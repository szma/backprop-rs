#![allow(dead_code)]
use std::{
    cell::RefCell,
    collections::HashSet,
    ops::{Add, Div, Mul, Neg, Sub},
};

pub type VariableIdx = usize;

/// Trait for creating variables in a computation graph
pub trait VarFactory<'a> {
    fn var(&self, data: f64) -> Var<'a>;
}

#[derive(Debug)]
pub struct Graph {
    vars: RefCell<Vec<Variable>>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            vars: RefCell::new(Vec::new()),
        }
    }

    pub fn var(&self, data: f64) -> Var<'_> {
        Var::new(self, data)
    }

    pub fn zero_grad(&self) {
        for var in self.vars.borrow_mut().iter_mut() {
            var.grad = None;
        }
    }

    pub fn len(&self) -> usize {
        self.vars.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.vars.borrow().is_empty()
    }

    pub fn truncate(&self, len: usize) {
        self.vars.borrow_mut().truncate(len);
    }

    pub fn neuron(&self, nin: i16, nonlin: bool) -> crate::nn::Neuron<'_> {
        crate::nn::Neuron::new(self, nin, nonlin)
    }

    pub fn layer(&self, nin: i16, nout: i16, nonlin: bool) -> crate::nn::Layer<'_> {
        crate::nn::Layer::new(self, nin, nout, nonlin)
    }

    pub fn mlp(&self, nin: i16, nouts: Vec<i16>) -> crate::nn::MLP<'_> {
        crate::nn::MLP::new(self, nin, nouts)
    }

    // Internal arena operations

    fn push_var(&self, children: Vec<VariableIdx>, op: Op) -> VariableIdx {
        let mut vars = self.vars.borrow_mut();
        let children_data: Vec<f64> = children.iter().map(|&c| vars[c].data).collect();
        let data = op.forward(&children_data);
        vars.push(Variable {
            data,
            grad: None,
            children,
            op,
        });
        vars.len() - 1
    }

    fn data(&self, idx: VariableIdx) -> f64 {
        self.vars.borrow()[idx].data
    }

    fn grad(&self, idx: VariableIdx) -> Option<f64> {
        self.vars.borrow()[idx].grad
    }

    fn set_data(&self, idx: VariableIdx, data: f64) {
        self.vars.borrow_mut()[idx].data = data;
    }

    fn zero_grad_single(&self, idx: VariableIdx) {
        self.vars.borrow_mut()[idx].grad = None;
    }

    fn add_op(&self, a: VariableIdx, b: VariableIdx) -> VariableIdx {
        self.push_var(vec![a, b], Op::Add)
    }

    fn mul_op(&self, a: VariableIdx, b: VariableIdx) -> VariableIdx {
        self.push_var(vec![a, b], Op::Mul)
    }

    fn pow_op(&self, a: VariableIdx, exp: f64) -> VariableIdx {
        self.push_var(vec![a], Op::Pow(exp))
    }

    fn relu_op(&self, a: VariableIdx) -> VariableIdx {
        self.push_var(vec![a], Op::ReLU)
    }

    fn neg_op(&self, a: VariableIdx) -> VariableIdx {
        let minus_one = self.var(-1.0).idx;
        self.mul_op(a, minus_one)
    }

    fn sub_op(&self, a: VariableIdx, b: VariableIdx) -> VariableIdx {
        let neg_b = self.neg_op(b);
        self.add_op(a, neg_b)
    }

    fn div_op(&self, a: VariableIdx, b: VariableIdx) -> VariableIdx {
        let b_inv = self.pow_op(b, -1.0);
        self.mul_op(a, b_inv)
    }

    fn add_grad(&self, idx: VariableIdx, delta: f64) {
        let mut vars = self.vars.borrow_mut();
        let grad = &mut vars[idx].grad;
        *grad = Some(grad.unwrap_or(0.0) + delta);
    }

    fn backward(&self, a: VariableIdx) {
        let vars = self.vars.borrow();
        let var = &vars[a];
        let children_data: Vec<f64> = var.children.iter().map(|&c| vars[c].data).collect();
        let grads = var
            .op
            .backward(&children_data, var.data, var.grad.unwrap_or(0.0));
        let children = var.children.clone();
        drop(vars);

        for (child, grad) in children.iter().zip(grads) {
            self.add_grad(*child, grad);
        }
    }

    fn backprop(&self, a: VariableIdx) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo(
            v: VariableIdx,
            topo: &mut Vec<VariableIdx>,
            visited: &mut HashSet<VariableIdx>,
            vars: &[Variable],
        ) {
            if !visited.contains(&v) {
                visited.insert(v);
                for c in &vars[v].children {
                    build_topo(*c, topo, visited, vars);
                }
                topo.push(v);
            }
        }

        build_topo(a, &mut topo, &mut visited, &self.vars.borrow());
        self.vars.borrow_mut()[a].grad = Some(1.0);

        for v in topo.iter().rev() {
            self.backward(*v);
        }
    }
}

impl<'a> VarFactory<'a> for &'a Graph {
    fn var(&self, data: f64) -> Var<'a> {
        Graph::var(self, data)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Var<'a> {
    idx: VariableIdx,
    graph: &'a Graph,
}

impl<'a> Var<'a> {
    fn new(graph: &'a Graph, data: f64) -> Self {
        let mut vars = graph.vars.borrow_mut();
        vars.push(Variable::new(data));
        Var {
            idx: vars.len() - 1,
            graph,
        }
    }

    pub fn backprop(self) {
        self.graph.backprop(self.idx);
    }

    pub fn data(self) -> f64 {
        self.graph.data(self.idx)
    }

    pub fn grad(self) -> Option<f64> {
        self.graph.grad(self.idx)
    }

    pub fn set_data(self, data: f64) {
        self.graph.set_data(self.idx, data);
    }

    pub fn zero_grad(self) {
        self.graph.zero_grad_single(self.idx);
    }

    pub fn pow(self, exp: f64) -> Self {
        let idx = self.graph.pow_op(self.idx, exp);
        Var {
            idx,
            graph: self.graph,
        }
    }

    pub fn relu(self) -> Self {
        let idx = self.graph.relu_op(self.idx);
        Var {
            idx,
            graph: self.graph,
        }
    }
}

impl<'a> Add for Var<'a> {
    type Output = Var<'a>;
    fn add(self, rhs: Self) -> Self {
        let idx = self.graph.add_op(self.idx, rhs.idx);
        Var {
            idx,
            graph: self.graph,
        }
    }
}

impl<'a> Sub for Var<'a> {
    type Output = Var<'a>;
    fn sub(self, rhs: Self) -> Self {
        let idx = self.graph.sub_op(self.idx, rhs.idx);
        Var {
            idx,
            graph: self.graph,
        }
    }
}

impl<'a> Mul for Var<'a> {
    type Output = Var<'a>;
    fn mul(self, rhs: Self) -> Self {
        let idx = self.graph.mul_op(self.idx, rhs.idx);
        Var {
            idx,
            graph: self.graph,
        }
    }
}

impl<'a> Div for Var<'a> {
    type Output = Var<'a>;
    fn div(self, rhs: Self) -> Self {
        let idx = self.graph.div_op(self.idx, rhs.idx);
        Var {
            idx,
            graph: self.graph,
        }
    }
}

impl<'a> Neg for Var<'a> {
    type Output = Var<'a>;

    fn neg(self) -> Self::Output {
        let idx = self.graph.neg_op(self.idx);
        Var {
            idx,
            graph: self.graph,
        }
    }
}

// Internal types

#[derive(Debug, Copy, Clone)]
enum Op {
    Value,
    Add,
    Mul,
    Pow(f64),
    ReLU,
}

impl Op {
    fn forward(&self, children_data: &[f64]) -> f64 {
        match self {
            Op::Add => children_data[0] + children_data[1],
            Op::Mul => children_data[0] * children_data[1],
            Op::Pow(exp) => children_data[0].powf(*exp),
            Op::ReLU => {
                if children_data[0] > 0.0 {
                    children_data[0]
                } else {
                    0.0
                }
            }
            Op::Value => unimplemented!(),
        }
    }

    fn backward(&self, children_data: &[f64], out_data: f64, out_grad: f64) -> Vec<f64> {
        match self {
            Op::Add => vec![out_grad, out_grad],
            Op::Mul => vec![children_data[1] * out_grad, children_data[0] * out_grad],
            Op::Pow(exp) => vec![exp * children_data[0].powf(exp - 1.0) * out_grad],
            Op::ReLU => vec![if out_data > 0.0 { out_grad } else { 0.0 }],
            Op::Value => vec![],
        }
    }
}

#[derive(Debug, Clone)]
struct Variable {
    data: f64,
    grad: Option<f64>,
    children: Vec<VariableIdx>,
    op: Op,
}

impl Variable {
    fn new(data: f64) -> Self {
        Self {
            data,
            grad: None,
            children: Vec::new(),
            op: Op::Value,
        }
    }
}
