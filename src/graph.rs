#![allow(dead_code)]
use std::{
    cell::RefCell,
    collections::HashSet,
    ops::{Add, Div, Mul, Neg, Sub},
};

pub type VariableDataIdx = usize;

#[derive(Debug)]
pub struct Graph {
    vars: RefCell<Vec<VariableData>>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            vars: RefCell::new(Vec::new()),
        }
    }

    pub fn variable(&self, data: f64) -> Variable<'_> {
        Variable::new(self, data)
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

    pub fn softmax<'a>(&'a self, logits: &[Variable<'a>]) -> Vec<Variable<'a>> {
        // Numerically stable softmax: subtract max before exp
        let max_val = logits
            .iter()
            .map(|v| v.data())
            .fold(f64::NEG_INFINITY, f64::max);
        let max_var = self.variable(max_val);

        let exps: Vec<Variable<'_>> = logits.iter().map(|x| (*x - max_var).exp()).collect();
        let sum_exp = exps.iter().skip(1).fold(exps[0], |acc, &x| acc + x);

        exps.iter().map(|&e| e / sum_exp).collect()
    }

    pub fn cross_entropy<'a>(&'a self, probs: &[Variable<'a>], target: usize) -> Variable<'a> {
        -probs[target].log()
    }

    // Internal arena operations

    fn push_var(&self, children: Vec<VariableDataIdx>, op: Op) -> VariableDataIdx {
        let mut vars = self.vars.borrow_mut();
        let children_data: Vec<f64> = children.iter().map(|&c| vars[c].data).collect();
        let data = op.forward(&children_data);
        vars.push(VariableData {
            data,
            grad: None,
            children,
            op,
        });
        vars.len() - 1
    }

    fn data(&self, idx: VariableDataIdx) -> f64 {
        self.vars.borrow()[idx].data
    }

    fn grad(&self, idx: VariableDataIdx) -> Option<f64> {
        self.vars.borrow()[idx].grad
    }

    fn set_data(&self, idx: VariableDataIdx, data: f64) {
        self.vars.borrow_mut()[idx].data = data;
    }

    fn zero_grad_single(&self, idx: VariableDataIdx) {
        self.vars.borrow_mut()[idx].grad = None;
    }

    fn add_op(&self, a: VariableDataIdx, b: VariableDataIdx) -> VariableDataIdx {
        self.push_var(vec![a, b], Op::Add)
    }

    fn mul_op(&self, a: VariableDataIdx, b: VariableDataIdx) -> VariableDataIdx {
        self.push_var(vec![a, b], Op::Mul)
    }

    fn pow_op(&self, a: VariableDataIdx, exp: f64) -> VariableDataIdx {
        self.push_var(vec![a], Op::Pow(exp))
    }

    fn relu_op(&self, a: VariableDataIdx) -> VariableDataIdx {
        self.push_var(vec![a], Op::ReLU)
    }

    fn exp_op(&self, a: VariableDataIdx) -> VariableDataIdx {
        self.push_var(vec![a], Op::Exp)
    }

    fn log_op(&self, a: VariableDataIdx) -> VariableDataIdx {
        self.push_var(vec![a], Op::Log)
    }

    fn neg_op(&self, a: VariableDataIdx) -> VariableDataIdx {
        let minus_one = self.variable(-1.0).idx;
        self.mul_op(a, minus_one)
    }

    fn sub_op(&self, a: VariableDataIdx, b: VariableDataIdx) -> VariableDataIdx {
        let neg_b = self.neg_op(b);
        self.add_op(a, neg_b)
    }

    fn div_op(&self, a: VariableDataIdx, b: VariableDataIdx) -> VariableDataIdx {
        let b_inv = self.pow_op(b, -1.0);
        self.mul_op(a, b_inv)
    }

    fn add_grad(&self, idx: VariableDataIdx, delta: f64) {
        let mut vars = self.vars.borrow_mut();
        let grad = &mut vars[idx].grad;
        *grad = Some(grad.unwrap_or(0.0) + delta);
    }

    fn backward(&self, a: VariableDataIdx) {
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

    fn backprop(&self, a: VariableDataIdx) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo(
            v: VariableDataIdx,
            topo: &mut Vec<VariableDataIdx>,
            visited: &mut HashSet<VariableDataIdx>,
            vars: &[VariableData],
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

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Variable<'a> {
    idx: VariableDataIdx,
    graph: &'a Graph,
}

impl<'a> Variable<'a> {
    fn new(graph: &'a Graph, data: f64) -> Self {
        let mut vars = graph.vars.borrow_mut();
        vars.push(VariableData::new(data));
        Variable {
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
        Variable {
            idx,
            graph: self.graph,
        }
    }

    pub fn relu(self) -> Self {
        let idx = self.graph.relu_op(self.idx);
        Variable {
            idx,
            graph: self.graph,
        }
    }

    pub fn exp(self) -> Self {
        let idx = self.graph.exp_op(self.idx);
        Variable {
            idx,
            graph: self.graph,
        }
    }

    pub fn log(self) -> Self {
        let idx = self.graph.log_op(self.idx);
        Variable {
            idx,
            graph: self.graph,
        }
    }
}

impl<'a> Add for Variable<'a> {
    type Output = Variable<'a>;
    fn add(self, rhs: Self) -> Self {
        let idx = self.graph.add_op(self.idx, rhs.idx);
        Variable {
            idx,
            graph: self.graph,
        }
    }
}

impl<'a> Sub for Variable<'a> {
    type Output = Variable<'a>;
    fn sub(self, rhs: Self) -> Self {
        let idx = self.graph.sub_op(self.idx, rhs.idx);
        Variable {
            idx,
            graph: self.graph,
        }
    }
}

impl<'a> Mul for Variable<'a> {
    type Output = Variable<'a>;
    fn mul(self, rhs: Self) -> Self {
        let idx = self.graph.mul_op(self.idx, rhs.idx);
        Variable {
            idx,
            graph: self.graph,
        }
    }
}

impl<'a> Div for Variable<'a> {
    type Output = Variable<'a>;
    fn div(self, rhs: Self) -> Self {
        let idx = self.graph.div_op(self.idx, rhs.idx);
        Variable {
            idx,
            graph: self.graph,
        }
    }
}

impl<'a> Neg for Variable<'a> {
    type Output = Variable<'a>;

    fn neg(self) -> Self::Output {
        let idx = self.graph.neg_op(self.idx);
        Variable {
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
    Exp,
    Log,
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
            Op::Exp => children_data[0].exp(),
            Op::Log => children_data[0].ln(),
            Op::Value => unimplemented!(),
        }
    }

    fn backward(&self, children_data: &[f64], out_data: f64, out_grad: f64) -> Vec<f64> {
        match self {
            Op::Add => vec![out_grad, out_grad],
            Op::Mul => vec![children_data[1] * out_grad, children_data[0] * out_grad],
            Op::Pow(exp) => vec![exp * children_data[0].powf(exp - 1.0) * out_grad],
            Op::ReLU => vec![if out_data > 0.0 { out_grad } else { 0.0 }],
            Op::Exp => vec![out_data * out_grad], // d/dx exp(x) = exp(x)
            Op::Log => vec![out_grad / children_data[0]], // d/dx ln(x) = 1/x
            Op::Value => vec![],
        }
    }
}

#[derive(Debug, Clone)]
struct VariableData {
    data: f64,
    grad: Option<f64>,
    children: Vec<VariableDataIdx>,
    op: Op,
}

impl VariableData {
    fn new(data: f64) -> Self {
        Self {
            data,
            grad: None,
            children: Vec::new(),
            op: Op::Value,
        }
    }
}
