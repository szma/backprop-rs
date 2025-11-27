#![allow(dead_code)]
use std::{
    cell::RefCell,
    collections::HashSet,
    ops::{Add, Div, Mul, Neg, Sub},
};

pub type VariableDataIdx = usize;

/// An Arena that holds all the variable data element in a vector.
/// The vector is guarded by interior mutability to allow convinience access from the "Variable" structs
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

    /// Construct a new variable with data
    pub fn variable(&self, data: f64) -> Variable<'_> {
        let mut vars = self.vars.borrow_mut();
        vars.push(VariableData::new(data));
        Variable {
            idx: vars.len() - 1,
            graph: self,
        }
    }

    /// Set all gradients to zero
    pub fn zero_grad(&self) {
        for var in self.vars.borrow_mut().iter_mut() {
            var.grad = None;
        }
    }

    /// The number of variables
    pub fn len(&self) -> usize {
        self.vars.borrow().len()
    }

    /// Are there any variables in the graph?
    pub fn is_empty(&self) -> bool {
        self.vars.borrow().is_empty()
    }

    /// Remove all elements from len onwards.
    /// Useful to reset graph after computation if model is initialized first
    pub fn truncate(&self, len: usize) {
        self.vars.borrow_mut().truncate(len);
    }

    /// Convenience function to create a single neuron
    pub fn neuron(&self, nin: i16, nonlin: bool) -> crate::nn::Neuron<'_> {
        crate::nn::Neuron::new(self, nin, nonlin)
    }

    /// Convenience function to create a layer of neurons
    pub fn layer(&self, nin: i16, nout: i16, nonlin: bool) -> crate::nn::Layer<'_> {
        crate::nn::Layer::new(self, nin, nout, nonlin)
    }

    /// Convenience function to create a MLP
    pub fn mlp(&self, nin: i16, nouts: Vec<i16>) -> crate::nn::MLP<'_> {
        crate::nn::MLP::new(self, nin, nouts)
    }

    /// Compute the softmax from logits
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

    /// Compute the cross entropy
    pub fn cross_entropy<'a>(&'a self, probs: &[Variable<'a>], target: usize) -> Variable<'a> {
        -probs[target].log()
    }

    // Internal arena operations

    /// Add a new computation variable, forward path is executed directly
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

    /// data Getter
    fn data(&self, idx: VariableDataIdx) -> f64 {
        self.vars.borrow()[idx].data
    }

    /// grad Getter
    fn grad(&self, idx: VariableDataIdx) -> Option<f64> {
        self.vars.borrow()[idx].grad
    }

    /// data Setter
    fn set_data(&self, idx: VariableDataIdx, data: f64) {
        self.vars.borrow_mut()[idx].data = data;
    }

    /// Remove grad for single variable
    fn zero_grad_single(&self, idx: VariableDataIdx) {
        self.vars.borrow_mut()[idx].grad = None;
    }

    /// Add add op variable, normally used by Variable
    fn add_op(&self, a: VariableDataIdx, b: VariableDataIdx) -> VariableDataIdx {
        self.push_var(vec![a, b], Op::Add)
    }

    /// Add mul op variable, normally used by Variable
    fn mul_op(&self, a: VariableDataIdx, b: VariableDataIdx) -> VariableDataIdx {
        self.push_var(vec![a, b], Op::Mul)
    }

    /// Add pow op variable, normally used by Variable
    fn pow_op(&self, a: VariableDataIdx, exp: f64) -> VariableDataIdx {
        self.push_var(vec![a], Op::Pow(exp))
    }

    /// Add relu op variable, normally used by Variable
    fn relu_op(&self, a: VariableDataIdx) -> VariableDataIdx {
        self.push_var(vec![a], Op::ReLU)
    }

    /// Add exp op variable, normally used by Variable
    fn exp_op(&self, a: VariableDataIdx) -> VariableDataIdx {
        self.push_var(vec![a], Op::Exp)
    }

    /// Add ln op variable, normally used by Variable
    fn log_op(&self, a: VariableDataIdx) -> VariableDataIdx {
        self.push_var(vec![a], Op::Log)
    }

    /// Add neg op variable, normally used by Variable
    fn neg_op(&self, a: VariableDataIdx) -> VariableDataIdx {
        let minus_one = self.variable(-1.0).idx;
        self.mul_op(a, minus_one)
    }

    /// Add sub op variable (using neg and add), normally used by Variable
    fn sub_op(&self, a: VariableDataIdx, b: VariableDataIdx) -> VariableDataIdx {
        let neg_b = self.neg_op(b);
        self.add_op(a, neg_b)
    }

    /// Add div op variable (using pow and mul), normally used by Variable
    fn div_op(&self, a: VariableDataIdx, b: VariableDataIdx) -> VariableDataIdx {
        let b_inv = self.pow_op(b, -1.0);
        self.mul_op(a, b_inv)
    }

    /// Backpropagate gradients for a single variable to its children
    fn backward_single(&self, a: VariableDataIdx) {
        let mut vars = self.vars.borrow_mut();

        // calc grads depending on op type
        let children_data: Vec<f64> = vars[a].children.iter().map(|&c| vars[c].data).collect();
        let grads = vars[a].op.backward(
            &children_data,
            vars[a].data,
            vars[a].grad.unwrap_or_default(),
        );

        // accumulate grads
        for (i, grad) in grads.iter().enumerate() {
            let child = vars[a].children[i];
            // initialized child's grad to 0.0 if None and increase by grad
            *vars[child].grad.get_or_insert_default() += grad;
        }
    }

    /// Backpropagate gradiants through the graph
    fn backward(&self, idx: VariableDataIdx) {
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

        build_topo(idx, &mut topo, &mut visited, &self.vars.borrow());
        self.vars.borrow_mut()[idx].grad = Some(1.0);

        // c = a + b => topo=vec![a,b,c] => rev() to start from c pushing the grads though the graph
        for v in topo.iter().rev() {
            self.backward_single(*v);
        }
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience Variable on which operations like +, *, /, etc. are defined for a nice API.
/// Can be cheaply copied, only holds index of data in graph/arena and a reference to the graph
#[derive(Debug, Copy, Clone)]
pub struct Variable<'a> {
    idx: VariableDataIdx,
    graph: &'a Graph,
}

impl<'a> Variable<'a> {
    fn new(idx: VariableDataIdx, graph: &'a Graph) -> Self {
        Variable { idx, graph }
    }

    pub fn backward(self) {
        self.graph.backward(self.idx);
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
        Variable::new(idx, self.graph)
    }

    pub fn relu(self) -> Self {
        let idx = self.graph.relu_op(self.idx);
        Variable::new(idx, self.graph)
    }

    pub fn exp(self) -> Self {
        let idx = self.graph.exp_op(self.idx);
        Variable::new(idx, self.graph)
    }

    pub fn log(self) -> Self {
        let idx = self.graph.log_op(self.idx);
        Variable::new(idx, self.graph)
    }
}

impl<'a> Add for Variable<'a> {
    type Output = Variable<'a>;
    fn add(self, rhs: Self) -> Self {
        let idx = self.graph.add_op(self.idx, rhs.idx);
        Variable::new(idx, self.graph)
    }
}

impl<'a> Sub for Variable<'a> {
    type Output = Variable<'a>;
    fn sub(self, rhs: Self) -> Self {
        let idx = self.graph.sub_op(self.idx, rhs.idx);
        Variable::new(idx, self.graph)
    }
}

impl<'a> Mul for Variable<'a> {
    type Output = Variable<'a>;
    fn mul(self, rhs: Self) -> Self {
        let idx = self.graph.mul_op(self.idx, rhs.idx);
        Variable::new(idx, self.graph)
    }
}

impl<'a> Div for Variable<'a> {
    type Output = Variable<'a>;
    fn div(self, rhs: Self) -> Self {
        let idx = self.graph.div_op(self.idx, rhs.idx);
        Variable::new(idx, self.graph)
    }
}

impl<'a> Neg for Variable<'a> {
    type Output = Variable<'a>;

    fn neg(self) -> Self::Output {
        let idx = self.graph.neg_op(self.idx);
        Variable::new(idx, self.graph)
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
