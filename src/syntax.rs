use crate::engine::Context;
use std::{
    cell::RefCell,
    ops::{Add, Div, Mul, Neg, Sub},
};

/// Trait for creating variables in a computation graph
pub trait VarFactory<'a> {
    fn var(&self, data: f64) -> Var<'a>;
}

pub struct Graph {
    ctx: RefCell<Context>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            ctx: RefCell::new(Context::new()),
        }
    }

    pub fn var(&self, data: f64) -> Var<'_> {
        Var::new(&self.ctx, data)
    }

    pub fn zero_grad(&self) {
        self.ctx.borrow_mut().zero_grad();
    }

    pub fn len(&self) -> usize {
        self.ctx.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.ctx.borrow().is_empty()
    }

    pub fn truncate(&self, len: usize) {
        self.ctx.borrow_mut().truncate(len);
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
    idx: usize,
    ctx: &'a RefCell<Context>,
}

impl<'a> Var<'a> {
    pub fn new(ctx: &'a RefCell<Context>, data: f64) -> Self {
        Var {
            idx: ctx.borrow_mut().var(data),
            ctx,
        }
    }

    pub fn backprop(self) {
        self.ctx.borrow_mut().backprop(self.idx);
    }

    pub fn data(self) -> f64 {
        self.ctx.borrow().data(self.idx)
    }

    pub fn grad(self) -> Option<f64> {
        self.ctx.borrow().grad(self.idx)
    }

    pub fn set_data(self, data: f64) {
        self.ctx.borrow_mut().set_data(self.idx, data);
    }

    pub fn zero_grad(self) {
        self.ctx.borrow_mut().zero_grad_single(self.idx);
    }

    pub fn pow(self, exp: f64) -> Self {
        let idx = self.ctx.borrow_mut().pow(self.idx, exp);
        Var { idx, ctx: self.ctx }
    }

    pub fn relu(self) -> Self {
        let idx = self.ctx.borrow_mut().relu(self.idx);
        Var { idx, ctx: self.ctx }
    }
}

impl<'a> Add for Var<'a> {
    type Output = Var<'a>;
    fn add(self, rhs: Self) -> Self {
        let idx = self.ctx.borrow_mut().add(self.idx, rhs.idx);
        Var { idx, ctx: self.ctx }
    }
}

impl<'a> Sub for Var<'a> {
    type Output = Var<'a>;
    fn sub(self, rhs: Self) -> Self {
        let idx = self.ctx.borrow_mut().sub(self.idx, rhs.idx);
        Var { idx, ctx: self.ctx }
    }
}

impl<'a> Mul for Var<'a> {
    type Output = Var<'a>;
    fn mul(self, rhs: Self) -> Self {
        let idx = self.ctx.borrow_mut().mul(self.idx, rhs.idx);
        Var { idx, ctx: self.ctx }
    }
}

impl<'a> Div for Var<'a> {
    type Output = Var<'a>;
    fn div(self, rhs: Self) -> Self {
        let idx = self.ctx.borrow_mut().div(self.idx, rhs.idx);
        Var { idx, ctx: self.ctx }
    }
}

impl<'a> Neg for Var<'a> {
    type Output = Var<'a>;

    fn neg(self) -> Self::Output {
        let idx = self.ctx.borrow_mut().neg(self.idx);
        Var { idx, ctx: self.ctx }
    }
}
