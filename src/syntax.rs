use crate::engine::Context;
use std::{
    cell::RefCell,
    ops::{Add, Div, Mul, Neg, Sub},
};

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
        self.ctx.borrow_mut().data(self.idx)
    }

    pub fn grad(self) -> Option<f64> {
        self.ctx.borrow_mut().grad(self.idx)
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
