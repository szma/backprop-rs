#![allow(dead_code)]

use std::cell::RefCell;

use backprop_rs::engine::Context;
use backprop_rs::syntax::Var;

#[cfg(test)]
mod tests;

fn main() {
    // Direct Arena syntax
    let mut ctx = Context::new();
    let a = ctx.var(1.0);
    let b = ctx.var(2.0);
    let c = ctx.add(a, b);

    let d0 = ctx.mul(c, c);
    let d = ctx.add(d0, c);

    ctx.backprop(d);
    dbg!(ctx.grad(a));

    // Syntactic sugar variant, maps to arena in the background
    let ctx = RefCell::new(Context::new());
    let a = Var::new(&ctx, 1.0);
    let b = Var::new(&ctx, 2.0);

    let c = a + b;
    let d = (c * c).relu() + c.pow(2.0);

    d.backprop();

    dbg!(a.grad());
}
