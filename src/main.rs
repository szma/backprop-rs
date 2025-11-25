#![allow(dead_code)]

use backprop_rs::engine::Context;
use backprop_rs::syntax::graph;

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

    // Syntactic sugar variant, maps to arena in the background: Scoped API
    graph(|var| {
        let a = var(1.0);
        let b = var(2.0);

        let c = a + b;
        let d = (c * c).relu() + c.pow(2.0);

        d.backprop();

        dbg!(a.grad());
    });
}
