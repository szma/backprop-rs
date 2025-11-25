use backprop_rs::engine::{Context, VariableIdx};

// Helper to get gradient, panics if None
fn grad(ctx: &Context, idx: VariableIdx) -> f64 {
    ctx.grad(idx).expect("gradient not computed")
}

#[test]
fn test_add() {
    let mut ctx = Context::new();
    let a = ctx.var(2.0);
    let b = ctx.var(3.0);
    let c = ctx.add(a, b);

    assert_eq!(ctx.data(c), 5.0);

    ctx.backprop(c);
    assert_eq!(grad(&ctx, a), 1.0);
    assert_eq!(grad(&ctx, b), 1.0);
}

#[test]
fn test_mul() {
    let mut ctx = Context::new();
    let a = ctx.var(2.0);
    let b = ctx.var(3.0);
    let c = ctx.mul(a, b);

    assert_eq!(ctx.data(c), 6.0);

    ctx.backprop(c);
    assert_eq!(grad(&ctx, a), 3.0); // dc/da = b
    assert_eq!(grad(&ctx, b), 2.0); // dc/db = a
}

#[test]
fn test_pow() {
    let mut ctx = Context::new();
    let a = ctx.var(2.0);
    let b = ctx.pow(a, 3.0); // a^3 = 8

    assert_eq!(ctx.data(b), 8.0);

    ctx.backprop(b);
    assert_eq!(grad(&ctx, a), 12.0); // 3 * 2^2 = 12
}

#[test]
fn test_relu_positive() {
    let mut ctx = Context::new();
    let a = ctx.var(2.0);
    let b = ctx.relu(a);

    assert_eq!(ctx.data(b), 2.0);

    ctx.backprop(b);
    assert_eq!(grad(&ctx, a), 1.0);
}

#[test]
fn test_relu_negative() {
    let mut ctx = Context::new();
    let a = ctx.var(-2.0);
    let b = ctx.relu(a);

    assert_eq!(ctx.data(b), 0.0);

    ctx.backprop(b);
    assert_eq!(grad(&ctx, a), 0.0);
}

#[test]
fn test_sub() {
    let mut ctx = Context::new();
    let a = ctx.var(5.0);
    let b = ctx.var(3.0);
    let c = ctx.sub(a, b);

    assert_eq!(ctx.data(c), 2.0);

    ctx.backprop(c);
    assert_eq!(grad(&ctx, a), 1.0);
    assert_eq!(grad(&ctx, b), -1.0);
}

#[test]
fn test_div() {
    let mut ctx = Context::new();
    let a = ctx.var(6.0);
    let b = ctx.var(2.0);
    let c = ctx.div(a, b); // 6/2 = 3

    assert_eq!(ctx.data(c), 3.0);

    ctx.backprop(c);
    assert_eq!(grad(&ctx, a), 0.5); // dc/da = 1/b = 0.5
    assert_eq!(grad(&ctx, b), -1.5); // dc/db = -a/b^2 = -6/4 = -1.5
}

#[test]
fn test_variable_reuse() {
    // b = a*a + a, db/da = 2a + 1
    let mut ctx = Context::new();
    let a = ctx.var(3.0);
    let a_sq = ctx.mul(a, a);
    let b = ctx.add(a_sq, a);

    assert_eq!(ctx.data(b), 12.0);

    ctx.backprop(b);
    assert_eq!(grad(&ctx, a), 7.0); // 2*3 + 1 = 7
}

#[test]
fn test_chain() {
    // d = (a + b) * c
    let mut ctx = Context::new();
    let a = ctx.var(1.0);
    let b = ctx.var(2.0);
    let c = ctx.var(3.0);
    let ab = ctx.add(a, b);
    let d = ctx.mul(ab, c);

    assert_eq!(ctx.data(d), 9.0);

    ctx.backprop(d);
    assert_eq!(grad(&ctx, a), 3.0); // c
    assert_eq!(grad(&ctx, b), 3.0); // c
    assert_eq!(grad(&ctx, c), 3.0); // a + b
}
