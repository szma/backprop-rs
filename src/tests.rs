use backprop_rs::graph::Graph;

#[test]
fn test_add() {
    let g = Graph::new();
    let a = g.variable(2.0);
    let b = g.variable(3.0);
    let c = a + b;

    assert_eq!(c.data(), 5.0);

    c.backprop();
    assert_eq!(a.grad().unwrap(), 1.0);
    assert_eq!(b.grad().unwrap(), 1.0);
}

#[test]
fn test_mul() {
    let g = Graph::new();
    let a = g.variable(2.0);
    let b = g.variable(3.0);
    let c = a * b;

    assert_eq!(c.data(), 6.0);

    c.backprop();
    assert_eq!(a.grad().unwrap(), 3.0); // dc/da = b
    assert_eq!(b.grad().unwrap(), 2.0); // dc/db = a
}

#[test]
fn test_pow() {
    let g = Graph::new();
    let a = g.variable(2.0);
    let b = a.pow(3.0); // a^3 = 8

    assert_eq!(b.data(), 8.0);

    b.backprop();
    assert_eq!(a.grad().unwrap(), 12.0); // 3 * 2^2 = 12
}

#[test]
fn test_relu_positive() {
    let g = Graph::new();
    let a = g.variable(2.0);
    let b = a.relu();

    assert_eq!(b.data(), 2.0);

    b.backprop();
    assert_eq!(a.grad().unwrap(), 1.0);
}

#[test]
fn test_relu_negative() {
    let g = Graph::new();
    let a = g.variable(-2.0);
    let b = a.relu();

    assert_eq!(b.data(), 0.0);

    b.backprop();
    assert_eq!(a.grad().unwrap(), 0.0);
}

#[test]
fn test_sub() {
    let g = Graph::new();
    let a = g.variable(5.0);
    let b = g.variable(3.0);
    let c = a - b;

    assert_eq!(c.data(), 2.0);

    c.backprop();
    assert_eq!(a.grad().unwrap(), 1.0);
    assert_eq!(b.grad().unwrap(), -1.0);
}

#[test]
fn test_div() {
    let g = Graph::new();
    let a = g.variable(6.0);
    let b = g.variable(2.0);
    let c = a / b; // 6/2 = 3

    assert_eq!(c.data(), 3.0);

    c.backprop();
    assert_eq!(a.grad().unwrap(), 0.5); // dc/da = 1/b = 0.5
    assert_eq!(b.grad().unwrap(), -1.5); // dc/db = -a/b^2 = -6/4 = -1.5
}

#[test]
fn test_variable_reuse() {
    // b = a*a + a, db/da = 2a + 1
    let g = Graph::new();
    let a = g.variable(3.0);
    let a_sq = a * a;
    let b = a_sq + a;

    assert_eq!(b.data(), 12.0);

    b.backprop();
    assert_eq!(a.grad().unwrap(), 7.0); // 2*3 + 1 = 7
}

#[test]
fn test_chain() {
    // d = (a + b) * c
    let g = Graph::new();
    let a = g.variable(1.0);
    let b = g.variable(2.0);
    let c = g.variable(3.0);
    let ab = a + b;
    let d = ab * c;

    assert_eq!(d.data(), 9.0);

    d.backprop();
    assert_eq!(a.grad().unwrap(), 3.0); // c
    assert_eq!(b.grad().unwrap(), 3.0); // c
    assert_eq!(c.grad().unwrap(), 3.0); // a + b
}
