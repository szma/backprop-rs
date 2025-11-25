# backprop-rs

A minimal autograd engine in Rust, inspired by [Karpathy's micrograd](https://github.com/karpathy/micrograd).

## Example

```rust
let mut ctx = Context::new();

// Create variables
let a = ctx.var(3.0);
let b = ctx.var(2.0);

// Build computation graph: c = a*a + b
let a_sq = ctx.mul(a, a);
let c = ctx.add(a_sq, b);

// Backward pass
ctx.backprop(c);

// Gradients: dc/da = 2*a = 6, dc/db = 1
assert_eq!(ctx.vars[a].grad, Some(6.0));
assert_eq!(ctx.vars[b].grad, Some(1.0));
```

## Supported Operations

- `add`, `sub`, `mul`, `div`
- `pow`, `relu`, `neg`

Run tests with `cargo test`.
