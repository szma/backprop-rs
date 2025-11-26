# backprop-rs

A minimal autograd engine in Rust, inspired by [Karpathy's micrograd](https://github.com/karpathy/micrograd).

## Architecture

Uses the **Arena pattern** for the computation graph. Variables live in a central `Context`, operations return indices. Avoids Rust's ownership complexity (no `Rc<RefCell<...>>`).

## Usage

### Graph API (ergonomic)

```rust
use backprop_rs::syntax::Graph;

let g = Graph::new();
let mlp = g.mlp(2, vec![8, 8, 1]);  // 2 inputs -> 8 -> 8 -> 1 output

let inputs = [g.var(1.0), g.var(0.0)];
let pred = mlp.forward(&inputs);

let target = g.var(1.0);
let loss = (pred[0] - target) * (pred[0] - target);  // MSE

loss.backprop();
```

### Raw Arena API (explicit)

```rust
use backprop_rs::engine::Context;
use backprop_rs::nn_raw::MLP;

let mut ctx = Context::new();
let mlp = MLP::new(&mut ctx, 2, vec![8, 8, 1]);

let x0 = ctx.var(1.0);
let x1 = ctx.var(0.0);
let pred = mlp.forward(&mut ctx, &[x0, x1]);

let target = ctx.var(1.0);
let diff = ctx.sub(pred[0], target);
let loss = ctx.mul(diff, diff);

ctx.backprop(loss);
```

Both APIs use the same underlying arena.

## Supported Operations

`+`, `-`, `*`, `/`, `pow`, `relu`, `neg`

Run `cargo run` to see XOR training with both APIs.
