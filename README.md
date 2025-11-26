# backprop-rs

A minimal autograd engine in Rust, inspired by [Karpathy's micrograd](https://github.com/karpathy/micrograd).

## Architecture

Uses the **Arena pattern** with interior mutability (`RefCell`). All variables live in a central `Graph`, operations return lightweight `Var` handles containing just an index and a reference to the graph. This avoids Rust's ownership complexity (no `Rc<RefCell<...>>`).

## Usage

```rust
use backprop_rs::graph::Graph;

let g = Graph::new();
let mlp = g.mlp(2, vec![8, 8, 1]);  // 2 inputs -> 8 -> 8 -> 1 output

let inputs = [g.var(1.0), g.var(0.0)];
let pred = mlp.forward(&inputs);

let target = g.var(1.0);
let loss = (pred[0] - target) * (pred[0] - target);  // MSE

loss.backprop();
```

## Supported Operations

`+`, `-`, `*`, `/`, `pow`, `relu`, `neg`

Run `cargo run` to see XOR training.
