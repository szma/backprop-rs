# backprop-rs

A minimal autograd engine in Rust, inspired by [Karpathy's micrograd](https://github.com/karpathy/micrograd).

## Architecture

Uses the **Arena pattern** to manage the computation graph. All variables are stored in a central `Context`, and operations return indices (`VariableIdx`) instead of owned values. This avoids Rust's ownership complexity with graph structures (no `Rc<RefCell<...>>` needed). Adds some syntactic sugar on top of it, see below.

## Usage

### Scoped API

Uses the **Scoped API pattern** (like `std::thread::scope`) to hide context management:

```rust
use backprop_rs::syntax::graph;

graph(|var| {
    let a = var(3.0);
    let b = var(2.0);

    let c = (a * a + b).relu();  // c = relu(a² + b)

    c.backprop();

    assert_eq!(a.grad(), Some(6.0));  // dc/da = 2a = 6
});
```

### Direct Arena Syntax

For a better insight on what's going on, use the arena directly:

```rust
use backprop_rs::engine::Context;

let mut ctx = Context::new();

let a = ctx.var(3.0);
let b = ctx.var(2.0);

let a_sq = ctx.mul(a, a);
let c = ctx.add(a_sq, b);  // c = a² + b

ctx.backprop(c);

assert_eq!(ctx.grad(a), Some(6.0));  // dc/da = 2a = 6
assert_eq!(ctx.grad(b), Some(1.0));  // dc/db = 1
```

Both APIs map to the same underlying arena.

## Supported Operations

- `+`, `-`, `*`, `/` (via operator overloading in scoped API)
- `pow`, `relu`, `neg`

Run tests with `cargo test`.
