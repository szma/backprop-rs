# backprop-rs

A minimal autograd engine in Rust, inspired by [Karpathy's micrograd](https://github.com/karpathy/micrograd).

## Architecture

Uses the **Arena pattern** to manage the computation graph. All variables are stored in a central `Context`, and operations return indices (`VariableIdx`) instead of owned values. This avoids Rust's ownership complexity with graph structures (no `Rc<RefCell<...>>` needed).

## Usage

### Direct Arena Syntax

```rust
let mut ctx = Context::new();

let a = ctx.var(3.0);
let b = ctx.var(2.0);

let a_sq = ctx.mul(a, a);
let c = ctx.add(a_sq, b);  // c = a² + b

ctx.backprop(c);

assert_eq!(ctx.grad(a), Some(6.0));  // dc/da = 2a = 6
assert_eq!(ctx.grad(b), Some(1.0));  // dc/db = 1
```

### Syntactic Sugar

Ergonomic wrapper with operator overloading (`+`, `-`, `*`, `/`):

```rust
let ctx = RefCell::new(Context::new());

let a = Var::new(&ctx, 3.0);
let b = Var::new(&ctx, 2.0);

let c = (a * a + b).relu();  // c = relu(a² + b)

c.backprop();

assert_eq!(a.grad(), Some(6.0));
```

Both map to the same underlying arena.

## Supported Operations

- `add`, `sub`, `mul`, `div`
- `pow`, `relu`, `neg`

Run tests with `cargo test`.
