#![allow(dead_code)]

#[cfg(test)]
mod tests;

fn main() {
    println!("=== Graph API ===\n");
    xor_graph_api();

    println!("\n=== Raw Arena API ===\n");
    xor_raw_api();
}

/// Graph API: Ergonomic syntax with operator overloading
fn xor_graph_api() {
    use backprop_rs::syntax::Graph;
    let xs: Vec<[f64; 2]> = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let ys: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];

    let g = Graph::new();
    let mlp = g.mlp(2, vec![8, 8, 1]);
    let params = mlp.parameters();

    for epoch in 0..500 {
        let mut total_loss = g.var(0.0);

        for (x, &target) in xs.iter().zip(ys.iter()) {
            let inputs = [g.var(x[0]), g.var(x[1])];
            let pred = mlp.forward(&inputs);
            let y_target = g.var(target);

            // Nice syntax: operators work directly on Var
            let diff = pred[0] - y_target;
            let loss = diff * diff;
            total_loss = total_loss + loss;
        }

        total_loss.backprop();

        for &p in &params {
            let grad = p.grad().unwrap_or(0.0);
            p.set_data(p.data() - 0.01 * grad);
        }

        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, total_loss.data());
        }

        g.zero_grad();
    }

    println!("\nResults:");
    for (x, &target) in xs.iter().zip(ys.iter()) {
        let inputs = [g.var(x[0]), g.var(x[1])];
        let pred = mlp.forward(&inputs);
        println!("  {:?} -> {:.3} (expected: {})", x, pred[0].data(), target);
    }
}

/// Raw Arena API: Direct access to Context with explicit operations
fn xor_raw_api() {
    use backprop_rs::engine::Context;
    use backprop_rs::nn_raw::MLP;

    let xs: Vec<[f64; 2]> = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let ys: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];

    let mut ctx = Context::new();
    let mlp = MLP::new(&mut ctx, 2, vec![8, 8, 1]);
    let params = mlp.parameters();
    let checkpoint = ctx.len();

    for epoch in 0..500 {
        let mut total_loss = ctx.var(0.0);

        for (x, &target) in xs.iter().zip(ys.iter()) {
            let x0 = ctx.var(x[0]);
            let x1 = ctx.var(x[1]);
            let inputs = vec![x0, x1];

            let pred = mlp.forward(&mut ctx, &inputs);
            let y_target = ctx.var(target);

            // Explicit operations on Context
            let diff = ctx.sub(pred[0], y_target);
            let loss = ctx.mul(diff, diff);
            total_loss = ctx.add(total_loss, loss);
        }

        ctx.backprop(total_loss);

        for &p in &params {
            let grad = ctx.grad(p).unwrap_or(0.0);
            let new_val = ctx.data(p) - 0.01 * grad;
            ctx.set_data(p, new_val);
        }

        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, ctx.data(total_loss));
        }

        ctx.zero_grad();
        ctx.truncate(checkpoint);
    }

    println!("\nResults:");
    for (x, &target) in xs.iter().zip(ys.iter()) {
        let x0 = ctx.var(x[0]);
        let x1 = ctx.var(x[1]);
        let pred = mlp.forward(&mut ctx, &[x0, x1]);
        println!(
            "  {:?} -> {:.3} (expected: {})",
            x,
            ctx.data(pred[0]),
            target
        );
    }
}
