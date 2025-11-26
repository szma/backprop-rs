#![allow(dead_code)]

use backprop_rs::engine::Context;
use backprop_rs::nn::MLP;

#[cfg(test)]
mod tests;

fn main() {
    // XOR problem: Learn the xor function
    // Inputs: (0,0) -> 0, (0,1) -> 1, (1,0) -> 1, (1,1) -> 0

    let xs: Vec<[f64; 2]> = vec![
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ];
    let ys: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];

    let mut ctx = Context::new();

    // MLP: 2 inputs -> 8 hidden -> 8 hidden -> 1 output
    let mlp = MLP::new(&mut ctx, 2, vec![8,8,1]);
    let params = mlp.parameters();

    let learning_rate = 0.01;
    let checkpoint = ctx.len(); // Save variables after graph was built

    for epoch in 0..1000 {
        // Forward pass for all samples
        let mut total_loss = ctx.var(0.0);

        for (x, &target) in xs.iter().zip(ys.iter()) {
            let x0 = ctx.var(x[0]);
            let x1 = ctx.var(x[1]);
            let inputs = vec![x0, x1];

            let pred = mlp.forward(&mut ctx, &inputs);
            let y_target = ctx.var(target);

            // MSE loss: (pred - target)^2
            let diff = ctx.sub(pred[0], y_target);
            let loss = ctx.mul(diff, diff);
            total_loss = ctx.add(total_loss, loss);
        }

        // Backward pass
        ctx.backprop(total_loss);

        // Parameter update (SGD)
        for &p in &params {
            let grad = ctx.grad(p).unwrap_or(0.0);
            let new_val = ctx.data(p) - learning_rate * grad;
            ctx.set_data(p, new_val);
        }

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, ctx.data(total_loss));
        }

        // Reset: Zero gradients and remove forward path variables (keep only initial graph)
        ctx.zero_grad();
        ctx.truncate(checkpoint);
    }

    // Test
    println!("\nResults:");
    for (x, &target) in xs.iter().zip(ys.iter()) {
        let x0 = ctx.var(x[0]);
        let x1 = ctx.var(x[1]);
        let pred = mlp.forward(&mut ctx, &[x0, x1]);
        println!("  {:?} -> {:.3} (expected: {})", x, ctx.data(pred[0]), target);
    }
}
