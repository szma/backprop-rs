#![allow(dead_code)]

#[cfg(test)]
mod tests;

fn main() {
    xor();
}

fn xor() {
    use backprop_rs::graph::Graph;
    let xs: Vec<[f64; 2]> = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let ys: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];

    let g = Graph::new();
    let mlp = g.mlp(2, vec![8, 8, 1]);
    let params = mlp.parameters();
    let checkpoint = g.len();

    for epoch in 0..500 {
        let mut total_loss = g.var(0.0);

        for (x, &target) in xs.iter().zip(ys.iter()) {
            let inputs = [g.var(x[0]), g.var(x[1])];
            let pred = mlp.forward(&inputs);
            let y_target = g.var(target);

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
        g.truncate(checkpoint);
    }

    println!("\nResults:");
    for (x, &target) in xs.iter().zip(ys.iter()) {
        let inputs = [g.var(x[0]), g.var(x[1])];
        let pred = mlp.forward(&inputs);
        println!("  {:?} -> {:.3} (expected: {})", x, pred[0].data(), target);
    }
}
