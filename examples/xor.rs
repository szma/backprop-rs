#![allow(dead_code)]

fn main() {
    xor();
}

fn xor() {
    use backprop_rs::graph::Graph;
    let xs: Vec<[f64; 2]> = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let ys: Vec<f64> = vec![0.0, 1.0, 1.0, 0.0];

    let g = Graph::new();
    let mlp = g.mlp(2, vec![32, 32, 1]);
    let params = mlp.parameters();
    let checkpoint = g.len();

    let lr = 0.01;

    for epoch in 0..500 {
        let mut total_loss = g.variable(0.0);

        for (x, &target) in xs.iter().zip(ys.iter()) {
            let inputs = [g.variable(x[0]), g.variable(x[1])];
            let pred = mlp.forward(&inputs);
            let y_target = g.variable(target);

            let loss = (pred[0] - y_target).pow(2.);
            total_loss = total_loss + loss;
        }

        total_loss.backward();

        for &p in &params {
            let grad = p.grad().unwrap_or(0.0);
            p.set_data(p.data() - lr * grad);
        }

        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, total_loss.data());
        }

        g.zero_grad();
        g.truncate(checkpoint);
    }

    println!("\nResults:");
    for (x, &target) in xs.iter().zip(ys.iter()) {
        let inputs = [g.variable(x[0]), g.variable(x[1])];
        let pred = mlp.forward(&inputs);
        println!("  {:?} -> {:.3} (expected: {})", x, pred[0].data(), target);
    }
}
