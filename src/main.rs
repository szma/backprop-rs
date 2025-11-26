#![allow(dead_code)]

#[cfg(test)]
mod tests;

fn main() {
    mnist();
}

fn mnist() {
    use backprop_rs::graph::Graph;
    use backprop_rs::mnist::MnistData;
    use std::path::Path;

    // Load MNIST data - download from http://yann.lecun.com/exdb/mnist/
    let train = MnistData::load(
        Path::new("data/train-images-idx3-ubyte"),
        Path::new("data/train-labels-idx1-ubyte"),
    )
    .expect("Failed to load MNIST training data");

    let test = MnistData::load(
        Path::new("data/t10k-images-idx3-ubyte"),
        Path::new("data/t10k-labels-idx1-ubyte"),
    )
    .expect("Failed to load MNIST test data");

    println!("Loaded {} training, {} test images", train.len(), test.len());

    let g = Graph::new();
    let mlp = g.mlp(784, vec![32, 10]);
    let params = mlp.parameters();
    let checkpoint = g.len();

    println!("Model has {} parameters", params.len());

    let lr = 0.01;
    let batch_size = 32;
    let epochs = 10;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;

        for batch_start in (0..train.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train.len());
            let mut batch_loss = g.variable(0.0);

            for i in batch_start..batch_end {
                let inputs: Vec<_> = train.images[i].iter().map(|&x| g.variable(x)).collect();
                let logits = mlp.forward(&inputs);
                let probs = g.softmax(&logits);
                let target = train.labels[i] as usize;

                let loss = g.cross_entropy(&probs, target);
                batch_loss = batch_loss + loss;

                // Track accuracy
                let pred = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.data().partial_cmp(&b.1.data()).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                if pred == target {
                    correct += 1;
                }
            }

            total_loss += batch_loss.data();
            batch_loss.backprop();

            if batch_start % 32 == 0 {
                println!("Epoch {} - Batch_start: {}: Loss = {:.4}", epoch, batch_start, batch_loss.data());
            }


            // SGD update
            for &p in &params {
                let grad = p.grad().unwrap_or(0.0);
                p.set_data(p.data() - lr * grad);
            }

            g.zero_grad();
            g.truncate(checkpoint);
        }

        let accuracy = correct as f64 / train.len() as f64 * 100.0;
        println!(
            "Epoch {}: Loss = {:.4}, Train Accuracy = {:.2}%",
            epoch + 1,
            total_loss / train.len() as f64,
            accuracy
        );

        // Test accuracy
        let mut test_correct = 0;
        for i in 0..test.len() {
            let inputs: Vec<_> = test.images[i].iter().map(|&x| g.variable(x)).collect();
            let logits = mlp.forward(&inputs);
            let pred = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.data().partial_cmp(&b.1.data()).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            if pred == test.labels[i] as usize {
                test_correct += 1;
            }
            g.truncate(checkpoint);
        }
        println!(
            "         Test Accuracy = {:.2}%",
            test_correct as f64 / test.len() as f64 * 100.0
        );
    }
}

#[allow(dead_code)]

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

            let loss = (pred[0]- y_target).pow(2.);
            total_loss = total_loss + loss;
        }

        total_loss.backprop();

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
