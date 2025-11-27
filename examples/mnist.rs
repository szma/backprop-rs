#[path = "dataloader/mnist_loader.rs"]
mod mnist_loader;

fn main() {
    mnist();
}

fn mnist() {
    use backprop_rs::graph::Graph;
    use mnist_loader::MnistData;
    use std::path::Path;

    // Load MNIST data
    let train = MnistData::load(
        Path::new("examples/data/train-images-idx3-ubyte"),
        Path::new("examples/data/train-labels-idx1-ubyte"),
    )
    .expect("Failed to load MNIST training data");

    let test = MnistData::load(
        Path::new("examples/data/t10k-images-idx3-ubyte"),
        Path::new("examples/data/t10k-labels-idx1-ubyte"),
    )
    .expect("Failed to load MNIST test data");

    println!(
        "Loaded {} training, {} test images",
        train.len(),
        test.len()
    );

    let g = Graph::new();
    let mlp = g.mlp(784, vec![16, 10]);
    let params = mlp.parameters();
    let checkpoint = g.len();

    println!("Model has {} parameters", params.len());

    let lr = 0.01;
    let batch_size = 32;
    let epochs = 10;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;

        let num_samples = 1000; // train.len()
        for batch_start in (0..num_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(num_samples);
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
            batch_loss.backward();

            if batch_start % 32 == 0 {
                println!(
                    "Epoch {} - Batch_start: {}: Loss = {:.4}",
                    epoch,
                    batch_start,
                    batch_loss.data()
                );
            }

            // SGD update
            for &p in &params {
                let grad = p.grad().unwrap_or(0.0);
                p.set_data(p.data() - lr * grad);
            }

            g.zero_grad();
            g.truncate(checkpoint);
        }

        let accuracy = correct as f64 / num_samples as f64 * 100.0;
        println!(
            "Epoch {}: Loss = {:.4}, Train Accuracy = {:.2}%",
            epoch + 1,
            total_loss / num_samples as f64,
            accuracy
        );

        // Test accuracy
        let mut test_correct = 0;
        let num_test_samples = 1000;
        for i in 0..num_test_samples {
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
            test_correct as f64 / num_test_samples as f64 * 100.0
        );
    }
}
