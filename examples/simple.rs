use backprop_rs::{graph::Graph, optim::stochastic_gradiant_descent};

fn main() {
    let g = Graph::new();

    let a = g.variable(1.0);
    let b = g.variable(2.0);

    for _ in 0..15 {
        let loss = (a - b).pow(2.);

        g.zero_grad();
        loss.backward();

        println!("_|_data_|_grad_");
        println!("b|{:.2} | {:?}", b.data(), b.grad());
        println!("a|{:.2} | {:?}", a.data(), a.grad());

        stochastic_gradiant_descent(&mut [a, b], 0.1);
    }
}
