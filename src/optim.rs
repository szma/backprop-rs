use crate::graph::Variable;

pub fn stochastic_gradiant_descent(parameters: &mut [Variable<'_>], lr: f64) {
    for p in parameters.iter_mut() {
        let grad = p.grad().unwrap_or_default();
        p.set_data(p.data() - lr * grad);
    }
}
