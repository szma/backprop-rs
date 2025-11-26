use crate::engine::Context;
use crate::engine::VariableIdx;

pub struct Neuron {
    w: Vec<VariableIdx>,
    b: VariableIdx,
    nonlin: bool,
}

impl Neuron {
    pub fn new(ctx: &mut Context, nin: i16, nonlin: bool) -> Self {
        let scale = (2.0 / nin as f64).sqrt();
        let w = (0..nin)
            .map(|_| ctx.var((rand::random::<f64>() * 2. - 1.) * scale))
            .collect();
        Self {
            w,
            b: ctx.var(0.0),
            nonlin,
        }
    }

    pub fn forward(&self, ctx: &mut Context, x: &[VariableIdx]) -> VariableIdx {
        let mut s = self.b;
        for (&wi, &xi) in self.w.iter().zip(x) {
            let prod = ctx.mul(wi, xi);
            s = ctx.add(s, prod);
        }
        if self.nonlin {
            s = ctx.relu(s)
        }
        s
    }

    pub fn parameters(&self) -> Vec<VariableIdx> {
        let mut params = self.w.clone();
        params.push(self.b);
        params
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(ctx: &mut Context, nin: i16, nout: i16, nonlin: bool) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(ctx, nin, nonlin)).collect();
        Self { neurons }
    }

    pub fn forward(&self, ctx: &mut Context, x: &[VariableIdx]) -> Vec<VariableIdx> {
        self.neurons.iter().map(|n| n.forward(ctx, x)).collect()
    }

    pub fn parameters(&self) -> Vec<VariableIdx> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(ctx: &mut Context, nin: i16, nouts: Vec<i16>) -> Self {
        let n = nouts.len();
        let mut layers = Vec::new();

        layers.push(Layer::new(ctx, nin, nouts[0], n > 1));

        for i in 1..n {
            let is_last = i == n - 1;
            layers.push(Layer::new(ctx, nouts[i - 1], nouts[i], !is_last));
        }

        Self { layers }
    }

    pub fn forward(&self, ctx: &mut Context, x: &[VariableIdx]) -> Vec<VariableIdx> {
        let mut out = x.to_vec();
        for layer in self.layers.iter() {
            out = layer.forward(ctx, &out);
        }
        out
    }

    pub fn parameters(&self) -> Vec<VariableIdx> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
