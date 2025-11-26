use crate::engine::Context;

use crate::engine::VariableIdx;


pub struct Neuron {
    w: Vec<VariableIdx>,
    b: VariableIdx,
    nonlin: bool,
}

impl Neuron {

    pub fn new(ctx: &mut Context, nin: i16, nonlin: bool) -> Self {
        let mut w = Vec::new();
        for _ in 0..nin {
            w.push(ctx.var(rand::random::<f64>() * 2. - 1.));
        }
        Self { w, b: ctx.var(0.0), nonlin }
    }
    
    pub fn forward(&self, ctx: &mut Context, x: &[VariableIdx]) -> VariableIdx{
        let mut s = self.b;
        for (wi, xi) in self.w.iter().zip(x) {
            let prod = ctx.mul(*wi, *xi);
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
        let mut neurons = Vec::new();
        for _ in 0..nout {
            neurons.push(Neuron::new(ctx, nin, nonlin));
        }

        Self { neurons }
    }

    pub fn forward(&self, ctx: &mut Context, x: &[VariableIdx]) -> Vec<VariableIdx> {
        let mut out = Vec::new();
        for neuron in self.neurons.iter() {
            out.push(neuron.forward(ctx, x));
        }
        out
    }

    pub fn parameters(&self) -> Vec<VariableIdx> {
        let mut params = Vec::new();
        for neuron in self.neurons.iter() {
            params.extend(neuron.parameters());
        }

        params
    }
}

pub struct MLP {
    layers: Vec<Layer>
}

impl MLP {
    pub fn new(ctx: &mut Context, nin: i16, nouts: Vec<i16>) -> Self {
        let mut layers = Vec::new();
        let n = nouts.len();
        
        layers.push(Layer::new(ctx, nin, nouts[0], n > 1)); // nonlin if not last
        
        for i in 1..n {
            let is_last = i == n - 1;
            layers.push(Layer::new(ctx, nouts[i-1], nouts[i], !is_last));
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
        let mut params = Vec::new();
        for layer in self.layers.iter() {
            params.extend(layer.parameters());
        }
        params
    }
}