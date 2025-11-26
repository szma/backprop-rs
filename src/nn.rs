use crate::syntax::Var;

pub struct Neuron<'a> {
    w: Vec<Var<'a>>,
    b: Var<'a>,
    nonlin: bool,
}

impl<'a> Neuron<'a> {
    pub fn new(var: impl Fn(f64) -> Var<'a>, nin: i16, nonlin: bool) -> Self {
        // He-Initialisierung: scale = sqrt(2 / fan_in)
        let scale = (2.0 / nin as f64).sqrt();
        let w = (0..nin)
            .map(|_| var((rand::random::<f64>() * 2. - 1.) * scale))
            .collect();
        Self {
            w,
            b: var(0.0),
            nonlin,
        }
    }

    pub fn forward(&self, x: &[Var<'a>]) -> Var<'a> {
        let mut s = self.b;
        for (&wi, &xi) in self.w.iter().zip(x) {
            s = s + wi * xi;
        }
        if self.nonlin {
            s = s.relu()
        }
        s
    }

    pub fn parameters(&self) -> Vec<Var<'a>> {
        let mut params = self.w.clone();
        params.push(self.b);
        params
    }
}

pub struct Layer<'a> {
    neurons: Vec<Neuron<'a>>,
}

impl<'a> Layer<'a> {
    pub fn new(var: impl Fn(f64) -> Var<'a>, nin: i16, nout: i16, nonlin: bool) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(&var, nin, nonlin)).collect();
        Self { neurons }
    }

    pub fn forward(&self, x: &[Var<'a>]) -> Vec<Var<'a>> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }

    pub fn parameters(&self) -> Vec<Var<'a>> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

pub struct MLP<'a> {
    layers: Vec<Layer<'a>>,
}

impl<'a> MLP<'a> {
    pub fn new(var: impl Fn(f64) -> Var<'a>, nin: i16, nouts: Vec<i16>) -> Self {
        let n = nouts.len();
        let mut layers = Vec::new();

        layers.push(Layer::new(&var, nin, nouts[0], n > 1)); // nonlin if not last

        for i in 1..n {
            let is_last = i == n - 1;
            layers.push(Layer::new(&var, nouts[i - 1], nouts[i], !is_last));
        }

        Self { layers }
    }

    pub fn forward(&self, x: &[Var<'a>]) -> Vec<Var<'a>> {
        let mut out = x.to_vec();
        for layer in self.layers.iter() {
            out = layer.forward(&out);
        }
        out
    }

    pub fn parameters(&self) -> Vec<Var<'a>> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
