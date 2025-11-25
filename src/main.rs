#![allow(dead_code)]
use std::collections::HashSet;
type VariableIdx = usize;

struct Context {
    vars: Vec<Variable>,
}

impl Context {
    pub fn new() -> Self {
        Self { vars: Vec::new() }
    }

    pub fn var(&mut self, data: f64) -> VariableIdx {
        let new_var = Variable::new(data);
        self.vars.push(new_var);
        self.vars.len() - 1
    }

    // Helper to construct and return new variable
    fn push_var(&mut self, children: Vec<VariableIdx>, op: Op) -> VariableIdx {
        let children_data: Vec<f64> = children.iter().map(|&c| self.vars[c].data).collect();
        let data = op.forward(&children_data);
        self.vars.push(Variable {
            data,
            grad: None,
            children,
            op,
        });
        self.vars.len() - 1
    }

    pub fn add(&mut self, a: VariableIdx, b: VariableIdx) -> VariableIdx {
        self.push_var(vec![a, b], Op::Add)
    }

    pub fn mul(&mut self, a: VariableIdx, b: VariableIdx) -> VariableIdx {
        self.push_var(vec![a, b], Op::Mul)
    }

    pub fn pow(&mut self, a: VariableIdx, exp: f64) -> VariableIdx {
        self.push_var(vec![a], Op::Pow(exp))
    }

    pub fn relu(&mut self, a: VariableIdx) -> VariableIdx {
        self.push_var(vec![a], Op::ReLU)
    }

    pub fn neg(&mut self, a: VariableIdx) -> VariableIdx {
        let minus_one = self.var(-1.0);
        self.mul(a, minus_one)
    }

    pub fn sub(&mut self, a: VariableIdx, b: VariableIdx) -> VariableIdx {
        let neg_b = self.neg(b);
        self.add(a, neg_b)
    }

    pub fn div(&mut self, a: VariableIdx, b: VariableIdx) -> VariableIdx {
        let b_inv = self.pow(b, -1.0);
        self.mul(a, b_inv)
    }

    fn add_grad(&mut self, idx: VariableIdx, delta: f64) {
        let grad = &mut self.vars[idx].grad;
        *grad = Some(grad.unwrap_or(0.0) + delta);
    }

    fn backward(&mut self, a: VariableIdx) {
        let var = &self.vars[a];
        let children_data: Vec<f64> = var.children.iter().map(|&c| self.vars[c].data).collect();
        let grads = var
            .op
            .backward(&children_data, var.data, var.grad.unwrap_or(0.0));
        let children = var.children.clone();

        for (child, grad) in children.iter().zip(grads) {
            self.add_grad(*child, grad);
        }
    }

    pub fn backprop(&mut self, a: VariableIdx) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo(
            v: VariableIdx,
            topo: &mut Vec<VariableIdx>,
            visited: &mut HashSet<VariableIdx>,
            ctx: &Context,
        ) {
            if !visited.contains(&v) {
                visited.insert(v);
                for c in &ctx.vars[v].children {
                    build_topo(*c, topo, visited, ctx);
                }
                topo.push(v);
            }
        }

        build_topo(a, &mut topo, &mut visited, self);
        self.vars[a].grad = Some(1.0);

        for v in topo.iter().rev() {
            self.backward(*v);
        }
    }
}

#[derive(Debug, Copy, Clone)]
enum Op {
    Value,
    Add,
    Mul,
    Pow(f64),
    ReLU,
}

impl Op {
    fn forward(&self, children_data: &[f64]) -> f64 {
        match self {
            Op::Add => children_data[0] + children_data[1],
            Op::Mul => children_data[0] * children_data[1],
            Op::Pow(exp) => children_data[0].powf(*exp),
            Op::ReLU => {
                if children_data[0] > 0.0 {
                    children_data[0]
                } else {
                    0.0
                }
            }
            Op::Value => unimplemented!(),
        }
    }
    fn backward(&self, children_data: &[f64], out_data: f64, out_grad: f64) -> Vec<f64> {
        match self {
            Op::Add => vec![out_grad, out_grad],
            Op::Mul => vec![children_data[1] * out_grad, children_data[0] * out_grad],
            Op::Pow(exp) => vec![exp * children_data[0].powf(exp - 1.0) * out_grad],
            Op::ReLU => vec![if out_data > 0.0 { out_grad } else { 0.0 }],
            Op::Value => vec![],
        }
    }
}

#[derive(Debug, Clone)]
struct Variable {
    data: f64,
    grad: Option<f64>,

    // Autograd graph
    children: Vec<VariableIdx>,
    op: Op,
}

impl Variable {
    fn new(data: f64) -> Self {
        Self {
            data,
            grad: None,
            children: Vec::new(),
            op: Op::Value,
        }
    }
}

#[cfg(test)]
mod tests;

fn main() {
    println!("Run `cargo test` to run the tests");
}
