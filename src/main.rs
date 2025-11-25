use std::collections::HashSet;

type VariableIdx = usize;

struct Context {
    vars: Vec<Variable>,
}

impl Context {
    fn new() -> Self {
        Self { vars: Vec::new() }
    }

    fn var(&mut self, data: f64) -> VariableIdx {
        let new_var = Variable::new(data);
        self.vars.push(new_var);
        self.vars.len() - 1
    }

    fn add(&mut self, a: VariableIdx, b: VariableIdx) -> VariableIdx {
        let va = &self.vars[a];
        let vb = &self.vars[b];
        let new_var = Variable {
            data: va.data + vb.data,
            grad: None,
            children: vec![a, b],
            op: Op::Add,
        };

        self.vars.push(new_var);
        self.vars.len() - 1
    }

    fn mul(&mut self, a: VariableIdx, b: VariableIdx) -> VariableIdx {
        let va = &self.vars[a];
        let vb = &self.vars[b];
        let new_var = Variable {
            data: va.data * vb.data,
            grad: None,
            children: vec![a, b],
            op: Op::Mul,
        };

        self.vars.push(new_var);
        self.vars.len() - 1
    }

    fn pow(&mut self, a: VariableIdx, exp: f64) -> VariableIdx {
        let va = &self.vars[a];
        let new_var = Variable {
            data: va.data.powf(exp),
            grad: None,
            children: vec![a],
            op: Op::Pow(exp),
        };

        self.vars.push(new_var);
        self.vars.len() - 1
    }

    fn relu(&mut self, a: VariableIdx) -> VariableIdx {
        let va = &self.vars[a];
        let new_var = Variable {
            data: if va.data > 0.0 { va.data } else { 0.0 },
            grad: None,
            children: vec![a],
            op: Op::ReLU,
        };

        self.vars.push(new_var);
        self.vars.len() - 1
    }

    fn neg(&mut self, a: VariableIdx) -> VariableIdx {
        let minus_one = self.var(-1.0);
        self.mul(a, minus_one)
    }

    fn sub(&mut self, a: VariableIdx, b: VariableIdx) -> VariableIdx {
        let neg_b = self.neg(b);
        self.add(a, neg_b)
    }

    fn div(&mut self, a: VariableIdx, b: VariableIdx) -> VariableIdx {
        let b_inv = self.pow(b, -1.0);
        self.mul(a, b_inv)
    }

    fn add_grad(&mut self, idx: VariableIdx, delta: f64) {
        let grad = &mut self.vars[idx].grad;
        *grad = Some(grad.unwrap_or(0.0) + delta);
    }

    fn backward(&mut self, a: VariableIdx) {
        let Variable {
            data,
            grad,
            children,
            op,
        } = self.vars[a].clone();

        let grad = grad.unwrap_or(0.0);

        match op {
            Op::Add => {
                self.add_grad(children[0], grad);
                self.add_grad(children[1], grad);
            }
            Op::Mul => {
                let data0 = self.vars[children[0]].data;
                let data1 = self.vars[children[1]].data;

                self.add_grad(children[0], data1 * grad);
                self.add_grad(children[1], data0 * grad);
            }
            Op::Pow(exp) => {
                // d/dx (x^n) = n * x^(n-1)
                let base_data = self.vars[children[0]].data;
                self.add_grad(children[0], exp * base_data.powf(exp - 1.0) * grad);
            }
            Op::ReLU => {
                self.add_grad(children[0], if data > 0.0 { grad } else { 0.0 });
            }
            Op::Value => {}
        }
    }

    fn backprop(&mut self, a: VariableIdx) {
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

fn main() {
    let mut ctx = Context::new();
    let a = ctx.var(1.0);
    let b = ctx.var(2.);
    let c = ctx.var(3.);
    let d = ctx.add(a, b);
    let e = ctx.mul(d, c);
    let f = ctx.div(e, b);
    let g = ctx.sub(f, c);
    let h = ctx.relu(g);

    ctx.backprop(h);

    println!("{:?}", ctx.vars);
    dbg!(
        &ctx.vars[a],
        &ctx.vars[b],
        &ctx.vars[c],
        &ctx.vars[d],
        &ctx.vars[e],
        &ctx.vars[f],
        &ctx.vars[g],
        &ctx.vars[h],
    );
}
