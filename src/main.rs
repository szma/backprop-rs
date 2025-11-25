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
    fn push_var(&mut self, data: f64, children: Vec<VariableIdx>, op: Op) -> VariableIdx {
        self.vars.push(Variable {
            data,
            grad: None,
            children,
            op,
        });
        self.vars.len() - 1
    }

    pub fn add(&mut self, a: VariableIdx, b: VariableIdx) -> VariableIdx {
        let data = self.vars[a].data + self.vars[b].data;
        self.push_var(data, vec![a, b], Op::Add)
    }

    pub fn mul(&mut self, a: VariableIdx, b: VariableIdx) -> VariableIdx {
        let data = self.vars[a].data * self.vars[b].data;
        self.push_var(data, vec![a, b], Op::Mul)
    }

    pub fn pow(&mut self, a: VariableIdx, exp: f64) -> VariableIdx {
        let data = self.vars[a].data.powf(exp);
        self.push_var(data, vec![a], Op::Pow(exp))
    }

    pub fn relu(&mut self, a: VariableIdx) -> VariableIdx {
        let adata = self.vars[a].data;
        let data = if adata > 0.0 { adata } else { 0.0 };
        self.push_var(data, vec![a], Op::ReLU)
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
    // Test: b = a*a + a, where a=3
    // b = 9 + 3 = 12
    // db/da = 2a + 1 = 7
    let mut ctx = Context::new();
    let a = ctx.var(3.0);
    let a_sq = ctx.mul(a, a);  // a*a = 9
    let b = ctx.add(a_sq, a);  // a*a + a = 12

    ctx.backprop(b);

    dbg!(&ctx.vars[a], &ctx.vars[a_sq], &ctx.vars[b]);
    // Expected: a.grad = 7 (2*3 + 1)
}
