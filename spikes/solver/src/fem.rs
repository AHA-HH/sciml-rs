//! Option C: second-order sparse SPD discretisations on the CGL tensor mesh, solved by
//! unpreconditioned CG (design §4.1). Unknowns are the interior nodes, p = (i − 1) +
//! (n − 2)(j − 1); the Dirichlet nodes are eliminated (u = 0 there).

use ndarray::{Array1, Array2};
use rlst::FromAij;
use rlst::operator::abstract_operator::Operator;
use rlst::operator::algorithms::conjugate_gradients::CgIteration;
use rlst::operator::space::zero_element;
use rlst::sparse::csr_mat::CsrMatrix;
use rlst::traits::abstract_operator::OperatorBase;
use std::cell::Cell;

/// Which option-C discretisation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// Vertex-centred finite volume, 5-point, lumped (dual-cell) load.
    Fv,
    /// Bilinear Q1 finite elements, 9-point, consistent mass and consistent load.
    Q1,
}

impl Variant {
    pub fn name(self) -> &'static str {
        match self {
            Variant::Fv => "C-fv",
            Variant::Q1 => "C-q1",
        }
    }
}

/// An assembled system: matrix in CSR, plus the 1D pieces needed for the load.
pub struct System {
    pub variant: Variant,
    pub n: usize,
    pub mat: CsrMatrix<f64>,
    /// Dual-cell widths Δ_i = (x_{i+1} − x_{i−1}) / 2 (FV load).
    dual: Array1<f64>,
    /// Full 1D P1 consistent mass on all n nodes (Q1 load).
    mass: Array2<f64>,
}

/// 1D P1 stiffness and consistent mass on all n nodes (tridiagonal, stored dense).
fn p1_1d(x: &Array1<f64>) -> (Array2<f64>, Array2<f64>) {
    let n = x.len();
    let mut k = Array2::zeros((n, n));
    let mut m = Array2::zeros((n, n));
    for e in 0..n - 1 {
        let h = x[e + 1] - x[e];
        k[[e, e]] += 1.0 / h;
        k[[e + 1, e + 1]] += 1.0 / h;
        k[[e, e + 1]] -= 1.0 / h;
        k[[e + 1, e]] -= 1.0 / h;
        m[[e, e]] += h / 3.0;
        m[[e + 1, e + 1]] += h / 3.0;
        m[[e, e + 1]] += h / 6.0;
        m[[e + 1, e]] += h / 6.0;
    }
    (k, m)
}

impl System {
    /// Assemble on n × n CGL nodes `x` (same nodes in x and y).
    pub fn assemble(variant: Variant, x: &Array1<f64>) -> Self {
        let n = x.len();
        let m = n - 2;
        let idx = |i: usize, j: usize| (i - 1) + m * (j - 1);
        let (k1, m1) = p1_1d(x);
        let dual = Array1::from_shape_fn(n, |i| {
            if i == 0 || i == n - 1 {
                0.0
            } else {
                (x[i + 1] - x[i - 1]) / 2.0
            }
        });
        let (mut rows, mut cols, mut data) = (Vec::new(), Vec::new(), Vec::new());
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let p = idx(i, j);
                match variant {
                    Variant::Fv => {
                        // −∫ over the dual cell of ∂x(∂x u) ≈ Δy_j [flux differences],
                        // fluxes (u_{i+1} − u_i)/(x_{i+1} − x_i); likewise in y.
                        let (hl, hr) = (x[i] - x[i - 1], x[i + 1] - x[i]);
                        let (kl, kr) = (x[j] - x[j - 1], x[j + 1] - x[j]);
                        let diag =
                            dual[j] * (1.0 / hl + 1.0 / hr) + dual[i] * (1.0 / kl + 1.0 / kr);
                        rows.push(p);
                        cols.push(p);
                        data.push(diag);
                        let nb = [
                            (i - 1, j, -dual[j] / hl),
                            (i + 1, j, -dual[j] / hr),
                            (i, j - 1, -dual[i] / kl),
                            (i, j + 1, -dual[i] / kr),
                        ];
                        for (a, b, v) in nb {
                            if (1..n - 1).contains(&a) && (1..n - 1).contains(&b) {
                                rows.push(p);
                                cols.push(idx(a, b));
                                data.push(v);
                            }
                        }
                    }
                    Variant::Q1 => {
                        // K = K_x ⊗ M_y + M_x ⊗ K_y, each 1D factor tridiagonal.
                        for b in j - 1..=j + 1 {
                            for a in i - 1..=i + 1 {
                                if (1..n - 1).contains(&a) && (1..n - 1).contains(&b) {
                                    let v = k1[[i, a]] * m1[[j, b]] + m1[[i, a]] * k1[[j, b]];
                                    rows.push(p);
                                    cols.push(idx(a, b));
                                    data.push(v);
                                }
                            }
                        }
                    }
                }
            }
        }
        let mat = CsrMatrix::<f64>::from_aij([m * m, m * m], &rows, &cols, &data);
        System {
            variant,
            n,
            mat,
            dual,
            mass: m1,
        }
    }

    /// Load vector from nodal f on all n × n nodes. FV: f_ij Δx_i Δy_j. Q1: the
    /// consistent load of the Q1 interpolant of f, (M_x F M_y)_ij.
    pub fn load(&self, f: &Array2<f64>) -> Vec<f64> {
        let n = self.n;
        let m = n - 2;
        let full = match self.variant {
            Variant::Fv => {
                Array2::from_shape_fn((n, n), |(i, j)| f[[i, j]] * self.dual[i] * self.dual[j])
            }
            Variant::Q1 => self.mass.dot(f).dot(&self.mass),
        };
        let mut b = vec![0.0; m * m];
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                b[(i - 1) + m * (j - 1)] = full[[i, j]];
            }
        }
        b
    }

    /// Solve with CG from a zero initial guess. Returns the n × n solution (zero
    /// boundary), the iteration count and the final relative residual.
    pub fn solve(&self, f: &Array2<f64>, tol: f64) -> (Array2<f64>, usize, f64) {
        let n = self.n;
        let m = n - 2;
        let b = self.load(f);
        let op = Operator::from(&self.mat);
        let mut rhs = zero_element(op.range());
        for (p, v) in b.iter().enumerate() {
            rhs.imp_mut()[[p]] = *v;
        }
        let mut x = zero_element(op.domain());
        let iters = Cell::new(0usize);
        let residual = CgIteration::new(&op, &rhs, &mut x)
            .set_tol(tol)
            .set_max_iter(50 * m * m)
            .set_callable(|_, _| iters.set(iters.get() + 1))
            .run();
        let mut u = Array2::zeros((n, n));
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                u[[i, j]] = x.imp()[[(i - 1) + m * (j - 1)]];
            }
        }
        (u, iters.get(), residual)
    }

    /// Dense copy of the matrix (tests only; small n).
    pub fn dense(&self) -> Array2<f64> {
        let m = self.n - 2;
        let mut a = Array2::zeros((m * m, m * m));
        for p in 0..m * m {
            let mut e = vec![0.0; m * m];
            e[p] = 1.0;
            let op = Operator::from(&self.mat);
            let mut v = zero_element(op.domain());
            for (q, val) in e.iter().enumerate() {
                v.imp_mut()[[q]] = *val;
            }
            let av = op.dot(&v);
            for q in 0..m * m {
                a[[q, p]] = av.imp()[[q]];
            }
        }
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use transfers_spike::grids::cheb_nodes;

    #[test]
    fn matrices_symmetric_positive_diagonal() {
        let x = cheb_nodes(9);
        for v in [Variant::Fv, Variant::Q1] {
            let a = System::assemble(v, &x).dense();
            let size = a.nrows();
            for p in 0..size {
                assert!(a[[p, p]] > 0.0, "{v:?}: diag {p}");
                for q in 0..size {
                    assert!((a[[p, q]] - a[[q, p]]).abs() < 1e-14, "{v:?}: ({p},{q})");
                }
            }
        }
    }

    // CG converges and the callback counts iterations: for an SPD system of size N, CG
    // in exact arithmetic needs at most N steps.
    #[test]
    fn cg_counts_iterations() {
        let x = cheb_nodes(9);
        let sys = System::assemble(Variant::Fv, &x);
        let f = Array2::from_elem((9, 9), 1.0);
        let (_, iters, res) = sys.solve(&f, 1e-12);
        assert!(res < 1e-12, "{res:e}");
        assert!(iters > 0 && iters <= 49 + 5, "{iters}");
    }
}
