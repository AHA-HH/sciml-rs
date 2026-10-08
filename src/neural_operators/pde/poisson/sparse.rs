//! Solver C: bilinear Q1 finite elements on the CGL tensor mesh, solved by unpreconditioned
//! conjugate gradients (design §4.1–4.2, §12 decision 10).
//!
//! C is the validation oracle for solver A: an independent, second-order discretisation on
//! the same CGL nodes (CONVENTIONS §12), so u_A and u_C compare node by node with no
//! interpolation (design §7, error 1). It never produces training labels.
//!
//! The unknowns are the interior nodes, ordered x fastest:
//!
//! p = (i − 1) + (n_x − 2)(j − 1), 1 ≤ i ≤ n_x − 2, 1 ≤ j ≤ n_y − 2.
//!
//! The Dirichlet nodes are eliminated (u = 0 there). With K and M the 1D P1 stiffness and
//! consistent mass restricted to the interior, the matrix is the Q1 stiffness
//! "K_x ⊗ M_y + M_x ⊗ K_y" of design §4.2, which in this ordering and the standard
//! Kronecker convention, (A ⊗ B)[(r, s), (c, t)] = A[r, c] B[s, t] with the B index fastest,
//! is M_y ⊗ K_x + K_y ⊗ M_x. It is symmetric positive definite, 9 points per row.

use ndarray::{Array1, Array2, ArrayView2};
use rlst::FromAij;
use rlst::operator::abstract_operator::Operator;
use rlst::operator::algorithms::conjugate_gradients::CgIteration;
use rlst::operator::space::zero_element;
use rlst::sparse::csr_mat::CsrMatrix;
use rlst::traits::abstract_operator::OperatorBase;
use std::cell::Cell;
use std::fmt;

use crate::neural_operators::chebyshev::nodes;

/// CG stopping tolerance on the relative residual ‖b − Ax‖₂ / ‖b‖₂ (design §12, decision
/// 10). rlst falls back to the absolute residual when b = 0.
pub const CG_TOL: f64 = 1e-6;

/// CG iteration cap per interior unknown: the cap is 10 (n_x − 2)(n_y − 2). Exact
/// arithmetic needs at most one iteration per unknown; Phase 0 T4 needed 2998 of 65 025
/// at n = 257.
const MAX_ITER_PER_UNKNOWN: usize = 10;

/// Q1 finite-element solver for −Δu = f on [−1, 1]², u = 0 on the boundary, on an
/// `[n_x, n_y]` CGL grid stored 'ij' (CONVENTIONS §12; design §4.1–4.2).
///
/// Built once per grid size: [`new`](Self::new) assembles the sparse matrix, and
/// [`solve`](Self::solve) runs CG for each forcing. The validation oracle for
/// [`CollocationSolver`](super::collocation::CollocationSolver); not a label source.
pub struct SparseSolver {
    nx: usize,
    ny: usize,
    /// Q1 stiffness on the interior unknowns, `[(n_x − 2)(n_y − 2)]²`, in the module's
    /// ordering.
    mat: CsrMatrix<f64>,
    /// 1D P1 consistent mass along x on all nodes, `[n_x, n_x]`.
    mass_x: Array2<f64>,
    /// 1D P1 consistent mass along y on all nodes, `[n_y, n_y]`.
    mass_y: Array2<f64>,
}

impl fmt::Debug for SparseSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseSolver")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .finish_non_exhaustive()
    }
}

/// A converged solve of [`SparseSolver::solve`].
#[derive(Debug, Clone)]
pub struct SparseSolution {
    /// u on the full grid, `[n_x, n_y]` 'ij', zero on the boundary rows and columns.
    pub u: Array2<f64>,
    /// CG iterations taken (0 when f = 0 on the interior rows of the load).
    pub iterations: usize,
    /// Final relative residual ‖r‖₂ / ‖b‖₂, at most [`CG_TOL`], with r the residual CG
    /// updates by recurrence (equal to b − Ax up to round-off).
    pub residual: f64,
}

/// CG reached its iteration cap with the relative residual still above the tolerance.
#[derive(Debug, Clone, PartialEq)]
pub struct CgNotConverged {
    /// Iterations taken: the cap, or fewer if CG broke down on a zero or non-finite
    /// search-direction denominator (for example a non-finite forcing, after 0).
    pub iterations: usize,
    /// Final relative residual ‖r‖₂ / ‖b‖₂, r as in [`SparseSolution::residual`].
    pub residual: f64,
    /// The tolerance that was not met, [`CG_TOL`].
    pub tol: f64,
}

impl fmt::Display for CgNotConverged {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CG did not converge: relative residual {:e} above {:e} after {} iterations",
            self.residual, self.tol, self.iterations
        )
    }
}

impl std::error::Error for CgNotConverged {}

impl SparseSolver {
    /// Assembles the Q1 system for an `[n_x, n_y]` CGL grid: the 1D P1 stiffness and
    /// consistent mass along each axis, then the 9-point interior matrix
    /// K_x[i, a] M_y[j, b] + M_x[i, a] K_y[j, b] in CSR.
    ///
    /// # Panics
    /// If `nx < 3` or `ny < 3` (no interior nodes).
    pub fn new(nx: usize, ny: usize) -> Self {
        assert!(
            nx >= 3 && ny >= 3,
            "SparseSolver::new: nx and ny must be >= 3 (interior nodes), got [{nx}, {ny}]"
        );
        let (kx, mass_x) = p1_1d(&nodes(nx));
        let (ky, mass_y) = p1_1d(&nodes(ny));
        let mx = nx - 2;
        let idx = |i: usize, j: usize| (i - 1) + mx * (j - 1);
        let interior_x = 1..nx - 1;
        let interior_y = 1..ny - 1;
        let (mut rows, mut cols, mut data) = (Vec::new(), Vec::new(), Vec::new());
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                for b in j - 1..=j + 1 {
                    for a in i - 1..=i + 1 {
                        if interior_x.contains(&a) && interior_y.contains(&b) {
                            rows.push(idx(i, j));
                            cols.push(idx(a, b));
                            data.push(kx[[i, a]] * mass_y[[j, b]] + mass_x[[i, a]] * ky[[j, b]]);
                        }
                    }
                }
            }
        }
        let size = mx * (ny - 2);
        let mat = CsrMatrix::<f64>::from_aij([size, size], &rows, &cols, &data);
        Self {
            nx,
            ny,
            mat,
            mass_x,
            mass_y,
        }
    }

    /// The CG iteration cap, 10 (n_x − 2)(n_y − 2).
    pub fn max_iter(&self) -> usize {
        MAX_ITER_PER_UNKNOWN * (self.nx - 2) * (self.ny - 2)
    }

    /// Solves for u given nodal f on the full grid.
    ///
    /// The load is the consistent load of the Q1 interpolant of f, (M_x F M_yᵀ) on the
    /// interior rows, i.e. (M_x ⊗ M_y) f; boundary values of f enter through it. CG starts
    /// from u = 0 and stops at relative residual [`CG_TOL`] or after
    /// [`max_iter`](Self::max_iter) iterations.
    ///
    /// `f: [n_x, n_y]`, 'ij' (CONVENTIONS §12). On success the solution is `[n_x, n_y]`
    /// with u = 0 on the boundary.
    ///
    /// # Errors
    /// [`CgNotConverged`] if the relative residual is still above [`CG_TOL`] (or NaN) when
    /// CG stops: at the iteration cap, or earlier if CG breaks down on a zero or non-finite
    /// search-direction denominator (for example a non-finite forcing).
    ///
    /// # Panics
    /// If `f` is not `[n_x, n_y]`.
    pub fn solve(&self, f: ArrayView2<f64>) -> Result<SparseSolution, CgNotConverged> {
        assert_eq!(
            f.dim(),
            (self.nx, self.ny),
            "SparseSolver::solve: forcing must be [nx, ny]"
        );
        let (nx, ny) = (self.nx, self.ny);
        let mx = nx - 2;
        let idx = |i: usize, j: usize| (i - 1) + mx * (j - 1);
        let load = self.mass_x.dot(&f).dot(&self.mass_y.t());

        let op = Operator::from(&self.mat);
        let mut rhs = zero_element(op.range());
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                rhs.imp_mut()[[idx(i, j)]] = load[[i, j]];
            }
        }
        let mut x = zero_element(op.domain());
        // `run()` returns only the residual; the callable runs once per CG update.
        let iterations = Cell::new(0usize);
        let residual = CgIteration::new(&op, &rhs, &mut x)
            .set_tol(CG_TOL)
            .set_max_iter(self.max_iter())
            .set_callable(|_, _| iterations.set(iterations.get() + 1))
            .run();
        let iterations = iterations.get();
        if residual.is_nan() || residual > CG_TOL {
            return Err(CgNotConverged {
                iterations,
                residual,
                tol: CG_TOL,
            });
        }

        let mut u = Array2::zeros((nx, ny));
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                u[[i, j]] = x.imp()[[idx(i, j)]];
            }
        }
        Ok(SparseSolution {
            u,
            iterations,
            residual,
        })
    }
}

/// 1D P1 stiffness K and consistent mass M on all nodes `x` (ascending), `[n, n]` each,
/// tridiagonal and stored dense. Element e = [x_e, x_{e+1}] of width h adds
/// (1/h)[[1, −1], [−1, 1]] to K and (h/6)[[2, 1], [1, 2]] to M.
fn p1_1d(x: &Array1<f64>) -> (Array2<f64>, Array2<f64>) {
    let n = x.len();
    let mut k = Array2::zeros((n, n));
    let mut m = Array2::zeros((n, n));
    for e in 0..n - 1 {
        let h = x[e + 1] - x[e];
        for (r, c, kv, mv) in [
            (e, e, 1.0, 2.0),
            (e + 1, e + 1, 1.0, 2.0),
            (e, e + 1, -1.0, 1.0),
            (e + 1, e, -1.0, 1.0),
        ] {
            k[[r, c]] += kv / h;
            m[[r, c]] += mv * h / 6.0;
        }
    }
    (k, m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::chebyshev::{clenshaw_curtis, rel_l2_error};
    use crate::neural_operators::pde::poisson::collocation::CollocationSolver;
    use crate::neural_operators::pde::poisson::manufactured::{
        EXP, Manufactured, POLY, SIN, sample,
    };
    use ndarray::s;
    use std::f64::consts::PI;

    /// Clenshaw–Curtis relative L² distance of `a` from `b` on the `[nx, ny]` CGL grid.
    fn rel_l2(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let (nx, ny) = a.dim();
        let (wx, wy) = (clenshaw_curtis(nx), clenshaw_curtis(ny));
        rel_l2_error(a.view(), b.view(), wx.view(), wy.view())
    }

    /// Converged solve of `f` on a fresh `[nx, ny]` solver.
    fn solve_c(nx: usize, ny: usize, f: &Array2<f64>) -> SparseSolution {
        SparseSolver::new(nx, ny)
            .solve(f.view())
            .unwrap_or_else(|e| panic!("[{nx}, {ny}]: {e}"))
    }

    /// Relative L² error of solver C on `m`.
    fn error(nx: usize, ny: usize, m: &Manufactured) -> f64 {
        let u = solve_c(nx, ny, &sample(nx, ny, m.f)).u;
        rel_l2(&u, &sample(nx, ny, m.u))
    }

    /// Observed orders log₂(e_n / e_{2n−1}) between consecutive entries of `e`.
    fn orders(e: &[f64]) -> Vec<f64> {
        e.windows(2).map(|w| (w[0] / w[1]).log2()).collect()
    }

    /// `e` as "[1.23e-3, …]".
    fn sci(e: &[f64]) -> String {
        let parts: Vec<String> = e.iter().map(|v| format!("{v:.2e}")).collect();
        format!("[{}]", parts.join(", "))
    }

    fn assert_order_two(name: &str, e: &[f64]) {
        for (k, p) in orders(e).into_iter().enumerate() {
            assert!((1.9..=2.1).contains(&p), "{name}: step {k} has order {p}");
        }
    }

    const NS: [usize; 3] = [33, 65, 129];

    #[test]
    fn q1_is_second_order() {
        for m in [&SIN, &POLY, &EXP] {
            let e: Vec<f64> = NS.iter().map(|&n| error(n, n, m)).collect();
            println!("{:22} L² {}, orders {:.3?}", m.name, sci(&e), orders(&e));
            assert_order_two(m.name, &e);
        }
    }

    #[test]
    fn cg_meets_tolerance() {
        let n = 33;
        let solver = SparseSolver::new(n, n);
        let f = sample(n, n, SIN.f);
        let sol = solver.solve(f.view()).expect("CG");

        // ‖b − Ax‖₂ / ‖b‖₂ recomputed from the dense matrix, not CG's recurrence.
        let a = solver.mat.todense();
        let load = solver.mass_x.dot(&f).dot(&solver.mass_y.t());
        let interior = |a: &Array2<f64>| -> Vec<f64> {
            (1..n - 1)
                .flat_map(|j| (1..n - 1).map(move |i| (i, j)))
                .map(|(i, j)| a[[i, j]])
                .collect()
        };
        let (b, x) = (interior(&load), interior(&sol.u));
        let (mut r2, mut b2) = (0.0_f64, 0.0_f64);
        for p in 0..b.len() {
            let ax: f64 = (0..x.len()).map(|q| a[[p, q]] * x[q]).sum();
            r2 += (b[p] - ax).powi(2);
            b2 += b[p].powi(2);
        }
        let true_residual = (r2 / b2).sqrt();
        println!(
            "n = {n}: {} iterations, residual {:.3e} (recomputed {true_residual:.3e})",
            sol.iterations, sol.residual
        );
        assert!(sol.residual <= CG_TOL, "{:e}", sol.residual);
        assert!(true_residual <= CG_TOL, "recomputed {true_residual:e}");
        assert!(sol.iterations > 0 && sol.iterations < solver.max_iter());
    }

    #[test]
    fn matrix_is_symmetric() {
        let a = SparseSolver::new(9, 9).mat.todense();
        let [r, c] = a.shape();
        assert_eq!((r, c), (49, 49));
        let mut max = 0.0_f64;
        let mut asym = 0.0_f64;
        for p in 0..r {
            for q in 0..c {
                max = max.max(a[[p, q]].abs());
                asym = asym.max((a[[p, q]] - a[[q, p]]).abs());
            }
        }
        assert!(asym <= 1e-14 * max, "max |K − Kᵀ| = {asym:e}");
    }

    /// (a ⊗ b)[(r, s), (c, t)] = a[r, c] b[s, t], the b index fastest.
    fn kron(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let ((ar, ac), (br, bc)) = (a.dim(), b.dim());
        Array2::from_shape_fn((ar * br, ac * bc), |(p, q)| {
            a[[p / br, q / bc]] * b[[p % br, q % bc]]
        })
    }

    #[test]
    fn kronecker_ordering() {
        // A non-square grid, so a swapped axis or a y-fastest ordering changes the matrix.
        let (nx, ny) = (9, 7);
        let (kx, mx) = p1_1d(&nodes(nx));
        let (ky, my) = p1_1d(&nodes(ny));
        let int = |a: &Array2<f64>, n: usize| a.slice(s![1..n - 1, 1..n - 1]).to_owned();
        let (kx, mx, ky, my) = (int(&kx, nx), int(&mx, nx), int(&ky, ny), int(&my, ny));
        let expected = kron(&my, &kx) + kron(&ky, &mx);
        let a = SparseSolver::new(nx, ny).mat.todense();
        assert_eq!(a.shape(), [expected.nrows(), expected.ncols()]);
        let scale = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for ((p, q), v) in expected.indexed_iter() {
            assert!(
                (a[[p, q]] - v).abs() <= 1e-14 * scale,
                "({p}, {q}): {} against {v}",
                a[[p, q]]
            );
        }
    }

    #[test]
    fn rectangular_grid() {
        let (e33, e65) = (error(33, 33, &EXP), error(65, 65, &EXP));
        for (nx, ny) in [(33, 65), (65, 33)] {
            let e = error(nx, ny, &EXP);
            println!(
                "{} on [{nx}, {ny}]: L² {e:.2e} (square: {e33:.2e}, {e65:.2e})",
                EXP.name
            );
            assert!(e65 < e && e < e33, "[{nx}, {ny}]: {e:e}");
        }
    }

    /// Σ a_kl sin(kπ(x + 1)/2) sin(lπ(y + 1)/2): zero on ∂[−1, 1]².
    fn sine_series(modes: &[(u32, u32, f64)]) -> impl Fn(f64, f64) -> f64 + '_ {
        move |x, y| {
            modes
                .iter()
                .map(|&(k, l, a)| {
                    a * (f64::from(k) * PI * (x + 1.0) / 2.0).sin()
                        * (f64::from(l) * PI * (y + 1.0) / 2.0).sin()
                })
                .sum()
        }
    }

    #[test]
    fn agrees_with_collocation() {
        // Hand-chosen low modes (k, l, a_kl); the GRF sampler is Phase 2.
        let forcings: [&[(u32, u32, f64)]; 3] = [
            &[(1, 1, 1.0), (2, 1, 0.5), (1, 3, -0.3)],
            &[(2, 2, 1.0), (3, 1, 0.4), (1, 4, 0.25)],
            &[(1, 2, 0.8), (3, 3, -0.5), (4, 2, 0.2)],
        ];
        for (k, modes) in forcings.iter().enumerate() {
            let e: Vec<f64> = NS
                .iter()
                .map(|&n| {
                    let f = sample(n, n, sine_series(modes));
                    let u_a = CollocationSolver::new(n, n).solve_full(f.view());
                    let u_c = solve_c(n, n, &f).u;
                    rel_l2(&u_c, &u_a)
                })
                .collect();
            println!(
                "forcing {k}: ‖u_A − u_C‖ L² {}, orders {:.3?}",
                sci(&e),
                orders(&e)
            );
            assert_order_two(&format!("forcing {k}"), &e);
        }
    }

    #[test]
    #[should_panic(expected = "nx and ny must be >= 3")]
    fn too_small_grid_panics() {
        SparseSolver::new(2, 9);
    }

    #[test]
    #[should_panic(expected = "forcing must be [nx, ny]")]
    fn solve_wrong_shape_panics() {
        let _ = SparseSolver::new(9, 7).solve(Array2::zeros((7, 9)).view());
    }
}
