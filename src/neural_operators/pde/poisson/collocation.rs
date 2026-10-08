//! Solver A: Chebyshev collocation solved by fast diagonalisation (design §4.1–4.2).
//!
//! On the interior CGL nodes (CONVENTIONS §12), −Δu = f with u = 0 on the boundary is
//! the separable equation
//!
//! −D_xx U − U D_yyᵀ = F,
//!
//! with D_xx, D_yy the interior blocks of D² along x and y. Each block is diagonalised
//! once, D_xx = V_x Λ V_x⁻¹ and D_yy = V_y M V_y⁻¹, so a solve is four matrix products
//! and a pointwise division (design §12, decision 10).
//!
//! V⁻¹ comes from the left eigenvectors, not from an LU-based inverse: OpenBLAS's
//! threaded `getrf` (m·n ≥ 10⁴, so n ≥ 103 here) can put a scratch array on the calling
//! thread's stack and overflow a 2 MB thread, as test and rayon worker threads have.

use ndarray::{Array2, ArrayView2, s};
use rlst::dense::linalg::lapack::eigenvalue_decomposition::EigMode;
use rlst::{DynArray, EigenvalueDecomposition};

use crate::neural_operators::chebyshev::diff2_matrix;

/// Largest imaginary part accepted in the eigendecomposition of D_xx or D_yy: relative
/// to the largest |λ| for eigenvalues, absolute for the entries of the unit-norm
/// eigenvectors. Phase 0 T4 measured exactly 0.
const IMAG_TOL: f64 = 1e-10;

/// Smallest accepted |u_jᵀ v_j| for unit-norm left and right eigenvectors, i.e. the
/// reciprocal of the largest accepted eigenvalue condition number.
const BIORTH_TOL: f64 = 1e-12;

/// Collocation solver for −Δu = f on [−1, 1]², u = 0 on the boundary, on an
/// `[n_x, n_y]` CGL grid stored 'ij' (CONVENTIONS §12; design §4.1–4.2).
///
/// Built once per grid size; [`solve`](Self::solve) and
/// [`solve_full`](Self::solve_full) reuse the eigendecomposition for every forcing.
pub struct CollocationSolver {
    nx: usize,
    ny: usize,
    /// Eigenvectors of D_xx, `[n_x − 2, n_x − 2]`.
    vx: DynArray<f64, 2>,
    /// V_x⁻¹.
    vx_inv: DynArray<f64, 2>,
    /// V_yᵀ, `[n_y − 2, n_y − 2]`.
    vy_t: DynArray<f64, 2>,
    /// V_y⁻ᵀ.
    vy_inv_t: DynArray<f64, 2>,
    /// 1 / −(λ_i + μ_j), `[n_x − 2, n_y − 2]`.
    inv_denom: DynArray<f64, 2>,
}

impl std::fmt::Debug for CollocationSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollocationSolver")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .finish_non_exhaustive()
    }
}

impl CollocationSolver {
    /// Builds the solver for an `[n_x, n_y]` CGL grid: takes the interior blocks of
    /// [`diff2_matrix`] and eigendecomposes each, O(n³) once.
    ///
    /// # Panics
    /// If `nx < 3` or `ny < 3` (no interior nodes); if an eigenvalue has an imaginary
    /// part above 1e-10 · max|λ|, or an entry of a unit-norm left or right eigenvector
    /// one above 1e-10; if a left and right eigenvector pair has |u_jᵀ v_j| below 1e-12
    /// (eigenvalue condition number above 1e12); or if LAPACK fails.
    pub fn new(nx: usize, ny: usize) -> Self {
        assert!(
            nx >= 3 && ny >= 3,
            "CollocationSolver::new: nx and ny must be >= 3 (interior nodes), got [{nx}, {ny}]"
        );
        let ex = real_eig(&interior_d2(nx));
        let ey = real_eig(&interior_d2(ny));
        let mut inv_denom = DynArray::<f64, 2>::from_shape([ex.lam.len(), ey.lam.len()]);
        for (i, a) in ex.lam.iter().enumerate() {
            for (j, b) in ey.lam.iter().enumerate() {
                inv_denom[[i, j]] = -1.0 / (a + b);
            }
        }
        Self {
            nx,
            ny,
            vy_t: transpose(&ey.v),
            vy_inv_t: transpose(&ey.v_inv),
            vx: ex.v,
            vx_inv: ex.v_inv,
            inv_denom,
        }
    }

    /// Solves on the interior nodes: Ĝ = V_x⁻¹ F V_y⁻ᵀ, Û = Ĝ ⊙ (1 / −(λ_i + μ_j)),
    /// U = V_x Û V_yᵀ (design §4.2).
    ///
    /// `f_interior: [n_x − 2, n_y − 2]` is f at the interior nodes, 'ij' (CONVENTIONS
    /// §12). Returns u at the same nodes, same shape.
    ///
    /// # Panics
    /// If `f_interior` is not `[n_x − 2, n_y − 2]`.
    pub fn solve(&self, f_interior: ArrayView2<f64>) -> Array2<f64> {
        assert_eq!(
            f_interior.dim(),
            (self.nx - 2, self.ny - 2),
            "CollocationSolver::solve: forcing must be [nx - 2, ny - 2]"
        );
        let g = self.vx_inv.dot(&to_dyn(f_interior).dot(&self.vy_inv_t));
        let [m, n] = g.shape();
        let mut uh = DynArray::<f64, 2>::from_shape([m, n]);
        for j in 0..n {
            for i in 0..m {
                uh[[i, j]] = g[[i, j]] * self.inv_denom[[i, j]];
            }
        }
        from_dyn(&self.vx.dot(&uh.dot(&self.vy_t)))
    }

    /// Solves on the full grid: reads the interior of `f` and returns u with u = 0 on
    /// the boundary rows and columns.
    ///
    /// `f: [n_x, n_y]`, 'ij' (CONVENTIONS §12); boundary values of `f` are ignored.
    /// Returns `[n_x, n_y]`.
    ///
    /// # Panics
    /// If `f` is not `[n_x, n_y]`.
    pub fn solve_full(&self, f: ArrayView2<f64>) -> Array2<f64> {
        assert_eq!(
            f.dim(),
            (self.nx, self.ny),
            "CollocationSolver::solve_full: forcing must be [nx, ny]"
        );
        let mut u = Array2::zeros((self.nx, self.ny));
        u.slice_mut(s![1..self.nx - 1, 1..self.ny - 1])
            .assign(&self.solve(f.slice(s![1..self.nx - 1, 1..self.ny - 1])));
        u
    }
}

/// Interior block of D², `[n − 2, n − 2]`.
fn interior_d2(n: usize) -> Array2<f64> {
    diff2_matrix(n).slice(s![1..n - 1, 1..n - 1]).to_owned()
}

/// Real eigendecomposition d = V diag(λ) V⁻¹ of an interior D² block.
struct RealEig {
    /// Eigenvalues λ_j.
    lam: Vec<f64>,
    /// Right eigenvectors v_j as columns, unit 2-norm.
    v: DynArray<f64, 2>,
    /// V⁻¹, with row j = u_jᵀ / (u_jᵀ v_j).
    v_inv: DynArray<f64, 2>,
    /// Discarded imaginary parts: max |Im λ| / max |λ|.
    im_lam: f64,
    /// Discarded imaginary parts: max |Im| over the entries of V and U.
    im_vec: f64,
    /// min_j |u_jᵀ v_j|, the reciprocal of the largest eigenvalue condition number.
    min_biorth: f64,
}

/// Real eigenvalues, right eigenvectors and V⁻¹ of `d: [m, m]`.
///
/// LAPACK's left eigenvectors satisfy u_jᵀ d = λ_j u_jᵀ and are paired with v_j by
/// column. For distinct eigenvalues u_iᵀ v_j = 0 when i ≠ j, so
/// V⁻¹ = diag(1 / u_jᵀ v_j) Uᵀ without a factorisation.
fn real_eig(d: &Array2<f64>) -> RealEig {
    let m = d.nrows();
    // `RightEigenvectors` alone would also panic in rlst 0.9.0 (`eig` passes `ldvl = n`
    // without a left-vector buffer and the geev wrapper asserts `ldvl == 1`), but the
    // left vectors are needed for V⁻¹ in any case.
    let (lam, v, u) = to_dyn(d.view())
        .eig(EigMode::BothEigenvectors)
        .expect("CollocationSolver::new: eig of the interior D² failed");
    let v = v.expect("CollocationSolver::new: eig returned no right eigenvectors");
    let u = u.expect("CollocationSolver::new: eig returned no left eigenvectors");
    let scale = (0..m).fold(0.0_f64, |a, i| a.max(lam[[i]].norm()));
    let mut im_lam = 0.0_f64;
    let mut im_vec = 0.0_f64;
    let mut v_re = DynArray::<f64, 2>::from_shape([m, m]);
    let mut v_inv = DynArray::<f64, 2>::from_shape([m, m]);
    let mut min_biorth = f64::INFINITY;
    for j in 0..m {
        im_lam = im_lam.max(lam[[j]].im.abs());
        let mut s_j = 0.0;
        for i in 0..m {
            v_re[[i, j]] = v[[i, j]].re;
            im_vec = im_vec.max(v[[i, j]].im.abs()).max(u[[i, j]].im.abs());
            s_j += u[[i, j]].re * v[[i, j]].re;
        }
        min_biorth = min_biorth.min(s_j.abs());
        for i in 0..m {
            v_inv[[j, i]] = u[[i, j]].re / s_j;
        }
    }
    RealEig {
        lam: (0..m).map(|i| lam[[i]].re).collect(),
        v: v_re,
        v_inv,
        im_lam: im_lam / scale,
        im_vec,
        min_biorth,
    }
    .checked()
}

impl RealEig {
    /// Panics if a discarded imaginary part exceeds `IMAG_TOL` or a left and right
    /// eigenvector pair is closer than `BIORTH_TOL` to orthogonal; returns `self`.
    fn checked(self) -> Self {
        assert!(
            self.im_lam <= IMAG_TOL && self.im_vec <= IMAG_TOL,
            "CollocationSolver::new: complex eigenpairs of the interior D² \
             (max |Im λ| / max |λ| = {:e}, max |Im v|, |Im u| = {:e})",
            self.im_lam,
            self.im_vec
        );
        assert!(
            self.min_biorth >= BIORTH_TOL,
            "CollocationSolver::new: ill-conditioned eigenvalue of the interior D² \
             (min |u_jᵀ v_j| = {:e})",
            self.min_biorth
        );
        self
    }
}

/// ndarray → rlst (column-major) copy.
fn to_dyn(a: ArrayView2<f64>) -> DynArray<f64, 2> {
    let (m, n) = a.dim();
    let mut out = DynArray::<f64, 2>::from_shape([m, n]);
    for ((i, j), v) in a.indexed_iter() {
        out[[i, j]] = *v;
    }
    out
}

/// rlst → ndarray copy.
fn from_dyn(a: &DynArray<f64, 2>) -> Array2<f64> {
    let [m, n] = a.shape();
    Array2::from_shape_fn((m, n), |(i, j)| a[[i, j]])
}

/// Owned transpose (rlst's `dot` needs contiguous storage, which a transposed view
/// lacks).
fn transpose(a: &DynArray<f64, 2>) -> DynArray<f64, 2> {
    let [m, n] = a.shape();
    let mut out = DynArray::<f64, 2>::from_shape([n, m]);
    for j in 0..n {
        for i in 0..m {
            out[[j, i]] = a[[i, j]];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::chebyshev::{clenshaw_curtis, rel_l2_error};
    use crate::neural_operators::pde::poisson::manufactured::{
        ANISO, ANISO_T, EXP, Manufactured, POLY, SIN, exp_f, exp_u, rel_max, sample, sin_f, sin_u,
    };
    use rlst::Solve;

    /// Relative max and Clenshaw–Curtis relative L² errors of solver A on `m`.
    fn errors(nx: usize, ny: usize, m: &Manufactured) -> (f64, f64) {
        let u = CollocationSolver::new(nx, ny).solve_full(sample(nx, ny, m.f).view());
        let exact = sample(nx, ny, m.u);
        let (wx, wy) = (clenshaw_curtis(nx), clenshaw_curtis(ny));
        (
            rel_max(&u, &exact),
            rel_l2_error(u.view(), exact.view(), wx.view(), wy.view()),
        )
    }

    /// Option B (design §4.1): −(I ⊗ D_xx + D_yy ⊗ I) vec U = vec F with column-major
    /// vec (p = i + m·j), solved by dense LU. `f: [n_x − 2, n_y − 2]`.
    fn kron_lu(nx: usize, ny: usize, f: &Array2<f64>) -> Array2<f64> {
        let (dxx, dyy) = (interior_d2(nx), interior_d2(ny));
        let (m, n) = f.dim();
        let mut a = DynArray::<f64, 2>::from_shape([m * n, m * n]);
        let mut rhs = DynArray::<f64, 1>::from_shape([m * n]);
        for j in 0..n {
            for i in 0..m {
                let p = i + m * j;
                for k in 0..m {
                    a[[p, k + m * j]] -= dxx[[i, k]];
                }
                for k in 0..n {
                    a[[p, i + m * k]] -= dyy[[j, k]];
                }
                rhs[[p]] = f[[i, j]];
            }
        }
        let x = a.solve(&rhs).expect("LU solve");
        Array2::from_shape_fn((m, n), |(i, j)| x[[i + m * j]])
    }

    #[test]
    fn manufactured_solutions_reach_floor() {
        for m in [&SIN, &POLY, &EXP] {
            for n in [9, 17, 33, 65, 129] {
                let (e_max, e_l2) = errors(n, n, m);
                println!("{:22} n = {n:3}: max {e_max:.1e}, L² {e_l2:.1e}", m.name);
                if n >= 33 {
                    assert!(
                        e_max <= 1e-12 && e_l2 <= 1e-12,
                        "{} at n = {n}: max {e_max:e}, L² {e_l2:e}",
                        m.name
                    );
                }
            }
        }
        // Spectral decay: n = 9 → 17 gains at least two orders of magnitude.
        for m in [&SIN, &EXP] {
            let (e9, _) = errors(9, 9, m);
            let (e17, _) = errors(17, 17, m);
            assert!(e17 * 100.0 <= e9, "{}: {e9:e} at 9, {e17:e} at 17", m.name);
        }
    }

    /// Runs `f` on a thread with a 64 MB stack and re-raises its panic. The dense LU of
    /// option B goes through OpenBLAS's threaded `getrf`, which can overflow the 2 MB
    /// test thread (see the module docs).
    fn with_large_stack(f: impl FnOnce() + Send + 'static) {
        let handle = std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(f)
            .expect("spawn a large-stack test thread");
        if let Err(panic) = handle.join() {
            std::panic::resume_unwind(panic);
        }
    }

    #[test]
    fn agrees_with_kronecker_lu() {
        with_large_stack(|| {
            let forcing = |x: f64, y: f64| x.exp() * (2.0 * y).cos() + x * y;
            for n in [9, 17, 33] {
                let f = sample(n, n, forcing)
                    .slice(s![1..n - 1, 1..n - 1])
                    .to_owned();
                let a = CollocationSolver::new(n, n).solve(f.view());
                let b = kron_lu(n, n, &f);
                let e = rel_max(&a, &b);
                println!("A against B, n = {n:2}: {e:.1e}");
                assert!(e <= 1e-10, "n = {n}: {e:e}");
            }
        });
    }

    /// max |a| over the entries.
    fn max_abs(a: &Array2<f64>) -> f64 {
        a.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
    }

    /// Infinity norm (max row sum of |entries|).
    fn norm_inf(a: &Array2<f64>) -> f64 {
        a.rows()
            .into_iter()
            .map(|r| r.iter().map(|v| v.abs()).sum::<f64>())
            .fold(0.0, f64::max)
    }

    #[test]
    fn residual_at_round_off() {
        // Scaled residual of −D_xx U − U D_yyᵀ = F on the interior:
        // ‖R‖ / ((‖D_xx‖_∞ + ‖D_yy‖_∞) ‖U‖ + ‖F‖), max norms.
        for m in [&SIN, &POLY, &EXP] {
            for n in [9, 17, 33, 65, 129] {
                let d = interior_d2(n);
                let f = sample(n, n, m.f).slice(s![1..n - 1, 1..n - 1]).to_owned();
                let u = CollocationSolver::new(n, n).solve(f.view());
                let r = -(d.dot(&u) + u.dot(&d.t())) - &f;
                let scale = 2.0 * norm_inf(&d) * max_abs(&u) + max_abs(&f);
                let e = max_abs(&r) / scale;
                println!("{:22} n = {n:3}: scaled residual {e:.1e}", m.name);
                assert!(e <= 1e-14, "{} at n = {n}: {e:e}", m.name);
            }
        }
    }

    #[test]
    fn rectangular_grid() {
        // ANISO is only resolved with n_y ≈ 33, so it reaches the floor on [17, 33] but
        // not on [33, 17]; its transpose does the opposite. A swapped axis fails both.
        for (nx, ny, m) in [(17, 33, &ANISO), (33, 17, &ANISO_T), (33, 65, &EXP)] {
            let (e_max, e_l2) = errors(nx, ny, m);
            println!(
                "{:26} on [{nx}, {ny}]: max {e_max:.1e}, L² {e_l2:.1e}",
                m.name
            );
            assert!(
                e_max <= 1e-12 && e_l2 <= 1e-12,
                "{} on [{nx}, {ny}]: max {e_max:e}, L² {e_l2:e}",
                m.name
            );
        }
        for (nx, ny, m) in [(33, 17, &ANISO), (17, 33, &ANISO_T)] {
            let (e_max, _) = errors(nx, ny, m);
            println!("{:26} on [{nx}, {ny}]: max {e_max:.1e}", m.name);
            assert!(e_max >= 1e-6, "{} on [{nx}, {ny}]: {e_max:e}", m.name);
        }
    }

    #[test]
    fn factorisation_reused() {
        let n = 33;
        let (f1, f2) = (sample(n, n, sin_f), sample(n, n, exp_f));
        let solver = CollocationSolver::new(n, n);
        // Interleaved solves on one struct: no state carries over between calls.
        let u1 = solver.solve_full(f1.view());
        let u2 = solver.solve_full(f2.view());
        let u1_again = solver.solve_full(f1.view());
        assert_eq!(u1, u1_again);
        assert_eq!(u1, CollocationSolver::new(n, n).solve_full(f1.view()));
        assert_eq!(u2, CollocationSolver::new(n, n).solve_full(f2.view()));
        // Each reused solve is correct, not just repeatable.
        assert!(rel_max(&u1, &sample(n, n, sin_u)) <= 1e-12);
        assert!(rel_max(&u2, &sample(n, n, exp_u)) <= 1e-12);
        // The stored factorisation is a linear map: solve(f1 − 3 f2) = u1 − 3 u2.
        let combo = solver.solve_full((&f1 - &(3.0 * &f2)).view());
        let e = rel_max(&combo, &(&u1 - &(3.0 * &u2)));
        assert!(e <= 1e-13, "linearity: {e:e}");
    }

    #[test]
    fn eigenpairs_are_real() {
        for n in [9, 17, 33, 65, 129] {
            let e = real_eig(&interior_d2(n));
            println!(
                "n = {n:3}: max |Im λ| / max |λ| = {:e}, max |Im v|, |Im u| = {:e}",
                e.im_lam, e.im_vec
            );
            assert!(e.im_lam <= IMAG_TOL && e.im_vec <= IMAG_TOL, "n = {n}");
        }
    }

    #[test]
    fn left_vectors_give_v_inverse() {
        for n in [9, 17, 33, 65, 129] {
            let d = interior_d2(n);
            let e = real_eig(&d);
            let (v, v_inv) = (from_dyn(&e.v), from_dyn(&e.v_inv));
            let eye = Array2::<f64>::eye(n - 2);
            let left = max_abs(&(v_inv.dot(&v) - &eye));
            let right = max_abs(&(v.dot(&v_inv) - &eye));
            let lam = Array2::from_diag(&ndarray::Array1::from(e.lam.clone()));
            let recon = max_abs(&(v.dot(&lam).dot(&v_inv) - &d)) / max_abs(&d);
            println!(
                "n = {n:3}: min |u_jᵀ v_j| {:.1e}, |V⁻¹V − I| {left:.1e}, \
                 |VV⁻¹ − I| {right:.1e}, |VΛV⁻¹ − D| / |D| {recon:.1e}",
                e.min_biorth
            );
            assert!(left <= 1e-10 && right <= 1e-10, "n = {n}");
            assert!(recon <= 1e-12, "n = {n}: {recon:e}");
        }
    }

    #[test]
    #[should_panic(expected = "must be >= 3")]
    fn too_small_grid_panics() {
        CollocationSolver::new(2, 9);
    }

    #[test]
    #[should_panic(expected = "forcing must be [nx - 2, ny - 2]")]
    fn solve_wrong_shape_panics() {
        CollocationSolver::new(9, 9).solve(Array2::zeros((9, 9)).view());
    }

    #[test]
    #[should_panic(expected = "forcing must be [nx, ny]")]
    fn solve_full_wrong_shape_panics() {
        CollocationSolver::new(9, 17).solve_full(Array2::zeros((17, 9)).view());
    }
}
