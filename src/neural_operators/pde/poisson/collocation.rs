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
//!
//! [`CollocationSolver3d`] is A3, the same method on [−1, 1]³ (3D proposal §5): one
//! interior D² eigendecomposition per axis and mode products along the three axes.

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, s};
use rlst::dense::linalg::lapack::eigenvalue_decomposition::EigMode;
use rlst::{DynArray, EigenvalueDecomposition};

use crate::neural_operators::chebyshev::{apply_nd, diff2_matrix};

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

/// Collocation solver A3 for −Δu = f on [−1, 1]³, u = 0 on the boundary, on an
/// `[n_x, n_y, n_z]` CGL grid stored 'ijk' (CONVENTIONS §12; 3D proposal §5).
///
/// The 2D solver with a third axis: with D_aa = V_a Λ_a V_a⁻¹ per axis,
/// u = −(V_x ⊗ V_y ⊗ V_z)(Λ_x ⊕ Λ_y ⊕ Λ_z)⁻¹(V_x⁻¹ ⊗ V_y⁻¹ ⊗ V_z⁻¹) f on the interior
/// nodes, six mode products ([`apply_nd`]) and a pointwise division, about 12 (n − 2)⁴
/// flops. Built once per grid size; every solve reuses the eigendecompositions.
pub struct CollocationSolver3d {
    /// `[n_x, n_y, n_z]`.
    shape: [usize; 3],
    /// V_a per axis, `[n_a − 2, n_a − 2]`.
    v: [Array2<f64>; 3],
    /// V_a⁻¹ per axis, `[n_a − 2, n_a − 2]`.
    v_inv: [Array2<f64>; 3],
    /// 1 / −(λ_i + μ_j + ν_k), `[n_x − 2, n_y − 2, n_z − 2]`.
    inv_denom: Array3<f64>,
}

impl std::fmt::Debug for CollocationSolver3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollocationSolver3d")
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl CollocationSolver3d {
    /// Builds the solver for an `[n_x, n_y, n_z]` CGL grid: eigendecomposes the interior
    /// block of [`diff2_matrix`] along each axis, O(n³) once.
    ///
    /// # Panics
    /// If any of `nx`, `ny`, `nz` is below 3 (no interior nodes), or as
    /// [`CollocationSolver::new`] for each eigendecomposition.
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let shape = [nx, ny, nz];
        assert!(
            shape.iter().all(|&n| n >= 3),
            "CollocationSolver3d::new: nx, ny and nz must be >= 3 (interior nodes), got {shape:?}"
        );
        let eig = shape.map(|n| real_eig(&interior_d2(n)));
        let inv_denom = Array3::from_shape_fn((nx - 2, ny - 2, nz - 2), |(i, j, k)| {
            -1.0 / (eig[0].lam[i] + eig[1].lam[j] + eig[2].lam[k])
        });
        Self {
            shape,
            v: eig.each_ref().map(|e| from_dyn(&e.v)),
            v_inv: eig.each_ref().map(|e| from_dyn(&e.v_inv)),
            inv_denom,
        }
    }

    /// Solves on the interior nodes: Ĝ = f ×₁ V_x⁻¹ ×₂ V_y⁻¹ ×₃ V_z⁻¹,
    /// Û = Ĝ ⊙ (1 / −(λ_i + μ_j + ν_k)), u = Û ×₁ V_x ×₂ V_y ×₃ V_z.
    ///
    /// `f_interior: [n_x − 2, n_y − 2, n_z − 2]` is f at the interior nodes, 'ijk'
    /// (CONVENTIONS §12). Returns u at the same nodes, same shape.
    ///
    /// # Panics
    /// If `f_interior` is not `[n_x − 2, n_y − 2, n_z − 2]`.
    pub fn solve(&self, f_interior: ArrayView3<f64>) -> Array3<f64> {
        assert_eq!(
            f_interior.shape(),
            self.shape.map(|n| n - 2),
            "CollocationSolver3d::solve: forcing must be [nx - 2, ny - 2, nz - 2]"
        );
        let g = apply_nd(&self.v_inv.each_ref().map(|m| m.view()), f_interior);
        apply_nd(
            &self.v.each_ref().map(|m| m.view()),
            (g * &self.inv_denom).view(),
        )
    }

    /// Solves on the full grid: reads the interior of `f` and returns u with u = 0 on the
    /// six faces.
    ///
    /// `f: [n_x, n_y, n_z]`, 'ijk' (CONVENTIONS §12); boundary values of `f` are ignored.
    /// Returns `[n_x, n_y, n_z]`.
    ///
    /// # Panics
    /// If `f` is not `[n_x, n_y, n_z]`.
    pub fn solve_full(&self, f: ArrayView3<f64>) -> Array3<f64> {
        let [nx, ny, nz] = self.shape;
        assert_eq!(
            f.shape(),
            self.shape,
            "CollocationSolver3d::solve_full: forcing must be [nx, ny, nz]"
        );
        let inner = s![1..nx - 1, 1..ny - 1, 1..nz - 1];
        let mut u = Array3::zeros((nx, ny, nz));
        u.slice_mut(inner).assign(&self.solve(f.slice(inner)));
        u
    }
}

/// The n − 2 eigenvalues of the interior block of the Chebyshev D² on n CGL nodes, the
/// 1D Dirichlet second derivative of solver A: real and negative. The 2D Laplacian's are the
/// sums of two, so its condition number is the ratio of the largest |λ| to the smallest
/// (Phase 10 T6, the PDE side of the conditioning comparison).
///
/// # Panics
/// If `n < 3`, or as [`CollocationSolver::new`] for the eigendecomposition.
pub fn interior_d2_eigenvalues(n: usize) -> Vec<f64> {
    assert!(n >= 3, "interior_d2_eigenvalues: need n >= 3, got {n}");
    real_eig(&interior_d2(n)).lam
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
        .expect("real_eig: eig of the interior D² failed");
    let v = v.expect("real_eig: eig returned no right eigenvectors");
    let u = u.expect("real_eig: eig returned no left eigenvectors");
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
            "real_eig: complex eigenpairs of the interior D² \
             (max |Im λ| / max |λ| = {:e}, max |Im v|, |Im u| = {:e})",
            self.im_lam,
            self.im_vec
        );
        assert!(
            self.min_biorth >= BIORTH_TOL,
            "real_eig: ill-conditioned eigenvalue of the interior D² \
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
    use crate::neural_operators::chebyshev::{cc_rel_l2_error, cc_rel_l2_error_nd, nodes};
    use crate::neural_operators::pde::poisson::grf3d;
    use crate::neural_operators::pde::poisson::manufactured::{
        ANISO, ANISO_T, ANISO3, EXP, EXP3, MODES3, Manufactured, Manufactured3, POLY, SIN, SIN3,
        exp_f, exp_u, rel_max, sample, sample3, sin_f, sin_u,
    };
    use ndarray::{Array, Dimension};
    use rlst::Solve;

    /// The interior D² eigenvalues are negative, the smallest |λ| that of −d²/dx² on
    /// [−1, 1], (π/2)², and the largest grows like n⁴: κ rises 15–17× when n − 1 doubles.
    #[test]
    fn interior_d2_eigenvalues_match_the_continuous_lowest_and_grow_like_n4() {
        let kappa = |n: usize| {
            let lam = interior_d2_eigenvalues(n);
            assert!(lam.iter().all(|&l| l < 0.0));
            let (lo, hi) = lam.iter().fold((f64::INFINITY, 0.0_f64), |(lo, hi), l| {
                (lo.min(l.abs()), hi.max(l.abs()))
            });
            assert!(
                (lo - std::f64::consts::PI.powi(2) / 4.0).abs() < 1e-8,
                "{lo}"
            );
            hi / lo
        };
        let (k33, k65) = (kappa(33), kappa(65));
        assert!((15.0..17.0).contains(&(k65 / k33)), "{k33} {k65}");
    }

    /// Relative max and Clenshaw–Curtis relative L² errors of solver A on `m`.
    fn errors(nx: usize, ny: usize, m: &Manufactured) -> (f64, f64) {
        let u = CollocationSolver::new(nx, ny).solve_full(sample(nx, ny, m.f).view());
        let exact = sample(nx, ny, m.u);
        (rel_max(&u, &exact), cc_rel_l2_error(u.view(), exact.view()))
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
    fn max_abs<D: Dimension>(a: &Array<f64, D>) -> f64 {
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

    // ---- A3, the 3D solver (3D proposal §5) ----

    /// Relative max and 3D Clenshaw–Curtis relative L² errors of A3 on `m`.
    fn errors3(nx: usize, ny: usize, nz: usize, m: &Manufactured3) -> (f64, f64) {
        let u = CollocationSolver3d::new(nx, ny, nz).solve_full(sample3(nx, ny, nz, m.f).view());
        let exact = sample3(nx, ny, nz, m.u);
        (
            rel_max(&u, &exact),
            cc_rel_l2_error_nd(u.view(), exact.view()),
        )
    }

    #[test]
    fn solver_3d_manufactured_solutions_reach_floor() {
        for m in [&SIN3, &MODES3, &EXP3] {
            for n in [9, 17, 33, 65] {
                let (e_max, e_l2) = errors3(n, n, n, m);
                println!("{:24} n = {n:2}: max {e_max:.1e}, L² {e_l2:.1e}", m.name);
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
        for m in [&SIN3, &MODES3, &EXP3] {
            let (e9, _) = errors3(9, 9, 9, m);
            let (e17, _) = errors3(17, 17, 17, m);
            assert!(e17 * 100.0 <= e9, "{}: {e9:e} at 9, {e17:e} at 17", m.name);
        }
    }

    /// The closed form of CL3.1's GRF: f and u = Σ c / μ evaluated exactly at the nodes.
    #[test]
    fn solver_3d_grf_closed_form_reaches_floor() {
        for (k, n, seed) in [(8, 33, 0), (8, 33, 7), (16, 65, 1)] {
            let c = grf3d::forcing_coeffs(grf3d::draw_xi(seed).view(), k);
            let x = nodes(n);
            let f = grf3d::evaluate(c.view(), x.view(), x.view(), x.view());
            let exact = grf3d::evaluate(
                grf3d::solution_coeffs(c.view()).view(),
                x.view(),
                x.view(),
                x.view(),
            );
            let u = CollocationSolver3d::new(n, n, n).solve_full(f.view());
            let (e_max, e_l2) = (
                rel_max(&u, &exact),
                cc_rel_l2_error_nd(u.view(), exact.view()),
            );
            println!("GRF K = {k:2}, n = {n}, seed {seed}: max {e_max:.1e}, L² {e_l2:.1e}");
            assert!(
                e_max <= 1e-12 && e_l2 <= 1e-12,
                "K = {k}, n = {n}, seed {seed}: max {e_max:e}, L² {e_l2:e}"
            );
        }
    }

    #[test]
    fn solver_3d_residual_at_round_off() {
        // Scaled residual of −(D_xx ×₁ + D_yy ×₂ + D_zz ×₃) U = F on the interior, applied
        // without the eigendecomposition: ‖R‖ / (3 ‖D‖_∞ ‖U‖ + ‖F‖), max norms.
        for m in [&SIN3, &EXP3] {
            for n in [9, 17, 33, 65] {
                let (d, eye) = (interior_d2(n), Array2::<f64>::eye(n - 2));
                let ops: [[ArrayView2<f64>; 3]; 3] = [
                    [d.view(), eye.view(), eye.view()],
                    [eye.view(), d.view(), eye.view()],
                    [eye.view(), eye.view(), d.view()],
                ];
                let f = sample3(n, n, n, m.f)
                    .slice(s![1..n - 1, 1..n - 1, 1..n - 1])
                    .to_owned();
                let u = CollocationSolver3d::new(n, n, n).solve(f.view());
                let lap = ops
                    .iter()
                    .fold(Array3::zeros(u.dim()), |acc, t| acc + apply_nd(t, u.view()));
                let r = -lap - &f;
                let e = max_abs(&r) / (3.0 * norm_inf(&d) * max_abs(&u) + max_abs(&f));
                println!("{:24} n = {n:2}: scaled residual {e:.1e}", m.name);
                assert!(e <= 1e-14, "{} at n = {n}: {e:e}", m.name);
            }
        }
    }

    #[test]
    fn solver_3d_non_cubic_grid() {
        // ANISO3 is exact along x and needs 33 nodes along z: it reaches the floor on
        // [17, 25, 33] but not with the axes reversed. A swapped axis fails both.
        let (e_max, e_l2) = errors3(17, 25, 33, &ANISO3);
        println!("ANISO3 on [17, 25, 33]: max {e_max:.1e}, L² {e_l2:.1e}");
        assert!(
            e_max <= 1e-12 && e_l2 <= 1e-12,
            "max {e_max:e}, L² {e_l2:e}"
        );
        let (e_max, _) = errors3(33, 25, 17, &ANISO3);
        println!("ANISO3 on [33, 25, 17]: max {e_max:.1e}");
        assert!(e_max >= 1e-6, "{e_max:e}");
    }

    /// Time per A3 solve (prints only; run in release with `--ignored --nocapture`).
    #[test]
    #[ignore = "timing, run by hand in release"]
    fn time_per_solve_3d() {
        for (n, reps) in [(33, 50), (65, 10)] {
            let t = std::time::Instant::now();
            let solver = CollocationSolver3d::new(n, n, n);
            let setup = t.elapsed();
            let f = sample3(n, n, n, EXP3.f);
            let t = std::time::Instant::now();
            for _ in 0..reps {
                std::hint::black_box(solver.solve_full(f.view()));
            }
            let per = t.elapsed() / reps;
            println!("n = {n}: setup {setup:.2?}, solve_full {per:.2?} (mean of {reps})");
        }
    }

    #[test]
    #[should_panic(expected = "must be >= 3")]
    fn solver_3d_too_small_grid_panics() {
        CollocationSolver3d::new(9, 9, 2);
    }

    #[test]
    #[should_panic(expected = "forcing must be [nx - 2, ny - 2, nz - 2]")]
    fn solver_3d_solve_wrong_shape_panics() {
        CollocationSolver3d::new(9, 11, 13).solve(Array3::zeros((7, 9, 9)).view());
    }

    #[test]
    #[should_panic(expected = "forcing must be [nx, ny, nz]")]
    fn solver_3d_solve_full_wrong_shape_panics() {
        CollocationSolver3d::new(9, 11, 13).solve_full(Array3::zeros((13, 11, 9)).view());
    }
}
