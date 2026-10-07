//! Option A (collocation as −D_xx U − U D_yyᵀ = F, design §4.1) and option B (the same
//! system as one dense Kronecker sum, LU).
//!
//! All solvers take and return interior values, `[n − 2, n − 2]`, 'ij' (axis 0 is x).

use crate::{from_dyn, to_dyn, transpose};
use ndarray::Array2;
use rlst::dense::linalg::lapack::eigenvalue_decomposition::EigMode;
use rlst::dense::linalg::lapack::interface::trsyl::{Trsyl, TrsylTranspose};
use rlst::dense::linalg::lapack::sylvester::{TrsylSign, TrsylStatus};
use rlst::{DynArray, EigenvalueDecomposition, Inverse, Solve, SylvesterSolve};

/// A-fd: fast diagonalisation. D_xx = V_x Λ_x V_x⁻¹ (and likewise y), computed once.
/// A solve is Ĝ = V_x⁻¹ F V_y⁻ᵀ, Û_ij = Ĝ_ij / −(λ_i + μ_j), U = V_x Û V_yᵀ.
pub struct FdSolver {
    vx: DynArray<f64, 2>,
    vx_inv: DynArray<f64, 2>,
    vy_t: DynArray<f64, 2>,
    vy_inv_t: DynArray<f64, 2>,
    /// 1 / −(λ_i + μ_j).
    inv_denom: DynArray<f64, 2>,
    /// Largest |Im| over the eigenvalues of D_xx and D_yy.
    pub max_imag_lambda: f64,
    /// Largest |Im| over the eigenvector entries (each column normalised by LAPACK).
    pub max_imag_v: f64,
}

/// Real parts of eig(D), and the largest imaginary parts seen.
///
/// Uses `EigMode::BothEigenvectors`: in rlst 0.9.0 `RightEigenvectors` panics, because
/// `eig` passes `ldvl = n` without a left-vector buffer and the geev wrapper asserts
/// `ldvl == 1` (`dense/linalg/lapack/interface/geev.rs`). The left vectors are discarded (setup cost only).
fn real_eig(d: &Array2<f64>) -> (Vec<f64>, DynArray<f64, 2>, f64, f64) {
    let m = d.nrows();
    let (lam, v, _) = to_dyn(d)
        .eig(EigMode::BothEigenvectors)
        .expect("eig of D_xx");
    let v = v.expect("right eigenvectors");
    let mut im_l = 0.0_f64;
    let mut im_v = 0.0_f64;
    let mut lam_re = Vec::with_capacity(m);
    let mut v_re = DynArray::<f64, 2>::from_shape([m, m]);
    for i in 0..m {
        lam_re.push(lam[[i]].re);
        im_l = im_l.max(lam[[i]].im.abs());
        for j in 0..m {
            v_re[[i, j]] = v[[i, j]].re;
            im_v = im_v.max(v[[i, j]].im.abs());
        }
    }
    (lam_re, v_re, im_l, im_v)
}

impl FdSolver {
    pub fn new(dxx: &Array2<f64>, dyy: &Array2<f64>) -> Self {
        let (lx, vx, ilx, ivx) = real_eig(dxx);
        let (ly, vy, ily, ivy) = real_eig(dyy);
        let vx_inv = vx.inverse().expect("V_x invertible");
        let vy_inv = vy.inverse().expect("V_y invertible");
        let mut inv_denom = DynArray::<f64, 2>::from_shape([lx.len(), ly.len()]);
        for (i, a) in lx.iter().enumerate() {
            for (j, b) in ly.iter().enumerate() {
                inv_denom[[i, j]] = -1.0 / (a + b);
            }
        }
        FdSolver {
            vy_t: transpose(&vy),
            vy_inv_t: transpose(&vy_inv),
            vx,
            vx_inv,
            inv_denom,
            max_imag_lambda: ilx.max(ily),
            max_imag_v: ivx.max(ivy),
        }
    }

    pub fn solve(&self, f: &Array2<f64>) -> Array2<f64> {
        let g = self.vx_inv.dot(&to_dyn(f).dot(&self.vy_inv_t));
        let [m, n] = g.shape();
        let mut uh = DynArray::<f64, 2>::from_shape([m, n]);
        for j in 0..n {
            for i in 0..m {
                uh[[i, j]] = g[[i, j]] * self.inv_denom[[i, j]];
            }
        }
        from_dyn(&self.vx.dot(&uh.dot(&self.vy_t)))
    }
}

/// A-syl with reuse: real Schur forms of A = −D_xx and B = −D_yyᵀ computed once
/// (`schur`, LAPACK gees). A solve is Y = Q_aᵀ F Q_b, then LAPACK trsyl on the
/// quasi-triangular factors (A X + X B = scale·C), then U = Q_a Y Q_bᵀ / scale.
pub struct SylReuseSolver {
    ra: DynArray<f64, 2>,
    qa: DynArray<f64, 2>,
    qa_t: DynArray<f64, 2>,
    sb: DynArray<f64, 2>,
    qb: DynArray<f64, 2>,
    qb_t: DynArray<f64, 2>,
}

impl SylReuseSolver {
    pub fn new(dxx: &Array2<f64>, dyy: &Array2<f64>) -> Self {
        let a = to_dyn(&dxx.mapv(|v| -v));
        let b = to_dyn(&dyy.t().mapv(|v| -v));
        let (ra, qa) = a.schur().expect("schur of A");
        let (sb, qb) = b.schur().expect("schur of B");
        SylReuseSolver {
            qa_t: transpose(&qa),
            qb_t: transpose(&qb),
            ra,
            qa,
            sb,
            qb,
        }
    }

    /// Returns the solution and the trsyl status.
    pub fn solve(&self, f: &Array2<f64>) -> (Array2<f64>, TrsylStatus) {
        let [m, _] = self.ra.shape();
        let [n, _] = self.sb.shape();
        let mut y = self.qa_t.dot(&to_dyn(f).dot(&self.qb));
        let out = <f64 as Trsyl>::trsyl(
            TrsylTranspose::NoTranspose,
            TrsylTranspose::NoTranspose,
            TrsylSign::Plus,
            m,
            n,
            self.ra.data().unwrap(),
            m,
            self.sb.data().unwrap(),
            n,
            y.data_mut().unwrap(),
            m,
        )
        .expect("trsyl");
        let mut u = from_dyn(&self.qa.dot(&y.dot(&self.qb_t)));
        u /= out.scale;
        (u, out.status)
    }
}

/// A-syl as shipped: `solve_sylvester` recomputes both Schur forms on every call.
pub fn syl_call(
    dxx: &Array2<f64>,
    dyy: &Array2<f64>,
    f: &Array2<f64>,
) -> (Array2<f64>, TrsylStatus) {
    let a = to_dyn(&dxx.mapv(|v| -v));
    let b = to_dyn(&dyy.t().mapv(|v| -v));
    let sol = a
        .solve_sylvester(&b, &to_dyn(f), TrsylSign::Plus)
        .expect("solve_sylvester");
    let mut u = from_dyn(sol.x());
    u /= sol.scale();
    (u, sol.status())
}

/// Option B: −(I ⊗ D_xx + D_yy ⊗ I) vec(U) = vec(F) with column-major vec
/// (p = i + m·j), solved by dense LU. Only for small n: the matrix is (n − 2)² square.
pub fn kron_lu(dxx: &Array2<f64>, dyy: &Array2<f64>, f: &Array2<f64>) -> Array2<f64> {
    let (m, n) = f.dim();
    let big = m * n;
    let mut a = DynArray::<f64, 2>::from_shape([big, big]);
    for j in 0..n {
        for i in 0..m {
            let p = i + m * j;
            for k in 0..m {
                a[[p, k + m * j]] -= dxx[[i, k]];
            }
            for k in 0..n {
                a[[p, i + m * k]] -= dyy[[j, k]];
            }
        }
    }
    let mut rhs = DynArray::<f64, 1>::from_shape([big]);
    for j in 0..n {
        for i in 0..m {
            rhs[[i + m * j]] = f[[i, j]];
        }
    }
    let x = a.solve(&rhs).expect("LU solve");
    Array2::from_shape_fn((m, n), |(i, j)| x[[i + m * j]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cheb::d2_interior;
    use crate::manufactured::ALL;
    use crate::{interior, rel_max, with_boundary};
    use transfers_spike::grids::{cheb_nodes, sample};

    fn manufactured(n: usize, k: usize) -> (Array2<f64>, Array2<f64>) {
        let x = cheb_nodes(n);
        (sample(&x, &x, ALL[k].u), sample(&x, &x, ALL[k].f))
    }

    #[test]
    fn fd_matches_kron_lu_at_n9() {
        let d = d2_interior(9);
        let fd = FdSolver::new(&d, &d);
        for k in 0..3 {
            let (_, f) = manufactured(9, k);
            let fi = interior(&f);
            let e = rel_max(&fd.solve(&fi), &kron_lu(&d, &d, &fi));
            assert!(e < 1e-12, "k={k}: {e:e}");
        }
    }

    // The polynomial solution has degree 4 in x and y: exact at n = 17 up to round-off.
    #[test]
    fn polynomial_solution_to_round_off_at_n17() {
        let d = d2_interior(17);
        let (u, f) = manufactured(17, 1);
        let fi = interior(&f);
        let fd = with_boundary(&FdSolver::new(&d, &d).solve(&fi));
        let (sr, st) = SylReuseSolver::new(&d, &d).solve(&fi);
        assert_eq!(st, TrsylStatus::Success);
        let (sc, _) = syl_call(&d, &d, &fi);
        for (name, got) in [
            ("fd", fd),
            ("syl", with_boundary(&sr)),
            ("call", with_boundary(&sc)),
        ] {
            let e = rel_max(&got, &u);
            assert!(e < 1e-12, "{name}: {e:e}");
        }
    }
}
