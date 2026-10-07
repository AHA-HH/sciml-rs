//! Smoke tests for RLST behind the `chebyshev` feature.
//!
//! They check that rlst and the BLAS/LAPACK provider link on this platform
//! (Accelerate on macOS, OpenBLAS on Linux) and that the two solve routes the
//! Chebyshev reference solver relies on work on the pinned rlst version: CSR
//! assembly with CG, and the dense Sylvester solve.
#![cfg(feature = "chebyshev")]

use rlst::dense::linalg::lapack::sylvester::{TrsylSign, TrsylStatus};
use rlst::operator::abstract_operator::Operator;
use rlst::operator::algorithms::conjugate_gradients::CgIteration;
use rlst::operator::space::zero_element;
use rlst::sparse::csr_mat::CsrMatrix;
use rlst::traits::abstract_operator::OperatorBase;
use rlst::{DynArray, FromAij, SylvesterSolve};
// The tests call rlst directly; linking the crate brings in the BLAS/LAPACK
// provider it names in `lib.rs`.
use sciml_rs as _;

/// Dense `a * b` for square `n x n` matrices.
fn matmul(a: &DynArray<f64, 2>, b: &DynArray<f64, 2>, n: usize) -> DynArray<f64, 2> {
    let mut out = DynArray::<f64, 2>::from_shape([n, n]);
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = (0..n).map(|k| a[[i, k]] * b[[k, j]]).sum();
        }
    }
    out
}

#[test]
fn cg_solves_1d_dirichlet_laplacian() {
    let n = 50;
    let (mut rows, mut cols, mut data) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        data.push(2.0);
        if i + 1 < n {
            rows.extend([i, i + 1]);
            cols.extend([i + 1, i]);
            data.extend([-1.0, -1.0]);
        }
    }
    let mat = CsrMatrix::<f64>::from_aij([n, n], &rows, &cols, &data);
    let op = Operator::from(&mat);

    let mut rhs = zero_element(op.range());
    for i in 0..n {
        rhs.imp_mut()[[i]] = 1.0;
    }
    let mut x = zero_element(op.domain());
    let tol = 1e-12;
    // `run()` returns the final relative residual, not a converged flag.
    let residual = CgIteration::new(&op, &rhs, &mut x).set_tol(tol).run();
    assert!(residual < tol, "CG residual {residual:e} >= {tol:e}");

    let exact: Vec<f64> = (0..n).map(|i| ((i + 1) * (n - i)) as f64 / 2.0).collect();
    let max_exact = exact.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let max_err = (0..n).fold(0.0_f64, |m, i| m.max((x.imp()[[i]] - exact[i]).abs()));
    let rel = max_err / max_exact;
    assert!(rel <= 1e-9, "relative max error {rel:e} > 1e-9");
}

#[test]
fn sylvester_route_solves_small_system() {
    let n = 8;
    let mut a = DynArray::<f64, 2>::from_shape([n, n]);
    let mut b = DynArray::<f64, 2>::from_shape([n, n]);
    let mut c = DynArray::<f64, 2>::from_shape([n, n]);
    a.fill_from_seed_normally_distributed(1);
    b.fill_from_seed_normally_distributed(2);
    c.fill_from_seed_normally_distributed(3);
    // Shift so the spectra of A and -B stay well apart and the equation is
    // well conditioned.
    for i in 0..n {
        a[[i, i]] += 8.0;
        b[[i, i]] += 8.0;
    }

    let solution = a.solve_sylvester(&b, &c, TrsylSign::Plus).unwrap();
    assert_eq!(solution.status(), TrsylStatus::Success);
    let scale = solution.scale();
    assert!(
        scale.is_finite() && scale > 0.0 && scale <= 1.0,
        "unexpected scale {scale}"
    );

    // The solution satisfies A X + X B = scale C.
    let x = solution.x();
    let ax = matmul(&a, x, n);
    let xb = matmul(x, &b, n);
    let (mut res_sq, mut c_sq) = (0.0_f64, 0.0_f64);
    for i in 0..n {
        for j in 0..n {
            let sc = scale * c[[i, j]];
            res_sq += (ax[[i, j]] + xb[[i, j]] - sc).powi(2);
            c_sq += sc.powi(2);
        }
    }
    let rel = (res_sq / c_sq).sqrt();
    assert!(rel <= 1e-12, "relative Frobenius residual {rel:e} > 1e-12");
}
