//! Chebyshev toolkit
//!
//! One-dimensional building blocks on the Chebyshev–Gauss–Lobatto (CGL) grid
//! (CONVENTIONS §12). Everything is f64, and the public API takes and returns ndarray
//! types only.
//!
//! - Always built: the closed-form CGL nodes (`nodes`), the grid transfers
//!   between the CGL grid and the model's uniform grid (`uniform_nodes`,
//!   `cheb_to_uniform`, `uniform_to_cheb`, `apply`, and the production pair `Transfers`
//!   with its degree `TRANSFER_DEGREE`), and the closed-form Clenshaw–Curtis weights with
//!   the norm and relative L² error built from them (`clencurt`, `cc_norm_sq`,
//!   `cc_rel_l2_error`, `cc_rel_l2_error_weighted`). They depend on ndarray alone, so training and
//!   evaluation on Chebyshev data need no RLST, BLAS or FFTW (design §12, decisions 13
//!   and 15).
//! - Behind the `datagen` feature: D and D² (`diff_matrix`, `diff2_matrix`), the only
//!   gated production items here. They wrap RLST's FFTW-backed `chebychev` transforms
//!   (design §12, decisions 12 and 27) and need a system libfftw3. RLST works in descending node order; every function here returns ascending
//!   order.
//! - Test oracles only (`#[cfg(all(test, feature = "datagen"))]`): the RLST nodes
//!   (`points::nodes`) and Clenshaw–Curtis weights (`quadrature::clenshaw_curtis`), which
//!   the closed forms above are checked against (design §12, decisions 12 and 26).

mod clencurt;
// Gated: wraps RLST's FFTW transforms (decision 12; T15 declined, decision 27).
#[cfg(feature = "datagen")]
mod differentiation;
#[cfg(all(test, feature = "datagen"))]
mod points;
#[cfg(all(test, feature = "datagen"))]
mod quadrature;
mod transfer;

pub use clencurt::{cc_norm_sq, cc_rel_l2_error, cc_rel_l2_error_weighted, clencurt};
#[cfg(feature = "datagen")]
pub use differentiation::{diff_matrix, diff2_matrix};
pub use transfer::{
    TRANSFER_DEGREE, Transfers, apply, cheb_to_uniform, nodes, uniform_nodes, uniform_to_cheb,
};

#[cfg(feature = "datagen")]
use rlst::DynArray;
#[cfg(feature = "datagen")]
use rlst::chebychev::ChebychevCoefficientsToData;

/// Chebyshev coefficients c_0..c_{n−1} of the CGL interpolant of the unit vector
/// e_j, with e_j indexed in RLST's descending node order.
#[cfg(feature = "datagen")]
fn unit_coeffs(n: usize, j: usize) -> DynArray<f64, 1> {
    let mut e = DynArray::<f64, 1>::from_shape([n]);
    e[[j]] = 1.0;
    e.chebychev_coeffs_from_data_second_kind()
}

/// Panic with a uniform message when a 1D grid has fewer than two nodes.
fn check_n(caller: &str, n: usize) {
    assert!(
        n >= 2,
        "{caller}: n must be >= 2 (CGL nodes include both endpoints), got {n}"
    );
}
