//! Chebyshev toolkit
//!
//! One-dimensional building blocks on the Chebyshev–Gauss–Lobatto (CGL) grid
//! (CONVENTIONS §12). Everything is f64, and the public API takes and returns ndarray
//! types only.
//!
//! - Always built: the grid transfers between the CGL grid and the model's uniform grid
//!   (`uniform_nodes`, `cheb_to_uniform`, `uniform_to_cheb`, `apply`). They use
//!   closed-form nodes and weights and depend on ndarray alone, so training on
//!   Chebyshev data needs no RLST, BLAS or FFTW (design §12, decision 13).
//! - Behind the `chebyshev` feature: the nodes, D, D², the Clenshaw–Curtis weights and
//!   the tensor-product L² norm built from them (`nodes`, `diff_matrix`, `diff2_matrix`,
//!   `clenshaw_curtis`, `l2_norm`, `rel_l2_error`). They wrap RLST's FFTW-backed
//!   `chebychev` transforms (design §12, decision 12) and need a system libfftw3. RLST
//!   works in descending node order; every function here returns ascending order.

#[cfg(feature = "chebyshev")]
mod differentiation;
#[cfg(feature = "chebyshev")]
mod norm;
#[cfg(feature = "chebyshev")]
mod points;
#[cfg(feature = "chebyshev")]
mod quadrature;
mod transfer;

#[cfg(feature = "chebyshev")]
pub use differentiation::{diff_matrix, diff2_matrix};
#[cfg(feature = "chebyshev")]
pub use norm::{l2_norm, rel_l2_error};
#[cfg(feature = "chebyshev")]
pub use points::nodes;
#[cfg(feature = "chebyshev")]
pub use quadrature::clenshaw_curtis;
pub use transfer::{apply, cheb_to_uniform, uniform_nodes, uniform_to_cheb};

#[cfg(feature = "chebyshev")]
use rlst::DynArray;
#[cfg(feature = "chebyshev")]
use rlst::chebychev::ChebychevCoefficientsToData;

/// Chebyshev coefficients c_0..c_{n−1} of the CGL interpolant of the unit vector
/// e_j, with e_j indexed in RLST's descending node order.
#[cfg(feature = "chebyshev")]
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
