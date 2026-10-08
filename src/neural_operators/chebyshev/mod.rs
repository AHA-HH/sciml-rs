//! Chebyshev toolkit
//!
//! One-dimensional building blocks on the Chebyshev–Gauss–Lobatto (CGL) grid
//! (CONVENTIONS §12): ascending nodes, differentiation matrices, Clenshaw–Curtis
//! weights, and the tensor-product L² norm built from them. Everything is f64.
//!
//! Nodes, D, D² and the weights wrap RLST's FFTW-backed `chebychev` transforms
//! (design §12, decision 12), so this module needs the `chebyshev` feature and a
//! system libfftw3. RLST works in descending node order; every function here
//! returns ascending order, and the public API takes and returns ndarray types only.

mod differentiation;
mod norm;
mod points;
mod quadrature;

pub use differentiation::{diff_matrix, diff2_matrix};
pub use norm::{l2_norm, rel_l2_error};
pub use points::nodes;
pub use quadrature::clenshaw_curtis;

use rlst::DynArray;
use rlst::chebychev::ChebychevCoefficientsToData;

/// Chebyshev coefficients c_0..c_{n−1} of the CGL interpolant of the unit vector
/// e_j, with e_j indexed in RLST's descending node order.
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
