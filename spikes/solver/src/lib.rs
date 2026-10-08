//! Phase 0 T4 spike: reference solvers for −Δu = f on [−1, 1]², u = 0 on the boundary.
//!
//! Throwaway measurement code (see `docs/phase0/T4-solver-spike.md`). Phase 1 rewrites
//! what is kept, with its own tests.

extern crate blas_src;
extern crate lapack_src;

pub mod cheb;
pub mod colloc;
pub mod fem;
pub mod manufactured;

use ndarray::Array2;
use rlst::DynArray;

/// ndarray → rlst (column-major) copy.
pub fn to_dyn(a: &Array2<f64>) -> DynArray<f64, 2> {
    let (m, n) = a.dim();
    let mut out = DynArray::<f64, 2>::from_shape([m, n]);
    for ((i, j), v) in a.indexed_iter() {
        out[[i, j]] = *v;
    }
    out
}

/// rlst → ndarray copy.
pub fn from_dyn(a: &DynArray<f64, 2>) -> Array2<f64> {
    let [m, n] = a.shape();
    Array2::from_shape_fn((m, n), |(i, j)| a[[i, j]])
}

/// Owned transpose (rlst's `dot` needs raw access, which a transpose view lacks).
pub fn transpose(a: &DynArray<f64, 2>) -> DynArray<f64, 2> {
    let [m, n] = a.shape();
    let mut out = DynArray::<f64, 2>::from_shape([n, m]);
    for i in 0..m {
        for j in 0..n {
            out[[j, i]] = a[[i, j]];
        }
    }
    out
}

/// Interior block `[1..n-1, 1..n-1]` of an n × n field.
pub fn interior(f: &Array2<f64>) -> Array2<f64> {
    let (m, n) = f.dim();
    f.slice(ndarray::s![1..m - 1, 1..n - 1]).to_owned()
}

/// Embed interior values into an n × n field with zero boundary.
pub fn with_boundary(u: &Array2<f64>) -> Array2<f64> {
    let (m, n) = u.dim();
    let mut out = Array2::zeros((m + 2, n + 2));
    out.slice_mut(ndarray::s![1..m + 1, 1..n + 1]).assign(u);
    out
}

/// max |a − b| / max |b|.
pub fn rel_max(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let err = a
        .iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    err / scale
}
