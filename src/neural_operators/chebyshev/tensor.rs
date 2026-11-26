//! N-D tensor helpers on fields of any rank (3D L1 gate, C4): one 1D matrix applied along
//! each axis ([`apply_nd`]), a field read as `[n_0, rest]` ([`as_matrix`]), the Kronecker
//! product of per-axis weights ([`kron`]), the rank-2 view ([`rank2`]) and the per-axis
//! shape check ([`check_per_axis`]). The transfers, the Clenshaw–Curtis norms and A3's
//! mode products are built on them.

use ndarray::{Array, Array1, ArrayD, ArrayView, ArrayView1, ArrayView2, ArrayViewD, CowArray};
use ndarray::{Dimension, Ix2, IxDyn};

use super::transfer::apply;

/// Apply one 1D transfer along each axis of a field of any rank (CONVENTIONS §12): along
/// x, then y, then z for an 'ijk' field.
///
/// `f: [n_0, …, n_{r−1}]` is stored with axis a the a-th coordinate; `ts[a]: [m_a, n_a]`
/// acts on axis a. Returns `[m_0, …, m_{r−1}]`, the same rank and layout. Rank 2 is
/// [`apply`] itself, so 2D outputs are bit for bit those of [`apply`] (3D D1, D2).
/// Otherwise `ts[0]` is applied as one matrix product on `f` read as `[n_0, rest]`, then
/// the remaining transfers on each slice along axis 0, so a rank-3 field's (y, z) step is
/// [`apply`] too.
///
/// # Panics
/// If `ts.len()` is not the rank of `f` (at least 1), or `ts[a]` has not `n_a` columns.
pub fn apply_nd<D: Dimension>(ts: &[ArrayView2<f64>], f: ArrayView<f64, D>) -> Array<f64, D> {
    check_per_axis(
        "apply_nd",
        ("transfer", "ts", "columns"),
        f.shape(),
        ts.iter().map(|t| t.ncols()),
    );
    apply_dyn(ts, f.into_dyn())
        .into_dimensionality()
        .expect("apply_dyn keeps the rank")
}

/// [`apply_nd`] on a dynamic-rank view whose shape the caller has checked.
fn apply_dyn(ts: &[ArrayView2<f64>], f: ArrayViewD<f64>) -> ArrayD<f64> {
    if let [tx, ty] = ts {
        return apply(*tx, *ty, rank2(f)).into_dyn();
    }
    let (t0, rest) = ts.split_first().expect("rank >= 1");
    let shape = f.shape().to_vec();
    let flat = as_matrix(f);
    let mut out_shape = shape;
    out_shape[0] = t0.nrows();
    let along_0 = t0
        .dot(&flat)
        .into_shape_with_order(IxDyn(&out_shape))
        .expect("standard layout");
    if rest.is_empty() {
        return along_0;
    }
    for (a, t) in rest.iter().enumerate() {
        out_shape[a + 1] = t.nrows();
    }
    let mut out = ArrayD::zeros(out_shape);
    for (mut dst, src) in out.outer_iter_mut().zip(along_0.outer_iter()) {
        dst.assign(&apply_dyn(rest, src));
    }
    out
}

/// `v: [n_0, …]` read as `[n_0, rest]` in row-major order, copied only if not contiguous.
pub(super) fn as_matrix<D: Dimension>(v: ArrayView<f64, D>) -> CowArray<f64, Ix2> {
    let rows = v.shape()[0];
    let cols = v.len() / rows.max(1);
    let v = if v.is_standard_layout() {
        CowArray::from(v)
    } else {
        CowArray::from(v.as_standard_layout().into_owned())
    };
    v.into_shape_with_order((rows, cols))
        .expect("standard layout")
}

/// A view the caller knows to be rank 2, as `[n_x, n_y]`.
pub(super) fn rank2<D: Dimension>(v: ArrayView<'_, f64, D>) -> ArrayView2<'_, f64> {
    v.into_dimensionality().expect("rank 2")
}

/// Panic unless `lens` holds one entry per axis of `shape` (at least one axis) and entry a
/// equals `shape[a]`. `(item, name, unit)` word the messages: "{caller}: one {item} per
/// axis, …" and "{caller}: {name}[a] must have n_a {unit}".
pub(super) fn check_per_axis(
    caller: &str,
    (item, name, unit): (&str, &str, &str),
    shape: &[usize],
    lens: impl ExactSizeIterator<Item = usize>,
) {
    assert!(
        lens.len() == shape.len() && !shape.is_empty(),
        "{caller}: one {item} per axis, got {} for rank {}",
        lens.len(),
        shape.len()
    );
    for (a, (len, &n)) in lens.zip(shape).enumerate() {
        assert_eq!(len, n, "{caller}: {name}[{a}] must have {n} {unit}");
    }
}

/// The Kronecker product of the weight vectors, earlier axes outer: the weights of the
/// trailing axes flattened in row-major order. `[1.0]` for none.
pub(super) fn kron(ws: &[ArrayView1<f64>]) -> Array1<f64> {
    ws.iter().fold(Array1::ones(1), |acc, w| {
        acc.iter()
            .flat_map(|&p| w.iter().map(move |&q| p * q))
            .collect()
    })
}
