//! The complex SVD of the Tensor GaLore projector (library CM3.6b): a host one-sided Jacobi
//! in f64, each vector turned to LAPACK `gesdd`'s phase and then tensorly's `svd_flip`
//! (see the [`galore`](super) module documentation).

use burn::prelude::*;
use num_complex::Complex64 as Cx;

use super::ComplexTensor;

/// The `r` leading left singular vectors of B + iC (`[m, n]`), `[m, r]`, with the
/// phase of LAPACK's `gesdd` (the first entry of each right singular vector real when
/// m ≥ n, of each left one when m < n) and then tensorly's complex `svd_flip` (each column
/// times sgn of its entry of largest modulus; the first on a tie). Computed on the host in
/// f64 by [`jacobi_left_vectors`]; returned in `b`'s precision. `r` ≤ min(m, n).
///
/// # Panics
/// If a reference entry is zero (e.g. the first column of a tall `matrix` is zero): the phase
/// is then undefined (LAPACK's arbitrary).
pub(super) fn complex_leading_vectors(matrix: ComplexTensor<2>, r: usize) -> ComplexTensor<2> {
    let (b, c) = (matrix.re, matrix.im);
    let [m, n] = b.dims();
    let (dtype, device) = (b.dtype(), b.device());
    let host = |t: Tensor<2>| {
        t.into_data()
            .convert::<f64>()
            .try_into_vec::<f64>()
            .unwrap()
    };
    let a: Vec<Cx> = host(b)
        .into_iter()
        .zip(host(c))
        .map(|(x, y)| Cx::new(x, y))
        .collect();
    let mut u = jacobi_left_vectors(a.clone(), m, n, r);
    for col in u.iter_mut() {
        // gesdd's phase: σ v_j[0] = (Aᴴ u_j)₀ = Σᵢ conj(aᵢ₀) uᵢⱼ real, or u₀ⱼ real.
        let w = if m >= n {
            (0..m).fold(Cx::ZERO, |acc, i| acc + a[i * n].conj() * col[i])
        } else {
            col[0]
        };
        assert!(
            w.norm_sqr() > 0.0,
            "complex SVD: zero phase reference entry, the phase is undefined"
        );
        let phase = w.conj() / w.norm();
        col.iter_mut().for_each(|x| *x = *x * phase);
        // svd_flip with torch.sgn: times u_max / |u_max|.
        let largest = col.iter().enumerate().fold(0, |best, (i, x)| {
            if x.norm_sqr() > col[best].norm_sqr() {
                i
            } else {
                best
            }
        });
        let sgn = col[largest] / col[largest].norm();
        col.iter_mut().for_each(|x| *x = *x * sgn);
    }
    let part = |f: fn(&Cx) -> f64| {
        let values: Vec<f64> = (0..m)
            .flat_map(|i| u.iter().map(move |col| col[i]))
            .map(|x| f(&x))
            .collect();
        Tensor::from_data(
            TensorData::new(values, [m, r]).convert_dtype(dtype),
            &device,
        )
    };
    ComplexTensor::new(part(|x| x.re), part(|x| x.im))
}

/// Jacobi sweeps allowed before [`jacobi_left_vectors`] gives up (it converges
/// quadratically; a handful suffice).
const JACOBI_SWEEPS: usize = 60;

/// The `r` leading left singular vectors (columns, by singular value, descending) of the
/// complex `m × n` matrix `a` (row-major), phases arbitrary: one-sided Jacobi (Hestenes) on
/// its rows, Wᴴ A = Z with orthogonal rows, so A = W Z and the columns of W, ordered by the
/// rows' norms, are U. Robust to repeated singular values, which stall Burn's QR `svd` on a
/// complex matrix's real embedding (each value twice) in f64.
///
/// # Panics
/// If the rows are not orthogonal after [`JACOBI_SWEEPS`] sweeps.
fn jacobi_left_vectors(mut a: Vec<Cx>, m: usize, n: usize, r: usize) -> Vec<Vec<Cx>> {
    // W accumulated as Q = Wᴴ, row by row with A's rows.
    let mut q: Vec<Cx> = (0..m * m)
        .map(|k| if k / m == k % m { Cx::ONE } else { Cx::ZERO })
        .collect();
    let tol = m as f64 * f64::EPSILON;
    // Rows below this (squared) norm are rounding noise (the m − n surplus rows of a tall
    // matrix end there): no relative orthogonality is reachable between them.
    let negligible = (tol * tol) * a.iter().map(Cx::norm_sqr).sum::<f64>();
    let mut converged = false;
    for _ in 0..JACOBI_SWEEPS {
        converged = true;
        for p in 0..m {
            for t in p + 1..m {
                let (row_p, row_t) = (&a[p * n..(p + 1) * n], &a[t * n..(t + 1) * n]);
                let alpha: f64 = row_p.iter().map(Cx::norm_sqr).sum();
                let beta: f64 = row_t.iter().map(Cx::norm_sqr).sum();
                let gamma = row_p
                    .iter()
                    .zip(row_t)
                    .fold(Cx::ZERO, |acc, (&x, &y)| acc + x * y.conj());
                let g = gamma.norm();
                if g <= tol * (alpha * beta).sqrt() || alpha.min(beta) <= negligible {
                    continue;
                }
                converged = false;
                // Row t times e^{iφ} (γ = |γ| e^{iφ}) makes the pair's inner product real;
                // then the real rotation that zeroes it, the smaller root of
                // t² + 2ζt − 1 = 0, ζ = (β − α) / 2|γ|.
                let phase = gamma / g;
                let zeta = (beta - alpha) / (2.0 * g);
                let tan = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let cos = 1.0 / (1.0 + tan * tan).sqrt();
                let sin = cos * tan;
                for (rows, len) in [(&mut a, n), (&mut q, m)] {
                    for k in 0..len {
                        let (x, y) = (rows[p * len + k], phase * rows[t * len + k]);
                        rows[p * len + k] = x * cos - y * sin;
                        rows[t * len + k] = x * sin + y * cos;
                    }
                }
            }
        }
        if converged {
            break;
        }
    }
    assert!(
        converged,
        "complex SVD: no convergence in {JACOBI_SWEEPS} Jacobi sweeps"
    );
    let norm = |i: usize| -> f64 { a[i * n..(i + 1) * n].iter().map(Cx::norm_sqr).sum() };
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| norm(j).total_cmp(&norm(i)));
    // U = Qᴴ: column i of U is row i of Q conjugated.
    order[..r]
        .iter()
        .map(|&i| q[i * m..(i + 1) * m].iter().map(|x| x.conj()).collect())
        .collect()
}
