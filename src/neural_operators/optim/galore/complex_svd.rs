//! The complex SVD of the Tensor GaLore projector (library CM3.6b): a host one-sided Jacobi
//! in f64, each vector turned to LAPACK `gesdd`'s phase and then tensorly's `svd_flip`
//! (see the [`galore`](super) module documentation).

use burn::prelude::*;

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
        .map(|(x, y)| Cx(x, y))
        .collect();
    let mut u = jacobi_left_vectors(a.clone(), m, n, r);
    for col in u.iter_mut() {
        // gesdd's phase: σ v_j[0] = (Aᴴ u_j)₀ = Σᵢ conj(aᵢ₀) uᵢⱼ real, or u₀ⱼ real.
        let w = if m >= n {
            (0..m).fold(Cx(0.0, 0.0), |acc, i| acc + a[i * n].conj() * col[i])
        } else {
            col[0]
        };
        assert!(
            w.norm_sq() > 0.0,
            "complex SVD: zero phase reference entry, the phase is undefined"
        );
        let phase = w.conj().unit();
        col.iter_mut().for_each(|x| *x = *x * phase);
        // svd_flip with torch.sgn: times u_max / |u_max|.
        let largest = col.iter().enumerate().fold(0, |best, (i, x)| {
            if x.norm_sq() > col[best].norm_sq() {
                i
            } else {
                best
            }
        });
        let sgn = col[largest].unit();
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
    ComplexTensor::new(part(|x| x.0), part(|x| x.1))
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
        .map(|k| Cx(if k / m == k % m { 1.0 } else { 0.0 }, 0.0))
        .collect();
    let tol = m as f64 * f64::EPSILON;
    // Rows below this (squared) norm are rounding noise (the m − n surplus rows of a tall
    // matrix end there): no relative orthogonality is reachable between them.
    let negligible = (tol * tol) * a.iter().map(Cx::norm_sq).sum::<f64>();
    let mut converged = false;
    for _ in 0..JACOBI_SWEEPS {
        converged = true;
        for p in 0..m {
            for t in p + 1..m {
                let (row_p, row_t) = (&a[p * n..(p + 1) * n], &a[t * n..(t + 1) * n]);
                let alpha: f64 = row_p.iter().map(Cx::norm_sq).sum();
                let beta: f64 = row_t.iter().map(Cx::norm_sq).sum();
                let gamma = row_p
                    .iter()
                    .zip(row_t)
                    .fold(Cx(0.0, 0.0), |acc, (&x, &y)| acc + x * y.conj());
                let g = gamma.norm_sq().sqrt();
                if g <= tol * (alpha * beta).sqrt() || alpha.min(beta) <= negligible {
                    continue;
                }
                converged = false;
                // Row t times e^{iφ} (γ = |γ| e^{iφ}) makes the pair's inner product real;
                // then the real rotation that zeroes it, the smaller root of
                // t² + 2ζt − 1 = 0, ζ = (β − α) / 2|γ|.
                let phase = gamma.unit();
                let zeta = (beta - alpha) / (2.0 * g);
                let tan = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let cos = 1.0 / (1.0 + tan * tan).sqrt();
                let sin = cos * tan;
                for (rows, len) in [(&mut a, n), (&mut q, m)] {
                    for k in 0..len {
                        let (x, y) = (rows[p * len + k], phase * rows[t * len + k]);
                        rows[p * len + k] = x.scale(cos) - y.scale(sin);
                        rows[t * len + k] = x.scale(sin) + y.scale(cos);
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
    let norm = |i: usize| -> f64 { a[i * n..(i + 1) * n].iter().map(Cx::norm_sq).sum() };
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| norm(j).total_cmp(&norm(i)));
    // U = Qᴴ: column i of U is row i of Q conjugated.
    order[..r]
        .iter()
        .map(|&i| q[i * m..(i + 1) * m].iter().map(|x| x.conj()).collect())
        .collect()
}

/// A complex number on the host, f64: (re, im). Not `num_complex::Complex64`: that crate is
/// only a transitive dependency, and the few operations needed do not justify a direct one.
#[derive(Clone, Copy, Debug)]
struct Cx(f64, f64);

impl Cx {
    fn conj(self) -> Self {
        Cx(self.0, -self.1)
    }

    fn norm_sq(&self) -> f64 {
        self.0 * self.0 + self.1 * self.1
    }

    /// self / |self|.
    fn unit(self) -> Self {
        self.scale(1.0 / self.norm_sq().sqrt())
    }

    fn scale(self, s: f64) -> Self {
        Cx(self.0 * s, self.1 * s)
    }
}

impl std::ops::Add for Cx {
    type Output = Cx;
    fn add(self, o: Cx) -> Cx {
        Cx(self.0 + o.0, self.1 + o.1)
    }
}

impl std::ops::Sub for Cx {
    type Output = Cx;
    fn sub(self, o: Cx) -> Cx {
        Cx(self.0 - o.0, self.1 - o.1)
    }
}

impl std::ops::Mul for Cx {
    type Output = Cx;
    fn mul(self, o: Cx) -> Cx {
        Cx(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
    }
}
