//! Optimisers CO3.1: Burn's `linalg::qr` (tensor ops, on the device) and `linalg::svd` (on
//! the host) at SOAP's factor sizes, in f32, against f64 host references; prints a
//! Markdown table of errors and median times. See `../../REPORT.md`.
//!
//! qr's input is SOAP's refresh A = P Q₀ (P symmetric PSD, eigenvalues 10^(−6 i/(n−1)),
//! Q₀ orthogonal); svd's is P itself (the first basis). Both are built in f64 and rounded
//! to f32, and the references decompose that same f32 matrix in f64.

use std::time::Instant;

use burn::{
    Tensor,
    linalg::{qr, svd},
    tensor::{Device, TensorData},
};
use num_traits::Float;

const SIZES: [usize; 4] = [12, 32, 128, 144];
const WARMUP: usize = 3;
const REPS: usize = 20;
/// QR sweeps per singular value for `svd` (LAPACK's `dbdsqr` allows 6).
const SWEEPS: usize = 30;

/// Row-major n × n matrices.
type Mat<T> = Vec<T>;

/// xorshift64* and Box–Muller: standard normals without a dependency.
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn matmul<T: Float>(a: &[T], b: &[T], n: usize) -> Mat<T> {
    let mut c = vec![T::zero(); n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] = c[i * n + j] + aik * b[k * n + j];
            }
        }
    }
    c
}

fn transpose<T: Float>(a: &[T], n: usize) -> Mat<T> {
    (0..n * n).map(|ij| a[(ij % n) * n + ij / n]).collect()
}

/// Householder QR of a square matrix in T's precision; R's diagonal made non-negative so
/// Q is unique for a non-singular A.
fn householder_qr<T: Float>(a: &[T], n: usize) -> (Mat<T>, Mat<T>) {
    let mut r = a.to_vec();
    let mut q: Mat<T> = (0..n * n)
        .map(|ij| {
            if ij / n == ij % n {
                T::one()
            } else {
                T::zero()
            }
        })
        .collect();
    for k in 0..n.saturating_sub(1) {
        let mut v: Vec<T> = (k..n).map(|i| r[i * n + k]).collect();
        let norm = v.iter().fold(T::zero(), |s, &x| s + x * x).sqrt();
        if norm == T::zero() {
            continue;
        }
        v[0] = v[0] + v[0].signum() * norm;
        let vv = v.iter().fold(T::zero(), |s, &x| s + x * x);
        let two = T::one() + T::one();
        // R ← H R, Q ← Q H with H = I − 2 v vᵀ / vᵀv on rows/columns k..n
        for j in 0..n {
            let dot = (k..n).fold(T::zero(), |s, i| s + v[i - k] * r[i * n + j]);
            let f = two * dot / vv;
            for i in k..n {
                r[i * n + j] = r[i * n + j] - f * v[i - k];
            }
        }
        for i in 0..n {
            let dot = (k..n).fold(T::zero(), |s, j| s + q[i * n + j] * v[j - k]);
            let f = two * dot / vv;
            for j in k..n {
                q[i * n + j] = q[i * n + j] - f * v[j - k];
            }
        }
    }
    sign_fix(&mut q, &mut r, n);
    (q, r)
}

/// Flips column j of Q and row j of R where R_jj < 0 (QR is unchanged).
fn sign_fix<T: Float>(q: &mut [T], r: &mut [T], n: usize) {
    for j in 0..n {
        if r[j * n + j] < T::zero() {
            (0..n).for_each(|i| q[i * n + j] = -q[i * n + j]);
            (0..n).for_each(|c| r[j * n + c] = -r[j * n + c]);
        }
    }
}

/// Eigenvalues of a symmetric matrix by cyclic Jacobi in f64, descending.
fn jacobi_eigenvalues(a: &[f64], n: usize) -> Vec<f64> {
    let mut a = a.to_vec();
    for _sweep in 0..100 {
        let off: f64 = (0..n * n)
            .filter(|ij| ij / n != ij % n)
            .map(|ij| a[ij] * a[ij])
            .sum();
        // absolute tolerance: valid because every P here has ‖P‖₂ = 1
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut ev: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    ev.sort_by(|x, y| y.total_cmp(x));
    ev
}

fn to_f64(a: &[f32]) -> Mat<f64> {
    a.iter().map(|&x| x as f64).collect()
}

fn fro(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// max |QᵀQ − I|.
fn orth_err(q: &[f64], n: usize) -> f64 {
    let qtq = matmul(&transpose(q, n), q, n);
    (0..n * n)
        .map(|ij| (qtq[ij] - if ij / n == ij % n { 1.0 } else { 0.0 }).abs())
        .fold(0.0, f64::max)
}

/// ‖A − QR‖_F / ‖A‖_F.
fn backward_err(a: &[f64], q: &[f64], r: &[f64], n: usize) -> f64 {
    let qr = matmul(q, r, n);
    let d: Vec<f64> = a.iter().zip(&qr).map(|(x, y)| x - y).collect();
    fro(&d) / fro(a)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn read(t: Tensor<2>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
}

fn median_secs(mut f: impl FnMut()) -> f64 {
    (0..WARMUP).for_each(|_| f());
    let mut ts: Vec<f64> = (0..REPS)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .collect();
    ts.sort_by(f64::total_cmp);
    ts[REPS / 2]
}

/// A random orthogonal matrix in f64 (Q of a Gaussian matrix).
fn orthogonal(n: usize, rng: &mut Rng) -> Mat<f64> {
    let g: Mat<f64> = (0..n * n).map(|_| rng.normal()).collect();
    householder_qr(&g, n).0
}

fn main() {
    let device = Device::default();
    println!("device: {device:?}; f32; median of {REPS} after {WARMUP} warm-up calls\n");
    println!(
        "| n | qr backward | qr orth | qr fwd vs f64 | f32 host baseline (bwd / orth / fwd) | qr time | host qr time | svd σ err / σ₁ | svd residual | svd orth | svd time |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for n in SIZES {
        let v = orthogonal(n, &mut rng);
        let q0 = orthogonal(n, &mut rng);
        let lambda: Vec<f64> = (0..n)
            .map(|i| 10f64.powf(-6.0 * i as f64 / (n - 1) as f64))
            .collect();
        let vl: Mat<f64> = (0..n * n).map(|ij| v[ij] * lambda[ij % n]).collect();
        let p64 = matmul(&vl, &transpose(&v, n), n);
        let a32: Vec<f32> = matmul(&p64, &q0, n).iter().map(|&x| x as f32).collect();
        let p32: Vec<f32> = p64.iter().map(|&x| x as f32).collect();
        let (a, p) = (to_f64(&a32), to_f64(&p32));

        // qr
        let (q_ref, _) = householder_qr(&a, n);
        let (q_h, r_h) = householder_qr(&a32, n);
        let (q_h, r_h) = (to_f64(&q_h), to_f64(&r_h));
        let a_t = || Tensor::<2>::from_data(TensorData::new(a32.clone(), [n, n]), &device);
        let (q_b, r_b) = qr(a_t(), true);
        let (mut q_b, mut r_b) = (to_f64(&read(q_b)), to_f64(&read(r_b)));
        sign_fix(&mut q_b, &mut r_b, n);
        let qr_time = median_secs(|| {
            let (q, r) = qr(a_t(), true);
            let _ = (q.into_data(), r.into_data());
        });
        // like for like with `qr_time`: A from the device, QR on the host, Q and R back
        let a_dev = a_t();
        let host_qr_time = median_secs(|| {
            let (q, r) = householder_qr(&read(a_dev.clone()), n);
            let q = Tensor::<2>::from_data(TensorData::new(q, [n, n]), &device);
            let r = Tensor::<2>::from_data(TensorData::new(r, [n, n]), &device);
            let _ = (q.into_data(), r.into_data());
        });

        // svd of the symmetric PSD P: σ = eigenvalues, U = V
        let ev = jacobi_eigenvalues(&p, n);
        let p_t = || Tensor::<2>::from_data(TensorData::new(p32.clone(), [n, n]), &device);
        let (u, s, _vt) = svd::<2, 1>(p_t(), SWEEPS);
        let u = to_f64(&read(u));
        let s: Vec<f64> = s
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|&x| x as f64)
            .collect();
        let sigma_err = max_abs_diff(&s, &ev) / ev[0];
        let pu = matmul(&p, &u, n);
        let us: Mat<f64> = (0..n * n).map(|ij| u[ij] * s[ij % n]).collect();
        let residual = fro(&pu.iter().zip(&us).map(|(x, y)| x - y).collect::<Vec<_>>()) / fro(&p);
        let svd_time = median_secs(|| {
            let (u, s, vt) = svd::<2, 1>(p_t(), SWEEPS);
            let _ = (u.into_data(), s.into_data(), vt.into_data());
        });

        println!(
            "| {n} | {:.1e} | {:.1e} | {:.1e} | {:.1e} / {:.1e} / {:.1e} | {:.2} ms | {:.2} ms | {:.1e} | {:.1e} | {:.1e} | {:.2} ms |",
            backward_err(&a, &q_b, &r_b, n),
            orth_err(&q_b, n),
            max_abs_diff(&q_b, &q_ref),
            backward_err(&a, &q_h, &r_h, n),
            orth_err(&q_h, n),
            max_abs_diff(&q_h, &q_ref),
            qr_time * 1e3,
            host_qr_time * 1e3,
            sigma_err,
            residual,
            orth_err(&u, n),
            svd_time * 1e3,
        );
    }
}
