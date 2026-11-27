//! The SOAP tests of the optimisers proposal §9 (CO3.2): the identity basis against Burn's
//! Adam, a quadratic with a Kronecker Hessian, ranks 1–5 against a host f64 step, the
//! refresh against host f64, and the record round trip.

use super::*;
use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams},
};

fn device() -> Device {
    Default::default()
}

/// A deterministic stream of values in [−1, 1) (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d);
        (x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn tensor<const D: usize>(values: &[f64], shape: [usize; D]) -> Tensor<D> {
    let values: Vec<f32> = values.iter().map(|&v| v as f32).collect();
    Tensor::from_data(TensorData::new(values, shape), &device())
}

fn host<const D: usize>(t: &Tensor<D>) -> Vec<f64> {
    t.clone()
        .into_data()
        .try_to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |m, x| m.max(x.abs()))
}

/// The stride of dimension `k` of a row-major tensor of `shape`.
fn stride(shape: &[usize], k: usize) -> usize {
    shape[k + 1..].iter().product()
}

/// The mode-k Gram matrix Gₖ Gₖᵀ (dₖ × dₖ, row-major) of a row-major tensor.
fn mode_gram(x: &[f64], shape: &[usize], k: usize) -> Vec<f64> {
    let (d, s) = (shape[k], stride(shape, k));
    let mut gram = vec![0.0; d * d];
    for (idx, &v) in x.iter().enumerate() {
        let a = (idx / s) % d;
        let base = idx - a * s;
        for b in 0..d {
            gram[a * d + b] += v * x[base + b * s];
        }
    }
    gram
}

/// The mode-k product with Qᵀ (`back` false: y_a = Σ_b Q[b, a] x_b) or Q (`back` true:
/// y_a = Σ_b Q[a, b] x_b) along dimension k.
fn mode_apply(x: &[f64], shape: &[usize], k: usize, q: &[f64], back: bool) -> Vec<f64> {
    let (d, s) = (shape[k], stride(shape, k));
    let mut y = vec![0.0; x.len()];
    for (idx, out) in y.iter_mut().enumerate() {
        let a = (idx / s) % d;
        let base = idx - a * s;
        *out = (0..d)
            .map(|b| {
                let q = if back { q[a * d + b] } else { q[b * d + a] };
                q * x[base + b * s]
            })
            .sum();
    }
    y
}

/// The projection of x into the bases (`back` false) or back from them, for the dimensions
/// with a basis.
fn project_host(x: &[f64], shape: &[usize], bases: &[Option<Vec<f64>>], back: bool) -> Vec<f64> {
    let mut y = x.to_vec();
    for (k, q) in bases.iter().enumerate() {
        if let Some(q) = q {
            y = mode_apply(&y, shape, k, q, back);
        }
    }
    y
}

fn bases<const D: usize>(state: &SoapState<D>) -> Vec<Option<Vec<f64>>> {
    state
        .factors
        .iter()
        .map(|f| f.as_ref().map(|f| host(&f.basis)))
        .collect()
}

/// max |QᵀQ − I| of an n × n row-major matrix.
fn orthogonality_error(q: &[f64], n: usize) -> f64 {
    let mut err: f64 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let dot: f64 = (0..n).map(|r| q[r * n + i] * q[r * n + j]).sum();
            err = err.max((dot - if i == j { 1.0 } else { 0.0 }).abs());
        }
    }
    err
}

/// Q of the QR of an n × n row-major matrix by modified Gram–Schmidt (columns, f64): an
/// algorithm independent of the module's Householder QR.
fn gram_schmidt(a: &[f64], n: usize) -> Vec<f64> {
    let mut q = a.to_vec();
    for j in 0..n {
        for i in 0..j {
            let dot: f64 = (0..n).map(|r| q[r * n + i] * q[r * n + j]).sum();
            for r in 0..n {
                q[r * n + j] -= dot * q[r * n + i];
            }
        }
        let norm = (0..n).map(|r| q[r * n + j].powi(2)).sum::<f64>().sqrt();
        for r in 0..n {
            q[r * n + j] /= norm;
        }
    }
    q
}

/// A random orthogonal n × n matrix (row-major).
fn orthogonal(rng: &mut Rng, n: usize) -> Vec<f64> {
    gram_schmidt(&rng.vec(n * n), n)
}

fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    c
}

fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    (0..n * m).map(|idx| a[(idx % m) * n + idx / m]).collect()
}

/// U diag(s) Vᵀ for n × n U and V.
fn with_spectrum(u: &[f64], s: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    let us: Vec<f64> = (0..n * n).map(|idx| u[idx] * s[idx % n]).collect();
    matmul(&us, &transpose(v, n, n), n, n, n)
}

/// Every column of `got` equal to the same column of `want` up to sign, within `tol`.
fn assert_columns_up_to_sign(got: &[f64], want: &[f64], n: usize, tol: f64, what: &str) {
    for j in 0..n {
        let col = |m: &[f64]| (0..n).map(|r| m[r * n + j]).collect::<Vec<_>>();
        let (g, w) = (col(got), col(want));
        let neg: Vec<f64> = w.iter().map(|x| -x).collect();
        let err = max_abs_diff(&g, &w).min(max_abs_diff(&g, &neg));
        assert!(err < tol, "{what}: column {j} differs by {err}");
    }
}

/// With every basis the identity (`max_precond_dim` 0, the reference code's own route to
/// Adam), SOAP after its first call equals Burn's Adam fed the same gradients.
#[test]
fn identity_basis_is_adam() {
    let mut rng = Rng(7);
    let shape = [3, 4, 5];
    let n = 60;
    let w0 = tensor(&rng.vec(n), shape);
    let soap = SoapConfig::new()
        .with_max_precond_dim(0)
        .with_weight_decay(0.0)
        .build();
    let adam = AdamConfig::new()
        .with_beta_1(0.95)
        .with_beta_2(0.95)
        .with_epsilon(1e-8)
        .build();
    let lr = 1e-2;

    let (w_soap, mut s_soap) = soap.step(lr, w0.clone(), tensor(&rng.vec(n), shape), None);
    assert_eq!(
        host(&w_soap),
        host(&w0),
        "the first call must not move the tensor"
    );
    assert!(s_soap.as_ref().unwrap().factors.iter().all(Option::is_none));

    let (mut w_soap, mut w_adam, mut s_adam) = (w_soap, w0, None);
    for _ in 0..20 {
        let g = tensor(&rng.vec(n), shape);
        (w_soap, s_soap) = soap.step(lr, w_soap, g.clone(), s_soap);
        (w_adam, s_adam) = adam.step(lr, w_adam, g, s_adam);
    }
    let err = max_abs_diff(&host(&w_soap), &host(&w_adam));
    assert!(
        err < 1e-6,
        "SOAP with identity bases differs from Adam by {err}"
    );
}

/// f(W) = ½⟨E, A E B⟩, E = W − W*, Hessian B ⊗ A with rotated A and B of κ = 10 each: the
/// first factors are (1 − β_s) G Gᵀ and (1 − β_s) Gᵀ G, and at the same decaying lr SOAP ends
/// below 1 % of Adam's loss (1.6e-7 and 1.5e-7 on flex and metal against 5.1e-2, from 2.0e2, when
/// written). At κ = 100 each the factors' κ is about 1e8, past f32: their trailing
/// eigenvectors are round-off and SOAP's result depends on the backend's rounding.
#[test]
fn kronecker_quadratic() {
    let (m, n) = (6, 5);
    let mut rng = Rng(11);
    let spd = |rng: &mut Rng, d: usize| {
        let u = orthogonal(rng, d);
        let s: Vec<f64> = (0..d)
            .map(|i| 10f64.powf(i as f64 / (d - 1) as f64))
            .collect();
        with_spectrum(&u, &s, &u, d)
    };
    let a = spd(&mut rng, m);
    let b = spd(&mut rng, n);
    let w_star = rng.vec(m * n);
    let w0 = vec![0.0; m * n];
    let grad = |w: &[f64]| {
        let e: Vec<f64> = w.iter().zip(&w_star).map(|(x, y)| x - y).collect();
        matmul(&matmul(&a, &e, m, m, n), &b, m, n, n)
    };
    let loss = |w: &[f64]| {
        let e: Vec<f64> = w.iter().zip(&w_star).map(|(x, y)| x - y).collect();
        0.5 * e.iter().zip(grad(w)).map(|(x, g)| x * g).sum::<f64>()
    };

    let soap = SoapConfig::new().with_weight_decay(0.0).build();
    // lr 0.05 · 0.98ⁱ: at a constant lr both stall at the lr's noise floor and the bases
    // cannot show.
    let (lr, decay, steps) = (0.05, 0.98f64, 400);

    let g0 = grad(&w0);
    let (_, state) = soap.step(lr, tensor(&w0, [m, n]), tensor(&g0, [m, n]), None);
    let state = state.unwrap();
    let g0_32: Vec<f64> = g0.iter().map(|&v| f64::from(v as f32)).collect();
    for (k, want) in [
        matmul(&g0_32, &transpose(&g0_32, m, n), m, n, m),
        matmul(&transpose(&g0_32, m, n), &g0_32, n, m, n),
    ]
    .into_iter()
    .enumerate()
    {
        let want: Vec<f64> = want.iter().map(|v| 0.05 * v).collect();
        let got = host(&state.factors[k].as_ref().unwrap().gram);
        let err = max_abs_diff(&got, &want) / max_abs(&want);
        assert!(err < 1e-5, "factor {k} relative error {err}");
    }

    let run = |opt: &dyn Fn(f64, Tensor<2>, Tensor<2>) -> Tensor<2>| {
        let mut w = tensor(&w0, [m, n]);
        for i in 0..steps {
            let g = tensor(&grad(&host(&w)), [m, n]);
            w = opt(lr * decay.powi(i), w, g);
        }
        loss(&host(&w))
    };
    let soap_state = std::cell::RefCell::new(None);
    let soap_loss = run(&|lr, w, g| {
        let (w, s) = soap.step(lr, w, g, soap_state.take());
        soap_state.replace(s);
        w
    });
    let adam = AdamConfig::new()
        .with_beta_1(0.95)
        .with_beta_2(0.95)
        .with_epsilon(1e-8)
        .build();
    let adam_state = std::cell::RefCell::new(None);
    let adam_loss = run(&|lr, w, g| {
        let (w, s) = adam.step(lr, w, g, adam_state.take());
        adam_state.replace(s);
        w
    });
    let start = loss(&w0);
    assert!(
        soap_loss < 1e-2 * adam_loss,
        "SOAP {soap_loss:.3e} not below 1 % of Adam's {adam_loss:.3e}"
    );
    assert!(
        soap_loss < 1e-5 * start,
        "SOAP {soap_loss:.3e} against a start of {start:.3e}"
    );
}

/// Ranks 1 to 5: the factors are the EMA of the mode-k Grams, the bases orthonormal, and a
/// step equals its host f64 recomputation from the state.
#[test]
fn ranks_one_to_five() {
    fn check<const D: usize>(shape: [usize; D], seed: u64) {
        let mut rng = Rng(seed);
        let n: usize = shape.iter().product();
        // wd · lr = 1e-2: decaying before the step instead of after moves the result by about
        // 1e-2 of the update, far above the tolerance.
        let (lr, wd, beta) = (1e-2, 1.0, 0.95);
        let soap = SoapConfig::new()
            .with_weight_decay(wd)
            .with_precondition_frequency(100)
            .build();
        let mut w = tensor(&rng.vec(n), shape);
        let mut state = None;
        let mut grams: Vec<Vec<f64>> = shape.iter().map(|&d| vec![0.0; d * d]).collect();
        for _ in 0..3 {
            let g = tensor(&rng.vec(n), shape);
            let g_host = host(&g);
            for (k, gram) in grams.iter_mut().enumerate() {
                for (x, y) in gram.iter_mut().zip(mode_gram(&g_host, &shape, k)) {
                    *x = beta * *x + (1.0 - beta) * y;
                }
            }
            (w, state) = soap.step(lr, w, g, state);
        }
        let s = state.unwrap();
        assert_eq!(s.factors.len(), D);
        for (k, f) in s.factors.iter().enumerate() {
            let Some(f) = f else {
                assert_eq!(D, 1, "rank {D}: dimension {k} has no factor");
                continue;
            };
            let err = max_abs_diff(&host(&f.gram), &grams[k]) / max_abs(&grams[k]);
            assert!(err < 1e-5, "rank {D}, factor {k}: relative error {err}");
            let orth = orthogonality_error(&host(&f.basis), shape[k]);
            assert!(orth < 1e-5, "rank {D}, basis {k}: max |QᵀQ − I| = {orth}");
        }

        // One more step, on the host in f64 from the state's Q, M and V.
        let g = tensor(&rng.vec(n), shape);
        let (g_h, w_h) = (host(&g), host(&w));
        let q = bases(&s);
        let (m, v) = (host(&s.moment_1), host(&s.moment_2));
        let t = (s.step + 1) as i32;
        let g_rot = project_host(&g_h, &shape, &q, false);
        let m: Vec<f64> = m
            .iter()
            .zip(&g_h)
            .map(|(m, g)| beta * m + (1.0 - beta) * g)
            .collect();
        let v: Vec<f64> = v
            .iter()
            .zip(&g_rot)
            .map(|(v, g)| beta * v + (1.0 - beta) * g * g)
            .collect();
        let m_rot = project_host(&m, &shape, &q, false);
        let n_rot: Vec<f64> = m_rot
            .iter()
            .zip(&v)
            .map(|(m, v)| m / (v.sqrt() + 1e-8))
            .collect();
        let dir = project_host(&n_rot, &shape, &q, true);
        let step_size = lr * (1.0 - beta.powi(t)).sqrt() / (1.0 - beta.powi(t));
        let want: Vec<f64> = w_h
            .iter()
            .zip(&dir)
            .map(|(w, d)| (w - step_size * d) * (1.0 - lr * f64::from(wd)))
            .collect();
        let (got, _) = soap.step(lr, w, g, Some(s));
        let delta: Vec<f64> = want.iter().zip(&w_h).map(|(a, b)| a - b).collect();
        let err = max_abs_diff(&host(&got), &want) / max_abs(&delta);
        assert!(err < 1e-4, "rank {D}: step relative error {err}");
    }
    check([7], 1);
    check([5, 6], 2);
    check([4, 3, 5], 3);
    check([3, 4, 1, 3], 4);
    check([2, 3, 2, 3, 2], 5);
}

/// The first basis against the eigenvectors it was built from, and a refresh (frequency 2)
/// against a host f64 sort, power step and Gram–Schmidt QR, with V permuted to follow.
#[test]
fn refresh_against_f64() {
    let n = 5;
    let shape = [n, n];
    let mut rng = Rng(23);
    let (u, v) = (orthogonal(&mut rng, n), orthogonal(&mut rng, n));
    let g1 = with_spectrum(&u, &[5.0, 4.0, 3.0, 2.0, 1.0], &v, n);
    let g2 = with_spectrum(&u, &[1.0, 2.0, 3.0, 4.0, 10.0], &v, n);
    let (beta, lr) = (0.95, 1e-2);
    let soap = SoapConfig::new()
        .with_precondition_frequency(2)
        .with_weight_decay(0.0)
        .build();

    let w = tensor(&rng.vec(n * n), shape);
    let (w, s) = soap.step(lr, w, tensor(&g1, shape), None);
    let s1 = s.unwrap();
    // Gram eigenvalues 25, 16, 9, 4, 1: gaps ≥ 3, so the f32 factor's eigenvectors are U's.
    assert_columns_up_to_sign(
        &host(&s1.factors[0].as_ref().unwrap().basis),
        &u,
        n,
        1e-5,
        "Q₀",
    );
    assert_columns_up_to_sign(
        &host(&s1.factors[1].as_ref().unwrap().basis),
        &v,
        n,
        1e-5,
        "Q₁",
    );

    let (w, s) = soap.step(lr, w, tensor(&g2, shape), Some(s1));
    let s2 = s.unwrap();
    assert_eq!(s2.step, 1);

    // The third call is Adam step 2: a refresh after the factor update.
    let g3 = tensor(&g2, shape);
    let g3_h = host(&g3);
    let q_old = bases(&s2);
    let g_rot = project_host(&g3_h, &shape, &q_old, false);
    let mut v_want: Vec<f64> = host(&s2.moment_2)
        .iter()
        .zip(&g_rot)
        .map(|(v, g)| beta * v + (1.0 - beta) * g * g)
        .collect();
    let mut q_want = Vec::new();
    for (k, q) in q_old.iter().enumerate() {
        let f = s2.factors[k].as_ref().unwrap();
        let gram: Vec<f64> = host(&f.gram)
            .iter()
            .zip(mode_gram(&g3_h, &shape, k))
            .map(|(a, b)| beta * a + (1.0 - beta) * b)
            .collect();
        let q = q.as_ref().unwrap();
        let lq = matmul(&gram, q, n, n, n);
        let est: Vec<f64> = (0..n)
            .map(|j| (0..n).map(|r| q[r * n + j] * lq[r * n + j]).sum())
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| est[b].total_cmp(&est[a]));
        assert_ne!(
            order,
            (0..n).collect::<Vec<_>>(),
            "the test should permute basis {k}"
        );
        let q_sorted: Vec<f64> = (0..n * n)
            .map(|idx| q[(idx / n) * n + order[idx % n]])
            .collect();
        q_want.push(gram_schmidt(&matmul(&gram, &q_sorted, n, n, n), n));
        let s = stride(&shape, k);
        v_want = (0..n * n)
            .map(|idx| {
                let a = (idx / s) % n;
                v_want[idx - a * s + order[a] * s]
            })
            .collect();
    }

    let (_, s) = soap.step(lr, w, g3, Some(s2));
    let s3 = s.unwrap();
    for (k, want) in q_want.iter().enumerate() {
        let got = host(&s3.factors[k].as_ref().unwrap().basis);
        assert_columns_up_to_sign(&got, want, n, 1e-5, &format!("refreshed Q{k}"));
    }
    let err = max_abs_diff(&host(&s3.moment_2), &v_want) / max_abs(&v_want);
    assert!(err < 1e-5, "V after the refresh: relative error {err}");
}

/// A bias (rank 1), a weight (rank 2) and a rank-3 parameter.
#[derive(Module, Debug)]
struct Toy {
    linear: Linear,
    cube: Param<Tensor<3>>,
}

impl Toy {
    fn loss(&self, x: &Tensor<2>) -> Tensor<1> {
        let y = self.linear.forward(x.clone());
        let c = self.cube.val();
        (y.powf_scalar(2.0).sum() + c.clone().mul(c).sin().sum()).reshape([1])
    }
}

/// Five steps, the optimiser saved to bytes and reloaded, five more across a refresh: the
/// same parameters bit for bit as without the reload.
#[test]
fn record_round_trip() {
    let dev = device().autodiff();
    let mut rng = Rng(31);
    let mut model = Toy {
        linear: LinearConfig::new(4, 3).init(&dev),
        cube: Param::from_tensor(tensor(&rng.vec(24), [2, 3, 4]).to_device(&dev)),
    };
    let x = tensor(&rng.vec(8), [2, 4]).to_device(&dev);
    let config = SoapConfig::new().with_precondition_frequency(3);
    let mut optim = config.init();
    let step = |optim: &mut ModuleOptimizer, model: Toy| {
        let grads = GradientsParams::from_grads(model.loss(&x).backward(), &model);
        optim.step(1e-2, model, grads)
    };
    for _ in 0..5 {
        model = step(&mut optim, model);
    }
    let mut reloaded = config
        .init()
        .from_bytes(optim.into_bytes().unwrap())
        .unwrap();
    let mut twin = model.clone();
    for _ in 0..5 {
        model = step(&mut optim, model);
        twin = step(&mut reloaded, twin);
    }
    let bits = |m: &Toy| {
        [
            host(&m.linear.weight.val().inner()),
            host(&m.linear.bias.as_ref().unwrap().val().inner()),
            host(&m.cube.val().inner()),
        ]
    };
    assert_eq!(bits(&model), bits(&twin));
}

/// A zero first gradient (a parameter outside the loss): zero factors, yet orthonormal
/// bases, and the next steps stay finite.
#[test]
fn zero_first_gradient() {
    let shape = [4, 3];
    let soap = SoapConfig::new().with_precondition_frequency(1).build();
    let w = tensor(&[0.5; 12], shape);
    let (w, s) = soap.step(1e-2, w, tensor(&[0.0; 12], shape), None);
    let s = s.unwrap();
    for (k, f) in s.factors.iter().enumerate() {
        let f = f.as_ref().unwrap();
        assert_eq!(max_abs(&host(&f.gram)), 0.0);
        let orth = orthogonality_error(&host(&f.basis), shape[k]);
        assert!(orth < 1e-6, "basis {k}: max |QᵀQ − I| = {orth}");
    }
    let mut rng = Rng(41);
    let (w, s) = soap.step(1e-2, w, tensor(&rng.vec(12), shape), Some(s));
    let (w, _) = soap.step(1e-2, w, tensor(&rng.vec(12), shape), s);
    assert!(host(&w).iter().all(|v| v.is_finite()));
}

/// A refresh frequency of 0 is refused when the optimiser is built.
#[test]
#[should_panic(expected = "precondition_frequency must be at least 1")]
fn zero_frequency_panics() {
    SoapConfig::new().with_precondition_frequency(0).build();
}

/// Optimisers CO3.3: the first refresh's log reads the eigenvalues of Q diag(λ) Qᵀ, a
/// spread of 1e10 included, to 1e-12 relative to the largest.
#[test]
fn symmetric_eigenvalues_of_a_known_spectrum() {
    let mut rng = Rng(7);
    for n in [1, 2, 12, 32] {
        let q = orthogonal(&mut rng, n);
        let want: Vec<f64> = (0..n)
            .map(|i| 10f64.powf(-10.0 * i as f64 / n as f64))
            .collect();
        let a = with_spectrum(&q, &want, &q, n);
        let a = ndarray::Array2::from_shape_vec((n, n), a).unwrap();
        let mut got = super::host::symmetric_eigenvalues(a);
        got.sort_by(|x, y| y.total_cmp(x));
        assert!(max_abs_diff(&got, &want) < 1e-12, "n = {n}: {got:?}");
    }
}

/// The first refresh's summary: a rank-2 factor of size 4 with round-off below the floor
/// has 2 eigenvalues resolved, and the resolved ratio is λ₁/λ₂.
#[test]
fn spectrum_summary_counts_the_resolved_eigenvalues() {
    let (max, min, resolved, min_resolved) =
        super::host::spectrum_summary(vec![1e-9, 2.0, -1e-9, 0.5]);
    assert_eq!((max, min, resolved, min_resolved), (2.0, -1e-9, 2, 0.5));
}
