//! Issue #8: spectral weight initialisation.
//!
//! - The default must stay bit-for-bit what it was before `SpectralInit`
//!   existed, given the same seed.
//! - Every scheme must draw from the distribution its docs promise.
//!
//! Own test binary (own process) because these tests reseed Flex's RNG, which
//! is process-wide: a draw or reseed from another test between the seed and
//! the draws would change the weights (a reseed can even make two "independent"
//! parts identical). Within this binary, every test holds [`rng`] for its
//! whole body.

use std::sync::{Mutex, MutexGuard};

use burn::{
    Tensor,
    store::{ModuleSnapshot, bridge::to_data},
    tensor::{Device, Distribution},
};
use sciml_rs::neural_operators::{
    layers::spectral_convolution::{SpectralConv, SpectralInit},
    models::fno::{FNO, FNOConfig},
};

/// Serialises use of the process-wide RNG across this binary's tests.
fn rng() -> MutexGuard<'static, ()> {
    static RNG: Mutex<()> = Mutex::new(());
    // A failed test poisons the lock; the guarded data is `()`, so carry on.
    RNG.lock().unwrap_or_else(|e| e.into_inner())
}

/// Every parameter, by name, as raw f32 values.
fn params<M: ModuleSnapshot>(module: &M) -> Vec<(String, Vec<f32>)> {
    let mut out: Vec<_> = module
        .collect(None, None, false)
        .iter()
        .map(|t| {
            let values = to_data(t).unwrap().try_to_vec::<f32>().unwrap();
            (t.name.clone(), values)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Exact equality of f32 bit patterns.
fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

// --- default parity ---

#[test]
fn default_spectral_init_is_bit_identical() {
    let _rng = rng();
    const SEED: u64 = 2024;
    let device = Device::default();

    // 2D with asymmetric channels: two corners, and I != O.
    let (i, o, modes) = (3usize, 5usize, [4usize, 3]);
    device.seed(SEED);
    let conv = SpectralConv::<4>::new(&device, i, o, &modes);

    // The pre-#8 constructor, replayed by hand: Uniform(0, 1/(I·O)), and per
    // corner the real part then the imaginary part.
    device.seed(SEED);
    let scale = 1.0 / (i * o) as f64;
    let shape = [i, o, modes[0], modes[1]];
    let mut replay = Vec::new();
    for corner in 0..2 {
        for part in ["re", "im"] {
            let t = Tensor::<4>::random(shape, Distribution::Uniform(0.0, scale), &device);
            let values = t.into_data().try_to_vec::<f32>().unwrap();
            replay.push((format!("weights_{part}.{corner}"), values));
        }
    }
    replay.sort_by(|a, b| a.0.cmp(&b.0));

    let got = params(&conv);
    let names: Vec<_> = got.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "weights_im.0",
            "weights_im.1",
            "weights_re.0",
            "weights_re.1"
        ]
    );
    for ((name, got), (_, want)) in got.iter().zip(&replay) {
        assert_eq!(bits(got), bits(want), "{name} differs from the old draws");
    }

    // `new` is `new_with_init(.., LiUniform)`.
    device.seed(SEED);
    let explicit = SpectralConv::<4>::new_with_init(&device, i, o, &modes, SpectralInit::LiUniform);
    for ((name, a), (_, b)) in got.iter().zip(params(&explicit)) {
        assert_eq!(bits(a), bits(&b), "{name}: new vs new_with_init(LiUniform)");
    }

    // At model level, `spectral_init: None` and `Some(LiUniform)` give the
    // same weights for every parameter, spectral or not.
    let model = |init: Option<SpectralInit>| -> FNO<3> {
        device.seed(SEED);
        FNOConfig::new(vec![4], 1, 1)
            .with_hidden_channels(6)
            .with_n_layers(2)
            .with_spectral_init(init)
            .init::<3>(&device)
    };
    let none = params(&model(None));
    let li = params(&model(Some(SpectralInit::LiUniform)));
    assert_eq!(none.len(), li.len());
    for ((name, a), (name_b, b)) in none.iter().zip(&li) {
        assert_eq!(name, name_b);
        assert_eq!(bits(a), bits(b), "FNO {name}: None vs Some(LiUniform)");
    }

    // And a different scheme does change the spectral weights (so the
    // equalities above are not vacuous).
    let normal = params(&model(Some(SpectralInit::Normal)));
    let (name, a) = none
        .iter()
        .find(|(n, _)| n.starts_with("conv.0.weights_re"))
        .expect("spectral weight in FNO");
    let b = &normal.iter().find(|(n, _)| n == name).unwrap().1;
    assert_ne!(bits(a), bits(b), "{name}: Normal gave the Li weights");
}

// --- distributions ---

const ALL_INITS: [SpectralInit; 3] = [
    SpectralInit::LiUniform,
    SpectralInit::Normal,
    SpectralInit::SymmetricUniform,
];

/// `(I, O)` pairs. The asymmetric ones catch an `in`/`out` swap in
/// `SymmetricUniform`, whose bound depends on `I` alone.
const CHANNELS: [(usize, usize); 3] = [(4, 4), (2, 8), (8, 2)];

/// Per-corner `(re, im)` samples from a seeded 2D layer: 2 corners of
/// `I·O·16·16` weights each, so `n = 2·I·O·256 = 8192` per part for every
/// pair in `CHANNELS`. Callers must hold [`rng`].
fn init_samples(i: usize, o: usize, init: SpectralInit) -> Vec<(Vec<f64>, Vec<f64>)> {
    let device = Device::default();
    device.seed(11);
    let conv = SpectralConv::<4>::new_with_init(&device, i, o, &[16, 16], init);
    let p = params(&conv);
    let part = |name: String| -> Vec<f64> {
        let values = &p.iter().find(|(n, _)| *n == name).unwrap().1;
        values.iter().map(|&x| f64::from(x)).collect()
    };
    (0..2)
        .map(|c| {
            (
                part(format!("weights_re.{c}")),
                part(format!("weights_im.{c}")),
            )
        })
        .collect()
}

/// Pooled real and imaginary samples over all corners.
fn pooled(i: usize, o: usize, init: SpectralInit) -> (Vec<f64>, Vec<f64>) {
    let (mut re, mut im) = (Vec::new(), Vec::new());
    for (r, m) in init_samples(i, o, init) {
        re.extend(r);
        im.extend(m);
    }
    (re, im)
}

fn mean_var(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var)
}

/// The per-part distribution each scheme promises, written out from the
/// formulas in the `SpectralInit` docs rather than taken from the
/// implementation, so the tests are not tautological.
struct Expected {
    mean: f64,
    var: f64,
    /// `E[(x - μ)⁴] / σ⁴`: 9/5 for a uniform, 3 for a normal.
    kurtosis: f64,
    /// `[low, high]` for the uniform schemes.
    support: Option<(f64, f64)>,
}

fn expected(init: &SpectralInit, i: usize, o: usize) -> Expected {
    let (i, o) = (i as f64, o as f64);
    let uniform = |low: f64, high: f64| Expected {
        mean: (low + high) / 2.0,
        var: (high - low).powi(2) / 12.0,
        kurtosis: 1.8,
        support: Some((low, high)),
    };
    match init {
        SpectralInit::LiUniform => uniform(0.0, 1.0 / (i * o)),
        SpectralInit::Normal => Expected {
            mean: 0.0,
            var: 1.0 / (i * o),
            kurtosis: 3.0,
            support: None,
        },
        SpectralInit::SymmetricUniform => uniform(-1.0 / i.sqrt(), 1.0 / i.sqrt()),
    }
}

/// Checks one part (re or im) against its promised distribution.
///
/// Every bound is 6 standard errors of the statistic. The seed is fixed, so
/// each check passes or fails deterministically; the bound is chosen so a
/// correct implementation would fail for fewer than 1 in 1e8 seeds, i.e. it
/// stays valid if the seed or RNG changes. With `n = 8192` that is about
/// 7% of σ for the mean and 6% (uniform) or 9% (normal) for the variance,
/// while the plausible wrong formulas are off by at least 2×: std vs variance
/// in `Normal` (16× here), `1/I` vs `1/√I` in `SymmetricUniform` (≥ 2×), `I`
/// vs `O` (4×), or `[0, s)` vs `[-s, s)` (mean off by s/2 ≈ 157 standard
/// errors).
fn check_part(label: &str, xs: &[f64], e: &Expected) {
    let n = xs.len() as f64;
    let (mean, var) = mean_var(xs);

    let mean_tol = 6.0 * e.var.sqrt() / n.sqrt();
    assert!(
        (mean - e.mean).abs() <= mean_tol,
        "{label}: mean {mean:e}, expected {:e} ± {mean_tol:e}",
        e.mean
    );

    // Var(s²) ≈ σ⁴ (κ - 1) / n.
    let var_tol = 6.0 * ((e.kurtosis - 1.0) / n).sqrt();
    let rel = var / e.var - 1.0;
    assert!(
        rel.abs() <= var_tol,
        "{label}: variance {var:e}, expected {:e} (rel err {rel:+.4}, tol {var_tol:.4})",
        e.var
    );

    if let Some((low, high)) = e.support {
        let width = high - low;
        let (min, max) = xs
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
                (a.min(x), b.max(x))
            });
        // Samples are f32, so allow one f32 rounding of a bound.
        let slack = 1e-6 * width;
        assert!(
            min >= low - slack && max <= high + slack,
            "{label}: samples span [{min:e}, {max:e}], outside [{low:e}, {high:e}]"
        );
        // With n = 8192, P(no sample in the outer 1%) = 0.99^n ≈ e^-82, so a
        // bound that is too tight is detected.
        assert!(
            min <= low + 0.01 * width && max >= high - 0.01 * width,
            "{label}: samples span [{min:e}, {max:e}], short of [{low:e}, {high:e}]"
        );
    }
}

#[test]
fn init_moments_and_support_match_formulas() {
    let _rng = rng();
    for init in ALL_INITS {
        for (i, o) in CHANNELS {
            let e = expected(&init, i, o);
            let (re, im) = pooled(i, o, init.clone());
            assert_eq!(re.len(), 8192);
            check_part(&format!("{init:?} I={i} O={o} re"), &re, &e);
            check_part(&format!("{init:?} I={i} O={o} im"), &im, &e);
        }
    }
}

/// The moment check alone would accept a uniform of the right variance.
/// `P(|x| ≤ σ)` is 0.6827 for a normal but `1/√3 ≈ 0.5774` for a uniform of
/// the same variance; the 6-standard-error band (≈ 0.031 at n = 8192)
/// separates them.
#[test]
fn init_normal_has_gaussian_shape() {
    let _rng = rng();
    let (i, o) = (4, 4);
    let sigma = (1.0 / (i * o) as f64).sqrt();
    let (re, im) = pooled(i, o, SpectralInit::Normal);
    for (label, xs) in [("re", &re), ("im", &im)] {
        let n = xs.len() as f64;
        let p = 0.682_689_492;
        let tol = 6.0 * (p * (1.0 - p) / n).sqrt();
        let frac = xs.iter().filter(|x| x.abs() <= sigma).count() as f64 / n;
        assert!(
            (frac - p).abs() <= tol,
            "Normal {label}: P(|x| <= σ) = {frac:.4}, expected {p:.4} ± {tol:.4}"
        );
    }
}

/// Real and imaginary parts, and different corners, are separate draws. For
/// independent samples the sample correlation has standard error `1/√n`, so
/// `|r| ≤ 6/√n`; reusing one draw would give `r = 1`.
#[test]
fn init_parts_and_corners_are_independent() {
    fn corr(a: &[f64], b: &[f64]) -> f64 {
        let (ma, va) = mean_var(a);
        let (mb, vb) = mean_var(b);
        let n = a.len() as f64;
        let cov = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - ma) * (y - mb))
            .sum::<f64>()
            / (n - 1.0);
        cov / (va * vb).sqrt()
    }

    let _rng = rng();
    for init in ALL_INITS {
        let corners = init_samples(4, 4, init.clone());
        let pairs = [
            ("re0/im0", &corners[0].0, &corners[0].1),
            ("re1/im1", &corners[1].0, &corners[1].1),
            ("re0/re1", &corners[0].0, &corners[1].0),
            ("im0/im1", &corners[0].1, &corners[1].1),
        ];
        for (label, a, b) in pairs {
            let tol = 6.0 / (a.len() as f64).sqrt();
            let r = corr(a, b);
            assert!(
                r.abs() <= tol,
                "{init:?} {label}: correlation {r:+.4} exceeds {tol:.4}"
            );
        }
    }
}
