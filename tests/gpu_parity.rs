//! GPU parity regression tests: the Burn fusion bugs reported in `upstream/burn/` and the
//! 2D replay check at the Poisson trainer's configuration.
//!
//! - The GPU `repro_*` tests reproduce the Burn bugs at the pinned rev, so on a fused
//!   backend that has them they fail (that is the finding). They are `#[ignore]`d until
//!   the fork is fixed; `-- --ignored` runs them.
//! - Their `*_flex` twins run the same bodies on Flex, against the same fixed values and
//!   naive f64 DFT. They are not behind a GPU feature, so CI runs them.
//! - `replay_2d_poisson` compares the GPU against Flex and must pass before trusting a
//!   GPU padding sweep.
//!
//! Run with `cargo test [--release] --features metal --test gpu_parity [-- --ignored]`
//! (or `--features cuda`). Only those two features have a GPU here: with `wgpu` alone, only
//! the Flex twins compile. The exploratory diagnostics behind the bug reports are in
//! `spikes/metal_diag/`.

use burn::prelude::*;
use burn::tensor::signal;

/// The GPU under test: CUDA when the `cuda` feature is on.
#[cfg(feature = "cuda")]
fn gpu() -> Device {
    Device::cuda(0)
}

/// The GPU under test: Metal when the `metal` feature is on and `cuda` is not.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn gpu() -> Device {
    Device::metal(Default::default())
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn v<const R: usize>(t: Tensor<R>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>();
    let den: f32 = b.iter().map(|y| y * y).sum::<f32>();
    (num / den.max(1e-30)).sqrt()
}

/// Burn bug 1 (`upstream/burn/cfft-kernel/`): a fused `cat` broadcast against a wider
/// tensor reads the wrong stride. Fixed values; asserts the exact product.
fn fused_cat_broadcast(device: &Device) {
    let a = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], device);
    let p = Tensor::<2>::from_floats([[10.0], [20.0]], device);
    let q = Tensor::<2>::from_floats([[30.0], [40.0]], device);

    let out = a * Tensor::cat(vec![p, q], 0); // [4, 2] * [4, 1] -> [4, 2]
    let got = out.into_data().try_to_vec::<f32>().unwrap();
    println!("got {got:?}");
    assert_eq!(got, [10.0, 20.0, 60.0, 80.0, 150.0, 180.0, 280.0, 320.0]);
}

/// Burn bug 1 as it reaches Bluestein: cfft of a `[6, 4]` tensor along axis 0 (non-last,
/// non-power-of-two) against a naive f64 DFT.
fn cfft_bluestein(device: &Device) {
    let (n, w) = (6usize, 4usize);
    let vals: Vec<f32> = (0..n * w).map(|k| ((k * 7 % 11) as f32) - 5.0).collect();
    let re = Tensor::<2>::from_data(TensorData::new(vals.clone(), [n, w]), device);
    let im = re.zeros_like();
    let (out_re, _) = signal::cfft(re, im, 0, Some(n));
    let got = out_re.into_data().try_to_vec::<f32>().unwrap();
    let mut max_err = 0.0f32;
    for k in 0..n {
        for c in 0..w {
            let want: f64 = (0..n)
                .map(|j| {
                    vals[j * w + c] as f64
                        * (2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64).cos()
                })
                .sum();
            max_err = max_err.max((got[k * w + c] - want as f32).abs());
        }
    }
    println!("max abs err vs naive DFT: {max_err}");
    assert!(
        max_err < 1e-3,
        "cfft [6, 4] axis 0 wrong: max abs err {max_err}"
    );
}

/// [`fused_cat_broadcast`] on the GPU.
#[cfg(any(feature = "metal", feature = "cuda"))]
#[test]
#[ignore = "Burn fusion bug at the pinned rev: upstream/burn/cfft-kernel/README.md"]
fn repro_fused_cat_broadcast() {
    fused_cat_broadcast(&gpu());
}

/// [`cfft_bluestein`] on the GPU.
#[cfg(any(feature = "metal", feature = "cuda"))]
#[test]
#[ignore = "Burn fusion bug at the pinned rev: upstream/burn/cfft-kernel/README.md"]
fn repro_cfft_bluestein() {
    cfft_bluestein(&gpu());
}

/// [`fused_cat_broadcast`] on Flex, the reference backend.
#[test]
fn repro_fused_cat_broadcast_flex() {
    fused_cat_broadcast(&Device::flex());
}

/// [`cfft_bluestein`] on Flex, the reference backend.
#[test]
fn repro_cfft_bluestein_flex() {
    cfft_bluestein(&Device::flex());
}

/// Bug-report repro 2, as published.
#[cfg(any(feature = "metal", feature = "cuda"))]
#[test]
#[ignore = "Burn fusion bug at the pinned rev: upstream/burn/fused-cat-reused-at-new-shape/README.md"]
fn repro_fused_cat_reused_at_new_shape() {
    fn chain(x: Tensor<3>) -> Tensor<3> {
        let [b, n, _] = x.dims();
        let g = Tensor::<1, Int>::arange(0..n as i64, &x.device())
            .float()
            .reshape([1, n, 1])
            .expand([b, n, 1]);
        let g = Tensor::cat(vec![g], 2); // single-input cat
        Tensor::cat(vec![x, g], 2) // [b, n, 2]: (x, index) pairs
    }
    let device = gpu();
    let upload = |b: usize, n: usize| {
        let vals: Vec<f32> = (0..b * n).map(|i| 10.0 * (i + 1) as f32).collect();
        Tensor::<3>::from_data(TensorData::new(vals, [b, n, 1]), &device)
    };

    // One run at [2, 21, 1].
    let _ = chain(upload(2, 21)).into_data();

    // Then three runs at [3, 8, 1], each with a freshly uploaded input.
    let want: Vec<f32> = (0..24)
        .flat_map(|i| [10.0 * (i + 1) as f32, (i % 8) as f32])
        .collect();
    for call in 0..3 {
        let got = chain(upload(3, 8)).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(got, want, "call {call}");
    }
}

/// One call in [`replay_2d_poisson`]'s schedule: a training step (forward + backward on
/// an autodiff device) or an evaluation forward.
#[cfg(any(feature = "metal", feature = "cuda"))]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Call {
    Train,
    Eval,
}

#[cfg(any(feature = "metal", feature = "cuda"))]
impl std::fmt::Display for Call {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self {
            Call::Train => "train",
            Call::Eval => "eval",
        })
    }
}

/// 2D replay check at the Poisson trainer's configuration: shape changes as in
/// training (full, partial and test batches), each shape called repeatedly,
/// inputs uploaded afresh per call; every GPU call compared to Flex
/// (forward output, and loss + input gradient through backward).
///
/// Default padding 0, 4, 8 and 16; `PADS=a,b,...` overrides.
#[cfg(any(feature = "metal", feature = "cuda"))]
#[test]
fn replay_2d_poisson() {
    use burn::module::Module;
    use sciml_rs::neural_operators::layers::spectral_convolution::SpectralInit;
    use sciml_rs::neural_operators::models::fno::FNOConfig;
    let pads: Vec<usize> = std::env::var("PADS")
        .map(|s| s.split(',').map(|p| p.parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![0, 4, 8, 16]);
    let (m, f) = (gpu(), Device::flex());
    let s = 64;
    let input = |b: usize, d: &Device| {
        let vals: Vec<f32> = (0..b * s * s)
            .map(|i| ((i as f32) * 0.0137).sin() + 0.3 * ((i as f32) * 0.0031).cos())
            .collect();
        Tensor::<4>::from_data(TensorData::new(vals, [b, s, s, 1]), d)
    };
    // Training-like order: train batches (20, partial 7), eval batches (20, 3).
    let schedule = [
        (Call::Train, 20),
        (Call::Train, 20),
        (Call::Train, 7),
        (Call::Eval, 20),
        (Call::Eval, 3),
        (Call::Train, 20),
        (Call::Train, 7),
        (Call::Eval, 20),
    ];
    let mut worst = 0.0f32;
    for p in pads {
        f.seed(5);
        let model_f = FNOConfig::new(vec![12, 12], 1, 1)
            .with_hidden_channels(32)
            .with_n_layers(4)
            .with_padding(Some(p))
            .with_spectral_init(Some(SpectralInit::SymmetricUniform))
            .init::<4>(&f);
        let model_m = model_f.clone().to_device(&m);
        let (fa, ma) = (f.clone().autodiff(), m.clone().autodiff());
        let (model_fa, model_ma) = (
            model_f.clone().to_device(&fa),
            model_f.clone().to_device(&ma),
        );

        // Flex references, once per (kind, batch).
        let mut refs = std::collections::HashMap::<(Call, usize), Vec<f32>>::new();
        let train_step = |model: &sciml_rs::neural_operators::models::fno::FNO<4>, x: Tensor<4>| {
            let x = x.require_grad();
            let loss = model.forward(x.clone()).powf_scalar(2.0).mean();
            let g = loss.backward();
            let gx = x.grad(&g).unwrap();
            let mut out = v(loss.inner().reshape([1, 1, 1, 1]));
            out.extend(v(gx));
            out
        };
        let mut line = Vec::new();
        for (call, &(kind, b)) in schedule.iter().enumerate() {
            let want = refs.entry((kind, b)).or_insert_with(|| match kind {
                Call::Eval => v(model_f.forward(input(b, &f))),
                Call::Train => train_step(&model_fa, input(b, &fa)),
            });
            let got = match kind {
                Call::Eval => v(model_m.forward(input(b, &m))),
                Call::Train => train_step(&model_ma, input(b, &ma)),
            };
            let e = rel(&got, want);
            worst = worst.max(e);
            line.push(format!(
                "{call}:{kind}{b}={e:.1e}{}",
                if e > 1e-5 { "!" } else { "" }
            ));
        }
        println!("p={p:<2} {}", line.join(" "));
    }
    println!("worst gpu/flex rel err: {worst:.2e}");
    assert!(worst <= 1e-5, "GPU differs from Flex: {worst:e}");
}
