//! GPU parity regression tests: the Burn fusion bugs reported in `upstream/burn/` and the
//! 2D replay check at the Poisson trainer's configuration.
//!
//! - The GPU `repro_*` tests reproduce the Burn bugs found on the fork `ddfa9af`; Burn
//!   0.23.0-pre.2 fixes all three on Metal, so they now run as regression tests.
//! - Their `*_flex` twins run the same bodies on Flex, against the same fixed values and
//!   naive f64 DFT. They are not behind a GPU feature, so CI runs them.
//! - `replay_2d_poisson` compares the GPU against Flex and must pass before trusting a
//!   GPU padding sweep.
//! - `grid_3d`, `spectral_conv_3d` and `replay_3d_poisson` (3D CL0.2) check the rank-5
//!   grid channels, spectral layer and FNO against Flex the same way, before 3D trains on
//!   a GPU.
//!
//! Run with `cargo test [--release] --features metal --test gpu_parity`
//! (or `--features cuda`). Only those two features have a GPU here: with `wgpu` alone, only
//! the Flex twins compile. The exploratory diagnostics behind the bug reports are in
//! `spikes/metal_diag/`.

use burn::prelude::*;
use burn::tensor::signal;
#[cfg(any(feature = "metal", feature = "cuda"))]
use sciml_rs::neural_operators::models::fno::FNO;

/// The GPU under test: CUDA when the `cuda` feature is on.
#[cfg(feature = "cuda")]
fn gpu() -> Device {
    Device::cuda(0)
}

/// The GPU under test: Metal when the `metal` feature is on and `cuda` is not.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn gpu() -> Device {
    Device::metal()
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
fn repro_fused_cat_broadcast() {
    fused_cat_broadcast(&gpu());
}

/// [`cfft_bluestein`] on the GPU.
#[cfg(any(feature = "metal", feature = "cuda"))]
#[test]
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

/// One call in a replay schedule ([`replay`]): a training step (forward + backward on
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

/// The replays' deterministic input pattern, at any shape.
#[cfg(any(feature = "metal", feature = "cuda"))]
fn wave<const R: usize>(dims: [usize; R], device: &Device) -> Tensor<R> {
    let vals: Vec<f32> = (0..dims.iter().product::<usize>())
        .map(|i| ((i as f32) * 0.0137).sin() + 0.3 * ((i as f32) * 0.0031).cos())
        .collect();
    Tensor::<R>::from_data(TensorData::new(vals, dims), device)
}

/// `[L, dL/dx..]` for L = mean(forward(x)²); `x` must be on an autodiff device.
#[cfg(any(feature = "metal", feature = "cuda"))]
fn loss_and_grad<const R: usize>(
    forward: impl Fn(Tensor<R>) -> Tensor<R>,
    x: Tensor<R>,
) -> Vec<f32> {
    let x = x.require_grad();
    let loss = forward(x.clone()).powf_scalar(2.0).mean();
    let g = loss.backward();
    let gx = x.grad(&g).unwrap();
    let mut out = v(loss.inner());
    out.extend(v(gx));
    out
}

/// max |a − b| / max |b|: unlike [`rel`], a few wrong entries cannot hide in the norm.
#[cfg(any(feature = "metal", feature = "cuda"))]
fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let num = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    let den = b.iter().map(|y| y.abs()).fold(0.0, f32::max);
    num / den.max(1e-30)
}

/// (model, autodiff model, device, autodiff device): one backend's side of [`replay`].
#[cfg(any(feature = "metal", feature = "cuda"))]
type Side<'a, const R: usize> = (&'a FNO<R>, &'a FNO<R>, &'a Device, &'a Device);

/// Replays `schedule` on an `FNO<R>` of width 32 and 4 layers with `modes`, once per
/// padding in `pads`: shapes change as in training, inputs are uploaded afresh per call,
/// and every GPU call is compared to Flex (forward output, and loss + input gradient
/// through backward). Prints one line per padding; returns the worst relative L² and the
/// worst max-entry error.
#[cfg(any(feature = "metal", feature = "cuda"))]
fn replay<const R: usize>(
    modes: &[usize],
    pads: &[usize],
    schedule: &[(Call, [usize; R])],
) -> (f32, f32) {
    use burn::module::Module;
    use sciml_rs::neural_operators::layers::spectral_convolution::SpectralInit;
    use sciml_rs::neural_operators::models::fno::FNOConfig;
    let (m, f) = (gpu(), Device::flex());
    let (fa, ma) = (f.clone().autodiff(), m.clone().autodiff());
    let (mut worst, mut worst_max) = (0.0f32, 0.0f32);
    for &p in pads {
        f.seed(5);
        let model_f = FNOConfig::new(modes.to_vec(), 1, 1)
            .with_hidden_channels(32)
            .with_n_layers(4)
            .with_padding(Some(p))
            .with_spectral_init(Some(SpectralInit::SymmetricUniform))
            .init::<R>(&f);
        let model_m = model_f.clone().to_device(&m);
        let (model_fa, model_ma) = (
            model_f.clone().to_device(&fa),
            model_f.clone().to_device(&ma),
        );
        let (flex, gpu_side): (Side<R>, Side<R>) = (
            (&model_f, &model_fa, &f, &fa),
            (&model_m, &model_ma, &m, &ma),
        );
        let run = |kind: Call, dims: [usize; R], (model, model_ad, plain, ad): Side<R>| match kind {
            Call::Eval => v(model.forward(wave(dims, plain))),
            Call::Train => loss_and_grad(|x| model_ad.forward(x), wave(dims, ad)),
        };
        // Flex references, once per (kind, shape).
        let mut refs = std::collections::HashMap::<(Call, [usize; R]), Vec<f32>>::new();
        let mut line = Vec::new();
        for (call, &(kind, dims)) in schedule.iter().enumerate() {
            let want = refs
                .entry((kind, dims))
                .or_insert_with(|| run(kind, dims, flex));
            let got = run(kind, dims, gpu_side);
            let (e, e_max) = (rel(&got, want), max_rel(&got, want));
            worst = worst.max(e);
            worst_max = worst_max.max(e_max);
            line.push(format!(
                "{call}:{kind}{dims:?}={e:.1e}/{e_max:.1e}{}",
                if e > 1e-5 { "!" } else { "" }
            ));
        }
        println!("p={p:<2} {}", line.join(" "));
    }
    println!("worst gpu/flex rel err: {worst:.2e}, max rel: {worst_max:.2e}");
    (worst, worst_max)
}

/// 2D replay check at the Poisson trainer's configuration (s = 64, modes 12²): train
/// batches 20, 20 and a partial 7, eval batches 20 and 3, each shape called repeatedly.
///
/// Default padding 0, 4, 8 and 16; `PADS=a,b,...` overrides.
#[cfg(any(feature = "metal", feature = "cuda"))]
#[test]
fn replay_2d_poisson() {
    let pads: Vec<usize> = std::env::var("PADS")
        .map(|s| s.split(',').map(|p| p.parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![0, 4, 8, 16]);
    let sq = |b: usize| [b, 64, 64, 1];
    let schedule = [
        (Call::Train, sq(20)),
        (Call::Train, sq(20)),
        (Call::Train, sq(7)),
        (Call::Eval, sq(20)),
        (Call::Eval, sq(3)),
        (Call::Train, sq(20)),
        (Call::Train, sq(7)),
        (Call::Eval, sq(20)),
    ];
    // L² only, the criterion on main; the max-entry error is printed.
    let (worst, _) = replay(&[12, 12], &pads, &schedule);
    assert!(worst <= 1e-5, "GPU differs from Flex: {worst:e}");
}

/// 3D replay check (CL0.2): [`replay_2d_poisson`]'s schedule at rank 5, s = 16 and
/// modes 6³, padding 0 and 4 (24³ goes through Bluestein), with an evaluation at the
/// non-cubic [2, 14, 13, 12, 1] in the middle (a resolution change). Asserts both the
/// relative L² and the max-entry error.
#[cfg(any(feature = "metal", feature = "cuda"))]
#[test]
fn replay_3d_poisson() {
    let cube = |b: usize| [b, 16, 16, 16, 1];
    let schedule = [
        (Call::Train, cube(20)),
        (Call::Train, cube(20)),
        (Call::Train, cube(7)),
        (Call::Eval, cube(20)),
        (Call::Eval, cube(3)),
        (Call::Eval, [2, 14, 13, 12, 1]),
        (Call::Train, cube(20)),
        (Call::Train, cube(7)),
        (Call::Eval, cube(20)),
    ];
    let (worst, worst_max) = replay(&[6, 6, 6], &[0, 4], &schedule);
    assert!(
        worst <= 1e-5 && worst_max <= 1e-5,
        "GPU differs from Flex at rank 5: rel {worst:e}, max rel {worst_max:e}"
    );
}

/// 3D grid channels (CL0.2): `FNO<5>` with no Fourier layers, so the output is the
/// lifting of (input, grid), the permutes and the projection only, evaluated on the GPU
/// and on Flex at three non-cubic shapes, the first repeated after the second (the 2D
/// cat-after-permute and fused-cat-at-new-shape bugs). Flex's rank-5 grid is pinned by
/// `grid_controls_3d`. Asserts both the relative L² and the max-entry error.
#[cfg(any(feature = "metal", feature = "cuda"))]
#[test]
fn grid_3d() {
    use burn::module::Module;
    use sciml_rs::neural_operators::models::fno::FNOConfig;
    let (m, f) = (gpu(), Device::flex());
    f.seed(5);
    let model_f = FNOConfig::new(vec![2, 2, 2], 1, 1)
        .with_hidden_channels(32)
        .with_n_layers(0)
        .init::<5>(&f);
    let model_m = model_f.clone().to_device(&m);
    let mut worst = 0.0f32;
    for (call, dims) in [[2, 6, 5, 4, 1], [3, 4, 7, 5, 1], [2, 6, 5, 4, 1]]
        .into_iter()
        .enumerate()
    {
        let want = v(model_f.forward(wave(dims, &f)));
        let got = v(model_m.forward(wave(dims, &m)));
        let (e, e_max) = (rel(&got, &want), max_rel(&got, &want));
        println!("{call}: {dims:?} rel {e:.1e}, max rel {e_max:.1e}");
        worst = worst.max(e).max(e_max);
    }
    assert!(worst <= 1e-5, "GPU grid differs from Flex: {worst:e}");
}

/// 3D spectral layer (CL0.2): `SpectralConv<5>` alone, forward and the input gradient
/// of mean(y²), on the GPU against Flex. [2, 3, 14, 13, 12] takes Bluestein on both
/// cfft axes (the 2D cfft stride bug); it is repeated after a power-of-two shape; the
/// last shape's rfft half-width is even (14 → 8), where the 2D cfft bug appeared.
/// Asserts both the relative L² and the max-entry error.
#[cfg(any(feature = "metal", feature = "cuda"))]
#[test]
fn spectral_conv_3d() {
    use burn::module::Module;
    use sciml_rs::neural_operators::layers::spectral_convolution::SpectralConv;
    let (m, f) = (gpu(), Device::flex());
    let (fa, ma) = (f.clone().autodiff(), m.clone().autodiff());
    f.seed(5);
    let conv_f = SpectralConv::<5>::new(&f, 3, 3, &[6, 6, 6]);
    let conv_m = conv_f.clone().to_device(&m);
    let (conv_fa, conv_ma) = (conv_f.clone().to_device(&fa), conv_f.clone().to_device(&ma));
    let mut worst = 0.0f32;
    let shapes = [
        [2, 3, 14, 13, 12],
        [2, 3, 16, 16, 16],
        [2, 3, 14, 13, 12],
        [2, 3, 13, 12, 14],
    ];
    for (call, dims) in shapes.into_iter().enumerate() {
        let (fwd_m, fwd_f) = (
            v(conv_m.forward(wave(dims, &m))),
            v(conv_f.forward(wave(dims, &f))),
        );
        let (bwd_m, bwd_f) = (
            loss_and_grad(|x| conv_ma.forward(x), wave(dims, &ma)),
            loss_and_grad(|x| conv_fa.forward(x), wave(dims, &fa)),
        );
        let e = [
            rel(&fwd_m, &fwd_f),
            max_rel(&fwd_m, &fwd_f),
            rel(&bwd_m, &bwd_f),
            max_rel(&bwd_m, &bwd_f),
        ];
        println!(
            "{call}: {dims:?} forward {:.1e}/{:.1e}, loss + grad {:.1e}/{:.1e} (rel/max rel)",
            e[0], e[1], e[2], e[3]
        );
        worst = e.into_iter().fold(worst, f32::max);
    }
    assert!(
        worst <= 1e-5,
        "GPU SpectralConv<5> differs from Flex: {worst:e}"
    );
}
