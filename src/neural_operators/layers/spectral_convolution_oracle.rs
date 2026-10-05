//! Test-only f64 oracle of [`SpectralConv::forward`].
//!
//! A plain-Rust separable DFT of the operator in design spectral-conv-perf §2.1, written
//! without Burn's FFTs. Today's layer is compared against it here, and the faster paths
//! of later phases reuse it unchanged. Transform order and normalisation follow
//! CONVENTIONS §4; the mode corners and the channel mix follow CONVENTIONS §5. Corner
//! indexing is derived here from the convention, not from `SpectralConv::corner_ranges`,
//! so that the two can disagree.
//!
//! Every test prints `case, rel_inf` lines; run with `--nocapture` to collect them.

use super::*; // SpectralConv, Param, Tensor, Device
use burn::tensor::TensorData; // not imported by the parent

use std::f64::consts::PI;

/// Batch size of every layer-vs-oracle case.
const BATCH: usize = 2;
/// Input channels; differs from [`OUT_CH`] so a transposed weight cannot pass.
const IN_CH: usize = 3;
/// Output channels.
const OUT_CH: usize = 2;
/// E-control threshold: a mutated oracle must differ from the layer by at least this,
/// relative to the unmutated oracle's ‖y‖∞ (phase README, "Error measures").
const CONTROL_MIN: f64 = 1e-3;

/// Fixed-seed host generator (SplitMix64), uniform in [-1, 1).
///
/// Used instead of `Tensor::random`: flex's RNG is process-wide and shared with
/// concurrently running tests.
struct HostRng(u64);

impl HostRng {
    /// Next raw 64-bit SplitMix64 output.
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1) on a grid of 2^-23, so every value is exact in f32.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

/// Row-major f64 array with its shape.
struct Array {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl Array {
    /// Exact f64 copy of f32 values, with `shape`.
    fn from_f32(values: &[f32], shape: &[usize]) -> Self {
        Self {
            data: values.iter().map(|&v| f64::from(v)).collect(),
            shape: shape.to_vec(),
        }
    }
}

/// Uploads host f32 values as a rank-`R` tensor of `shape`.
fn upload<const R: usize>(values: &[f32], shape: &[usize], device: &Device) -> Tensor<R> {
    Tensor::<R>::from_data(TensorData::new(values.to_vec(), shape.to_vec()), device)
}

/// Layer under test with deterministic weights. Returns the layer, plus every corner's
/// (re, im) weights as the f32 values actually uploaded, cast to f64,
/// each shaped [I, O, modes..].
///
/// `SpectralConv::new` draws random weights first; they are overwritten, never used.
/// The new values are uniform in [-1, 1) scaled by 1/√(I·O), drawn per corner, real
/// part then imaginary part.
fn layer_with_weights<const R: usize>(
    device: &Device,
    i: usize,
    o: usize,
    modes: &[usize],
    rng: &mut HostRng,
) -> (SpectralConv<R>, Vec<(Array, Array)>) {
    let mut conv = SpectralConv::<R>::new(device, i, o, modes);
    let mut shape = vec![i, o];
    shape.extend_from_slice(modes);
    let len: usize = shape.iter().product();
    let scale = 1.0 / ((i * o) as f32).sqrt();

    let mut weights = Vec::with_capacity(conv.weights_re.len());
    for (w_re, w_im) in conv.weights_re.iter_mut().zip(conv.weights_im.iter_mut()) {
        let re: Vec<f32> = (0..len).map(|_| rng.next_f32() * scale).collect();
        let im: Vec<f32> = (0..len).map(|_| rng.next_f32() * scale).collect();
        *w_re = Param::from_tensor(upload::<R>(&re, &shape, device));
        *w_im = Param::from_tensor(upload::<R>(&im, &shape, device));
        weights.push((Array::from_f32(&re, &shape), Array::from_f32(&im, &shape)));
    }
    (conv, weights)
}

/// What the oracle computes; the controls switch one rule off.
#[derive(Clone, Copy, Default)]
struct Variant {
    /// Exchange the weights of corners 0 and 1 (D ≥ 2 only).
    swap_corners: bool,
    /// Weight the retained Nyquist bin of the last-axis inverse by 2 instead of 1.
    nyquist_weight_two: bool,
    /// Let Im Y_0 contribute `Im Y_0 / n` to the last-axis inverse, as a backend that
    /// does not ignore it would.
    keep_dc_imag: bool,
}

/// Design §2.1 in f64:
/// - forward DFT per axis, unnormalised, in CONVENTIONS §4's axis order;
/// - keep the CONVENTIONS §5 corners, and mix each with `out[b,o,k] = Σ_i x[b,i,k] w[i,o,k]`;
/// - inverse complex DFT (1/n) on axes D−1..1;
/// - then `irfft` on the last axis, which ignores Im at DC and, for even n, at Nyquist.
///
/// x: [B, I, s_1..s_D] -> y: [B, O, s_1..s_D]. `weights[mask]` is corner `mask`'s
/// (re, im) pair, each [I, O, modes..].
fn oracle_forward(x: &Array, weights: &[(Array, Array)], modes: &[usize], v: Variant) -> Array {
    let d = modes.len();
    let (batch, in_ch) = (x.shape[0], x.shape[1]);
    let out_ch = weights[0].0.shape[1];
    let spatial = &x.shape[2..];
    let grid: usize = spatial.iter().product();
    let flat_modes: usize = modes.iter().product();

    // Checked here rather than through `check_modes_fit`, so the oracle stays
    // independent of the layer: overlapping corners would silently overwrite bins.
    assert_eq!(
        spatial.len(),
        d,
        "oracle: x has {} spatial axes",
        spatial.len()
    );
    for (j, (&m, &n)) in modes.iter().zip(spatial).enumerate() {
        let limit = if j == d - 1 { n / 2 + 1 } else { n / 2 };
        assert!(
            m <= limit,
            "oracle: modes[{j}] = {m} exceeds {limit} for n = {n}"
        );
    }
    let mut w_shape = vec![in_ch, out_ch];
    w_shape.extend_from_slice(modes);
    assert_eq!(
        weights.len(),
        1 << (d - 1),
        "oracle: one weight pair per corner"
    );
    for (re, im) in weights {
        assert!(
            re.shape == w_shape && im.shape == w_shape,
            "oracle: weights must be {w_shape:?}"
        );
    }

    // Forward: last spatial axis first, then axes R − 2 down to 2 (CONVENTIONS §4).
    // The full spectrum is kept on every axis, including the last; the corners below
    // only read bins 0..m_D ≤ s_D/2 + 1 of it, which is the rfft half-spectrum.
    let mut x_re = x.data.clone();
    let mut x_im = vec![0.0; x_re.len()];
    for axis in (2..2 + d).rev() {
        dft_axis(&mut x_re, &mut x_im, &x.shape, axis, false);
    }

    // Corner `mask`: bit j set selects the high block s_j − m_j..s_j on non-last axis
    // j; the last axis is always the low block 0..m_D (CONVENTIONS §5). Every other
    // output bin stays zero.
    let mut out_shape = vec![batch, out_ch];
    out_shape.extend_from_slice(spatial);
    let mut y_re = vec![0.0; batch * out_ch * grid];
    let mut y_im = vec![0.0; y_re.len()];
    for mask in 0..1usize << (d - 1) {
        let corner = if v.swap_corners && d >= 2 && mask < 2 {
            1 - mask
        } else {
            mask
        };
        let (w_re, w_im) = (&weights[corner].0.data, &weights[corner].1.data);
        let start: Vec<usize> = (0..d)
            .map(|j| {
                if j < d - 1 && (mask >> j) & 1 == 1 {
                    spatial[j] - modes[j]
                } else {
                    0
                }
            })
            .collect();

        for local in 0..flat_modes {
            // Unravel `local` over `modes` and ravel start + index over the grid.
            let (mut rem, mut k, mut stride) = (local, 0, 1);
            for j in (0..d).rev() {
                k += (start[j] + rem % modes[j]) * stride;
                rem /= modes[j];
                stride *= spatial[j];
            }
            for b in 0..batch {
                for o in 0..out_ch {
                    let (mut acc_re, mut acc_im) = (0.0, 0.0);
                    for i in 0..in_ch {
                        let xi = (b * in_ch + i) * grid + k;
                        let wi = (i * out_ch + o) * flat_modes + local;
                        acc_re += x_re[xi] * w_re[wi] - x_im[xi] * w_im[wi];
                        acc_im += x_re[xi] * w_im[wi] + x_im[xi] * w_re[wi];
                    }
                    let yi = (b * out_ch + o) * grid + k;
                    y_re[yi] = acc_re;
                    y_im[yi] = acc_im;
                }
            }
        }
    }

    // Inverse complex DFT on the non-last spatial axes, D − 1 down to 1.
    for axis in (2..1 + d).rev() {
        dft_axis(&mut y_re, &mut y_im, &out_shape, axis, true);
    }
    let data = irfft_last_axis(&y_re, &y_im, spatial[d - 1], v);
    Array {
        data,
        shape: out_shape,
    }
}

/// cos and sin of 2π·j/n for j in 0..n. Phases are looked up at `(k·x) mod n`, so the
/// argument never grows with k·x.
fn twiddles(n: usize) -> (Vec<f64>, Vec<f64>) {
    (0..n)
        .map(|j| {
            let theta = 2.0 * PI * j as f64 / n as f64;
            (theta.cos(), theta.sin())
        })
        .unzip()
}

/// Full complex DFT along `axis` of the row-major array `(re, im)` of `shape`, in place.
///
/// Forward: `X[k] = Σ_x x[x] e^{−2πi k x / n}`, unscaled. Inverse: `e^{+2πi k x / n}`,
/// scaled by 1/n (NumPy `norm="backward"`, CONVENTIONS §4).
fn dft_axis(re: &mut [f64], im: &mut [f64], shape: &[usize], axis: usize, inverse: bool) {
    let n = shape[axis];
    let stride: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    let (cos, sin) = twiddles(n);
    let (sign, scale) = if inverse {
        (1.0, 1.0 / n as f64)
    } else {
        (-1.0, 1.0)
    };

    let mut line_re = vec![0.0; n];
    let mut line_im = vec![0.0; n];
    for block in 0..outer {
        for t in 0..stride {
            let base = block * n * stride + t;
            for (x, (lr, li)) in line_re.iter_mut().zip(line_im.iter_mut()).enumerate() {
                *lr = re[base + x * stride];
                *li = im[base + x * stride];
            }
            for k in 0..n {
                let (mut acc_re, mut acc_im) = (0.0, 0.0);
                for (x, (&a, &b)) in line_re.iter().zip(&line_im).enumerate() {
                    let p = (k * x) % n;
                    let (c, s) = (cos[p], sign * sin[p]);
                    // (a + bi)(c + si)
                    acc_re += a * c - b * s;
                    acc_im += a * s + b * c;
                }
                re[base + k * stride] = acc_re * scale;
                im[base + k * stride] = acc_im * scale;
            }
        }
    }
}

/// `irfft` of length n along the last axis (design §2.1), reading bins 0..=n/2 of each
/// full-spectrum line of `(re, im)`:
///
/// `y[x] = (1/n)[Re Y_0 + Σ_{0<k<n/2} 2(Re Y_k cos θ − Im Y_k sin θ) + [n even] Re Y_{n/2} cos θ]`,
/// θ = 2π·((k·x) mod n)/n. Im Y_0 and, for even n, Im Y_{n/2} are ignored unless `v`
/// switches that rule off.
fn irfft_last_axis(re: &[f64], im: &[f64], n: usize, v: Variant) -> Vec<f64> {
    let (cos, sin) = twiddles(n);
    let nyquist_weight = if v.nyquist_weight_two { 2.0 } else { 1.0 };
    let mut out = vec![0.0; re.len()];
    for ((out_line, re_line), im_line) in out.chunks_mut(n).zip(re.chunks(n)).zip(im.chunks(n)) {
        for (x, y) in out_line.iter_mut().enumerate() {
            let mut acc = re_line[0];
            if v.keep_dc_imag {
                acc += im_line[0];
            }
            for (k, (&yr, &yi)) in re_line
                .iter()
                .zip(im_line)
                .enumerate()
                .take(n / 2 + 1)
                .skip(1)
            {
                let p = (k * x) % n;
                if 2 * k == n {
                    acc += nyquist_weight * yr * cos[p];
                } else {
                    acc += 2.0 * (yr * cos[p] - yi * sin[p]);
                }
            }
            *y = acc / n as f64;
        }
    }
    out
}

/// Largest absolute value, ‖a‖∞.
fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |m, v| m.max(v.abs()))
}

/// ‖a − b‖∞, the largest absolute difference.
fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

/// ‖a − b‖∞ / ‖b‖∞.
fn rel_inf(a: &[f64], b: &[f64]) -> f64 {
    max_abs_diff(a, b) / max_abs(b)
}

/// One case's layer output and the inputs the oracle needs.
struct Case {
    /// [B, I, spatial..], the values uploaded to the layer.
    x: Array,
    /// Per corner (re, im), each [I, O, modes..], the values set on the layer.
    weights: Vec<(Array, Array)>,
    /// The layer's f32 output [B, O, spatial..], cast to f64.
    y_layer: Vec<f64>,
}

/// Runs `SpectralConv<R>` on `Device::default()` with B = 2, I = 3, O = 2 and inputs
/// and weights drawn from `HostRng(seed)`.
fn run_layer<const R: usize>(spatial: &[usize], modes: &[usize], seed: u64) -> Case {
    let device = Device::default();
    let mut rng = HostRng(seed);
    let (conv, weights) = layer_with_weights::<R>(&device, IN_CH, OUT_CH, modes, &mut rng);

    let mut shape = vec![BATCH, IN_CH];
    shape.extend_from_slice(spatial);
    let len: usize = shape.iter().product();
    let x: Vec<f32> = (0..len).map(|_| rng.next_f32()).collect();

    let y_layer = conv
        .forward(upload::<R>(&x, &shape, &device))
        .into_data()
        .try_to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect();
    Case {
        x: Array::from_f32(&x, &shape),
        weights,
        y_layer,
    }
}

/// Printed label of a case.
fn label(spatial: &[usize], modes: &[usize]) -> String {
    format!("{}d s={spatial:?} modes={modes:?}", modes.len())
}

/// One E-oracle case: prints `case, rel_inf` and records a failure rather than
/// panicking, so every case of a test is printed even when one fails.
fn check_oracle<const R: usize>(
    spatial: &[usize],
    modes: &[usize],
    seed: u64,
    tol: f64,
    failures: &mut Vec<String>,
) {
    let case = run_layer::<R>(spatial, modes, seed);
    let y = oracle_forward(&case.x, &case.weights, modes, Variant::default());
    let err = rel_inf(&case.y_layer, &y.data);
    let name = label(spatial, modes);
    println!("{name}, {err:.3e}");
    if err.is_nan() || err > tol {
        failures.push(format!("{name}: E-oracle {err:.3e} > {tol:.0e}"));
    }
}

/// One E-control case: the oracle mutated by `v` must miss the layer by at least
/// [`CONTROL_MIN`] relative to the unmutated oracle's ‖y‖∞.
fn check_control<const R: usize>(
    spatial: &[usize],
    modes: &[usize],
    seed: u64,
    mutation: &str,
    v: Variant,
    failures: &mut Vec<String>,
) {
    let case = run_layer::<R>(spatial, modes, seed);
    let y_ref = oracle_forward(&case.x, &case.weights, modes, Variant::default());
    let y_wrong = oracle_forward(&case.x, &case.weights, modes, v);
    let err = max_abs_diff(&case.y_layer, &y_wrong.data) / max_abs(&y_ref.data);
    let name = format!("control {mutation} {}", label(spatial, modes));
    println!("{name}, {err:.3e}");
    if err.is_nan() || err < CONTROL_MIN {
        failures.push(format!("{name}: E-control {err:.3e} < {CONTROL_MIN:.0e}"));
    }
}

/// Panics listing every failed case, after all cases have printed.
fn assert_no_failures(failures: &[String]) {
    assert!(
        failures.is_empty(),
        "failed cases:\n{}",
        failures.join("\n")
    );
}

#[test]
fn oracle_1d() {
    let mut failures = Vec::new();
    let cases: [(usize, usize, f64); 6] = [
        (16, 9, 1e-5),    // retained Nyquist (9 = 16/2 + 1), power of two
        (15, 8, 1e-5),    // odd, at the last-axis limit, Bluestein
        (12, 7, 1e-5),    // even non-power-of-two with retained Nyquist
        (94, 16, 1e-5),   // Darcy padded extent, Bluestein
        (256, 16, 3e-5),  // Burgers example
        (1024, 16, 3e-5), // Burgers at r = 8
    ];
    for (seed, (s, m, tol)) in (100..).zip(cases) {
        check_oracle::<3>(&[s], &[m], seed, tol, &mut failures);
    }
    assert_no_failures(&failures);
}

#[test]
fn oracle_2d() {
    let mut failures = Vec::new();
    let cases: [([usize; 2], [usize; 2]); 3] = [
        ([94, 94], [12, 12]), // Darcy padded grid
        ([16, 16], [8, 9]),   // non-last axis at 16/2, retained Nyquist on the last
        ([15, 17], [7, 9]),   // both odd, both at their limits
    ];
    for (seed, (s, m)) in (200..).zip(cases) {
        check_oracle::<4>(&s, &m, seed, 1e-5, &mut failures);
    }
    assert_no_failures(&failures);
}

#[test]
fn oracle_3d() {
    let mut failures = Vec::new();
    check_oracle::<5>(&[8, 6, 10], &[2, 3, 4], 300, 1e-5, &mut failures);
    assert_no_failures(&failures);
}

#[test]
fn oracle_controls() {
    let mut failures = Vec::new();
    let swap = Variant {
        swap_corners: true,
        ..Variant::default()
    };
    let nyquist = Variant {
        nyquist_weight_two: true,
        ..Variant::default()
    };
    let dc = Variant {
        keep_dc_imag: true,
        ..Variant::default()
    };
    check_control::<4>(&[16, 16], &[8, 9], 400, "swap_corners", swap, &mut failures);
    check_control::<3>(
        &[16],
        &[9],
        401,
        "nyquist_weight_two",
        nyquist,
        &mut failures,
    );
    check_control::<3>(
        &[12],
        &[7],
        402,
        "nyquist_weight_two",
        nyquist,
        &mut failures,
    );
    check_control::<4>(
        &[16, 16],
        &[8, 9],
        403,
        "nyquist_weight_two",
        nyquist,
        &mut failures,
    );
    check_control::<3>(&[94], &[16], 404, "keep_dc_imag", dc, &mut failures);
    assert_no_failures(&failures);
}

#[test]
fn oracle_reproduces_pinned_reference() {
    // Input and weights of `tests::forward_2d_matches_pytorch_reference`, whose
    // expected values are pinned against PyTorch's SpectralConv2d.
    let x = Array {
        data: (1..=16).map(f64::from).collect(),
        shape: vec![1, 1, 4, 4],
    };
    let w = |v: f32| Array::from_f32(&[v], &[1, 1, 1, 1]);
    let weights = [(w(0.1), w(0.2)), (w(0.3), w(0.4))];

    let y = oracle_forward(&x, &weights, &[1, 1], Variant::default());

    let expected = [
        1.05, 1.05, 1.05, 1.05, -0.55, -0.55, -0.55, -0.55, 0.65, 0.65, 0.65, 0.65, 2.25, 2.25,
        2.25, 2.25,
    ];
    let err = max_abs_diff(&y.data, &expected);
    println!("pinned 2d s=[4, 4] modes=[1, 1] (absolute), {err:.3e}");
    assert!(
        err <= 1e-4,
        "oracle misses the pinned reference by {err:.3e}"
    );
}
