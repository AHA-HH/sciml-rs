//! Spectral convolution, generic over spatial dimensionality.
//!
//! The core operation of a Fourier Neural Operator: transform to the frequency
//! domain, apply a learned linear map to the channel axis at each retained
//! frequency, and transform back. Because the map is applied per-frequency
//! rather than per-grid-point, the layer is a global convolution and is
//! independent of the input discretisation - the same weights apply at any
//! resolution.
//!
//! # Rank
//!
//! `R` is the total tensor rank: `R = D + 2` for `D` spatial axes, plus the
//! batch and channel axes. Burn encodes rank in the tensor *type*, so it must
//! be a compile-time constant. Mode counts, by contrast, are ordinary runtime
//! data (`modes: Vec<usize>`); `new` asserts the two agree. `R >= 3` (at least
//! one spatial axis) is enforced at compile time.

use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    tensor::signal,
    tensor::{Device, Distribution},
};

use std::ops::Range;

use crate::neural_operators::utils::fft::icfft_full_spectrum;

/// Initialisation of the complex spectral weights of [`SpectralConv`].
///
/// Every weight is `w = a + bi`, with the real part `a` and imaginary part `b`
/// drawn independently from the same real distribution. `I` is
/// `in_channels` and `O` is `out_channels`. Means and variances below are per
/// part.
#[derive(Config, Debug, PartialEq)]
pub enum SpectralInit {
    /// `U(0, s)` with `s = 1/(I·O)`: mean `s/2`, variance `s²/12`.
    ///
    /// The reference initialisation of Li et al. (`scale * torch.rand(cfloat)`)
    /// and the default. Every part is non-negative, so the mean phase is π/4,
    /// and the scale shrinks with the square of the width.
    LiUniform,
    /// `N(0, σ²)` with variance `σ² = 1/(I·O)` (standard deviation
    /// `1/√(I·O)`): mean 0.
    Normal,
    /// `U(−1/√I, 1/√I)`: mean 0, variance `1/(3I)`. Depends on `in_channels`
    /// only.
    SymmetricUniform,
}

// Manual: `#[derive(Default)]` needs a `#[default]` variant attribute, which
// `#[derive(Config)]` copies into its generated code, where it does not resolve.
#[allow(clippy::derivable_impls)]
impl Default for SpectralInit {
    fn default() -> Self {
        Self::LiUniform
    }
}

impl SpectralInit {
    /// The per-part distribution for an `in_channels × out_channels` layer.
    fn distribution(&self, in_channels: usize, out_channels: usize) -> Distribution {
        match self {
            Self::LiUniform => {
                let scale = 1.0 / (in_channels * out_channels) as f64;
                Distribution::Uniform(0.0, scale)
            }
            // Burn's `Normal` takes the standard deviation, not the variance.
            Self::Normal => {
                let std = (1.0 / (in_channels * out_channels) as f64).sqrt();
                Distribution::Normal(0.0, std)
            }
            Self::SymmetricUniform => {
                let bound = 1.0 / (in_channels as f64).sqrt();
                Distribution::Uniform(-bound, bound)
            }
        }
    }
}

/// Learned spectral convolution over `R - 2` spatial axes.
#[derive(Module, Debug)]
pub struct SpectralConv<const R: usize> {
    /// One weight per mode corner; `len() == 2^(D-1)`.
    /// Each is `[in_channels, out_channels, modes[0], .., modes[D-1]]`.
    weights_re: Vec<Param<Tensor<R>>>,
    weights_im: Vec<Param<Tensor<R>>>,
    /// Retained frequency count per spatial axis; `len() == R - 2`.
    modes: Vec<usize>,
    flat_modes: usize,
}

impl<const R: usize> SpectralConv<R> {
    /// Compile-time guard: at least one spatial axis. Evaluated in `new` and
    /// `forward`, so `SpectralConv<R>` (and `FNO<R>`) with `R < 3` fails to
    /// build instead of underflowing `R - 2` / `modes.len() - 1` at runtime.
    const RANK_OK: () = assert!(
        R >= 3,
        "SpectralConv<R> needs at least one spatial axis (R >= 3)"
    );

    /// `R` must be at least 3 (one or more spatial axes):
    ///
    /// ```
    /// use burn::tensor::Device;
    /// use sciml_rs::neural_operators::layers::spectral_convolution::SpectralConv;
    /// use sciml_rs::neural_operators::models::fno::FNOConfig;
    ///
    /// let device = Device::default();
    /// let _conv = SpectralConv::<3>::new(&device, 1, 1, &[4]);
    /// let _model = FNOConfig::new(vec![4], 1, 1).init::<3>(&device);
    /// ```
    ///
    /// Smaller ranks are rejected at compile time, including through `FNO`:
    ///
    /// ```compile_fail,E0080
    /// use burn::tensor::Device;
    /// use sciml_rs::neural_operators::layers::spectral_convolution::SpectralConv;
    ///
    /// let device = Device::default();
    /// let _conv = SpectralConv::<2>::new(&device, 1, 1, &[]);
    /// ```
    ///
    /// ```compile_fail,E0080
    /// use burn::tensor::Device;
    /// use sciml_rs::neural_operators::models::fno::FNOConfig;
    ///
    /// let device = Device::default();
    /// let _model = FNOConfig::new(vec![], 1, 1).init::<2>(&device);
    /// ```
    ///
    /// Weights use the Li et al. default, [`SpectralInit::LiUniform`]; see
    /// [`Self::new_with_init`] to choose another scheme.
    ///
    /// # Panics
    ///
    /// If `modes.len() != R - 2`.
    pub fn new(device: &Device, in_channels: usize, out_channels: usize, modes: &[usize]) -> Self {
        Self::new_with_init(
            device,
            in_channels,
            out_channels,
            modes,
            SpectralInit::LiUniform,
        )
    }

    /// As [`Self::new`], with the weights drawn according to `init`.
    ///
    /// ```
    /// use burn::tensor::Device;
    /// use sciml_rs::neural_operators::layers::spectral_convolution::{SpectralConv, SpectralInit};
    ///
    /// let device = Device::default();
    /// let _conv = SpectralConv::<3>::new_with_init(&device, 64, 64, &[16], SpectralInit::Normal);
    /// ```
    ///
    /// # Panics
    ///
    /// If `modes.len() != R - 2`.
    pub fn new_with_init(
        device: &Device,
        in_channels: usize,
        out_channels: usize,
        modes: &[usize],
        init: SpectralInit,
    ) -> Self {
        let () = Self::RANK_OK;
        assert_eq!(
            modes.len(),
            R - 2,
            "modes must have one entry per spatial axis (R - 2)"
        );
        let modes = modes.to_vec();
        let flat_modes = modes.iter().product();
        let num_corners = 1usize << (modes.len() - 1);

        let mut shape = vec![in_channels, out_channels];
        shape.extend_from_slice(&modes);
        let shape: [usize; R] = shape.try_into().unwrap();

        // Draw order (per corner: real, then imaginary) is part of the
        // contract: with a fixed seed, `LiUniform` reproduces the weights of
        // earlier releases bit for bit.
        let distribution = init.distribution(in_channels, out_channels);
        let mut weights_re = Vec::with_capacity(num_corners);
        let mut weights_im = Vec::with_capacity(num_corners);
        for _ in 0..num_corners {
            weights_re.push(Param::from_tensor(Tensor::<R>::random(
                shape,
                distribution,
                device,
            )));
            weights_im.push(Param::from_tensor(Tensor::<R>::random(
                shape,
                distribution,
                device,
            )));
        }

        Self {
            weights_re,
            weights_im,
            modes,
            flat_modes,
        }
    }

    /// Spatial-axis slices selecting one mode corner.
    ///
    /// Bit `j` of `mask` picks the low (`0`) or high (`1`) frequency block on
    /// axis `j`. The last axis is always low: the real FFT leaves it with only
    /// non-negative frequencies, so it has no mirror to select. This is why
    /// `mask` spans `2^(D-1)` values rather than `2^D`.
    fn corner_ranges(mask: usize, modes: &[usize], full_dims: &[usize]) -> Vec<Range<usize>> {
        let d = modes.len();
        (0..d)
            .map(|axis| {
                if axis == d - 1 || (mask >> axis) & 1 == 0 {
                    0..modes[axis]
                } else {
                    full_dims[axis] - modes[axis]..full_dims[axis]
                }
            })
            .collect()
    }

    /// Checks that every mode count fits its spatial extent `n`.
    ///
    /// Non-last axes keep a low block `0..m` and a high block `n-m..n`, which
    /// are disjoint only if `m <= n/2`. The last axis keeps `0..m` of the
    /// `n/2 + 1` bins the real FFT produces.
    fn check_modes_fit(modes: &[usize], spatial: &[usize]) {
        let last = modes.len() - 1;
        for (axis, (&m, &n)) in modes.iter().zip(spatial).enumerate() {
            let (limit, reason) = if axis == last {
                (n / 2 + 1, "the rfft axis has n/2 + 1 frequency bins")
            } else {
                (n / 2, "low and high mode blocks would overlap")
            };
            assert!(
                m <= limit,
                "SpectralConv: modes[{axis}] = {m} exceeds {limit} for spatial extent {n} ({reason})"
            );
        }
    }

    /// Complex channel mixing at every retained frequency.
    ///
    /// `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`, batched over modes.
    ///
    /// The mode axes are flattened to a single axis `M` so the contraction is a
    /// plain batched matmul `[M, B, I] @ [M, I, O]`. This keeps every
    /// intermediate at literal rank 3 - the broadcast-and-sum alternative would
    /// need rank `R + 1`, which stable Rust cannot express as a type.
    fn complex_multiplication(
        x_re: Tensor<R>,
        x_im: Tensor<R>, // [B, I, modes..]
        w_re: Tensor<R>,
        w_im: Tensor<R>, // [I, O, modes..]
        modes: &[usize],
        flat: usize,
    ) -> (Tensor<R>, Tensor<R>) {
        let (b, i) = (x_re.dims()[0], x_re.dims()[1]);
        let o = w_re.dims()[1];

        let x_re = x_re.reshape([b, i, flat]).permute([2, 0, 1]);
        let x_im = x_im.reshape([b, i, flat]).permute([2, 0, 1]);
        let w_re = w_re.reshape([i, o, flat]).permute([2, 0, 1]);
        let w_im = w_im.reshape([i, o, flat]).permute([2, 0, 1]);

        let ac = x_re.clone().matmul(w_re.clone());
        let bd = x_im.clone().matmul(w_im.clone());

        let ab_cd = (x_re + x_im).matmul(w_re + w_im);

        let out_re = ac.clone() - bd.clone();
        let out_im = ab_cd - ac - bd;

        let mut out_shape = vec![b, o];
        out_shape.extend_from_slice(modes);
        let out_shape: [usize; R] = out_shape.try_into().unwrap();

        (
            out_re.permute([1, 2, 0]).reshape(out_shape),
            out_im.permute([1, 2, 0]).reshape(out_shape),
        )
    }

    /// Forward N-d transform: real FFT on the last spatial axis, complex FFT on
    /// the rest. Halves the storage and the corner count versus a full complex
    /// transform, since a real signal's spectrum is Hermitian.
    fn fft_ctensor(x: Tensor<R>) -> (Tensor<R>, Tensor<R>) {
        let dims = x.dims(); // read before x is consumed
        let last = R - 1;
        let (mut re, mut im) = signal::rfft(x, last, Some(dims[last]));
        for axis in (2..last).rev() {
            let (r, i) = signal::cfft(re, im, axis, Some(dims[axis]));
            re = r;
            im = i;
        }
        (re, im)
    }

    /// Inverse of [`Self::fft_ctensor`], applying the axes in the reverse order.
    fn ifft_ctensor(mut re: Tensor<R>, mut im: Tensor<R>, orig_dims: &[usize]) -> Tensor<R> {
        let last = R - 1;
        for (axis, &dim) in orig_dims.iter().enumerate().take(last).skip(2) {
            let (r, i) = icfft_full_spectrum(re, im, axis, dim);
            re = r;
            im = i;
        }
        signal::irfft(re, im, last, Some(orig_dims[last]))
    }

    /// `[B, in_channels, spatial..] -> [B, out_channels, spatial..]`.
    ///
    /// Spatial extents are preserved; the channel count changes.
    ///
    /// # Panics
    ///
    /// If a mode count does not fit the input grid: `modes[i] > n_i / 2` on a
    /// non-last spatial axis, or `modes[last] > n_last / 2 + 1`.
    pub fn forward(&self, x: Tensor<R>) -> Tensor<R> {
        // Captured before the transform: irfft needs the original extent to
        // recover the correct length from the truncated half-spectrum.
        let () = Self::RANK_OK;
        let orig_dims = x.dims();
        Self::check_modes_fit(&self.modes, &orig_dims[2..]);
        let (batch, in_ch) = (orig_dims[0], orig_dims[1]);
        let out_ch = self.weights_re[0].val().dims()[1];
        let num_corners = self.weights_re.len();

        let (x_ft_re, x_ft_im) = Self::fft_ctensor(x);
        // Post-transform extents - the last axis is now n/2 + 1.
        let spec_dims: Vec<usize> = x_ft_re.dims()[2..].to_vec();

        let mut out_shape = vec![batch, out_ch];
        out_shape.extend_from_slice(&spec_dims);
        let out_shape: [usize; R] = out_shape.try_into().unwrap();

        // Frequencies outside the retained modes stay zero - this is the
        // low-pass truncation that makes the operator resolution-independent.
        let mut out_ft_re = Tensor::<R>::zeros(out_shape, &x_ft_re.device());
        let mut out_ft_im = Tensor::<R>::zeros(out_shape, &x_ft_im.device());

        for corner in 0..num_corners {
            let ranges = Self::corner_ranges(corner, &self.modes, &spec_dims);

            let mut in_ranges = vec![0..batch, 0..in_ch];
            in_ranges.extend(ranges.clone());
            let in_ranges: [Range<usize>; R] = in_ranges.try_into().unwrap();

            let x_re = x_ft_re.clone().slice(in_ranges.clone());
            let x_im = x_ft_im.clone().slice(in_ranges);

            let (o_re, o_im) = Self::complex_multiplication(
                x_re,
                x_im,
                self.weights_re[corner].val(),
                self.weights_im[corner].val(),
                &self.modes,
                self.flat_modes,
            );

            let mut out_ranges = vec![0..batch, 0..out_ch];
            out_ranges.extend(ranges);
            let out_ranges: [Range<usize>; R] = out_ranges.try_into().unwrap();

            out_ft_re = out_ft_re.slice_assign(out_ranges.clone(), o_re);
            out_ft_im = out_ft_im.slice_assign(out_ranges, o_im);
        }

        Self::ifft_ctensor(out_ft_re, out_ft_im, &orig_dims)
    }

    /// `(re, im)` weight values per mode corner, for tests outside this module.
    #[cfg(test)]
    pub(crate) fn corner_weights(&self) -> Vec<(Tensor<R>, Tensor<R>)> {
        self.weights_re
            .iter()
            .zip(&self.weights_im)
            .map(|(re, im)| (re.val(), im.val()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    #[test]
    fn corner_ranges_2d_matches_manual_indices() {
        let modes = vec![4, 3];
        let full_dims = vec![16, 9];

        // Last axis is always the low block — rfft leaves it non-negative only.
        assert_eq!(
            SpectralConv::<4>::corner_ranges(0, &modes, &full_dims),
            vec![0..4, 0..3],
        );
        assert_eq!(
            SpectralConv::<4>::corner_ranges(1, &modes, &full_dims),
            vec![12..16, 0..3],
        );
    }

    #[test]
    fn corner_ranges_3d_covers_all_four_combinations() {
        let modes = vec![4, 3, 2];
        let full_dims = vec![16, 12, 5];
        let expected = [
            vec![0..4, 0..3, 0..2],
            vec![12..16, 0..3, 0..2],
            vec![0..4, 9..12, 0..2],
            vec![12..16, 9..12, 0..2],
        ];
        for (corner, expected) in expected.iter().enumerate() {
            assert_eq!(
                &SpectralConv::<5>::corner_ranges(corner, &modes, &full_dims),
                expected,
                "corner {corner} mismatch",
            );
        }
    }

    #[test]
    fn complex_multiplication_matches_python_einsum() {
        let device = Device::default();

        let x_re = Tensor::<3>::from_data([[[1.0f32, 2.0], [3.0, 4.0]]], &device);
        let x_im = Tensor::<3>::from_data([[[0.5f32, 1.5], [2.5, 3.5]]], &device);
        let w_re = Tensor::<3>::from_data(
            [[[1.0f32, 0.5], [0.5, 1.0]], [[0.5, 1.5], [1.5, 0.5]]],
            &device,
        );
        let w_im = Tensor::<3>::from_data(
            [[[0.1f32, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]],
            &device,
        );

        let (out_re, out_im) =
            SpectralConv::<3>::complex_multiplication(x_re, x_im, w_re, w_im, &[2], 2);

        let re_vec = out_re.into_data().try_to_vec::<f32>().unwrap();
        let im_vec = out_im.into_data().try_to_vec::<f32>().unwrap();

        let expected_re = [1.2f32, 4.6, 3.1, 0.6];
        let expected_im = [3.35f32, 8.8, 6.4, 7.25];

        for (i, (r, e)) in re_vec.iter().zip(expected_re.iter()).enumerate() {
            assert!((r - e).abs() < 1e-4, "re[{i}]: got {r}, expected {e}");
        }
        for (i, (r, e)) in im_vec.iter().zip(expected_im.iter()).enumerate() {
            assert!((r - e).abs() < 1e-4, "im[{i}]: got {r}, expected {e}");
        }
    }

    fn round_trip_check<const R: usize>(shape: [usize; R]) {
        let device = Device::default();
        let x = Tensor::<R>::random(shape, Distribution::Normal(0.0, 1.0), &device);
        let orig_dims = x.dims();

        let (re, im) = SpectralConv::<R>::fft_ctensor(x.clone());
        let recon = SpectralConv::<R>::ifft_ctensor(re, im, &orig_dims);

        let x_vec = x.into_data().try_to_vec::<f32>().unwrap();
        let recon_vec = recon.into_data().try_to_vec::<f32>().unwrap();

        let max_diff = x_vec
            .iter()
            .zip(recon_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        assert!(max_diff < 1e-4, "round-trip diverged: max_diff={max_diff}");
    }

    #[test]
    fn fft_round_trip_1d() {
        round_trip_check::<3>([2, 4, 16]);
    }

    #[test]
    fn fft_round_trip_2d() {
        round_trip_check::<4>([2, 4, 16, 16]);
    }

    #[test]
    fn fft_round_trip_3d() {
        round_trip_check::<5>([2, 4, 8, 8, 8]);
    }

    #[test]
    fn forward_2d_matches_pytorch_reference() {
        let device = Device::default();
        let modes = vec![1, 1];

        // corner 0 = low-H (Python's weights1), corner 1 = high-H (weights2).
        let mut conv = SpectralConv::<4>::new(&device, 1, 1, &modes);
        conv.weights_re[0] = Param::from_tensor(Tensor::<4>::from_data([[[[0.1]]]], &device));
        conv.weights_im[0] = Param::from_tensor(Tensor::<4>::from_data([[[[0.2]]]], &device));
        conv.weights_re[1] = Param::from_tensor(Tensor::<4>::from_data([[[[0.3]]]], &device));
        conv.weights_im[1] = Param::from_tensor(Tensor::<4>::from_data([[[[0.4]]]], &device));

        let x = Tensor::<4>::from_data(
            [[[
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]]],
            &device,
        );

        let out_vals = conv.forward(x).into_data().try_to_vec::<f32>().unwrap();

        // Pinned against SpectralConv2d.forward, weights1 = 0.1+0.2i, weights2 = 0.3+0.4i.
        let expected = [
            1.05, 1.05, 1.05, 1.05, -0.55, -0.55, -0.55, -0.55, 0.65, 0.65, 0.65, 0.65, 2.25, 2.25,
            2.25, 2.25,
        ];

        for (i, (a, e)) in out_vals.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-4, "mismatch at index {i}: {a} vs {e}");
        }
    }

    #[test]
    fn forward_3d_matches_numpy_reference() {
        let device = Device::default();
        let modes = vec![1, 1, 1];

        // corner 0 = (low H1, low H2), 1 = (high H1, low H2),
        // 2 = (low H1, high H2), 3 = (high, high).
        let mut conv = SpectralConv::<5>::new(&device, 1, 1, &modes);
        let pairs = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.7, 0.8)];
        for (i, (re, im)) in pairs.iter().enumerate() {
            conv.weights_re[i] = Param::from_tensor(Tensor::<5>::from_data([[[[[*re]]]]], &device));
            conv.weights_im[i] = Param::from_tensor(Tensor::<5>::from_data([[[[[*im]]]]], &device));
        }

        let x = Tensor::<5>::from_data(
            [[[
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ],
                [
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0],
                    [29.0, 30.0, 31.0, 32.0],
                ],
                [
                    [33.0, 34.0, 35.0, 36.0],
                    [37.0, 38.0, 39.0, 40.0],
                    [41.0, 42.0, 43.0, 44.0],
                    [45.0, 46.0, 47.0, 48.0],
                ],
                [
                    [49.0, 50.0, 51.0, 52.0],
                    [53.0, 54.0, 55.0, 56.0],
                    [57.0, 58.0, 59.0, 60.0],
                    [61.0, 62.0, 63.0, 64.0],
                ],
            ]]],
            &device,
        );

        let out_vals = conv.forward(x).into_data().try_to_vec::<f32>().unwrap();

        let expected = [
            4.25, 4.25, 4.25, 4.25, 1.85, 1.85, 1.85, 1.85, 3.85, 3.85, 3.85, 3.85, 6.25, 6.25,
            6.25, 6.25, -2.15, -2.15, -2.15, -2.15, -4.55, -4.55, -4.55, -4.55, -2.55, -2.55,
            -2.55, -2.55, -0.15, -0.15, -0.15, -0.15, 2.65, 2.65, 2.65, 2.65, 0.25, 0.25, 0.25,
            0.25, 2.25, 2.25, 2.25, 2.25, 4.65, 4.65, 4.65, 4.65, 9.05, 9.05, 9.05, 9.05, 6.65,
            6.65, 6.65, 6.65, 8.65, 8.65, 8.65, 8.65, 11.05, 11.05, 11.05, 11.05,
        ];

        for (i, (a, e)) in out_vals.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-4, "mismatch at index {i}: {a} vs {e}");
        }
    }

    fn gradient_flow_check<const R: usize>(modes: Vec<usize>, x_shape: [usize; R]) {
        let device = Device::default().autodiff();
        let channels = x_shape[1];

        let conv = SpectralConv::<R>::new(&device, channels, channels, &modes);
        let x =
            Tensor::<R>::random(x_shape, Distribution::Normal(0.0, 1.0), &device).require_grad();

        let out = conv.forward(x.clone());
        let grads = out.powf_scalar(2.0).sum().backward();

        assert!(
            x.grad(&grads).is_some(),
            "gradients did not flow through SpectralConv<{R}>"
        );

        for (idx, w) in conv.weights_re.iter().enumerate() {
            let g = w
                .grad(&grads)
                .unwrap_or_else(|| panic!("weights_re[{idx}] got no gradient"));
            let sum: f32 = g.abs().sum().into_scalar();
            assert!(sum > 1e-9, "weights_re[{idx}] gradient is effectively zero");
        }
        for (idx, w) in conv.weights_im.iter().enumerate() {
            let g = w
                .grad(&grads)
                .unwrap_or_else(|| panic!("weights_im[{idx}] got no gradient"));
            let sum: f32 = g.abs().sum().into_scalar();
            assert!(sum > 1e-9, "weights_im[{idx}] gradient is effectively zero");
        }
    }

    #[test]
    fn gradients_flow_1d() {
        gradient_flow_check::<3>(vec![4], [2, 4, 16]);
    }

    #[test]
    fn gradients_flow_2d() {
        gradient_flow_check::<4>(vec![2, 2], [1, 1, 4, 4]);
    }

    #[test]
    fn gradients_flow_3d() {
        gradient_flow_check::<5>(vec![2, 2, 2], [1, 1, 4, 4, 4]);
    }

    // --- regression: modes must fit the grid (REVIEW.md 2.2) ---

    fn modes_fit(modes: &[usize], spatial: &[usize]) -> bool {
        std::panic::catch_unwind(|| SpectralConv::<4>::check_modes_fit(modes, spatial)).is_ok()
    }

    #[test]
    fn check_modes_fit_limits_odd_and_even() {
        // Non-last axis: m <= n/2, so the low and high blocks are disjoint.
        assert!(modes_fit(&[4, 1], &[8, 8]));
        assert!(!modes_fit(&[5, 1], &[8, 8]));
        assert!(modes_fit(&[3, 1], &[7, 8]));
        assert!(!modes_fit(&[4, 1], &[7, 8]));
        // Last axis: m <= n/2 + 1 rfft bins.
        assert!(modes_fit(&[1, 5], &[8, 8]));
        assert!(!modes_fit(&[1, 6], &[8, 8]));
        assert!(modes_fit(&[1, 4], &[8, 7]));
        assert!(!modes_fit(&[1, 5], &[8, 7]));
    }

    fn forward_with_modes<const R: usize>(modes: Vec<usize>, x_shape: [usize; R]) -> Tensor<R> {
        let device = Device::default();
        let conv = SpectralConv::<R>::new(&device, 1, 1, &modes);
        conv.forward(Tensor::<R>::random(
            x_shape,
            Distribution::Normal(0.0, 1.0),
            &device,
        ))
    }

    #[test]
    #[should_panic(expected = "modes[0] = 5 exceeds 4 for spatial extent 8")]
    fn forward_rejects_overlapping_modes() {
        // Previously ran silently: low 0..5 and high 3..8 overlap on rows 3..5.
        let _ = forward_with_modes::<4>(vec![5, 2], [1, 1, 8, 8]);
    }

    #[test]
    #[should_panic(expected = "modes[1] = 6 exceeds 5 for spatial extent 8")]
    fn forward_rejects_modes_beyond_rfft_bins() {
        // Previously an opaque out-of-range slice panic inside Burn.
        let _ = forward_with_modes::<4>(vec![2, 6], [1, 1, 8, 8]);
    }

    #[test]
    #[should_panic(expected = "modes[0] = 5 exceeds 2 for spatial extent 4")]
    fn forward_rejects_modes_larger_than_axis() {
        // Previously `n - m` underflowed usize in corner_ranges.
        let _ = forward_with_modes::<4>(vec![5, 2], [1, 1, 4, 8]);
    }

    /// With I = O = 1, unit weights on every corner and modes at the limit on
    /// every axis, each frequency bin is kept exactly once, so the layer is
    /// the identity. Requires an even extent on non-last axes (an odd extent
    /// drops the middle frequency).
    fn identity_at_mode_limit<const R: usize>(modes: Vec<usize>, x_shape: [usize; R]) {
        let device = Device::default();
        let mut conv = SpectralConv::<R>::new(&device, 1, 1, &modes);
        let mut w_shape = [1usize; R];
        w_shape[2..].copy_from_slice(&modes);
        for c in 0..conv.weights_re.len() {
            conv.weights_re[c] = Param::from_tensor(Tensor::<R>::ones(w_shape, &device));
            conv.weights_im[c] = Param::from_tensor(Tensor::<R>::zeros(w_shape, &device));
        }

        let x = Tensor::<R>::random(x_shape, Distribution::Normal(0.0, 1.0), &device);
        let out = conv.forward(x.clone());
        assert_eq!(out.dims(), x_shape);

        let x_vec = x.into_data().try_to_vec::<f32>().unwrap();
        let out_vec = out.into_data().try_to_vec::<f32>().unwrap();
        for (i, (a, e)) in out_vec.iter().zip(&x_vec).enumerate() {
            assert!((a - e).abs() < 1e-4, "mismatch at index {i}: {a} vs {e}");
        }
    }

    #[test]
    fn forward_at_mode_limit_is_identity_1d_even() {
        identity_at_mode_limit::<3>(vec![9], [1, 1, 16]);
    }

    #[test]
    fn forward_at_mode_limit_is_identity_2d_odd_last_axis() {
        identity_at_mode_limit::<4>(vec![4, 4], [1, 1, 8, 7]);
    }

    // --- regression: output must carry out_channels, not in_channels ---

    #[test]
    fn forward_shape_with_asymmetric_channels() {
        let device = Device::default();
        let (in_channels, out_channels) = (2, 3);
        let modes = vec![2, 2, 2];
        let (batch, h1, h2, m) = (1, 8, 8, 8);

        let conv = SpectralConv::<5>::new(&device, in_channels, out_channels, &modes);
        let x = Tensor::<5>::random(
            [batch, in_channels, h1, h2, m],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );

        assert_eq!(
            conv.forward(x).dims(),
            [batch, out_channels, h1, h2, m],
            "output shape should carry out_channels — regression test for the zeros-shape bug"
        );
    }

    // --- weight initialisation (issue #8) ---
    //
    // The distribution and bit-for-bit parity tests live in
    // `tests/spectral_init.rs`: they reseed Flex's process-wide RNG, so they
    // need their own test binary.

    #[test]
    fn init_default_is_li_uniform() {
        assert_eq!(SpectralInit::default(), SpectralInit::LiUniform);
    }
}
