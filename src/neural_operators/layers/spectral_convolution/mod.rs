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
    tensor::{Device, Distribution, ops::PadMode},
};

use std::ops::Range;

use crate::neural_operators::utils::fft::{fft_ctensor, ifft_ctensor, ifft_ctensor_to};

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

/// `[a, b, rest..]` as a fixed-size array: the batch/channel prefix followed
/// by one entry per spatial axis.
///
/// # Panics
/// If `rest.len() != R - 2`.
fn with_bc<T: Clone, const R: usize>(a: T, b: T, rest: &[T]) -> [T; R] {
    let mut v = vec![a, b];
    v.extend_from_slice(rest);
    v.try_into()
        .unwrap_or_else(|v: Vec<T>| panic!("with_bc: {} entries for rank {R}", v.len()))
}

/// `x` with `axis` trimmed to its first `len` entries, or zero-padded at its end to `len`:
/// torch's `n` of a 1D FFT, along one axis.
fn resize_end<const R: usize>(x: Tensor<R>, axis: usize, len: usize) -> Tensor<R> {
    let dims = x.dims();
    let n = dims[axis];
    if len <= n {
        let ranges: [Range<usize>; R] =
            core::array::from_fn(|i| if i == axis { 0..len } else { 0..dims[i] });
        x.slice(ranges)
    } else {
        let pairs: [(usize, usize); R] =
            core::array::from_fn(|i| if i == axis { (0, len - n) } else { (0, 0) });
        x.pad(pairs, PadMode::Constant(0.0))
    }
}

/// Learned spectral convolution over `R - 2` spatial axes.
///
/// The weights are stored per mode corner (CONVENTIONS §5). The modes used by
/// [`forward`](Self::forward) may be fewer than the stored ones
/// ([`with_modes`](Self::with_modes)), as `neuraloperator`'s `n_modes` below its
/// `max_n_modes`.
#[derive(Module, Debug)]
pub struct SpectralConv<const R: usize> {
    /// One weight per mode corner; `len() == 2^(D-1)`.
    /// Each is `[in_channels, out_channels, stored[0], .., stored[D-1]]`, or
    /// `[1, channels, stored..]` if `separable`; the stored modes are its extents.
    weights_re: Vec<Param<Tensor<R>>>,
    weights_im: Vec<Param<Tensor<R>>>,
    /// Retained frequency count per spatial axis, at most the stored count;
    /// `len() == R - 2`.
    modes: Vec<usize>,
    /// One weight per channel and mode (a diagonal channel map) instead of a
    /// full `in_channels × out_channels` map.
    separable: bool,
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
        let distribution = init.distribution(in_channels, out_channels);
        Self::draw(
            device,
            [in_channels, out_channels],
            modes,
            distribution,
            false,
        )
    }

    /// A separable layer: `channels` in and out, one complex weight per channel
    /// and retained mode, multiplying each channel's spectrum on its own
    /// (`neuraloperator`'s `separable=True`). The weights are drawn as for
    /// `in_channels = out_channels = channels`.
    ///
    /// ```
    /// use burn::tensor::Device;
    /// use sciml_rs::neural_operators::layers::spectral_convolution::{SpectralConv, SpectralInit};
    ///
    /// let device = Device::default();
    /// let _conv = SpectralConv::<4>::new_separable(&device, 8, &[6, 4], SpectralInit::Normal);
    /// ```
    ///
    /// # Panics
    ///
    /// If `modes.len() != R - 2`.
    pub fn new_separable(
        device: &Device,
        channels: usize,
        modes: &[usize],
        init: SpectralInit,
    ) -> Self {
        let distribution = init.distribution(channels, channels);
        Self::draw(device, [1, channels], modes, distribution, true)
    }

    /// Draws every corner's weights, `[lead[0], lead[1], modes..]`.
    ///
    /// # Panics
    ///
    /// If `modes.len() != R - 2`.
    fn draw(
        device: &Device,
        lead: [usize; 2],
        modes: &[usize],
        distribution: Distribution,
        separable: bool,
    ) -> Self {
        let () = Self::RANK_OK;
        assert_eq!(
            modes.len(),
            R - 2,
            "modes must have one entry per spatial axis (R - 2)"
        );
        let modes = modes.to_vec();
        let num_corners = 1usize << (modes.len() - 1);

        let shape: [usize; R] = with_bc(lead[0], lead[1], &modes);

        // Draw order (per corner: real, then imaginary) is part of the
        // contract: with a fixed seed, `LiUniform` reproduces the weights of
        // earlier releases bit for bit.
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
            separable,
        }
    }

    /// The layer with `modes` retained per spatial axis, at most the stored
    /// modes: `neuraloperator`'s `n_modes` below `max_n_modes`. On a non-last
    /// axis the low corner keeps frequencies `0..m` and the high corner `-m..-1`
    /// (the last `m` rows of its stored block); the last axis keeps `0..m`.
    /// The weights are kept, so the modes can be raised again up to the stored
    /// ones.
    ///
    /// ```
    /// use burn::tensor::Device;
    /// use sciml_rs::neural_operators::layers::spectral_convolution::SpectralConv;
    ///
    /// let device = Device::default();
    /// let _conv = SpectralConv::<4>::new(&device, 4, 4, &[5, 8]).with_modes(&[3, 4]);
    /// ```
    ///
    /// # Panics
    ///
    /// If `modes.len() != R - 2`, or a mode count exceeds the stored one.
    pub fn with_modes(mut self, modes: &[usize]) -> Self {
        assert_eq!(
            modes.len(),
            R - 2,
            "modes must have one entry per spatial axis (R - 2)"
        );
        for (axis, (&m, &stored)) in modes.iter().zip(&self.stored_modes()).enumerate() {
            assert!(
                m <= stored,
                "SpectralConv: modes[{axis}] = {m} exceeds the stored {stored}"
            );
        }
        self.modes = modes.to_vec();
        self
    }

    /// The stored mode counts: the weights' spatial extents.
    fn stored_modes(&self) -> Vec<usize> {
        self.weights_re[0].val().dims()[2..].to_vec()
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
    /// Computes `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`, batched over modes,
    /// with Gauss's three-matmul form: `ac`, `bd` and `(a + b)(c + d)` give
    /// `re = ac - bd` and `im = (a + b)(c + d) - ac - bd`. That is 3 matmuls
    /// instead of the 4 (`ac`, `bd`, `ad`, `bc`) of the direct product, at the
    /// cost of a few extra additions.
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
    ) -> (Tensor<R>, Tensor<R>) {
        let (b, i) = (x_re.dims()[0], x_re.dims()[1]);
        let o = w_re.dims()[1];
        let modes = &w_re.dims()[2..];
        let flat: usize = modes.iter().product();

        let x_re = x_re.reshape([b, i, flat]).permute([2, 0, 1]);
        let x_im = x_im.reshape([b, i, flat]).permute([2, 0, 1]);
        let w_re = w_re.reshape([i, o, flat]).permute([2, 0, 1]);
        let w_im = w_im.reshape([i, o, flat]).permute([2, 0, 1]);

        let ac = x_re.clone().matmul(w_re.clone());
        let bd = x_im.clone().matmul(w_im.clone());

        let ab_cd = (x_re + x_im).matmul(w_re + w_im);

        let out_re = ac.clone() - bd.clone();
        let out_im = ab_cd - ac - bd;

        let out_shape: [usize; R] = with_bc(b, o, modes);

        (
            out_re.permute([1, 2, 0]).reshape(out_shape),
            out_im.permute([1, 2, 0]).reshape(out_shape),
        )
    }

    /// Per-channel complex product `(a + bi)(c + di)` of `x` `[B, C, modes..]`
    /// and separable weights `[1, C, modes..]`, broadcast over the batch.
    fn separable_multiplication(
        x_re: Tensor<R>,
        x_im: Tensor<R>,
        w_re: Tensor<R>,
        w_im: Tensor<R>,
    ) -> (Tensor<R>, Tensor<R>) {
        (
            x_re.clone() * w_re.clone() - x_im.clone() * w_im.clone(),
            x_re * w_im + x_im * w_re,
        )
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
        let dims = x.dims();
        self.forward_to(x, &dims[2..])
    }

    /// `[B, in_channels, spatial..] -> [B, out_channels, out..]`: [`Self::forward`] with the
    /// output on `out`'s spatial extents, as 2.0.0's `SpectralConv` with a
    /// `resolution_scaling_factor` or an `output_shape` (library CM1.5b). The spectrum is
    /// built at the input's size, then trimmed or zero-padded at the end of each axis (torch's
    /// `irfftn(s=out)`), and scaled by `N_out / N_in` (products of the spatial extents), so the
    /// amplitude is 2.0.0's `fft_norm="forward"`'s (an extension of CONVENTIONS §4's backward
    /// norm, which it equals when the extents agree). At the input's own extents it is
    /// [`Self::forward`], the same operations.
    ///
    /// # Panics
    ///
    /// As [`Self::forward`]; if `out` has not `R - 2` entries or one is 0.
    pub fn forward_to(&self, x: Tensor<R>, out: &[usize]) -> Tensor<R> {
        // Captured before the transform: irfft needs the original extent to
        // recover the correct length from the truncated half-spectrum.
        let () = Self::RANK_OK;
        let orig_dims = x.dims();
        assert!(
            out.len() == R - 2 && out.iter().all(|&n| n > 0),
            "SpectralConv: output extents {out:?} for {} spatial axes",
            R - 2
        );
        Self::check_modes_fit(&self.modes, &orig_dims[2..]);
        let (batch, in_ch) = (orig_dims[0], orig_dims[1]);
        let out_ch = self.weights_re[0].val().dims()[1];
        let num_corners = self.weights_re.len();
        let stored = self.stored_modes();
        // Slice the weights only when truncated, so the default path is unchanged.
        let truncated = stored != self.modes;

        let (x_ft_re, x_ft_im) = fft_ctensor(x);
        // Post-transform extents - the last axis is now n/2 + 1.
        let spec_dims: Vec<usize> = x_ft_re.dims()[2..].to_vec();

        let out_shape: [usize; R] = with_bc(batch, out_ch, &spec_dims);

        // Frequencies outside the retained modes stay zero - this is the
        // low-pass truncation that makes the operator resolution-independent.
        let mut out_ft_re = Tensor::<R>::zeros(out_shape, &x_ft_re.device());
        let mut out_ft_im = Tensor::<R>::zeros(out_shape, &x_ft_im.device());

        for corner in 0..num_corners {
            let ranges = Self::corner_ranges(corner, &self.modes, &spec_dims);

            let in_ranges: [Range<usize>; R] = with_bc(0..batch, 0..in_ch, &ranges);

            let x_re = x_ft_re.clone().slice(in_ranges.clone());
            let x_im = x_ft_im.clone().slice(in_ranges);

            let (mut w_re, mut w_im) =
                (self.weights_re[corner].val(), self.weights_im[corner].val());
            if truncated {
                let lead = w_re.dims();
                let w_ranges: [Range<usize>; R] = with_bc(
                    0..lead[0],
                    0..lead[1],
                    &Self::corner_ranges(corner, &self.modes, &stored),
                );
                w_re = w_re.slice(w_ranges.clone());
                w_im = w_im.slice(w_ranges);
            }
            let (o_re, o_im) = if self.separable {
                Self::separable_multiplication(x_re, x_im, w_re, w_im)
            } else {
                Self::complex_multiplication(x_re, x_im, w_re, w_im)
            };

            let out_ranges: [Range<usize>; R] = with_bc(0..batch, 0..out_ch, &ranges);

            out_ft_re = out_ft_re.slice_assign(out_ranges.clone(), o_re);
            out_ft_im = out_ft_im.slice_assign(out_ranges, o_im);
        }

        if out == &orig_dims[2..] {
            return ifft_ctensor(out_ft_re, out_ft_im, &orig_dims);
        }
        let out_dims: [usize; R] = with_bc(batch, out_ch, out);
        for (axis, &n) in out_dims.iter().enumerate().skip(2) {
            let len = if axis == R - 1 { n / 2 + 1 } else { n };
            out_ft_re = resize_end(out_ft_re, axis, len);
            out_ft_im = resize_end(out_ft_im, axis, len);
        }
        ifft_ctensor_to(out_ft_re, out_ft_im, &orig_dims, &out_dims)
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
mod tests;
