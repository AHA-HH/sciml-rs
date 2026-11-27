//! The positional embeddings of `neuraloperator` 2.0.0 (`layers/embeddings.py`, library
//! CM1.4): [`GridEmbeddingNd`] (the FNO's optional grid, [`FNOConfig::positional_embedding`]),
//! [`SinusoidalEmbedding`] and [`RotaryEmbedding2d`] with its helpers [`rotate_half`],
//! [`apply_rotary_pos_emb`] and [`apply_2d_rotary_pos_emb`].
//!
//! None has parameters, so each is its own config. 2.0.0 builds its constants (the grid, the
//! frequencies) in f32, torch's default dtype, whatever the input's; these build them in the
//! input's dtype, so they agree with 2.0.0 to f32 rounding.
//!
//! [`FNOConfig::positional_embedding`]: crate::neural_operators::models::fno::FNOConfig::positional_embedding

// Burn's `#[derive(Config)]` writes `field: field` in the `new` it generates.
#![allow(clippy::redundant_field_names)]

use burn::{
    Tensor,
    config::Config,
    tensor::{DType, Device, FloatDType, Int},
};

/// 2.0.0's `GridEmbeddingND`: appends `D` coordinate channels to a tensor, one per spatial axis
/// in axis order, axis `a` holding `linspace(start_a, stop_a, s_a + 1)[:-1]`, i.e.
/// `start_a + (stop_a − start_a) · j / s_a` (on `[0, 1)` by default, without the endpoint,
/// unlike CONVENTIONS §2's grid).
#[derive(Config, Debug)]
pub struct GridEmbeddingNd {
    /// `(start, stop)` of each spatial axis; its length is `D`.
    pub boundaries: Vec<(f64, f64)>,
}

impl GridEmbeddingNd {
    /// The grid on `[0, 1)` along each of `d` axes, 2.0.0's FNO default (`"grid"`).
    pub fn unit(d: usize) -> Self {
        Self::new(vec![(0.0, 1.0); d])
    }

    /// The `D` coordinate channels for a tensor of shape `shape` with its channels on axis
    /// `channel_axis`: `shape` with that axis `D` wide, in `dtype` on `device`, the same in
    /// every batch row (axis 0). A constant, never tracked by autodiff.
    ///
    /// # Panics
    /// If `channel_axis` is 0 or not below `R`, or the spatial axes (all but 0 and
    /// `channel_axis`) are not as many as the boundaries.
    pub fn grid<const R: usize>(
        &self,
        shape: [usize; R],
        channel_axis: usize,
        dtype: DType,
        device: &Device,
    ) -> Tensor<R> {
        assert!(
            (1..R).contains(&channel_axis),
            "channel axis {channel_axis} of a rank-{R} tensor"
        );
        let spatial: Vec<usize> = (1..R).filter(|&a| a != channel_axis).collect();
        assert_eq!(
            spatial.len(),
            self.boundaries.len(),
            "a grid of {} axes for {} spatial axes",
            self.boundaries.len(),
            spatial.len()
        );
        let lines = spatial
            .iter()
            .zip(&self.boundaries)
            .map(|(&axis, &(start, stop))| {
                let n = shape[axis];
                let coords = Tensor::<1, Int>::arange(0..n as i64, device)
                    .cast(FloatDType::from(dtype))
                    .mul_scalar((stop - start) / n as f64)
                    .add_scalar(start);
                (axis, coords)
            });
        coordinate_channels(shape, channel_axis, lines)
    }

    /// `[B, C, s_1..s_D]` → `[B, C + D, s_1..s_D]`: the input with the grid appended, as
    /// 2.0.0's `forward`.
    ///
    /// # Panics
    /// If `R − 2` is not the number of boundaries.
    pub fn forward<const R: usize>(&self, x: Tensor<R>) -> Tensor<R> {
        let grid = self.grid(x.dims(), 1, x.dtype(), &x.device());
        Tensor::cat(vec![x, grid], 1)
    }
}

/// Coordinate channels for a tensor of shape `shape` with its channels on `channel_axis`:
/// each `(axis, line)` (`line` of length `shape[axis]`) broadcast over the other axes, the
/// channels concatenated in the order given. Shared by [`GridEmbeddingNd::grid`] and the
/// FNO's CONVENTIONS §2 grid.
pub(crate) fn coordinate_channels<const R: usize>(
    shape: [usize; R],
    channel_axis: usize,
    lines: impl Iterator<Item = (usize, Tensor<1>)>,
) -> Tensor<R> {
    let mut full = shape;
    full[channel_axis] = 1;
    let mut channels: Vec<Tensor<R>> = lines
        .map(|(axis, coords)| {
            let mut line = [1; R];
            line[axis] = shape[axis];
            coords.reshape(line).expand(full)
        })
        .collect();
    // One channel (1D) is returned as is: a single-input `cat` of the expanded channel,
    // followed by the caller's `cat`, gives wrong values on the fused GPU backends once the
    // chain has run at another shape (`upstream/burn/fused-cat-reused-at-new-shape/`).
    if channels.len() == 1 {
        return channels.pop().expect("one channel");
    }
    Tensor::cat(channels, channel_axis)
}

/// The frequencies of a [`SinusoidalEmbedding`].
#[derive(Config, Debug, PartialEq)]
pub enum SinusoidalKind {
    /// Vaswani et al.: level `k` of `L` has frequency `max_positions^(−2k / L)`.
    Transformer {
        /// 2.0.0's `max_positions`, default 10 000.
        max_positions: f64,
    },
    /// Mildenhall et al. (NeRF): level `k` has frequency `2^k π`.
    Nerf,
}

/// 2.0.0's `SinusoidalEmbedding`: each input channel `p` becomes `2L` channels,
/// `sin(ω_k p), cos(ω_k p)` for `k = 0..L`, interleaved, with `ω_k` by [`SinusoidalKind`].
#[derive(Config, Debug)]
pub struct SinusoidalEmbedding {
    /// Channels in (2.0.0's `in_channels`).
    pub in_channels: usize,
    /// Frequency levels `L` (2.0.0's `num_frequencies`, which has no working default).
    pub num_frequencies: usize,
    /// The frequencies (2.0.0's `embedding_type`, default `"transformer"`).
    #[config(default = "SinusoidalKind::Transformer { max_positions: 10000.0 }")]
    pub kind: SinusoidalKind,
}

impl SinusoidalEmbedding {
    /// Channels out, `2 · L · in_channels`.
    pub fn out_channels(&self) -> usize {
        2 * self.num_frequencies * self.in_channels
    }

    /// `[B, N, in_channels]` → `[B, N, 2 · L · in_channels]`; channel `2(c·L + k)` is
    /// `sin(ω_k x_c)` and `2(c·L + k) + 1` is `cos(ω_k x_c)`.
    ///
    /// # Panics
    /// If the last axis is not `in_channels` wide.
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, n, c] = x.dims();
        assert_eq!(
            c, self.in_channels,
            "expected {} channels",
            self.in_channels
        );
        let l = self.num_frequencies;
        let levels =
            Tensor::<1, Int>::arange(0..l as i64, &x.device()).cast(FloatDType::from(x.dtype()));
        // ω_k = exp(k · ln ω_1) for nerf's 2^k (times π), exp(−2k/L · ln m) for transformer.
        let freqs = match self.kind {
            SinusoidalKind::Nerf => levels
                .mul_scalar(core::f64::consts::LN_2)
                .exp()
                .mul_scalar(core::f64::consts::PI),
            SinusoidalKind::Transformer { max_positions } => levels
                .mul_scalar(-2.0 * max_positions.ln() / l as f64)
                .exp(),
        };
        let angles = x.unsqueeze_dim::<4>(3) * freqs.reshape([1, 1, 1, l]);
        Tensor::stack::<5>(vec![angles.clone().sin(), angles.cos()], 4).reshape([b, n, 2 * l * c])
    }
}

/// 2.0.0's `RotaryEmbedding2D` (Su et al., RoFormer): the rotation angles of a rotary
/// embedding for one coordinate; [`apply_2d_rotary_pos_emb`] applies two of them.
#[derive(Config, Debug)]
pub struct RotaryEmbedding2d {
    /// Angle channels `d`: `⌈d / 2⌉` frequencies `10000^(−2j / d)`, each used twice.
    pub dim: usize,
    /// 2.0.0's `min_freq`: coordinates are scaled by `scale / min_freq`.
    #[config(default = "1.0 / 64.0")]
    pub min_freq: f64,
    /// 2.0.0's `scale`.
    #[config(default = 1.0)]
    pub scale: f64,
}

impl RotaryEmbedding2d {
    /// Coordinates `[B, N]` → angles `[B, N, 2⌈d / 2⌉]`: `θ_j = c · scale / min_freq ·
    /// 10000^(−2j / d)`, the `⌈d / 2⌉` angles repeated once.
    pub fn forward(&self, coordinates: Tensor<2>) -> Tensor<3> {
        let half = self.dim.div_ceil(2);
        let inv_freq = Tensor::<1, Int>::arange(0..half as i64, &coordinates.device())
            .cast(FloatDType::from(coordinates.dtype()))
            .mul_scalar(-2.0 * 10000_f64.ln() / self.dim as f64)
            .exp();
        let freqs = coordinates
            .mul_scalar(self.scale / self.min_freq)
            .unsqueeze_dim::<3>(2)
            * inv_freq.reshape([1, 1, half]);
        Tensor::cat(vec![freqs.clone(), freqs], 2)
    }
}

/// 2.0.0's `rotate_half`: the last axis split in halves `(x_1, x_2)` becomes `(−x_2, x_1)`.
///
/// # Panics
/// If the last axis has odd length.
pub fn rotate_half<const R: usize>(x: Tensor<R>) -> Tensor<R> {
    let d = x.dims()[R - 1];
    assert!(
        d.is_multiple_of(2),
        "rotate_half needs an even last axis, got {d}"
    );
    let x1 = x.clone().narrow(R - 1, 0, d / 2);
    let x2 = x.narrow(R - 1, d / 2, d / 2);
    Tensor::cat(vec![x2.neg(), x1], R - 1)
}

/// 2.0.0's `apply_rotary_pos_emb`: `t · cos(freqs) + rotate_half(t) · sin(freqs)`, `freqs`
/// broadcast to `t`'s shape.
///
/// # Panics
/// As [`rotate_half`], or if the shapes do not broadcast.
pub fn apply_rotary_pos_emb<const R: usize>(t: Tensor<R>, freqs: Tensor<R>) -> Tensor<R> {
    t.clone() * freqs.clone().cos() + rotate_half(t) * freqs.sin()
}

/// 2.0.0's `RotaryEmbedding2D.apply_2d_rotary_pos_emb`: the first half of `t`'s last axis
/// rotated by `freqs_x`, the second by `freqs_y` (each from [`RotaryEmbedding2d::forward`]).
/// `t`: `[B, N, 2d]`; `freqs_x`, `freqs_y`: `[B, N, d]`.
///
/// # Panics
/// If `t`'s last axis is not twice the frequencies', or as [`rotate_half`].
pub fn apply_2d_rotary_pos_emb(t: Tensor<3>, freqs_x: Tensor<3>, freqs_y: Tensor<3>) -> Tensor<3> {
    let d = t.dims()[2];
    let t_x = t.clone().narrow(2, 0, d / 2);
    let t_y = t.narrow(2, d / 2, d - d / 2);
    Tensor::cat(
        vec![
            apply_rotary_pos_emb(t_x, freqs_x),
            apply_rotary_pos_emb(t_y, freqs_y),
        ],
        2,
    )
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::*;

    #[test]
    fn rotate_half_by_hand() {
        let device = Device::default();
        let x = Tensor::<2>::from_data([[1.0, 2.0, 3.0, 4.0]], &device);
        rotate_half(x).into_data().assert_approx_eq::<f32>(
            &Tensor::<2>::from_data([[-3.0, -4.0, 1.0, 2.0]], &device).into_data(),
            Tolerance::default(),
        );
    }

    /// Axis 1 of a 3×2 grid on [0, 2) × [−1, 1): 0, 2/3, 4/3; axis 2: −1, 0.
    #[test]
    fn grid_values_by_hand() {
        let device = Device::default();
        let g = GridEmbeddingNd::new(vec![(0.0, 2.0), (-1.0, 1.0)]).grid(
            [1, 2, 3, 2],
            1,
            DType::F32,
            &device,
        );
        let expected = Tensor::<4>::from_data(
            [[
                [[0.0, 0.0], [2.0 / 3.0, 2.0 / 3.0], [4.0 / 3.0, 4.0 / 3.0]],
                [[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
            ]],
            &device,
        );
        g.into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
    }

    #[test]
    fn out_channels_and_shapes() {
        let device = Device::default();
        let s = SinusoidalEmbedding::new(3, 4);
        assert_eq!(s.out_channels(), 24);
        assert_eq!(
            s.forward(Tensor::zeros([2, 5, 3], &device)).dims(),
            [2, 5, 24]
        );
        let r = RotaryEmbedding2d::new(7);
        assert_eq!(r.forward(Tensor::zeros([2, 5], &device)).dims(), [2, 5, 8]);
    }
}
