//! `neuraloperator` 2.0.0's `resample` (`layers/resample.py`, library CM1.5b): a
//! channels-first activation brought to other spatial extents, as 2.0.0's FNO blocks do to
//! their skips when a layer changes the resolution.

use std::ops::Range;

use burn::{
    Tensor,
    tensor::module::{CoordinateTransformMode, InterpolateMode, InterpolateOptions, interpolate},
};

use super::spectral_convolution::SpectralConv;

/// `[B, C, s..]` → `[B, C, out..]` by 2.0.0's method for the rank: linear interpolation in
/// 1D and bicubic (`a = −0.75`, edges clamped) in 2D, both with `align_corners=True` (torch's
/// `F.interpolate`, here Burn's `interpolate`); for three or more axes a Fourier resize, the
/// lowest `min(s, out)` modes of each axis kept (on a non-last axis `m / 2` non-negative and
/// `⌈m / 2⌉` negative frequencies) at 2.0.0's `norm="forward"` amplitude (as
/// [`SpectralConv::forward_to`]; CONVENTIONS §4 for the transforms). `x` itself when
/// `out` is its own spatial extent (2.0.0 compares a list with a tuple there and always
/// resamples, an exact identity in 1D and 2D).
///
/// # Panics
/// If `out` has not `R - 2` entries or one is 0.
pub fn resample<const R: usize>(x: Tensor<R>, out: &[usize]) -> Tensor<R> {
    let dims = x.dims();
    assert!(
        out.len() == R - 2 && out.iter().all(|&n| n > 0),
        "resample: output extents {out:?} for {} spatial axes",
        R - 2
    );
    if out == &dims[2..] {
        return x;
    }
    let shape: [usize; R] = core::array::from_fn(|i| if i < 2 { dims[i] } else { out[i - 2] });
    let (b, c) = (dims[0], dims[1]);
    let coordinate_transform = CoordinateTransformMode::AlignCorners;
    match R {
        3 => {
            let mode = InterpolateMode::Bilinear {
                coordinate_transform,
            };
            let x = x.reshape([b, c, 1, dims[2]]);
            interpolate(
                x,
                InterpolateOptions::new(mode).with_output_size([1, out[0]]),
            )
            .reshape(shape)
        }
        4 => {
            let mode = InterpolateMode::Bicubic {
                coordinate_transform,
                a: -0.75,
            };
            let x = x.reshape([b, c, dims[2], dims[3]]);
            interpolate(
                x,
                InterpolateOptions::new(mode).with_output_size([out[0], out[1]]),
            )
            .reshape(shape)
        }
        _ => fourier(x, shape),
    }
}

/// The Fourier resize of [`resample`] to the full shape `shape` (`[B, C, out..]`).
fn fourier<const R: usize>(x: Tensor<R>, shape: [usize; R]) -> Tensor<R> {
    let dims = x.dims();
    let device = x.device();
    let (re, im) = SpectralConv::<R>::fft_ctensor(x);
    let old = re.dims();
    let spec: [usize; R] = core::array::from_fn(|i| {
        if i == R - 1 {
            shape[i] / 2 + 1
        } else {
            shape[i]
        }
    });
    let mut out_re = Tensor::<R>::zeros(spec, &device);
    let mut out_im = Tensor::<R>::zeros(spec, &device);
    // One corner per sign pattern of the non-last axes; the last (rfft) axis is non-negative.
    // Not `SpectralConv`'s corners: 2.0.0 splits m modes as m / 2 low and ⌈m / 2⌉ high here.
    for corner in 0..1usize << (R - 3) {
        let mut src: [Range<usize>; R] = core::array::from_fn(|i| 0..old[i]);
        let mut dst = src.clone();
        for axis in 2..R {
            let m = old[axis].min(spec[axis]);
            let low = axis == R - 1 || (corner >> (axis - 2)) & 1 == 0;
            (src[axis], dst[axis]) = if low {
                let n = if axis == R - 1 { m } else { m / 2 };
                (0..n, 0..n)
            } else {
                let n = m - m / 2;
                (old[axis] - n..old[axis], spec[axis] - n..spec[axis])
            };
        }
        if src.iter().any(|r| r.is_empty()) {
            continue;
        }
        out_re = out_re.slice_assign(dst.clone(), re.clone().slice(src.clone()));
        out_im = out_im.slice_assign(dst, im.clone().slice(src));
    }
    let n_in: usize = dims[2..].iter().product();
    let n_out: usize = shape[2..].iter().product();
    SpectralConv::<R>::ifft_ctensor(out_re, out_im, &shape).mul_scalar(n_out as f64 / n_in as f64)
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, Distribution, Tensor, TensorData, Tolerance};

    use super::resample;

    #[test]
    fn same_extent_is_the_identity() {
        let device = Device::default();
        let x = Tensor::<5>::random([1, 2, 4, 3, 5], Distribution::Default, &device);
        resample(x.clone(), &[4, 3, 5])
            .into_data()
            .assert_eq(&x.into_data(), true);
    }

    /// Linear with `align_corners` reproduces a linear function at any size.
    #[test]
    fn linear_1d_is_exact_on_a_line() {
        let device = Device::default();
        let line = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| 3.0 * i as f32 / (n - 1) as f32 - 1.0)
                .collect()
        };
        let x = Tensor::<3>::from_data(TensorData::new(line(5), [1, 1, 5]), &device);
        resample(x, &[9]).into_data().assert_approx_eq::<f32>(
            &TensorData::new(line(9), [1, 1, 9]),
            Tolerance::absolute(1e-6),
        );
    }

    /// The Fourier resize keeps a band-limited signal: `cos(2π x) · sin(2π z)` on a periodic
    /// 6 × 4 × 8 grid, upsampled to 12 × 5 × 10, is the same function there.
    #[test]
    fn fourier_3d_keeps_a_band_limited_signal() {
        let device = Device::default();
        let field = |[a, b, c]: [usize; 3]| -> Vec<f32> {
            let mut v = Vec::new();
            for i in 0..a {
                for _ in 0..b {
                    for k in 0..c {
                        let (x, z) = (i as f64 / a as f64, k as f64 / c as f64);
                        v.push(
                            ((2.0 * std::f64::consts::PI * x).cos()
                                * (2.0 * std::f64::consts::PI * z).sin())
                                as f32,
                        );
                    }
                }
            }
            v
        };
        let x = Tensor::<5>::from_data(TensorData::new(field([6, 4, 8]), [1, 1, 6, 4, 8]), &device);
        resample(x, &[12, 5, 10])
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::new(field([12, 5, 10]), [1, 1, 12, 5, 10]),
                Tolerance::absolute(1e-5),
            );
    }

    #[test]
    #[should_panic(expected = "output extents")]
    fn wrong_rank_of_extents_panics() {
        let x = Tensor::<4>::zeros([1, 1, 4, 4], &Device::default());
        resample(x, &[4]);
    }
}
