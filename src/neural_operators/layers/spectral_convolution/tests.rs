//! The spectral layer's tests: corners, the PyTorch and NumPy references, gradients, the mode
//! limits, `with_modes` and separable weights (the FFTs' own are in `utils/fft.rs`).

use super::*;
use burn::tensor::Distribution;

#[test]
fn with_bc_prepends_and_panics_on_wrong_rank() {
    assert_eq!(with_bc::<usize, 4>(1, 2, &[3, 4]), [1, 2, 3, 4]);
    let bad = std::panic::catch_unwind(|| with_bc::<usize, 4>(1, 2, &[3]));
    assert!(bad.is_err());
}

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

    let (out_re, out_im) = SpectralConv::<3>::complex_multiplication(x_re, x_im, w_re, w_im);

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
        4.25, 4.25, 4.25, 4.25, 1.85, 1.85, 1.85, 1.85, 3.85, 3.85, 3.85, 3.85, 6.25, 6.25, 6.25,
        6.25, -2.15, -2.15, -2.15, -2.15, -4.55, -4.55, -4.55, -4.55, -2.55, -2.55, -2.55, -2.55,
        -0.15, -0.15, -0.15, -0.15, 2.65, 2.65, 2.65, 2.65, 0.25, 0.25, 0.25, 0.25, 2.25, 2.25,
        2.25, 2.25, 4.65, 4.65, 4.65, 4.65, 9.05, 9.05, 9.05, 9.05, 6.65, 6.65, 6.65, 6.65, 8.65,
        8.65, 8.65, 8.65, 11.05, 11.05, 11.05, 11.05,
    ];

    for (i, (a, e)) in out_vals.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < 1e-4, "mismatch at index {i}: {a} vs {e}");
    }
}

fn gradient_flow_check<const R: usize>(modes: Vec<usize>, x_shape: [usize; R]) {
    let device = Device::default().autodiff();
    let channels = x_shape[1];

    let conv = SpectralConv::<R>::new(&device, channels, channels, &modes);
    let x = Tensor::<R>::random(x_shape, Distribution::Normal(0.0, 1.0), &device).require_grad();

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

// --- regression: modes must fit the grid (merge `04ec800`) ---

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

/// `forward_to` at the input's extents is `forward`, bit for bit; elsewhere its output has
/// the requested extents.
#[test]
fn forward_to_at_the_input_size_is_forward() {
    let device = Device::default();
    let conv = SpectralConv::<4>::new(&device, 2, 3, &[3, 2]);
    let x = Tensor::<4>::random([2, 2, 8, 6], Distribution::Default, &device);
    assert_eq!(
        bits(conv.forward_to(x.clone(), &[8, 6])),
        bits(conv.forward(x.clone()))
    );
    assert_eq!(conv.forward_to(x, &[11, 5]).dims(), [2, 3, 11, 5]);
}

#[test]
fn init_default_is_li_uniform() {
    assert_eq!(SpectralInit::default(), SpectralInit::LiUniform);
}

// --- truncation below the stored modes and separable weights (library CM1.2) ---

fn bits<const R: usize>(t: Tensor<R>) -> Vec<u32> {
    let v = t.into_data().try_to_vec::<f32>().unwrap();
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn with_modes_matches_layer_built_at_smaller_modes() {
    let device = Device::default();
    let (stored, active) = ([4, 3], [2, 2]);
    let conv = SpectralConv::<4>::new(&device, 2, 3, &stored);
    // The smaller layer holds each corner's retained block of the stored weights.
    let mut small = SpectralConv::<4>::new(&device, 2, 3, &active);
    for c in 0..conv.weights_re.len() {
        let ranges: [Range<usize>; 4] = with_bc(
            0..2,
            0..3,
            &SpectralConv::<4>::corner_ranges(c, &active, &stored),
        );
        small.weights_re[c] = Param::from_tensor(conv.weights_re[c].val().slice(ranges.clone()));
        small.weights_im[c] = Param::from_tensor(conv.weights_im[c].val().slice(ranges));
    }
    let x = Tensor::<4>::random([2, 2, 12, 9], Distribution::Normal(0.0, 1.0), &device);
    let truncated = conv.with_modes(&active).forward(x.clone());
    assert_eq!(bits(truncated), bits(small.forward(x)));
}

#[test]
fn with_modes_at_stored_modes_changes_nothing() {
    let device = Device::default();
    let conv = SpectralConv::<5>::new(&device, 2, 2, &[3, 2, 2]);
    let x = Tensor::<5>::random([1, 2, 8, 6, 5], Distribution::Normal(0.0, 1.0), &device);
    let before = conv.forward(x.clone());
    assert_eq!(bits(conv.with_modes(&[3, 2, 2]).forward(x)), bits(before));
}

#[test]
#[should_panic(expected = "modes[1] = 4 exceeds the stored 3")]
fn with_modes_rejects_more_than_stored() {
    let _ = SpectralConv::<4>::new(&Device::default(), 1, 1, &[4, 3]).with_modes(&[2, 4]);
}

#[test]
#[should_panic(expected = "one entry per spatial axis")]
fn with_modes_rejects_wrong_length() {
    let _ = SpectralConv::<4>::new(&Device::default(), 1, 1, &[4, 3]).with_modes(&[2]);
}

/// Truncation applies to separable weights as to dense ones.
#[test]
fn with_modes_truncates_separable_weights() {
    let device = Device::default();
    let (stored, active) = ([4, 3], [3, 2]);
    let conv = SpectralConv::<4>::new_separable(&device, 2, &stored, SpectralInit::Normal);
    let mut small = SpectralConv::<4>::new_separable(&device, 2, &active, SpectralInit::Normal);
    for c in 0..conv.weights_re.len() {
        let ranges: [Range<usize>; 4] = with_bc(
            0..1,
            0..2,
            &SpectralConv::<4>::corner_ranges(c, &active, &stored),
        );
        small.weights_re[c] = Param::from_tensor(conv.weights_re[c].val().slice(ranges.clone()));
        small.weights_im[c] = Param::from_tensor(conv.weights_im[c].val().slice(ranges));
    }
    let x = Tensor::<4>::random([2, 2, 12, 9], Distribution::Normal(0.0, 1.0), &device);
    let (a, e) = (
        conv.with_modes(&active).forward(x.clone()),
        small.forward(x),
    );
    // Bit for bit on Flex; Metal's fused element-wise kernel rounds a strided slice
    // differently (a few ulps), while a wrong mode selection would be O(1).
    let scale: f32 = e.clone().abs().max().into_scalar();
    let err: f32 = (a - e).abs().max().into_scalar();
    assert!(
        err <= 1e-6 * scale,
        "truncated separable: {err:e} of {scale:e}"
    );
}

/// A separable layer is the dense layer whose channel map is diagonal at every mode.
#[test]
fn separable_equals_dense_with_diagonal_weights() {
    let device = Device::default();
    let (c, modes) = (3, [3, 2]);
    let sep = SpectralConv::<4>::new_separable(&device, c, &modes, SpectralInit::Normal);
    let mut dense = SpectralConv::<4>::new(&device, c, c, &modes);
    let eye = Tensor::<2>::eye(c, &device).reshape([c, c, 1, 1]);
    for k in 0..sep.weights_re.len() {
        // [C, C, 1, 1] * [1, C, m..]: entry (i, o) is δ_io w[o].
        dense.weights_re[k] = Param::from_tensor(eye.clone() * sep.weights_re[k].val());
        dense.weights_im[k] = Param::from_tensor(eye.clone() * sep.weights_im[k].val());
    }
    let x = Tensor::<4>::random([2, c, 10, 7], Distribution::Normal(0.0, 1.0), &device);
    let (a, e) = (sep.forward(x.clone()), dense.forward(x));
    let scale: f32 = e.clone().abs().max().into_scalar();
    let err: f32 = (a - e).abs().max().into_scalar();
    assert!(
        err <= 1e-6 * scale,
        "separable vs dense: {err:e} of {scale:e}"
    );
}
