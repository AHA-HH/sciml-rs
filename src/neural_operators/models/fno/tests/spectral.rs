//! The spectral-convolution options (library CM1.5a): what they build. Parity with 2.0.0's
//! `FNO` is `tests/parity_fno.rs`.

use super::*;

/// 2 layers, width 4, 2D modes (3, 2).
fn config() -> FNOConfig {
    test_config(vec![3, 2], 1, 4)
}

/// Each parameter's name and number of values.
fn sizes(fno: &FNO<4>) -> Vec<(String, usize)> {
    params(fno).into_iter().map(|(n, v)| (n, v.len())).collect()
}

/// Every option given its "off" value builds the parameters of every option absent.
#[test]
fn options_off_build_today_s_parameters() {
    let device = Device::default();
    let off = config()
        .with_conv_bias(Some(false))
        .with_separable(Some(false))
        .with_max_modes(None)
        .with_tanh_stabilizer(Some(false))
        .init::<4>(&device);
    assert_eq!(sizes(&off), sizes(&config().init::<4>(&device)));
    assert!(off.conv_biases.is_empty() && !off.tanh_stabilizer);
}

#[test]
fn conv_bias_adds_one_width_vector_per_layer() {
    let fno: FNO<4> = config().with_conv_bias(Some(true)).init(&Device::default());
    let biases: Vec<_> = sizes(&fno)
        .into_iter()
        .filter(|(n, _)| n.starts_with("conv_biases."))
        .collect();
    assert_eq!(
        biases,
        vec![("conv_biases.0".into(), 4), ("conv_biases.1".into(), 4)]
    );
}

/// Separable: `W` weights per mode and corner, not `W · W`.
#[test]
fn separable_stores_one_weight_per_channel() {
    let device = Device::default();
    let weights = |fno: &FNO<4>| -> usize {
        sizes(fno)
            .into_iter()
            .filter(|(n, _)| n.starts_with("conv.0.weights_re."))
            .map(|(_, len)| len)
            .sum()
    };
    let dense = weights(&config().init(&device));
    let separable = weights(&config().with_separable(Some(true)).init(&device));
    assert_eq!((dense, separable), (2 * 4 * 4 * 3 * 2, 2 * 4 * 3 * 2));
}

/// `max_modes` sets the stored weights; the forward uses `modes` of them.
#[test]
fn max_modes_stores_more_than_it_uses() {
    let device = Device::default();
    let fno: FNO<4> = config().with_max_modes(Some(vec![5, 4])).init(&device);
    let corner = sizes(&fno)
        .into_iter()
        .find(|(n, _)| n == "conv.0.weights_re.0")
        .unwrap();
    assert_eq!(corner.1, 4 * 4 * 5 * 4);
    let x = Tensor::<4>::random([2, 12, 10, 1], Distribution::Default, &device);
    assert_eq!(fno.forward(x).dims(), [2, 12, 10, 1]);
}

#[test]
#[should_panic(expected = "exceeds")]
fn max_modes_below_modes_panics() {
    config()
        .with_max_modes(Some(vec![2, 2]))
        .init::<4>(&Device::default());
}
