//! The block options (library CM1.3b): what they build, and AdaIN's embedding. Parity with
//! 2.0.0's `FNOBlocks` is `tests/parity_fno_blocks.rs`.

use super::*;
use crate::neural_operators::layers::norm::NormKind;

fn config() -> FNOConfig {
    FNOConfig::new(vec![2, 2], 1, 1)
        .with_hidden_channels(6)
        .with_n_layers(3)
}

/// With every option absent, no block parameters beyond today's `conv` and `w`.
#[test]
fn absent_options_build_today_s_block() {
    let fno: FNO<4> = config().init(&Device::default());
    assert_eq!((fno.conv.len(), fno.w.len()), (3, 3));
    assert!(fno.fno_skips.is_empty() && fno.channel_mlp.is_empty());
    assert!(fno.channel_mlp_skips.is_empty() && fno.norms.is_empty());
    assert_eq!(fno.activation, Activation::Relu);
    assert!(!fno.preactivation && !fno.repeat_activation);
}

#[test]
fn options_build_2_0_0_s_layers() {
    let fno: FNO<4> = config()
        .with_fno_skip(Some(SkipKind::Linear))
        .with_channel_mlp(Some(BlockMlpConfig::new()))
        .with_norm(Some(NormKind::Group))
        .init(&Device::default());
    assert!(fno.w.is_empty());
    assert_eq!(fno.fno_skips.len(), 3);
    assert_eq!(fno.channel_mlp_skips.len(), 3);
    assert_eq!(fno.norms.len(), 6);
    // Hidden width round(6 · 0.5) = 3, as 2.0.0's `round(out_channels * expansion)`:
    // 6 → 3 → 6 with biases is 6·3 + 3 + 3·6 + 6 = 45 parameters.
    for mlp in &fno.channel_mlp {
        assert_eq!(mlp.num_params(), 45);
    }
}

fn ada_in_fno() -> FNO<4> {
    config()
        .with_norm(Some(NormKind::AdaIn { embed_dim: 2 }))
        .init(&Device::default())
}

#[test]
fn ada_in_conditions_on_the_embedding() {
    let device = Device::default();
    let fno = ada_in_fno();
    let x = Tensor::<4>::random([2, 8, 8, 1], Distribution::Default, &device);
    let y =
        |e: [f32; 2]| fno.forward_with_embedding(x.clone(), Some(Tensor::from_data(e, &device)));
    let (a, b) = (y([1.0, -1.0]), y([0.5, 2.0]));
    assert_eq!(a.dims(), [2, 8, 8, 1]);
    assert_ne!(
        a.into_data(),
        b.into_data(),
        "the output must depend on the embedding"
    );
}

#[test]
#[should_panic(expected = "an AdaIN norm needs an embedding")]
fn ada_in_forward_needs_an_embedding() {
    let x = Tensor::<4>::zeros([1, 8, 8, 1], &Device::default());
    ada_in_fno().forward(x);
}

/// Python's `round` takes a tie to the even neighbour: width 5 at expansion 0.5 has 2 hidden
/// channels in 2.0.0, not 3. 5 → 2 → 5 with biases is 5·2 + 2 + 2·5 + 5 = 27 parameters.
#[test]
fn mlp_width_rounds_half_to_even() {
    let fno: FNO<4> = config()
        .with_hidden_channels(5)
        .with_channel_mlp(Some(BlockMlpConfig::new()))
        .init(&Device::default());
    assert_eq!(fno.channel_mlp[0].num_params(), 27);
}

/// Without a 2.0.0 block option the activation is applied once (the original block, CO1.1's
/// GELU included); for ReLU a second application would change nothing, bit for bit.
#[test]
fn skipped_relu_repeat_is_relu_twice() {
    let device = Device::default();
    let fno: FNO<4> = config().init(&device);
    assert!(!fno.repeat_activation);
    let x = Tensor::<4>::random([2, 6, 8, 8], Distribution::Default, &device);
    let once = fno.fourier_layers(x.clone(), None);
    let mut twice = fno.clone();
    twice.repeat_activation = true;
    assert_eq!(once.into_data(), twice.fourier_layers(x, None).into_data());
}
