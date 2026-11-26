//! The lifting, projection and embedding options (library CM1.4): what they build. Parity
//! with 2.0.0's `FNO` is `tests/parity_embeddings.rs`.

use super::*;

/// 3 data channels, 2D, width 5.
fn config() -> FNOConfig {
    FNOConfig::new(vec![2, 2], 3, 1)
        .with_hidden_channels(5)
        .with_n_layers(1)
}

/// `[in, out]` of each lifting and projection layer.
fn widths(fno: &FNO<4>) -> Vec<[usize; 2]> {
    let layers = [&fno.fc0].into_iter().chain(&fno.lifting);
    layers
        .chain([&fno.fc1, &fno.fc2])
        .map(|l| l.weight.dims())
        .collect()
}

#[test]
fn absent_options_build_today_s_lifting_and_projection() {
    let fno: FNO<4> = config().init(&Device::default());
    assert_eq!(widths(&fno), vec![[5, 5], [5, 128], [128, 1]]);
    assert!(fno.positional_embedding.is_none());
}

/// 2.0.0's `int(r · W)`: 1.5 · 5 = 7.5 → 7; 0.1 · 5 = 0.5 → 0, one lifting layer.
#[test]
fn ratios_truncate_as_python_int() {
    let fno: FNO<4> = config()
        .with_lifting_channel_ratio(Some(1.5))
        .with_projection_channel_ratio(Some(1.5))
        .init(&Device::default());
    assert_eq!(widths(&fno), vec![[5, 7], [7, 5], [5, 7], [7, 1]]);
    let fno: FNO<4> = config()
        .with_lifting_channel_ratio(Some(0.1))
        .init(&Device::default());
    assert_eq!(widths(&fno), vec![[5, 5], [5, 128], [128, 1]]);
}

#[test]
#[should_panic(expected = "width of 0")]
fn zero_projection_width_panics() {
    config()
        .with_projection_channel_ratio(Some(0.1))
        .init::<4>(&Device::default());
}

/// Without an embedding `fc0` takes the data channels alone, and the forward accepts them.
#[test]
fn disabled_embedding_lifts_data_channels_only() {
    let device = Device::default();
    let fno: FNO<4> = config()
        .with_positional_embedding(Some(PositionalEmbedding::Disabled))
        .init(&device);
    assert_eq!(fno.fc0.weight.dims(), [3, 5]);
    let x = Tensor::<4>::random([2, 6, 5, 3], Distribution::Default, &device);
    assert_eq!(fno.forward(x).dims(), [2, 6, 5, 1]);
}

#[test]
#[should_panic(expected = "the grid needs 2 axes")]
fn grid_of_the_wrong_rank_panics() {
    config()
        .with_positional_embedding(Some(PositionalEmbedding::GridNd(GridEmbeddingNd::unit(3))))
        .init::<4>(&Device::default());
}
