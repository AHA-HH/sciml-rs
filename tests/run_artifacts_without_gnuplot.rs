//! REVIEW.md 5.2: a missing gnuplot must not abort `write_run_artifacts`,
//! which the training examples call before saving the weights.
//!
//! Own test binary (own process) because it clears `PATH`, which is global.

use sciml_rs::neural_operators::{
    data::loaders::{base_dataset::DatasetConfig, burgers::BurgersConfig},
    metrics::io::write_run_artifacts,
    models::fno::FNOConfig,
    training::{metrics::EpochMetrics, trainer::TrainingConfig},
};

#[test]
fn run_artifacts_survive_missing_gnuplot() {
    // SAFETY: this binary has a single test, so no other thread reads the env.
    unsafe { std::env::set_var("PATH", "") };

    let metrics = vec![EpochMetrics {
        epoch: 0,
        train_mse: 0.5,
        train_l2: 0.3,
        test_l2: 0.4,
        current_lr: 1e-3,
    }];
    let dir = write_run_artifacts(
        "test_no_gnuplot",
        &metrics,
        &FNOConfig::new(vec![4], 1, 1),
        &TrainingConfig::new(),
        &BurgersConfig::new(
            DatasetConfig {
                n_train: 1,
                n_test: 1,
            },
            32,
        ),
    )
    .expect("a missing gnuplot must only skip the plot");

    let written = [
        "metrics.csv",
        "model_cfg.json",
        "train_cfg.json",
        "data_cfg.json",
    ]
    .map(|f| dir.join(f).exists());
    let plotted = dir.join("plots.png").exists();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(written, [true; 4], "artifacts missing in {}", dir.display());
    assert!(!plotted, "no gnuplot, so there should be no plot");
}
