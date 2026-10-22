//! A missing gnuplot must not abort `RunDir::save_plot`, which the training examples
//! call before saving the weights.
//!
//! Own test binary (own process) because it clears `PATH`, which is global.

use sciml_rs::neural_operators::{
    data::loaders::{base_dataset::DatasetConfig, burgers::BurgersConfig},
    models::fno::FNOConfig,
    runs::{EpochMetrics, RunDir},
    training::config::TrainingConfig,
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
    let root = std::env::temp_dir().join(format!("sciml_rs_no_gnuplot_{}", std::process::id()));
    let run = RunDir::create(&root, "test_no_gnuplot").expect("create the run directory");
    run.save_configs(
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
    .expect("save the configs");
    run.append_metrics(&metrics[0])
        .expect("append the metrics row");
    // must return without panicking: a missing gnuplot only skips the plot
    run.save_plot(&metrics);

    let dir = run.path();
    let written = [
        "metrics.csv",
        "model_cfg.json",
        "train_cfg.json",
        "data_cfg.json",
    ]
    .map(|f| dir.join(f).exists());
    let plotted = dir.join("plots.png").exists();
    std::fs::remove_dir_all(&root).ok();
    assert_eq!(written, [true; 4], "artifacts missing in {}", dir.display());
    assert!(!plotted, "no gnuplot, so there should be no plot");
}
