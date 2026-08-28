//! FNO experiment on 1D Burgers dataset

use crate::neural_operators::{
    data::{
        dataset::{BurgersConfig, load_burgers_uniform},
        pipeline::DatasetConfig,
        visual::{write_metrics_csv_nd, plot_metrics_nd},
    },
    train_fno::{FNOConfig, train_burgers_fno},
};
use burn::tensor::Device;

pub fn run_burgers_fno() {
    let device = Device::default().autodiff();

    let dataset_config = BurgersConfig::new(
        DatasetConfig { n_train: 1024, n_test: 100 },
        32, // 4
    );

    let (train_data, test_data) = load_burgers_uniform(
        "/Users/aneeshussain/Code/Datasets/burgers_data_R10.mat",
        // "/Users/aneeshussain/Code/Datasets/burgers_uniform_jens.mat",
        &dataset_config,
    );

    let config = FNOConfig {
        modes: vec![16], // was modes1: 16 // 128
        width: 64, // 32
        data_channels: 1,  // Burgers: scalar field — see note below, this is inferred, not confirmed
        out_channels: 1,
        n_layers: 4,
        epochs: 500,
        batch_size: 20, // 4
        learning_rate: 1e-3,
        weight_decay: 1e-4,
        min_lr: 1e-5,
        n_train: 1024,
        n_test: 100,
        seed: 42,
    };

    let (_model, metrics) = train_burgers_fno(train_data, test_data, &config, &device);

    let timestamp = std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH)
    .unwrap()
    .as_secs();
    let csv_path = format!("burgers_1nd_fno_{}.csv", timestamp);
    let png_path = format!("burgers_1nd_fno_{}.png", timestamp);
    write_metrics_csv_nd(&csv_path, &metrics);
    plot_metrics_nd(&metrics, &png_path);
    println!("csv written, png saved");

    let final_test_l2 = metrics.last().unwrap().test_l2;
    println!("final test_l2: {:.6}", final_test_l2);
    // FNO paper benchmark
    println!("z.li l2: 0.0149");
}
