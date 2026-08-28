//! FNO experiment on 2D Darcy Flow dataset

use crate::neural_operators::{
    data::{
        dataset::{DarcyFlowConfig, load_darcy_flow_uniform},
        pipeline::DatasetConfig,
        visual::{write_metrics_csv_nd, plot_metrics_nd},
    },
    train_fno::{FNOConfig, train_darcy_flow_fno},
};
use burn::tensor::Device;

pub fn run_darcy_flow_fno() {
    let device = Device::default().autodiff();

    let dataset_config = DarcyFlowConfig::new(
        DatasetConfig { n_train: 1000, n_test: 100 },
        28, // subsample_rate -> s=16, matches the Python reference run
    );

    let (train_data, test_data, y_normalizer) = load_darcy_flow_uniform(
        "/Users/aneeshussain/Code/Datasets/piececonst_r421_N1024_smooth1.mat",
        "/Users/aneeshussain/Code/Datasets/piececonst_r421_N1024_smooth2.mat",
        &dataset_config,
    );

    let config = FNOConfig {
        modes: vec![4, 4], 
        width: 32, 
        data_channels: 1, 
        out_channels: 1,
        n_layers: 4,
        epochs: 3, // 500
        batch_size: 20, 
        learning_rate: 1e-3,
        weight_decay: 1e-4,
        min_lr: 1e-5,
        n_train: 1000,
        n_test: 100,
        seed: 42,
    };

    let (_model, metrics) = train_darcy_flow_fno(train_data, test_data, &y_normalizer, &config, &device);

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let csv_path = format!("darcy_2nd_fno_{}.csv", timestamp);
    let png_path = format!("darcy_2nd_fno_{}.png", timestamp);
    write_metrics_csv_nd(&csv_path, &metrics);
    plot_metrics_nd(&metrics, &png_path);
    println!("csv written, png saved");

    let final_test_l2 = metrics.last().unwrap().test_l2;
    println!("final test_l2: {:.6}", final_test_l2);
    // Python reference run (this project, s=16, modes=4, cosine LR, epoch 499)
    println!("python reference test_l2: 0.0345");
}