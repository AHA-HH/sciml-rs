//! Optimisers CO3.1, constraint 1: one hash of the bits of a seeded `FNO<4>` forward, two
//! Adam epochs of `train` and a three-epoch hybrid Gauss–Newton run, on tiny 2D
//! datasets. Built with and without Burn's `linalg` feature, the hashes must be equal (see
//! `../../REPORT.md`). Uses only API that exists on `main` before CO3.1.

use burn::{
    Tensor,
    store::{ModuleSnapshot, bridge::to_data},
    tensor::{Device, Distribution},
};
use ndarray::{ArrayD, IxDyn};
use sciml_rs::neural_operators::{
    data::dataset::OperatorDataset,
    models::fno::{FNO, FNOConfig},
    optim::gauss_newton::GaussNewtonConfig,
    runs::{EpochMetrics, RunDir},
    training::{
        config::{OptimizerConfig, TrainingConfig},
        trainer::train,
    },
};

/// FNV-1a over 32-bit words.
struct Fnv(u64);
impl Fnv {
    fn add(&mut self, words: impl IntoIterator<Item = u32>) {
        for w in words {
            for b in w.to_le_bytes() {
                self.0 = (self.0 ^ b as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
    }
    fn f32s(&mut self, xs: &[f32]) {
        self.add(xs.iter().map(|x| x.to_bits()));
    }
}

/// `n` Darcy-shaped samples, as `training::fixtures::tiny_dataset_2d`.
fn dataset(n: usize) -> OperatorDataset<f64> {
    let inputs = ArrayD::from_shape_fn(IxDyn(&[n, 4, 4, 1]), |i| {
        ((i[0] * 16 + i[1] * 4 + i[2]) as f64 * 0.3).sin()
    });
    let targets = ArrayD::from_shape_fn(IxDyn(&[n, 4, 4]), |i| {
        2.0 + ((i[0] * 16 + i[1] * 4 + i[2]) as f64 * 0.7).cos()
    });
    OperatorDataset::new(inputs, targets)
}

fn weights(h: &mut Fnv, model: &FNO<4>) {
    let mut ps: Vec<_> = model
        .collect(None, None, false)
        .iter()
        .map(|t| {
            (
                t.name.clone(),
                to_data(t).unwrap().try_to_vec::<f32>().unwrap(),
            )
        })
        .collect();
    ps.sort_by(|a, b| a.0.cmp(&b.0));
    ps.iter().for_each(|(_, v)| h.f32s(v));
}

fn metrics(h: &mut Fnv, ms: &[EpochMetrics]) {
    for m in ms {
        h.add([
            m.train_mse.to_bits(),
            m.train_l2.to_bits(),
            m.test_l2.to_bits(),
        ]);
        let lr = m.current_lr.to_bits();
        h.add([lr as u32, (lr >> 32) as u32]);
    }
}

fn main() {
    let model_cfg = FNOConfig::new(vec![2, 2], 1, 1)
        .with_hidden_channels(4)
        .with_n_layers(2);
    let root = std::env::temp_dir().join(format!("linalg_spike_{}", std::process::id()));
    let mut h = Fnv(0xcbf2_9ce4_8422_2325);

    // forward
    let device = Device::default();
    device.seed(7);
    let model: FNO<4> = model_cfg.init(&device);
    let x = Tensor::<4>::random([2, 4, 4, 1], Distribution::Default, &device);
    let y = model.forward(x).into_data().try_to_vec::<f32>().unwrap();
    h.f32s(&y);
    println!("forward     {:016x}", h.0);

    let device = Device::default().autodiff();
    let base = TrainingConfig::new()
        .with_batch_size(2)
        .with_test_batch_size(2);
    let gn = GaussNewtonConfig::new().with_cg_max_iters(5);
    let arms = [
        ("adam", base.clone().with_epochs(2)),
        (
            "hybrid_gn",
            base.with_epochs(3)
                .with_optimizer(Some(OptimizerConfig::GaussNewton(gn)))
                .with_adam_epochs(Some(1)),
        ),
    ];
    for (name, cfg) in arms {
        let run = RunDir::create(&root, name).unwrap();
        let (model, ms) = train::<FNO<4>, 4, 3, _>(
            dataset(5),
            dataset(3),
            None,
            &model_cfg,
            &cfg,
            &device,
            &run,
        );
        weights(&mut h, &model);
        metrics(&mut h, &ms);
        println!("{name:<11} {:016x}", h.0);
    }
    let _ = std::fs::remove_dir_all(&root);
    println!("fingerprint {:016x}", h.0);
}
