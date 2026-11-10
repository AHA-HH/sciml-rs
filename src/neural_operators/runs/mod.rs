//! Run directories: the one owner of what a training run writes and a predict example
//! reads.
//!
//! A run directory `<root>/<name>_<unix_secs>/` holds:
//! - `model_cfg.json`, `train_cfg.json`, `data_cfg.json`: the three configs;
//! - `metrics.csv` (one row per epoch) and `plots.png` (only if gnuplot is installed);
//! - `x_normalizer.json`, `y_normalizer.json`: Darcy and Poisson runs only;
//! - `model_weights.bpk`: the model;
//! - `costs.json` ([`RunCosts`]): Poisson runs since Phase 5 T4;
//! - `checkpoint.bin` ([`Checkpoint`]): the latest checkpoint, Poisson runs since Phase 9
//!   T1, replaced every k epochs and after the last;
//! - `gn_steps.csv` ([`GnStepRecord`]): one row per Gauss–Newton step, Gauss–Newton runs
//!   only (Phase 10 T3).
//!
//! [`RunDir`] writes these and reads them back. File names live only here. Runs written
//! since Phase 3 open, including Learner runs from before Phase 4s T2, which lack
//! `data_cfg.json` and `metrics.csv`: [`RunDir::open`] requires only the directory, and
//! each reader fails when its own file is missing (or returns `None`, for the normalizers
//! and the costs).
//!
//! Errors: writers return [`io::Result`]; readers that parse a file (configs,
//! normalizers, weights) return `Box<dyn Error>`, since Burn's config, record and store
//! errors have no common type.

mod checkpoint;
mod costs;
mod csv;
mod gn_steps;
mod plots;

pub use checkpoint::{Checkpoint, CheckpointState};
pub use costs::RunCosts;
pub use gn_steps::{GnPasses, GnStepRecord};

use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use burn::{
    config::Config,
    optim::{ModuleOptimizer, lr_scheduler::module_lr_scheduler::ModuleLrScheduler},
    store::{BurnpackStore, ModuleSnapshot},
    tensor::Device,
};

use crate::neural_operators::{
    data::{
        loaders::common::Normalizers,
        transforms::normalizers::{NormalizerRecord, UnitGaussianNormalizer},
    },
    metrics::poisson::ERRORS_JSON,
    models::fno::{FNO, FNOConfig},
    optim::gauss_newton::GaussNewtonState,
    training::config::TrainingConfig,
};

const MODEL_CFG: &str = "model_cfg.json";
const TRAIN_CFG: &str = "train_cfg.json";
const DATA_CFG: &str = "data_cfg.json";
const METRICS_CSV: &str = "metrics.csv";
const GN_STEPS_CSV: &str = "gn_steps.csv";
const PLOTS_PNG: &str = "plots.png";
const X_NORMALIZER: &str = "x_normalizer.json";
const Y_NORMALIZER: &str = "y_normalizer.json";
const COSTS: &str = "costs.json";
const CHECKPOINT: &str = "checkpoint.bin";
/// Written first, then renamed over `checkpoint.bin`, so a crash mid-write leaves the
/// previous checkpoint intact.
const CHECKPOINT_TMP: &str = "checkpoint.bin.tmp";
/// Burn's `BurnpackStore` appends `.bpk` to an extensionless path on save and load
/// alike, so both use this stem and the file is `model_weights.bpk`.
const WEIGHTS_STEM: &str = "model_weights";

/// Snapshot of training/eval state after one epoch; one row of `metrics.csv`.
///
/// `train_mse` is averaged over batches (the MSE is already a per-batch mean);
/// `train_l2` and `test_l2` are averaged over samples (LpLoss sums per batch).
/// `current_lr` is the rate after the epoch's final step.
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Mean training MSE over the epoch's batches.
    pub train_mse: f32,
    /// Mean training relative L² per sample.
    pub train_l2: f32,
    /// Mean test relative L² per sample.
    pub test_l2: f32,
    /// Learning rate after the epoch's final step. For a Gauss–Newton run (Phase 10 T3), which has no learning rate, the damping λ the
    /// next step will use.
    pub current_lr: f64,
    /// Wall-clock seconds of the epoch, training and evaluation; `None` in `metrics.csv`
    /// files written before Phase 5 T4, which have no such column.
    pub epoch_time_s: Option<f64>,
}

/// A training run's directory. See the [module docs](self) for its files.
#[derive(Debug, Clone)]
pub struct RunDir {
    path: PathBuf,
    name: String,
}

impl RunDir {
    /// Creates `root/<name>_<unix_secs>/` (and `root` if needed). `name` is also the
    /// title of the metrics plot.
    ///
    /// Examples pass `Path::new(env!("CARGO_MANIFEST_DIR")).join("runs")` as `root`.
    ///
    /// # Errors
    /// If the clock is before 1970 or the directory can't be created.
    pub fn create(root: &Path, name: &str) -> io::Result<Self> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_secs();
        let path = root.join(format!("{name}_{stamp}"));
        std::fs::create_dir_all(&path)?;
        Ok(Self {
            path,
            name: name.to_string(),
        })
    }

    /// Opens an existing run directory. Requires only that `dir` is a directory; each
    /// reader checks its own file.
    ///
    /// # Errors
    /// [`io::ErrorKind::NotFound`] if `dir` is not a directory.
    pub fn open(dir: impl Into<PathBuf>) -> io::Result<Self> {
        let path = dir.into();
        if !path.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("no run directory at {}", path.display()),
            ));
        }
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        Ok(Self { path, name })
    }

    /// The run directory itself.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Writes `model_cfg.json`, `train_cfg.json` and `data_cfg.json`.
    ///
    /// # Errors
    /// If a file can't be written.
    pub fn save_configs(
        &self,
        model: &impl Config,
        train: &impl Config,
        data: &impl Config,
    ) -> io::Result<()> {
        model.save(self.path.join(MODEL_CFG))?;
        train.save(self.path.join(TRAIN_CFG))?;
        data.save(self.path.join(DATA_CFG))
    }

    /// Appends one epoch's row to `metrics.csv`, writing the header first if the file is
    /// new: the file's only writer. Training calls this after every epoch (`train_fno`), so
    /// a run that stops early keeps the rows of its finished epochs.
    ///
    /// # Errors
    /// If the file can't be opened or written.
    pub fn append_metrics(&self, m: &EpochMetrics) -> io::Result<()> {
        csv::append_metrics_row(self.path.join(METRICS_CSV), m)
    }

    /// Drops the rows of `metrics.csv` for epochs `next_epoch` and later, keeping the
    /// others byte for byte: a resumed run calls this before training on from its
    /// checkpoint, so the epochs after it are not listed twice.
    ///
    /// # Errors
    /// If the file can't be read or written, or a row's epoch doesn't parse
    /// ([`io::ErrorKind::InvalidData`]).
    pub fn truncate_metrics(&self, next_epoch: usize) -> io::Result<()> {
        csv::truncate_metrics_csv(self.path.join(METRICS_CSV), next_epoch)
    }

    /// Appends one Gauss–Newton step's row to `gn_steps.csv`, writing the header first if
    /// the file is new: the file's only writer.
    ///
    /// # Errors
    /// If the file can't be opened or written.
    pub fn append_gn_step(&self, record: &GnStepRecord) -> io::Result<()> {
        gn_steps::append_gn_step_row(self.path.join(GN_STEPS_CSV), record)
    }

    /// Drops the rows of `gn_steps.csv` for epochs `next_epoch` and later, as
    /// [`truncate_metrics`](Self::truncate_metrics) does for `metrics.csv`; nothing to do
    /// for a run without the file (an Adam run).
    ///
    /// # Errors
    /// As [`truncate_metrics`](Self::truncate_metrics).
    pub fn truncate_gn_steps(&self, next_epoch: usize) -> io::Result<()> {
        let path = self.path.join(GN_STEPS_CSV);
        if !path.exists() {
            return Ok(());
        }
        // its first column is the epoch, as in metrics.csv
        csv::truncate_metrics_csv(path, next_epoch)
    }

    /// The forward and backward passes summed over `gn_steps.csv`, and its row count (the
    /// steps); `None` for a run without the file (an Adam run).
    ///
    /// # Errors
    /// If the file can't be read, lacks the pass columns or has a field that doesn't
    /// parse ([`io::ErrorKind::InvalidData`]).
    pub fn gn_passes(&self) -> io::Result<Option<GnPasses>> {
        let path = self.path.join(GN_STEPS_CSV);
        if !path.exists() {
            return Ok(None);
        }
        gn_steps::sum_passes(path).map(Some)
    }

    /// Draws `plots.png` (loss curves and learning rate) from `metrics`, titled with the
    /// run's name: an end-of-run action. A missing gnuplot only skips the plot, with a
    /// warning on stderr.
    pub fn save_plot(&self, metrics: &[EpochMetrics]) {
        if let Err(e) = plots::plot_metrics(self.path.join(PLOTS_PNG), metrics, &self.name) {
            eprintln!("warning: skipping {PLOTS_PNG} ({e})");
        }
    }

    /// Writes `x_normalizer.json` and `y_normalizer.json`.
    ///
    /// # Errors
    /// If a file can't be written.
    pub fn save_normalizers(&self, normalizers: &Normalizers) -> io::Result<()> {
        normalizers
            .x
            .to_record()
            .save(self.path.join(X_NORMALIZER))?;
        normalizers.y.to_record().save(self.path.join(Y_NORMALIZER))
    }

    /// Writes the model's weights to `model_weights.bpk`.
    ///
    /// # Errors
    /// If the file can't be written.
    pub fn save_weights<const R: usize>(&self, model: &FNO<R>) -> Result<(), Box<dyn Error>> {
        model.save_into(&mut BurnpackStore::from_file(self.path.join(WEIGHTS_STEM)))?;
        Ok(())
    }

    /// Loads the run's model: the architecture from `model_cfg.json`, the weights from
    /// `model_weights.bpk`, on `device`. Pass a plain (non-autodiff) device for inference,
    /// so forward passes record no backward graph.
    ///
    /// # Errors
    /// If `model_cfg.json` can't be read or parsed, or `model_weights.bpk` can't be read
    /// or doesn't fit the architecture.
    pub fn load_fno<const R: usize>(&self, device: &Device) -> Result<FNO<R>, Box<dyn Error>> {
        let mut model = self.model_config()?.init::<R>(device);
        model.load_from(&mut BurnpackStore::from_file(self.path.join(WEIGHTS_STEM)))?;
        Ok(model)
    }

    /// The model's architecture, from `model_cfg.json`.
    ///
    /// # Errors
    /// If the file can't be read or parsed.
    pub fn model_config(&self) -> Result<FNOConfig, Box<dyn Error>> {
        self.load_config(MODEL_CFG)
    }

    /// The training config, from `train_cfg.json`.
    ///
    /// # Errors
    /// If the file can't be read or parsed.
    pub fn training_config(&self) -> Result<TrainingConfig, Box<dyn Error>> {
        self.load_config(TRAIN_CFG)
    }

    /// The dataset config, from `data_cfg.json`, as the PDE's config type `C` (for example
    /// `DarcyConfig`).
    ///
    /// # Errors
    /// If the file can't be read or doesn't parse as `C`.
    pub fn data_config<C: Config>(&self) -> Result<C, Box<dyn Error>> {
        self.load_config(DATA_CFG)
    }

    fn load_config<C: Config>(&self, file: &str) -> Result<C, Box<dyn Error>> {
        let path = self.path.join(file);
        C::load(&path).map_err(|e| format!("{}: {e}", path.display()).into())
    }

    /// The run's normalizers, or `None` for a run that has none (Burgers).
    ///
    /// # Errors
    /// If `y_normalizer.json` exists but either normalizer file can't be read or parsed.
    pub fn normalizers(&self) -> Result<Option<Normalizers>, Box<dyn Error>> {
        if !self.path.join(Y_NORMALIZER).exists() {
            return Ok(None);
        }
        let load = |name: &str| -> Result<UnitGaussianNormalizer, Box<dyn Error>> {
            let path = self.path.join(name);
            let record =
                NormalizerRecord::load(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(UnitGaussianNormalizer::from_record(&record))
        };
        Ok(Some(Normalizers {
            x: load(X_NORMALIZER)?,
            y: load(Y_NORMALIZER)?,
        }))
    }

    /// Writes `costs.json`, replacing it if present.
    ///
    /// # Errors
    /// If the file can't be written.
    pub fn save_costs(&self, costs: &RunCosts) -> io::Result<()> {
        costs.save(self.path.join(COSTS))
    }

    /// The run's costs, or `None` for a run without `costs.json` (any run before Phase 5
    /// T4).
    ///
    /// # Errors
    /// If `costs.json` exists but can't be read or parsed.
    pub fn costs(&self) -> Result<Option<RunCosts>, Box<dyn Error>> {
        if !self.path.join(COSTS).exists() {
            return Ok(None);
        }
        self.load_config(COSTS).map(Some)
    }

    /// Reads `metrics.csv`, one [`EpochMetrics`] per row.
    ///
    /// # Errors
    /// I/O errors from opening or reading the file, and [`io::ErrorKind::InvalidData`]
    /// naming the line for a row with missing or unparsable fields.
    pub fn metrics(&self) -> io::Result<Vec<EpochMetrics>> {
        csv::read_metrics_csv(self.path.join(METRICS_CSV))
    }

    /// The run's files worth viewing beside its metrics, those present, sorted by name:
    /// the three configs, `costs.json`, `errors.json` ([`ERRORS_JSON`]), `gn_steps.csv`
    /// (Gauss–Newton runs) and every `.png`
    /// (`plots.png`, `prediction_<k>.png`). Not the weights, checkpoint, normalizers or
    /// `metrics.csv`, and no symbolic link, so nothing outside the directory is picked up.
    /// The MLflow exporter (Phase 8 T1) uploads these.
    ///
    /// # Errors
    /// If the directory can't be listed.
    pub fn artifacts(&self) -> io::Result<Vec<PathBuf>> {
        let named = [
            MODEL_CFG,
            TRAIN_CFG,
            DATA_CFG,
            COSTS,
            ERRORS_JSON,
            GN_STEPS_CSV,
        ];
        let mut files = Vec::new();
        for entry in std::fs::read_dir(&self.path)? {
            let entry = entry?;
            // `DirEntry::file_type` does not follow symbolic links, so links are skipped
            let is_file = entry.file_type()?.is_file();
            let path = entry.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if is_file && (named.contains(&&*name) || name.ends_with(".png")) {
                files.push(path);
            }
        }
        files.sort();
        Ok(files)
    }

    /// Writes `checkpoint.bin`: `state`, the model's weights and the optimiser's and
    /// scheduler's records, replacing any earlier checkpoint. Atomic: the file is written
    /// and synced under a temporary name, then renamed.
    ///
    /// # Errors
    /// If the model or a record can't be serialised, or the file can't be written.
    pub fn save_checkpoint<const R: usize>(
        &self,
        state: &CheckpointState,
        model: &FNO<R>,
        optim: &ModuleOptimizer,
        scheduler: &ModuleLrScheduler,
    ) -> Result<(), Box<dyn Error>> {
        self.save_checkpoint_gn(state, model, optim, scheduler, None)
    }

    /// [`save_checkpoint`](Self::save_checkpoint) with the Gauss–Newton state of a
    /// Gauss–Newton run, if `gn` is given, as the checkpoint's optional fourth section.
    ///
    /// # Errors
    /// As [`save_checkpoint`](Self::save_checkpoint).
    pub fn save_checkpoint_gn<const R: usize>(
        &self,
        state: &CheckpointState,
        model: &FNO<R>,
        optim: &ModuleOptimizer,
        scheduler: &ModuleLrScheduler,
        gn: Option<&GaussNewtonState>,
    ) -> Result<(), Box<dyn Error>> {
        let bytes = checkpoint::encode(state, model, optim.to_record(), scheduler.to_record(), gn)?;
        let tmp = self.path.join(CHECKPOINT_TMP);
        let mut file = std::fs::File::create(&tmp)?;
        io::Write::write_all(&mut file, &bytes)?;
        file.sync_all()?;
        std::fs::rename(&tmp, self.path.join(CHECKPOINT))?;
        Ok(())
    }

    /// The run's checkpoint, with the model built from `model_cfg.json` on `device`, or
    /// `None` for a run without `checkpoint.bin`. Pass the training (autodiff) device to
    /// resume.
    ///
    /// # Errors
    /// If `checkpoint.bin` exists but can't be read, is truncated or malformed, or
    /// doesn't fit the architecture in `model_cfg.json`.
    pub fn checkpoint<const R: usize>(
        &self,
        device: &Device,
    ) -> Result<Option<Checkpoint<R>>, Box<dyn Error>> {
        let path = self.path.join(CHECKPOINT);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path)?;
        checkpoint::decode(&bytes, &self.model_config()?, device)
            .map(Some)
            .map_err(|e| format!("{}: {e}", path.display()).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::{
        data::loaders::{base_dataset::DatasetConfig, darcy::DarcyConfig},
        models::predict::predict_host,
    };
    use ndarray::{Array4, ArrayD, IxDyn};

    fn model_config() -> FNOConfig {
        FNOConfig {
            modes: vec![3, 3],
            hidden_channels: 4,
            data_channels: 1,
            out_channels: 1,
            n_layers: 1,
            spectral_init: None,
            padding: Some(2),
        }
    }

    fn data_config() -> DarcyConfig {
        DarcyConfig::new(
            DatasetConfig {
                n_train: 2,
                n_test: 1,
            },
            5,
        )
    }

    /// A fresh, empty root under the system temp dir, unique to `test` and this process.
    fn fresh_root(test: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("sciml_rs_runs_{test}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    fn metrics() -> Vec<EpochMetrics> {
        vec![
            EpochMetrics {
                epoch: 0,
                train_mse: 0.5,
                train_l2: 0.3,
                test_l2: 0.4,
                current_lr: 1e-3,
                epoch_time_s: Some(2.5),
            },
            EpochMetrics {
                epoch: 1,
                train_mse: 0.25,
                train_l2: 0.15,
                test_l2: 0.2,
                current_lr: 5e-4,
                epoch_time_s: Some(2.25),
            },
        ]
    }

    #[test]
    fn save_open_load_fno_round_trips_bit_identically() {
        let device = Device::default();
        let root = fresh_root("round_trip");
        let run = RunDir::create(&root, "test_fno").unwrap();
        assert!(run.path().starts_with(&root));
        let model: FNO<4> = model_config().init::<4>(&device);
        run.save_configs(&model_config(), &TrainingConfig::new(), &data_config())
            .unwrap();
        run.save_weights(&model).unwrap();
        assert!(run.path().join("model_weights.bpk").exists());

        let opened = RunDir::open(run.path()).unwrap();
        let loaded: FNO<4> = opened.load_fno(&device).unwrap();
        let x = Array4::from_shape_fn((2, 8, 8, 1), |(k, i, j, _)| {
            ((k * 64 + i * 8 + j) as f32 * 0.37).sin()
        });
        let saved_out = predict_host(&model, x.view().into_dyn(), 2, &device);
        let loaded_out = predict_host(&loaded, x.view().into_dyn(), 2, &device);
        assert_eq!(saved_out, loaded_out);

        let data: DarcyConfig = opened.data_config().unwrap();
        assert_eq!(data.s(), data_config().s());
        assert_eq!(opened.model_config().unwrap().modes, model_config().modes);
        assert_eq!(
            opened.training_config().unwrap().seed,
            TrainingConfig::new().seed
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn metrics_and_normalizers_round_trip() {
        let root = fresh_root("metrics_normalizers");
        let run = RunDir::create(&root, "test_fno").unwrap();
        assert!(
            run.normalizers().unwrap().is_none(),
            "no normalizer files yet"
        );

        for m in &metrics() {
            run.append_metrics(m).unwrap();
        }
        let read = run.metrics().unwrap();
        assert_eq!(read.len(), 2);
        assert_eq!(read[1].epoch, 1);
        assert_eq!(read[0].train_mse, 0.5);
        assert_eq!(read[1].current_lr, 5e-4);
        assert_eq!(read[1].epoch_time_s, Some(2.25));

        let field = |seed: f64| {
            ArrayD::from_shape_fn(IxDyn(&[3, 2, 2]), |i| {
                (i[0] * 4 + i[1] * 2 + i[2]) as f64 + seed
            })
        };
        let saved = Normalizers {
            x: UnitGaussianNormalizer::fit(&field(0.0)),
            y: UnitGaussianNormalizer::fit(&field(1.0).mapv(|v| v * v)),
        };
        run.save_normalizers(&saved).unwrap();
        let loaded = run.normalizers().unwrap().expect("normalizers were saved");
        assert_eq!(loaded.x.mean(), saved.x.mean());
        assert_eq!(loaded.x.std(), saved.x.std());
        assert_eq!(loaded.y.mean(), saved.y.mean());
        assert_eq!(loaded.y.std(), saved.y.std());

        // predict needs both: a missing x normalizer names the run and the file
        std::fs::remove_file(run.path().join(X_NORMALIZER)).unwrap();
        let err = run
            .normalizers()
            .err()
            .expect("x_normalizer.json is missing");
        let msg = err.to_string();
        assert!(msg.contains(&run.path().display().to_string()), "{msg}");
        assert!(msg.contains("x_normalizer.json"), "{msg}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn costs_round_trip() {
        let root = fresh_root("costs");
        let run = RunDir::create(&root, "test_fno").unwrap();
        assert!(run.costs().unwrap().is_none(), "no costs.json yet");

        let mut costs = RunCosts {
            dataset_train_wall_time_s: Some(1.1),
            dataset_test_wall_time_s: Some(0.2),
            load_time_s: 3.5,
            training_time_s: 120.25,
            inference_time_s: None,
            host_to_device_bytes_train: 1000 * 64 * 64 * 2 * 4,
            host_to_device_bytes_test: 200 * 64 * 64 * 2 * 4,
            peak_gpu_memory: "unavailable".to_string(),
            backend_features: vec!["flex".to_string()],
            device: "Flex(Cpu)".to_string(),
            git_commit: "abc123".to_string(),
            num_params: 1_188_353,
            optimisation_steps: Some(500 * 50),
            resumes: Some(1),
            forward_passes: Some(500 * 50),
            backward_passes: Some(500 * 50),
        };
        run.save_costs(&costs).unwrap();
        // predict completes the file written by training
        let mut read = run.costs().unwrap().expect("costs were saved");
        assert_eq!(read.inference_time_s, None);
        read.inference_time_s = Some(0.75);
        run.save_costs(&read).unwrap();
        costs.inference_time_s = Some(0.75);
        let read = run.costs().unwrap().expect("costs were saved");
        assert_eq!(
            burn::config::config_to_json(&read),
            burn::config::config_to_json(&costs)
        );

        // a costs.json as Phase 5 runs wrote it, before optimisation_steps
        let phase5 = r#"{"dataset_train_wall_time_s": 1.1, "dataset_test_wall_time_s": 0.2,
            "load_time_s": 3.5, "training_time_s": 120.25, "inference_time_s": null,
            "host_to_device_bytes_train": 8, "host_to_device_bytes_test": 4,
            "peak_gpu_memory": "unavailable", "backend_features": ["cuda"],
            "device": "Cuda(0)", "git_commit": "abc123", "num_params": 2368001}"#;
        std::fs::write(run.path().join(COSTS), phase5).unwrap();
        let read = run.costs().unwrap().expect("costs were saved");
        assert_eq!(read.optimisation_steps, None);
        assert_eq!(read.resumes, None);
        assert_eq!((read.forward_passes, read.backward_passes), (None, None));

        std::fs::write(run.path().join(COSTS), "{").unwrap();
        let msg = run.costs().expect_err("a broken costs.json").to_string();
        assert!(msg.contains("costs.json"), "{msg}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// No `checkpoint.bin` reads as `None`; a truncated, overlong or foreign one is an
    /// error naming the file, not a panic.
    #[test]
    fn missing_or_broken_checkpoint() {
        use burn::{lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig, optim::AdamConfig};
        let device = Device::default();
        let root = fresh_root("checkpoint");
        let run = RunDir::create(&root, "test_fno").unwrap();
        run.save_configs(&model_config(), &TrainingConfig::new(), &data_config())
            .unwrap();
        assert!(run.checkpoint::<4>(&device).unwrap().is_none());

        let model: FNO<4> = model_config().init::<4>(&device);
        let optim: ModuleOptimizer = AdamConfig::new().init();
        let scheduler = CosineAnnealingLrSchedulerConfig::new(1e-3, 10)
            .init()
            .unwrap();
        let state = CheckpointState {
            next_epoch: 25,
            resumes: 0,
            training_time_s: 12.5,
        };
        run.save_checkpoint(&state, &model, &optim, &scheduler)
            .unwrap();
        assert!(!run.path().join(CHECKPOINT_TMP).exists());
        let read = run.checkpoint::<4>(&device).unwrap().expect("saved");
        assert_eq!(read.state, state);

        let path = run.path().join(CHECKPOINT);
        let bytes = std::fs::read(&path).unwrap();
        let longer = [bytes.as_slice(), b"x"].concat();
        for broken in [
            &bytes[..bytes.len() / 2],
            &bytes[..12],
            &longer[..],
            &b"not a checkpoint at all"[..],
        ] {
            std::fs::write(&path, broken).unwrap();
            let msg = run
                .checkpoint::<4>(&device)
                .err()
                .expect("a broken checkpoint")
                .to_string();
            assert!(msg.contains("checkpoint.bin"), "{msg}");
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn open_needs_only_the_directory() {
        // a pre-T2 Learner run: no data_cfg.json, no metrics.csv, no normalizers
        let root = fresh_root("open_partial");
        std::fs::create_dir_all(&root).unwrap();
        let run = RunDir::open(&root).unwrap();
        assert!(run.normalizers().unwrap().is_none());
        assert!(run.costs().unwrap().is_none());
        assert!(run.data_config::<DarcyConfig>().is_err());
        assert_eq!(run.metrics().unwrap_err().kind(), io::ErrorKind::NotFound);
        assert!(run.load_fno::<4>(&Device::default()).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn open_reports_a_missing_run() {
        let err = RunDir::open(std::env::temp_dir().join("sciml_rs_no_such_run")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[cfg(unix)]
    #[test]
    fn artifacts_skip_symbolic_links() {
        let root = fresh_root("artifacts_symlink");
        let run = RunDir::create(&root, "test_fno").unwrap();
        std::fs::write(run.path().join("plots.png"), b"x").unwrap();
        std::fs::write(root.join("outside.png"), b"x").unwrap();
        std::os::unix::fs::symlink(root.join("outside.png"), run.path().join("linked.png"))
            .unwrap();
        assert_eq!(run.artifacts().unwrap(), vec![run.path().join("plots.png")]);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
