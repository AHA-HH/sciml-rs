//! The model registry (library CM0.4; proposal §6.2): a model's kind, saved as the `kind`
//! field of the run directory's `model_cfg.json`, maps to its config and constructor, the
//! analogue of `neuraloperator`'s `get_model`. A `model_cfg.json` without `kind` (every run
//! before CM0.4) reads as [`DEFAULT_KIND`], the FNO.
//!
//! Static dispatch, as for the [`Operator`] trait (D4): [`ModelConfig`] and [`Model`] are
//! enums with one variant per kind, so adding a kind is a compile error at every `match`
//! on them. [`KINDS`] and [`ModelConfig::from_json`]'s arms are kept by hand; a test
//! round-trips every variant through both. Rank stays a compile-time parameter
//! (`Model<R>`).

// see `models/fno/mod.rs`: Burn's `#[derive(Config)]` writes `field: field`
#![allow(clippy::redundant_field_names)]

use std::{error::Error, path::Path};

use burn::{
    config::Config,
    serde::{Deserialize, Serialize},
    tensor::{Device, Tensor},
};

use crate::neural_operators::models::{
    fno::{FNO, FNOConfig},
    operator::Operator,
};

/// The kind a `model_cfg.json` without a `kind` field reads as.
pub const DEFAULT_KIND: &str = FNOConfig::KIND;

/// Every registered kind, in the order of [`ModelConfig`]'s variants.
pub const KINDS: &[&str] = &[FNOConfig::KIND];

/// The config of a registered operator: its [`KIND`](OperatorConfig::KIND) is the name the
/// registry knows it by and the `kind` written to `model_cfg.json`.
pub trait OperatorConfig: Config + Clone {
    /// The registry name, one of [`KINDS`].
    const KIND: &'static str;
}

impl OperatorConfig for FNOConfig {
    const KIND: &'static str = "fno";
}

/// A model config of any registered kind, as read from `model_cfg.json`.
#[derive(Clone, Debug)]
pub enum ModelConfig {
    /// [`FNOConfig`], kind `"fno"`.
    Fno(FNOConfig),
}

impl ModelConfig {
    /// The config's kind, one of [`KINDS`].
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Fno(_) => FNOConfig::KIND,
        }
    }

    /// Parses a `model_cfg.json` by its `kind` (missing: [`DEFAULT_KIND`]).
    ///
    /// # Errors
    /// If `json` doesn't parse, its kind isn't registered, or it doesn't parse as that
    /// kind's config.
    pub fn from_json(json: &str) -> Result<Self, Box<dyn Error>> {
        match kind_of(json)?.as_str() {
            FNOConfig::KIND => Ok(Self::Fno(FNOConfig::load_binary(json.as_bytes())?)),
            kind => Err(format!(
                "unknown model kind `{kind}`; registered: {}",
                KINDS.join(", ")
            )
            .into()),
        }
    }

    /// Reads the `model_cfg.json` at `path` ([`from_json`](Self::from_json)).
    ///
    /// # Errors
    /// If the file can't be read, or as [`from_json`](Self::from_json); the message starts
    /// with the path.
    pub fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        std::fs::read_to_string(path)
            .map_err(Box::<dyn Error>::from)
            .and_then(|json| Self::from_json(&json))
            .map_err(|e| format!("{}: {e}", path.display()).into())
    }
}

/// An operator of any registered kind on rank-`R` channels-last tensors, built by name
/// from a [`ModelConfig`].
#[derive(Debug)]
pub enum Model<const R: usize> {
    /// [`FNO`].
    Fno(FNO<R>),
}

impl<const R: usize> Model<R> {
    /// A freshly initialised operator of `config`'s kind on `device` ([`Operator::init`]).
    ///
    /// # Panics
    /// As the kind's `init` (the FNO: if `modes.len() != R - 2`).
    pub fn init(config: &ModelConfig, device: &Device) -> Self {
        match config {
            ModelConfig::Fno(c) => Self::Fno(FNO::init(c, device)),
        }
    }

    /// The operator applied to `x`, `[B, *spatial, C]` in, `[B, *spatial, C_out]` out
    /// (CONVENTIONS §1).
    ///
    /// # Panics
    /// As the kind's `forward`.
    pub fn forward(&self, x: Tensor<R>) -> Tensor<R> {
        match self {
            Self::Fno(m) => Operator::forward(m, x),
        }
    }
}

/// A `model_cfg.json`'s kind: its `kind` field, or [`DEFAULT_KIND`] without one.
fn kind_of(json: &str) -> Result<String, Box<dyn Error>> {
    Ok(KindField::load_binary(json.as_bytes())?.kind())
}

/// The `kind` field alone; every other field is ignored.
#[derive(Config, Debug)]
pub(crate) struct KindField {
    kind: Option<String>,
}

impl KindField {
    /// The kind, or [`DEFAULT_KIND`] without one.
    pub(crate) fn kind(self) -> String {
        self.kind.unwrap_or_else(|| DEFAULT_KIND.to_string())
    }
}

/// `model_cfg.json` as written since CM0.4: `kind` first, then the config's own fields.
/// Burn's config readers ignore unknown fields, so a plain `C::load` still reads it.
#[derive(Serialize, Deserialize, Debug)]
#[serde(crate = "burn::serde")]
pub(crate) struct Tagged<C> {
    pub(crate) kind: String,
    #[serde(flatten)]
    pub(crate) config: C,
}

impl<C: OperatorConfig> Tagged<C> {
    /// `config` with its kind.
    pub(crate) fn new(config: &C) -> Self {
        Self {
            kind: C::KIND.to_string(),
            config: config.clone(),
        }
    }
}

impl<C: OperatorConfig> Config for Tagged<C> {}

#[cfg(test)]
mod tests {
    use burn::{config::config_to_json, module::Module, tensor::TensorData};

    use super::*;

    fn values<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    fn fno_cfg() -> FNOConfig {
        FNOConfig::new(vec![2], 1, 1)
            .with_hidden_channels(4)
            .with_n_layers(1)
    }

    /// Every variant, written as `model_cfg.json` and read back by its kind.
    #[test]
    fn every_kind_round_trips_through_from_json() {
        let every = [ModelConfig::Fno(fno_cfg())];
        assert_eq!(KINDS.len(), every.len());
        for config in every {
            let json = match &config {
                ModelConfig::Fno(c) => config_to_json(&Tagged::new(c)),
            };
            assert!(KINDS.contains(&config.kind()));
            assert_eq!(ModelConfig::from_json(&json).unwrap().kind(), config.kind());
        }
    }

    #[test]
    fn kinds_lists_every_variant() {
        assert_eq!(KINDS, &["fno"]);
        assert_eq!(ModelConfig::Fno(fno_cfg()).kind(), "fno");
    }

    #[test]
    fn tagged_writes_kind_first_and_reads_as_the_plain_config() {
        let json = config_to_json(&Tagged::new(&fno_cfg()));
        assert!(json.starts_with("{\n  \"kind\": \"fno\",\n"), "{json}");
        let plain = FNOConfig::load_binary(json.as_bytes()).unwrap();
        assert_eq!(config_to_json(&plain), config_to_json(&fno_cfg()));
        assert_eq!(kind_of(&json).unwrap(), "fno");
    }

    #[test]
    fn from_json_reads_a_missing_kind_as_the_fno() {
        let json = config_to_json(&fno_cfg());
        assert!(!json.contains("kind"));
        let ModelConfig::Fno(c) = ModelConfig::from_json(&json).unwrap();
        assert_eq!(config_to_json(&c), json);
    }

    #[test]
    fn from_json_reads_a_config_older_than_its_optional_fields() {
        // as written before `spectral_init` and `padding` existed
        let json = r#"{"modes": [2], "hidden_channels": 4, "data_channels": 1,
            "out_channels": 1, "n_layers": 1}"#;
        let ModelConfig::Fno(c) = ModelConfig::from_json(json).unwrap();
        assert_eq!((c.modes, c.spectral_init, c.padding), (vec![2], None, None));
    }

    #[test]
    fn from_json_rejects_an_unknown_kind() {
        let err = ModelConfig::from_json(r#"{"kind": "uno", "modes": [2]}"#).unwrap_err();
        assert_eq!(err.to_string(), "unknown model kind `uno`; registered: fno");
    }

    #[test]
    fn model_builds_by_name_and_forwards_as_the_fno() {
        let device = Device::default();
        let json = r#"{"kind": "fno", "modes": [2, 2], "hidden_channels": 4,
            "data_channels": 1, "out_channels": 1, "n_layers": 1}"#;
        let Model::Fno(built) = Model::<4>::init(&ModelConfig::from_json(json).unwrap(), &device);
        // the same weights in a directly built FNO (Flex's RNG is process-wide, so two
        // seeded inits are not compared; see `models/operator.rs`)
        let direct: FNO<4> = FNOConfig::load_binary(json.as_bytes())
            .unwrap()
            .init(&device)
            .load_record(built.clone().into_record());
        let x = Tensor::<4>::from_data(
            TensorData::new(
                (0..2 * 8 * 8).map(|i| i as f32 / 64.0).collect(),
                [2, 8, 8, 1],
            ),
            &device,
        );
        let model = Model::Fno(built);
        assert_eq!(values(model.forward(x.clone())), values(direct.forward(x)));
    }
}
