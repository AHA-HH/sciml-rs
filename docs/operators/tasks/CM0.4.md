# CM0.4 The registry

Started 2026-11-25 on `operators/cm0-4`. Roadmap: "CM0.4 The registry. Proposal §6.2:
`kind` in `model_cfg.json`, absent meaning the FNO. *Done when:* every existing run
directory loads unchanged; a model builds by name. *Needs:* CM0.3. Also the MLflow export's
model config (`tracking/builder` reads `model_cfg.json` as `FNOConfig`; CM0.3 left it)."
Proposal: §10; §1.4, §2, §6.1, §6.2.

## Goal

`model_cfg.json` names its operator's kind, and a registry turns a kind and its config into
a model, with every existing run directory (no `kind`) read as the FNO, unchanged.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- Roadmap: "every existing run directory loads unchanged; a model builds by name."
- Proposal §6.2: "`model_cfg.json` gains a `kind` field; a missing `kind` reads as the FNO,
  so every existing run directory loads unchanged. The registry maps a kind to its config
  and constructor, the analogue of 2.0.0's `get_model`. Rank stays a compile-time
  parameter."
- Roadmap line: the MLflow export's model config goes through the registry.
- Constraint 1 (§1.4): every existing FNO output bit for bit identical.

## Read first

- `src/neural_operators/models/operator.rs` (`Operator`, `NAME`), `models/fno/mod.rs`
  (`FNOConfig`).
- `src/neural_operators/runs/mod.rs` (`save_configs`, `load_model`, `model_config`,
  `checkpoint_of`, `load_config`).
- `src/neural_operators/tracking/builder.rs` (`start_requests`: `run.model_config()`).
- Burn 0.23.0-pre.2 `burn-core/src/config.rs` (`Config::save`/`load`: serde_json, pretty;
  the derive sets no `deny_unknown_fields`, so an extra `kind` key is ignored).

## Changes

- New `models/registry.rs`:
  - `OperatorConfig: Config` with `const KIND`; `FNOConfig` is `"fno"`. `Operator::Config`
    is bound by it, and `Operator::NAME` defaults to the config's `KIND` (one name).
  - `DEFAULT_KIND = "fno"` (what a missing `kind` reads as), `KINDS` (every kind).
  - `ModelConfig` (`Fno(FNOConfig)`): `from_json` dispatches on `kind`; unknown kinds are an
    error naming `KINDS`. `Model<R>` (`Fno(FNO<R>)`): `init(&ModelConfig, device)` and
    `forward`: the constructor by name. Static dispatch by `match`, no trait objects (D4).
  - The file format: a crate-private `Tagged<C>` writes `{"kind": …, <C's fields>}` (serde
    `flatten`, through Burn's re-exported serde; no new dependency).
- `runs`: `save_configs` takes `&impl OperatorConfig` and writes `kind` first (callers pass
  `FNOConfig` and compile unchanged). One private reader checks the file's kind (missing =
  `"fno"`) against the requested config's: `model_config`, `load_model`, `checkpoint_of`
  use it, so a run of another kind fails with a message naming both kinds. New
  `RunDir::any_model_config() -> ModelConfig`.
- `tracking/builder`: reads `any_model_config()`; FNO-only tags (padding) by `match`, so a
  new kind must be handled there to compile. Params flatten the config without `kind`, so an
  FNO run's export is unchanged.
- Conventions: none.

## Tests that prove it

- `runs`: a `model_cfg.json` written by `main` (no `kind`, and an older one without
  `spectral_init`/`padding`) loads through `model_config`, `any_model_config` and
  `load_model` (weights round trip, forward bit-identical to the saved model).
- `runs`: a new run's `model_cfg.json` starts with `"kind": "fno"` and still parses as a
  plain `FNOConfig` (`FNOConfig::load`), so readers built from `main` read new runs.
- `runs`: a file of an unknown kind, or of another registered kind, is an error naming it.
- `registry`: `ModelConfig::from_json` by name (`"fno"`, missing, unknown);
  `Model::<3>::init` then `forward` equals the `FNO<3>` built from the same weights, bit for
  bit; `Model::<4>` for a 2D config.
- `tracking/builder`: existing tests pass unedited (params identical); one test that an
  export of a no-`kind` (pre-CM0.4) directory gives the same requests as a new one.
- Constraint 1: `tests/fno_bit_identity.rs` against `main` (as CM0.3); every existing test
  unedited.

## Checks

`/verify` (fmt, clippy default/examples/all-targets/datagen, `cargo test`,
`cargo test --features datagen`, `cargo test --features mlflow`, `cargo doc`, the
bit-identity dumps against `main`). Metal: not backend-sensitive (JSON only); run
`--features metal --lib` if time allows and say so.

## VM runs

None.

## Constraints and risks

- Constraint 1: no tensor code changes; only the config file gains `kind`. New runs'
  `model_cfg.json` bytes differ (the key and its order); weights, metrics and outputs do
  not.
- Old binaries reading new runs: unknown keys are ignored by Burn's Config, so a `main`
  build reads a CM0.4 run (test above).
- 2D objective and the other lanes: `examples/` untouched; `save_configs`, `model_config`
  and `load_fno` keep their signatures for every existing caller.

## Pending runs

None.

## Open questions

None yet.
