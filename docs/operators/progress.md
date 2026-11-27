# Neural operator library progress log

Newest entries at the bottom. One entry per task, in that task's pull request, using this
template:

```
## YYYY-MM-DD: <ID> <title>
- Done:
- Tests:
- Checks: ran …; pending …
- Decisions: none, or D… signed off; questions for sign-off …
- Review: plan review (blocking items fixed); code quality APPROVE or BLOCKED → APPROVE (rounds, blockers fixed, findings left as follow-ups)
- FNO: unchanged bit for bit (how shown), or not touched
- Matrix: rows set
- Follow-ups:
```

## 2026-11-11: Planning
- Done: the proposal, draft 0.1, in `docs/design/neural-operator-library.md`, with its
  development wireframes (§4); its Section 10 as `docs/operators/roadmap.md`; the feature
  matrix seeded from the 2.0.0 file tree in `docs/operators/feature-matrix.md`.
- Tests: none.
- Checks: none run; the 2.0.0 facts come from the tag's file tree and source read on
  2026-11-11.
- Decisions: D1 (`neuraloperator` 2.0.0) taken by the author 2026-11-11; D0 comes next.
- FNO: not touched.
- Matrix: seeded, unverified (CM0.1).
- Follow-ups: review the draft; D0; then CM0.1 and CM0.2.

## 2026-11-24: D0 and D2
- Done: D0 signed off by the author: draft 0.1 of `docs/design/neural-operator-library.md`
  as the plan of record. D2 signed off as recommended: the Python side in `tools/parity/`,
  outside CI, with a lock file; fixtures in `tests/fixtures/parity/`.
- Tests: none.
- Checks: none (docs).
- Decisions: D0 and D2 signed off 2026-11-24.
- FNO: not touched.
- Matrix: unchanged.
- Follow-ups: CM0.1 and CM0.2 (no Rust, can run beside the 3D tasks); CM0.3 before the
  optimisers tasks that change `training/`.

## 2026-11-24: CM0.1 Feature matrix at 2.0.0
- Done: `feature-matrix.md` read from the source of tag 2.0.0 (`b307030`): every public
  top-level class and function of `neuralop/` (tests excluded) in a table row with its
  source path, the sciml-rs counterpart, a status and a task; an appendix of 186
  signatures with defaults, extracted with `ast`. New against the seed: `data/transforms`,
  `mpu`, `utils.py`, `BatchNorm`, the equation-loss classes, the differentiation APIs, the
  registry's contents (UNO and CODANO are not in `get_model`). Corrected: 2.0.0's Burgers
  dataset is space-time, so not the crate's Burgers task (missing, not partial).
- Tests: an independent agent re-read its own clone of the tag and checked every row and
  signature: 183 of 185 signatures exact, one item missing from the appendix, an
  inconsistent "exported" rule, two wrong parentheticals and three thin sciml-rs cells;
  all fixed. A second pass found one more (`LocalNO` does need torch_harmonics: the DISCO
  module builds a dict from names it failed to import); fixed, and nothing else remained
  (the "exported" marks match all five `__init__.py` files; 186 signatures exact).
- Checks: ran the independent row check (twice); `/verify` not run: docs only, no Rust
  check applies and CI's `changes` filter skips its jobs. Pending none.
- Decisions: none taken; questions for sign-off in the PR (scope of the not-planned rows,
  owners for the rows without a task, proposal §3 and CM1.3 corrections).
- Review: plan review PASS, no blocker; its findings fixed (this entry's evidence, the
  normaliser's eps, the legend's qualified statuses) or put to the author (owners of the
  "–" rows); code-quality review skipped (docs only).
- FNO: not touched.
- Matrix: every row set.
- Follow-ups: owners for the "–" rows (datasets, transforms, utilities): no roadmap line
  until the author answers the PR's question 2.

## 2026-11-24: CM0.2 Pinned Python environment and fixture generator
- Done: `tools/parity/`, a uv project (Python 3.12, `uv.lock`) pinning `neuraloperator`
  2.0.0 (PyPI wheel, its Python code identical to the tag's), torch 2.9.0, tensorly 0.9.0,
  tensorly-torch 0.5.0, torch-harmonics 0.7.4 (the tag's `doc` pin), numpy 2.3.4;
  `generate.py` writes 2.0.0's `FNO` (defaults but sizes) as `.npz` (input, output,
  `param.<name>`, complex as a trailing (re, im) axis) with a `.json` sidecar:
  `fno1d` (P = 3,529) and `fno2d` (P = 7,129), each in f32 and f64, 28–103 kB.
- Tests: the generator asserts P ≤ 10⁴, size ≤ 1 MB and f32/f64-only arrays, and reloads
  each file into a zeroed fresh model, reproducing the output bit for bit; `--check`
  regenerates and compares arrays and sidecars bit for bit (passes).
- Checks: ran `uv sync --locked`, `generate.py`, `generate.py --check`, the wheel-against-tag
  diff; the reviewer ran `cargo fmt --check`, clippy (default, `--all-targets`,
  `datagen`), `cargo test`, `cargo test --features datagen`, `cargo doc` (all pass); metal
  not run (no Rust changed).
- Decisions: D8 (fixture size) signed off 2026-11-24 as recommended (P ≲ 10⁴, under about
  1 MB per fixture).
- Review: plan review PASS (advisories fixed: round trip from zeroed weights, strict load,
  sidecars in `--check`); code quality APPROVE with findings (F1–F3 fixed; F4, F5 left in
  the PR text).
- FNO: not touched (no Rust changed).
- Matrix: no rows (environment only).
- Follow-ups: CM0.5 reads these fixtures (the CM0.5 line).

## 2026-11-24: CM0.3 The operator trait and a generic stack
- Done: D4 signed off by the author (option A: a generic trait with associated config and
  input types, static dispatch, no trait objects). `models/operator.rs`: `Operator`
  (`Config`, `Input`, `Output`, `NAME`, `init`, `forward`) and `GridOperator<R>` (rank-R
  tensors in and out, a blanket impl); `FNO<R>` implements it. Generic over the operator:
  `train_epoch`, `training_loop`, `train_components`, the new `train` and
  `build_components`, the data-parallel epoch, `train_epoch_gn` and `FnoResidual`'s
  `LeastSquares` impl, `evaluate`, `predict_host`, `RunDir::save_weights`,
  `save_checkpoint(_gn)`, the new `load_model` and `checkpoint_of`, `OperatorCheckpoint<M>`;
  `EpochObserver` and `TrainingComponents` take `M = FNO<R>`. The FNO-named functions stay
  as wrappers, so every existing caller compiles unchanged; file formats unchanged.
- Tests: `tests/fno_bit_identity.rs` (ignored by default; release): a fixed Phase 5 FNO
  forward, one Phase 5-shaped Adam epoch (two batches, checkpoint, reload, `predict_host`)
  and one Gauss–Newton epoch (with its checkpointed state), dumped as bytes; the same file
  run on an export of `main` and on the branch gives byte-identical dumps (65,536,
  28,481,592 and 22,476 bytes). Files are compared by content: Burnpack and
  `GaussNewtonState::to_bytes` record random `ParamId`s, so even `main` never writes the
  same bytes twice. `models/operator.rs`: the trait's forward is the FNO's; a test-only
  second operator trains, checkpoints, reloads and predicts through the generic stack.
- Checks: ran `cargo fmt --check`, clippy (default, `--examples`, `--all-targets`,
  `--features datagen --all-targets`), `cargo test` (305 lib + integration, all pass),
  `cargo test --features datagen` (379 lib, all pass), `cargo doc --no-deps` (no
  warnings), `cargo test --features metal --test gpu_parity` (9 pass) and `--features metal --lib`
  (305 pass; built at 851b6e4, later commits change only tests and docs); the main-against-
  branch `cmp` (twice, and independently by the plan reviewer). No hunk inside any
  existing test module; `tests/` only gains the new file.
- Decisions: D4 signed off 2026-11-24. Question for the author in the PR: does moving a
  test module's imports count as editing a test (CM0.6)?
- Review: plan review PASS (one blocker, a test comparing two seeded inits under Flex's
  process-wide RNG, fixed; advisories taken: two batches, the GN state, comment fixes, the
  CM0.4 sentence); code quality APPROVE with 7 findings, no blockers (F6 fixed; F7 and the
  checkpoint naming in the plan; F1–F5 to CM0.6).
- FNO: unchanged bit for bit (the dumps above against `main`).
- Matrix: `BaseModel`, `Trainer`, training state set (partial; CM0.3's part done).
- Follow-ups: CM0.4 (also the MLflow export's model config); CM0.6 (the review's F1–F5).

## 2026-11-25: CM0.4 The registry
- Done: `models/registry.rs`: `OperatorConfig` (`KIND`; `FNOConfig` is `"fno"`), which
  `Operator::Config` now requires (`Operator::NAME` defaults to it); `DEFAULT_KIND`,
  `KINDS`; `ModelConfig` (`from_json`, `load`: by `kind`, a missing `kind` reading as the
  FNO, an unknown one an error naming the registered kinds) and `Model<R>` (`init`,
  `forward`): the constructor by name, one enum variant per kind (D4, static dispatch).
  `RunDir::save_configs` writes `kind` first in `model_cfg.json` (callers unchanged);
  `model_config`, `load_model` and `checkpoint_of` check the file's kind against the
  config type; new `RunDir::any_model_config`. The MLflow export reads the model config
  through the registry; its requests are unchanged.
- Tests: a pre-CM0.4 `model_cfg.json` (no `kind`) loads through `model_config`,
  `any_model_config` and `load_model`, the loaded model predicting bit for bit as the
  saved one; a new file starts with `"kind": "fno"` and still parses as a plain
  `FNOConfig` (what a reader built from `main` does); wrong and unknown kinds fail with
  messages naming them; every kind round-trips through `from_json`; a config older than
  `spectral_init`/`padding` parses (at the parse level, not through `RunDir`); `Model::<4>`
  built by name forwards bit for bit as the FNO with the same weights; the MLflow
  `start_requests` of a directory with and without `kind` are equal.
- Checks: ran `cargo fmt --check`, clippy (default, `--examples`, `--all-targets`,
  `--features mlflow --all-targets`, `--features datagen --all-targets`), `cargo test`,
  `cargo test --features datagen`, `cargo test --features mlflow --lib`, `cargo doc
  --no-deps` (all pass); `cargo test --features metal --lib` (326 pass, at 2a01cf2, before
  the JSON-only review fixes); `tests/fno_bit_identity.rs` on the branch and on `main`,
  dumps identical (65,536, 22,476 and 28,481,592 bytes).
- Decisions: none. Question for the author in the PR: keep `Operator::NAME` (now always its
  config's `KIND`) or retire it (CM0.6).
- Review: plan review PASS (advisories: `DEFAULT_KIND` from `FNOConfig::KIND` and one
  reader, fixed; the older-format test is parse-level, stated above); code quality APPROVE
  with findings (F1, F2 fixed; F3 a question in the PR; F4 to CM2.1).
- FNO: unchanged bit for bit (the dumps above against `main`); only new runs'
  `model_cfg.json` gains the `kind` key.
- Matrix: `BaseModel`, `get_model`/`available_models` set to have (infrastructure).
- Follow-ups: CM2.1 (the second kind: its variants, the export's arm, `Model<R>` as a
  `Module` if needed; review F4); CM0.6 (`Operator::NAME`, on the author's answer).

## 2026-11-25: D3 and the CM0.3 test-import question
- Done: D3 signed off by the author as recommended: parity holds when the relative max
  error is ≤ 1e-5 in f32 and ≤ 1e-12 in f64, on the CPU. The author's answer to CM0.3's
  question: moving a test module's `use` lines into that module is not editing a test, so
  CM0.6 moves the four `#[cfg(test)]` imports.
- Tests: none.
- Checks: none (docs).
- Decisions: D3 signed off 2026-11-25.
- FNO: not touched.
- Matrix: unchanged.
- Follow-ups: CM0.5 (the parity harness) and CM0.6 (the tidy-up) are runnable.

## 2026-11-25: CM0.5 The parity harness
- Done: `tests/parity_common/mod.rs`, the harness: `Fixture` (a CM0.2 `.npz`, its precision
  from the name), `Take` (`Whole`, or the `Re`/`Im` half of a complex tensor's trailing
  axis), `load_into` (a hand-written map of 2.0.0 names to Burn paths, applied with
  `ModuleSnapshot::apply`; panics unless every fixture parameter is mapped, complex ones by
  both halves, every entry applies and no module parameter is left out), `rel_max_error`
  (max |a − e| / max |e|; panics on a NaN or infinity) and `assert_parity`, with D3's
  bounds. `tests/parity_common/fno1d.rs`: `Fno1dRef`, 2.0.0's 1D `FNO` at its defaults in
  test code (Burn `Conv1d` and `gelu`, the crate's `SpectralConv`), and its 44-entry map
  of the fixture's 40 arrays.
- Tests: `fno1d_f32_matches_2_0_0` (3.514e-7 ≤ 1e-5, Flex); `fno1d_f64_matches_2_0_0` in
  its own binary, `#[ignore]`d (1.211e-8 > 1e-12: Flex's f64 `rfft` twiddles are f32); a
  1e-3 weight shift fails parity; the load rejects an unmapped array, an unknown path and
  half a complex tensor; the re/im split; `rel_max_error` on a hand-computed pair and on NaN.
- Checks: ran `cargo fmt --check`, clippy (default, `--examples`, `--all-targets`,
  `--features datagen --all-targets`), `cargo test`, `cargo doc --no-deps` (all pass);
  `cargo test --features datagen` (pass, run by the plan review before its NaN fix, which
  changes only `tests/`). Not run: `--features metal` (no backend-sensitive change), the
  bit-identity dumps (nothing under `src/` changes).
- Decisions: none. Questions for the author in the PR: D3's f64 bound on the CPU while the
  Flex bug stands; the reading of "relative max error".
- Review: plan review FAIL on one blocker (`rel_max_error` let NaN through), fixed with a
  test; its advisories (blanket `allow`, unused `d3`, plan text, grid dtype) fixed. Code
  quality APPROVE with findings (F2, F4, F6, F7 and a nit fixed; F1, F3, F5, F8 in the
  roadmap line).
- FNO: not touched (tests only).
- Matrix: unchanged (the mirror is test code; no row reaches **have**).
- Follow-ups: in CM0.5's roadmap line (for CM0.6, CM1.1 and CM1.5).

## 2026-11-25: D3 amended (CM0.5's questions)
- Done: the author's answers to CM0.5's two questions (#148), as the coordinating session
  recommended. "Relative max error" is max |a − e| / max |e| over the tensor, so elements
  near zero do not dominate. f32 parity on the CPU is enough for M1 while Flex's f64 `rfft`
  uses f32 twiddles (`upstream/burn/flex-f64-fft-f32-twiddles`): each f64 parity test is
  `#[ignore]`d with its measured error recorded, as CM0.5 does (1.211e-8 for the 1D FNO).
- Tests: none.
- Checks: none (docs).
- Decisions: D3 amended 2026-11-25.
- FNO: not touched.
- Matrix: unchanged.
- Follow-ups: CM1.1 runs to D3 as amended; the f64 bound returns when the upstream fix lands.

## 2026-11-25: CM1.1 Legacy spectral convolution parity
- Done: three configurations of 2.0.0's generic legacy `SpectralConv`
  (`layers/legacy_spectral_convolution.py`, defaults, bias on) in `tools/parity/generate.py`,
  in ≠ out and odd and non-power-of-two extents; fixtures
  `legacy_spectral_conv{1,2,3}d_{f32,f64}`. `tests/parity_common/legacy_conv.rs`:
  `LegacyConv<R>` (the crate's `SpectralConv<R>` plus a test-only bias) and the hand-written
  maps; 2.0.0's corners run axis 0 most significant, so in 3D its `weight.1` and `weight.2`
  are the crate's corners 2 and 1. The reference models (`fno1d.rs`, `legacy_conv.rs`) are no
  longer modules of `parity_common/mod.rs`: each binary includes its own with `#[path]`, so
  none compiles dead code (CM0.5's test files change only in `mod`/`use` lines).
- Tests: `legacy_spectral_conv{1,2,3}d_f32_matches_2_0_0` (2.116e-7, 4.722e-7, 3.053e-7 ≤
  1e-5, Flex, M2 Pro); `legacy_3d_corner_order_matters` (2.0.0's corner order unpermuted:
  4.853e-1, fails D3); the f64 three `#[ignore]`d (1.588e-7, 2.907e-7, 2.498e-7): 2.0.0's
  legacy `forward` builds its spectrum as `cfloat`, so its f64 output is f32-accurate
  (7.8e-8 against numpy in f64, measured), whatever Burn does.
- Checks: ran `cargo fmt --check`, clippy (default, `--examples`, `--all-targets`,
  `--features datagen --all-targets`), `cargo test`, `cargo test --features datagen`,
  `cargo doc --no-deps` (all pass); `generate.py --check` (all ten fixtures the same, CM0.2's
  four unchanged). Not run: `--features metal` and the bit-identity dumps (nothing under
  `src/` changes).
- Decisions: none. The author's answer on the review's F2 (2026-11-25): keep the legacy f64
  tests `#[ignore]`d, as D3 as amended asks.
- Review: plan review PASS (advisories fixed: the negative control prints its error, the bias
  doc; the docs committed); code quality APPROVE with findings (F3, F4 fixed; F1, one
  `flex_f64` for the f64 binaries, left to CM1.5 as CM0.5's line has it; F2 answered; F5
  informational).
- FNO: not touched (tests, fixtures and the generator only).
- Matrix: the legacy `SpectralConv` row stays partial, its note naming what is at parity
  and what is missing.
- Follow-ups: CM0.5's sidecar and grid items pass to CM1.2 and CM1.5 (CM1.1's roadmap line);
  `flex_f64` shared (CM1.5, now easy with `#[path]`).

## 2026-11-25: D5
- Done: D5 signed off by the author: A, as recommended. The spectral weights stay in the
  corner layout internally (CONVENTIONS §5 unchanged); 2.0.0's centred weights are
  converted when fixtures are loaded, the FFT normalisation and the mode counts included
  (CM1.2). A centred layout may come later only as an opt-in option, off by default, in a
  task of its own, if a need (checkpoint interchange with PyTorch) or a measured gain
  appears. The outputs are the same either way: the layouts store the same coefficients.
- Tests: none.
- Checks: none (docs).
- Decisions: D5 signed off 2026-11-25.
- FNO: not touched.
- Matrix: unchanged.
- Follow-ups: CM1.2 runnable.

## 2026-11-25: CM1.6a Finite differences and the data losses
- Done: CM1.6 split into CM1.6a–c (roadmap). `losses/differentiation.rs`: `FiniteDiff<D>`,
  one rank-generic stencil (`partial`, orders 1 and 2, periodic by `roll_dim` or one-sided
  with 2.0.0's coefficients) for 2.0.0's eighteen per-dimension methods; `dx`/`dy`/`dz`,
  `partial_sum`, `laplacian`, `gradient`, `divergence`, `curl_2d`, `curl_3d`.
  `losses/sobolev.rs`: `SobolevLoss<D>` (H1, Hdiv; 2.0.0's quadrature measure / s, which
  is also the step, and eps in `rel`; zero subgradient at an exact match).
  `losses/pointwise.rs`: `PointwiseQuantileLoss`, `mse_loss`. Fixtures
  `losses{1,2,3}d_{f32,f64}` from `tools/parity/generate_losses.py` (beside CM1.1's
  `generate.py`).
- Tests: parity on every `FiniteDiff` operator, H1 and Hdiv (rel, abs; periodic, one-sided,
  mixed; sum, mean), the quantile loss and MSE: f32 ≤ 1.7e-7 (D3 1e-5), f64 ≤ 6e-16 (D3
  1e-12, its own binary as Flex's dtype is process-wide; no FFT, so not `#[ignore]`d).
  `LpLoss::rel` against 2.0.0's: 9e-8 in f32, 6e-10 to 1.9e-9 in f64, 2.0.0's eps of 1e-8
  in the denominator (the crate's has none, CONVENTIONS §11), bounded by 1e-8. A swapped
  axis fails parity. Unit tests: sin within truncation error, one-sided ends exact on
  polynomials, curl ∘ gradient ≈ 0, panics, hand-computed H1, MSE and quantile values,
  exact matches with zero (MSE, Sobolev) or finite (quantile) gradients.
- Checks: ran `cargo fmt --check`, clippy (default, `--examples`, `--all-targets`,
  `--features datagen --all-targets`), `cargo test` (381 passed), `cargo test --features
  datagen` (462 passed), `cargo doc --no-deps`, `generate_losses.py --check` (six fixtures
  the same): all pass. Not run: `--features metal` (no backend-sensitive change).
- Decisions: none. Question for the author in the PR: should CONVENTIONS §11 record the
  2.0.0 losses' quadrature and eps (outside §1–§6, no version bump)?
- Review: plan review PASS (advisories: the lp.rel exception now on the roadmap line, an
  exact-match test for `pointwise.rs`, CM2.5 needs CM1.6a, Panics docs; all done; the two
  harness `allow`s stay until the follow-up below). Code quality APPROVE with findings: F1,
  F2, F5, F8, F9 fixed; F3, F4, F6, F7 to CM1.6b's line.
- FNO: unchanged bit for bit (`fno_bit_identity` dumps on `main` bb96d0e and on the
  branch: `forward`, `phase5_epoch`, `gauss_newton_epoch` byte-identical); no existing
  source file changed but `losses/mod.rs`.
- Matrix: `H1Loss`/`HdivLoss`/`PointwiseQuantileLoss`/`MSELoss` and `FiniteDiff` set to
  **have** (the first rows); `LpLoss` stays partial (rel parity up to eps); the
  aggregator and Fourier rows point at CM1.6b and CM1.6c.
- Follow-ups: CM1.6b's line (the review's F3, F4, F6, F7); CM1.6c (`FourierDiff`,
  `non_uniform_fd`); the §11 question.

## 2026-11-25: CM1.3a The parts
- Done: CM1.3 split into CM1.3a (the parts) and CM1.3b (the FNO's block options); CM1.4 now
  needs CM1.3a, CM1.5 also CM1.3b. `layers/channel_mlp.rs` (`ChannelMlpConfig`,
  `ChannelMlp`: 2.0.0's `ChannelMLP`, any rank, the activation a Burn `ActivationConfig`,
  default GELU; `conv1x1`, a kernel-1 `Conv1d` as one matmul), `layers/skip.rs` (`SkipKind`,
  `Skip<R>`: identity, bias-free linear, soft gating), `layers/norm.rs` (`AdaInConfig`,
  `AdaIn`, the embedding a `forward` argument). Instance and group norm are Burn's, tested.
  Seven 2.0.0 fixtures (`channel_mlp{1,3}d`, `linear_skip3d`, `soft_gating2d`,
  `instance_norm1d`, `group_norm3d`, `ada_in2d`), constant initialisations redrawn; the
  harness gains `Take::Transposed` (torch `Linear` weights).
- Tests: the seven parts in f32 (`tests/parity_block_parts.rs`, ≤ 1.9e-7) and f64
  (`tests/parity_block_parts_f64.rs`, ≤ 7.1e-16; none ignored: no FFT); unit tests of each
  module (a channel MLP by hand, soft gating per channel, AdaIN's scale and shift, widths,
  the panics).
- Checks: see the PR (fmt, clippy ×5, `cargo test`, `cargo test --features datagen`,
  `cargo doc`, the bit-identity dumps against `main`, `generate.py --check`).
- Decisions: none.
- Review: plan review PASS (advisories: the plan's `mod` count, the identity skip's
  missing fixture and the 1D–3D reading, now in the plan). Code quality APPROVE with findings:
  F1 `conv1x1` moved to `layers/pointwise.rs` (the FNO's copy goes in CM1.3b); F2 one function
  per case; F4 the width assert in the match, `split_last`; F5 one `FCS` map; F6 docs; F3
  (`flex_f64`, `Precision::suffix`) left to CM1.6b's harness tidy, as the suffix would be
  dead code in some binaries.
- FNO: not touched (new modules only).
- Matrix: `ChannelMLP` partial, skips have, norms partial (no `BatchNorm`), `FNOBlocks` to
  CM1.3b.
- Follow-ups: CM1.3b's line: the FNO's pointwise matmul replaced by
  `layers::pointwise::conv1x1`. One `flex_f64` in `parity_common`: CM1.6b's line (this branch
  adds a fourth copy, `tests/parity_block_parts_f64.rs`).

## 2026-11-25: CM1.6b Aggregators
- Done: `losses/meta_losses.rs`: `WeightedSumLoss`, `FieldwiseAggregatorLoss` (a field is a
  channel range on one axis, prediction and target sliced alike; 2.0.0 reshapes only the
  prediction, the two agree on its one-channel `(N, C)` use), `SoftAdapt`, `Relobralo` (ρ
  from a seeded `ChaCha8Rng`, or replayed by `forward_with_rho`), over `LossFn<R>`
  closures; adaptive weights on the host in f64, constants in the aggregate as under
  2.0.0's `no_grad`. CM1.6a's follow-ups: `Reduction::apply` and `root_zero_safe` in
  `data_losses.rs`, shared by `LpLoss`, `SobolevLoss` and `PointwiseQuantileLoss`; one
  `tests/flex_f64/` for the five f64 binaries (CM1.3a's, merged from `main`, included); `Fixture::tensor(key, device)` public.
  Fixtures `meta_losses_{f32,f64}` from `generate_losses.py`.
- Tests: per step, the aggregate, the weights and d aggregate / d loss of SoftAdapt and
  ReLoBRaLo (with and without weights; ReLoBRaLo also at α 0.9, β 0.5, τ 2, replaying
  torch's draws, both branches of ρ), the weighted sum (default and given weights) and the
  fieldwise loss: f32 ≤ 2.0e-7, f64 ≤ 2.0e-16 (D3 1e-5, 1e-12). Unit tests: the seeded
  draw exact at β = 0 and 1 and reproducible, a hand-computed SoftAdapt step, panics.
- Checks: on the branch merged with `main` (CM1.3a, CO0.2): ran `cargo fmt --check`,
  clippy (default, `--examples`, `--all-targets`, `--features datagen --all-targets`),
  `cargo test` (433 passed), `cargo test --features datagen` (514 passed), `cargo doc
  --no-deps`, all pass; `generate_losses.py --check` (8 fixtures the same) after the
  review fixes. Not run: `--features metal` (nothing backend-sensitive), a
  build without `flex`.
- Decisions: none.
- Review: plan review PASS (advisories: the `#[allow(dead_code)]` on `mod parity_common`
  in the loss binaries kept, as the plan says; `parity_losses_common` behind `flex` in the
  f64 binary, done; weights checked once, done). Code quality APPROVE with findings: F1,
  F2, F3 (weights only), F4–F8, F10 fixed; F9 (one device read per loss per step) left as
  a note, harmless on the CPU.
- FNO: unchanged bit for bit (`fno_bit_identity` dumps on `main` 99edf66 and the merged
  branch, each in its own build directory: `forward`, `phase5_epoch`, `gauss_newton_epoch`
  byte-identical). CM1.6a's comparison shared one build directory between `main` and the
  branch, so it did not show this; CM1.6a changed no FNO code.
- Matrix: the aggregator row set to **have**.
- Follow-ups: CM1.6c; the CONVENTIONS §11 question of CM1.6a (#154) stands.

## 2026-11-25: CM1.2 Centred layout, separable, `max_n_modes`
- Done: `SpectralConv::new_separable` (weights `[1, C, modes..]`, an element-wise complex
  product) and `with_modes` (modes below the stored ones: low `0..m`, high the last `m` rows,
  last axis `0..m`), both off by default; the converter `centred_to_corners` of 2.0.0's
  centred weights to corners at fixture load, with `Fixture::replace` (D5), in
  `tests/parity_common/centred_conv.rs`; seven centred fixtures `spectral_conv*_{f32,f64}`.
  `fft_norm="forward"` needs no conversion (one forward and one inverse at the same N).
  2.0.0's centred layer applies `fftshift` twice, so on an odd extent of a non-last axis its
  output spectrum moves by one bin: not representable in corners (numpy check in the plan).
- Tests: six f32 parity tests within D3 (2.6e-7 to 4.0e-7, Flex, M2 Pro), including an odd
  count (7; 5 in 3D), `fft_norm="backward"`, separable and stored (10, 8) used (6, 4); the
  odd extent fails as expected (1.771); unconverted halves fail (1.029); the converter's
  exact values; six layer unit tests; the f64 six `#[ignore]`d (1.5e-7 to 2.7e-7).
- Checks: ran `cargo fmt --check`, clippy (default, `--examples`, `--all-targets`,
  `--features datagen --all-targets`), `cargo test`, `cargo test --features datagen`,
  `cargo doc --no-deps`, the layer tests and `gpu_parity` under `--features metal` (pass,
  27 and 9; the separable truncation test compares to 1e-6, since Metal rounds the strided
  slice a few ulps apart), `generate.py` (the ten earlier fixtures unchanged); all in the
  task's own build directory. `cargo test --features datagen` failed once on
  `fno::tests::grid::grid_controls_1d` and passed alone (twice) and on a full rerun: it seeds
  Flex's process-wide RNG while other tests draw from it (pre-existing).
- Decisions: D5 signed off 2026-11-25 (#153), applied as signed off.
- Review: plan review PASS (advisories fixed: doc wording, a separable truncation test);
  code quality APPROVE with findings (one presumptive blocker, the file over 1,000 lines with
  its tests, waived; L1, D2 and the doc slip fixed; M1, B1 and the split are follow-ups).
- FNO: unchanged bit for bit (`fno_bit_identity` dumps on `main` and the branch, `cmp`
  same for the forward, the Phase 5-shaped epoch and the Gauss–Newton epoch).
- Matrix: the centred `SpectralConv` row set to partial.
- Follow-ups: in CM1.2's roadmap line (for CM1.5 and CM1.8).

## 2026-11-25: CM1.3b The FNO's block options
- Done: `FNOConfig` gains `activation`, `fno_skip`, `channel_mlp` (`BlockMlpConfig`:
  expansion 0.5, rounded half to even as Python's `round`; skip, default soft gating),
  `norm` (`layers::norm::NormKind`: instance, group, AdaIN) and `preactivation`, each `None`
  the original block. `FNO::fourier_layers` runs 2.0.0's `FNOBlocks` per layer (`post_block`,
  `pre_block`); `FNO::forward_with_embedding` conditions AdaIN. 2.0.0 applies the activation
  twice without MLP or norm; for ReLU the repeat is skipped (`relu ∘ relu = relu`), so the
  original block's ops are unchanged. The FNO's pointwise matmul is `layers::pointwise`'s
  (`conv1x1`, `linear_cf`). Nine `FNOBlocks` fixtures (`generate.py`: the legacy layer with
  `bias=False`; f32 only; AdaIN at one layer and 2 channels for D8); the harness gains
  `load_into_part` (a `PathFilter` to the block's paths).
- Tests: `tests/parity_fno_blocks.rs`, post- and pre-activation for no norm, instance, group
  and AdaIN, plus the repeated tanh, 1D–3D: ≤ 2.3e-6 (D3 1e-5). Unit tests: absent options
  build no new parameters, the options build 2.0.0's layers, half-to-even width, the skipped
  ReLU repeat equals ReLU twice bit for bit, AdaIN needs and uses its embedding,
  `linear_cf` against `Linear::forward`.
- Checks: see the PR (fmt, clippy ×4, `cargo test`, `cargo test --features datagen`,
  `cargo doc`, the bit-identity dumps against `main`, `--features metal` block parity, FNO
  unit tests and `gpu_parity`, `generate.py --check`).
- Decisions: none.
- Review: plan review FAIL → fixed: the channel MLP's width rounded half away from zero, not
  half to even as 2.0.0 (width 5: 3 against 2); advisories (plan text, the ReLU test) done.
  Code quality APPROVE with findings: F1 (post/pre split), F2 (repeat decided at `init`), F3
  (one `fno_skip` helper, `w` documented), F4 (the test map follows the config), F5 (a
  `shortcut:` note on the redraw rule), F6 (plan), F7 (doc) fixed; F8 (struct literals) to
  CM1.5's line.
- FNO: unchanged bit for bit (`fno_bit_identity` dumps on `main` 99edf66 and on the branch:
  `forward`, `phase5_epoch`, `gauss_newton_epoch` byte-identical); existing tests edited
  only to complete `FNOConfig` literals with `None` and to call the moved pointwise helpers.
- Matrix: `FNOBlocks` partial (CM1.3b; missing `batch_norm`, `stabilizer`, precision,
  complex data, `fno_skip=None`, the convolution's bias).
- Merge with optimisers CO1.1: its `FNOConfig::activation` (ReLU or GELU, layers and
  projection) is the one activation option; this task's `ActivationConfig` field is dropped,
  2.0.0's repeated activation keys on the block options (so CO1.1's GELU FNO is unchanged),
  and, at the author's request, CO1.1's enum widened: `Tanh`, `Sigmoid`, `Silu`, `LeakyRelu`,
  `Selu`, `Mish`, `HardSwish`, `Softsign`, each against torch (`activations1d`, ≤ 9.4e-8).
- Follow-ups: CM1.5's line (the convolution's bias in the block; the `FNOConfig` literals).

## 2026-11-26: CM2.5 Equation losses (PINO)
- Done: `losses/equation_losses.rs`: `BurgersEqnLoss` (2.0.0's `fdm`, or any `GridDiff`),
  `ic_loss`, `nonlinear_poisson_lhs`, `poisson_interior_loss`, `poisson_boundary_loss`,
  `PoissonEqnLoss` (`points`, 2.0.0's point-list form with the interior derivatives
  supplied; `grid`, the interior residual and the boundary MSE on a grid, which 2.0.0
  lacks). In `differentiation.rs`, `GridDiff` (implemented by `FiniteDiff`) and
  `MatrixDiff`, a differentiation matrix per axis; `MatrixDiff::chebyshev` (behind
  `datagen`) is the crate's Chebyshev path on the CGL grid (CONVENTIONS §12). In
  `pointwise.rs`, `mean_squared_error`, 2.0.0's base loss `F.mse_loss` (the mean over every
  entry, not `MSELoss`). Fixtures `equation_losses_{f32,f64}` from `generate_losses.py`.
- Tests: parity of Burgers (defaults; visc 0.05, domain (2, 3)), IC and the three Poisson
  losses (the left-hand side formed from 2.0.0's autograd derivatives stored in the
  fixture): f32 ≤ 6.9e-8, f64 ≤ 5.3e-16 (D3 1e-5, 1e-12); swapped Burgers axes fail (> 0.1).
  Unit tests (f32, Flex, M2 Pro): `MatrixDiff` with periodic central-difference matrices
  equals `FiniteDiff` (≤ 9.8e-8); second-order convergence of the grid Poisson loss (MSE
  ratio 17.1 for h halved); Chebyshev at 17 nodes 5.6e-11 against finite differences at
  65 points 5.4e-5; exact on a cubic; gradients through Burgers, the grid loss and
  `MatrixDiff`; panics.
- Checks: ran `cargo fmt --check`, clippy (default, `--examples`, `--all-targets`,
  `--features datagen --all-targets`), `cargo test` (474 passed), `cargo test --features
  datagen` (561 passed), `cargo doc --no-deps`, `generate_losses.py --check` (the ten
  fixtures the same); all pass, in the task's own build directory. Not run: `--features
  metal` (nothing backend-specific beyond matmul and slicing).
- Decisions: none.
- Review: plan review FAIL on one blocker (a `clippy::type_complexity` error under
  `datagen`), fixed; advisories (panics documented, shape assert) fixed. Code quality
  BLOCKED (the same clippy error) → APPROVE with findings (r2): F1–F5, F7 and the nits
  fixed; r2's R2 and R4 fixed, R1 mostly (one Laplacian fold for `GridDiff`; `FiniteDiff`
  keeps its own `partial_sum`); left: `generate.py`'s own provenance block, the argument
  order of `PoissonEqnLoss::points`.
- FNO: unchanged bit for bit (`fno_bit_identity` dumps on `main` 6ea636b and the branch,
  each in its own build directory: `forward`, `phase5_epoch`, `gauss_newton_epoch` `cmp`
  same).
- Matrix: the equation-loss row set to partial (the `autograd` method with CM3.4).
- Follow-ups: CM3.4's line (2.0.0's `autograd` method on point clouds).

## 2026-11-26: CM1.4 Lifting, projection and embeddings
- Done: `layers::embedding`: 2.0.0's `GridEmbeddingNd` (any rank and boundaries, `j / s`
  without the endpoint, axis order), `SinusoidalEmbedding` (`transformer`, `nerf`; sin and
  cos interleaved), `RotaryEmbedding2d` with `rotate_half`, `apply_rotary_pos_emb`,
  `apply_2d_rotary_pos_emb`; each parameter-free, so its own `Config`. `FNOConfig` gains
  `lifting_channel_ratio` (`fc0` `(C + D) → L`, the activation, `lifting_out` `L → W`,
  `L = int(r · W)`; 0 keeps one layer), `projection_channel_ratio` (`fc1`'s width instead of
  128) and `positional_embedding` (`GridNd`, or `Disabled` for 2.0.0's `None`), each `None`
  today's model. Sinusoidal and rotary stay standalone: 2.0.0's FNO does not use them.
- Tests: `tests/parity_embeddings.rs`, eight f32 fixtures (the four `FNO`s with
  `n_layers=0`: 1D defaults, 2D ratio 0 and 1.5 with tanh, 3D without a grid, 2D with
  boundaries; the grid alone; both sinusoidal kinds; rotary): ≤ 7.8e-7 (D3 1e-5).
  `_f64.rs`: the FNO without a grid at 1.7e-16; seven `#[ignore]`d with their errors
  (1.6e-6 to 5.0e-10), as 2.0.0 builds its grid and frequencies in f32. Unit tests: the
  widths truncate as Python's `int`, `Disabled` lifts C channels, absent options build
  today's layers, the zero-width and wrong-rank panics, `rotate_half` and the grid by hand.
- Checks: see the PR (fmt, clippy ×4, `cargo test`, `cargo test --features datagen`,
  `cargo doc`, `generate.py --check`, the bit-identity dumps, `--features metal`).
- Decisions: none. Question for sign-off: a sentence in CONVENTIONS §2 naming the opt-in
  `GridNd` grid.
- Review: plan review PASS (advisories: the plan's departures noted, a doc index made
  clear; the f64 fixture idea to CM1.5's line). Code quality APPROVE with findings: F2 (one
  coordinate-channel count) and F3 (`lifting_out`, before it became a checkpoint path)
  fixed; F1 (one grid builder) and F4 (shared parity helpers) to CM1.5's line.
- FNO: unchanged bit for bit (`fno_bit_identity` dumps on `main` cb7265d and the branch:
  `forward`, `phase5_epoch`, `gauss_newton_epoch` byte-identical); existing tests edited only
  to complete `FNOConfig` literals with `None`.
- Matrix: `GridEmbeddingND` and `SinusoidalEmbedding`/`RotaryEmbedding2D` rows have; `FNO`
  partial, noting the new options.
- Follow-ups: CM1.5's line (F1, F4, f64 fixtures with f64 constants).

## 2026-11-26: CM1.6c Fourier and non-uniform derivatives
- Done: `losses/differentiation.rs`: `FourierDiff<D>` (`l`, 2.0.0's `L`), any order by a
  per-axis `rfft`/`irfft` with (i k)ᵖ, k = 2π m / L, the odd-order Nyquist multiplier 0
  (equal to 2.0.0's `fftn` and real part for a one-axis derivative). `GridDiff` now carries
  `dx`/`dy`/`dz`, `partial_sum`, `laplacian`, `gradient`, `divergence`, `curl_2d`, `curl_3d`
  for every grid operator (moved from `FiniteDiff`, bodies unchanged). New
  `losses/non_uniform_fd.rs`: `NonUniformFd` (2.0.0's arguments and defaults), `stencils`
  on the host in f64 (k nearest points; minimum-norm weights by QR of Aᵀ as gelsy, or 2.0.0's
  ridge system by Cholesky), `Stencils::apply` on any device with gradients to the values.
  Fixtures `derivatives_{f32,f64}` from `generate_losses.py` (now writing every array
  C-contiguous; the existing fixtures unchanged).
- Tests: parity of `FourierDiff` 1D–3D (orders 1–3 per axis, Laplacian, gradient, divergence,
  curl; grids 32, 16 × 15, 8 × 6 × 5): f32 ≤ 4.9e-7; f64 `#[ignore]`d, 4.0e-7 (Flex's f32
  twiddles, D3 as amended). `non_uniform_fd` (2D default, radius 0.2, ridge; 3D 7
  neighbours): f32 ≤ 6.3e-7, f64 ≤ 1.2e-15; the ridge case (condition ≈ 5e6, 2.0.0's own f32
  result 4.0e-2 from its f64) f32 against the f64 reference 7.3e-8, f64 6.4e-11 ≤ 1e-9. A
  swapped-axis control fails (> 0.1). Unit tests: Fourier exact on a trigonometric
  polynomial (n = 15, 16), the Nyquist mode, panics; stencils exact on affine functions (both
  solvers), the minimum-norm weights against Aᵀ(AAᵀ)⁻¹b, the radius mask, the gradient to the
  values against the weights, panics.
- Checks: see the PR (fmt, clippy ×4, `cargo test`, `cargo test --features datagen`,
  `cargo doc`, `generate_losses.py --check`, the bit-identity dumps against `main`). Not run:
  `--features metal` (nothing backend-specific beyond Burn's `rfft`/`irfft`, `select`).
- Decisions: the ridge case's bounds (f32 against 2.0.0's f64 reference within D3; f64
  within 1e-9, not 1e-12) signed off 2026-11-26 by the author: accept.
- Review: plan review PASS (advisories fixed: plan text, the gradient test, the tie note).
  Code quality APPROVE with findings: F1 (`partial_sum` on `GridDiff`), F2 (one
  vector-calculus check), F3 (the ridge reference explicit), F4, F5 (`Stencils`), F8
  (`FourierDiff`'s `partial` in its trait impl), F10, F11 fixed; F6 (one factorisation per
  point), F7 (an enum for the solver), F9 (the ridge solve by QR) left, optional, in the PR.
- FNO: unchanged bit for bit (`fno_bit_identity` dumps on `main` and the branch, `cmp`).
- Matrix: `FourierDiff` partial, `non_uniform_fd` have.
- Follow-ups: CM2.4's line (`use_fc`, `low_pass_filter_ratio`, the mixed derivative).

## 2026-11-26: CM1.5a Whole-FNO parity: defaults and block sweeps
- Done: CM1.5 split into CM1.5a–d (roadmap). `FNOConfig` gains `conv_bias` (2.0.0's
  `SpectralConv` bias, `[W]` per layer in `FNO::conv_biases`, added before the norm, drawn
  as 2.0.0's), `separable`, `max_modes` (2.0.0's `max_n_modes`, per corner) and
  `tanh_stabilizer` (2.0.0's `stabilizer="tanh"`, on the convolution's input only), each
  `None` today's model; one `FNO::spectral` helper used by both block forms. Six sweep
  fixtures from `generate.py` (`SWEEPS`, f32; its redraw now skips complex weights, the
  existing fixtures unchanged). Test harness: `tests/parity_common/fno.rs` (the whole-FNO
  map, centred weights converted per layer, D5) and `fno_map.rs` (2.0.0's defaults, the
  lifting/projection and block-part maps, the permutes) shared by the FNO-family binaries;
  `Fixture::open_as`, `check_d3`, `centred_to_corners` and `Fixture::replace` in `mod.rs`
  (CM1.4's F4).
- Tests: `tests/parity_fno.rs`: 2.0.0's defaults `fno1d` (Burgers-shaped) 3.5e-7, `fno2d`
  (Darcy-shaped) 4.1e-7; sweeps `fno3d` 4.5e-7, `fno2d_instance_pre` 6.3e-7,
  `fno1d_group_skips` 2.6e-7, `fno2d_separable` 8.1e-7, `fno2d_odd_stabilizer` 1.8e-6,
  `fno1d_pre_no_mlp` 3.4e-7 (D3 1e-5, f32, Flex, M2 Pro); the stabiliser left off fails at
  1.4e-3; swapped corners fail (control). `_f64.rs` `#[ignore]`d: 1.2e-8, 3.8e-9. Unit tests
  (`models/fno/tests/spectral.rs`): options off build today's parameters, the bias's shape,
  separable and `max_modes` storage, the two `max_modes` panics.
- Checks: ran `cargo fmt --check`, clippy (default, `--examples`, `--all-targets`,
  `--features datagen --all-targets`), `cargo test` (546 passed), `cargo test --features
  datagen` (633 passed), `cargo doc --no-deps`, `generate.py --check` (every fixture the
  same), the bit-identity dumps, `cargo test --features metal --test gpu_parity` (see the
  PR).
- Decisions: none. Question for sign-off: 2.0.0's `fno_skip=None` (no FNO skip) is not
  representable, since `FNOConfig::fno_skip: None` is the original 1×1 convolution; add it?
- Review: plan review PASS (advisories: the off value of `max_modes` and its wrong-rank panic
  tested). Code quality APPROVE with findings: F1, F2, F3, F5, F9, F10, F12 fixed; F4, F6, F7
  to CM1.5d's line; F8 (the struct literals) already there; F11 accepted (integration tests
  cannot see the crate's `pub(crate)` permutes).
- FNO: unchanged bit for bit (`fno_bit_identity` dumps on `main` 73afdd3 and the branch,
  before and after the review fixes: `forward`, `phase5_epoch`, `gauss_newton_epoch` `cmp`
  same); existing tests edited only to complete `FNOConfig` literals with `None`.
- Matrix: `FNO` (CM1.5a–c), centred `SpectralConv`, `FNOBlocks` rows updated.
- Follow-ups: CM1.5b (padding, resampling), CM1.5c (batch norm; `fno_skip=None` on the
  author's answer), CM1.5d's line (F4, F6, F7 with the earlier tidies).

## 2026-11-27: CM1.5c Batch norm, and no FNO skip
- Done: `NormKind::Batch` (Burn's `BatchNorm`: torch's eps, momentum and initial values;
  the running statistics without autodiff, as torch's eval) and, on the author's answer to
  CM1.5a's question 1, `SkipKind::Disabled` (2.0.0's `fno_skip=None`): `SkipKind::init`
  returns an `Option`, and `FNOConfig::fno_skip: Some(Disabled)` builds neither a skip nor
  `w` (`fno_skip: None` is still the 1×1 convolution with bias). Two f32 fixtures from
  `generate.py` (`BATCH_NORM`, the running statistics redrawn, the variance positive;
  integer `num_batches_tracked` left out of the fixture). The author's M1-gate rule for
  boolean options recorded on CM1.5d's line.
- Tests: `tests/parity_fno.rs`: `fno2d_batch_norm` 1.5e-6, `fno1d_pre_no_skip` (batch norm,
  pre-activation) 3.0e-7 (D3 1e-5, f32, Flex, M2 Pro); the first two norms' running
  variances swapped fail (control). Unit tests: the batch norm's initial values and its eval
  closed form; `Disabled` builds nothing; an `FNO` with it has no `w` and runs.
- Checks: ran `cargo fmt --check`, clippy (default, `--examples`, `--all-targets`,
  `--features datagen --all-targets`), `cargo test` (562 passed), `cargo test --features
  datagen` (653 passed), `cargo doc --no-deps`, `cargo test --features metal --test
  gpu_parity` (9) and `--test parity_fno` (12), `generate.py --check` (72 fixtures the same).
  One `cargo test` run failed `training::trainer::tests::lbfgs_smoke_run_resumes_exactly`
  (CO2.1's; train L² rose over three epochs); it passed alone three times and in the next
  full run: flaky under parallel tests, outside this task (`training/`), reported in the PR.
- Decisions: the author's of 2026-11-27 (CM1.5a's questions 1 and 2), recorded on CM1.5c's
  and CM1.5d's lines.
- Review: plan review PASS (advisories fixed: the control's wording, the initial running
  statistics asserted, the frozen-layer wording). Code quality APPROVE with findings: F2
  (indexing keeps a mismatch a panic), F3 (training hazards in `Norm::Batch`'s rustdoc and
  the roadmap), F4 (one remap helper), F5 (`named_buffers`, `SWEEPS` defined once), F6
  fixed; F1 (`SkipKind::init`'s `Option` and the `.unwrap()`s in tests) kept, with
  `Some(Disabled)` documented on the channel MLP's skip.
- FNO: unchanged bit for bit (`fno_bit_identity` dumps on `main` f7e769a and the branch,
  separate target directories, `forward`, `phase5_epoch`, `gauss_newton_epoch` `cmp` same).
- Matrix: `FNO`, `FNOBlocks`, `skip_connection`, the norms rows.
- Follow-ups: training with batch norm (CM1.5c's line: Burn's biased running variance,
  Gauss–Newton, L-BFGS, data parallel), a task only if a batch-norm model is trained.
