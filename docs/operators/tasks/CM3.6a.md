# CM3.6a Tensor GaLore projector, real tensors

Started 2026-11-27 on `operators/cm3-6` (the author asked for CM3.6 that day; it needs only
CM0.3 and does not wait for the M1 gate). Roadmap: "CM3.6 GaLore. `tensor_galore_projector`.
*Needs:* CM0.3", split here into CM3.6a (real tensors) and CM3.6b (complex tensors).
Proposal: §10; §1.4, §2, §5, §6.4. D1, D2, D3 (as amended 2026-11-25), D8.

## Goal

2.0.0's `TensorGaLoreProjector` for real tensors of any rank: Tucker factors by 2.0.0's
higher-order orthogonal iteration (tensorly 0.9.0's `tucker`), the projection into the
low-rank space and back, matching 2.0.0 on committed fixtures within D3.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- Roadmap: no *Done when* of its own; the M3 recipe (proposal §6.4) and §5: output parity
  against fixtures. Here: the factors, `project` and `project_back` against 2.0.0 within D3
  (≤ 1e-5 f32, ≤ 1e-12 f64, CPU). No FFT, so the f64 test runs.
- Proposal §1.4, constraint 1: every existing FNO output bit for bit identical. The task
  adds a module and touches no existing code path.

## Why the split

2.0.0's projector also takes complex tensors (its main target, the FNO's spectral
weights): tensorly decomposes the whole centred `cfloat` weight, with a complex SVD and a
phase-normalised sign rule (`torch.sgn`). Burn has no complex SVD, and this crate stores
each spectral weight as real `weights_re`/`weights_im` per mode corner, so which tensor the
projector sees is a wiring question too. CM3.6b: the complex projector (host complex SVD,
or the real embedding [[B, −C], [C, B]]), parity on complex fixtures. *Needs:* CM3.6a.

## Read first

- 2.0.0 `neuralop/training/tensor_galore_projector.py`, `training/adamw.py` (its only user);
  tensorly 0.9.0 `decomposition/_tucker.py` (`tucker`, `partial_tucker`,
  `initialize_tucker`), `tucker_tensor.py` (`validate_tucker_rank`), `tenalg/svd.py`
  (`svd_interface`, `truncated_svd`, `svd_flip`), all in `tools/parity/.venv`.
- `src/neural_operators/optim/soap/mod.rs` (the module pattern; Burn's host `svd`).
- `tests/parity_common/`, `tools/parity/generate_losses.py` (fixture writing, `--check`).

## 2.0.0's behaviour, reproduced

- `project(grad, iter)` computes factors only when it has none and `iter % update_proj_gap
  == 0`; it never recomputes them (`proj_tensor` is never reset). So `update_proj_gap` only
  gates the first call, `warm_restart` never applies (it needs existing factors inside the
  recomputation that never runs), and `tucker_n_iter_max` is never passed to `tucker`
  (tensorly's defaults apply: 100 iterations, tol 1e-4). Reproduced as is; the PR asks the
  author whether a later task should add the intended periodic refresh.
- `tucker(x, rank, init="svd")`: rank validation (`validate_tucker_rank`: an int for every
  mode; a float a fraction c of the parameters, c the root of N cᴰ + Σ Iₖ² c = rank · N on
  [0, max(rank, 1)], ranks max(round-half-even(Iₖ c), 1); a list as is); SVD
  initialisation (leading `rank` left singular vectors of each mode-k unfolding); HOOI sweeps
  (each factor from the unfolding of x multiplied by the other factors transposed); stop
  after iteration ≥ 2 when the change in √|‖x‖² − ‖core‖²| / ‖x‖ is below 1e-4.
- Every SVD's U is sign-fixed (`svd_flip`, U-based): each column's largest-magnitude entry
  (first on a tie) made positive.
- `project` = x ×ₖ Fₖᵀ for every mode k; `project_back` = scale · (y ×ₖ Fₖ).
- Not reproduced: `activation_checkpoint` (a PyTorch memory option), the device move of the
  factors (they are made on the gradient's device), tensorly's full-matrix SVD when a rank
  exceeds the product of the other dimensions (rejected by a panic instead; never the case
  for weight gradients).

## Changes

- New `src/neural_operators/optim/galore.rs` (imports nothing from `models/` or
  `training/`): `GaLoreRank` (`Fraction(f64)`, `Uniform(usize)`, `PerMode(Vec<usize>)`,
  `ranks(shape)`), `TensorGaLoreConfig` (`rank`, `update_proj_gap` 200, `scale` 1.0;
  `init::<D>()`), `TensorGaLoreProjector<D>` (`project`, `project_back`, `factors`),
  `tucker_factors` (the HOOI) and `multi_mode_dot`. Unit tests in the file.
- `optim/mod.rs`: the module and its line.
- `tools/parity/generate_galore.py` (+ `--check`), fixtures `tests/fixtures/parity/galore_{f32,f64}.npz`
  with sidecars; `tools/parity/README.md` rows.
- `tests/parity_galore.rs`, `tests/parity_galore_f64.rs` (Flex in f64, as the other f64
  binaries).
- CLAUDE.md Layout line, feature matrix row, roadmap split and tick, progress entry.
- Conventions: none.

## Tests that prove it

- Parity (fixtures, D3): three cases, each a structured gradient (Tucker low-rank plus
  noise, so the HOOI converges in a few sweeps and the leading singular values are
  separated): `matrix` (12 × 7, int rank 3, gap 5), `tensor4` (6 × 5 × 4 × 3, ranks
  [3, 2, 3, 2], scale 0.25, a spectral-weight shape), `fraction` (8 × 6 × 5, rank 0.5).
  Each: the ranks; the factors; `project(grad, 0)`; `project_back` of it; and
  `project(grad2, gap)` with the first factors (2.0.0 does not refresh).
- Control: the factors of another case's gradient fail parity.
- Unit: ranks (`Uniform`, `PerMode`, `Fraction` against hand-solved values, the panics);
  orthonormal factors; full ranks make `project_back ∘ project` the identity; an exact
  Tucker tensor of rank r is reproduced; the sign rule; no refresh at the next gap; the
  panics (first call off the gap, `project_back` before `project`, a rank above its
  dimension).

## Checks

`/verify` (fmt, clippy default/`--examples`/`--all-targets`/datagen, `cargo test`,
`cargo test --features datagen`, `cargo doc --no-deps`), `uv run python generate_galore.py
--check`, the existing fixtures' `--check` untouched. Constraint 1: no existing file under
`src/` changes except `optim/mod.rs`'s `pub mod` line, so FNO outputs cannot change; the
`fno_bit_identity` dumps are run anyway if `/verify` asks for them. `CARGO_TARGET_DIR=~/sciml-rs-target/cm3-6`.

## VM runs

None.

## Wiring later (not in this task; `training/` belongs to CO1.5, then CM0.6 and CO3.3)

2.0.0 uses the projector only inside its `AdamW`: per GaLore parameter, `project` the
gradient at the step count, run Adam's moments in the low-rank space, `project_back` the
normalised step (with `scale`), then the usual update and decoupled weight decay. Here that
needs: an `OptimizerConfig` arm (say `AdamwGalore { rank, update_proj_gap, scale }`) and
the parameter selection (2.0.0 passes `galore_params` explicitly; the neuralop examples
pick the spectral weights, which needs CM3.6b and a choice of whether the corners'
`weights_re`/`weights_im` form one complex tensor as in 2.0.0); a Burn per-parameter
`Optimizer` like `optim/soap`, whose state holds the projector's factors (a `Record` by hand,
as `SoapState`) so checkpoints resume; the run's `costs.json` unchanged.

## Constraints and risks

- Constraint 1: holds by construction (new module only).
- Burn's `svd` is host-side and panics on a tensor that requires gradients: the projector
  takes plain (inner-backend) gradients, as Burn's `GradientsParams` give them.
- Risk: f32 rounding could change the HOOI's stopping iteration; the structured gradients
  keep the stop well inside tolerance.

## Pending runs

None.

## Open questions

1. Should a later task add the refresh 2.0.0 evidently intends (new factors every
   `update_proj_gap` steps, `warm_restart` starting the HOOI from the old ones), as an
   option off by default? Not needed for parity.
