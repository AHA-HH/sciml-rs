# T17: `y_train` stored in physical units ⚑

Phase 4s, task 17. Branch `phase4s/T17-y-train-physical`. Depends on T13. Can run in
parallel with T18 and T19, with small overlaps in the examples, so rebase.

> **⚑ Decision needed before any code changes: CONVENTIONS §10.** Record the answer in design
> §12 (a new decision), and propose the convention change in the PR description as CLAUDE.md
> requires.
>
> **Refresh at start:** re-read `train_fno` (T3), `evaluate` (T13), `loaders::common`
> (T9) and `Postprocess` / `identity` on `main`, and list the hook's remaining callers.

## Read first
- `docs/CONVENTIONS.md` §10 (the y asymmetry) and §9 (the version-bumping set is §1–§6 and
  §12, so §10 is outside it).
- Review §6 T3-1, §7 step D1.
- `training/trainer.rs`:
  - `Postprocess = dyn Fn(...)` :49 and `identity` :51
  - the `train_post` / `eval_post` pair threaded through `train_epoch`, `eval_epoch` and
    `training_loop`
- `data/transforms/normalizers.rs::FlatDecoder`.

## Decision: store `y_train` in physical units and decode only the prediction?
**Today:**
- `y_train` is stored encoded and `y_test` physical.
- Training decodes both prediction and target, while eval decodes only the prediction.
- This takes two closures and a public `identity`, and every caller has to wire them
  correctly.

**Proposed:**
- `y_train` is stored physical, like `y_test`.
- One hook, `Option<FlatDecoder>`, is applied to the model output in both training and eval.
- `Postprocess`, `identity` and the closure pair are deleted.

| Option | Consequence |
| --- | --- |
| **Accept** (recommended) | The training loss is unchanged mathematically. Numerically it changes only by f32 round-off: the target is no longer encode→decoded, and the forward pass is identical. The loader encodes only x (and fits y). CONVENTIONS §10 prose changes, with no `CONVENTION_VERSION` bump. |
| Bolder variant | Put the decoder inside the model, so `load_fno` / `predict_host` return physical units. This changes the model's record format, so it is out of scope for 4s. Mention it as a follow-up only. |
| Decline | Keep the closures. T13's single `evaluate` has already shrunk the callers. |

**Recommendation: accept.** The round-off change is invisible next to training noise. Check
it with the test below.

## Deliverables (if accepted)
1. **`normalize_split_pair`** (T9) no longer encodes `y_train`. It still fits the y
   normaliser on the physical `y_train`.
2. **The training path:**
   - `train_epoch`, `eval_epoch` / `evaluate` and `training_loop` take
     `decoder: Option<&FlatDecoder>`, applied to `out` only.
   - Delete `Postprocess`, `identity` and the pair.
   - Callers to update: `train_fno`; the learner test that uses `identity` (learner.rs
     :75, :129); `predict_darcy`'s third copy (`predict/darcy.rs:66-67` before T13).
3. **CONVENTIONS §10:** "`y_test` stays in physical units …" becomes "both y splits are
   stored in physical units; the prediction is decoded before the loss". Update the
   "Verification" names if tests are renamed.
4. **Saved runs:** `y_normalizer.json` keeps its meaning, so old runs still evaluate. Check
   that no predict path relied on `y_train` being encoded.

## Tests
- `train_loss_matches_old_asymmetric_path`: one epoch on a tiny Darcy-shaped set, old path
  against new, with relative difference ≤ 1e-6 (f32 round-off). Write it against the old
  code first, then keep it with the old path inlined in the test as the oracle.
- `normalizer_asymmetry_matches_darcy` is renamed or updated to the new contract. Name the
  rename in the PR.
- `inputs_hold_the_normalized_coefficient_only` and the loader tests pass.

## Acceptance
- `predict_darcy` and `predict_poisson` on existing runs print the same numbers to the
  printed precision, because evaluation was already "decode the prediction only".
- A short `train_darcy` run (epochs lowered locally) shows train-loss curves matching
  before/after to about 1e-6 relative over a few epochs.

## Checks to run and report
The six from the README. The chebyshev pair is not needed.

## Do not
- Put the decoder inside the model (the bold variant).
- Change the x normalisation or `LpLoss`.
