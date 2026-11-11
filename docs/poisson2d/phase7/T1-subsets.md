# T1: nested training subsets

Phase 7, task 1. Branch `phase7/T1-subsets`. Depends on Phase 5 T7 (merged).

## Read first
- `src/neural_operators/data/loaders/poisson/mod.rs` (`PoissonConfig`, `DatasetSource`)
  and `uniform.rs` (`load_poisson_uniform`, its tests, `poisson_output_is_pinned`).
- `src/neural_operators/data/loaders/common.rs` (`normalize_split_pair`, `check_n_train`).
- `src/neural_operators/training/config.rs` (`TrainingConfig`) and `trainer.rs`
  (`steps_per_epoch`, the cosine schedule).
- `examples/train/poisson.rs`; `Cargo.toml` (`rand_chacha` under `datagen`).
- Design §13.5 (nested subsets), §13.9 (the `data_cfg.json` and `train_cfg.json` fields);
  CONVENTIONS §10; this phase's README (Planning decisions).

## Goal
Train on N samples drawn from the stage 3 train split by a seeded permutation, such that
each smaller N is contained in every larger one for the same seed, with the normalisers
fitted on those N alone.

## Deliverables
1. **Subset selection** (ungated, in `data/loaders/poisson/`): for a train split of
   `count` samples, a subset seed and N ≤ `count`, the indices are the first N of a
   ChaCha8 Fisher–Yates permutation of `0..count`, then sorted. `rand_chacha` becomes a
   non-optional dependency at its pinned version (remove it from `datagen`'s list, keep
   the `=0.10.0` pin and the comment about the dataset format). Document the rule.
2. **`PoissonConfig`** gains an optional subset: the seed and a hash of the sorted
   indices (FNV-1a 64 over the indices as little-endian u64, written in hex; no new
   dependency). `None` (and every existing `data_cfg.json`) means today's prefix of
   `n_train`. N stays `base.n_train`.
3. **`load_poisson_uniform`** reads the whole train split when a subset is set, selects
   the indices before the transfer, and checks that the recomputed hash matches a
   configured one (an `Invalid` error, not a panic). The normalisers are fitted on the
   subset (`normalize_split_pair` unchanged). The test split is untouched.
4. **Total optimisation steps** recorded with the run (design §13.9): `epochs ×
   ceil(N / batch_size)`, in `train_cfg.json` (an `Option` field so old files load) or in
   `costs.json`; pick one and say why in the PR.
5. **`train_poisson -- <p> <n> <s> <seed> [<N> <subset_seed>]`:** N defaults to 1000
   with no subset (today's behaviour, bit-identical). With N given, the subset seed is
   required. The run name carries both when set, e.g.
   `poisson_fno_n129_s64_p0_N25_sub0_seed42`. Update the doc comment and `CLAUDE.md`'s
   Commands and Layout lines.
6. **D7 recorded:** confirm in the PR that design §13.15 and decision 30's 2026-10-29
   amendment state the threshold (done at planning); no change unless they disagree.

## Tests
- Nesting: for one seed, the index set at N = 25 ⊂ 50 ⊂ 100 ⊂ 250 ⊂ 500 ⊂ 1000.
- Reproducible: the same seed gives the same indices and hash; different seeds give
  different 25-sample sets.
- At N = count the indices are `0..count` whatever the seed.
- A pinned hash for one (seed, N), so a change to the RNG or the rule fails a test.
- On a synthetic split: the loaded subset equals the transfer of the selected samples,
  and the fitted normaliser mean equals the subset's mean, not the full split's.
- An old-format `data_cfg.json` (no subset) deserialises to `None`; a hash mismatch is
  rejected.
- `poisson_output_is_pinned` and every existing loader test unchanged.

## Acceptance
- With no subset, a stage 2 load gives tensors bit-identical to `main`.
- A short stage 3 run at s = 64, N = 25, subset seed 0 starts on flex (a few epochs; not
  a result) and writes N, the seed, the hash and the steps.
- `predict_poisson` evaluates that run unchanged.
- The T1 Results row filled in.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`.

## Do not
- Change the trainer loop, the evaluation cadence (per epoch, decided), the model or the
  hyperparameters.
- Change the test split or how the normalisers are fitted beyond feeding them the subset.
- Add a step-budget mode (D6 deferred it).
