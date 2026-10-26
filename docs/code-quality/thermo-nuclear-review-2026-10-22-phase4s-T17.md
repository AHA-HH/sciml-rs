# Thermo-nuclear code quality review: Phase 4s T17 (`y_train` stored in physical units)

- **Repo:** sciml-rs (`AHA-HH/sciml-rs`)
- **Branch:** `phase4s/T17-y-train-physical` (HEAD `ab7df42`, PR #55)
- **Base:** `main` (`703f7ea`). The range is `git diff main...phase4s/T17-y-train-physical`, one commit.
- **Date:** 2026-10-22
- **Diff size:** 13 files, +202 / −136. Of that, `src` and `examples` are 10 files, +174 / −132 (PR body: prod +81 / −77, test +93 / −55). The rest is docs (CONVENTIONS §10, design decision 28, the phase README).

## Verdict: APPROVE (with follow-ups)

This is the deletion the brief asked for, and it is done in the right layer:
- `Postprocess`, the public `identity` and the train/eval closure pair are gone. The hook is one `Option<&FlatDecoder>`, applied to the output only, in one place (`forward_loss`, `step.rs:100-103`).
- Callers can no longer wire training and evaluation inconsistently: `training_loop` takes one decoder and derives the evaluation copy itself (`trainer.rs:234`).
- The loader change is a one-line deletion (`common.rs:61`). The normaliser is still fitted on the same physical `y_train`, and the T9 pins confirm it: only the `train.targets()` checksum moved.
- No predict path reads `y_train`, so saved runs keep their meaning.

No file crosses 1000 lines. No branching was added: one `match` replaces a closure call. The public surface shrinks by one type alias and one function and grows by one method (`FlatDecoder::inner`).

What keeps this from a clean approve is what the PR left behind, not what it added. None of it is a presumptive blocker:
- `train_built` is now a two-line pass-through.
- The test pinning it, `train_fno_none_matches_identity_path`, compares a function with its own body.
- Decision 28 cites acceptance evidence in a form that does not match what was measured.

## Presumptive blockers

None.
- No file crosses 1000 lines (`trainer.rs` 732 → 763; the growth is the new oracle test).
- No new ad-hoc branching. The one `match` in `forward_loss` replaces the closure call and sits in the one function that owns the loss.
- No wrapper, cast or optionality was added. `Option<&FlatDecoder>` replaces `&dyn Fn` and is narrower than it.
- No duplicated helper. `FlatDecoder::inner` uses Burn's `Tensor::inner` (idempotent in the pinned fork, `burn-tensor/src/tensor/api/autodiff.rs:183-191`) and has no existing equivalent.

## File-size table

| File | Before → after | Note |
| --- | --- | --- |
| `docs/design/2d-chebyshev-poisson-fno.md` | 1325 → 1347 | already over 1k (a doc; decision 28 appended, house format) |
| `src/neural_operators/training/trainer.rs` | 732 → 763 | +31: the inlined-oracle test |
| `src/neural_operators/data/loaders/poisson.rs` | 719 → 718 | |
| `docs/CONVENTIONS.md` | 327 → 329 | |
| `src/neural_operators/data/loaders/darcy.rs` | 327 → 327 | |
| `src/neural_operators/data/transforms/normalizers.rs` | 318 → 327 | `FlatDecoder::inner` |
| `src/neural_operators/training/step.rs` | 239 → 244 | |
| `docs/phase4s/README.md` | 234 → 234 | |
| `src/neural_operators/training/learner.rs` | 187 → 187 | |
| `examples/predict/burgers_resolution.rs` | 102 → 102 | |
| `src/neural_operators/data/loaders/common.rs` | 90 → 89 | |
| `examples/predict/darcy.rs` | 66 → 65 | |
| `examples/predict/burgers.rs` | 57 → 57 | |

## Findings

### 1. Structural regressions

None.

### 2. Missed code-judo simplifications

**2.1 `train_built` is now a pass-through, and the test that pins it is a tautology** (`trainer.rs:324-335`, `trainer.rs:515-540`).
- Before T17, `train_built` held real logic: it chose between `identity`/`identity` and a pair of closures built on two devices. After T17 its body is:

  ```rust
  let decoder = y_norm.map(|y| FlatDecoder::new(y, device));
  training_loop(components, train_cfg, decoder.as_ref(), on_epoch)
  ```

- Its stated reason, "Split out so tests can give two runs the same initial model", no longer holds. The only test that needs `Some` (`train_loss_matches_old_asymmetric_path`) already builds `FlatDecoder::new(&y, &device)` for its own oracle at `:585`, so it can call `training_loop` with that decoder directly.
- `train_fno_none_matches_identity_path` (`:517-540`) runs `train_built(new, None, …)` against `training_loop(old, …, None, …)`. With `None`, the first is the second, line for line. The test pins nothing, and its doc ("against the old `train_burgers` body") claims an oracle that is no longer there. The PR touched this test and kept it.
- **Move:**
  - Inline the `map` into `train_fno` (`:319`: `let decoder = y_norm.map(|y| FlatDecoder::new(y, device)); let trained = training_loop(components, train_cfg, decoder.as_ref(), &mut on_epoch);`).
  - Delete `train_built` and `train_fno_none_matches_identity_path`.
  - In `train_loss_matches_old_asymmetric_path`, call `training_loop(new, &train_cfg, Some(&decoder), …)`, hoisting the `decoder` it already builds.
- `train_fno`'s `Some` branch stays covered end to end by `train_poisson_smoke` (`:665`), and its `None` branch by `metrics_csv_holds_every_epoch` (`:717`).
- Net: about −35 lines and one fewer private function. The lib test count drops 209 → 208, which goes in the README row.

**2.2 (Option, needs a check first.) One decoder on the inner device could serve both training and evaluation** (`normalizers.rs:138-145`, `trainer.rs:233-234`).
- The pinned fork documents that a non-autodiff tensor "combined with a tensor that still uses autodiff … is treated as a constant for that operation" (`autodiff.rs:195-196`). If that holds for `out * scale + mean` on Flex and Metal, then a `FlatDecoder` built on `device.inner()` decodes autodiff outputs too, with the gradient flowing through `out`.
- The gain: `FlatDecoder::inner`, the `eval_decoder` line in `training_loop` and the "on the training (autodiff) device" clauses in three docs (`trainer.rs:56-57`, `:222-223`, `step.rs:118`) all go away. Every caller (`train_fno`, `predict_darcy`) would build its decoder the same way, on the plain device.
- **Do not take this on prose alone.** First add a test: one `train_epoch` with an inner-device decoder against one with an autodiff-device decoder, bit-equal metrics on Flex, and run it once with `--features metal`. If the fork rejects the mixed op or the bits differ, keep `inner()` as it is.

**2.3 (Considered, not recommended.)** `build_training_components` already computes `train_device` and `eval_device` (`trainer.rs:149-155`), so `TrainingComponents` could own both decoders. That moves the device pairing to where the devices are named, but it adds a parameter to a builder that ten tests call, and a field pair to a struct. It does not delete more than 2.1 and 2.2 do. Recorded so it is a declined option.

### 3. Spaghetti / branching growth

None. The only new conditional is `forward_loss`'s `match decoder` (`step.rs:100-103`), and the function exists so that the loss is defined once. `decoder.map_or(out, …)` cannot replace it without a clone, because `out` is moved, so the `match` is the direct form.

### 4. Boundary, abstraction and type-contract problems

**4.1 `FlatDecoder::inner` is public but pinned only to a tolerance** (`normalizers.rs:138-145`).
- `train_loss_matches_old_asymmetric_path` compares the new path, which uses `inner()`, with an oracle built by `FlatDecoder::new(&y, &device.inner())` (`trainer.rs:604`), at 1e-6 relative. A bit-level difference between the two constructions would pass unnoticed.
- They should be bit-identical: the same upload and `+ eps` run on the same inner backend, and `without_autodiff` only drops the graph.
- If 2.2 is not taken, add a three-line test in `normalizers.rs`: `FlatDecoder::new(&n, &ad).inner().decode(x)` against `FlatDecoder::new(&n, &ad.inner()).decode(x)`, bit for bit. Also add "idempotent: a decoder already on a plain device is returned unchanged" to the doc. That is what makes passing a plain-device decoder to `training_loop` safe.

**4.2 The decoder's device is a convention in prose, not in the type** (`trainer.rs:56-57`, `:222-223`; `step.rs:118`). `train_epoch` wants an autodiff-device decoder and `evaluate` wants a loader-device one, and nothing checks either. This is acceptable at this size, and 2.2 deletes it. Noted only so that 2.2 is weighed against it.

### 5. File size and decomposition

None. `trainer.rs` is at 763 lines, about 430 of them tests. Taking 2.1 brings it back to about 730.

### 6. Modularity

**6.1 The acceptance evidence that decision 28 cites does not exist in the cited place, and is worded more strongly than measured** (`docs/design/2d-chebyshev-poisson-fno.md:1326-1329`, `docs/phase4s/README.md:219-220`).
- Decision 28 says the check includes "a short `train_darcy` run whose train-loss curve matches the old one to about 1e-6 relative (`docs/phase4s/README.md` Results)". The README Results table has no such entry. Its two "unchanged after" rows (`:219`, `:220`) still list only T10, T11 and T13, although the PR body shows `predict_darcy` and `predict_poisson` unchanged on this branch.
- The PR body's own `train_darcy` table meets 1e-6 at epoch 0 only. Epochs 1–2 are at 1.5e-5 to 1.0e-4 relative. The explanation (round-off compounding over about 750 Adam steps per epoch) is plausible. However, main's run-to-run spread was not measured, so "matches to about 1e-6" is not what was shown.
- **Fix (docs only):**
  - Add T17 to the README rows at `:219` and `:220` (the same numbers as before, with T17 named).
  - Add the `train_darcy` per-epoch differences to the Results table.
  - Reword decision 28 to "epoch 0 within 1e-6, later epochs within 1e-4 (round-off compounding); the per-step equivalence is pinned at 1e-6 by `train_loss_matches_old_asymmetric_path`".

**6.2 `learner.rs:12-14`** still says datasets that need a decoder "are not covered: the step passes no decoder". This is accurate. Since the hook is now a plain `Option`, a Darcy `Learner` would only need a decoder field on a wrapper. That is out of scope here; noted for T2 or whoever next touches the `Learner` showcase.

### 7. Legibility

**7.1 Task archaeology in the pinned checksums** (`darcy.rs:316`, `poisson.rs:478`: `// y_train physical since 4s T17`). Git blame and decision 28 already hold that history. The pin's neighbours carry no such notes, so the comment will read as noise after the next re-pin. Drop it, or say what the value is ("physical `y_train`").

**7.2 The `train_loss_matches_old_asymmetric_path` oracle repeats `train_epoch`'s body** (`trainer.rs:584-612`, about 30 lines). The brief requires this ("keep it with the old path inlined in the test as the oracle"), and an oracle should not call the code under test. This is the right choice and needs no change. It only checks one epoch, which is what the brief specifies.

## Brief acceptance (T17)

| Item | Status |
| --- | --- |
| `normalize_split_pair` no longer encodes `y_train`, still fits `y` | Met (`common.rs:61`; pins show both normalisers bit-identical) |
| `train_epoch`, `evaluate`, `training_loop` take `Option<&FlatDecoder>`, output only | Met (`trainer.rs:64`, `:230`; `step.rs:96`, `:126`) |
| `Postprocess`, `identity` and the pair deleted; callers updated | Met. `git grep 'Postprocess\|identity'` finds no live use in `src` or `examples` |
| CONVENTIONS §10 prose, no `CONVENTION_VERSION` bump, Verification names | Met (§10 is outside the bumping set, §9) |
| Saved runs still evaluate | Met (no predict path reads `y_train`) |
| `train_loss_matches_old_asymmetric_path` ≤ 1e-6 | Met (`trainer.rs:550`) |
| `normalizer_asymmetry_matches_darcy` renamed and named in the PR | Met (`normalization_matches_darcy`) |
| `predict_darcy` / `predict_poisson` unchanged | Reported in the PR body. **Not recorded in the README Results rows** (6.1). Not re-run in this review (no datasets or runs needed for the review) |
| Short `train_darcy` curve about 1e-6 | Partly: epoch 0 only; see 6.1 |
| Decision recorded in design §12 | Met (decision 28) |

## Checks run (local, macOS, Flex backend)

All passed:
- `cargo fmt -- --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo clippy --no-deps --examples -- -D warnings`
- `cargo test`: lib 209 tests (208 passed, 1 ignored), plus the integration and doc-test binaries, all ok
- `cargo doc --no-deps`

Not run:
- `--features datagen`: the brief says it is not needed, and no gated code changed.
- `--features metal`: no backend-sensitive op changed. Re-run it if 2.2 is taken.
- The predict and train examples: they need `datasets/` and `runs/`. The PR body reports them.

## Next steps

Each step preserves behaviour. Steps 1 and 3 are one small commit on this branch. Step 2 is optional and can be its own commit.

1. **Delete `train_built` and the tautological test** (2.1).
   - Inline the decoder construction into `train_fno`.
   - Point `train_loss_matches_old_asymmetric_path` at `training_loop` with the decoder it already builds.
   - Delete `train_fno_none_matches_identity_path`.
   - Verify: `cargo test` gives lib 208 (one deliberate deletion), and `train_poisson_smoke`, `metrics_csv_holds_every_epoch` and `train_loss_matches_old_asymmetric_path` pass. `cargo clippy --all-targets -- -D warnings` is clean. Update the README T17 test-count note.
2. **Optional: one inner-device decoder** (2.2), or else the `inner()` bit test (4.1).
   - First write the test: `train_epoch` with an inner-device decoder against an autodiff-device decoder, metrics bit-equal on Flex. Run it with `--features metal` too.
   - Only if it passes, delete `FlatDecoder::inner` and `eval_decoder`, and simplify the three device clauses in the docs.
   - If it fails, keep `inner()` and add the bit test of 4.1 instead.
3. **Docs** (6.1, 7.1).
   - Add T17 to README Results rows `:219` and `:220`, plus the `train_darcy` per-epoch differences.
   - Reword decision 28's acceptance sentence to what was measured.
   - Drop the two `since 4s T17` checksum comments.
   - Verify: by reading the diff; `cargo fmt -- --check` if a Rust comment changes.
