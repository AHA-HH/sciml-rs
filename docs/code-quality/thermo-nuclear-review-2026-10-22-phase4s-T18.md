# Thermo-nuclear code quality review: Phase 4s T18 (predict reads the saved normalisers)

- **Repo:** sciml-rs (`AHA-HH/sciml-rs`)
- **Branch:** `phase4s/T18-x-normalizer` (HEAD `023b0bb`, PR #56)
- **Base:** `phase4s/T17-y-train-physical` (stacked; PR #55). The range is `git diff phase4s/T17-y-train-physical...phase4s/T18-x-normalizer`, one commit.
- **Date:** 2026-10-22
- **Diff size:** 10 files, +320 / −100. Of that, `src` and `examples` are 8 files, +291 / −96 (PR body: prod +243 / −96, test +48 / −0). The rest is docs (design decision 29, the phase README Results rows).

## Verdict: BLOCKED

The decision is implemented in the right layer, and most of it is good:
- The encode step is shared, not copied. `encode_split` (`common.rs:93-99`) is the one place that encodes inputs, adds the channel axis and casts. `normalize_split_pair` and the new `normalize_test_split` both call it. The T9 pins (`darcy_output_is_pinned`, `poisson_output_is_pinned`) still pass unchanged, so the refactor is bit-identical.
- The split readers moved out of closures into named functions (`darcy.rs:120`, `poisson.rs:193`), so the two entry points per PDE share one reader.
- A normaliser fitted at another resolution becomes `LoadError::Invalid` instead of an ndarray broadcast panic (`common.rs:81-87`).
- `poisson_resolution` is untouched, and its refit already cites decision 18 (`poisson_resolution.rs:79`).

One presumptive blocker stands: the PR copies validation logic instead of reusing it.
- The Darcy test-split loader repeats both of `load_darcy_uniform`'s checks: the `subsample_rate` guard and the grid check.
- In doing so it drops the file's own pattern. `darcy_from_fields` exists "so the output can be tested without `.mat` files" (`darcy.rs:147-149`), but the new tail is inlined into the function that reads the file.
- As a result `load_darcy_test`, the entry point `predict_darcy` and `darcy_plot` now depend on, has no test. Its doc claims bit-for-bit equality with the refit path, and nothing checks that claim.
- The fix is small and deletes code.

## Presumptive blockers

1. **Duplicated validation, and the canonical home was skipped** (`darcy.rs:96-98` and `darcy.rs:105-111`, against `darcy.rs:69-71` and `darcy.rs:157-170`). See finding 2.1.

Not met:
- No file crosses 1000 lines (the largest code file, `loaders/poisson.rs`, goes 718 → 805).
- No ad-hoc branching was added to an existing flow.
- No wrapper, cast or optionality was added. `encode_split` is a one-line private helper, but it is the shared step the brief asked for, so it earns its keep.

## File-size table

| File | Before → after | Note |
| --- | --- | --- |
| `docs/design/2d-chebyshev-poisson-fno.md` | 1348 → 1373 | already over 1k (a doc; decision 29 appended, house format) |
| `src/neural_operators/data/loaders/poisson.rs` | 718 → 805 | +87: the test loader, `check_transfer_n`, `read_uniform_split`, one test (+39) |
| `src/neural_operators/runs/mod.rs` | 385 → 395 | test extension only |
| `src/neural_operators/data/loaders/darcy.rs` | 327 → 369 | +42, all prod; no test added |
| `examples/predict/poisson.rs` | 279 → 285 | |
| `docs/phase4s/README.md` | 234 → 234 | |
| `examples/predict/poisson_plot.rs` | 189 → 196 | |
| `src/neural_operators/data/loaders/common.rs` | 89 → 120 | `normalize_test_split`, `encode_split` |
| `examples/predict/darcy_plot.rs` | 93 → 98 | |
| `examples/predict/darcy.rs` | 65 → 72 | |

## Findings

### 1. Structural regressions

None. The data path got simpler: predict no longer reads (or, for Poisson, transfers) a split only to rebuild statistics the run already saved.

### 2. Missed code-judo simplifications

**2.1 (Blocking.) `load_darcy_test` copies two checks and skips the testable-tail pattern** (`darcy.rs:91-113`).
- `if config.subsample_rate == 0` appears twice, at `:69-71` and `:96-98`. The value it guards is `read_split`'s `r` parameter. With `r == 0`, `take_subsampled` would slice with step 0 and panic. So the guard belongs at the top of `read_split`, before any I/O. Moving it there deletes both copies, and `zero_subsample_rate_is_invalid_not_a_panic` (`:231`) still holds, because `read_split` runs after `check_n_train` and before `MatFileReader::new`.
- The grid check at `:105-111` re-implements the check in `darcy_from_fields` (`:157-170`) with a different message. It also reports `x_test`'s shape even when `y_test` is the field that failed.
- `darcy_from_fields` was split out of the reader so that its output can be tested without `.mat` files (`:147-149`). The new loader does not follow that split: its tail is inlined after `read_split`, so it cannot be tested without a `.mat` fixture, and it has no test. The only bit-for-bit test is the Poisson one (`poisson.rs:542`). Darcy's half of the claim "with the normalizers `load_darcy_uniform` fits, the dataset equals its test split bit for bit" (`darcy.rs:83-84`) is unpinned.
- **Move.** Make the tail look the same for both entry points:

  ```rust
  fn read_split(path, split, n, r) -> Result<..> {
      if r == 0 { return Err(LoadError::Invalid("subsample_rate must be > 0".into())); }
      ...
  }

  /// Every field is [_, s, s] with s = config.s().
  fn check_grid(config: &DarcyConfig, fields: &[&ArrayD<f64>]) -> Result<(), LoadError> {
      let s = config.s();
      match fields.iter().find(|a| a.shape()[1..] != [s, s]) {
          Some(a) => Err(LoadError::Invalid(format!(
              "subsampled grids must all be {s}x{s} (config.s()), got {:?}", &a.shape()[1..]))),
          None => Ok(()),
      }
  }

  fn darcy_from_fields(x_train, y_train, x_test, y_test, config) -> Result<(.., .., Normalizers), _> {
      check_grid(config, &[&x_train, &y_train, &x_test, &y_test])?;
      Ok(normalize_split_pair(x_train, y_train, x_test, y_test))
  }

  fn darcy_test_from_fields<T>(x_test, y_test, config, normalizers) -> Result<OperatorDataset<T>, _> {
      check_grid(config, &[&x_test, &y_test])?;
      normalize_test_split(x_test, y_test, normalizers)
  }
  ```

  - `load_darcy_uniform` becomes `check_n_train`, two `read_split`s and `darcy_from_fields`.
  - `load_darcy_test` becomes one `read_split` and `darcy_test_from_fields`.
- **Then add the missing Darcy test.** Next to `inputs_hold_the_normalized_coefficient_only` (`:272`), build the fixture fields and call `darcy_from_fields`. Assert that `darcy_test_from_fields(x_test, y_test, &cfg, &norms)` equals the returned test split bit for bit. This needs no `.mat` file and no `RunDir`: the Poisson test already covers the JSON round trip.
- **Check first:** no test asserts the old message text "got train … / test …" (`grep -n "must all be" darcy.rs` finds only `:165`). If the author wants to keep the train/test wording, `check_grid` can take a label. That is not needed.
- **Net:** about −10 prod lines and +15 test lines, one less copy of each check, and both PDEs pinned.

**2.2 (Considered, not recommended.)** Folding `normalize_test_split` into `normalize_split_pair` (fit, then call the test path twice) would put the shape check on the training path too, where it can never fail, because `x` is fitted on `x_train`. The current split is the direct one.

### 3. Spaghetti / branching growth

None in the library. In the examples, the four predict binaries each gained the same 10-line `normalizers()` → `unwrap_or_else(panic)` → `unwrap_or_else(|| panic!("{} has no normalizers; a … run has them"))` block (`examples/predict/darcy.rs:43`, `darcy_plot.rs:34`, `poisson.rs:141`, `poisson_plot.rs:100`). It replaces a 4-line `.expect` in each, so the four files grow by 24 lines of identical boilerplate. This is not blocking, because examples are leaf code. See 4.1 for the cheaper shape.

### 4. Boundary, abstraction and type-contract problems

**4.1 `RunDir::normalizers`'s `Option` leaks into every caller that needs both** (`runs/mod.rs:218-232`; the four example sites above).
- Since T18, every non-test caller treats `None` as fatal. Only Burgers has no normalisers, and no Burgers path calls this method.
- When both files are missing, the panic names the run but not the file. That falls short of the brief's "naming the run and the file", although the T18 brief only requires it for a missing `x_normalizer.json`, which is met.
- **Option (non-blocking):**
  - Keep `normalizers()` as is for the Burgers/`None` tests.
  - Have the examples call `.ok_or_else(|| format!("{}: no y_normalizer.json", run.path().display()))` inline, or add one `RunDir::require_normalizers() -> Result<Normalizers, Box<dyn Error>>` that returns that error.
  - Either way, each example's block shrinks to 3 lines and every failure names the file.
  - Only add the method if a fifth caller appears. Until then the inline `ok_or_else` is enough.

**4.2 `read_uniform_split` re-derives n from s** (`poisson.rs:199`: `let n = transfers.s() + 1;`). `Transfers::n()` already exists (`chebyshev/transfer.rs:215`) and reads it from the matrix. Use `transfers.n()`. This is the only `s() + 1` in the repository.

**4.3 Only `x` is shape-checked against the test split** (`common.rs:81-87`). A `y` normaliser at the wrong s still panics later, in `decode` in the examples. This is acceptable: `x` and `y` come from the same run and the same `data_cfg.json`, and the check exists for the encode that happens in the loader. Noted only so it is a deliberate choice. A one-line clause in the doc would make it explicit ("`y` is not used here; it decodes predictions downstream").

### 5. File size and decomposition

None. `loaders/poisson.rs` is at 805 lines, about 450 of them tests. It is worth watching: the next loader entry point added here should come with a split of the tests into `loaders/poisson/tests.rs`, not more growth.

### 6. Modularity

**6.1 The Poisson test checks two things** (`poisson.rs:542-575`). `test_loader_with_saved_normalizers_matches_refit` checks the bit-for-bit equality the brief names, then (`:563`) a separate rejection of a wrong-shape normaliser, with its own fixture file. Splitting the second half into `test_loader_rejects_normalizers_fitted_at_another_s` would make a failure name its cause. Non-blocking.

**6.2 The speed acceptance was not met, and the docs say so plainly** (decision 29; README `:219`). The brief expected "read it" to be faster. The measurement shows the train load is 0.167 s out of a 30 s run, so it is not. Reporting this honestly is the right call, and the decision stands on correctness: predict now uses the statistics the model was trained with. No change needed.

### 7. Legibility

**7.1 Section comment with task archaeology** (`poisson.rs:539`: `// --- phase 4s T18: predict encodes the test split with the run's saved normalizers ---`). The file already uses this style (`darcy.rs:204` cites `REVIEW1.md N1`), so it is consistent with the house style. No change.

**7.2 `nodes(n)` is now computed once per split instead of once per call** (`poisson.rs:201`). The cost is negligible (n ≤ 257 cosines), and it keeps `read_uniform_split` self-contained. No change.

## Brief acceptance (T18, "read it")

| Item | Status |
| --- | --- |
| Decision recorded in design §12 and the README row | Met (decision 29; README "Decisions taken" row) |
| Test-split loaders `load_darcy_test` / `load_poisson_uniform_test` taking `&Normalizers` | Met (`darcy.rs:91`, `poisson.rs:167`) |
| They share the encode step with `normalize_split_pair` | Met (`encode_split`, `common.rs:93`) |
| Four predict examples use `RunDir::normalizers()` and the test loader | Met |
| Missing `x_normalizer.json` fails naming the run and the file | Met (`RunDir::normalizers` error `<dir>/x_normalizer.json: …`; pinned at `runs/mod.rs:365-373`) |
| Phase 3+ runs all have it | Reported in the PR body and decision 29 (6 Darcy, 7 Poisson; two pre-Phase-3 Darcy runs have neither file). Not re-checked here |
| `poisson_resolution` unchanged, decision 18 comment present | Met (`poisson_resolution.rs:79`, already there since `2e15bb5`) |
| `test_loader_with_saved_normalizers_matches_refit` bit for bit | Met for Poisson (`poisson.rs:542`). **Darcy has no equivalent** (2.1) |
| `RunDir` round trip of both normalisers | Met (`metrics_and_normalizers_round_trip`, extended) |
| `predict_poisson` same five errors, `predict_darcy` same L² | Reported in the PR body (stdout and `errors.json` identical). Not re-run in this review |
| Wall-clock before / after | Reported; not faster (6.2) |
| Do not: change the refit, interpolate normalisers, change the JSON format | Respected |

## Checks run (local, macOS, Flex backend, on HEAD `023b0bb`)

All passed:
- `cargo fmt -- --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo clippy --no-deps --examples -- -D warnings`
- `cargo test`: lib 210 tests (209 passed, 1 ignored), plus the integration and doc-test binaries, all ok
- `cargo doc --no-deps`

Not run:
- `--features datagen`: the brief says it is not needed, and no gated code changed.
- `--features metal`: the change is host-side loading only.
- The predict examples: they need `datasets/` and `runs/`. The PR body reports them.

## Next steps

Each step preserves behaviour and can be one commit on this branch. Step 1 clears the blocker.

1. **Factor the Darcy checks and pin `load_darcy_test`** (2.1).
   - Move the `r == 0` guard into `read_split`.
   - Add `check_grid`, and have `darcy_from_fields` and a new `darcy_test_from_fields` both use it.
   - Reduce `load_darcy_test` to `read_split` followed by `darcy_test_from_fields`.
   - Add a Darcy bit-for-bit test on the existing in-memory fixture.
   - Verify: `cargo test` gives lib 211, and `darcy_output_is_pinned`, `zero_subsample_rate_is_invalid_not_a_panic`, `single_train_sample_is_invalid_not_nan` and `missing_file_is_a_reader_error` pass unchanged. Run `cargo clippy --all-targets -- -D warnings`. `predict_darcy` on `runs/darcy_fno_1790729160` should still print 0.0095942197, if the dataset is present.
2. **One-liners** (4.2, 6.1).
   - `transfers.n()` at `poisson.rs:199`.
   - Optionally, split the rejection half of the Poisson test into its own test.
   - Verify: `cargo test` (lib 211 or 212), `poisson_output_is_pinned` unchanged.
3. **Optional: shorter example boilerplate** (3, 4.1). Replace the inner `unwrap_or_else(|| panic!(..))` with an `ok_or_else` that names `y_normalizer.json`, in all four examples.
   - Verify: `cargo clippy --no-deps --examples -- -D warnings`. Run one predict example on a run with its normaliser files moved aside, and read the message.
4. **Docs.** Update the T18 line delta and test count in the README Results row after steps 1–2.

## Re-review after 1d8c86b

- **Range:** `git diff phase4s/T17-y-train-physical...phase4s/T18-x-normalizer`, now two commits (`023b0bb`, `1d8c86b`). 10 files, +383 / −116.
- **Fix commit:** `1d8c86b` changes `data/loaders/darcy.rs` (+74 / −27, of it +38 test) and the T18 line in `docs/phase4s/README.md`.

### Verdict: APPROVE

`1d8c86b` resolves the one blocker (2.1). It does what step 1 of the first review asked, with no extra moving parts:
- **One guard.** The `subsample_rate == 0` guard now exists once, at the top of `read_split` (`darcy.rs:113-115`), before any I/O. Both entry points call `read_split` before anything that calls `config.s()` (which divides by the rate), so neither loader can reach that division with a zero rate. `zero_subsample_rate_is_invalid_not_a_panic` still passes.
- **One grid check.** `check_grid` (`darcy.rs:169-182`) owns the resolution check. `darcy_from_fields` (`:147`) and the new `darcy_test_from_fields` (`:160`) both call it. The error names the shapes that failed, so a bad `y_test` is no longer reported as `x_test`'s shape (pinned at `:342-348`). It is equivalent to the old two-part check for 3-D fields, and for any other rank it returns an error where the old check could index out of bounds.
- **Testable tail.** `load_darcy_test` is now `read_split` followed by `darcy_test_from_fields` (`:93-99`), the same split as the train path. The Darcy half of the bit-for-bit claim is pinned without a `.mat` fixture (`darcy.rs:316`).
- **Net change:** about +9 prod lines, because `check_grid` gained a doc and the new tail function gained its own. No copy of either check is left. The README T18 row records the delta and lib 210 → 211.

Read fresh, the whole diff meets the Approval Bar:
- Each check and the encode step has one owner: `encode_split`, `check_grid`, `check_transfer_n`.
- Both PDEs follow the same entry-point shape (reader, then a `*_from_fields` tail or a validated read).
- No file crosses 1000 lines (`darcy.rs` 327 → 416, `poisson.rs` 718 → 805).
- No ad-hoc branching was added, and no wrapper or cast churn.

### Presumptive blockers

None. The earlier one (duplicated validation, canonical home skipped) is cleared, as described above.

### File-size table (changed since the first review)

| File | Before → after |
| --- | --- |
| `src/neural_operators/data/loaders/darcy.rs` | 327 → 416 (was 369 at `023b0bb`; +47, of it 38 test) |
| `docs/phase4s/README.md` | 234 → 234 |

The other files are unchanged since `023b0bb`; see the first table.

### Non-blocking suggestions still open

From the first review, not addressed (still optional):
- **4.2** `poisson.rs:199`: `transfers.s() + 1` re-derives n; use `Transfers::n()` (`chebyshev/transfer.rs:215`).
- **3 / 4.1** `examples/predict/darcy.rs:43`, `darcy_plot.rs:34`, `poisson.rs:141`, `poisson_plot.rs:100`: the four identical nested `unwrap_or_else(panic)` blocks. An inline `ok_or_else` naming `y_normalizer.json` would cut each to 3 lines and name the file.
- **4.3** `common.rs:81-87`: only `normalizers.x` is shape-checked. That is deliberate; one doc clause saying `y` is used downstream would make it explicit.
- **6.1** `poisson.rs:542` (rejection half at `:563`): one test checks two things. Split out `test_loader_rejects_normalizers_fitted_at_another_s`.

New in `1d8c86b` (all minor):
- **R1** `darcy.rs:316` (second half at `:342`): the new Darcy test has the same two-in-one shape as 6.1. If 6.1 is split, split this one too (`check_grid_names_the_offending_shape`), for symmetry.
- **R2** `darcy.rs:348`: the assertion matches the `Debug` rendering `got [[4, 4]]`, which ties the test to the `{bad:?}` format. This is acceptable for a message test. `contains("[4, 4]")` would be a little looser.
- **R3** `darcy.rs:38-39`: the `DarcyConfig::s` doc still says the value is checked "in `load_darcy_uniform`". It is now checked in `check_grid`, for both loaders. This is a one-word doc fix on a line outside the diff.

### Checks run (local, macOS, Flex backend, on HEAD `1d8c86b`)

All passed:
- `cargo fmt -- --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo clippy --no-deps --examples -- -D warnings`
- `cargo test`: lib 211 tests (210 passed, 1 ignored), plus the integration and doc-test binaries, all ok
- `cargo test neural_operators::data::loaders::darcy::tests::test_loader_with_saved_normalizers_matches_refit -- --exact`: 1 test ran, and it passed
- `cargo doc --no-deps`

Not run:
- `--features datagen`: no gated code changed.
- `--features metal`: host-side loading only.
- The predict examples: they need `datasets/` and `runs/`. The README reports `predict_darcy` unchanged at 0.0095942197, but that figure predates `1d8c86b`. The fix only refactors validation, and the pinned Darcy tests pass, but `predict_darcy` was not re-run on this commit.
