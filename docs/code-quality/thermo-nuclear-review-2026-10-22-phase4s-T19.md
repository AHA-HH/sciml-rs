# Thermo-nuclear code quality review: Phase 4s T19 (typed sidecar, Poisson evaluation in the library)

- **Repo:** sciml-rs (`AHA-HH/sciml-rs`)
- **Branch:** `phase4s/T19-sidecar-and-eval` (HEAD `ee24655`, PR #57)
- **Base:** `phase4s/T18-x-normalizer` (stacked; PR #56, itself on #55). The range is `git diff phase4s/T18-x-normalizer...phase4s/T19-sidecar-and-eval`, one commit.
- **Date:** 2026-10-22
- **Diff size:** 13 files, +613 / −233. Of that, `src` and `examples` are 11 files; the rest is the phase README Results rows and the 30-line sidecar fixture (`tests/fixtures/poisson_n65_K32_test.json`).

## Verdict: BLOCKED

Most of the task is done well:
- The sidecar is one type. `write_split` builds a `SidecarRecord` (`dataset.rs:255-284`) and `predict_poisson` loads the same type (`examples/predict/poisson.rs:82`). The hand-synced pair (untyped `json!` writer, three-field reader) is gone. The bytes are pinned twice: by the ungated fixture test (`loaders/poisson.rs:453`) and, byte for byte, by the old `json!` kept as an oracle in `files_round_trip` (`dataset.rs:387-417`).
- The evaluation policy left the example. `examples/predict/poisson.rs` went 280 → 103 lines. The worst sample, both cross-checks and the boundary verdict are now a library function with four unit tests on synthetic arrays, including the NaN worst sample and both sides of the threshold.
- The crate-wide `#[allow]` at `lib.rs:47` is gone. The replacement is one inner allow per file that has a `Config` derive (7 files), each with the burn-derive reason. That is the scope the repo review asked for (S5, "scope the allow to the files with `Config` derives") and the README row states ("scoped per file"). The PR's reason for not scoping per type checks out: burn-derive emits `#name: #name` (`crates/burn-derive/src/config/analyzer_struct.rs:196`) in a separate `impl` block, which does not inherit the struct's attributes.

One presumptive blocker stands: `PoissonErrorsRecord` is now two types in one.
- It is the persisted schema of `errors.json`, and it also carries three `#[serde(skip)]` fields that exist only for its `Display`: `run_dir`, `sidecar_count` and `error3_check: Option<Error3Check>`.
- `Option` there encodes no domain fact. `evaluate` always sets `Some`, and `None` only happens when serde builds the record from disk.
- A record loaded from `errors.json` therefore prints a silently wrong report: an empty `poisson_plot --` path, "the sidecar covers 0 samples" whenever error 2 was not compared, and no error 3 check line.
- The split is also why `PoissonErrorsRecord` lost `#[derive(Config)]` for a hand-written `Serialize`/`Deserialize` plus an empty `impl Config`, with a doc paragraph to justify it. And `error3_metrics_csv` now duplicates `error3_check.test_l2` in the same struct.
- There is a plain fix. Put the report on an evaluation type that holds the record and the two checks. The skip fields, the `Option`, the duplicated `test_l2` and the hand-written serde all go away, and stdout and `errors.json` stay byte-identical.

## Presumptive blockers

1. **An unnecessary optional, non-persisted contract on a persisted record** (`metrics/poisson.rs:185-241`, `:318-320`, `:383`). See finding 4.1. The brief asked for `Display` on `PoissonErrorsRecord`. The author resolved the conflict between that and "`errors.json` must not change" by making the schema type lie about its own fields. The brief's intent (the policy in the library, the printout from a library type, exit 1 driven by the result) is better met by the structure in 4.1.

Not met:
- No file crosses 1000 lines. `metrics/poisson.rs` goes 384 → 735, of which about 330 are prod and docs; `loaders/poisson.rs` goes 805 → 924. `models/fno.rs` was already over (1595 → 1599, the 4-line allow only).
- No ad-hoc branching was added to an existing flow. The error 2 "compared or skipped" branch moved; it was not added.
- No logic is in the wrong layer: `evaluate` lives next to `poisson_errors` and `check_error3`, which it builds on.

## File-size table

| File | Before → after | Note |
| --- | --- | --- |
| `src/neural_operators/models/fno.rs` | 1595 → 1599 | already over 1k; +4 is the allow and its reason |
| `src/neural_operators/data/loaders/poisson.rs` | 805 → 924 | +119: `SidecarRecord` and its two nested types (+86), the fixture test (+33). T22 splits this file |
| `src/neural_operators/metrics/poisson.rs` | 384 → 735 | +351: `evaluate`, `PoissonErrorsRecord`, its `Display`, the constants (about +240), four tests (+111) |
| `src/neural_operators/pde/poisson/dataset.rs` | 451 → 485 | the writer is the same length; +32 is the `json!` oracle in the test |
| `src/neural_operators/data/loaders/darcy.rs` | 416 → 420 | allow |
| `src/neural_operators/data/transforms/normalizers.rs` | 327 → 331 | allow |
| `examples/predict/poisson.rs` | 280 → 103 | policy and records moved out |
| `docs/phase4s/README.md` | 234 → 234 | Results rows |
| `src/neural_operators/data/loaders/burgers.rs` | 189 → 193 | allow |
| `src/neural_operators/metrics/stats.rs` | 71 → 75 | allow |
| `src/lib.rs` | 69 → 68 | crate-wide allow deleted |
| `tests/fixtures/poisson_n65_K32_test.json` | 0 → 30 | new fixture |
| `src/neural_operators/data/loaders/base_dataset.rs` | 15 → 19 | allow |

## Findings

### 1. Structural regressions

None in the dataflow. The example is now load, predict, `evaluate`, print, save, exit. That is the shape the review's §2 "Evaluation policy" edge asked for.

### 2. Missed code-judo simplifications

**2.1 The error 2 cross-check should be a value like `Error3Check`, not three loose pieces.** Today error 2's check is spread out:
- a stored `error2_compared` and a sidecar `ErrorStats` (`metrics/poisson.rs:208-211`);
- a `sidecar_count` stored only for printing (`:235`);
- `error2_matches` and `rel_diff`, called once in `evaluate` (`:294`) and again in `Display` (`:362-372`) to rebuild the pass/fail it never stored.

Error 3's check is already one value, `Error3Check`, with its own `Display` (`:135-158`). Make error 2 the same, and the two cross-checks read alike. This is the core of the fix in 4.1, so it is described there.

**2.2 The per-file allows are a workaround for one token in a fork the project pins.** Burn is the fork `ax1s-x1zz/burn` at `ddfa9af` (`Cargo.toml:109`). The lint comes from `#name: #name,` at `burn-derive/src/config/analyzer_struct.rs:196`. Writing `#name,` there removes the cause, and all seven `#![allow(clippy::redundant_field_names)]` (28 lines) go with it. This is out of T19's scope, because it needs a fork commit and a pin bump. It is a clean follow-up, not a blocker: until then, per file is the narrowest scope that works, and S5 asked for it.

### 3. Spaghetti / branching growth

None added. `Display` holds two `if warranted` ternaries and the compared-or-skipped branch, all moved verbatim from the example; stdout is byte-identical (PR acceptance).

### 4. Boundary, abstraction and type-contract problems

**4.1 (Blocking.) `PoissonErrorsRecord` mixes the `errors.json` schema with the report** (`metrics/poisson.rs:185-241`, `:318-320`, `:333-403`).

This works, but it makes the persisted type dishonest:
- Three fields are `#[serde(skip)]` (`:231-238`), so the struct's fields no longer describe the file it is the schema for.
- `error3_check: Option<Error3Check>` is `Some` on the only constructor (`:320`) and `None` only after a load. The `if let Some` in `Display` (`:383`) is a silent fallback for an invariant the type could state.
- `Display` on a loaded record is wrong without saying so: an empty path in the `poisson_plot` hint (`:358`), "covers 0 samples" (`:380`), and no error 3 line. The doc (`:189-190`) admits this rather than preventing it.
- `error3_metrics_csv` (`:215`) and `error3_check.test_l2` hold the same number.
- The doc paragraph at `:188-190`, the hand-written serde derive, the `burn::config::Config` and `burn::serde` imports (`:29-30`) and `impl Config for PoissonErrorsRecord {}` (`:241`) exist only because `Display` lives on the schema type, which collides with `#[derive(Config)]`'s own `Display`.

**Move.** Keep the record as pure schema, and put the report on the evaluation:

```rust
/// The error 2 cross-check against the test sidecar, from [`evaluate`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Error2Check {
    pub sidecar: ErrorStats,
    pub sidecar_count: usize,
    /// None when the sidecar covers other samples than those evaluated (skipped, counts as passed).
    pub rel_diff: Option<(f64, f64)>,   // (mean, max)
    pub passed: bool,
}
impl fmt::Display for Error2Check { /* the two lines now in PoissonErrorsRecord::fmt, :362-381 */ }

/// What `predict_poisson` reports: the record `errors.json` holds, and the two checks behind it.
pub struct PoissonEvaluation {
    pub record: PoissonErrorsRecord,
    pub error2: Error2Check,
    pub error3: Error3Check,
    pub run_dir: PathBuf,
}
impl fmt::Display for PoissonEvaluation { /* today's PoissonErrorsRecord::fmt, reading self.error2 / self.error3 */ }

pub fn evaluate(...) -> PoissonEvaluation
```

What this changes:
- `PoissonErrorsRecord` keeps its 17 persisted fields and nothing else.
- With no `Display` of its own, it can go back to the `#[derive(Config)]` it had in the example. Add the file's allow line as the other seven files do, or keep the plain serde derive. Either way the doc paragraph and the three skip fields go.
- `error2_matches` and the second `rel_diff` computation in `Display` go: `Error2Check` stores the result once.
- `cross_checks_passed` stays in the record, so `errors.json` is unchanged. It is `error2.passed && error3.passed`.

The example becomes:
```rust
let eval = evaluate(...);
println!("\n{eval}");
eval.record.save(&out)?;
if !eval.record.cross_checks_passed { exit(1) }
```
Stdout and `errors.json` stay byte-identical, so the acceptance diff is the proof. `evaluate_flags_a_cross_check_mismatch` then asserts `eval.error3.passed` directly instead of `.map(|c| c.passed) == Some(false)` (`:707`).

**4.2 `evaluate` reads the build's backend features itself** (`metrics/poisson.rs:304`). It takes `device` as an argument but calls `backend_features()` internally, so a "pure policy" function in `metrics` depends on `utils::backend` and on cargo features. It is harmless today, because the value is only recorded. If the record moves to an evaluation type (4.1), pass both environment facts the same way, or let the example fill them in. Non-blocking.

**4.3 `SidecarRecord`'s byte compatibility rests on alphabetical field declaration** (`loaders/poisson.rs:116-122`). The doc says so, and two tests enforce it. A future field added in "logical" order would change the bytes. The datagen oracle catches that, but the ungated fixture test catches it only as a line-count or key mismatch. The doc comment is the right mitigation. Consider "add new fields in sorted position" as one more line there. Non-blocking.

### 5. File size and decomposition

No threshold crossed. `loaders/poisson.rs` (924) keeps growing toward T22's split. The sidecar types belong to T22's dataset catalogue module, and the PR places them where the brief said ("or T22's catalogue module, if T22 has merged"). `metrics/poisson.rs` at 735 is cohesive: errors, checks, evaluation.

### 6. Modularity

**6.1 `rel_diff` is a helper that `check_error3` does not use** (`metrics/poisson.rs:324-326` against `:164`). The same `|a − b| / |b|` is written inline in `check_error3`. Use `rel_diff` there too. One line.

**6.2 The fixture is `include_str!`ed from two test modules with two relative paths** (`loaders/poisson.rs:450`, `metrics/poisson.rs` test helper `sidecar`, `:633`). It is fine as is. If T22 adds a catalogue module, a `#[cfg(test)] pub(crate) fn fixture_sidecar()` there would give the fixture one home.

### 7. Legibility

**7.1 `files_round_trip` keeps its old `Value` assertions after the new byte check** (`dataset.rs:417` onward). Once the text equals the old writer's output byte for byte, the `v["split"]`, `v["n"]` and similar lookups that follow re-check a subset of the same thing. Keep the ones that state facts the oracle does not (e.g. that `K` is the K in effect), and drop the rest. Low value either way.

**7.2 The README row says `poisson_resolution` calls `evaluate`** (`docs/phase4s/README.md:143`); it does not, by a documented deviation. The PR body explains why. Amend the row's wording when the Results are filled in, so the plan and the code agree.

## Brief acceptance (T19)

| Item | Status |
| --- | --- |
| Test first: fixture parsed by the new type | Met (`sidecar_fixture_parses_and_round_trips`, `loaders/poisson.rs:453`) |
| `SidecarRecord` typed, ungated, written by the generator, loaded by the example | Met. It derives serde with a hand `impl Config`, not `#[derive(Config)]`, for the renamed keys; justified (`:116-121`) |
| JSON byte-compatible | Met: byte check against the old `json!` (`dataset.rs:417`, datagen), keys and layout plus numbers to 1e-15 ungated |
| `evaluate` with the worst sample, both cross-checks and the boundary verdict; constants moved and documented | Met (`metrics/poisson.rs:173-183`, `:260`) |
| `PoissonErrorsRecord` moves and implements `Display` | Met to the letter, at the cost of blocker 4.1 |
| Exit 1 driven by the record | Met (`record.cross_checks_passed`) |
| `poisson_resolution` reuses it where it applies | Deviation, documented: it keeps its table and already calls `check_error3` at the trained n. Accepted |
| Scoped allows, `lib.rs` allow deleted, example's allow deleted | Met per file, not per type; the per-type route does not work (verified in burn-derive) |
| `predict_poisson` same text and `errors.json` | Reported in the PR body (empty diffs). Not re-run in this review (needs `datasets/` and the run) |
| Do not: change the sidecar or `errors.json` format; split `loaders/poisson.rs` | Respected |

## Checks run (local, macOS, Flex backend, on HEAD `ee24655`)

| Check | Result |
| --- | --- |
| `cargo fmt -- --check` | ok |
| `cargo clippy --no-deps -- -D warnings` | ok |
| `cargo clippy --no-deps --examples -- -D warnings` | ok |
| `cargo clippy --all-targets -- -D warnings` | ok |
| `cargo clippy --no-deps --features datagen --all-targets -- -D warnings` | ok |
| `cargo test` | ok: lib 216 run, 215 passed, 1 ignored; integration tests 2 + 1 + 4, doctests 3 + 4 |
| `cargo test --features datagen` | ok: lib 263 run, 262 passed, 1 ignored (131 s); integration tests and doctests as above |
| `cargo doc --no-deps` | ok |

Not run: `--features metal` and `gpu_parity` (the change is host-side records and f64 ndarray policy, not backend-sensitive), and the `predict_poisson` / `poisson_resolution` acceptance runs (need `datasets/poisson/` and the run directory).

## Next steps

1. **Split the report from the record (blocker 4.1, with 2.1).** Add `Error2Check` with its `Display`, and `PoissonEvaluation { record, error2, error3, run_dir }` with today's `Display`. `evaluate` returns it. `PoissonErrorsRecord` loses its three skip fields and goes back to plain schema, and `error2_matches` and the second `rel_diff` computation go. Verify: `cargo test neural_operators::metrics::poisson` (the four `evaluate` tests, adapted), the CI checks, and the `predict_poisson` acceptance diff (stdout and `errors.json` empty against `main`). One commit on this branch.
2. **`check_error3` uses `rel_diff` (6.1).** One line, same commit or the next. Verify: `error3_check_uses_the_last_epoch`.
3. **Optional, own PR on the fork: `#name,` in burn-derive (2.2).** Bump the pin, then delete the seven per-file allows. Verify: `cargo clippy --all-targets -- -D warnings`, also with `--features datagen`, and `git grep -n redundant_field_names` empty in `src/`. It depends on nothing here. It can land after T22, which moves the sidecar types.
4. **T22 picks up 4.3 and 6.2** when the catalogue module takes `SidecarRecord`: one home for the fixture helper, and the sorted-field note.

## Re-review after 8299dd2

- **Range:** `git diff phase4s/T18-x-normalizer...phase4s/T19-sidecar-and-eval`, now two commits (HEAD `8299dd2`). 13 files, +637 / −234. 8299dd2 itself touches `metrics/poisson.rs`, `examples/predict/poisson.rs` and the README line-delta row (+121 / −98).
- **Date:** 2026-10-22

### Verdict: APPROVE

Blocker 4.1 is resolved, in the shape the first review proposed:
- `PoissonErrorsRecord` (`metrics/poisson.rs:225-264`) is pure schema again: its 17 persisted fields, no `#[serde(skip)]` fields, no `Option`, and `#[derive(Config, Debug)]` back in place of the hand-written serde and empty `impl Config`. The file's `redundant_field_names` allow (`:23-25`) matches the other seven files.
- `Error2Check` (`:189-223`) is a value like `Error3Check`, with its own `Display`. The pass/fail is computed once in `evaluate` (`:330-342`); `error2_matches` and the recomputation in `Display` are gone.
- `PoissonEvaluation { record, error2, error3, run_dir }` (`:266-278`) owns the report `Display` (`:379-427`). A record loaded from `errors.json` can no longer print a wrong report, because it has no report.
- `evaluate` (`:297-373`) builds both checks first, then derives the record's `error2_compared`, `error3_metrics_csv` and `cross_checks_passed` from them in one constructor. The duplication left between check and record is a projection written in one place, not two sources of truth.
- The example (`examples/predict/poisson.rs:85-102`) is now load, predict, `evaluate`, print, save `eval.record`, exit on `!eval.record.cross_checks_passed`.
- The four `evaluate` tests keep their coverage and now assert `bad.error2.passed` / `bad.error3.passed` directly (`:707-724`).

No presumptive blocker is met on the full diff:
- No file crosses 1000 lines. `metrics/poisson.rs` is 384 → 758 (735 before the fix); `loaders/poisson.rs` 805 → 924; `models/fno.rs` was already over (1595 → 1599, the allow only).
- No ad-hoc branching was added; no wrapper, cast or optional obscures a contract; no logic is in the wrong layer.

The brief (step 3) asked for `evaluate(...) -> PoissonErrorsRecord` with `Display` on the record. Returning `PoissonEvaluation` is a deviation, but it meets the brief's intent (policy in the library, report from a library type, exit 1 driven by the result) and was the remedy this review asked for. The PR comment on 8299dd2 describes it.

### Remaining non-blocking suggestions

New in this re-review:
- **R1** `metrics/poisson.rs:198-203`: `Error2Check` encodes "compared or skipped" as `rel_diff: Option<(f64, f64)>` plus `sidecar_count` / `n_test`, which only matter when skipped. An enum `Error2Outcome { Compared { mean, max }, Skipped }` (or named fields instead of the tuple) would say which numbers belong to which case. Small, local.
- **R2** `metrics/poisson.rs:197`: `Error2Check::n_test` duplicates `record.n_test`; it is there only for the skipped line of `Display`. Acceptable, since `Error2Check` displays on its own.

Carried over from the first review, still open:
- **6.1** `metrics/poisson.rs:168`: `check_error3` writes `|a − b| / |b|` inline; use `rel_diff` (`:375`). One line.
- **4.2** `metrics/poisson.rs:352`: `evaluate` calls `backend_features()` itself while taking `device` as an argument; pass both environment facts the same way.
- **7.2** `docs/phase4s/README.md:143`: the T19 row still says `poisson_resolution` calls `evaluate`; it does not (documented deviation). Amend the row; it could also say `evaluate` returns `PoissonEvaluation`.
- **2.2** the seven per-file `#![allow(clippy::redundant_field_names)]`: a one-token fix in the pinned burn fork (`burn-derive/src/config/analyzer_struct.rs:196`) deletes them all. Own PR.
- **4.3** `loaders/poisson.rs:116-122`: add "new fields in sorted position" to `SidecarRecord`'s doc. For T22.
- **6.2** the fixture is `include_str!`ed from two test modules (`loaders/poisson.rs:450`, `metrics/poisson.rs:632`). For T22's catalogue module.
- **7.1** `pde/poisson/dataset.rs:417` onward: the `Value` assertions after the byte check mostly re-check the same bytes.

### Checks run (local, macOS, Flex backend, on HEAD `8299dd2`)

| Check | Result |
| --- | --- |
| `cargo fmt -- --check` | ok |
| `cargo clippy --no-deps --features datagen --all-targets -- -D warnings` | ok |
| `cargo clippy --all-targets -- -D warnings` | ok |
| `cargo clippy --no-deps --examples -- -D warnings` | ok |
| `cargo test` | ok: lib 216 run, 215 passed, 1 ignored; integration tests 2 + 1 + 4; doctests 3 + 4 |
| `cargo doc --no-deps` | ok |

Not run: `cargo test --features datagen` (the fix touches no gated code; the PR comment reports it passing), `cargo clippy --no-deps -- -D warnings` on its own (covered by `--all-targets`), `--features metal` / `gpu_parity` (not backend-sensitive), and the `predict_poisson` byte-identity acceptance (needs `datasets/poisson/` and the run; the PR comment reports empty diffs for stdout and `errors.json`).

### Next steps

1. Optional, on this branch: 6.1 and the 7.2 README wording. Verify: `cargo test neural_operators::metrics::poisson`, and read the row.
2. Optional: R1, if the author wants the skipped case typed. Verify: the `evaluate` tests and the `predict_poisson` stdout diff.
3. Later, as planned: 2.2 (own PR on the fork), 4.3 and 6.2 (T22).
