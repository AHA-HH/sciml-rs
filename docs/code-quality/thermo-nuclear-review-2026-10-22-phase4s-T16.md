# Thermo-nuclear code quality review: Phase 4s T16 (ungate `chebyshev/`, feature renamed `datagen`)

- **Repo:** sciml-rs (`AHA-HH/sciml-rs`)
- **Branch:** `phase4s/T16-ungate-chebyshev` (HEAD `6c1e3c4`, PR #53, stacked on #52)
- **Base:** `phase4s/T14-generator-ungated-stack` (T14, PR #52). The range is the one commit `6c1e3c4`.
- **Date:** 2026-10-22
- **Diff size:** 22 files, +95 / −67. Of that, `src` and `examples` are 13 files, +40 / −36. The rest is docs, `Cargo.toml` and CI.

## Verdict: APPROVE (with nits)

T16 is a rename with no change to any numeric code, and it is done thoroughly:
- Every `cfg`, `required-features`, the CI job and every live doc moved to `datagen`.
- `git grep 'feature = "chebyshev"'` finds nothing outside historical briefs and old reviews.
- The design decision (27) records both the T15 refusal and "no alias".
- The README Results row is filled in, and its test counts match what I measured: 209 tests default, 264 with `datagen`.

T14 had already ungated the production toolkit, so this PR adds no branching, wrapper or new abstraction. It also crosses no size threshold, and no blocker condition applies.

One small structural move was missed: the RLST helper `unit_coeffs` and its two `rlst` imports still sit gated in `chebyshev/mod.rs`, when they could live in `differentiation.rs`. The rest is stale wording. None of it blocks.

## Presumptive blockers

None.
- No file crosses 1000 lines. `docs/design/2d-chebyshev-poisson-fno.md` was already 1309 lines; it gains a 14-line decision entry, which is the house format.
- No new conditionals. Every changed `cfg` is a 1:1 rename.
- No new wrappers, casts or optionality.
- No duplicated helper.

## File-size table

| File | Before → after | Note |
| --- | --- | --- |
| `docs/design/2d-chebyshev-poisson-fno.md` | 1309 → 1323 | already over 1k (a doc; decision 27 appended) |
| `src/neural_operators/chebyshev/transfer.rs` | 602 → 602 | cfg rename only |
| `src/neural_operators/metrics/poisson.rs` | 384 → 384 | doc only |
| `src/neural_operators/chebyshev/clencurt.rs` | 344 → 344 | doc + 2 test cfgs |
| `examples/predict/poisson.rs` | 279 → 279 | doc only |
| `CLAUDE.md` | 185 → 191 | |
| `Cargo.toml` | 130 → 133 | feature comment added |
| `src/lib.rs` | 66 → 69 | one crate-doc paragraph |
| `src/neural_operators/chebyshev/mod.rs` | 62 → 63 | |
| All other touched files | unchanged length, < 300 | |

## Findings

### 1. Structural regressions

None.

### 2. Missed code-judo simplifications

**2.1 `unit_coeffs` and the `rlst` imports should move into `differentiation.rs`, so `mod.rs` is free of RLST** (`src/neural_operators/chebyshev/mod.rs:43-55`).
- After this PR, `mod.rs` still carries three `#[cfg(feature = "datagen")]` items that are neither module declarations nor re-exports:
  - `use rlst::DynArray` (:43-44);
  - `use rlst::chebychev::ChebychevCoefficientsToData` (:45-46);
  - `fn unit_coeffs` (:50-55).
- Their only callers are `differentiation.rs:6` (gated `datagen`) and the test oracle `quadrature.rs:8` (gated `all(test, datagen)`). `quadrature`'s gate implies `differentiation`'s, so the helper can move:

```rust
// differentiation.rs
use rlst::DynArray;
use rlst::chebychev::{ChebychevCoefficientsToData, DerivativeOfChebychevSeries1d};
pub(super) fn unit_coeffs(n: usize, j: usize) -> DynArray<f64, 1> { … }

// quadrature.rs
use super::{check_n, differentiation::unit_coeffs};
```

- This deletes 3 of the 9 `datagen` gates in `mod.rs` (4 `cfg` lines counting the doc'd function).
- It also makes the brief's deliverable literally true: "`differentiation.rs` alone stays gated". Today decision 27 has to add "(and the `unit_coeffs` it calls)" to cover the helper left in `mod.rs`.
- `mod.rs` then holds only the toolkit's doc, its module list, its re-exports and `check_n`. Every gate is on a `mod` or `pub use` line.
- Behaviour is unchanged.

**2.2 (Optional, for the author's judgement, not a request.)** D and D² have one consumer, `pde/poisson/collocation.rs:20`. They could move to `pde/poisson/`, which would leave `chebyshev/` with only test oracles behind the feature.
- I do **not** recommend it now. Decision 24 keeps the toolkit (and Phase 6's θ-map) in `chebyshev/`, and D/D² are toolkit operators, not solver logic.
- The brief also explicitly keeps `differentiation.rs` gated in place.
- Mentioned only so it is a recorded, declined option.

### 3. Spaghetti / branching growth

None. The `cfg` count in `chebyshev/` is unchanged (renamed, not added).

### 4. Boundary, abstraction and type-contract problems

**4.1 The `mod.rs` doc says D and D² are "the only gated production items here"** (`chebyshev/mod.rs:15-16`). The private `unit_coeffs` is gated production code too.
- Applying 2.1 makes the sentence true.
- Otherwise, say "the only gated public items".

### 5. File size and decomposition

None.

### 6. Modularity

**6.1 Rename history in the code** (`Cargo.toml:77-79` "renamed from `chebyshev`, no alias"; `chebyshev/mod.rs:25` "T15 declined, decision 27").
- Both point at decision 27, which is the right place for that history.
- In code, keep the *reason* and drop the task archaeology:
  - `mod.rs:25`: "Gated: wraps RLST's FFTW transforms (design §12, decision 12)."
  - `Cargo.toml`: drop the "renamed from" clause.
- This is minor, and the brief did ask for a one-line reason.

### 7. Legibility

**7.1 An unwrapped doc line, 143 characters** (`chebyshev/mod.rs:17`). The edit inserted "(design §12, decisions 12 and 27) and need a system libfftw3." and did not re-wrap, so "RLST works in descending node order; every function here returns ascending" runs on. `rustfmt` does not wrap comments. Re-wrap to the file's ~92-column style.

**7.2 Stale live mentions of the old name, outside the briefs and old reviews:**
- `spikes/solver/Cargo.toml:12` and `spikes/solver/REPORT.md:58` say "the root crate's `chebyshev` feature". The spike is never built, but the comment is now wrong. Either "`datagen` (named `chebyshev` at the time)" or leave it as history, deliberately.
- `docs/design/2d-chebyshev-poisson-fno.md` §9 (:463-480) still specifies "Optional feature `chebyshev`" and the old CI commands. The design doc's convention is to amend by decision, and decision 27 does that. A one-line pointer at the top of §9 ("renamed `datagen`, decision 27"), as earlier amendments did, would stop a reader of §9 alone from using the wrong name. This is a nit.

**7.3 CI rename is safe.** I checked `main`'s branch protection (`gh api …/branches/main/protection`): `required_status_checks.contexts` is empty and there are no rulesets. Renaming the job `run-tests-chebyshev` → `run-tests-datagen` therefore cannot leave a required check pending forever. No action needed.

## Brief acceptance (T16)

| Item | Status |
| --- | --- |
| `chebyshev/` production items build without features, except `differentiation.rs` | Met (plus `unit_coeffs`, see 2.1) |
| RLST oracles `#[cfg(all(test, feature = "datagen"))]` | Met (`mod.rs:28-33`, `clencurt.rs` 2 tests, `transfer.rs` `rlst_oracle`) |
| Feature renamed; it gates `pde/`, the `lib.rs` extern crates and the OS check | Met (`neural_operators/mod.rs:9`, `lib.rs:37-45`) |
| Every `cfg`, `required-features`, CI job and doc updated | Met for live code and docs; 7.2 lists the two stale leftovers |
| Alias decision recorded | Met: no alias, decision 27 |
| `lib.rs` crate-doc line | Met (`lib.rs:20-21`) |
| Test count: list the newly ungated tests | None newly ungated (T14 already did it). Recorded in the README; counts verified at 209 / 264 |
| `git grep 'feature = "chebyshev"'` empty | Met in `src`, `Cargo.toml`, CI and examples (hits only in historical briefs) |
| Renamed CI job green | **Not verified**: CI does not run on a PR whose base is not `main`. Local equivalents pass (below) |

## Checks run (local, macOS, Flex backend)

All passed:
- `cargo fmt -- --check`
- `cargo clippy --no-deps -- -D warnings`
- `cargo clippy --no-deps --examples -- -D warnings`
- `cargo test`: lib 209 tests (208 passed, 1 ignored), plus integration and doc tests, all ok
- `cargo doc --no-deps`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test --features datagen`: lib 264 tests (263 passed, 1 ignored), all ok
- `cargo clippy --all-targets --features datagen -- -D warnings`
- `cargo clippy --no-deps --features datagen --all-targets -- -D warnings` (the CI job's exact command)
- `cargo doc --no-deps --features datagen`

Not run:
- `--features metal`: no backend-sensitive code changed.
- GitHub CI: the base is not `main`.

## Next steps

Each step preserves behaviour. All of them together are one small commit on this branch.

1. **Move `unit_coeffs` and its `rlst` imports into `differentiation.rs`** as `pub(super)`, and import it in `quadrature.rs` from `super::differentiation` (2.1, fixes 4.1).
   - Verify: `cargo test --features datagen` still runs 264 lib tests, and `cargo test` still runs 209.
   - Also verify that `cargo clippy --all-targets [--features datagen] -- -D warnings` is clean in both configurations, because the non-test `datagen` build must not warn about `unit_coeffs` being unused outside tests. `differentiation` uses it, so it will not.
2. **Re-wrap `chebyshev/mod.rs:15-18`** and trim the task history in `mod.rs:25` and `Cargo.toml:79` (7.1, 6.1). Verify with `cargo fmt -- --check` and `cargo doc --no-deps --features datagen`.
3. **Optional:** a pointer to decision 27 at the head of design §9, and fix or annotate the `spikes/solver` comment (7.2). These are docs only.
