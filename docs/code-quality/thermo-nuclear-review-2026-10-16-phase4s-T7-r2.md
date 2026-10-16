# Thermo-nuclear code quality review, round 2: Phase 4s T7 (`metal_diag` split)

- **Repo:** sciml-rs
- **Branch:** `phase4s/T7`
- **Base:** `79f9c62` (merge-base with `main`)
- **Date:** 2026-10-16
- **Diff:** working tree, uncommitted. Excluding the two review reports: 7 files, +338 / −219.
  - That includes the rename of `tests/metal_diag.rs` to
    `spikes/metal_diag/tests/metal_diag.rs`.
  - `git diff` counts the root `Cargo.lock` as +8.
- **Follows:** `thermo-nuclear-review-2026-10-16-phase4s-T7.md` (round 1, BLOCKED; its
  Resolution section lists the fixes).

## Verdict: APPROVE

Both round-1 blockers are resolved, and nothing new rises to a blocker.
- The workspace (`Cargo.toml:102-113`) gives one lockfile and one Burn pin.
  `default-members = ["."]` keeps every root build command crate-only.
- `gpu_parity.rs` has no duplicated bodies: each repro is one `fn(&Device)` with two
  thin, named entry points.
- The env-var bisection variant is back in the spike.
- `replay_2d_poisson` uses a closed `Call` type.

What remains is cosmetic or one-line. The largest item (finding 6.1) is an optional
restructuring that would remove seven repeated `#[cfg]` attributes.

## Presumptive blockers

None.
- No file crosses 1,000 lines.
- No branching was added to shared code; `src/` is untouched.
- No new duplicate helpers: `v`, `rel` and `gpu()` existed only in the old
  `metal_diag.rs`, and `src/` has no test-side equivalent to reuse.
- No layer leaks.

## File sizes

| File | Before | After | Note |
| --- | ---: | ---: | --- |
| `tests/metal_diag.rs` → `spikes/metal_diag/tests/metal_diag.rs` | 1,674 | about 1,465 | already over 1k; shrank; now outside the crate's targets |
| `tests/gpu_parity.rs` | 0 | 248 | matches the brief's about 250 |
| `Cargo.toml` | — | +10 net | `[workspace]` and `[workspace.dependencies]` |
| `Cargo.lock` | — | +8 | the `metal-diag-spike` entry only |
| `spikes/metal_diag/Cargo.toml` | 0 | 19 | |
| `spikes/metal_diag/README.md` | 0 | 30 | |
| `CLAUDE.md` | 162 | 172 | |

## Findings

### 1. Structural regressions

None.

### 2. Missed dramatic simplifications

None left at this scale. The round-1 code-judo moves (shared workspace, device-parameterised
bodies) are applied.

### 3. Spaghetti / branching

None. The two `match kind` blocks in `replay_2d_poisson` (`tests/gpu_parity.rs:229-236`)
are exhaustive over `Call`. The pair is a reference call against a call under test, so
merging them would hide the comparison.

### 4. Boundary / type contracts

**4.1 Workspace side effect: `cargo fmt` now covers the spike** (`Cargo.toml:105-107`)

`cargo fmt` acts on every workspace member, unlike `test`, `clippy` and `doc`, which follow
`default-members`.
- So CI's `cargo fmt -- --check` now checks `spikes/metal_diag`.
- It passes today, and the effect is benign, arguably desirable. But it contradicts the
  absolute "never built by CI" wording only loosely: formatting is not building.

Remedy: add "(`cargo fmt` does check it)" to the Cargo.toml comment, so that nobody is
surprised by a fmt failure in a spike. One line.

**4.2 The spike's default feature is `metal`** (`spikes/metal_diag/Cargo.toml:11`)

On a CUDA box, `cargo test -p metal-diag-spike` fails trying to build Metal. The README
documents `--no-default-features --features cuda`, and the package is literally named for
Metal. Accept as is.

### 5. File size / decomposition

None.

### 6. Modularity

**6.1 Seven copies of the same GPU `#[cfg]`** (`tests/gpu_parity.rs:32,37,87,95,115,149,156,172`)

Every GPU item repeats `#[cfg(any(feature = "metal", feature = "cuda"))]`. A new GPU test
that forgets the attribute breaks the default build, which is caught by CI, but noisily.

The move is to gate one module:

```rust
#[cfg(any(feature = "metal", feature = "cuda"))]
mod gpu {
    use super::*;
    fn device() -> Device { /* cuda / metal split */ }
    fn v(…), fn rel(…), enum Call, …
    #[test] #[ignore = "…"] fn repro_fused_cat_broadcast() { fused_cat_broadcast(&device()) }
    …
}
```

Cost: test paths become `gpu::repro_…`.
- The upstream READMEs cite only function names inside code blocks, with no `cargo test`
  or `--exact` commands (grep-checked).
- Substring filters (`cargo test --test gpu_parity repro_cfft`) still match.
- The brief's rule "keep the test names" is about the names, which stay.

This leaves one gate instead of seven, and makes the GPU/Flex split visible as structure,
not as attributes. Optional: do it now if you want the file to read as two sections. It is
not required for approval.

### 7. Legibility

- **7.1** `tests/gpu_parity.rs:156-164`: the `Display` impl spends 9 lines on lowercase
  labels. `#[derive(Debug)]` with `{kind:?}` would print `Train20` instead of `train20`,
  which nothing parses. Either form is fine; it is a nit.
- **7.2** `CLAUDE.md:126`: the Burn line is now 113 characters, while the surrounding prose
  wraps at about 90. Re-wrap it.
- **7.3** `tests/gpu_parity.rs:114`: "Bug-report repro 2, as published." reads oddly now that
  `fused_cat_broadcast` is introduced as "Burn bug 1". Write "Burn bug 2
  (`upstream/burn/fused-cat-reused-at-new-shape/`), as published" for symmetry.

## Next steps

All are optional polish and independent; they could be one commit.

1. **4.1, 7.2, 7.3: comment and wording fixes.**
   - Verify: `cargo fmt -- --check` and `cargo doc --no-deps`.
2. **6.1: `mod gpu` wrapper, if wanted.**
   - Verify:
     - `cargo test --test gpu_parity` (2 pass);
     - `cargo clippy --features metal --all-targets -- -D warnings`;
     - `cargo test --release --features metal --test gpu_parity` (3 pass, 3 ignored;
       replay worst 2.33e-6).
   - Note the path change in the PR.
3. **7.1: only if doing step 2 anyway.**
