# Thermo-nuclear code quality review: Phase 4s T7 (`metal_diag` split)

- **Repo:** sciml-rs
- **Branch:** `phase4s/T7`
- **Base:** `79f9c62` (merge-base with `main`)
- **Date:** 2026-10-16
- **Diff:** working tree, uncommitted
  - 6 files, +8,557 / −263.
  - The new `spikes/metal_diag/Cargo.lock` accounts for 8,185 of the added lines.
  - Without it: 5 files, +372 / −263.

## Verdict: BLOCKED

The split does what T7 asks:
- the 7 asserting tests are in `tests/gpu_parity.rs`;
- the Flex twins run in CI;
- the 26 print-only diagnostics compile in a spike;
- the docs are updated.

It has no structural regression in the library: `src/` is untouched. Two presumptive
blockers remain, and both are cheap to fix:
- The spike adds a **second copy of the Burn pin and an 8,185-line copy of the root
  lockfile**. Every future Burn bump must then be made in two places, and the two locks
  will drift. The repository already has a home for both: one Cargo workspace.
- `gpu_parity.rs` **keeps two pairs of copy-pasted test bodies**, the GPU repros and
  their Flex twins. It also keeps an env-var-driven variant
  (`repro_fusion_cache_new_shape`), although T7's own source review says env-var tests
  belong in the spike. The file is 311 lines where about 200 would do.

## Presumptive blockers

1. **Duplicated canonical config (lockfile and Burn pin).**
   - `spikes/metal_diag/Cargo.toml:20` repeats the root `Cargo.toml:103` Burn git URL,
     rev and features.
   - `spikes/metal_diag/Cargo.lock` (8,185 lines) is a hand-seeded copy of the root
     `Cargo.lock`.
   - Without that copy, the spike does not resolve: a fresh resolve picks yanked
     `zip 6.0.0`. The copy is load-bearing duplication with no mechanism to keep it in
     sync.
2. **Preserved incidental complexity with a visible code-judo move.**
   - `tests/gpu_parity.rs:44-129`: `repro_fused_cat_broadcast` / `_flex` and
     `repro_cfft_bluestein` / `_flex` are line-for-line identical apart from the device.
   - `tests/gpu_parity.rs:133-216`: `repro_fusion_cache_new_shape` is
     `repro_fused_cat_reused_at_new_shape` plus `SHAPES` / `WARMS` knobs.

## File sizes

| File | Before | After | Note |
| --- | ---: | ---: | --- |
| `tests/metal_diag.rs` → `spikes/metal_diag/tests/metal_diag.rs` | 1,674 | 1,419 | already over 1k; shrank; now outside the crate |
| `tests/gpu_parity.rs` | 0 | 311 | brief targets about 250; about 200 is reachable (finding 2.1) |
| `spikes/metal_diag/Cargo.lock` | 0 | 8,185 | generated; see blocker 1 |
| `spikes/metal_diag/Cargo.toml` | 0 | 20 | |
| `spikes/metal_diag/README.md` | 0 | 27 | |
| `CLAUDE.md` | 162 | 170 | |

No file crosses 1,000 lines because of this change.

## Findings

### 1. Structural code-quality regressions

**1.1 Two dependency graphs for one crate** (`spikes/metal_diag/Cargo.toml:9,20`,
`spikes/metal_diag/Cargo.lock`)

This creates a second source of truth for the Burn fork pin and the whole resolved
dependency set.
- The next Burn bump (CLAUDE.md says Burn is "a fork pinned in `Cargo.toml`") has to touch
  `Cargo.toml`, `spikes/metal_diag/Cargo.toml` and both locks.
- If anyone forgets, the spike either fails to unify (two Burns, type mismatches across
  the `sciml_rs` boundary) or silently tests a different Burn than the one the bug reports
  are about.
- The spike also builds its own multi-GB `target/`, which the README has to warn about.

**Code-judo move: make the root a workspace whose only default member is the crate.**

```toml
# Cargo.toml (root)
[workspace]
members = [".", "spikes/metal_diag"]
default-members = ["."]          # cargo test / clippy / doc at the root stay crate-only

[workspace.dependencies]
burn = { git = "https://github.com/ax1s-x1zz/burn", rev = "ddfa9af…", features = ["autodiff", "signal", "store", "train"] }

[dependencies]
burn.workspace = true
```

```toml
# spikes/metal_diag/Cargo.toml: drop the empty [workspace] table
[dependencies]
sciml-rs = { path = "../..", default-features = false, features = ["flex"] }
burn.workspace = true
```

This removes:
- the second lockfile (8,185 lines);
- the duplicated pin;
- the separate `target/`, along with the README's warning about it;
- the CLAUDE.md caveat that the spike's lock "pins the root's versions".

CI is unchanged: `default-members` keeps the root commands crate-only. The other spikes
keep their own `[workspace]` tables and stay out.

The spike is run with `cargo test -p metal-diag-spike --release -- --nocapture`.

One cost to check: with the spike as a member, the root `Cargo.lock` gains one package
entry. Its dependency set is the same as the root's, which is why the seeded lock diffs by
only about 10 lines of real content.

### 2. Missed opportunities for dramatic simplification

**2.1 Device-parameterised repro bodies** (`tests/gpu_parity.rs:44-129`)

The GPU repros and their Flex twins duplicate about 60 lines, differing only in
`gpu()` versus `Device::flex()`.
- Verbatim preservation was the stated reason, but it does not hold. The upstream READMEs
  quote self-contained versions with `Device::metal(Default::default())` inlined
  (`upstream/burn/cfft-kernel/README.md:63-66`), not the `gpu()` in this file.
- The file is therefore already not verbatim. Keeping two copies only buys a chance for the
  pair to drift apart. For example, a tolerance fixed in one copy but not the other would
  make the Flex twin stop guarding what the GPU repro checks.

```rust
fn fused_cat_broadcast(device: &Device) {
    let a = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], device);
    // … unchanged body …
    assert_eq!(got, [10.0, 20.0, 60.0, 80.0, 150.0, 180.0, 280.0, 320.0]);
}

fn cfft_bluestein(device: &Device) { /* unchanged body, `device` instead of `&device` */ }

#[cfg(any(feature = "metal", feature = "cuda"))]
#[test]
#[ignore = "Burn fusion bug at the pinned rev: upstream/burn/cfft-kernel/README.md"]
fn repro_fused_cat_broadcast() { fused_cat_broadcast(&gpu()) }

#[test]
fn repro_fused_cat_broadcast_flex() { fused_cat_broadcast(&Device::flex()) }
// same for repro_cfft_bluestein{,_flex}
```

The test names the reports cite stay the same, the pair can no longer diverge, and the
file loses about 40 lines.

**2.2 Move `repro_fusion_cache_new_shape` to the spike** (`tests/gpu_parity.rs:133-183`)

This test should not be in the regression file:
- No upstream report cites it (`grep` of `upstream/`). The review row's claim that
  "`upstream/burn/*` cites these" is true for six of the seven names, not this one.
- It is the bisection variant of `repro_fused_cat_reused_at_new_shape`, with `SHAPES`
  and `WARMS` knobs. Review T2-12 says env-var tests go to the spike.
- It is intermittent: on 2026-10-16 it failed in 3 of 4 isolated runs and passed in the
  full `--ignored` run. A regression test that passes by luck is weaker than none, since
  someone could read a pass as "fixed in the fork".

Moving it deletes the second `chain` / `upload` copy and the only env-parsing code in the
file except `PADS`, about 50 lines. It is a deliberate deviation from the brief's list of
7 tests, so record it in the PR. If you'd rather keep it, inline the defaults and delete
the knobs. Either way the file should not have two copies of the chain.

### 3. Spaghetti / branching complexity

**3.1 Stringly-typed call kinds in `replay_2d_poisson`** (`tests/gpu_parity.rs:246-298`)

`("train", 20)` / `("eval", 3)` are dispatched with `match kind { "eval" => …, _ => … }`
twice. A typo such as `"evel"` silently becomes a training call.

Remedy: `enum Call { Train, Eval }`, matched exhaustively, with the schedule written as
`[(Call::Train, 20), …]`. The `_ =>` arms disappear and the cache key becomes
`(Call, usize)`.

### 4. Boundary / type-contract problems

**4.1 `RefCache` carries a dead field** (`tests/gpu_parity.rs:220,292-294`)

The value type is `(Vec<f32>, Vec<f32>)`, but the second `Vec` is always `Vec::new()` and
only `.0` is read. It is a leftover from the notebook.

Remedy: `HashMap<(Call, usize), Vec<f32>>`, or drop the type alias. Either way the
`(o, Vec::new())` / `.0` dance goes. Now that this is the maintained regression file
rather than the notebook, it is the right time to remove it.

**4.2 No-GPU-feature builds silently skip `gpu()`** (`tests/gpu_parity.rs:19-29`)

This is correct. Note that `wgpu` alone compiles no GPU tests, though: only the cfgs for
`metal` and `cuda` exist, and the module doc does not say so. Add one sentence to the
module doc, nothing more.

### 5. File size / decomposition

None. The spike file (1,419 lines) is a notebook outside the crate. Splitting it is not
worth it.

### 6. Modularity

**6.1 The ignore reasons repeat path strings four times** (`tests/gpu_parity.rs:47,63,140,188`)

This is acceptable: `#[ignore = …]` needs a literal. After 2.2 there are three
occurrences. No action.

### 7. Legibility

- **7.1** `tests/gpu_parity.rs:90,104`: the Flex twins' doc comments read "Bug-report
  repro: …". They are reference checks, not repros. 2.1 makes this moot, because the doc
  then lives on the shared body and each wrapper says which backend it covers.
- **7.2** `CLAUDE.md` "Checks": the added sentence is long. It would read better as its own
  bullet.

## Next steps

Each step is behaviour-preserving and sized as its own commit. They can all go into the T7
PR before merge.

1. **Workspace with the spike as a non-default member** (fixes blocker 1). No dependencies
   on other steps.
   - Add `[workspace]` with `default-members = ["."]` and `[workspace.dependencies] burn`.
   - Change the spike to `burn.workspace = true` and remove its `[workspace]` table and
     `Cargo.lock`.
   - Update the spike README's run command (`-p metal-diag-spike`) and the CLAUDE.md
     `spikes/` line.
   - Verify:
     - the six README checks, where `cargo test` must still list the same binaries and
       counts (lib 208, `gpu_parity` 2, …);
     - `git diff Cargo.lock` adds only the `metal-diag-spike` entry;
     - `cargo clippy -p metal-diag-spike --all-targets -- -D warnings` and one spike
       test on Metal.
2. **Device-parameterised repro bodies** (fixes blocker 2, part 1).
   - Verify: `cargo test --test gpu_parity` (2 pass) and
     `cargo test --release --features metal --test gpu_parity [-- --ignored]`. The same
     pass / ignore / fail pattern as before is expected: 3 pass and 4 ignored, then the
     ignored repros fail.
3. **Move `repro_fusion_cache_new_shape` to the spike** (fixes blocker 2, part 2).
   - Verify: the spike's clippy, and the Metal `gpu_parity` run now shows 3 pass and 3
     ignored.
   - Note the deviation from the brief's list in the PR.
4. **`enum Call` and `RefCache` cleanup in `replay_2d_poisson`** (3.1, 4.1). Independent
   of steps 2–3.
   - Verify: the Metal run gives worst rel err 2.33e-6, unchanged.
5. **Polish** (4.2 doc sentence, 7.1, 7.2), then `cargo fmt` and `cargo doc --no-deps`.

After steps 1–3, `gpu_parity.rs` should be at about 200 lines, and T7's net addition
outside generated files should be under 300 lines.

## Resolution (2026-10-16, same branch, uncommitted)

All five steps were applied.

- **Workspace.**
  - The root `Cargo.toml` has `[workspace]` with `default-members = ["."]` and Burn under
    `[workspace.dependencies]`.
  - The spike uses `burn.workspace = true`, and its `Cargo.lock` is deleted.
  - The root `Cargo.lock` gains only the 8-line `metal-diag-spike` entry.
  - Root `cargo test` does not build the spike.
- **Shared bodies.** `fused_cat_broadcast(&Device)` and `cfft_bluestein(&Device)` back
  both the GPU repros and the Flex twins; all test names are unchanged.
- **`repro_fusion_cache_new_shape`** moved to `spikes/metal_diag/`, ungated and no longer
  ignored. It still fails on Metal: first wrong call 1.
- **`replay_2d_poisson`** uses `enum Call { Train, Eval }` and a
  `HashMap<(Call, usize), Vec<f32>>` cache.
- **Docs.** The module doc notes that `wgpu` alone compiles no GPU tests. The CLAUDE.md
  Checks line is now its own paragraph.

`tests/gpu_parity.rs` is now 248 lines. The `Call` `Display` impl costs about 15 of the
review's estimated 200.

Checks run:
- The six README checks pass, with test counts unchanged from the first T7 pass.
- On Metal, `cargo clippy --features metal --all-targets` and
  `cargo clippy -p metal-diag-spike --all-targets` are clean.
- On Metal, `cargo test --features metal --test gpu_parity` gives 3 passed and 3 ignored.
  `replay_2d_poisson`'s per-call errors are identical to before, with worst 2.33e-6.
- `-- --ignored` gives 3 failures, as expected.
