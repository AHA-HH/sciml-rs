# Thermo-nuclear code quality review: phase4s/T1

| | |
|---|---|
| Repo | sciml-rs (`sciml_rs`), worktree `lusca` |
| Branch | `phase4s/T1` (task brief: `docs/phase4s/T1-ci-manifest-hygiene.md`) |
| Base | `a0652eb` (`git merge-base HEAD main`, the tip of `main`) |
| Date | 2026-10-16 |
| Diff | 6 files, +46 / −110 (uncommitted working tree against base, including the staged deletion) |

## Verdict: **APPROVE**

The diff is net deletion: 64 more lines removed than added, and no new code paths. It removes:
- a CI workflow that did nothing
- 40 lines of commented-out apt steps
- two empty manifest tables
- an invalid crates.io category
- two commented-out `cfg` arms

It adds no logic, branches, helpers, types or casts, so none of the structural blockers can
apply. The one Rust change makes a `compile_error!` message shorter and true. The remaining
findings are small consistency gaps in the same files. Each is a few lines and fits inside this
PR; none of them is a reason to hold it.

## Presumptive blockers

None met.
- No incidental complexity kept where a code-judo move exists. The task is itself the
  deletion move.
- No file crosses 1,000 lines.
- No ad-hoc branching.
- No scattered feature checks. The diff removes two disabled `cfg` arms from `lib.rs`.
- No new abstraction, wrapper or cast.
- No duplicated helper.

## File sizes

| File | Before | After | Note |
|---|---:|---:|---|
| `.github/workflows/run-examples.yml` | 30 | 0 | deleted |
| `.github/workflows/run-tests.yml` | 123 | 81 | |
| `CLAUDE.md` | 152 | 158 | |
| `Cargo.toml` | 128 | 125 | |
| `docs/design/2d-chebyshev-poisson-fno.md` | 1,282 | 1,292 | already over 1k; +10 lines of prose (Phase 5 note). Waived: a design document, not code, and §10 is the agreed home for roadmap rows |
| `src/lib.rs` | 70 | 66 | |

## Findings

### Tiers 1–3: structural regressions, missed simplification, spaghetti growth
None. Removing `run-examples.yml` outright, instead of uncommenting its step, was the right call.
Every example needs git-ignored data (`datasets/`, `runs/`), so the step would have failed on
every push. `cargo clippy --no-deps --examples` already gives the compile guarantee.

### Tier 4: boundaries and contracts

**F1. `Cargo.toml:79-80` keeps commented-out `# rocm` / `# vulkan` feature lines, with no reason
given.** This is the same smell T1 removes everywhere else: configuration that is commented out
instead of deleted. It's now also inconsistent. `lib.rs` dropped these arms, and CLAUDE.md:125
says only "`rocm`/`vulkan` are disabled", while the manifest still shows them as if one
uncomment would bring them back. History doesn't help. They were commented out in `ca6857a`
("selectable spectral weight initialisation"), a commit with no connection to backends, and
burn-signal at `ddfa9af` does list both features (`crates/burn-signal/src/ops.rs:10-13`).
- **Remedy:** delete both lines and put a one-line comment in the features block saying why the
  two backends aren't offered. That needs the actual reason, so ask the author. A
  commented-out line can't explain itself.
- **Related:** the priority list at `Cargo.toml:71` (CUDA > Metal > ROCm > Vulkan > wgpu > Flex)
  is Burn's `Device::default()` order and is correct as written. Keep it.

**F2. The CI command list now exists in three places, and the third has already drifted.**
- The workflow is the source.
- CLAUDE.md "Checks" (`CLAUDE.md:94-108`) now copies all seven commands. The brief asks for
  this, and it's correct.
- `docs/phase4s/README.md:67` still says "The CI five". With the chebyshev job included, CI
  runs seven commands. The five it means are those of `run-tests` and `style-checks`.

T1 exists because these copies drifted apart, and this structure lets them drift again. The cheapest
fix is to stop restating the commands and point at the one owner.
- **Remedy:** reword `README.md:67` to "The five commands of the `run-tests` and `style-checks`
  jobs (CLAUDE.md "Checks"), plus …". Better still, drop the restated block and refer to
  CLAUDE.md, so the workflow and CLAUDE.md are the only two copies.
- **Scope:** this edits `docs/phase4s/README.md`, which the brief doesn't name. Say so in the PR
  description, or leave it for T24 (docs pass).

### Tiers 5–6: file size, modularity
Nothing to flag. The only file over 1k lines is the design document (see the table).

### Tier 7: legibility

**F3. Every job installs `clippy, rustfmt` whether it uses them or not** (`run-tests.yml:23, 43,
65`).
- `run-tests` runs neither tool.
- `run-tests-chebyshev` never runs `rustfmt`.

The list tells a reader that each job lints, and only `style-checks` does.
- **Remedy:**
  - `run-tests`: drop the `components:` line.
  - `run-tests-chebyshev`: install `clippy` only.

  This is a small cut in setup time, not new caching, so it stays within the brief's "Do not".
- **Not recommended:** merging the three jobs to remove the repeated
  toolchain/checkout/cache prefix. They run in parallel now, and merging would serialise them to
  save about 9 lines. A composite action is too heavy for three jobs.

## Next steps

Each step is behaviour-preserving and small enough to fold into this PR. They're independent of
each other.

1. **F1:** delete `Cargo.toml:79-80` and state why ROCm/Vulkan are off (needs the author's
   reason). Verify with `cargo metadata --no-deps --format-version 1`,
   `cargo clippy --all-targets -- -D warnings`, and
   `git grep -n -i "rocm\|vulkan" -- Cargo.toml src`, which should then match only the priority
   comment.
2. **F3:** trim the `components:` lists. Verify with the YAML parse
   (`ruby -ryaml -e 'YAML.load_file(".github/workflows/run-tests.yml")'`) and with all three
   jobs green on the PR. This can only be checked on GitHub.
3. **F2:** reword `docs/phase4s/README.md:67`, or defer it to T24 and say so in the PR. Verify
   with `git grep -n "CI five"`, which should return nothing.
