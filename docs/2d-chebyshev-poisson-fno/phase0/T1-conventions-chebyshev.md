# 2d-chebyshev-poisson-fno phase 0 / T1 - CONVENTIONS §9: Chebyshev grids (C0.1)

Can start at once.

Read first:
- `docs/CONVENTIONS.md`, in full. The preamble's rules on versioning and on how a design proposes a
  convention matter most here.
- `docs/design/2d-chebyshev-poisson-fno.md`: §2.2, §2.3, §2.6, §2.7 and §2.8.

This task adds the Chebyshev-grid conventions as a new §9 of `docs/CONVENTIONS.md`. Phases 1–3 cite
it as `CONVENTIONS §9` in doc comments and tests.

It is a documentation-only change. It needs a human sign-off before any Phase 1 task starts.

Do:
- `docs/CONVENTIONS.md`: append a section `## 9. Chebyshev grids` after §8.
  - Use the text of the design §2.8 diff. Keep all six of its bullets:
    1. nodes;
    2. uniform grid;
    3. domain, with ξ = (x+1)/2 ↔ §2 `arange(s)/(s−1)`;
    4. indexing;
    5. norms;
    6. transfers.
  - Add a seventh bullet for the floored relative error (design §2.7). See below.
  - Phrase it in the style of §1–§8: short declarative bullets, naming the code that implements
    each rule.
  - Wherever that code does not exist yet, name the planned item and mark it "(planned, Phase 1)" or
    "(planned, Phase 2)":
    - `chebyshev::nodes::gl_nodes`, `cc_weights` (Phase 1);
    - `chebyshev::transfer::{ChebToUniform, UniformToCheb}` (Phase 2).
  - State the node formula with its exact order of operations, and why:
    - exact antisymmetry;
    - x_mid = 0;
    - endpoints exactly ±1;
    - bitwise nesting, n = 2^k+1 nodes ⊂ 2^{k+1}+1 nodes.
  - State that RLST's `chebychev_points` is descending, cos form, and is not used for nodes.
  - State that the floored relative error of design §2.7 (τ = 10⁻³ × the test-set median norm) is
    the convention for reporting relative errors on Chebyshev grids.
- In the preamble, extend the sentence "Any change to §1–§6 bumps…" so that it also says what a
  change to §9 needs. Recommended:
  - §9 changes need a dataset regeneration note in the PR;
  - they need no version bump, because §9 does not affect checkpoints.
- **Do not change `CONVENTION_VERSION`.** It stays 1: the preamble requires a bump only for §1–§6.
- PR description:
  - say "adds CONVENTIONS §9 (no version bump: §1–§6 untouched)";
  - quote the seven rules (six from design §2.8, plus the §2.7 floored error);
  - add the line "Needs human sign-off before Phase 1".

Tests that define done:
- No code changes, so there are no new tests.
- Review checks, each stated in the PR:
  - `git diff main -- docs/CONVENTIONS.md` shows only additions: §9 and the one preamble sentence.
  - `grep -n "CONVENTION_VERSION = 1" docs/CONVENTIONS.md` still matches.
  - Every formula in §9 agrees in meaning with design §2.2, §2.6, §2.7 and §2.8.
    The reviewer compares them side by side.

Must pass: the repository's full checks, to show nothing else changed:

```sh
cargo fmt -- --check
cargo clippy --no-deps -- -D warnings
cargo clippy --no-deps --examples -- -D warnings
cargo test
cargo doc --no-deps
cargo clippy --all-targets -- -D warnings
```

Do not:
- edit §1–§8 of `docs/CONVENTIONS.md`, or bump `CONVENTION_VERSION`;
- edit the design document, beyond fixing a contradiction you find (report it in the PR);
- add code, or cite §9 from code. Phase 1 does that.
