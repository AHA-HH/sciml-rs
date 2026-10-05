# pytorch-parity phase 0: baseline, the open FFT point, protection of defaults

Phase 0 prepares the ground before any neuraloperator feature is added. It does four
things:
- records the PyTorch number that the Darcy benchmark in Phase 3 must reproduce;
- closes the one open point in CONVENTIONS §4 (how `irfft` treats the imaginary parts of
  the DC and Nyquist bins), on which following neuraloperator main #702 relies;
- adds a test that pins every parameter a default `FNOConfig` draws at a fixed seed, so
  that no later `Option` field can change today's model without being noticed;
- lands CONVENTIONS version 2, the convention lines for every option the later phases
  add, in one diff with one version bump.

The phase ends when all four are merged and `cargo test` is green.

Companion documents:
- [docs/design/pytorch-parity.md](../../design/pytorch-parity.md): §2.1 (DC/Nyquist),
  §2.3 (module structure and RNG rules), §2.6 (the CONVENTIONS diff) and §8.3 (the
  benchmark). Components C0.1–C0.4 are in §7.
- `docs/CONVENTIONS.md`: §4 and §6, plus the preamble's versioning rule.

## Scope

In scope:
- `docs/design/pytorch-parity.md` §8.3: the PyTorch baseline numbers (T1).
- `src/neural_operators/layers/spectral_convolution.rs`, `mod tests` only: the `irfft`
  DC/Nyquist tests (T2).
- `tests/fno_default_params.rs` (new): the pin on the default FNO's parameters (T3).
- `docs/CONVENTIONS.md`: version 2 (T4).

Out of scope:
- Any change to library code under `src/` outside `mod tests`. Every feature belongs to
  Phases 1–5 (design §7).
- Q7, the `output_shape` resizing semantics. T4 writes that line as pending, and Phase 4
  decides it.
- Any Python file in the repository. The baseline script stays outside the repo
  (design A1; the user's answer to Q2).

## Design decisions for this phase

These hold for every task, so that no task decides them on its own.

- **References are pasted, not generated.**
  - NumPy and PyTorch values go into Rust tests as literals, with a comment naming the
    tool, its version and the snippet that produced them.
  - No fixture files and no Python in the repo.
- **One convention diff.**
  - T4 is the only task that edits `docs/CONVENTIONS.md`. Its content is design §2.6,
    nothing more.
  - The §4 line names the test `irfft_ignores_dc_nyquist_imag` that T2 adds. Both briefs
    fix that name, so the two tasks do not conflict.
- **The RNG is process-wide** (Flex). Any test that seeds goes in its own test binary,
  and holds a mutex for its whole body, as `tests/spectral_init.rs` does.
- **Burn parameters are lazy.**
  - `Linear` and `Conv1d` parameters are drawn on *first access*: `Initializer::init_with`
    returns `Param::uninitialized`
    (`~/.cargo/git/checkouts/burn-4aa1ec9707f89bee/ddfa9af/crates/burn-nn/src/initializer.rs:100-125`).
  - `SpectralConv` draws its weights *eagerly* in `new_with_init`
    (`layers/spectral_convolution.rs:181-200`).
  - So a model's weights depend on two orders: the order of construction, and the order
    of first access.
- **Error measures.** Every test names the one it uses.

  | Name | Rule | Used by |
  |---|---|---|
  | `E-bits` | exact equality of f32 bit patterns (`f32::to_bits`) | T3 |
  | `E-ref` | ‖a − b‖_∞ ≤ 1e-5 · ‖b‖_∞ + 1e-7, f32, against pasted NumPy float64 values rounded to 9 decimals | T2 |
  | `E-same` | ‖a − b‖_∞ ≤ 1e-6 · ‖b‖_∞ + 1e-7, f32, between two outputs of the same code path | T2 |
  | `E-diff` | ‖a − b‖_∞ ≥ 1e-2, a control that a test is not vacuous | T2, T3 |
  | `E-grad` | ‖a − b‖_∞ ≤ 1e-4 · ‖b‖_∞ + 1e-7, f32, against pasted NumPy float64 gradients (design §8.2) | T2 (gradients) |
| `E-zero` | exactly 0.0; only on the power-of-two `irfft` path, where Burn sets these exactly. On Bluestein the zeros hold to round-off (≈ 1e-7) and are checked with `E-grad` | T2 (n = 8 gradients) |
  | baseline | final metric per seed, as printed by neuraloperator; mean and sample std (ddof = 1) over 5 seeds | T1 |

## Exit gate

- Every acceptance test in the task briefs passes in CI (`.github/workflows/run-tests.yml`).
- Design §8.3 contains the five-seed PyTorch baseline: 16_l2, 16_h1, 32_l2 and 32_h1
  (mean and std each), the parameter count, and the versions and hardware used.
- `docs/CONVENTIONS.md` is at `CONVENTION_VERSION = 2`, and its §4 open point is closed
  by T2's test.
- `tests/fno_default_params.rs` is green on flex. Whether it was also run with
  `--features metal` is recorded in the T3 PR.
- **Sign-off.** The user approves the T4 diff in its PR. No Phase 1 task starts before
  T3 and T4 are merged (design §7).
- Measured results are in the PRs, and are copied into the design document (§7 phase 0
  status, §8.3).

## Tasks

One pull request each. **All four can start at once.** They touch disjoint files, so no
merge conflicts are expected. **Merge T4 after T3**, because T4's preamble cites T3's
test file:
- T1 edits only design §8.3;
- T2 edits only the tests module of `spectral_convolution.rs`;
- T3 adds one new test file;
- T4 edits only `docs/CONVENTIONS.md`.

T1 needs the user. Its numbers come from running neuraloperator in Python, which an agent
can do only if the user's machine has the environment and the user agrees.

| Task | Brief | Delivers | Component | Depends on |
| --- | --- | --- | --- | --- |
| T1 | [T1-darcy-baseline.md](T1-darcy-baseline.md) | Five-seed PyTorch Darcy baseline in design §8.3 | C0.1 | none |
| T2 | [T2-irfft-dc-nyquist.md](T2-irfft-dc-nyquist.md) | Tests of `irfft` on DC and Nyquist imaginary parts, values and gradients | C0.2 | none |
| T3 | [T3-default-param-pin.md](T3-default-param-pin.md) | `tests/fno_default_params.rs`: default FNO draws pinned, 1-D and 2-D | C0.3 | none |
| T4 | [T4-conventions-v2.md](T4-conventions-v2.md) | `docs/CONVENTIONS.md` version 2 (design §2.6) | C0.4 | none to start; merge after T3 (its preamble cites T3's file); user sign-off in the PR |

## How to run a task with Claude Code

Each task gets its own branch and worktree, as `CLAUDE.md` requires: "One task per
branch and PR". For example, `git worktree add ../goldeye-p0t2 -b
anees/pytorch-parity-p0-t2 main`.

In that worktree, start `claude` and say
"/do-task docs/pytorch-parity/phase0/T<k>-<name>.md". Review and merge before starting
a task that depends on it. In this phase, no task depends on another.

## Exit checklist

- [ ] T1 merged: baseline mean ± std for 16_l2, 16_h1, 32_l2, 32_h1 and the parameter
      count recorded in design §8.3
- [ ] T2 merged: `irfft` ignores the DC (all n) and Nyquist (even n) imaginary parts
      (exactly for power-of-two n, to round-off via Bluestein); the gradients match
      NumPy; metal run recorded
- [ ] T3 merged: default 1-D and 2-D FNO draws pinned bit for bit against a
      hand-built mirror module; the two vacuity controls differ
- [ ] T4 merged and signed off: `CONVENTION_VERSION = 2`
- [ ] Design document updated: §7 phase 0 status and measured results; §9.1 risks
      retired ("a new option accidentally changes a default", "`irfft` DC/Nyquist")
- [ ] `CLAUDE.md` "Active objectives" points to `docs/pytorch-parity/phase1/README.md`
