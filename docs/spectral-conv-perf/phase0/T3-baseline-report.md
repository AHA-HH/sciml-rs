# spectral-conv-perf phase 0 / T3 - baseline report (C0.5)

Starts after T1 and T2 have both merged.

Read first:
- the phase README (exit gate, "Baseline" tables);
- `docs/design/spectral-conv-perf.md` §7 phase 0, §8.2 and §9.1 (R4);
- T1's and T2's PR descriptions, for their flex and Metal outputs.

This task runs the phase 0 tools on today's code and records the numbers that phases 1–3
must beat and match. It is a documentation task: it changes no code.

Do:
- **Rerun on the merged `main`**, so that both tools see the same code. Use one machine,
  and say which (chip, OS) in the PR.
  - `cargo test --lib neural_operators::layers::spectral_convolution::oracle:: -- --nocapture`
  - `cargo test --lib --features metal neural_operators::layers::spectral_convolution::oracle:: -- --nocapture`
  - `cargo run --release --example bench_spectral_conv`
  - `cargo run --release --features metal --example bench_spectral_conv`

  If Metal is unavailable, record "not run (no Apple GPU)" in its rows. Label the
  backend by build feature (phase README, "Backend labels").
- `docs/spectral-conv-perf/phase0/README.md`:
  - fill in the two "Baseline" tables: every benchmark row for both backends, and every
    oracle case with its observed E-oracle error and tolerance;
  - add the commit hash measured and the machine;
  - **R8 (Metal `cfft`).** Record every Metal oracle failure with its value. Mark the
    Metal rows of the 2D layer case and the Darcy FNO step "computes wrong values (R8)":
    they time a wrong computation, so they are not a valid baseline.
- `docs/design/spectral-conv-perf.md`:
  - add a dated revision line;
  - add to §7 phase 0 a "Status: done" line with the commit hash and a short results
    summary (the slowest and fastest cases on Metal);
  - add an "Observed (phase 0)" column to §8.2's two oracle rows;
  - update §9.1 R4:
    - "retired" if every observed error **on flex** is within its tolerance;
    - otherwise, "open", with the measured value;
  - update §9.1 R8 with the measured Metal values.
- **Ask the user explicitly, in the PR, to decide R8:** does the Metal gate for the
  2D/Darcy settings stand as is, or must a fork fix land first? Do not decide it.
- **Tolerance for s = 256 and 1024.**
  - The numerics review observed 1.8e-7 to 5.5e-7 there, so 3e-5 is about 60× looser
    than needed.
  - If the measurements confirm this, propose 1e-5 for all sizes in the PR, for the
    user's sign-off. Otherwise, say why 3e-5 stays.
- **Only if some observed E-oracle error on flex exceeds its tolerance** (a T1 case marked
  `#[ignore]`):
  - propose a revised §8.2 row in the PR. Choose the smallest round tolerance at least
    3× the observed value.
  - Check that it stays below the E-control mutations T1 measured (≥ 1e-3) by ≥ 10×.
  - Ask the user explicitly for sign-off.
  - Do not un-ignore the test in this PR. That follows the sign-off, in its own small PR.
- `CLAUDE.md`: no change in this task. The pointer moves to phase 1 when phase 1 is
  planned.

Tests that define done (checks, since this task is docs only):
- Every oracle case of T1 (1D ×5, 2D ×3, 3D ×1) has a row with an observed value and its
  tolerance.
- Every benchmark case of T2 (6 cases) has a row for flex, and one for Metal or
  "not run".
- Every number in the README appears in the PR's pasted command output.
- `git diff --stat main` lists only the phase 0 README and the design document.

Must pass:
- `cargo fmt -- --check` and `cargo doc --no-deps`. No Rust changes, so these confirm
  nothing else was touched.
- The four commands above, with their outputs pasted in the PR.

Do not:
- Change any code, test or tolerance in code, or `docs/CONVENTIONS.md`.
- Re-time selectively or drop outlier runs. Report what the protocol produced. If a run
  looks wrong, rerun the whole table and say so.
- Tick the phase 0 exit checklist's sign-off items (§8.2 revision, R8 decision)
  yourself.
- Try to fix the Metal `cfft` bug. It is in the Burn fork, outside this objective.
