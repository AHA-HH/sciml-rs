# CO0.2 Cost profile of a Gauss–Newton step

Started 2026-11-25 on `optimisers/co0-2`. Roadmap: "CO0.2 Cost profile of a Gauss–Newton
step. Time per phase of one GN200-D step (the gradient, CG's JVPs and VJPs, backtracking,
the Hutchinson refresh), passes per second on Burn 0.23 against Phase 10's." Proposal:
§1.2 (O0), §1.3 (changing Gauss–Newton's algorithm is a non-goal; "O0's profile may
propose such work"), §1.4, §2, §3, §7.

## Goal
Measure where the time of one GN200-D step goes on Burn 0.23 (A100, cuda, f32), and its
passes per second against Phase 10's fork-era run, without changing any library code.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
"the profile table and a short list of where passes could be saved (for later work;
nothing changed here)." Proposal §1.2: "a cost profile of one Gauss–Newton step"; §1.3:
the profile may propose changes to Gauss–Newton; they are not part of this objective.

## Read first
- `src/neural_operators/optim/gauss_newton/step.rs` (`GaussNewton::step`: the gradient by
  `residual_and_gradient`, the Hutchinson refresh every `precond_every` steps, CG with the
  curvature closure `jvp` + `vjp`, `select`'s backtracking forwards, the adjoint check every
  `adjoint_check_every` steps, the pass counts of `GnStep`).
- `src/neural_operators/optim/gauss_newton/{products,cg}.rs`; `training/gn.rs`
  (`train_epoch_gn`'s one metric forward per step, outside `GaussNewton::step`).
- `docs/poisson2d/phase10/results.md` ("Cost per update": GN200-D 141.0 forward and 47.1
  backward passes per update, 17.3 passes per second; "Follow-ups": GN200-D at μ = 0).
- The reference run: Phase 10's GN200-D at μ = 0, seed 42, equal time
  (`runs/gauss_newton_followups/mu0_seeds/poisson_fno_n129_s64_p0_gn_1794488531`, commit
  `9dbeb91`, the fork): its weights, `checkpoint.bin` (with the Gauss–Newton state),
  `train_cfg.json` (cap 50, μ = 0, Hutchinson 10 probes every 10 steps, adjoint check every
  10, batch 200) and `gn_steps.csv` (per-step passes and wall time on the fork).

## Changes
- **No library code.** A new example `gn_profile` (`examples/profile/gn_profile.rs`,
  registered in `Cargo.toml` with `test = true`), built from the public Gauss–Newton API
  (`vjp`, `jvp`, `cg`, `hutchinson_diagonal`, `adjoint_error`, `GaussNewton::from_state`,
  `FnoResidual`, `RunDir`). Nothing in `training/`, `optim/`, `runs/` or `models/` changes,
  so constraint 1 holds by construction.
- `gn_profile -- <out.md> <run_dir> [<cg_cap> ...]` (caps default 50 and 150): reads the
  run's checkpoint (model and Gauss–Newton state), configs and normalisers, batches the
  run's training split at its batch size (200) without shuffling, and on the first batch:
  1. **Unit costs** (2 warm-up calls, then the median of 10, each call bracketed by
     `Device::sync`): Φ's forward without autodiff (`select`'s backtracking and
     `train_epoch_gn`'s metric forward), the gradient (`vjp` with u = r: one tracked forward
     and one backward, as `residual_and_gradient`), the JVP (finite differences, two
     forwards), the Hutchinson refresh (`hutchinson_diagonal`, the run's probes), the
     adjoint check (`adjoint_error`).
  2. **CG in place**, per cap: `cg` on the step's system (−g, the checkpoint's diagonal
     preconditioner, the run's λ and μ), its curvature closure timing its JVP and VJP; CG's
     own work per iteration = (CG's time − the products') / iterations.
  3. **Real steps**, per cap: 10 `GaussNewton::step`s from the checkpoint's state, cycling
     the split's batches, each timed (synced) with its `GnStep`. Per step, the counts
     (curvature products = CG's iterations, plus one when CG starts warm, known from the
     optimiser's state before the step, plus one on a negative-curvature stop; the
     adjoint check from `GnStep`; the refresh from the backward passes; backtracking
     forwards = forward passes − 1 − 3·products − probes·refresh − 3·adjoint) times the unit
     costs give a predicted time; the gap to the measured time is reported.
  4. **The profile table**: per phase (gradient, Hutchinson refresh, CG's JVPs, CG's VJPs,
     CG's own work, backtracking, adjoint check, the metric forward, unexplained), the mean
     count and time per step and its share; and passes per second of the real steps.
     **Amended 2026-11-25 (plan review):** the reference run's own passes per second (Σ
     passes / Σ `wall_time_s` of its `gn_steps.csv`, its last 10 steps and all of them)
     are computed in the report by one `awk` line, not by the example: run-file names
     appear only in `runs/` (root `CLAUDE.md`), whose per-row reader would be new code in
     the library objective's CM0.4 lane. On the Mac's copy of the run: all 80 steps 17.36
     passes per second, 11.12 s per step; the last 10 (cap 50 reached on each) 17.41, 12.03
     s per step (11.88 s without a refresh).
- The report `docs/optimisers/results/CO0.2.md`: the example's tables, the comparison with
  Phase 10's 17.3 passes per second (`costs.json`'s definition, the whole training loop) and
  the list of where passes could be saved.
- Conventions: none.

## Tests that prove it
- Unit tests in the example (run by `cargo test`): the per-step decomposition from a
  `GnStep`'s counts against hand-computed counts for a refresh step, an adjoint step and a
  plain one, from three rows of the reference run's `gn_steps.csv` (steps 0, 70, 71); an
  adjoint check without a refresh and a negative-curvature stop; a step without the
  preconditioner; unexplained backward passes rejected; the phases by hand. (The
  `gn_steps.csv` reader and its test were dropped by the amendment above.)
- The measurement itself is validated by its reconciliation: predicted against measured step
  time; a gap above 15 % on the mean is reported as such and is a question for the author
  (an in-library, off-by-default timer would then be the follow-up).
- Constraint 1: no library file changes (`git diff --stat origin/main` lists only the
  example, `Cargo.toml` and docs); the existing suite runs unedited.

## Checks
`/verify` (fmt, clippy with `--examples` and `--all-targets`, `cargo test`, `cargo doc`).
A smoke run on the Mac (metal) with a small cap against the local copy of the reference run,
proving the fork-era checkpoint loads on Burn 0.23 and the tables are written:
`cargo run --release --features metal --example gn_profile -- target/co0-2-smoke.md ~/Code/sciml-rs/runs/gauss_newton_followups/mu0_seeds/poisson_fno_n129_s64_p0_gn_1794488531 5`.
Metal timings are not reported.

## VM runs
One run, about 15 minutes after the build (units under a minute; CG in place up to 150
iterations at about 0.25 s; 10 real steps at about 11 s per step at cap 50 and up to three
times that at cap 150). **Alone on the VM**: the step is host-bound (Phase 9 T3), so it
waits until no other timed run is on the machine (3D CL0.1's `co31-cl01` must have
finished), and runs before CO0.1's runs start or between two of them. At the author's go:

```sh
ssh cricket
tmux ls                                    # no other timed run
ls ~/sciml-rs/runs/gauss_newton_followups/mu0_seeds/poisson_fno_n129_s64_p0_gn_1794488531
cd ~/sciml-rs && git fetch origin
git worktree add --detach ~/sciml-rs-wt/co0-2 origin/optimisers/co0-2
cd ~/sciml-rs-wt/co0-2
find ~/sciml-rs/datasets -mindepth 1 -maxdepth 1 ! -name README.md -exec ln -s {} datasets/ \;
source ~/sciml-gpu.sh
export CARGO_TARGET_DIR=~/sciml-rs-target/optimisers
tmux new -d -s co0-2 "cargo run --release --no-default-features --features cuda --example gn_profile -- docs/optimisers/results/CO0.2-vm.md ~/sciml-rs/runs/gauss_newton_followups/mu0_seeds/poisson_fno_n129_s64_p0_gn_1794488531 50 150 2>&1 | tee ~/sciml-rs/logs/co0-2.log"
```

It decides nothing on its own: it fills the profile table and the savings list (the roadmap's
*Done when*), and its cap-50 and cap-150 step costs feed CO0.1's D1 proposal.

## Constraints and risks
- Constraint 1 (proposal §1.4): binds; met by changing no library code.
- The checkpoint is fork-era (`9dbeb91`); if Burn 0.23 cannot read it, the smoke run shows
  it, and the profile starts from CO0.1's seed-42 GN200-D run instead (waits for CO0.1).
- Unit costs measured in isolation may differ from the same work inside a step (fusion,
  overlap); the reconciliation gap measures that.
- The real steps at both caps continue from the same cap-50 state (λ, warm start,
  diagonal); counts at cap 150 are those of cap-150 steps after cap-50 training.
- 2D objective: no output changes; the run directory is only read.

## Pending runs
<filled in step 6>

## Open questions
<filled when they arise>
