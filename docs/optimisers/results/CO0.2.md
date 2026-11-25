# CO0.2 Cost profile of a Gauss–Newton step

2026-11-25. Plan: `docs/optimisers/tasks/CO0.2.md`. One GN200-D step (batch 200, Hutchinson
diagonal, CG cap 50, μ = 0) on Burn 0.23 (`=0.23.0-pre.2`), A100 (GPU 0 of the VM, alone),
cuda, f32, from the checkpoint of Phase 10's GN200-D run at μ = 0, seed 42
(`poisson_fno_n129_s64_p0_gn_1794488531`, step 80, trained on the fork at `9dbeb91`).
`gn_profile` at commit `f893620`, 13:58–14:06 UTC; its report is reproduced below
("The profiler's report"). No library code changed.

## Summary

- **A step at cap 50 takes 15.3 s (mean of 10), 13.7 passes per second.** On the fork the
  same run's last ten steps (all at cap 50, the same 155 forward and 52 backward passes)
  took 11.88 s without a refresh and 13.34 s with one, 17.4 passes per second (`awk` over
  the run's `gn_steps.csv`; the same definition, the step only). Burn 0.23 is **1.27×
  slower per step** (15.11 s against 11.88 s, steps without a refresh; 17.10 s against
  13.34 s with one), 0.073 s per pass against 0.057 s.
- **CG's curvature products are 98 % of the step.** Per product (cap 50, CG in place): JVP
  0.124 s (two forwards at θ ± hv), VJP 0.119 s (a fresh forward at θ and a backward), CG's
  own work 0.063 s (vector updates and five host reads per iteration). Of the
  measured step: JVPs 41 %, VJPs 40 %, CG's own work 21 %, the unexplained row −3.8 %.
- **Everything else is under 3 %** at cap 50: the gradient (0.8 %), the Hutchinson refresh
  (1.17 s every 10 steps, 0.8 %), backtracking (0.3 %), the adjoint check (0.2 %), and
  outside the step `train_epoch_gn`'s metric forward (0.047 s).
- **At cap 150** the mean step is 25.8 s (13.8 passes per second): three of the ten steps
  reach the cap (44–46 s), the rest stop by Martens' rule after 31–66 products; the cost
  per product is the same as at cap 50 (0.29 s), and the shares barely move (products 98 %,
  backtracking 1.3 %). The cap is CO0.1's question (D1), not a saving.
- **The measurement holds.** Predicted (counts × unit costs) against measured step time:
  −3.8 % on the mean at cap 50 (per step −0.3 % to −5.9 %), −0.6 % at cap 150 (+1.0 % to
  −4.4 %), under the plan's 15 % flag, so no in-library timer is needed. The counts agree
  with `GnStep` on every step (a capped step: 51 products, 155 forward and 52 backward
  passes).

## Where passes could be saved (for later work; nothing changed here)

In order of the time at stake at cap 50 (15.3 s per step). Each is a proposal under proposal
§1.3 (changing Gauss–Newton's algorithm is a non-goal of this objective), not a decision.

1. **The Burn 0.23 regression, about 3.2 s per step (21 %).** The same passes cost 1.27×
   what they did on the fork. CO0.1's stage 1 found Adam (batch 20, host-bound) about 3.5×
   slower on Burn 0.23, and its bisect over the fork, #119, CM0.3 and `main` is running. A
   regression that grows as the batch shrinks would fit a per-kernel or per-sync overhead
   (a hypothesis, not measured here). If the bisect finds a cause and it is fixed, Gauss–Newton
   gains more than from any change below; this profile cannot split the fork's step into
   phases (the fork has no profiler run), so which phase regressed is open.
2. **CG's own work, up to 3.2 s per step (21 %).** Each iteration reads five scalars to the
   host (p·Ap, ‖r‖, r·z and the two dots of Martens' quadratic x·b and x·r, which one dot
   x·(b + r) could replace; `cg.rs`), and each `ParamVec` operation launches one kernel per parameter
   tensor. One flat parameter vector, the two quadratic dots merged, or the stop test every
   few iterations would cut syncs and launches; how much of the 3.2 s each recovers is not
   measured.
3. **One-sided finite-difference JVPs, about 2.4 s per step (16 %).** The central difference
   costs two forwards per product; r(θ + hv) − r(θ), with r(θ) computed once per step (the
   gradient already has it), costs one: 51 forwards fewer per capped step, a third of the
   step's forwards. The cost is accuracy, O(h) against O(h²) (Phase 10 chose central,
   design §13.17); the adjoint check and CG's convergence would have to show it is harmless.
4. **Reuse the forward at θ in the VJP, up to 2.4 s per step (16 %).** Every product's VJP
   recomputes the forward at the same θ (0.047 s); only u changes between iterations.
   Backward-only VJPs need the graph kept across backward calls, which Burn's autodiff
   consumes; whether Burn 0.23 allows it is not checked.
5. **Not worth it:** the gradient, the refresh, the adjoint check, backtracking and the
   metric forward are together under 3 % of a capped step.

## The profiler's report

Written on the VM by `gn_profile -- docs/optimisers/results/CO0.2-vm.md
~/sciml-rs/runs/poisson_fno_n129_s64_p0_gn_1794488531 50 150` (log `~/sciml-rs/logs/co0-2.log`),
verbatim below its title.

Run `/home/ucaba82/sciml-rs/runs/poisson_fno_n129_s64_p0_gn_1794488531` at step 80 (λ 1.760e-5, μ 0), batch 200 of 819200 residual entries; backend ["cuda"], commit f893620bc69825203801ceb48e6821fcfc87ab39. Unit costs: median of 10 after 2 warm-up calls, each synced.

### Unit costs

| unit | s |
| --- | --- |
| Φ forward (no autodiff) | 0.0465 |
| gradient (vjp, u = r) | 0.1176 |
| JVP (finite differences) | 0.1040 |
| Hutchinson refresh | 1.1679 |
| adjoint check | 0.2346 |

### CG cap 50

CG in place: 51 products, 0.1236 s per JVP, 0.1188 s per VJP, 0.0631 s of CG's own work per product.

| step | products | refresh | adjoint | backtracking | forward | backward | measured (s) | predicted (s) | gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | 51 | true | true | 1 | 168 | 63 | 17.104 | 17.151 | -0.3 % |
| 81 | 51 | false | false | 1 | 155 | 52 | 15.323 | 15.749 | -2.8 % |
| 82 | 51 | false | false | 1 | 155 | 52 | 15.145 | 15.749 | -4.0 % |
| 83 | 51 | false | false | 1 | 155 | 52 | 15.033 | 15.749 | -4.8 % |
| 84 | 51 | false | false | 1 | 155 | 52 | 15.050 | 15.749 | -4.6 % |
| 85 | 51 | false | false | 1 | 155 | 52 | 15.013 | 15.749 | -4.9 % |
| 86 | 51 | false | false | 1 | 155 | 52 | 14.875 | 15.749 | -5.9 % |
| 87 | 51 | false | false | 1 | 155 | 52 | 15.100 | 15.749 | -4.3 % |
| 88 | 51 | false | false | 1 | 155 | 52 | 15.434 | 15.749 | -2.0 % |
| 89 | 51 | false | false | 1 | 155 | 52 | 14.971 | 15.749 | -5.2 % |

| phase (mean per step) | s | share |
| --- | --- | --- |
| gradient (1 forward, 1 backward) | 0.118 | 0.8 % |
| Hutchinson refresh | 0.117 | 0.8 % |
| CG's JVPs (2 forwards each) | 6.305 | 41.2 % |
| CG's VJPs (1 forward, 1 backward each) | 6.060 | 39.6 % |
| CG's own work (vectors, host reads) | 3.219 | 21.0 % |
| backtracking (Φ forwards) | 0.047 | 0.3 % |
| adjoint check (JVP and VJP) | 0.023 | 0.2 % |
| unexplained | -0.584 | -3.8 % |
| **step, measured** | **15.305** | 100 % |
| metric forward (`train_epoch_gn`, 1 per step, outside the step) | 0.047 | – |

Passes per second of the steps: 13.68 (10 steps, 2094 passes, 153.0 s).

### CG cap 150

CG in place: 151 products, 0.1223 s per JVP, 0.1204 s per VJP, 0.0510 s of CG's own work per product.

| step | products | refresh | adjoint | backtracking | forward | backward | measured (s) | predicted (s) | gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | 151 | true | true | 1 | 468 | 163 | 45.718 | 45.913 | -0.4 % |
| 81 | 151 | false | false | 1 | 455 | 152 | 44.958 | 44.510 | +1.0 % |
| 82 | 151 | false | false | 1 | 455 | 152 | 44.315 | 44.510 | -0.4 % |
| 83 | 66 | false | false | 15 | 214 | 67 | 20.084 | 20.199 | -0.6 % |
| 84 | 44 | false | false | 14 | 147 | 45 | 13.592 | 13.691 | -0.7 % |
| 85 | 36 | false | false | 13 | 122 | 37 | 11.129 | 11.295 | -1.5 % |
| 86 | 49 | false | false | 14 | 162 | 50 | 14.831 | 15.160 | -2.2 % |
| 87 | 31 | false | false | 12 | 106 | 32 | 9.365 | 9.780 | -4.4 % |
| 88 | 32 | false | false | 2 | 99 | 33 | 9.213 | 9.609 | -4.3 % |
| 89 | 151 | false | false | 1 | 455 | 152 | 44.426 | 44.510 | -0.2 % |

| phase (mean per step) | s | share |
| --- | --- | --- |
| gradient (1 forward, 1 backward) | 0.118 | 0.5 % |
| Hutchinson refresh | 0.117 | 0.5 % |
| CG's JVPs (2 forwards each) | 10.541 | 40.9 % |
| CG's VJPs (1 forward, 1 backward each) | 10.380 | 40.3 % |
| CG's own work (vectors, host reads) | 4.395 | 17.1 % |
| backtracking (Φ forwards) | 0.344 | 1.3 % |
| adjoint check (JVP and VJP) | 0.023 | 0.1 % |
| unexplained | -0.154 | -0.6 % |
| **step, measured** | **25.763** | 100 % |
| metric forward (`train_epoch_gn`, 1 per step, outside the step) | 0.047 | – |

Passes per second of the steps: 13.84 (10 steps, 3566 passes, 257.6 s).
