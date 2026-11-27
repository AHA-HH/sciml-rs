# CO0.3 Cost profile of an Adam step

2026-11-27. Plan: `docs/optimisers/tasks/CO0.3.md`. CO0.1's Adam setting (stage 3, n = 129,
s = 64, p = 0, modes [12, 12], width 32, 4 layers, N = 1,000, batch 20, 50 steps per epoch,
200 test samples), `adam_profile`, A100 (GPU 0 of the VM, no other compute process at
the start of each run), cuda, f32.
Three runs, one after another (driver `~/sciml-rs/logs/co0-3/driver.sh`):

| run | build | Burn | TF32 | finished (UTC) |
| --- | --- | --- | --- | --- |
| A | `bc26896` (this branch) | `=0.23.0-pre.2` | off (`cubek.toml`, as CO0.1 ran) | 17:20:32 |
| B | `4541f02` + the example (uncommitted copy) | the fork, `ddfa9af` | the fork's default (no `cubek.toml`) | 17:25:21 |
| C | `bc26896`, `allow_tf32 = true` | `=0.23.0-pre.2` | on | 17:26:44 |

Each run: 3 warm-up epochs, then 3 epochs of each kind; the numbers are means. No library
code changed. The reports and logs are kept outside git
(`~/Code/sciml-rs/runs/co0-3/{burn023,fork,burn023-tf32}.{md,log}` on the Mac).

## Summary

- **The profiler reproduces CO0.1's bisect.** An epoch as `metrics.csv` times it (training
  and evaluation) takes 6.051 s on Burn 0.23 and 1.699 s on the fork, **3.56×**; CO0.1's
  bisect had 5.930 s (#119) and 1.723 s (`4541f02`). The training part alone is 3.61×.
- **Every phase that does tensor work slowed by about the same factor.** Forward and loss
  3.36×, backward 3.32×, the optimiser update 4.13×, evaluation 2.97×. Their shares of the
  step hardly move (forward 31 % on both, backward 55 % against 56 %, the update 13 %
  against 11 %): no single phase went wrong.
- **Where the extra time goes:** the training epoch is 4.12 s longer; of the 4.03 s by
  which the synced phases differ, backward is +2.20 s (55 %), forward and loss +1.25 s
  (31 %), the optimiser update +0.57 s (14 %). The loader, metrics, scheduler, the epoch's
  host reads and the checkpoint together add under 0.01 s per epoch.
- **The step looks host-bound on both Burns.** Per phase, the host's time without a sync is
  close to the time until the device has finished it (on Burn 0.23 within a few per cent;
  on the fork the device times are 7–18 % higher in the large phases), and syncing every
  phase adds only 0.4 % to the epoch on Burn 0.23 and 7.2 % on the fork: there is almost
  no overlap of host and device to lose. That fits a GPU waiting for the host to issue
  work, but also blocking inside each phase (implicit syncs, allocator waits); device
  activity data (Nsight, candidate 1 below) would tell them apart. Either way the extra
  time is in issuing the work: Burn 0.23's host time per step is 3.46× the fork's for the
  forward, 3.49× for the backward and 5.02× for the update.
- **TF32 is not the cause.** With TF32 on (run C) the training epoch takes 5.766 s against
  5.705 s with it off (+1.1 %; each setting ran once, and the warm-up epochs of run A
  alone spread 5.57–5.87 s): a step that waits on the host does not speed up when the
  kernels do. #119's other change to the GPU path, `cubek.toml`, is ruled out;
  what is left is the Burn version.
- **Why a Gauss–Newton step slowed only 1.27×** (CO0.2: 15.11 s against 11.88 s; CO0.1 and
  the roadmap say about 24 %): consistent with this, though not measured here. A GN200 step works on batches of 200, ten times the work per kernel at the same
  number of kernels, so a fixed host cost per kernel is a smaller part of it.

## The profile

Per step (device: synced before and after each phase; host: no sync), seconds; the ratio is
Burn 0.23 (A) over the fork (B), on device times.

| phase | A host | A device | B host | B device | A / B | C device (TF32 on) |
| --- | --- | --- | --- | --- | --- | --- |
| loader (`next`) | 0.00025 | 0.00027 | 0.00007 | 0.00018 | 1.5× | 0.00028 |
| forward and loss | 0.03413 | 0.03566 | 0.00987 | 0.01060 | 3.36× | 0.03698 |
| metrics (MSE, accumulators) | 0.00036 | 0.00047 | 0.00014 | 0.00048 | 1.0× | 0.00042 |
| backward (`backward`, `from_grads`) | 0.06384 | 0.06299 | 0.01827 | 0.01898 | 3.32× | 0.06288 |
| scheduler | 0.00000 | 0.00006 | 0.00000 | 0.00002 | – | 0.00007 |
| optimiser update | 0.01555 | 0.01508 | 0.00310 | 0.00365 | 4.13× | 0.01523 |
| epoch's host reads (both, once per epoch) | 0.00082 | 0.00042 | 0.00064 | 0.00012 | – | 0.00065 |

Per epoch, seconds:

| part | A (Burn 0.23) | B (fork) | A / B | C (TF32 on) |
| --- | --- | --- | --- | --- |
| `train_epoch` (synced at its ends) | 5.7047 | 1.5824 | 3.61× | 5.7658 |
| `evaluate` (test split) | 0.3459 | 0.1163 | 2.97× | 0.3323 |
| **epoch as `metrics.csv` times it** | **6.0506** | **1.6987** | **3.56×** | **6.0981** |
| `save_checkpoint` (one) | 0.3296 | 0.3243 | 1.02× | 0.3404 |
| checkpoint per epoch (one every 25) | 0.0132 | 0.0130 | 1.02× | 0.0136 |
| synced phases, summed | 5.7267 | 1.6957 | 3.38× | 5.7941 |

The extra time per training epoch, from the synced phases (A − B, × 50 steps):

| phase | A − B (s per epoch) | share of the extra |
| --- | --- | --- |
| backward | +2.2008 | 54.6 % |
| forward and loss | +1.2526 | 31.1 % |
| optimiser update | +0.5712 | 14.2 % |
| loader, metrics, scheduler, host reads | +0.0065 | 0.2 % |
| **sum** | **+4.0311** | 100 % |

(The library epochs differ by 4.1223 s; the 0.09 s left over is the fork's larger cost of
syncing every phase, 7.2 % of its epoch against 0.4 %.)

## The measurement holds

- The inline copy of `train_epoch`'s loop, unsynced, against the library's `train_epoch`:
  +0.1 % (A), −0.4 % (B), −0.6 % (C), all within the plan's 10 %. The per-phase times are
  of the loop the library runs.
- The synced phases sum to within 0.4–0.5 % of `train_epoch` on Burn 0.23 and 7.2 % on the
  fork: syncing costs little, because there is almost no host–device overlap to lose.
- Each run started with no compute process on either GPU (the driver's check before each
  run; nothing checked during them). On the Mac a fork smoke that shared the GPU with a
  Burn 0.23 smoke read +162 %, so overlapping runs would have shown up.

## Where the extra time goes

1. **Into issuing tensor work, in Burn 0.23.** The time grows with the phases that issue
   kernels (forward, backward, update, evaluation, 3.0–4.1×); in those that issue few or
   none (the loader, scheduler, metrics, the checkpoint's file writes) the change is
   negligible in absolute terms (under 0.01 s per epoch together). The device's own work
   does not appear to be the limit (above).
2. **Not into one phase or one operation.** Forward and backward both slowed 3.3–3.4×; the
   update slowed most (4.1× on the device, 5.0× on the host), probably because Adam is many
   small element-wise ops per parameter, the case where per-op host cost dominates most.
3. **Not into TF32, the loader, the checkpoint or the metrics** (run C; the per-phase
   table).

This profile cannot tell whether Burn 0.23 issues more operations per step (for example,
less fusion) or spends longer on each one: both look the same from outside the library.
That split is the first item below.

## Candidate fixes (later tasks; nothing changed here)

Any fix moves Adam's equal-time budget (CO0.1's B), so it needs its own task and a decision
on whether to rerun the reference arms.

1. **Count kernel launches per step on both Burns** (Nsight Systems on the VM, the same
   example). This tells more operations (fusion) apart from a higher cost per operation.
2. **Burn 0.23's fusion and autotune settings on cuda**: whether fusion is on in this build
   and what each setting changes, against the fork's, using this example's per-phase table
   as the measure.
3. **A fused (multi-tensor) Adam update**: the update's 4–5× slowdown and 13 % share make
   it the cheapest single target; it would have to keep Adam's outputs bit for bit
   (constraint 1) or be a new optimiser arm.
4. **Report upstream** with this example as the reproduction (batch 20, a small FNO,
   host-bound), and retest on later Burn releases.
5. **Larger batches** reduce the host's share but change the Adam arm itself: an
   experiment-design question for the author, not a fix.
