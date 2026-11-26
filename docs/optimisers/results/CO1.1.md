# CO1.1 results: the reference recipe

2026-11-25, branch `optimisers/co1-1`. Plan: `docs/optimisers/tasks/CO1.1.md`.

## A1 on seed 42 (D10's band, an early reading)

PETScML's reference solver (Zampini et al. 2024, §4.1.1) on our code: Li et al.'s Burgers
data at s = 128 (subsample rate 64), the first 1,000 samples for training and the last 100
for testing; the FNO of 549,569 parameters with GELU; the batch mean of the squared relative
L² as the objective; AdamW, ε 1e-8, weight decay 1e-4, batch 20, lr 1e-3 halved every 100
epochs, 500 epochs; f32 (D10).

```sh
RECIPE=petscml OPTIMIZER=adamw cargo run --release --features metal --example train_burgers
```

Mac (M2 Pro), metal, commit `ae67c38`; run directory `runs/burgers_fno_1795613835`
(git-ignored). Training time 498 s, with `/verify` compiling beside it, so the time is not a
measurement (CO1.3 times the cells on the VM).

| | test L² (mean relative L², eq. 7) |
| --- | --- |
| final (epoch 499) | **7.11e-4** |
| best so far (PETScML's legend metric; epoch 413) | 7.07e-4 |
| PETScML's reference | 1.4e-3 |
| ratio, final / reference | 0.51 |

D10's band is a factor 2 either side of 1.4e-3, [7.0e-4, 2.8e-3]: seed 42 lies inside it, at
its lower edge, and below PETScML's value. One seed is not D10's verdict, which is A1's mean
over seeds 42–46 in CO1.3; seed 42 lying near the lower edge means a few seeds could take
the mean out of the band below, which CO1.3 would then report as a gap (on the good side),
with the differences known so far: f32 against their f64, our erf GELU against Haiku's
default tanh form (open question 1 of the plan), and their unpublished script.

| epoch | train L² | test L² | lr |
| --- | --- | --- | --- |
| 0 | 3.27e-1 | 8.31e-2 | 1e-3 |
| 99 | 2.22e-2 | 1.56e-2 | 1e-3 |
| 100 | 7.67e-3 | 4.26e-3 | 5e-4 |
| 199 | 5.81e-3 | 4.13e-3 | 5e-4 |
| 200 | 2.53e-3 | 2.25e-3 | 2.5e-4 |
| 299 | 9.60e-3 | 7.00e-3 | 2.5e-4 |
| 300 | 3.11e-3 | 1.95e-3 | 1.25e-4 |
| 399 | 1.64e-3 | 2.26e-3 | 1.25e-4 |
| 400 | 1.01e-3 | 7.90e-4 | 6.25e-5 |
| 499 | 6.35e-4 | 7.11e-4 | 6.25e-5 |

The learning rate halves at epochs 100, 200, 300 and 400, as the schedule states. Within
each constant-rate stretch the error oscillates (epochs 99 and 299 are local peaks); each
halving drops it. So the final and the best-so-far values differ little here, but a seed
whose last epoch sits on a peak would show the gap between PETScML's best-so-far metric and
our final one; CO1.3 reports both.

## Constraint 1

Flex, on the Mac, `origin/main` (`14caf47`) against this branch (`b04b318`, before the
review fixes, which change documentation and tests only, and the optimiser `match`'s arms
without changing what any arm builds):

| check | `origin/main` | branch |
| --- | --- | --- |
| `spikes/linalg` `fingerprint` (seeded `FNO<4>` forward, two Adam epochs, a hybrid Gauss–Newton run) | `84e63f28a21024db` | `84e63f28a21024db` |
| `train_burgers`, nothing set, epochs capped at 2 (a local edit of the literal on both sides, not committed): `metrics.csv` columns `epoch` to `current_lr` | 2 rows | equal, bit for bit |
| `train_darcy`, the same | 2 rows | equal, bit for bit |

The existing test suite passes with its only edits the `activation: None` field added to
`FNOConfig { … }` literals (a compile-only change).

## The source check

In the plan (`docs/optimisers/tasks/CO1.1.md`, "The source check"): s = 128 and the
1,000/100 split confirmed against PETScML's text, Lu et al.'s code (s) and Li et al.'s code
(the last 100 samples as the test set, which our loader takes); PETScML's repository has no
Burgers script.
