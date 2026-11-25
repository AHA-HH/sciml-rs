# CM1.6b Aggregators

Started 2026-11-25 on `operators/cm1-6b`. Roadmap: "CM1.6b Aggregators. `WeightedSumLoss`,
`FieldwiseAggregatorLoss`, `SoftAdapt`, `Relobralo` (its Bernoulli draw taken from a seeded
source). *Done when:* each against a fixture of a sequence of steps (the weights and the
aggregated loss per step). *Needs:* CM0.5." With CM1.6a's code-quality follow-ups on the
same line. Proposal: §10; §1.2, §1.4, §2, §5, §6.4. D1, D2, D3 (as amended 2026-11-25), D8.

## Goal

2.0.0's loss aggregators in `losses/`, each matching 2.0.0 on a committed fixture: the
aggregated loss, the per-loss weights and the gradient with respect to each loss, step by
step.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- Roadmap: "each against a fixture of a sequence of steps (the weights and the aggregated
  loss per step)".
- D3 (amended 2026-11-25): max |a − e| / max |e| ≤ 1e-5 in f32, ≤ 1e-12 in f64, on the CPU.
  No FFT: the f64 tests run.
- Proposal §5: output parity; gradient parity for parts with custom backward work. The
  adaptive weights are computed without gradient in 2.0.0 (`torch.no_grad`), so the
  gradient of the aggregate with respect to each loss is part of the fixture.
- Constraint 1 (§1.4): every existing FNO output bit for bit identical.

## Read first

- 2.0.0 `neuralop/losses/meta_losses.py` (`FieldwiseAggregatorLoss`, `WeightedSumLoss`,
  `Aggregator`, `SoftAdapt`, `Relobralo`).
- `losses/{data_losses,sobolev,pointwise}.rs`; `tests/parity_common/mod.rs`,
  `tests/parity_losses_common/mod.rs`; `tools/parity/generate_losses.py`.

## Changes

- `src/neural_operators/losses/meta_losses.rs` (new):
  - `LossFn<R>`: a boxed `Fn(Tensor<R>, Tensor<R>) -> Tensor<1>`, so any of the crate's
    losses can be aggregated through a closure.
  - `WeightedSumLoss<R>`: Σ wᵢ Lᵢ(pred, truth); weights default to 1/n (2.0.0's default).
  - `FieldwiseAggregatorLoss<R>`: per field a name, a channel range on an axis and a loss;
    the mean of the field losses (`loss`), and each field's loss (`field_losses`, in place
    of 2.0.0's `logging`, which returns only the last field's loss). 2.0.0 indexes with
    any tuple of slices and flattens the prediction to `(-1, 1)` but not the target; the two
    agree only for one-channel slices of `(N, C)` outputs, its use (GINO-style point
    outputs). Here both are sliced the same way and keep their shape; parity is shown on
    that use.
  - `SoftAdapt` and `Relobralo`: the state (previous and initial losses, Relobralo's EMA
    of the weights) is kept on the host in f64, read from the scalar losses each step; the
    aggregate is Σ λᵢ Lᵢ with the λᵢ constants, so gradients flow as in 2.0.0. Optional
    per-loss `weights` (2.0.0's `Aggregator.weigh_losses`, its dict keyed by position;
    missing ones 1). `forward(losses, step)`, step 0 initialising, as 2.0.0.
  - `Relobralo`'s ρ ~ Bernoulli(β) drawn from a `ChaCha8Rng` seeded at construction
    (`rand_chacha`, already a dependency; ρ = 1 when a 53-bit uniform draw
    `(next_u64 >> 11) · 2⁻⁵³` is below β, exact at β = 0 and 1), and
    `forward_with_rho(losses, step, rho)` to replay a given draw (the parity test replays
    torch's).
  - 2.0.0's `Aggregator` base class is the shared weighting; it gets no type of its own.
- `losses/data_losses.rs` (CM1.6a review F3, F4): `Reduction::apply`, used by `LpLoss`,
  `SobolevLoss` and `PointwiseQuantileLoss`; `root_zero_safe(sum, p)`, `lp_norm`'s masked
  root, shared with `SobolevLoss`. Same ops in the same order, so `LpLoss` is bit-identical;
  the bit-identity dumps confirm it.
- Harness (CM1.6a review F6, F7): `tests/flex_f64/`, one `flex_f64` declared (behind
  `flex`) by the four f64 binaries, `gauss_newton_f64` among them, so it sits beside
  `parity_common/` rather than in it, and no f32 binary compiles it; `Fixture::tensor(key, device)` public, replacing the copy in
  `parity_losses_common`, which drops the `#[allow(dead_code)]` on that module. The
  `#[allow(dead_code)]` on `mod parity_common` in the two loss binaries stays: they use
  no parameter map (`load_into`, `Take`), and splitting the harness for that is out of
  scope.
- `tools/parity/generate_losses.py`: fixtures `meta_losses_{f32,f64}`; README.
- Conventions: none.

## Tests that prove it

- Parity, in the existing loss binaries (f32 `tests/parity_losses.rs`, f64
  `tests/parity_losses_f64.rs`), against `meta_losses_{f32,f64}`:
  - `WeightedSumLoss([H1Loss(d=1), MSELoss()])` on a pair (2, 2, 32), default weights and
    (0.3, 0.7);
  - `FieldwiseAggregatorLoss` on (16, 3): field `u` channel 0 with `MSELoss`, field `p`
    channel 2 with `PointwiseQuantileLoss(0.1)`;
  - `SoftAdapt` and `Relobralo` over 6 steps of 3 seeded positive losses: per step the
    aggregate, the λ (SoftAdapt) or EMA weights (Relobralo), and the gradient with respect
    to each loss; each without weights and with one (2.0.0 dict `{"b": 2.0}` /
    `{"c": 0.5}`); Relobralo at its defaults and at α = 0.9, β = 0.5, τ = 2, replaying
    torch's recorded ρ draws.
  - The f64 fixtures are made with torch's default dtype f64, so 2.0.0's buffers (created
    with the default dtype) are f64 too.
- Unit tests (`meta_losses.rs`): the seeded Bernoulli at β = 0 and 1 is constant and at
  0.5 is reproducible from the seed; a step before step 0 panics; a wrong number of losses
  panics; mismatched weights panic.
- Constraint 1: `tests/fno_bit_identity.rs` dumps on `main` and the branch, `cmp`'d (the
  change touches `data_losses.rs`).

## Checks

`/verify` (the CI set, `--all-targets`, datagen, the bit-identity dumps) and
`uv run python generate_losses.py --check`.

## VM runs

None.

## Constraints and risks

- Constraint 1: `data_losses.rs` changes by refactor only (`Reduction::apply`,
  `root_zero_safe`); the dumps decide.
- `training/` untouched.
- Risk: 2.0.0's f32 SoftAdapt and Relobralo compute their weights in f32; here in f64 on
  the host. The difference is f32 rounding, inside D3's f32 bound.

## Pending runs

None.

## Open questions

None. The roadmap line's "the two `#[allow(dead_code)]` … gone" is met for the one on
`parity_losses_common`; the one on `mod parity_common` in each loss binary stays, as
"Changes" says (the plan review accepted the reason; recorded in the PR and progress).
