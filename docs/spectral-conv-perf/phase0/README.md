# spectral-conv-perf phase 0: references and baseline

This phase builds the two things every later phase is judged against, before any change
to `SpectralConv`:
- a trusted f64 oracle of the layer, written without Burn's FFTs;
- a benchmark of the layer and of a full FNO training step, with baseline numbers
  recorded on flex and Metal.

Phase 1 (restructured FFT path) and phase 2 (truncated DFT path) must match the oracle
and beat the baseline. No library code changes in this phase. The phase ends when the
oracle passes against today's code at the tolerances below, and the baseline report has
recorded timings and oracle errors.

Companion documents:
- [docs/design/spectral-conv-perf.md](../../design/spectral-conv-perf.md): §2.1 (the
  operator), §5.1, §7 (components C0.1, C0.2, C0.5; C0.3 dropped, C0.4 moved to phase 2),
  §8.1–§8.3 (tests, tolerances, benchmark protocol), §9.1 (R4);
- `docs/CONVENTIONS.md`: §1 layouts, §3 padding, §4 FFT and normalisation, §5 spectral
  layout.

## Scope

In scope:
- `src/neural_operators/layers/spectral_convolution_oracle.rs` (new, test-only child
  module) and the one `mod` line that includes it (T1).
- `examples/bench/spectral_conv.rs` (new), and its `[[example]]` entry in `Cargo.toml` (T2).
- This README and the design document's §7/§8.2, for the baseline numbers (T3).

Out of scope:
- **Non-test code in `SpectralConv`, `FNO` or `utils/fft.rs`.** Owned by phases 1–2.
- **The `irfft` DC/Nyquist tests.** These are pytorch-parity phase 0 T2
  (`docs/pytorch-parity/phase0/T2-irfft-dc-nyquist.md`), a dependency of *phase 2*,
  not of this phase.
- **Edits to `docs/CONVENTIONS.md`.** The non-normative §9 note is C2.0, in phase 2.
  pytorch-parity T4 owns the v2 bump.
- **Timings on CUDA, wgpu or ROCm**, and peak-memory measurement. Burn exposes no portable
  API for peak memory; report it as "not measured".

## Design decisions for this phase

These hold for every task, so that no task decides them on its own:

- **The oracle is a child module, not an integration test.** `weights_re`/`weights_im`
  are private (`spectral_convolution.rs:87-88`), and `corner_weights` is
  `#[cfg(test)] pub(crate)` (`:378-386`). A file in `tests/` can neither set nor read
  them. A `#[cfg(test)] #[path = "spectral_convolution_oracle.rs"] mod oracle;` child of
  `spectral_convolution` can do both. Design §5.1 was revised accordingly.
- **Deterministic data.**
  - Oracle inputs and weights come from a fixed-seed host generator inside the test,
    uploaded with `TensorData::new` (the pattern of `data_losses.rs:397`) and set with
    `Param::from_tensor` (the pattern of `spectral_convolution.rs:502-506`).
  - Do not use `Tensor::random` or `device.seed`. Flex's RNG is process-wide and shared
    with concurrently running tests (see `tests/spectral_init.rs:7-9`).
  - The benchmark may use `Tensor::random`, since its values do not matter.
- **The f64 oracle is host-only** (design §8.2 note).
  - It takes the f32 values that were uploaded, cast exactly to f64, and computes
    design §2.1 with plain Rust `f64` loops.
  - It never calls Burn FFTs, and never runs the layer in f64: flex's f64 `irfft` casts
    to f32 (fork `burn-flex/src/ops/fft.rs:1549-1562`).
- **Error measures.** Every test names the one it uses. ‖·‖∞ is the max-abs over all
  elements.
  - **E-oracle.**
    - ‖y_layer − y_oracle‖∞ / ‖y_oracle‖∞ ≤ 1e-5 for every spatial extent ≤ 128.
    - ≤ 3e-5 for the 1D cases s = 256 and s = 1024.
    - y_layer is f32 on the default backend; y_oracle is f64.
    - Design §8.2, rows 1–2.
  - **E-control.** A deliberately wrong oracle differs from the layer by
    ‖y_layer − y_wrong‖∞ / ‖y_oracle‖∞ ≥ 1e-3.
    - This proves the E-oracle comparison can fail.
    - The numerics review measured 1.8e-2 to 1.7 for these mutations. 1e-3 leaves a
      margin for other data, and is still 100× above E-oracle.
  - **Timing.**
    - 5 warm-up steps, then ≥ 20 timed steps.
    - Call `device.sync()` (fork `burn-tensor/src/device.rs:707`) before starting and
      before stopping each step's clock.
    - Report the median and the inter-quartile range (IQR) in milliseconds. Use
      nearest-rank quartiles: with 20 samples these are x(5), x(10) and x(15), so the
      "median" is the lower middle value.
- **Backend labels.** Label runs by build feature: "flex", or "Metal" for
  `--features metal`. The device debug string on macOS reads
  `Cube(Wgpu(.. backend: Auto))` and never says "Metal".
    - Design §8.3.
- **Benchmark settings** (design §1, §8.3). B = 20.
  - Layer, 1D: I = O = 64, s ∈ {256, 1024}, `modes = [16]`.
  - Layer, 2D: I = O = 32, s = 94 × 94, `modes = [12, 12]`.
  - FNO step: the `FNOConfig` literals of `examples/train/burgers.rs` (at s = 256 and
    1024) and of `examples/train/darcy.rs` (s = 85, `padding: Some(9)`).

## Known issue: Metal `cfft` (design §9.1 R8)

The phase 0 numerics review (2026-10-05) ran the T1 oracle against today's layer.
- **On flex**, every case passes with errors from 1e-7 to 7e-7.
- **On Metal**, two cases are wrong by O(1): 2D (94, 94) [12, 12] gives an E-oracle error
  of about 1.0, and 3D (8, 6, 10) [2, 3, 4] about 0.5.
- **Isolated cause:** `signal::cfft` on a non-last axis of even, non-power-of-two length
  (the Bluestein path) is wrong when the trailing extent is even. For example
  `[2,3,94,48]` is wrong and `[2,3,94,47]` is correct. The root cause is not verified.
- **Consequence:** Darcy training on Metal (94 × 94) is presumably affected today.

This bug is in the Burn fork, outside this objective. Phase 0 records it and does not fix
it:
- T1 reports the Metal values. It does not `#[ignore]` them, and CI (flex only) stays
  green.
- T3 marks the affected Metal timings "computes wrong values".

## Exit gate
- Every acceptance test in T1 passes in CI, on flex.
- T1's tests were run with `--features metal`, and the PR reports every case's value.
  Metal failures on the R8 shapes are expected and reported, not ignored. If no Apple GPU
  was available, the PR says so.
- T3 has recorded, in this README and in design §7/§8.2:
  - the baseline timings for flex and Metal;
  - the observed E-oracle error for every T1 case.
- **Sign-off needed only if** an observed E-oracle error on today's code, **on flex**,
  exceeds its tolerance (risk R4: Bluestein at s = 94). The user must then approve the
  revised §8.2 row before phase 1 starts.
- **User decision needed on R8** before phase 1: does the Metal gate for the 2D/Darcy
  settings stand as is, or does a fork fix come first?

## Tasks

One pull request each. T1 and T2 touch disjoint files and can start at once, in parallel.
The only shared file is `Cargo.toml` (T2 only), so no merge conflicts are expected.
T3 starts after both have merged.

| Task | Brief | Delivers | Component | Depends on |
| --- | --- | --- | --- | --- |
| T1 | [T1-layer-oracle.md](T1-layer-oracle.md) | f64 naive-DFT oracle tests of `SpectralConv::forward`, D = 1, 2, 3 | C0.2 | none |
| T2 | [T2-benchmark.md](T2-benchmark.md) | `bench_spectral_conv` example: layer fwd+bwd and FNO train-step timings | C0.1 | none |
| T3 | [T3-baseline-report.md](T3-baseline-report.md) | Baseline timings and oracle errors recorded; §8.2 revision proposed if needed | C0.5 | T1, T2 |

## How to run a task with Claude Code

In this objective's worktree (`feat/gpu-optimisations`), start `claude` and say
"/do-task docs/spectral-conv-perf/phase0/T<k>-<name>.md". Review and merge before
starting a task that depends on it.

## Baseline (filled in by T3)

| Case | Backend | Median ms | IQR ms |
| --- | --- | --- | --- |
| _pending T3_ | | | |

| Oracle case | Observed E-oracle | Tolerance |
| --- | --- | --- |
| _pending T3_ | | |

## Exit checklist
- [ ] T1 merged: oracle passes on flex for all cases; Metal run reported
- [ ] T2 merged: `bench_spectral_conv` runs on flex and Metal
- [ ] T3 merged: baseline and oracle errors recorded above and in design §7/§8.2
- [ ] §8.2 revision signed off, only if T3 needed one
- [ ] User decision on R8 (Metal `cfft`) recorded
- [ ] Design document updated: §7 phase 0 status and measured results; §9.1 R4 retired
      or revised; R8 status
- [ ] `CLAUDE.md` points to phase 1
