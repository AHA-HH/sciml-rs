# f64 FFTs on cuda launched as f32 kernels (garbage values, NaN downstream)

**Status:** fixed upstream by cubek #677, "fix(fft): launch the kernels at the input's
element type" (merged 2026-09-28,
<https://github.com/tracel-ai/cubek/pull/677>), in cubek 0.3.0 and later, so in Burn
0.22.0 and later. We found it on the old fork; not reported by us. Kept as the record
behind 2D Phase 10 T3 (decision G7).

## Summary

`cubek-fft` at rev `333c406` (pinned by the Burn fork `ax1s-x1zz/burn@ddfa9af`) launched
every FFT kernel as `launch::<f32>` whatever the dtype, into buffers sized for f64, with
f32 twiddles. Burn's `burn-cubecl` wrapper passed the dtype correctly. Every f64 FFT
(`rfft`, `cfft`, `irfft`) on cuda returned values such as 1e275 and 1e-69, and stale
memory; basic f64 ops were correct.

## How it showed

`tests/gauss_newton_f64.rs`, `f64_oracle_and_adjoint_on_cuda` (Phase 10 T3, VM,
2026-11-11, commit `9dbeb91`): NaN for the vjp, the finite-difference jvp and the adjoint
error alike (P = 591). A throwaway probe on the VM's second GPU isolated the FFTs.

## After the fix

On Burn 0.23.0-pre.2 the same test gets past the FFTs and fails in the matmul instead:
[../f64-matmul-declined-on-cuda/](../f64-matmul-declined-on-cuda/README.md).
