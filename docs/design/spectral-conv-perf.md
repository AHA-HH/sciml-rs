# Spectral convolution performance

As of 2026-10-05.
2026-10-05: numerics review applied. §2.7 no longer edits CONVENTIONS §1–§6, and the bases are built on the host in f64. Revised §2.1–§2.5, §5.2–§5.3, §6, §7, §8.2 and §9.

> Where this document and `docs/CONVENTIONS.md` differ, the conventions file takes
> precedence.

Recommendation in one line: first restructure the existing FFT path so that each layer runs one
stacked, real-packed GEMM with no zero-fill or scatter. Then add a truncated DFT, computed as
dense matmuls, as a second transform path selected by a flag. Both paths compute today's
operator, and the parameter layout is unchanged.

## 1. Purpose, scope and assumptions

**Goal.** Make `SpectralConv::forward` and its backward cheaper in time and memory traffic,
mainly on GPUs, without changing the operator it computes or the checkpoints it reads.

**Definition of done.**
1. Every new code path matches the current FFT path to f32 tolerance on the default `flex`
   backend (§8.2), in forward values and in gradients. All existing reference tests
   (`forward_2d_matches_pytorch_reference`, `forward_3d_matches_numpy_reference`,
   `forward_at_mode_limit_is_identity_*`) pass unchanged.
2. On Metal, one training step (forward + backward + optimiser) of the FNO is
   measurably faster than the phase 0 baseline at both benchmark settings (§8.3):
   - Burgers 1D: `s = 256` as in `examples/train/burgers.rs`, plus `s = 1024`; `modes = [16]`, H = 64.
   - Darcy 2D: `s = 85`, `padding = Some(9)`, so the spectral layers run at 94 × 94;
     `modes = [12, 12]`, H = 32.
   "Measurably" means the median over ≥ 20 timed steps is lower, and the inter-quartile
   ranges do not overlap. Flex timings are reported as well; they are not a gate.
3. Saved `.bpk` checkpoints and saved `FNOConfig` JSON from before this objective load
   and predict as before (§5.4).

**In scope.** The six items from the performance review of
`layers/spectral_convolution.rs`. Its line numbers are stale; current locations are given here:
- (a) The full FFT is replaced by a truncated DFT as matmul. Today the full spectrum is
  transformed (`:294-304`) and the inverse runs over zeros (`:307-315`).
- (b) The zero-fill + `slice_assign` (`:345-346`, `:371-372`) is replaced by `cat`/`pad`.
- (c) The `2^(D−1)` per-corner matmuls are stacked into one (`:348-373`).
- (d) The three Gauss matmuls and five elementwise ops become one packed real GEMM (`:273-279`).
- (e) The weight reshape and permute on every forward (`:270-271`) is reduced. It is
  not removed, because parameters keep their layout (§5.4).
- (f) The input permute `[B, I, M] → [M, B, I]` (`:268-269`), which today runs once per
  corner on that corner's slice, becomes one permute of the stacked tensor.

Also in scope: a benchmark harness and an f64 oracle test of the layer.

**Out of scope.**
- **Dropping the Burn fork.** The FFT path stays (for large m/n, §4). `Tensor<R>` with no
  backend generic and `burn-signal` are fork APIs used throughout the crate.
- **Changing the parameter layout** (`weights_re/im[corner]`, each `[I, O, modes..]`,
  CONVENTIONS §5), and loading PyTorch weights.
- **Optimising the rest of the model.** That covers the pointwise layers, `FNO::forward`'s permutes
  and the data pipeline.
- **The features pytorch-parity adds** to `SpectralConv`: bias, separable, factorised
  weights (§9.1 R5).
- **Mixed precision and f16/bf16.**
- **CUDA, wgpu and ROCm timings.** These backends are tested if available; they are not gated.

**Assumptions.**
- Typical FNO settings have small m/n: m ≤ 32 and n ≤ 1024 per axis.
- f32 matmul accumulation on Metal and flex is at least as accurate as the FFT kernels at
  these sizes (checked in phase 0, §8.2).
- Burn's autodiff of `matmul`, `cat`, `pad`, `slice` and `permute` is correct. These ops
  are core and already in use, and the gradient tests in §8.1 check them again.

## 2. Foundations

Notation as in CONVENTIONS: B batch, D spatial dimension, R = D + 2, s_a extent of spatial
axis a (a = 1..D), I/O in/out channels. m_a = `modes[a]`.

### 2.1 The operator today (unchanged by this objective)

The layer maps channels-first `x ∈ ℝ^{B×I×s_1×…×s_D}` to `y ∈ ℝ^{B×O×s_1×…×s_D}`
(CONVENTIONS §1). With the unnormalised forward DFT on each axis (CONVENTIONS §4):

```math
\hat x[b,i,k] = \sum_{x} x[b,i,x]\, \prod_{a=1}^{D} e^{-2\pi \mathrm{i}\, k_a x_a / s_a}
```

The retained set is 𝒦 = 𝒦_1 × … × 𝒦_D (CONVENTIONS §5), where
- 𝒦_a = {0..m_a} ∪ {s_a − m_a..s_a} for a < D, and
- 𝒦_D = {0..m_D} (the rfft half-spectrum).

Write k ∈ 𝒦 for each retained frequency and w_k for the complex weight
assembled from the corner that contains k. Then

```math
\hat y[b,o,k] = \begin{cases} \sum_i \hat x[b,i,k]\, w_k[i,o] & k \in \mathcal K \\ 0 & \text{otherwise}\end{cases}
\qquad
y = \mathrm{irfftn}(\hat y)
```

Here `irfftn` is defined as follows:
- First, the inverse complex DFT (scaled by 1/s_a) along axes D−1 down to 1. This is
  `icfft_full_spectrum`, `utils/fft.rs:8-25`.
- Then `irfft` along axis D with output length s_D:

```math
y[\dots, x_D] = \frac{1}{s_D}\Big[\mathrm{Re}\,Y_0 + \sum_{0<k<s_D/2} 2\big(\mathrm{Re}\,Y_k \cos\theta_k - \mathrm{Im}\,Y_k \sin\theta_k\big) + [s_D \text{ even}]\,\mathrm{Re}\,Y_{s_D/2}\,(-1)^{x_D}\Big],
\quad \theta_k = 2\pi k x_D / s_D
```

`irfft` ignores Im Y_0 and, for even s_D, Im Y_{s_D/2}. In the fork's Bluestein branch
this is visible directly: `burn-signal/src/functions/fft.rs:147-156` keeps only the real
part of the Hermitian-extended inverse. For the native power-of-two kernels, CONVENTIONS §4
lists it as not yet pinned by a test. This matters in every forward, including 1D. Im Y_0 = Σ_i X_0·Im W_0 is nonzero because
the weights are complex, and the same holds for Im Y_{s_D/2} whenever Nyquist is retained.
In 2D and 3D, the inverse along the non-last axes adds a further contribution. The flex
native kernel uses only `x_re[0]` and `x_re[half]` (`burn-flex/src/ops/fft.rs:1277-1278`),
which is consistent with this (numerics review, 2026-10-05).

### 2.2 Truncated DFT bases (item a)

For an axis of length n and an ordered list of retained frequencies κ = (κ_1..κ_q), define
the phase **in integer arithmetic first** as

```math
\phi[x, j] = \frac{2\pi}{n}\,\big((\kappa_j \cdot x) \bmod n\big), \quad x \in 0..n
```

so the argument of cos/sin stays in [0, 2π) whatever the size of κ_j·x. Then:

| Basis | Shape | Entry | Use |
|---|---|---|---|
| `F_re`, `F_im` | `[n, q]` | `cos φ`, `−sin φ` | forward along any axis |
| `G_re`, `G_im` | `[q, n]` | `cos φᵀ / n`, `sin φᵀ / n` | inverse along a non-last axis |
| `H_re`, `H_im` | `[q, n]` | `c_j cos φᵀ / n`, `c_j sin φᵀ / n` | inverse along the last (rfft) axis |

On the last axis, c_j = 1 if κ_j = 0, or if κ_j = n/2 with n even; otherwise c_j = 2. The
retained frequency lists are:
- non-last axis a: κ = (0, 1, …, m_a − 1, s_a − m_a, …, s_a − 1), so q = 2m_a,
  low block then high block;
- last axis: κ = (0, …, m_D − 1), so q = m_D.

Per axis:
- forward: `X = x @ F` (real input; complex input expands to four real matmuls, or one
  packed matmul as in §2.4);
- inverse, non-last axis: `x = X @ G`, complex;
- inverse, last axis: `y = X_re @ H_re − X_im @ H_im`, real output.

The H formula is exactly the `irfft` formula of §2.1 restricted to the κ_j < m_D that
can be nonzero. Because sin φ = 0 at κ = 0 and at κ = n/2, the DFT path ignores Im Y_0
and Im Y_{n/2} as `irfft` does, in exact arithmetic. In f32, sin(fl(π)) ≈ −8.7e-8, so the
builders set the `F_im`, `G_im` and `H_im` columns at κ ∈ {0, n/2} to exact 0, which makes
this hold in floating point too. Since ŷ is zero outside 𝒦, the truncated
inverse equals the full inverse **exactly in exact arithmetic**. The difference between the
two paths is f32 rounding only.

### 2.3 The corner-stacked spectral grid (items b, c)

Concatenate the low and high blocks on each non-last axis in κ order. This turns the
`2^(D−1)` corners into one dense spectral grid of extent `(2m_1, …, 2m_{D−1}, m_D)`,
with M = 2^(D−1) Π m_a entries. The corner with mask bits β (CONVENTIONS §5,
`corner_ranges` `:216-227`) occupies, on axis a < D, rows `0..m_a` if β_a = 0 and
`m_a..2m_a` if β_a = 1.

The stacked weight `W ∈ ℂ^{I×O×2m_1×…×m_D}` is assembled from the stored corners. On
each non-last axis, from axis 1 up, `cat` the (β_a = 0) half with the (β_a = 1) half.
Parameters themselves are not changed (§5.4).

For the **FFT path**:
- *Gather*: on each non-last axis a, build the stacked input from the full spectrum as
  `cat([slice 0..m_a, slice s_a−m_a..s_a], a)`, and on the last axis as
  `slice 0..m_D`. That is 2(D−1) + 1 slices, against 2^(D−1)·D today.
- *Scatter*: embed the stacked output back with, per non-last axis,
  `cat([low, zeros(s_a − 2m_a), high], a)`, and on the last axis `pad` with zeros up to
  s_D/2 + 1. This replaces `zeros` + `slice_assign`. When s_a = 2m_a the zero block is
  omitted, not allocated empty; existing tests hit this case (`gradients_flow_2d`,
  `forward_at_mode_limit_is_identity_2d_odd_last_axis`).
- Axis numbering: spatial axis a (1-based) is tensor dim a + 1; in code, 0-based spatial
  index j is dim 2 + j.

For the **DFT path**, the forward bases emit the stacked grid directly, and the inverse
bases consume it. There is no gather or scatter at all.

### 2.4 Packed real GEMM (item d)

Flatten the stacked grid to M and lay out modes first. The complex product is one real
batched matmul:

```math
\underbrace{[\,X_{re} \mid X_{im}\,]}_{[M,\,B,\,2I]}
\;@\;
\underbrace{\begin{bmatrix} W_{re} & W_{im} \\ -W_{im} & W_{re}\end{bmatrix}}_{[M,\,2I,\,2O]}
=
\underbrace{[\,Y_{re} \mid Y_{im}\,]}_{[M,\,B,\,2O]}
```

This is exact: block multiplication reproduces (ac − bd) + (ad + bc)i term by term. Its
cost is 4·B·I·O·M multiply-adds, against 3·B·I·O·M for today's Gauss trick (`:273-279`).
So (d) trades +33 % FLOPs for one kernel launch instead of eight ops, and for a GEMM with 2× K
and 2× N. This is a bet on launch overhead and occupancy, not on arithmetic, and it is
benchmarked (phase 1 exit gate). If it loses on flex but wins on Metal, the Gauss form is
kept behind the same internal switch; see §9.2 Q2.

`W_pack` is assembled each forward from the parameters:
- `cat` the four blocks;
- reshape `[I, O, M] → [M, I, O]` with a permute (item e);
- this is O(I·O·M) work, small next to the O(B·I·O·M) GEMM when B ≥ 4.

### 2.5 Contraction order (item f)

Today the permute to modes-first already acts on truncated corner slices, once per corner.
After stacking it is one permute of `[B, I, M]` on both paths. This saves launches, not
bytes. Emitting `[M, B, I]` with no copy at all
would need the bases applied as left-multiplications `Fᵀ @ x` on a transposed view. That
is a matmul-kernel layout detail. It is left to phase 2 to measure, and it is not required.

### 2.6 Cost model

These are real multiply-adds per layer forward, for orientation only. Phase 0 measures
the real numbers. Let N = Π s_a.

| Stage | FFT path today | DFT path |
|---|---|---|
| forward transform | ~c·B·I·N·log₂N; Bluestein (non-power-of-two s) pads to ≥ 2s−1 and does 3 FFTs | 2·B·I·N·m_D, then 4·B·I·(N/s_D)·m_D·2m_a per further axis (shrinking) |
| mode mixing | 3·B·I·O·M (Gauss) → 4·B·I·O·M packed | 4·B·I·O·M packed |
| inverse transform | full length on every axis, over mostly zeros | mirror of forward with O instead of I |
| kernel launches | ~2D FFT calls, 2^(D−1)·(2 slices + 3 matmuls + 5 elementwise + 2 slice_assign) | ~2D matmuls + basis build + 1 GEMM |

At m ≪ s the DFT path does more FLOPs than an ideal FFT. It wins on GPUs by being
compute-bound GEMM rather than memory-bound strided FFT passes. At large m/n the FFT wins,
so both paths are kept (§4).

### 2.7 Conventions

**No edit to CONVENTIONS §1–§6, and no `CONVENTION_VERSION` bump.** The operator,
parameter shapes, corner order and draw order are unchanged. CONVENTIONS' preamble says
any change to §1–§6 bumps the version, with no exception for informative notes, and
pytorch-parity plans its own v2 bump (§9.1 R5). So this objective keeps out of §1–§6
altogether:
- **§4 DC/Nyquist.** C0.3 adds the tests only. Updating the §4 "not yet fixed by a test"
  bullet is left to pytorch-parity's v2 change (its T4 cites the same test names). If
  this objective lands first, the bullet stays as it is until then.
- **Transform choice.** It goes in a new, non-normative section after §8, which the
  preamble's bump rule does not cover:

```diff
+## 9. Implementation notes (non-normative)
+
+Nothing here affects saved checkpoints or the operator; changes need a PR note only.
+
+- `SpectralConv` may compute its transforms with FFTs or with truncated DFT matmuls
+  (`SpectralTransform`, design spectral-conv-perf §2.2). Both compute the operator of §4
+  and §5 and differ only by f32 rounding. The choice is not stored in module records
+  (`.bpk`); it is stored in `FNOConfig` as an `Option`, where `None` means `Fft`.
```

This is still an edit to the conventions file, so it needs the user's sign-off (§9.3).

## 3. State of the art

1. **Full FFT + truncation + zero-filled inverse.** This is what sciml-rs does today, and
   it is also Li et al.'s reference (`torch.fft.rfftn`, `out_ft = zeros`, corner assign,
   `irfftn`) and neuraloperator's `SpectralConv`.
   - Cost: O(N log N) per channel per direction. Memory-bound on GPU.
   - Works for any m.
2. **Truncated DFT as matmul** ("pruned DFT" / partial DFT).
   - Cost: O(N·m) per axis. Maps to tensor-core GEMMs, so it is compute-bound.
   - Exact up to rounding. It needs basis tensors of size n·q per axis.
   - Used in several GPU FNO reimplementations because cuFFT plus scatter dominates at
     small m (cited from memory, unverified; no specific source pinned).
3. **Pruned FFT algorithms**, which compute only the wanted output bins (Markel 1971;
   Sorensen & Burrus 1993, unverified). They are O(N log m), but need custom kernels that
   Burn does not have.
4. **Mixed-radix or plan-cached FFT** (cuFFT/VkFFT-style). This is a backend change in
   the Burn fork (`burn-signal`), outside this crate.
5. **Complex-mixing kernels.**
   - Gauss three-multiply (today): 3 GEMMs.
   - Packed real block GEMM: 1 GEMM, 4/3 of the FLOPs.
   - Native complex GEMM: Burn has no complex dtype in this fork.

## 4. Comparison and recommended strategy

Criteria from §1: same operator to f32 tolerance; Metal step time; works for any s
(odd, non-power-of-two); checkpoint-compatible; stays within this crate; size of change.

| Option | Exact operator | Metal speed (expected) | Any s | Checkpoints | In crate | Size |
|---|---|---|---|---|---|---|
| 1. Today | ✓ | baseline | ✓ (Bluestein) | ✓ | ✓ | – |
| 1 + (b, c, d, e) restructure | ✓ | modest gain: fewer launches, no scatter | ✓ | ✓ | ✓ | S–M |
| 2. Truncated DFT (a, f) + restructure | ✓ (rounding) | largest at small m/n | ✓ | ✓ | ✓ | M |
| 3. Pruned FFT | ✓ | unknown | ✓ | ✓ | ✗ custom kernels | L |
| 4. Better FFT backend | ✓ | helps FFT only | ✓ | ✓ | ✗ fork work | L |
| (e) stored as `[M, 2I, 2O]` | ✓ | saves an O(I·O·M) copy | ✓ | ✗ convention bump | ✓ | M |

**Choice.** Do option 1's restructure first (phase 1). It is safe, it carries over to
the DFT path (stacked grid and packed GEMM are shared), and it isolates what (b)–(e) are
worth on their own. Then add option 2 behind a `SpectralTransform` flag (phase 2) and wire
it into `FNOConfig` with a measured `Auto` heuristic (phase 3). The FFT path remains for
large m/n and as the trusted reference.

The user decided to keep the parameter layout (2026-10-05). Storing the weights as `[M, 2I, 2O]`
is rejected for now, and §9.2 Q3 revisits it only if phase 1 shows the per-forward
assembly matters.

## 5. Architecture

### 5.1 Modules

| File | Change |
|---|---|
| `src/neural_operators/utils/dft.rs` (new) | basis builders (§2.2); `utils/mod.rs` gains `pub mod dft` |
| `src/neural_operators/layers/spectral_convolution.rs` | `SpectralTransform` enum; stacked gather/scatter, packed mixing, DFT forward/inverse; `forward` dispatches. `complex_multiplication` is kept as the test oracle until phase 3 removes or retains it (§9.2 Q2) |
| `src/neural_operators/models/fno.rs` | `FNOConfig::spectral_transform: Option<SpectralTransform>`, passed to each `SpectralConv` (phase 3) |
| `examples/bench/spectral_conv.rs` (new) + `[[example]] bench_spectral_conv` in `Cargo.toml` | layer and training-step timings (phase 0) |
| `tests/spectral_conv_oracle.rs` (new) | f64 naive-DFT oracle of the layer (phase 0) |
| `docs/CONVENTIONS.md` | §4 test citation, §5 note (§2.7) |

Nothing else changes: loaders, training, metrics, losses.

### 5.2 Core interfaces

```rust
// utils/dft.rs: all constant tensors (no autodiff), computed on the host in f64,
// converted to `dtype` and uploaded to `device` (§5.3).

/// Retained frequency list for axis `a` (§2.2): `0..m` then `n-m..n` on a non-last
/// axis (`len 2m`), `0..m` on the last axis (`len m`).
pub fn retained_frequencies(n: usize, m: usize, last_axis: bool) -> Vec<usize>;

/// Forward basis `(F_re, F_im)`, each `[n, q]`, q = `freqs.len()`.
/// Phase computed as `(k·x) mod n` in integers before scaling (§2.2).
pub fn forward_basis(n: usize, freqs: &[usize], dtype: DType, device: &Device)
    -> (Tensor<2>, Tensor<2>);

/// Inverse basis `(G_re, G_im)`, each `[q, n]`, including 1/n; with
/// `hermitian = true`, the rfft-axis weights c_j (§2.2, `H_*`).
/// # Panics: if `hermitian` and some `freqs[j] > n/2`.
pub fn inverse_basis(n: usize, freqs: &[usize], hermitian: bool, dtype: DType, device: &Device)
    -> (Tensor<2>, Tensor<2>);
```

```rust
// layers/spectral_convolution.rs

/// How `SpectralConv` computes its transforms. Not persisted (`#[module(skip)]`).
// `Config` on an enum already derives `Clone`; `Default` is manual, as for `SpectralInit`
// (`spectral_convolution.rs:52-59`).
#[derive(Config, Debug, PartialEq, Copy)]
pub enum SpectralTransform { Fft, Dft, Auto }   // impl Default -> Fft

impl<const R: usize> SpectralConv<R> {
    /// Builder; `new`/`new_with_init` keep `Fft`.
    pub fn with_transform(self, t: SpectralTransform) -> Self;

    /// Stacked complex weight (§2.3) packed as `[M, 2I, 2O]` (§2.4).
    fn packed_weight(&self) -> Tensor<3>;
    /// `[M, B, 2I] @ [M, 2I, 2O] -> [M, B, 2O]`.
    fn mix(x_pack: Tensor<3>, w_pack: Tensor<3>) -> Tensor<3>;
    /// FFT path: full spectrum `[B, I, s_1.., s_D/2+1]` (re, im) -> stacked `[B, I, 2m_1.., m_D]`.
    fn gather_stacked(re: Tensor<R>, im: Tensor<R>, modes: &[usize]) -> (Tensor<R>, Tensor<R>);
    /// FFT path: stacked `[B, O, 2m_1.., m_D]` -> full spectrum `[B, O, s_1.., s_D/2+1]`.
    fn scatter_stacked(re: Tensor<R>, im: Tensor<R>, spec_dims: &[usize]) -> (Tensor<R>, Tensor<R>);
    /// DFT path: `[B, I, s..]` real -> stacked `[B, I, 2m_1.., m_D]` (re, im).
    fn dft_forward(&self, x: Tensor<R>) -> (Tensor<R>, Tensor<R>);
    /// DFT path: stacked `[B, O, 2m_1.., m_D]` -> `[B, O, s..]` real.
    fn dft_inverse(&self, re: Tensor<R>, im: Tensor<R>, spatial: &[usize]) -> Tensor<R>;
}
```

The public `forward(&self, x: [B, I, s..]) -> [B, O, s..]` keeps its signature, panics
(`check_modes_fit`) and doc comment, and gains a note on `SpectralTransform`.

Per-axis DFT application on a rank-R tensor:
1. Move the axis last with `swap_dims` (`burn-tensor/src/tensor/api/base.rs:500`).
2. `matmul` with the `[n, q]` basis reshaped to rank R as `[1,…,1,n,q]`. The fork's
   `matmul` takes two tensors of the same rank and broadcasts batch dims of size 1
   (`numeric.rs:825`, `check.rs:524-530`).
3. Swap back.

The last axis needs no swap.

### 5.3 Data layout and flow

**FFT path after phase 1:**
1. `x [B,I,s..]` → `fft_ctensor` (unchanged).
2. `gather_stacked` → `[B,I,2m..,m_D]`.
3. Reshape `[B,I,M]` → permute `[M,B,I]`, `cat` re|im → `[M,B,2I]`.
4. `mix` with `packed_weight()` → `[M,B,2O]`.
5. Split and permute back to `[B,O,2m..,m_D]`.
6. `scatter_stacked` → `ifft_ctensor` (unchanged) → `[B,O,s..]`.

**DFT path (phase 2):**
1. `x` → last-axis `F` (real → complex), then the non-last axes from D−1 down to 1 (complex).
   This is the same axis order as `fft_ctensor` (CONVENTIONS §4); the order does not
   change the exact result, but it keeps rounding comparable.
2. Stacked `[B,I,2m..,m_D]` → permute on the small tensor (§2.5) → `mix`.
3. Inverse non-last axes 1..D−1 with `G`, then the last axis with `H` → `[B,O,s..]`.

**Bases.** Built on the host in f64 for each forward, then converted to the input's
dtype (`TensorData::convert_dtype`) and uploaded to its device:
- the phase uses integer `(k·x) mod n`;
- columns at κ ∈ {0, n/2} of the sine bases are set to exact 0 (§2.2).

This follows the fork's own Bluestein precedent, which computes `chirp_phase` and
`broadcast_const` on the host in f64 (`burn-signal/src/functions/fft.rs:187-206`). Entries
are within 0.5 ulp whatever the backend's `cos`/`sin` accuracy, so the basis tests do not
depend on the backend. The cost is O(Σ_a n_a·q_a) host work plus one upload per axis per
forward. The bases are rebuilt whenever s changes, which keeps the discretisation
invariance of CONVENTIONS §5. Caching is deferred (§9.2 Q4). On-device generation, as in
`FNO::grid_cl` (`fno.rs:197`), was rejected: f32 `cos`/`sin` gives only ~3.5e-7 accuracy at
best, and Metal may compile them with fast-math (numerics review, 2026-10-05).

### 5.4 Compatibility

- **Parameters**: `weights_re`, `weights_im` keep their names, count and shape
  `[I, O, modes..]`. Draw order is unchanged, so `tests/spectral_init.rs` bit-parity holds.
- **`SpectralConv` fields**: `transform` is `#[module(skip)]` (see the attribute docs at
  `burn-derive/src/lib.rs:34`). It adds no record key, so old `.bpk` files load.
- **Config**: `FNOConfig::spectral_transform: Option<SpectralTransform>`. `None` means `Fft`,
  so configs saved before this field load. A test mirrors
  `config_without_spectral_init_loads_as_default` (`fno.rs:443`).
- **Public API**: additive only (`SpectralTransform`, `with_transform`, `utils::dft`).
- **Examples**: unchanged until phase 3, which may set `spectral_transform` in the
  training examples after the benchmark.

## 6. Platform and dependency considerations

These were read in the pinned fork,
`~/.cargo/git/checkouts/burn-4aa1ec9707f89bee/ddfa9af/crates/`:
- `burn-signal/src/functions/fft.rs`:
  - `rfft` at `:53`, `irfft` at `:118`, `cfft` at `:354`;
  - non-power-of-two `irfft` rebuilds the Hermitian spectrum (`hermitian_extend`,
    `:389-420`) and runs Bluestein (`:147-156`), keeping only the real part;
  - power-of-two sizes dispatch to the backend kernel.
  - Autodiff for these lives in `burn-signal/src/backends/autodiff.rs` (`rfft` `:13`,
    `irfft` `:88`).
- `burn-tensor/src/tensor/api/`:
  - `numeric.rs:825` `matmul`, with batch handling for D ≥ 3;
  - `base.rs:2573` `cat`, `base.rs:500` `swap_dims`;
  - `pad.rs:129` `pad` (`PadMode::Constant`);
  - `TensorData::convert_dtype` and `Tensor::from_data` for the host-built bases (§5.3).
- `burn-core/src/module/param/constant.rs:141`: `Ignored<T>` is deprecated in favour of
  `#[module(skip)]`.

**Gaps.**
- No inverse complex FFT; `icfft_full_spectrum` fills it, unchanged.
- No complex dtype, hence the packed real GEMM.
- `matmul` is same-rank only, so the bases are reshaped to rank R with leading 1s (§5.2).

**Backends.**
- Flex is the correctness gate (CLAUDE.md).
- Metal is the performance gate. Note the known Metal quirk: a channels-first `cat` after
  a permute zeroed grid entries (CLAUDE.md, Burn section). The stacked gather and scatter
  use `cat` after FFTs, and the packed GEMM uses `cat` after a permute, so every new path
  runs with `--features metal` in tests as well as benchmarks. If the quirk reappears,
  a test catches it, and the workaround is to `cat` before the permute.

**Precision.** f32 throughout:
- Bases are computed in f64 on the host and rounded once to f32 (§5.3). Phase arguments
  are kept in [0, 2π) by integer reduction (§2.2).
- The DFT sums n terms. Rounding grows ~√n·u for random data, u = 6e-8, which is far
  inside the §8.2 tolerances for n ≤ 1024.
- Bluestein (s = 94) has its own error, larger than radix-2's. Phase 0 measures it
  against the f64 oracle, so tolerances are set from evidence.

**Performance.**
- Per-forward basis construction is O(Σ_a n_a·q_a) host work plus one upload per basis.
- Weight packing is O(I·O·M).
- Phase 1 and phase 2 benchmarks report both, so their cost is visible.

## 7. Phased implementation plan

Every phase leaves `main` green (CLAUDE.md "Checks"). Each component is one PR unless
noted.

### Phase 0: oracles and baseline

Goal: trusted references and numbers to beat, before any change to the layer.
- **C0.1** Benchmark example `bench_spectral_conv`:
  - times `SpectralConv` forward+backward and a full FNO training step at the §1
    settings (B = 20) with warm-up;
  - reports median and IQR over ≥ 20 steps;
  - runs on flex and `--features metal`.
  - Baseline numbers are recorded in the phase 0 README.
- **C0.2** `tests/spectral_conv_oracle.rs`: an f64 host naive DFT implementation of §2.1
  (no Burn FFTs), compared with `SpectralConv::forward` for:
  - D = 1, 2, 3;
  - s even, odd, and non-power-of-two (incl. 94);
  - modes at and below the limits, including a retained Nyquist (m_D = s_D/2 + 1, s_D even);
  - I ≠ O, B > 1, random weights.
- **C0.3** The `irfft` DC/Nyquist tests (CONVENTIONS §4), using pytorch-parity T2's test
  names and content verbatim. They cover a power-of-two length (native kernel), an even
  non-power-of-two length and an odd length (Bluestein). Tests only: the §4 bullet is
  updated by pytorch-parity's v2 change (§2.7, §9.1 R5).
- **C0.4** New non-normative CONVENTIONS §9 "Implementation notes" (§2.7). Docs only;
  needs sign-off.

Dependencies: none. **Exit gate:**
- the C0.2 oracle passes on flex at §8.2 tolerances against **today's** code;
- C0.3 passes on flex (and on Metal if available);
- baseline timings are recorded for flex, and for Metal if available.

### Phase 1: restructured FFT path (b, c, d, e)

Goal: one stacked, packed GEMM; no zeros or `slice_assign`; parameters unchanged.
- **C1.1** `gather_stacked` / `scatter_stacked` (§2.3), replacing the corner loop's
  slicing and the zero-fill.
- **C1.2** `packed_weight` + `mix` (§2.4). Keep `complex_multiplication` as the in-test
  oracle for `mix`.

Dependencies: phase 0. **Exit gate:**
- every existing `spectral_convolution.rs` test and the C0.2 oracle pass unchanged;
- new gradient-agreement tests against the pre-phase-1 implementation (copied into the
  test module as `forward_reference`) pass at §8.2;
- benchmarks show no regression on flex and Metal (median within IQR or better);
- a Metal speedup is reported, not required.

### Phase 2: truncated DFT path (a, f)

Goal: the DFT path, selectable per layer.
- **C2.1** `utils/dft.rs` bases (§5.2), each tested against a direct f64 `cos`/`sin` table,
  and `F`/`H` round-trip tested at the mode limit.
- **C2.2** `SpectralTransform`, `with_transform`, `dft_forward`, `dft_inverse`, and
  `forward` dispatch. The DFT path shares `packed_weight`/`mix` with phase 1.

Dependencies: phase 1 (shared stacked layout). **Exit gate:**
- `Dft` matches `Fft` in forward and in input/weight gradients at §8.2 tolerances, for
  every case of C0.2;
- the C0.2 oracle passes with `Dft`;
- `forward_*_matches_*_reference` pass with `Dft`;
- tests run on flex and with `--features metal`;
- the Metal benchmark shows `Dft` faster than phase 1's `Fft` at both §1 settings.

### Phase 3: integration

Goal: users can choose the path; the default is decided on evidence.
- **C3.1** `FNOConfig::spectral_transform: Option<SpectralTransform>` (§5.4), passed to
  every layer, with an old-config deserialisation test.
- **C3.2** `Auto` heuristic: a rule over (s_a, m_a) fitted to a phase 2 sweep, recorded
  in this document (revision) and CONVENTIONS §5's note.
- **C3.3** Training-step benchmark on Metal for both §1 settings; README/docs update.
  Whether examples default to `Auto` is decided by §9.2 Q1.

Dependencies: phase 2. **Exit gate:** §1 definition of done items 1–3.

## 8. Validation and benchmarking

### 8.1 Test layers
1. **Unit:**
   - bases against f64 tables;
   - gather→scatter round trip is exact (it only moves values);
   - `mix` against `complex_multiplication` and the existing einsum values
     (`complex_multiplication_matches_python_einsum`, `:430`).
2. **Layer oracle (C0.2):** f64 naive DFT, independent of Burn's FFTs.
3. **Path agreement:** `Dft` against `Fft`, and the restructured `Fft` against the
   pre-phase-1 `forward_reference`, in values and in gradients (with respect to input and
   every `weights_re/im[c]`).
4. **Existing pinned references:** the PyTorch 2D and NumPy 3D values, and the identity at
   the mode limit. These must pass unchanged on every path.
5. **Model:**
   - `FNO` forward and gradient on both paths;
   - checkpoint save on `Fft`, load into a `Dft` model, and compare outputs (`fno.rs:464-498` pattern);
   - old-config deserialisation.

Full-path test filters are used for single tests (CLAUDE.md Gotchas).

### 8.2 Error metrics and tolerances

All metrics are max-abs errors, scaled by the reference's max-abs value
(`‖a − b‖_∞ / ‖b‖_∞`), unless stated.

| Comparison | Precision | Tolerance | Relative to |
|---|---|---|---|
| f32 layer vs f64 oracle, s ≤ 128 | f32 vs f64 | 1e-5 | ‖y_oracle‖_∞ |
| f32 layer vs f64 oracle, s = 256, 1024 (1D) | f32 vs f64 | 3e-5 | ‖y_oracle‖_∞ |
| restructured `Fft` vs `forward_reference`, values | f32 | 1e-5 | ‖y_ref‖_∞ (transforms identical; different complex-product formula, Gauss vs block; simulated 4.3e-7) |
| restructured `Fft` vs `forward_reference`, gradients | f32 | 1e-5 | g_max (see note) |
| `Dft` vs `Fft`, values | f32 | 1e-4 | ‖y_fft‖_∞ |
| `Dft` vs `Fft`, gradients | f32 | 1e-4 | g_max (see note) |
| `F` vs f64 cos/sin table | f32 | 1e-6 absolute | entries in [−1, 1] |
| `n·G`, `n·H/c_j` vs the same table | f32 | 1e-6 absolute | entries in [−1, 1]; c_j checked separately as exact values (1 or 2) |
| `mix` vs `complex_multiplication` | f32 | 1e-5 | ‖out‖_∞ |
| existing pinned tests | f32 | 1e-4 absolute (as written) | as today |

Note on g_max. Comparing gradients per parameter tensor fails when a tensor's true
gradient is 0. The operator discards Im Y_0, so with m_D = 1 `weights_im[0]` has zero
gradient in 1D, and so does corner 0's in 2D with `modes = [1, 1]`; both paths then return
rounding noise. Gradients are therefore compared elementwise, scaled by
g_max = max(‖∇x‖_∞, max over all `weights_re/im[c]` of ‖∇w‖_∞) of the reference path.

The C0.2 oracle runs on the host in f64 only. It takes the layer's own f32 inputs and
weights, cast exactly to f64, and never runs the layer itself in f64: flex's f64 `irfft`
casts to f32 (`burn-flex/src/ops/fft.rs:1549-1562`).

Phase 0 records the observed oracle errors of today's code. If Bluestein at s = 94 exceeds
1e-5, the row is revised in this document, with the measurement, before phase 1 begins.

### 8.3 Reference results and performance measurement

- **References.** The pinned PyTorch/NumPy values in `spectral_convolution.rs:497-588`, and
  the C0.2 oracle.
- **Benchmark.** `cargo run --release --example bench_spectral_conv [--features metal]`.
  - Settings:
    - layer only: B = 20; 1D (I = O = 64, s ∈ {256, 1024}, m = 16) and
      2D (I = O = 32, 94², m = (12, 12));
    - full FNO training step: the `train_burgers` and `train_darcy` configs on random
      data (no datasets needed).
  - Measurement: 5 warm-up steps, then ≥ 20 timed steps, with device sync before reading
    the clock. Report median, IQR and peak memory where the backend exposes it.
  - Results are recorded in the phase README of the phase that produced them.
- **Sweep (phase 3).** s ∈ {64, 94, 128, 256, 512, 1024}, m ∈ {4, 8, 16, 32}, 1D and 2D,
  to fit `Auto`.

## 9. Risks, open questions and working with Claude Code

### 9.1 Risks

| # | Risk | Retired by |
|---|---|---|
| R1 | Packed GEMM's extra 33 % FLOPs outweigh its fewer launches on some backend | Phase 1 benchmark; fall back to Gauss behind the internal switch (Q2) |
| R2 | Per-forward basis construction or weight packing costs more than it saves at small B | Phase 2 benchmark reports it separately; Q4 caching |
| R3 | Metal `cat`-after-permute quirk returns in gather/scatter/packing | Metal test runs in phases 1–2 |
| R4 | Bluestein's f32 error at s = 94 makes 1e-5 oracle tolerance fail on today's code | Phase 0 measurement; §8.2 revised from evidence |
| R5 | **Overlap with pytorch-parity** (goldeye worktree): its phase 0 T2 adds the same `irfft` DC/Nyquist tests, and later phases add bias, separable and factorised weights (with empty corner vectors) to `spectral_convolution.rs`, plus CONVENTIONS v2 §5 lines | C0.3 copies T2's test names and content verbatim, so whichever lands second drops its duplicate. The stacked layout of §2.3 is built from the corner vectors, so factorised weights need a reconstruction step before `packed_weight`. Recorded here; the second objective to land rebases. User decision 2026-10-05: independent objectives |
| R6 | Size-1 batch broadcasting in `matmul` is slow on some backend (it materialises the basis) | Phase 2 benchmark; fall back to reshaping the data to rank 3 `[B·…, n, q]` |
| R7 | Native power-of-two `irfft` kernels treat Im Y_0 differently from §2.1 on some backend, so `Dft` ≠ `Fft` there | C0.3 on flex/Metal; the DFT path follows NumPy/torch semantics regardless |

### 9.2 Open questions

- **Q1.** Should `None` mean `Auto` after phase 3, if the gates pass? The operator is the
  same, so outputs change only at f32 rounding.
  *Recommendation:* keep `None` = `Fft` (today), and set `Auto` explicitly in the
  examples. Revisit after a release.
- **Q2.** If packed GEMM loses to Gauss on flex but wins on Metal, should both be kept?
  *Recommendation:* keep both behind a private switch chosen by backend only if the gap
  exceeds the IQR; otherwise keep packed only.
- **Q3.** Should the weights be stored as `[M, 2I, 2O]`, which would need a convention
  bump? *Recommendation:* no, unless phase 1 shows the per-forward packing above 10 % of
  layer time.
- **Q4.** Should the bases be cached per (n, m, device, dtype)?
  *Recommendation:* not before phase 2 measures the build cost. If needed, cache them in a
  `#[module(skip)]` field keyed on s, never persisted.

- **Q5.** Do you sign off the new non-normative CONVENTIONS §9 "Implementation notes"
  (§2.7), as the way to record the transform choice without a version bump?
  *Recommendation:* yes. The alternative is to fold the note into pytorch-parity's v2 bump.

None of Q1–Q4 blocks phase 0. Q5 blocks C0.4 only.

### 9.3 Handing components to Claude Code

- Self-contained, can run in parallel: C0.1, C0.2, C0.3 and C0.4. C0.3 is a verbatim copy of
  pytorch-parity T2.
- C1.1 and C1.2 can run in parallel. Both are gated on C0.2 having merged, since it is
  their oracle.
- C2.1 is self-contained after phase 1. C2.2 needs C2.1.
- **Human sign-off needed:**
  - C0.4: the new non-normative CONVENTIONS §9 (§2.7), which avoids a version bump;
  - the §8.2 tolerance revision, if phase 0 forces one;
  - Q1 before C3.1 lands;
  - choosing `Auto`'s rule (C3.2).
- The `numerics` agent reviews C0.2, C1.x and C2.x diffs. The `reviewer` agent runs at the
  end of every `/do-task`.

## 10. References

- `docs/CONVENTIONS.md` §1, §4, §5, §6 (version 1).
- `src/neural_operators/layers/spectral_convolution.rs` and `utils/fft.rs` at `478c2cb`.
- Burn fork `ax1s-x1zz/burn@ddfa9af`, files cited in §6, read in the local checkout.
- `docs/design/pytorch-parity.md` and `docs/pytorch-parity/phase0/T2-irfft-dc-nyquist.md`
  on branch `anees/design-neural-operator` (goldeye worktree), read for §9.1 R5.
- Z. Li et al., *Fourier Neural Operator for Parametric PDEs*, ICLR 2021. Reference
  scripts `fourier_1d.py`/`fourier_2d.py`, from memory (unverified for line numbers).
- J. D. Markel, "FFT pruning", IEEE Trans. Audio Electroacoust., 1971 (from memory,
  unverified).
- H. V. Sorensen, C. S. Burrus, "Efficient computation of the DFT with only a subset of
  input or output points", IEEE Trans. Signal Process., 1993 (from memory, unverified).
- The user-supplied performance review, §4.3 "Spectral convolution" (pasted 2026-10-05;
  its line numbers predate the current file).
