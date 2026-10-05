# Feature parity with the PyTorch `neuraloperator` library

As of 2026-10-05.

Revisions:
- **2026-10-05, Q1.** The user decided Q1: the spectral convolution follows neuraloperator
  main `00b7d86`.
- **2026-10-05, rewrite.** The document is rewritten as a gap list against the existing
  code. The earlier draft proposed a parallel model family, a new CONVENTIONS §9, a
  weight-import layer and a fixture framework. Those are withdrawn: none of them is
  needed to add the missing features.
- **2026-10-05, second numerics review.** Added the CONVENTIONS v2 diff (§2.6, one bump in
  T0.4) and the default-parameter pin (T0.3). Also fixed the module-structure rules, the
  double activation, the factorised storage and the #702 resize case.

> Where this document and `docs/CONVENTIONS.md` differ, the conventions file takes
> precedence.

**Recommendation in one line.** Add each missing neuraloperator feature to the module
where it belongs, as an `Option<_>` config field or a new function whose default keeps
today's behaviour. Write new files only for the models sciml-rs does not have (UNO,
SFNO, GINO) and their layers.

**References and citation forms.**
- neuraloperator: tag `2.0.0`, commit `b307030d9b7ee530823d6a05bddb8238ba22a743`,
  released 2025-10-22. Sources: GitHub releases, and PyPI at
  https://pypi.org/pypi/neuraloperator/json.
- `@2.0.0 path:line` cites `neuraloperator/neuraloperator` at that tag. `@main` cites
  main at `00b7d86`, dated 2026-08-06.
- sciml-rs paths are relative to `src/neural_operators/` at `478c2cb`, unless they start
  with `docs/`, `tests/` or `examples/`.
- Burn paths are relative to `~/.cargo/git/checkouts/burn-4aa1ec9707f89bee/ddfa9af/crates/`,
  written `<B>/`.

---

## 1. Purpose, scope and assumptions

### 1.1 Goal

Bring sciml-rs up to the functionality of neuraloperator 2.0.0 for regular-grid neural
operators, their training and data stack, and the geometry models.

Existing behaviour stays exactly as it is:
- every current config, checkpoint, example and test keeps working unchanged;
- new behaviour is opt-in, following the precedent of `FNOConfig::spectral_init` and
  `FNOConfig::padding` (`models/fno.rs:30-51`), as `CLAUDE.md` requires.

### 1.2 Definition of done

A feature is done when both of these hold:
- It matches neuraloperator's definition, cited `@2.0.0 path:line`.
- A test checks it against an independent reference: a formula or naive computation in
  the test, or small PyTorch reference values pasted into the test. The existing
  `forward_2d_matches_pytorch_reference` (`layers/spectral_convolution.rs:497`) shows the
  pattern.

The objective as a whole is done when, in addition, a Darcy training run with
neuraloperator's settings reproduces the error of the same run in PyTorch (§8.3).

Loading neuraloperator checkpoints is **not** part of done (§9.2 Q8).

### 1.3 In scope

- **SpectralConv features:** bias, `separable`, factorised weights (TFNO),
  `max_n_modes`, `output_shape`.
- **FNO features:**
  - activation choice, lifting and projection MLPs, more than one output channel;
  - channel MLP, skip types, normalisation layers, symmetric fractional padding;
  - neuraloperator's grid, and the `tanh` stabiliser.
- **Losses:** `H1Loss`, `FiniteDiff`, and neuraloperator's `LpLoss` options.
- **Data:** `.pt` datasets (Darcy, Navier-Stokes), channel-wise normalisation.
- **Training:** AdamW, StepLR, ReduceLROnPlateau, scheduler stepping per epoch,
  evaluation at several resolutions, saving and resuming the training state, incremental
  training.
- **New models:** UNO, SFNO and GINO / FNOGNO.

### 1.4 Out of scope

Recorded for later objectives:
- models: CODANO, UQNO, LocalNO, RNO, OTNO, DeepONet;
- layers: DISCO convolutions, attention kernel integrals, Fourier-continuation layers;
- PINO equation losses and the meta-loss aggregators;
- `mpu` / distributed training, mixed precision, Tensor-GaLore;
- The Well datasets, zarr/HDF5 datasets, and generating spherical SWE data;
- complex-valued data (`complex_data=True`), and `preactivation`;
- importing neuraloperator checkpoints (Q8).

### 1.5 Assumptions

- **A1.** The user can run neuraloperator in Python once, by hand, to produce the Darcy
  baseline (§8.3) and the small reference values for tests. No Python is committed and
  CI never runs Python (user's answer to Q2).
- **A2.** neuraloperator models are used with even mode counts. Its docstring asks for
  this (`@2.0.0 neuralop/layers/spectral_convolution.py:210`).

---

## 2. Foundations

Notation follows CONVENTIONS: B batch, D spatial dimension, s_a the extent of axis a,
I and O the input and output channels, C a channel count, H the hidden width.

### 2.1 SpectralConv needs no structural change

This was checked by the `numerics` review in numpy, in 2-D and 3-D, at even and odd
extents.

**Same frequencies for even modes.**
- neuraloperator keeps a centred block of K modes on each non-last axis, after
  `fftshift` (`@2.0.0 spectral_convolution.py:487-497`). It keeps k//2 + 1 bins on the
  last axis (`:398-399, :499-502`).
- For even K, the centred block holds the frequencies {−K/2, .., K/2 − 1}. That is
  exactly the sciml-rs low block plus the high block with `modes[a] = K/2` (CONVENTIONS
  §5).
- So neuraloperator's `n_modes = (K_1, .., K_D)` corresponds to
  `modes = (K_1/2, .., K_{D−1}/2, K_D//2 + 1)`.
- The review measured agreement to 1.8e-15. Only a helper that converts mode counts is
  needed.

**FFT normalisation.**
- neuraloperator defaults to `fft_norm="forward"`; sciml-rs uses "backward"
  (CONVENTIONS §4).
- The layer is linear, so the two give identical outputs when the output grid equals the
  input grid, which covers every FNO layer. They differ by the factor N_out/N_in when the
  grid changes (§2.2).

**Odd non-last extents.** 2.0.0 shifts back with `fftshift` instead of `ifftshift`
(`:516`), which is wrong at odd extents; main #688 fixes this (`@main :532`).
sciml-rs's corner layout already computes main's (correct) operator, so following main
(Q1) needs no change.

**main #702.** Enforces Hermitian symmetry: it inverts the non-last axes, then drops the
imaginary parts of the DC and Nyquist bins before `irfft` (`@main :299, :547-559`).
Flex already behaves this way:
- the flex power-of-two `irfft` reads only `x_re[0]` and `x_re[half]`
  (`<B>/burn-flex/src/ops/fft.rs:1277-1278`);
- the Bluestein path, which is backend-independent, takes the real part
  (`<B>/burn-signal/src/functions/fft.rs:147-155`), and `hermitian_extend` skips Nyquist
  (`<B>/burn-signal/src/functions/fft.rs:402-413`);
- the CubeCL kernels write only the real part, and their mirror indices skip DC and the
  last bin (`cubek 333c406 crates/cubek-fft/src/fft/irfft.rs:179-212`,
  `rfft_large.rs:399-410`);
- autodiff is consistent with this (`<B>/burn-signal/src/backends/autodiff.rs:112-115, :169-194`).

The numerics review verified all of this by reading the sources. T0.2 tests it on flex
and metal, which closes the open point in CONVENTIONS §4.

### 2.2 SpectralConv additions

**Bias** (`@2.0.0 :361-364, :525-526`).
- A real b ∈ ℝ^{O×1×..×1}, added after the inverse transform.
- Drawn as σ·randn, with σ = √(2/(I+O)) (`:312-313`).

**Normal init.**
- neuraloperator draws the complex weight with `normal_(0, σ)` (`:355`). The weight is a
  tltorch `DenseTensor`, because `factorization=None` becomes `"Dense"` (`:325-326`).
- For a complex tensor, torch is believed to give each of the real and imaginary parts
  variance σ²/2. **Unverified:** T1.1 confirms this with one PyTorch run, before the new
  `SpectralInit` variant is fixed in CONVENTIONS §6.

**Separable** (`:21-52, :328-337`).
- The weight is `[C, modes..]` and requires I = O.
- The contraction is elementwise per channel instead of mixing channels.

**`max_n_modes` and `set_n_modes`** (`:449-474`).
- Weights are allocated at `max_n_modes`. A smaller active `n_modes` uses the centre of
  the weight.
- In sciml-rs's corner form, the centre is the *first* K'/2 entries of the low corner and
  the *last* K'/2 entries of the high corner, on each non-last axis. On the last axis it
  is the first K_D'//2 + 1 bins.
- Incremental training uses this.

**Mode clipping.**
- neuraloperator clips modes to the grid without warning (`:450-453`); sciml-rs panics
  (`check_modes_fit`, `layers/spectral_convolution.rs:234-247`).
- New option `clip_modes: Option<bool>`. `None` keeps the panic. `Some(true)` keeps
  K' = min(K, 2⌊s/2⌋) on non-last axes and min(K_D//2 + 1, s_D//2 + 1) on the last axis.
- At odd extents neuraloperator can keep an odd count, which sciml-rs cannot express;
  this is a documented deviation (§9.1).

**`output_shape` and `resolution_scaling_factor`** (`:509-523`). The output grid t may
differ from the input grid s. neuraloperator builds the output spectrum on the
**input** grid. `irfftn(s = t)` then zero-pads or trims at the **end** of each non-last
axis. The result is scaled by the "forward" normalisation factor N_out/N_in.
- This is *not* Fourier interpolation on non-last axes: upsampling moves the
  negative-frequency block onto positive frequencies.
- The review measured an O(1) difference against correct placement, e.g. 6.75 for
  8² → 16².
- To reproduce it in sciml-rs: compute the corners on the input grid, then call
  `icfft_full_spectrum(re, im, axis, t_a)` on each non-last axis; Burn's `cfft` resizes at
  the end (`<B>/burn-signal/src/functions/fft.rs:172-184, :222-223`). Then
  `irfft(n = t_D)`, then × N_out/N_in.
- Changing only the last axis agrees exactly either way, with **one exception from
  #702** (Q1).
  - main zeros the imaginary part of the last bin of the *input* half-spectrum,
    `out_fft[..., -1].imag`, when the output length t_D is even
    (`@main spectral_convolution.py:555-556`).
  - When the last axis is upsampled, that bin becomes an interior bin, and Burn's
    `irfft` keeps its imaginary part.
  - The review measured differences of 2.8e-2 for s → t = 8 → 16, and 0.18 for 9 → 10.
    The differences appear only when `modes[D] = s_D//2 + 1`.
  - So the `output_shape` path zeros the imaginary part of input bin s_D//2 when t_D is
    even and t_D ≠ s_D.
- Whether to copy this behaviour is **Q7, still open**. It only blocks UNO.

**Factorised weights** (TFNO; `@2.0.0 spectral_convolution.py:347-354`, via
`tltorch.FactorizedTensor`).
- The dense weight W ∈ ℂ^{I×O×K..} is replaced by a factorised form:
  - **Tucker:** a core G times one factor matrix per mode;
  - **CP:** a sum of R rank-one outer products;
  - **TT:** a chain of 3-way cores.
- `rank` is a fraction of the dense parameter count. TFNO uses Tucker with rank 0.1
  (`@2.0.0 models/fno.py:476-479`).
- The FNO default is `rank = 1.0` (`fno.py:188`); the SpectralConv default of 0.5 is not
  what FNO passes.
- There are two contractions with the same result:
  - **reconstructed:** build W, then use the existing contraction;
  - **factorised:** contract x with the factors directly, using `einsum`.
- **Storage.**
  - A factorised layer holds the factors of the *full centred* K-block, each factor a
    real/imaginary pair of `Param`s.
  - Its per-corner `weights_re`/`weights_im` vectors are **empty**. An empty `Vec`
    produces no keys and no parameters, so nothing is counted twice.
  - Dense layers are unchanged.
  - Factorising each corner separately would be a different model with a different
    parameter count, so it is ruled out.
- **Reconstruction into corners** (the "reconstructed" path).
  - On a non-last axis, centred index j means frequency j − K/2. The low corner takes
    rows K/2..K of that axis's factor, and the high corner takes rows 0..K/2.
  - The last axis is unshifted, with K_D//2 + 1 rows.
  - Reconstruction is a sequence of complex mode products (3–4 real matmuls each), not
    one expanded real einsum.
- **Inputs needed for exact parity**, ported in T2.1 from the tltorch and tensorly
  versions that neuraloperator 2.0.0 installs:
  - **Rank resolution**: `validate_tucker_rank`, `validate_cp_rank` and
    `validate_tt_rank`, called at `tltorch/factorized_tensors/factorized_tensors.py:236,
    :97, :383` on tltorch main. The parameter-count gate depends on these.
  - **Factor initialisation**: 2.0.0 calls `normal_(0, σ)` on the factorised tensor
    (`spectral_convolution.py:355`). For Tucker, tltorch draws each factor with std
    (σ / ∏√r_i)^(1/(order+1)) (`factorized_tensors.py:310-320`); CP is at `:132-139` and
    TT at `:417-425`.
  - These line numbers are from tltorch **main**. The pinned tltorch version is
    unverified, so T2.1 confirms them first.

### 2.3 FNO additions

Source: `@2.0.0 neuralop/models/fno.py:164-402` and `neuralop/layers/fno_block.py:301-338`.

**neuraloperator's block, in post-activation form.**

```math
\begin{aligned}
\tilde v &= \sigma\big(\mathrm N(\mathcal K v) + S^{\text{fno}} v\big) \\
v' &= \sigma\big(\mathrm N(\mathrm{MLP}\,\tilde v + S^{\text{mlp}} v)\big)
\end{aligned}
```

- σ is skipped on the last layer (`fno_block.py:323, :335`).
- Both skips take the block *input* v (`:302-308`).

**Today's sciml-rs block.**

```math
v' = \mathrm{ReLU}(\mathcal K v + W v)
```

- W is a 1×1 `Conv1d` **with** a bias (`models/fno.rs:98-102`; Burn's `Conv1dConfig`
  defaults to `bias = true`, `<B>/burn-nn/src/modules/conv/conv1d.rs:36-38`).
- The block loop is at `fno.rs:277-281`. There is no ReLU after the last block.

**How the two blocks relate.**
- Same function class: today's block is the neuraloperator block with σ = ReLU, a linear
  S^fno, no norm and no channel MLP.
- Same parameter count: neuraloperator's linear skip has no bias, but its spectral conv
  has one (`skip_connections.py:37-42`, `spectral_convolution.py:361`). So
  𝒦v + b + Sv has as many parameters as today's 𝒦v + Wv + b.
- Only the initialisation differs.

**Double activation.** Without a channel MLP and without a norm, neuraloperator applies
σ **twice** on every non-last layer: once at `fno_block.py:323-324` and again at
`:335-336`.
- That is harmless for ReLU, which is idempotent.
- `activation: Some(Gelu)` with `channel_mlp: None` must compute GELU(GELU(𝒦v + Sv)) to
  match. T1.4 pins a PyTorch reference with `use_channel_mlp=False` and GELU.

**Features to add**, each as an `Option` field on `FNOConfig`. `None` keeps today's
model.

| neuraloperator | Definition | `FNOConfig` field (None = today) |
|---|---|---|
| `non_linearity=F.gelu` | exact GELU (`<B>/burn-tensor/src/tensor/activation/base.rs:93`) | `activation: Option<Activation>` (ReLU) |
| `use_channel_mlp`, `channel_mlp_expansion=0.5`, `channel_mlp_dropout` | 2-layer Conv1d(k=1) MLP, C → round(0.5C) → C, GELU between (`fno_block.py:203-213`, `channel_mlp.py:66-110`); Python `round` is half-to-even | `channel_mlp: Option<ChannelMlpConfig>` (none) |
| `channel_mlp_skip="soft-gating"` | x ⊙ w, w ∈ ℝ^{1×C×1..} init ones, no bias (`skip_connections.py:82-93`) | part of `ChannelMlpConfig` |
| `fno_skip="linear"/"identity"/"soft-gating"` | linear = Conv1d(k=1), no bias (`skip_connections.py:37-43`) | `fno_skip: Option<SkipKind>` (today's `w`, which **has** a bias) |
| `norm` | instance (no affine), group (1 group), batch, AdaIN; 2 per block (`fno_block.py:238-273`) | `norm: Option<NormKind>` (none) |
| `lifting_channel_ratio=2` | 2-layer MLP (C_in+D) → ⌊2H⌋ → H (`fno.py:212, 303-316`) | `lifting_ratio: Option<f64>` (today's single `fc0`) |
| `projection_channel_ratio=2` | 2-layer MLP H → ⌊2H⌋ → C_out (`fno.py:215, 332-339`) | `projection_hidden: Option<usize>` (today's fixed 128, `fno.rs:54-64`) |
| `out_channels` | any | lift the `out_channels == 1` assert (`fno.rs:75-78`, `training/trainer.rs:98`) |
| `domain_padding=δ` | symmetric, p_a = round_half_even(δ s_a) each side, zeros, after lifting (`padding.py:78-128`) | `domain_padding: Option<Vec<f64>>`, alongside today's `padding` (one-sided cells); setting both panics |
| `positional_embedding="grid"` | data first, then one channel per axis in axis order, `linspace(0,1,s+1)[:-1]` ('ij', periodic [0,1)) (`embeddings.py:163, 380-381`) | `grid: Option<GridKind>` (today's closed [0,1], reverse order, CONVENTIONS §2) |
| `stabilizer="tanh"` | tanh before each spectral conv (`fno_block.py:310-314`) | `stabilizer: Option<Stabilizer>` |
| `factorization`, `rank`, `separable`, `max_n_modes` | §2.2 | passed through to `SpectralConv` |

The constructor `FNOConfig::neuralop(n_modes, in_channels, out_channels, hidden)` sets
all of these to neuraloperator's defaults (`fno.py:164-195`), and `FNOConfig::tfno(..)`
adds Tucker with rank 0.1.

**Implementation note: module structure and checkpoints.**

*How Burn loads a model.*
- The model's structure is fixed by the config at `init`; `load_from` then fills in the
  values.
- A `None` `Option<Module>` or `Option<Param>` visits nothing and adds no key
  (`<B>/burn-core/src/module/param/primitive.rs:14-18`).
- `Option` adds no path segment either, so `fc0: Option<Linear>` still has the key
  `fc0.weight`.
- So an all-`None` config has exactly today's keys, and old `.bpk` files load.

*What this means for new parameters.*
- New parameters go in new optional fields, for example `lifting: Option<ChannelMlp>` or
  a `channel_mlp: Vec<ChannelMlp>` that is empty when off.
- Today's `fc0`, `w`, `fc1` and `fc2` become `Option`, or an empty `Vec`, when a new
  option replaces them (lifting MLP, identity or soft-gating skip). Otherwise the
  parameter-count gate fails. This stays checkpoint-compatible, since `Option` keeps the
  same keys.

*Errors when loading.*
- If an option is `Some` and the file lacks its keys, `BurnpackStore` returns
  `Err("Missing tensors")`, unless `allow_partial(true)` is set
  (`<B>/burn-store/src/burnpack.rs:95, :452-457`).
- Extra keys in the file are only reported as unused.

*Other rules.*
- Fields that are not parameters (activation, grid kind) need `#[module(skip)]`
  (`<B>/burn-derive/src/lib.rs:34`).
- No existing parameter is renamed or reshaped.
- **`None` branches must make no RNG calls.** This includes building a module and then
  discarding it. A stray draw shifts every later layer's weights for a given seed; T0.3
  pins this.

### 2.4 Losses

Source: `@2.0.0 neuralop/losses/data_losses.py`.

**`LpLoss(d, p, measure=1, reduction="sum", eps=1e-8)`** (`:68-216`).
- `rel = ‖x−y‖_p / (‖y‖_p + eps)`, over the last d axes. It is then reduced over *all*
  leading axes, batch and channel alike.
- `abs` uses quadrature h_a = measure / s_a.
- sciml-rs's `LpLoss` (`losses/data_losses.rs:27-133`) differs in three ways:
  - it has no eps;
  - its h_a is 1/(s_a − 1) (CONVENTIONS §8);
  - it takes flattened `[B, N]` input.
- New options `eps: Option<f64>` and `measure: Option<Vec<f64>>` are added, with `None`
  keeping today's behaviour. A `rel_nd` takes `[B, C, s..]` input.

**`H1Loss(d, measure=1, reduction="sum", eps=1e-8, periodic_in_x/y/z=True)`**
(`:266-474`).

```math
\frac{\sqrt{\sum|u-v|^2 + \sum_j\sum|\partial_j u-\partial_j v|^2}}{\sqrt{\sum|v|^2+\sum_j\sum|\partial_j v|^2}+\varepsilon}
```

**`FiniteDiff`** (`@2.0.0 losses/differentiation.py:15-71`) gives ∂_j with
h = measure/s:
- central differences with wrap-around on periodic axes;
- third-order one-sided stencils at the boundaries of non-periodic axes.

**Gradient at an exact match.** For p > 1, torch's gradient at x = y is NaN. sciml-rs
returns 0 (CONVENTIONS §8); this is kept as a documented deviation.

### 2.5 Data normalisation

**`UnitGaussianNormalizer(eps=1e-7, dim)`** (`@2.0.0 data/transforms/normalizers.py:65-158`).
- It reduces over `dim` with ddof = 1.
- PTDataset's `"channel-wise"` encoding reduces over every axis except the channel axis
  (`pt_dataset.py:164-167`).
- sciml-rs's `UnitGaussianNormalizer` (`data/transforms/normalizers.rs:36-108`) is
  pointwise. A `with_dims` option is added, and `with_eps` already exists.

**`partial_fit`.** It computes std = √(E[x²] − μ²) · n/(n−1) (`:148-152`): the
correction factor sits outside the square root. This is not ported. PTDataset uses `fit`
(`pt_dataset.py:171, :184`).

### 2.6 Proposed CONVENTIONS diff (version 2)

`docs/CONVENTIONS.md` requires two things:
- any change to its §1–§6 bumps `CONVENTION_VERSION`;
- a new convention lands in that file *before* the code that relies on it.

So every option below lands together, in one diff with one bump, in task T0.4. That
happens before T0.2's note or any Phase 1 code, and needs the user's sign-off. Each line
states the default, which is today's behaviour, and the alternative. Later tasks only
cite these lines; they do not edit CONVENTIONS §1–§6.

```diff
-`CONVENTION_VERSION = 1`.
+`CONVENTION_VERSION = 2`.
@@ preamble
+Version 2 adds opt-in options (feature parity with neuraloperator 2.0.0). Every
+option's default is the version-1 behaviour; a config with all new fields `None`
+is bit-identical to version 1, including its RNG draw sequence.
@@ §1 Tensor layouts
-- `out_channels` is 1 (asserted in `FNOConfig::init`).
+- `out_channels` may be any positive number; losses then reduce over batch and channel.
@@ §2 Coordinate grid channels
+- Option `FNOConfig::grid = Some(GridKind::Periodic)`: channels `[data, x(s_1), .., x(s_D)]`
+  (axis order), each `x(s_a) = arange(s_a) / s_a` on [0, 1) (neuraloperator `GridEmbeddingND`).
@@ §3 Domain padding
+- Option `FNOConfig::domain_padding = Some(δ)`: symmetric, p_a = round_half_even(δ_a s_a)
+  zeros on each side of every spatial axis after lifting, cropped before projection.
+  Mutually exclusive with `padding`.
@@ §4 FFT and normalisation
-- Not yet fixed by a test: how `irfft` treats a nonzero imaginary part in the DC bin and,
-  for even s_D, the Nyquist bin s_D/2 of the last axis. ...
+- `irfft` ignores the imaginary part of the DC bin and, for even s_D, of the Nyquist bin
+  (test `irfft_ignores_dc_nyquist_imag`, T0.2), so neuraloperator main #702's Hermitian
+  enforcement needs no extra step on equal grids.
+- Option `output_shape` (t ≠ s): spectrum built on the input grid, resized at the end of
+  each non-last axis, imaginary part of input bin s_D//2 zeroed when t_D is even and
+  t_D ≠ s_D, `irfft(n = t_D)`, then × N_out/N_in.  [pending Q7]
@@ §5 Spectral convolution layout
+- neuraloperator `n_modes = (K_1, .., K_D)` (K_a even for a < D) ↔
+  `modes = (K_1/2, .., K_{D−1}/2, K_D//2 + 1)`: same retained frequencies.
+- Option bias: real `[O, 1, .., 1]` added after the inverse transform.
+- Option `clip_modes`: K_a' = min(K_a, 2⌊s_a/2⌋), last axis min(K_D//2+1, s_D//2+1).
+- Option `max_modes` / `set_modes`: active modes are the first m of the low corner and
+  the last m of the high corner on non-last axes, the first m on the last axis.
+- Option separable: weights `[C, modes..]` per corner (I = O), elementwise product.
+- Option factorised (Tucker/CP/TT): factors of the full centred block, each a re/im
+  `Param` pair; corner vectors empty; reconstruction per design pytorch-parity §2.2.
@@ §6 Spectral weight initialisation
+| `NeuralopNormal` | re, im each N(0, σ²/2), σ = √(2/(I+O)); bias N(0, σ²) |  [σ²/2 confirmed in T1.1]
@@ §8 Losses
+- Options on `LpLoss`: `eps` (added to the denominator norm), `measure` (h_a = measure_a/s_a);
+  `rel_nd` on `[B, C, s..]` reduces over batch and channel. `H1Loss` per pytorch-parity §2.4.
```

The two lines marked as pending are written as "to be confirmed" in T0.4. The task that
settles each one then updates it: T1.1 for the σ²/2 variance, T4.2 for Q7. Neither
changes a default, so neither needs another version bump.

---

## 3. neuraloperator today

### 3.1 Package map at 2.0.0

The map comes from the recursive tree `git/trees/2.0.0`. Top-level exports are at
`neuralop/__init__.py:3-19`.

| Area | Contents | Notes |
|---|---|---|
| `models/` | `FNO`, `TFNO`, `SFNO`, `UNO`, `UQNO`, `FNOGNO`, `GINO`, `CODANO`, `LocalNO`, `get_model` | TFNO = FNO + Tucker, rank 0.1 (`fno.py:476-479`); SFNO = FNO + `SphericalConv` (`sfno.py:10`); SFNO/LocalNO need torch_harmonics (`models/__init__.py:3-8`) |
| `layers/` | `spectral_convolution`, `spherical_convolution`, `fno_block`, `channel_mlp`, `skip_connections`, `embeddings`, `padding`, `normalization_layers`, `complex`, `resample`, `einsum_utils`, `integral_transform`, `gno_block`, `gno_weighting_functions`, `neighbor_search`, `segment_csr`, `attention_kernel_integral`, `coda_layer`, `differential_conv`, `discrete_continuous_convolution`, `local_no_block`, `fourier_continuation`, `spectral_projection` | factorised weights via tltorch (`spectral_convolution.py:8-16`) |
| `losses/` | `LpLoss`, `H1Loss`, `HdivLoss`, `PointwiseQuantileLoss`, `MSELoss`; equation losses; meta losses; `differentiation` | |
| `data/datasets/` | Darcy, Navier-Stokes, `PTDataset`, Burgers 1D-time, car-cfd, mesh data module, nonlinear Poisson, spherical SWE, The Well; bundled `darcy_{train,test}_16.pt`, `darcy_test_32.pt`, `mini_car.pt` | `.pt` dicts `{x, y}` (`pt_dataset.py:105-134`); Zenodo downloads (`web_utils.py:152-153`) |
| `data/transforms/` | `UnitGaussianNormalizer`, `DataProcessor`, `DefaultDataProcessor`, `IncrementalDataProcessor`, `MGPatchingDataProcessor` | |
| `training/` | `Trainer`, `IncrementalFNOTrainer`, complex-aware `AdamW`, `save/load_training_state` | eval errors are per-sample means (`trainer.py:459-462`) |
| `mpu/` | model-parallel helpers | out of scope |

**How the parts fit together.**
1. A dataset yields `{"x", "y"}`.
2. The `DataProcessor` moves a batch to the device and normalises it (`preprocess`).
3. The model predicts.
4. The processor decodes the prediction (`postprocess`).
5. The `Trainer` owns the loop, evaluation at several resolutions, and checkpoints.

sciml-rs has the same shape:
- Loaders return `OperatorDataset`s (`data/dataset.rs:13-67`).
- `DeviceBatcher` moves batches to the device (`data/device_batcher.rs:54-122`).
- The `Postprocess` closure does the decoding (`training/trainer.rs:49-53`).
- `training_loop` runs training (`training/trainer.rs:315-366`).

So the training stack is extended in place; nothing is replaced.

### 3.2 Releases, renames and deprecations

| Tag | Date | Relevant changes |
|---|---|---|
| 2.0.0 | 2025-10-22 | `FNO1d/2d/3d`, `TFNO1d/2d/3d` and `**kwargs` removed (#646); one-sided padding and `domain_padding_mode` removed, symmetric only (#525, #630); `SpectralConvNd` removed (#511); `localFNO` → `LocalNO` (#526); mode centring fixed (#508, #647); `use_channel_mlp`, float channel ratios, `batch_norm` restored (#553, #598, #610); configs YAML → zencfg (#592) |
| 1.0.2 | 2024-12-30 | packaging only |
| 1.0.1 | 2024-12-20 | Darcy dataset version bump; `state_dict` metadata and version tag (#493) |
| 1.0.0 | 2024-12-17 | new `Trainer` (callbacks removed); `DataProcessor`; complex spectral parameters (#401) and complex-aware `AdamW` (#420); GINO, UQNO |

Constructor names were renamed between 0.3.0 and 1.0:
- `lifting_channels` → `lifting_channel_ratio`;
- `use_mlp` → `use_channel_mlp`;
- `output_scaling_factor` → `resolution_scaling_factor`.

See `@0.3.0 models/fno.py:92-125` and `@1.0.2 :162-194`. This document uses the 2.0.0
names.

### 3.3 Changes on main after 2.0.0

main is 65 commits ahead of the tag. The changes that matter here:
- **#688:** `ifftshift` fix (followed, Q1).
- **#702:** Hermitian enforcement (followed, Q1; no change needed, §2.1).
- **New models:** RNO and OTNO.
- **New FNO options:** `conv_bias_kernel` (#722) and `norm_groups` (#715).
- **Optimisers:** the `tensorgrad` optimiser (#730).
- **Data:** PTDataset keeps the input dtype (#727).

**Stability.**
- *Unstable:* SpectralConv's inverse path, FNOBlocks' skip options, optimisers, and
  dataset dtypes.
- *Stable since 1.0:* the `Trainer`/`DataProcessor` split, the `LpLoss`/`H1Loss`
  definitions, and the `.pt` format.

---

## 4. Gap analysis

### 4.1 Gap table

The core of this design. "Where" names the file that changes. Every change is additive.

**FNO family on regular grids**

| Feature (neuraloperator) | sciml-rs today | Where, and how | Burn | Size |
|---|---|---|---|---|
| spectral bias (`spectral_convolution.py:361`) | none | `layers/spectral_convolution.rs`: `bias: Option<Param<..>>`, set by new `SpectralConv::with_bias` | ✓ | S |
| normal init N(0, 2/(I+O)) | `LiUniform`, `Normal`, `SymmetricUniform` (`spectral_convolution.rs:36-80`) | same file: new `SpectralInit::NeuralopNormal` variant | ✓ | S |
| `n_modes` as neuraloperator counts them | `modes` per corner | same file: `modes_from_neuralop(&[usize])` helper (§2.1) | ✓ | S |
| mode clipping | panics (`:234-247`) | same file: `clip_modes` option | ✓ | S |
| `separable` | none | same file: separable weight variant | ✓ | S |
| `max_n_modes` / `set_n_modes` | none | same file | ✓ | M |
| `output_shape` / `resolution_scaling_factor` | none | same file, behind Q7 | ✓ (`cfft` resize) | M |
| factorised weights Tucker/CP/TT (TFNO) | none | new `layers/factorized.rs`, used by `SpectralConv` | `einsum` (`<B>/burn-tensor/src/tensor/api/einsum/mod.rs:67`) | L |
| GELU and other activations | ReLU only (`models/fno.rs:277-281`) | `models/fno.rs`: `activation` option | ✓ `gelu` | S |
| channel MLP + soft-gating skip | none | new `layers/channel_mlp.rs`, `layers/skip.rs`; used in `fno.rs` | ✓ | M |
| skip types for the spectral branch | 1×1 conv with bias | `fno.rs` + `layers/skip.rs` | ✓ | S |
| norms: instance, group, batch, AdaIN | none | new `layers/norm.rs`; used in `fno.rs` | Instance/Group/Batch present; Batch running var biased (§6) | M |
| lifting / projection MLPs with ratios | `fc0` and fixed `fc1` = 128 (`fno.rs:54-64`) | `fno.rs`: `lifting_ratio`, `projection_hidden` | ✓ | S |
| `out_channels > 1` | asserted 1 (`fno.rs:75-78`, `training/trainer.rs:98`) | `fno.rs`, `training/trainer.rs` (`flatten_pair`) | ✓ | S |
| symmetric fractional padding | one-sided cells (`fno.rs:163-171`) | `fno.rs`: `domain_padding` | ✓ | S |
| grid [0,1), axis order, data first | closed [0,1], reverse order (`fno.rs:197-232`) | `fno.rs`: `grid` option in `grid_cl` | ✓ | S |
| `tanh` stabiliser | none | `fno.rs` | ✓ | S |
| `FNOConfig::neuralop`, `::tfno` | none | `fno.rs` | — | S |
| UNO (`models/uno.py:112-159`) | none | new `models/uno.rs`, `layers/resample.rs` | `interpolate` 2-D only (§6) | L |
| SFNO (`sfno.py`, `spherical_convolution.py`) | none | new `layers/spherical.rs` (SHT), `models/sfno.rs` | no SHT | L |

**Losses, data and training**

| Feature | sciml-rs today | Where, and how | Burn | Size |
|---|---|---|---|---|
| `H1Loss` (`data_losses.py:266-474`) | none | new `losses/h1.rs` | ✓ | M |
| `FiniteDiff` (`differentiation.py:15-71`) | none | new `losses/differentiation.rs` | ✓ | S |
| `LpLoss` `eps`, `measure`, N-d input | no eps, closed-grid h (`losses/data_losses.rs:27-133`) | same file: options + `rel_nd` | ✓ | S |
| `.pt` dataset files | `.mat`, `.npy`, `.npz` (`data/io/readers/`) | new `data/io/readers/pt.rs` | `pytorch-reader` (`<B>/burn-store/src/pytorch/store.rs:73`); reading a dict of tensors is unverified (T3.1) | M |
| Darcy (16-421) and Navier-Stokes (128) PT loaders | Darcy and Burgers `.mat` (`data/loaders/`) | new `data/loaders/pt_darcy.rs`, `pt_navier_stokes.rs` | — | M |
| channel-wise normaliser | pointwise (`data/transforms/normalizers.rs:36-108`) | same file: `with_dims` | — | S |
| AdamW | Adam + L2 (`training/trainer.rs:274-277`) | same file: `optimizer` option | `AdamWConfig` (`<B>/burn-optim/src/optim/adamw.rs:17`) | S |
| StepLR, ReduceLROnPlateau | cosine only (`trainer.rs:280-287`) | same file + new `training/schedulers.rs` (plateau) | StepLR ✓; plateau absent | M |
| scheduler stepped per epoch | per batch (`trainer.rs:200`) | `trainer.rs`: option | — | S |
| evaluation at several resolutions | one test set (`trainer.rs:315-366`) | `trainer.rs`: several test loaders | — | M |
| save and resume the training state | weights only, no resume (`examples/train/burgers.rs:114`) | new `training/checkpoint.rs` | burn-store `BurnpackStore` | M |
| incremental training (`IncrementalFNOTrainer`) | none | new `training/incremental.rs` | — | M |
| H1 as training and evaluation loss in the trainer | Lp only (`trainer.rs:306`) | `trainer.rs`: a loss enum | — | S |

**Geometry**

| Feature | sciml-rs today | Where, and how | Burn | Size |
|---|---|---|---|---|
| native neighbour search (`neighbor_search.py:84-120`: `cdist`, ≤ r, CSR) | none | new `layers/gno/neighbor_search.rs` | `nonzero`, `cumsum` (`<B>/burn-tensor/src/tensor/api/bool.rs:224`, `numeric.rs:566`) | M |
| `segment_csr` | none | new `layers/gno/segment.rs` | `scatter` + `Add` (`base.rs:1969`) | S |
| `IntegralTransform`, `GNOBlock` (`integral_transform.py:70-224`) | none | new `layers/gno/integral_transform.rs`, `gno_block.rs` | ✓ | M |
| GINO, FNOGNO (`models/gino.py:163-243`) | none | new `models/gino.rs` | ✓ | L |
| mini car-cfd data | none | new `data/loaders/car_cfd.rs` | — | S |

Size: S is under about 200 lines including tests; M is a few hundred; L is a new model
or algorithm.

### 4.2 What does *not* need to change

- The FFT path (`fft_ctensor`, `ifft_ctensor`, `icfft_full_spectrum`), the corner
  layout, and the Gauss complex product: §2.1 shows they already compute neuraloperator
  main's operator.
- `FNO::forward`'s data flow, permuting to channels-first once
  (`fno.rs:249-297`): new options plug into it.
- The loaders, batchers, `DeviceBatcher`, run artefacts and the existing examples.
- CONVENTIONS §1–§8 as they apply today. New options add lines to §2, §3, §5, §6 and §8,
  stating the default (= today) and the alternative (§9.3).

---

## 5. Architecture

### 5.1 New and changed files

```
layers/spectral_convolution.rs   + bias, NeuralopNormal init, clip_modes, separable,
                                   max_n_modes / set_n_modes, output_shape, factorized hook
layers/factorized.rs             new: Tucker / CP / TT spectral weights
layers/channel_mlp.rs            new: ChannelMlp
layers/skip.rs                   new: Skip { Linear, SoftGating, Identity }
layers/norm.rs                   new: BlockNorm { Instance, Group, Batch, AdaIn }
layers/resample.rs               new (UNO)
layers/spherical.rs              new (SFNO): RealSht, InverseRealSht, SphericalConv
layers/gno/                      new (GINO): neighbor_search, segment, integral_transform, gno_block
models/fno.rs                    + Option fields of §2.3, FNOConfig::{neuralop, tfno}
models/uno.rs, sfno.rs, gino.rs  new
losses/data_losses.rs            + eps, measure, rel_nd
losses/h1.rs, differentiation.rs new
data/io/readers/pt.rs            new
data/loaders/pt_darcy.rs, pt_navier_stokes.rs, car_cfd.rs   new
data/transforms/normalizers.rs   + with_dims
training/trainer.rs              + optimizer, scheduler, step-per-epoch, several test sets, loss enum
training/schedulers.rs, checkpoint.rs, incremental.rs       new
examples/train/darcy_neuralop.rs, examples/predict/darcy_neuralop.rs   new (benchmark)
```

### 5.2 Interfaces

The new types follow the existing style: `Config`-derived configs, `Option` fields, and
documented shapes and panics. Exact signatures are settled in the task briefs.

```rust
// layers/spectral_convolution.rs (additions)
impl<const R: usize> SpectralConv<R> {
    /// Adds a real bias [O, 1, .., 1] after the inverse transform, drawn as
    /// N(0, 2/(I+O)) regardless of the weight init (neuraloperator `:361-364`).
    pub fn with_bias(self, device: &Device) -> Self;
    /// x: [B, I, s..] -> [B, O, t..], t = output_shape.unwrap_or(s); see §2.2.
    pub fn forward_with_shape(&self, x: Tensor<R>, output_shape: Option<[usize; R - 2]>) -> Tensor<R>;
    /// Active modes <= allocated modes, corner form (§2.2).
    pub fn set_modes(&mut self, modes: &[usize]);
}
/// neuraloperator n_modes (K_1..K_D, K_a even for a < D) -> sciml-rs modes (§2.1).
pub fn modes_from_neuralop(n_modes: &[usize]) -> Vec<usize>;

// models/fno.rs (additions to FNOConfig; all Option, None = today)
pub activation: Option<Activation>,
pub channel_mlp: Option<ChannelMlpConfig>,
pub fno_skip: Option<SkipKind>,
pub norm: Option<NormKind>,
pub lifting_ratio: Option<f64>,
pub projection_hidden: Option<usize>,
pub domain_padding: Option<Vec<f64>>,
pub grid: Option<GridKind>,
pub stabilizer: Option<Stabilizer>,
pub spectral_bias: Option<bool>,
pub factorization: Option<FactorizationConfig>,
pub max_modes: Option<Vec<usize>>,
impl FNOConfig {
    pub fn neuralop(n_modes: &[usize], in_channels: usize, out_channels: usize, hidden: usize) -> Self;
    pub fn tfno(n_modes: &[usize], in_channels: usize, out_channels: usize, hidden: usize) -> Self;
}

// losses/h1.rs
pub struct H1Loss { pub d: usize, pub measure: Vec<f64>, pub reduction: Reduction,
                    pub eps: f64, pub periodic: Vec<bool> }
impl H1Loss { /// x, y: [B, C, s_1..s_d]; returns [1] (Sum/Mean) or [B·C] (None).
              pub fn rel<const R: usize>(&self, x: Tensor<R>, y: Tensor<R>) -> Tensor<1>; }

// layers/gno/neighbor_search.rs
/// data: [N, d], queries: [M, d] -> CSR { indices: [nnz], row_splits: [M+1] }.
pub fn neighbor_search(data: Tensor<2>, queries: Tensor<2>, radius: f64) -> Csr;
```

### 5.3 Data flow

`FNO::forward` keeps its shape (`fno.rs:259-297`):
1. grid;
2. permute to channels-first;
3. lift: `fc0`, or the lifting MLP;
4. pad: one-sided or symmetric;
5. L blocks: spectral conv (+ bias), plus the skip, optional norm, activation, optional
   channel MLP with its skip;
6. crop;
7. project: `fc1`/`fc2`, or the projection MLP;
8. permute back.

Pointwise layers keep using one batched matmul along the channel axis
(`FNO::pointwise`, `fno.rs:141`). This avoids the permute-then-`cat` that triggered the
known Metal quirk.

### 5.4 Compatibility

- **Configs.** Every new `FNOConfig` field is `Option` and defaults to today's behaviour.
  Old `model_cfg.json` files load, which the existing test at `fno.rs:443` covers; that
  test is extended.
- **Parameters.** New parameters are new optional module fields; no existing parameter
  is renamed or reshaped.
- **Defaults are protected in two ways.**
  - T0.3's parameter pin catches stray RNG draws.
  - The existing tests (pinned PyTorch and NumPy references, the init replay, the
    padding and grid tests) must pass with no change other than `field: None` added to
    struct literals.
- **Struct literals.** Six exhaustive `FNOConfig` struct literals exist:
  `models/fno.rs:314, :365, :471` and the three train examples. Each gains `None` for the
  new fields, in T1.4. The examples' behaviour is unchanged, and the benchmark gets its
  own example.

---

## 6. Burn considerations

All of these were confirmed in `ddfa9af`.

**FFT.**
- Only 1-D `rfft`, `irfft` and `cfft` exist (`<B>/burn-signal/src/functions/fft.rs:53, :118, :354`).
  They use "backward" normalisation and take no `norm` argument; there is no inverse
  `cfft`, which `utils/fft.rs:8-25` already fills in.
- Flex supports powers of two natively (`<B>/burn-flex/src/ops/fft.rs:818-819`). Other
  lengths use Bluestein, which is correct but slower (`fft.rs:64-69, :213-258`).
- CubeCL backends support f32 and f64 only (`<B>/burn-cubecl/src/kernel/fft/base.rs:38-43`).
- On flex, `irfft_f64` casts to f32 and back (`<B>/burn-flex/src/ops/fft.rs:1549-1563`),
  although `rfft_f64` has its own f64 path (`:1018`). So f64 runs on flex are not true
  f64 through the spectral layers, and tests do not rely on them.

**Complex numbers.**
- There is no complex dtype (`<B>/burn-std/src/tensor/dtype.rs:10-26`), so complex
  weights stay as real/imaginary `Param` pairs.
- burn-store refuses complex tensors (`<B>/pytorch-reader/src/pickle_reader.rs:345-379`).
  This only matters for checkpoint import (Q8).

**Contractions.** `Tensor::einsum` and the `einsum!` macro are available, so Tucker, CP
and TT contractions are written as real einsums over the real and imaginary parts.

**Resampling.**
- `interpolate` is 2-D only (`<B>/burn-tensor/src/tensor/module.rs:509`).
- `align_corners` defaults to true (`<B>/burn-std/src/ops.rs:303-307`). That matches
  neuraloperator's explicit `align_corners=True` (`@2.0.0 layers/resample.py:50-52`).
- Bicubic uses a = −0.75, as torch does (`<B>/burn-flex/src/ops/interpolate.rs:517`).
- 1-D is done as 2-D with height 1. Three or more dimensions are spectral
  (`resample.py:54-69`) and reuse the FFT path.

**Normalisation layers.**
- `InstanceNorm`, `GroupNorm` and `BatchNorm` all use eps 1e-5
  (`<B>/burn-nn/src/modules/norm/instance.rs:15-22`, `group.rs:23`).
- `InstanceNorm` has an `affine` flag, defaulting to true. neuraloperator uses
  `F.instance_norm` without affine, so set it to false.
- `BatchNorm`'s `running_var` uses the **biased** batch variance
  (`<B>/burn-tensor/src/tensor/module.rs:69`), whereas torch uses the unbiased one. Eval
  mode diverges after training unless `layers/norm.rs` applies the n/(n−1) correction.

**Activation.** Exact GELU is `activation::gelu` (`<B>/burn-tensor/src/tensor/activation/base.rs:93`);
the tanh approximation is a separate function.

**Optimisers and schedulers.**
- Available: AdamW (`<B>/burn-optim/src/optim/adamw.rs:17`), StepLR
  (`lr_scheduler/step.rs:23`) and cosine (`cosine.rs:21`). ReduceLROnPlateau is
  **absent**.
- neuraloperator 2.0.0 steps every scheduler once per epoch (`trainer.py:310`).
- It steps ReduceLROnPlateau with the *summed* training error, not the evaluation loss
  (`trainer.py:301, 307-308, 314`).
- neuraloperator's AdamW treats a complex parameter as one number (#420). Real and
  imaginary pairs accumulate second moments per part, so training trajectories differ.
  §8.3 compares errors, not trajectories.

**Graph operations.**
- Available: `nonzero`, `cumsum`, `gather`, and `scatter` with `IndexingUpdateOp::Add`
  (`<B>/burn-tensor/src/tensor/api/base.rs:1901, :1969`; `<B>/burn-std/src/ops.rs:456-464`).
- GPU scatter-add with duplicate indices may be non-deterministic (`base.rs:1948-1952`).

**Spherical harmonics.** No SHT exists. `layers/spherical.rs` builds associated Legendre
matrices on the host in f64, and applies them with `einsum` after an `rfft` over
longitude. This targets torch_harmonics `RealSHT(norm="ortho", grid="equiangular")`
(`@2.0.0 layers/spherical_convolution.py:220-243`).

**`.pt` files.** burn-store contains a pickle reader. Whether it can read a plain dict
of tensors is unverified (T3.1). The fallback is a one-off conversion to `.npz`, which
the existing `NpzFileReader` reads (`data/io/readers/npz.rs:14`).

---

## 7. Phased implementation plan

Rules for every phase:
- One task is one branch, one worktree and one PR.
- "Files" lists every file a task edits.
- ∥ marks tasks that can run in parallel. They share no file, except where a shared
  file is named explicitly.
- `models/fno.rs` and `training/trainer.rs` are each edited by several tasks. Those tasks
  are **sequenced**, never run in parallel.
- New files need a `mod` line in their parent `mod.rs`. Tasks in the same phase that add
  `mod` lines to the same `mod.rs` merge cleanly when the lines are kept in alphabetical
  order. The phase README lists them.

### Phase 0: baseline and the one open FFT point

- **T0.1: PyTorch Darcy baseline (also the user's guided tour of neuraloperator).**
  - Files: §8.3 of this document only. The script stays outside the repo (A1).
  - Run `examples/models/plot_FNO_darcy.py` at 2.0.0 (lines 52-162) with five seeds:
    `FNO(n_modes=(8,8), hidden_channels=32, projection_channel_ratio=2)`, AdamW lr 8e-3,
    cosine with T_max 30, H1 loss, 20 epochs, n_train 1000, batch 32.
  - Record the final 16 and 32 L2/H1 errors, the parameter count and the hardware.
  - ∥ T0.2.
- **T0.2: `irfft` DC/Nyquist test.**
  - Files: `layers/spectral_convolution.rs` (tests module only). The CONVENTIONS §4 line
    comes from T0.4; this task fills in the test name.
  - Compare against `numpy.fft.irfft` values pasted into the test, at a power-of-two
    length, a non-power-of-two even length and an odd length.
  - ∥ T0.1.

- **T0.3: pin the default FNO's parameters.** Required before any Phase 1 code.
  - Files: a new test binary, `tests/fno_default_params.rs`. Each test binary is its own
    process, which matters because Flex's RNG is process-wide.
  - For a default `FNOConfig`, 1-D and 2-D, at a fixed seed, pin the bits of every
    parameter. Use either a by-hand replay of `FNOConfig::init`'s draw order (`fc0`, then
    per layer `conv[i]` re/im per corner, `w[i]`, then `fc1`, `fc2`) or checksums
    recorded at `478c2cb`.
  - Why it is needed: the existing tests cannot see a stray RNG draw at FNO level.
    `tests/spectral_init.rs` replays draws for a bare `SpectralConv` only, and the
    seeded FNO tests compare against the model's own weights.
  - ∥ T0.1, T0.2.
- **T0.4: CONVENTIONS version 2.** Lands the §2.6 diff, docs only, and needs the user's
  sign-off.
  - Files: `docs/CONVENTIONS.md`.
  - ∥ T0.1, T0.3. T0.2 adds its test name and result after T0.4 merges.

**Exit gate.**
- The baseline numbers are recorded.
- CONVENTIONS version 2 is merged, with the §4 open point closed by T0.2.
- The default-parameter pin (T0.3) is green.
- `cargo test` is green.

### Phase 1: FNO features

- **T1.1: SpectralConv additions.** Bias, the `NeuralopNormal` init,
  `modes_from_neuralop`, `clip_modes` and `separable`.
  - Files: `layers/spectral_convolution.rs` and `tests/spectral_init.rs` (new cases
    only). The CONVENTIONS §5 and §6 lines come from T0.4; this task confirms σ²/2.
  - Depends on: Phase 0. ∥ T1.2, T1.3, T1.5.
- **T1.2: new pointwise layers.** `ChannelMlp`, `Skip` and `BlockNorm`.
  - Files: `layers/channel_mlp.rs`, `layers/skip.rs`, `layers/norm.rs`, and
    `layers/mod.rs`.
  - ∥ T1.1, T1.3, T1.5.
- **T1.3: losses.** `H1Loss`, `FiniteDiff`, and the `LpLoss` options.
  - Files: `losses/h1.rs`, `losses/differentiation.rs`, `losses/data_losses.rs`,
    `losses/mod.rs`.
  - ∥ T1.1, T1.2, T1.5.
- **T1.4: FNO options.** Activation, lifting and projection, `out_channels`, symmetric
  padding, grid, stabiliser, and wiring in the channel MLP, skips, norms and bias.
  `FNOConfig::neuralop`.
  - Files: `models/fno.rs`, `training/trainer.rs` (`flatten_pair` for `out_channels`),
    the six `FNOConfig` struct literals (`fno.rs:314, :365, :471`;
    `examples/train/{burgers,darcy,burgers_learner}.rs`), which gain `..: None`. Also a
    PyTorch reference with `use_channel_mlp=False` and GELU (the double activation of
    §2.3).
  - Depends on: T1.1 and T1.2.
- **T1.5: normaliser `with_dims`.**
  - Files: `data/transforms/normalizers.rs`.
  - ∥ T1.1, T1.2, T1.3.

**Exit gate.**
- Every new option matches its reference (§8.2).
- All existing tests pass unmodified.
- An `FNOConfig::neuralop` model for the T0.1 configuration has the **same parameter
  count** as PyTorch's (`count_model_params` counts a complex parameter as 2).

### Phase 2: TFNO and incremental modes

- **T2.1: factorised weights.** Tucker, CP and TT, with both contractions.
  - Files: `layers/factorized.rs`, a hook in `layers/spectral_convolution.rs`, and
    `layers/mod.rs`.
  - Depends on: T1.1.
- **T2.2: `max_n_modes` / `set_modes`.**
  - Files: `layers/spectral_convolution.rs`.
  - Depends on: T2.1, which edits the same file, so the two are sequenced.
- **T2.3: TFNO and `max_modes` on `FNOConfig`.** Adds `FNOConfig::tfno`.
  - Files: `models/fno.rs`.
  - Depends on: T1.4, T2.1 and T2.2.

**Exit gate.** TFNO's parameter count equals PyTorch's for the T0.1 configuration with
`factorization="Tucker", rank=0.1`. Factorised layers match the dense contraction (§8.2).

### Phase 3: training stack, datasets, benchmark

- **T3.1: `.pt` reader.**
  - Files: `data/io/readers/pt.rs` and `data/io/readers/mod.rs`.
  - First check the pytorch-reader assumption (§6).
  - ∥ T3.2.
- **T3.2: optimiser and schedulers.** AdamW, StepLR, ReduceLROnPlateau, stepping per
  epoch, and the loss enum (H1).
  - Files: `training/trainer.rs`, `training/schedulers.rs` and `training/mod.rs`.
  - Depends on: T1.3. ∥ T3.1.
- **T3.3: PT loaders.** Darcy and Navier-Stokes.
  - Files: `data/loaders/pt_darcy.rs`, `data/loaders/pt_navier_stokes.rs`,
    `data/loaders/mod.rs` and `datasets/README.md`.
  - Depends on: T3.1 and T1.5.
- **T3.4: evaluation at several resolutions, and resuming training state.**
  - Files: `training/trainer.rs` and `training/checkpoint.rs`.
  - Depends on: T3.2, which edits the same file.
- **T3.5: incremental training.**
  - Files: `training/incremental.rs`.
  - Depends on: T3.4 and T2.2.
- **T3.6: Darcy benchmark example.**
  - Files: `examples/train/darcy_neuralop.rs`, `examples/predict/darcy_neuralop.rs`,
    and `Cargo.toml` (two `[[example]]` entries).
  - Depends on: T1.4, T3.3 and T3.4.

**Exit gate.** The §8.3 benchmark criterion holds, and resuming continues
bit-identically on flex.

### Phase 4: UNO and SFNO (two independent strands)

- **T4.1: `layers/resample.rs`.**
  - Depends on: Phase 1. ∥ T4.3.
- **T4.2: `models/uno.rs`.** Uses `SpectralConv::forward_with_shape`.
  - Files: `models/uno.rs`, `models/mod.rs`, and `layers/spectral_convolution.rs` (the
    `output_shape` path).
  - Depends on: T4.1, T2.2, and a decision on **Q7**.
- **T4.3: `layers/spherical.rs` (SHT and `SphericalConv`).**
  - ∥ T4.1.
- **T4.4: `models/sfno.rs`.**
  - Depends on: T4.3 and T1.4.

**Exit gate.** UNO and SFNO match their reference values (§8.2).

### Phase 5: geometry

- **T5.1: neighbour search.**
  - Files: `layers/gno/neighbor_search.rs` and `layers/gno/mod.rs`.
  - ∥ T5.2.
- **T5.2: `segment_csr`.**
  - Files: `layers/gno/segment.rs`.
  - ∥ T5.1.
- **T5.3: `IntegralTransform` and `GNOBlock`.**
  - Depends on: T5.1, T5.2 and T1.2.
- **T5.4: `models/gino.rs` (GINO and FNOGNO).**
  - Depends on: T5.3 and T1.4.
- **T5.5: car-cfd loader and example.**
  - Depends on: T5.4 and T3.1.

**Exit gate.** GINO trains on mini car-cfd and matches its reference values.

### Dependency summary

```
T0.1, T0.2, T0.3, T0.4  (T0.3 and T0.4 gate all of Phase 1)
  └─ T1.1 ─┬─ T1.4 ─ T2.3        T1.2 ─ T1.4        T1.3 ─ T3.2 ─ T3.4 ─ T3.5
           └─ T2.1 ─ T2.2 ─┬─ T2.3                   T1.5 ─ T3.3 ─ T3.6
                           └─ T4.2 (+ T4.1, Q7)      T3.1 ─ T3.3, T5.5
  T4.3 ─ T4.4      T5.1, T5.2 ─ T5.3 ─ T5.4 ─ T5.5
```

---

## 8. Validation and benchmarking

### 8.1 Test layers

1. **Defaults unchanged.**
   - T0.3's default-parameter pin stays green.
   - Existing tests pass with only `None` added to struct literals.
2. **Per feature, an independent reference in the test.** One of:
   - a formula or naive computation;
   - a brute-force loop (neighbour search);
   - the dense einsum (factorised weights);
   - small PyTorch reference values pasted in, as in
     `layers/spectral_convolution.rs:497`. The test comment records the neuraloperator
     version and the snippet that produced them.
3. **Properties:**
   - `set_modes` at the maximum reproduces the full layer;
   - the round trip of symmetric padding and cropping is exact;
   - old configs load (extends `fno.rs:443`);
   - resuming continues bit-identically.
4. **Backends.** Everything runs on flex. Spectral and norm changes are also run with
   `--features metal` where available, and the PR says whether they were.
5. **Benchmark** (§8.3).

### 8.2 Tolerances

All tolerances are f32.

| Check | Tolerance | Relative to |
|---|---|---|
| layer forward against pasted PyTorch values | ‖a − b‖_∞ ≤ 1e-5 · ‖b‖_∞ + 1e-7 | the pasted values |
| layer gradients | ≤ 1e-4 · ‖g‖_∞ + 1e-7 | PyTorch float64 gradients, pasted, or an analytic adjoint. Not f32 finite differences, whose best accuracy is about ε^(2/3) ≈ 2.4e-5 |
| spectral additions against a naive DFT in the test | ≤ 1e-4 · ‖b‖_∞ + 1e-7 | a DFT computed in f64 |
| factorised against dense | ≤ 1e-5 · ‖b‖_∞ + 1e-7 | dense einsum on the reconstructed weight |
| `FiniteDiff` on sin(2πx), periodic, s = 64 | ≤ 1e-5 relative L2 | the exact discrete derivative (sin(kh)/h)·cos(kx), k = 2π, h = 1/s |
| `FiniteDiff` against the analytic derivative (sanity check) | ≤ 2.5e-3 relative L2 | 2π cos(2πx); the stencil error is (kh)²/6 = 1.61e-3 |
| `FiniteDiff` third-order boundary on x³ | ≤ 1e-4 · ‖f′‖_∞ | analytic (exact for cubics) |
| SHT round trip, band-limited input | ≤ 1e-4 · max\|f\| | the input field |
| neighbour search | exact set equality | brute-force loop |

Why these values:
- 1e-5 is about 100 f32 ulps at unit scale. It catches the plausible bugs: a missing 1/n,
  a swapped corner, a transposed grid channel (all O(1)), and the wrong `FiniteDiff`
  spacing 1/(s − 1) (a 63/64 scaling at s = 64, i.e. 1.56e-2).
- Measured f32 FFT error in pocketfft is about 2e-7 at 64² to 128². Burn's error is
  checked by these tests.
- The absolute floor of 1e-7 avoids hiding errors in tensors much smaller than 1.

### 8.3 Darcy benchmark

**Setup.**
- Model: `FNOConfig::neuralop(&[8, 8], 1, 1, 32)` with projection hidden ⌊2·32⌋.
- Training: the T0.1 settings, five seeds.
- Metrics: final 16_l2, 16_h1, 32_l2 and 32_h1.

**Criterion.** For each metric separately:

```math
|\bar e_{\text{rs}} - \bar e_{\text{pt}}| \;\le\; \max\!\big(0.10\,\bar e_{\text{pt}},\; 3\,\hat\sigma_{\text{pooled}}\sqrt{2/5}\big)
```

Here ē is the five-seed mean and σ̂_pooled the pooled per-seed standard deviation. The
benchmark passes when all four metrics pass.
- The 3σ term keeps the false-failure rate low. By the numerics review's Monte Carlo
  estimate, a three-seed [min, max] band failed about 80% of the time with no bug.
- The 10% floor covers AdamW on real/imaginary pairs (§6) and differences in random
  number generators.

**Context only.** The dev docs report 16_l2 = 0.1329 and 32_l2 = 0.1616 at epoch 10
(https://neuraloperator.github.io/dev/auto_examples/models/plot_FNO_darcy.html). Those
were built from main with a different configuration, so they are not a target.

**Performance** (reported, not gated): seconds per epoch on flex and metal.

---

## 9. Risks, open questions and working with Claude Code

### 9.1 Risks and documented deviations

| Item | Handling |
|---|---|
| A new option accidentally changes a default, including a stray RNG draw | T0.3 pins every default parameter's bits; `None` branches make no RNG calls (§2.3) |
| `FNO::forward` becomes branch-heavy | options are grouped into small private helpers per block stage; T1.4's review checks readability |
| Complex `normal_` variance assumption | T1.1 checks it with one PyTorch run before it is written into CONVENTIONS §6 |
| `output_shape` resizing is not Fourier interpolation | Q7; only UNO depends on it |
| Mode clipping at odd extents differs from neuraloperator | documented; clipping tests use even extents |
| Gradient at x = y: sciml-rs gives 0, torch gives NaN (p > 1) | documented; tests avoid x = y |
| Burn `BatchNorm` running variance is biased | `layers/norm.rs` applies the correction (T1.2) |
| AdamW on real/imaginary pairs differs from complex AdamW | accepted; the benchmark compares errors |
| `.pt` dict unreadable by pytorch-reader | T3.1 falls back to a one-off conversion to `.npz` |
| Bluestein cost at 421² or on padded grids | measured; FFT size padding is a later objective |
| GPU scatter-add non-determinism | GNO tests on flex; tolerances on other backends |
| O(N·M) memory in neighbour search | same algorithm as neuraloperator's fallback; queries are chunked if needed |
| `partial_fit` std formula | not ported (PTDataset uses `fit`) |

### 9.2 Open questions

- **Q1 (decided 2026-10-05).** The spectral convolution follows main `00b7d86` (#688,
  #702). This needs no change to sciml-rs (§2.1).
- **Q2 (answered).** No Python or fixture files are committed. Reference values are
  pasted into the tests, and the benchmark baseline is recorded in §8.3.
- **Q3 (answered).** There is no separate model family and no prefix; features extend
  the existing modules.
- **Q7 (open).** Should `output_shape` copy neuraloperator's end-resize (§2.2), or do
  correct Fourier interpolation? *Recommendation:* copy it by default, for parity, and
  offer correct interpolation later as an opt-in option. This blocks T4.2 only.
- **Q8 (new).** Is importing neuraloperator checkpoints wanted later? It would need a
  name map and a centred-to-corner weight relayout. The relayout is well defined for even
  modes (§2.1); the review verified it. *Recommendation:* a separate later objective.

### 9.3 Conventions and sign-off

New options add lines to the existing CONVENTIONS sections, each stating the default
(= today) and the alternative:

| Section | Options it gains | Added by |
|---|---|---|
| §2 | the grid option | T1.4 |
| §3 | symmetric fractional padding | T1.4 |
| §4 | the DC/Nyquist result | T0.2 (test name) |
| §5 | bias, clipping, separable, factorised weights, `max_n_modes` | T1.1, T2.x |
| §6 | `NeuralopNormal` | T1.1 |
| §8 | `eps`/`measure`/H1 | T1.3 |

CONVENTIONS requires a bump for any change to §1–§6. So all of these lines land
together, with a single bump to version 2, in **T0.4**, before any code that relies on
them (§2.6). The table shows which task *implements and tests* each line. **T0.4 needs
the user's sign-off.**

**Self-contained once Phase 0 lands:** T1.1, T1.2, T1.3, T1.5, T3.1, T4.3, T5.1 and
T5.2. Run the `numerics` agent on T0.2, T1.1, T1.3, T2.1 and T4.3.

---

## 10. References

**neuraloperator.** Read at tag `2.0.0` (`b307030`), and at main `00b7d86` for #688
and #702, via `gh api` on 2026-10-05.
- Re-read directly:
  - `layers/spectral_convolution.py`: 296-370, 385-402, 420-528;
  - `models/fno.py`: 164-402;
  - `layers/fno_block.py`: 180-338;
  - `layers/channel_mlp.py` and `layers/skip_connections.py`;
  - `layers/embeddings.py`: 147-163, 340-383;
  - `layers/padding.py`: 75-128;
  - `losses/data_losses.py`: 68-216;
  - `layers/resample.py`: 36-70;
  - `layers/neighbor_search.py`: 84-120;
  - outlines of `models/uno.py`, `layers/integral_transform.py` and `models/gino.py`.
- The numerics review additionally read `data_losses.py:266-474`, `trainer.py`,
  `pt_dataset.py` and `utils.py`.
- **Unverified:** Zenodo file contents (the fetch failed with a TLS error); the dev-docs
  numbers (built from main); the complex `normal_` variance; the tltorch and
  tensorly versions that neuraloperator 2.0.0 installs. The CubeCL `irfft` DC/Nyquist
  behaviour was verified by reading the source (§2.1).

**neuraloperator project pages.**
- Releases: https://github.com/neuraloperator/neuraloperator/releases
- Docs: https://neuraloperator.github.io/dev/

**Burn.** Fork `ax1s-x1zz/burn` at `ddfa9af`, checked out locally under
`~/.cargo/git/checkouts/burn-4aa1ec9707f89bee/ddfa9af/`.

**sciml-rs.** Commit `478c2cb`: `docs/CONVENTIONS.md` (v1), `CLAUDE.md`, and the files
cited above.

**Papers** (cited from memory, unverified):
- Li et al., FNO, ICLR 2021.
- Kossaifi et al., Multi-Grid Tensorized FNO, TMLR 2024.
- Rahman et al., U-NO, TMLR 2023.
- Bonev et al., SFNO, ICML 2023.
- Li et al., GINO, NeurIPS 2023.
