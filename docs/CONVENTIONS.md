# Conventions

`CONVENTION_VERSION = 1`.

This file is the single source of truth for tensor layouts, FFT normalisation, mode
truncation, coordinate grids, padding, initialisation and normalisation in `sciml-rs`.
Code cites it as `CONVENTIONS §n`. Version 1 records what the code already does as of
commit `1208638`; it introduces nothing new.

Any change to §1–§6 bumps `CONVENTION_VERSION`. Those sections fix what saved
checkpoints and trained weights depend on (parameter shapes, the order of grid
channels, the spectral layout), so a change there can make an old checkpoint load but
predict differently. §7 and §8 describe preprocessing and losses; a change to them
needs a note in the PR but no version bump.

A design document that needs a new convention proposes it as a diff to this file in its
§2, and the change lands in this file before any code that relies on it.

Notation: B batch, D spatial dimension, R = D + 2 tensor rank, s_a the extent of
spatial axis a (a = 1..D), C data channels, H hidden channels.

## 1. Tensor layouts

- Model input and output are **channels-last**: `FNO::forward` takes
  `[B, s_1, .., s_D, C]` and returns `[B, s_1, .., s_D, out_channels]`.
- Inside the model everything is **channels-first**, `[B, H, s_1, .., s_D]`. The input
  is permuted once, while it is only C + D wide; the output once, while it is one
  channel wide.
- `SpectralConv<R>` takes and returns channels-first `[B, I, s..]` → `[B, O, s..]`.
- Pointwise layers (`fc0`, the `w` 1×1 convolutions, `fc1`, `fc2`) are applied as one
  batched matmul along the channel axis (`FNO::pointwise`), using the layers' own
  parameters. `Linear` stores `[d_in, d_out]`; `Conv1d` stores `[O, I, 1]`.
- `out_channels` is 1 (asserted in `FNOConfig::init`).
- Losses take flattened `[B, n_points]` (`LpLoss::rel`) or unflattened `[B, s_1, .., s_D]`
  (`LpLoss::abs`).

## 2. Coordinate grid channels

- The model generates its own coordinate channels (`FNO::grid_cl`); inputs carry data
  channels only, and `forward` panics if the last axis is not `data_channels` wide.
- Each coordinate is uniform over the closed interval [0, 1]: `arange(n) / (n − 1)`, and
  a single point is 0.
- The D coordinate channels are appended **after** the data, one per spatial axis, in
  **reverse axis order**: `[data, x(s_D), .., x(s_1)]`, where `x(s_a)` varies along axis
  a only. For D = 2 this is `np.meshgrid` 'xy' order, as in Li et al.'s 2D script; for
  D = 1 it is `[data, x]`. `fc0`'s grid rows are trained against this order.
- The grid is built in the input's dtype on its device, and is a constant (no autodiff).
- Host-side `data::grids` helpers produce 'ij'-indexed grids and offer both
  `[grids.., data..]` and `[data.., grids..]` placement. They are for data preparation,
  not the model.

## 3. Domain padding

- `FNOConfig::padding = Some(p)` appends p zero cells at the **end** of every spatial
  axis after `fc0`, and crops back to the original extent after the last spectral
  layer, before `fc1` (Li et al.'s one-sided `F.pad(x, [0, p, 0, p])`).
- `None` and `Some(0)` mean no padding. The padded extent s_a + p is used as is (no
  rounding up to a power of two), and `modes` are checked against the padded extent.
- Every axis gets the same p, also in 3D (unlike Li et al.'s `fourier_3d.py`).

## 4. FFT and normalisation

- Forward transform of a channels-first tensor (`SpectralConv::fft_ctensor`):
  `signal::rfft` along the **last** spatial axis (length s_D, giving s_D/2 + 1 bins),
  then `signal::cfft` along the remaining spatial axes, from axis R − 2 down to axis 2.
- Inverse (`ifft_ctensor`): `icfft_full_spectrum` along axes 2..R − 2, then
  `signal::irfft` along the last axis with output length s_D (the original length, so
  odd lengths reconstruct correctly).
- Normalisation is NumPy's `norm="backward"`: forward transforms are unscaled, inverse
  transforms scale by 1/n. `icfft_full_spectrum` computes `conj(cfft(conj(x))) / n`.
  `irfft ∘ rfft` is the identity to 1e-4 in f32 (tests `fft_round_trip_*`).
- Burn has no native inverse complex FFT; `icfft_full_spectrum` is the only one used,
  and it is valid for non-Hermitian spectra (test `round_trips_an_asymmetric_signal`).
- Any length is allowed; non-powers of two go through Bluestein's algorithm.
- Not yet fixed by a test: how `irfft` treats a nonzero imaginary part in the DC bin and,
  for even s_D, the Nyquist bin s_D/2 of the last axis. A design that relies on this
  must add a test and record the result here.

## 5. Spectral convolution layout

- `modes[a]` is the number of retained modes on spatial axis a; `modes.len() == D`.
- Retained frequencies: on the last axis, bins `0..modes[D]` of the half spectrum. On
  every other axis a, two blocks: the low block `0..modes[a]` and the high (negative
  frequency) block `s_a − modes[a] .. s_a`.
- This gives 2^(D−1) **corners**. Corner index `mask` selects, for each non-last axis
  a (bit a of `mask`, axes counted from 0), the low block (bit 0) or the high block
  (bit 1) (`SpectralConv::corner_ranges`). For D = 2, corner 0 is (low, ·), corner 1
  (high, ·).
- Limits, asserted in `forward` (`check_modes_fit`): `modes[D] ≤ s_D/2 + 1` on the
  last axis, and `modes[a] ≤ s_a/2` elsewhere, so low and high blocks never overlap.
- Weights: per corner, separate real and imaginary tensors
  `weights_re[corner]`, `weights_im[corner]`, each `[I, O, modes..]`. Output
  coefficients outside the retained corners are zero.
- Mode mixing is `out[b, o, k] = Σ_i x[b, i, k] · w[i, o, k]` per retained frequency k
  (Li et al.'s `einsum("bix,iox->box")`), computed with three real matmuls
  (Gauss's trick) in `complex_multiplication`.

## 6. Spectral weight initialisation

`SpectralInit`, chosen by `FNOConfig::spectral_init` (`None` means `LiUniform`). The
real and imaginary parts of every corner are drawn independently from:

| Variant | Distribution |
|---|---|
| `LiUniform` (default) | U(0, 1/(I·O)), as Li et al.'s `scale * torch.rand` |
| `Normal` | N(0, σ²) with σ = √(1/(I·O)) |
| `SymmetricUniform` | U(−1/√I, 1/√I) |

## 7. Data normalisation

- `UnitGaussianNormalizer`: pointwise. Mean and std (ddof = 1) per spatial point across
  the sample axis; `encode(x) = (x − mean)/(std + eps)` and `decode` the inverse.
  Default `eps = 1e-5`, configurable with `with_eps`. Needs at least 2 samples.
- `GaussianNormalizer`: one scalar mean and std over all values; `eps = 1e-5` fixed.
- Where a loader normalises (Darcy: `DarcyNormalizers`, both x and y pointwise),
  statistics are fitted on the training split only, in f64 on the host. `x_test` is
  encoded with the training `x` statistics, but `y_test` stays in physical units:
  predictions are decoded with the `y` normaliser before the test loss (as in Li et
  al.). The Burgers loader does not normalise.
- Subsampling keeps every `rate`-th point along an axis (`transforms::subsample`).

## 8. Losses

- `LpLoss::rel(x, y) = ‖x − y‖_p / ‖y‖_p` per example over `[B, n_points]`, then reduced
  by `Reduction::{Mean, Sum, None}`. Undefined (inf or NaN) for an all-zero target.
- `LpLoss::abs` weights by the uniform-mesh quadrature factor Π_a h_a^(1/p) with
  h_a = 1/(s_a − 1), and needs the unflattened `[B, s_1, .., s_D]` shape.
- At an exact match both return 0 with gradient 0 (the `torch.norm` subgradient).
