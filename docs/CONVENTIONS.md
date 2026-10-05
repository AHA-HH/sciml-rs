# Conventions

`CONVENTION_VERSION = 1`.

This file is the single source of truth for tensor layouts, coordinate grids, padding,
Fourier transforms, mode truncation and weight initialisation in `sciml-rs`, and records
the data normalisation and losses the training code uses. Code cites it as
`CONVENTIONS §n`. Any change to §1–§6 bumps `CONVENTION_VERSION`, which can make a saved
checkpoint load but predict differently. §10 and §11 describe preprocessing and losses; a
change to them does not bump the version (§9).

Summary of the choices: channels-last at the model boundary and channels-first inside;
the model builds its own coordinate channels on the closed [0, 1], appended after the
data in reverse axis order; one-sided zero padding at the end of every axis; NumPy's
`norm="backward"` FFTs with a real transform on the last spatial axis; 2^(D−1) corner
blocks of retained modes, each with its own complex weight stored as two real tensors;
and Li et al.'s U(0, 1/(I·O)) initialisation by default.

Version 1 introduces nothing new: it records the code as of commit `f1ab313`. Every
statement below was checked against that code and the pinned Burn fork while drafting.
The tests named under each section's Verification are what confirm it; no separate
numerical check was run.

Notation: B batch, D spatial dimensions, R = D + 2 tensor rank, s_a the extent of spatial
axis a (a = 1..D), C data channels, H hidden channels, I and O the input and output
channels of a layer, m_a the retained modes on axis a.

## 1. Tensor layouts and data flow

`FNO::forward` maps channels-last `[B, s_1, .., s_D, C]` to `[B, s_1, .., s_D, 1]`.
Inside the model everything is channels-first, `[B, H, s_1, .., s_D]`: the input is
permuted once, while it is C + D wide, and the output once, while it is one channel wide.

```text
x ──append grid (§2)──► [B, s.., C+D] ──permute──► fc0 (C+D → H) ──pad (§3)──►
  n_layers × { v ← σ(K_ℓ v + W_ℓ v) }   (σ = ReLU, none after the last layer)
──crop──► fc1 (H → 128) ──ReLU──► fc2 (128 → 1) ──permute──► [B, s.., 1]
```

- K_ℓ is `SpectralConv<R>` (§4, §5), channels-first `[B, I, s..]` → `[B, O, s..]`;
  W_ℓ is the 1×1 convolution `w[ℓ]`.
- Pointwise layers (`fc0`, `w`, `fc1`, `fc2`) are applied as one batched matmul along the
  channel axis (`FNO::pointwise`), with the layers' own parameters. `Linear` stores
  `[d_in, d_out]`; `Conv1d` stores `[O, I, 1]`.
- Defaults: H = 32, n_layers = 4. `out_channels` must be 1 (asserted in
  `FNOConfig::init`), and `modes.len() + 2 == R`.
- Configs are Burn `Config`s saved as JSON. Fields added after release are `Option<_>`
  (`spectral_init`, `padding`), so older configs load them as `None`.

### Verification
`forward_shape_{1d,2d,3d}`, `forward_matches_reference_*` and `gradients_match_reference`
(against a plain reference forward in the test), `lift_width_includes_grid_channels`,
`config_without_spectral_init_loads_as_default`, `checkpoint_round_trip_matches_old_forward`.

## 2. Coordinate grid channels

The model generates its coordinate channels (`FNO::grid_cl`); inputs carry data channels
only, and `forward` panics if the last axis is not C wide. Along axis a,

```math
x_a[j] = \frac{j}{s_a - 1}, \qquad j = 0, \dots, s_a - 1, \qquad x_a[0] = 0 \text{ if } s_a = 1
```

- The D channels are appended **after** the data in **reverse axis order**:
  `[data, x_D, .., x_1]`, where x_a varies along axis a only. D = 1 gives `[data, x]`;
  D = 2 is `np.meshgrid` 'xy' order, as in Li et al.'s 2D script. `fc0`'s grid rows are
  trained against this order.
- The grid is built channels-last, in the input's dtype on its device, for the input's
  own resolution, and is a constant (not tracked by autodiff). Building it channels-first
  after the permute zeroed entries on Metal.
- Host-side `data::grids` helpers produce 'ij'-indexed grids for data preparation; the
  model does not use them.

### Verification
`grid_2d_order_is_pinned`, `grid_controls_{1d,2d,3d,3d_padded}`,
`grid_matches_old_loader_grid`, `resolution_change_builds_that_grid`,
`batch_rows_see_identical_grids`; host helpers in `data::grids` tests
(`reversed_uniform_grid_2d_matches_numpy_meshgrid`).

## 3. Domain padding

`FNOConfig::padding = Some(p)` appends p zeros at the **end** of every spatial axis after
`fc0` and crops back to s_a after the last spectral layer, before `fc1` (Li et al.'s
one-sided `F.pad(x, [0, p, 0, p])`).

- `None` and `Some(0)` mean no padding. The padded extent s_a + p is used as is (no
  rounding to a power of two), and §5's mode limits apply to it.
- Every axis gets the same p, also in 3D (Li et al.'s `fourier_3d.py` pads only time).
- Padding is not a parameter and is not stored in the weights.
- On the padded grid every layer is circular-shift-equivariant, so symmetric padding
  with a centre crop equals end padding shifted back; only the crop alignment matters.

### Verification
`padding_zero_is_bitwise_identical`, `padded_forward_matches_reference_{1d,2d,3d}`,
`padded_gradients_match_reference`, `padded_input_gradient_finite_difference`,
`symmetric_padding_with_centre_crop_matches_end_padding`, `pad_crop_helpers_exact`,
`padding_adds_no_parameters`, `padding_does_not_change_checkpoint`.

## 4. Fourier transforms and normalisation

NumPy's `norm="backward"`: forward unscaled, inverse scaled by 1/N (Burn fork,
`burn-signal/src/functions/fft.rs`):

```math
X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-2\pi i k n / N}, \qquad
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k]\, e^{2\pi i k n / N}
```

- Forward (`SpectralConv::fft_ctensor`): `signal::rfft` along the **last** spatial axis
  (s_D → s_D/2 + 1 bins), then `signal::cfft` along the other spatial axes, from tensor
  axis R − 2 down to axis 2.
- Inverse (`ifft_ctensor`): `icfft_full_spectrum` along axes 2..R − 2, then
  `signal::irfft` along the last axis with output length s_D, so odd lengths reconstruct
  correctly.
- Burn has no inverse complex FFT. `icfft_full_spectrum` uses

  ```math
  \mathcal{F}^{-1}(X) = \frac{1}{N}\, \overline{\mathcal{F}\big(\overline{X}\big)}
  ```

  and is valid for non-Hermitian spectra.
- Any N is allowed; non-powers of two go through Bluestein's algorithm.
- Not yet fixed by a test: how `irfft` treats a nonzero imaginary part in the DC bin and,
  for even s_D, in the Nyquist bin s_D/2. A design that relies on it adds a test and
  records the result here.

### Verification
`fft_round_trip_{1d,2d,3d}`, `round_trips_an_asymmetric_signal`,
`round_trips_a_short_asymmetric_signal`.

## 5. Spectral convolution layout

For axes a < D the retained frequencies are a low and a high (negative-frequency) block;
on the last axis only the low block of the half spectrum:

```math
\mathcal{K}_a = \{0, \dots, m_a - 1\} \cup \{s_a - m_a, \dots, s_a - 1\} \ (a < D), \qquad
\mathcal{K}_D = \{0, \dots, m_D - 1\}
```

- This gives 2^(D−1) **corners**. In corner index `mask`, bit j (spatial axes counted
  from 0) selects the low (0) or high (1) block on axis j; the last axis is always low
  (`SpectralConv::corner_ranges`).
- Limits, asserted in `forward` (`check_modes_fit`): m_a ≤ s_a/2 for a < D, so the
  blocks never overlap, and m_D ≤ s_D/2 + 1.
- Weights: per corner, real and imaginary parameters `weights_re[corner]`,
  `weights_im[corner]`, each `[I, O, m_1, .., m_D]`, indexed by the position k' of the
  frequency k within its corner. Coefficients outside the corners are zero.
- Mode mixing, Li et al.'s `einsum("bix,iox->box")`:

  ```math
  \hat y_{b,o}(\mathbf{k}) = \sum_{i} \hat x_{b,i}(\mathbf{k})\, W^{c}_{i,o}(\mathbf{k}'),
  \qquad \mathbf{k} \in \text{corner } c
  ```

  computed in `complex_multiplication` with three real matmuls (Gauss's trick):
  for (a + bi)(c + di), Re = ac − bd and Im = (a + b)(c + d) − ac − bd.

### Verification
`corner_ranges_2d_matches_manual_indices`, `corner_ranges_3d_covers_all_four_combinations`,
`complex_multiplication_matches_python_einsum`, `forward_2d_matches_pytorch_reference`,
`forward_3d_matches_numpy_reference`, `check_modes_fit_limits_odd_and_even`,
`forward_at_mode_limit_is_identity_*`, `gradients_flow_{1d,2d,3d}`.

## 6. Spectral weight initialisation

`SpectralInit`, chosen by `FNOConfig::spectral_init` (`None` means `LiUniform`). The real
and imaginary parts of every weight are drawn independently from the same distribution:

| Variant | Distribution (per part) |
| --- | --- |
| `LiUniform` (default) | U(0, 1/(I·O)), as Li et al.'s `scale * torch.rand` |
| `Normal` | N(0, σ²), σ = 1/√(I·O) |
| `SymmetricUniform` | U(−1/√I, 1/√I) |

The draw order is per corner, real then imaginary. With a fixed seed it reproduces
earlier weights bit for bit, so it is part of this convention.

### Verification
`tests/spectral_init.rs` (own binary, reseeds Flex's global RNG):
`default_spectral_init_is_bit_identical`, `init_moments_and_support_match_formulas`,
`init_normal_has_gaussian_shape`, `init_parts_and_corners_are_independent`; plus
`spectral_init_reaches_every_layer`, `init_default_is_li_uniform`.

## 7. Identities the tests must confirm

- Transform round trips: `irfft ∘ rfft` and `icfft_full_spectrum ∘ cfft` are the identity
  (§4).
- Full retention: with I = O = 1, unit weights in every corner and m_a at its limit on
  every axis, `SpectralConv` is the identity, provided non-last extents are even (an odd
  extent drops its middle frequency).
- Padding off: `Some(0)` gives bit-identical output to `None` (§3).
- Shift equivariance on the padded grid (§3).
- Exact match: both losses of §11 are 0 with gradient 0.

A new fast path is tested against a slower trusted one (a reference forward, an einsum, or
a published PyTorch/NumPy value), with its tolerance stated relative to that reference.

## 8. Precision and range

- Model tensors use Burn's default float, f32; host-side reading and normalisation are
  f64, and datasets are cast to the storage type after normalisation.
- The identities of §7 hold to 1e-4 absolute in f32 at the tested sizes.
- Results must hold on the default `flex` backend. Other backends are not bit-identical
  to it: the generated grid may differ by up to 1 ulp (division rounding).
- `grid_cl` is bit-identical to the old f64 `linspace` rounded to f32 while s_a − 1 is
  exact in f32 (s_a ≤ 2^24).

## 9. Versioning

`CONVENTION_VERSION = 1`. Any change to §1–§6 bumps it: those sections fix parameter
names and shapes, the grid-channel order, the spectral layout and the initialisation draw
order, which saved checkpoints depend on. A design document that needs a new convention
proposes it as a diff to this file, and the change lands here before any code that
relies on it. §7 and §8 change only with the tests they describe. §10 and §11 do not bump
the version; a change to them is noted in the pull request and changes the code that
implements it in the same pull request.

## 10. Data normalisation

```math
\text{encode}(x) = \frac{x - \mu}{\sigma + \varepsilon}, \qquad
\text{decode}(z) = z\,(\sigma + \varepsilon) + \mu
```

- `UnitGaussianNormalizer`: μ and σ per spatial point across the sample axis, σ with
  ddof = 1; ε = 1e-5 by default (`with_eps` to change). Needs at least 2 samples.
- `GaussianNormalizer`: one scalar μ and σ over all values; ε = 1e-5, fixed.
- `RangeNormalizer`: pointwise min/max map to [low, high] (default [0, 1]); a constant
  point gets scale 1 and is shifted to the midpoint.
- Darcy (`DarcyNormalizers`) normalises x and y pointwise; statistics are fitted on the
  training split only, in f64 on the host. `x_test` is encoded with the training x
  statistics; `y_test` stays in physical units, and predictions are decoded with the y
  normaliser before the test loss (as in Li et al.). Burgers does not normalise.
- Subsampling keeps every `rate`-th point along an axis (`transforms::subsample`).

### Verification
`encode_decode_round_trips`, `unit_gaussian_two_samples_uses_sample_std`,
`gaussian_accepts_one_sample_with_several_values`,
`range_constant_points_encode_to_midpoint_and_round_trip`,
`flat_decoder_matches_ndarray_decode`, `inputs_hold_the_normalized_coefficient_only`,
`single_train_sample_is_invalid_not_nan`.

## 11. Losses

```math
\mathrm{rel}(x, y) = \frac{\lVert x - y \rVert_p}{\lVert y \rVert_p}, \qquad
\mathrm{abs}(x, y) = \Big(\prod_{a=1}^{D} h_a^{1/p}\Big) \lVert x - y \rVert_p, \quad
h_a = \frac{1}{s_a - 1}
```

- Per example, then reduced by `Reduction::{Mean, Sum, None}`.
- `LpLoss::rel` takes flattened `[B, n_points]`; undefined (inf or NaN) for an all-zero
  target.
- `LpLoss::abs` takes unflattened `[B, s_1, .., s_D]`; every s_a ≥ 2.
- At an exact match both return 0 with gradient 0 (the `torch.norm` subgradient).

### Verification
`test_rel_known_value`, `test_abs_known_value`, `abs_2d_non_square_known_value`,
`abs_3d_known_value`, `abs_gradient_matches_analytic`,
`rel_gradient_at_exact_match_is_zero_not_nan`, `abs_gradient_at_exact_match_is_zero_not_nan`,
`rel_zero_target_is_inf_or_nan_as_documented`.
