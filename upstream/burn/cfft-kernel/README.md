# Fused `cat` ignores broadcasting on the vector axis (wrong `cfft` results on Metal)

## Summary

On the Metal backend, an elementwise op that broadcasts the output of `Tensor::cat`
along the vectorised (last) axis reads the wrong elements: each lane of the vector
steps along the `cat` input's real stride instead of repeating the broadcast value.
The result is silently wrong, with no panic or error.

`burn_signal::cfft` hits this whenever it falls back to Bluestein (non-power-of-two
length) along a non-last axis and the product of the trailing axes is even. There the
error is 20–85 % relative. That breaks the spectral layers of Fourier Neural Operators
on padded grids, which is how we found it.

The cause appears to be the `concat` codegen in `burn-cubecl-fusion`; see
[Root cause](#root-cause).

## Environment

| | |
|---|---|
| Burn | fork `ax1s-x1zz/burn` at `ddfa9af03ab9fb4b8491889e55c9d0ceffbfda22` (`0.22.0-pre.4`), features `autodiff, signal, store, train` + `metal` |
| CubeCL | `tracel-ai/cubecl` at `04676cd906552cb9a2b38e869d4f3c4064d73dec` (`0.11.0-pre.4`) |
| Backend | Metal (`Device::metal(Default::default())`), fusion enabled (default) |
| Reference backend | Flex (`Device::flex()`), correct |
| Hardware / OS | Apple M2 Pro, macOS 26.6.2 |
| Rust | 1.99.0 (b940084d7 2026-09-28) |

### Other backends

Only Metal has been run. From reading the source, CUDA (and wgpu/Vulkan) should hit
the same bug:

- **Fusion is on for CUDA by default.** `burn-cuda`'s default features include
  `fusion`, and `cargo tree --features cuda -i burn-cubecl-fusion` shows it linked via
  `burn-cubecl` → `burn-cuda` → `burn-core` → `burn`. Metal gets it the same way, through
  `burn-wgpu`'s default `fusion` feature.
- **The kernel is runtime-agnostic.** `concat` in
  `burn-cubecl-fusion/src/engine/codegen/kernel.rs` is one `#[cube]` function compiled for
  every CubeCL runtime. Neither it nor the fuser's `BaseOperationIr::Cat` arm
  (`engine/fuser.rs`) has a runtime-specific branch.
- **The vector widths are the same.** The fusion planner takes candidate widths from
  `ComputeClient::io_optimized_vector_sizes`, which works out to the load width (bits)
  divided by the element size (bits), capped at `max_vector_size`.

  | Runtime | `load_width` | `max_vector_size` | f32 candidates |
  |---|---|---|---|
  | CUDA (`cubecl-cuda/src/runtime.rs`) | 128 | `VectorSize::MAX` | 4, 2, 1 |
  | wgpu / Metal (`cubecl-wgpu/src/runtime.rs`) | 128 | 4 | 4, 2, 1 |

  So an even-width broadcast operand is vectorised by 2 or 4 on both.

Not ruled out: the planner's per-tensor choices (broadcast detection, multi-read
updates) were not traced in full, so CUDA could in principle pick width 1 for this
kernel. Nothing there depends on the runtime, though. CPU backends without CubeCL
fusion (Flex, NdArray) are unaffected.

## Minimal reproduction

```rust
use burn::prelude::*;

#[test]
fn repro_fused_cat_broadcast() {
    let device = Device::metal(Default::default());
    let a = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], &device);
    let p = Tensor::<2>::from_floats([[10.0], [20.0]], &device);
    let q = Tensor::<2>::from_floats([[30.0], [40.0]], &device);

    let out = a * Tensor::cat(vec![p, q], 0); // [4, 2] * [4, 1] -> [4, 2]
    let got = out.into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(got, [10.0, 20.0, 60.0, 80.0, 150.0, 180.0, 280.0, 320.0]);
}
```

### Expected

```
[10.0, 20.0, 60.0, 80.0, 150.0, 180.0, 280.0, 320.0]
```

(row `i` of `a` scaled by element `i` of `[10, 20, 30, 40]`). Flex gives this.

### Actual (Metal)

```
[10.0, 40.0, 60.0, 0.0, 150.0, 240.0, 280.0, 0.0]
```

Column 0 is correct; column 1 is wrong:

| row | expected | got | element read |
|---|---|---|---|
| 0 | `2 × 20` | `2 × 20 = 40` | `p[1]`, should be `p[0]` |
| 1 | `4 × 20` | `0` | `p[2]`, out of bounds |
| 2 | `6 × 30` | `6 × 40 = 240` | `q[1]`, should be `q[0]` |
| 3 | `8 × 40` | `0` | `q[2]`, out of bounds |

So lane `j` of row `i` reads element `i_local + j` of its `cat` input instead of
`i_local`. The broadcast stride is not zeroed.

The same happens with `+`, with the operands swapped (`cat(..) * a`) and with an
intermediate elementwise op (`a * (cat(..) + 0.0)`).

### Does not reproduce when

- the broadcast axis has odd length or length 1 (e.g. `a: [n, 7]`), so the kernel's
  vector width is 1;
- the `cat` output is explicitly expanded first: `a * cat(..).expand([4, 2])`;
- the `cat` output is materialised before the multiply (read to host and re-uploaded);
- the `[n, 1]` operand comes from `flip`, `narrow`, `reshape`, `swap_dims` or `+ 0.0`
  without a `cat`;
- the backend is Flex.

## Downstream impact: `burn_signal::cfft`

```rust
use burn::prelude::*;
use burn::tensor::{TensorData, signal};

#[test]
fn repro_cfft_bluestein() {
    let device = Device::metal(Default::default());
    let (n, w) = (6usize, 4usize);
    let vals: Vec<f32> = (0..n * w).map(|k| ((k * 7 % 11) as f32) - 5.0).collect();
    let re = Tensor::<2>::from_data(TensorData::new(vals.clone(), [n, w]), &device);
    let im = re.zeros_like();
    let (out_re, _) = signal::cfft(re, im, 0, Some(n));
    let got = out_re.into_data().try_to_vec::<f32>().unwrap();

    let mut max_err = 0.0f32;
    for k in 0..n {
        for c in 0..w {
            let want: f64 = (0..n)
                .map(|j| {
                    let t = 2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
                    vals[j * w + c] as f64 * t.cos()
                })
                .sum();
            max_err = max_err.max((got[k * w + c] - want as f32).abs());
        }
    }
    assert!(max_err < 1e-3, "max abs err {max_err}"); // Metal: 11.26
}
```

Relative L² error against a naive f64 DFT, `cfft` along axis 0 of an `[n, w]` tensor:

| shape | Flex | Metal |
|---|---|---|
| `[17, 8]` | 1.5e-7 | **8.2e-1** |
| `[17, 7]` | 1.6e-7 | 2.2e-7 |
| `[6, 4]` | 1.1e-7 | **5.4e-1** |
| `[16, 8]` (power of two, no Bluestein) | 8.6e-8 | 7.8e-8 |

Failing pattern: transform length not a power of two, transformed axis not the last,
product of the trailing axes even. `rfft`/`irfft` are affected through the same path.
Metal also gave different results for the same input across runs in some longer
chains.

Path inside `burn-signal` (`crates/burn-signal/src/functions/fft.rs`): `rfft` (non-power-of-two
length) → `bluestein_dft` → `cfft(b_re, b_im, dim, Some(m))` on the `[m, 1]` chirp kernel
→ `hermitian_extend` (`Tensor::cat` of the half-spectrum and its flipped mirror) →
`fa * fb` broadcasts that `[m, 1]` `cat` output against the `[m, w]` signal spectrum.

Bisection of `bluestein_dft` with host round-trips at each stage: the error goes away
**only** when `fb = cfft(b)` is materialised before `conv = fa * fb`. Every stage's
output is correct when read individually.

Confirmed as a fusion bug: with fusion disabled in a local copy of the fork
(`burn_cubecl::Cube = CubeBackend`), `repro_fused_cat_broadcast` and
`repro_cfft_bluestein` both pass on Metal. See also
[`../fused-cat-reused-at-new-shape/`](../fused-cat-reused-at-new-shape/README.md), a second
fused-`cat` bug.

## Root cause

`crates/burn-cubecl-fusion/src/engine/codegen/kernel.rs`, `fn concat`, branch
`comptime![dim != config.vector_axis]`:

```rust
let stride_tensor_vector =
    global_stride(inputs, comptime![config.vector_axis], pos_tensor);

#[unroll]
for i in 0..vector_size_ref {
    let input = read_input::<C, Const<1>>(
        inputs,
        &*locals,
        pos_tensor,
        index + i * stride_tensor_vector,
        LayoutInfo::IsRef,
        config,
        None,
    );
    result.insert(i, input.extract(0usize));
}
```

The `cat` inputs are `[2, 1]` with stride 1 on the vector axis (the last axis, size 1).
The reference layout is the broadcast `[4, 2]`, vectorised by 2. Lane `i` reads
`index + i * 1`, the next element of the input, where it should read `index` again,
since an axis of size 1 broadcast to 2 has an effective stride of 0. With vector width 1
(odd last axis) the loop only reads `i = 0`, which is why odd widths are unaffected.

The most recent commit touching this file is `3da67a217 perf(fusion): let a
concatenation choose its own layout (#5417)`. That commit is a candidate, not confirmed.

## Suggested fix

In the `dim != vector_axis` branch, treat the input's vector-axis stride as 0 when
its shape on that axis is 1 and the reference shape is larger, e.g.

```rust
let shape_tensor_vector = global_shape(inputs, comptime![config.vector_axis], pos_tensor);
let stride_tensor_vector = select(
    shape_tensor_vector == 1,
    0,
    global_stride(inputs, comptime![config.vector_axis], pos_tensor),
);
```

Or decline to fuse a `cat` whose inputs are broadcast on the vector axis. A regression
test along the lines of `repro_fused_cat_broadcast` above, for vector widths 2, 4 and
8, would cover it.

## Workaround

Until fixed, any of:

- Run each complex FFT on the last axis: swap the axis there, transform, swap back
  (below). This is what we use.
- Avoid even trailing widths on non-power-of-two `cfft` axes (for us: padded grid
  sizes whose `n_last / 2 + 1` is odd, or power-of-two lengths on the other axes).
- `.expand(..)` a `cat` output to the full shape before broadcasting it.
- Run on a non-fused backend (Flex), or build with fusion off (see "Confirmed as a
  fusion bug" above).

### What sciml-rs does

`SpectralConv::fft_ctensor` / `ifft_ctensor`
(`src/neural_operators/layers/spectral_convolution.rs`) run every complex FFT on the
last axis. With the transform axis last, Bluestein's `[.., m]` kernel spectrum is
concatenated along the vectorised axis itself, so the faulty `dim != vector_axis`
branch of `concat` is never taken:

```rust
for axis in (2..last).rev() {
    let (r, i) = signal::cfft(
        re.swap_dims(axis, last),
        im.swap_dims(axis, last),
        last,
        Some(dims[axis]),
    );
    re = r.swap_dims(axis, last);
    im = i.swap_dims(axis, last);
}
```

The inverse does the same around `icfft_full_spectrum`. The transform itself is
unchanged; only the memory layout around each FFT differs.

Results on Metal (relative error against Flex):

| Case | Before | After |
|---|---|---|
| `SpectralConv<5>` `[2, 6, 6, 8, 7]` | 6.7e-1 | 2.9e-7 |
| `SpectralConv<4>` `[2, 6, 17, 15]` | 8.7e-1 | 3.4e-7 |
| `SpectralConv<4>` 66 × 66 | 5.2e-1 | 7.6e-7 |
| `FNO<4>` 64², padding 2 / 4 / 8 / 16 | up to 9.5e-4 | ≤ 2.4e-7 |

The model tests that failed only on Metal (`padded_forward_matches_reference_3d`,
`grid_controls_3d_padded`, `padding_does_not_change_checkpoint`) pass.

Cost on Flex, one `cfft` along axis 2 in release mode:

| Shape | Old | New | Change |
|---|---|---|---|
| `[20, 32, 64, 33]` (power of two, no Bluestein) | 0.018 s | 0.022 s | 19 % slower |
| `[20, 32, 72, 37]` (Bluestein) | 0.339 s | 0.266 s | 22 % faster |

Regression test: `fft_correct_at_bluestein_sizes_with_even_trailing_width` checks
`fft_ctensor` against an f64 host DFT, and the round trip, at `[2, 3, 6, 7]` and
`[1, 2, 6, 5, 7]`. Those shapes hit the faulty case on every complex axis. It passes on
Flex and Metal. Whether it fails without the swap was not checked.
