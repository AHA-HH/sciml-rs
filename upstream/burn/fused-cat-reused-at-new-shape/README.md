# Fused `cat` chain gives wrong values when replayed at a new shape (Metal)

## Summary

On the Metal backend, with fusion enabled (the default), a short chain of ops that
includes a single-input `Tensor::cat` gives wrong values once it has already run at a
different shape in the same process. The first call at the new shape is correct.
From the second call on, one element is wrong, deterministically. No panic or
error is raised.

The same build with fusion disabled (`burn_cubecl::Cube = CubeBackend`) is correct.

We hit this in a Fourier Neural Operator's forward pass. The coordinate grid is built
with `arange → reshape → expand → cat` and concatenated onto the input. After a model
of one size had run, a model of another size gave 1–13 % relative error on every
forward after its first. That broke tests comparing two forwards of the same model,
which passed when run alone and failed after other tests.

Related to, but separate from, [`../cfft-kernel/`](../cfft-kernel/README.md). That
report is a wrong broadcast stride in the same fused `concat` codegen. This one needs
no broadcast on the vector axis and appears only on re-execution at a new shape.

## Environment

| | |
|---|---|
| Burn | fork `ax1s-x1zz/burn` at `ddfa9af03ab9fb4b8491889e55c9d0ceffbfda22` (`0.22.0-pre.4`), features `autodiff, signal, store, train` + `metal` |
| CubeCL | `tracel-ai/cubecl` at `04676cd906552cb9a2b38e869d4f3c4064d73dec` (`0.11.0-pre.4`) |
| Backend | Metal (`Device::metal(Default::default())`), fusion enabled (default) |
| Hardware / OS | Apple M2 Pro, macOS 26.6.2 |
| Rust | 1.99.0 (b940084d7 2026-09-28) |

Only Metal has been run. CUDA and wgpu/Vulkan use the same `burn-fusion` /
`burn-cubecl-fusion` path (see `../cfft-kernel/README.md`, "Other backends"), so they
may be affected too.

## Minimal reproduction

```rust
use burn::prelude::*;
use burn::tensor::TensorData;

#[test]
fn repro_fused_cat_reused_at_new_shape() {
    fn chain(x: Tensor<3>) -> Tensor<3> {
        let [b, n, _] = x.dims();
        let g = Tensor::<1, Int>::arange(0..n as i64, &x.device())
            .float()
            .reshape([1, n, 1])
            .expand([b, n, 1]);
        let g = Tensor::cat(vec![g], 2); // single-input cat
        Tensor::cat(vec![x, g], 2) // [b, n, 2]: (x, index) pairs
    }
    let device = Device::metal(Default::default());
    let upload = |b: usize, n: usize| {
        let vals: Vec<f32> = (0..b * n).map(|i| 10.0 * (i + 1) as f32).collect();
        Tensor::<3>::from_data(TensorData::new(vals, [b, n, 1]), &device)
    };

    // One run at [2, 21, 1].
    let _ = chain(upload(2, 21)).into_data();

    // Then three runs at [3, 8, 1], each with a freshly uploaded input.
    let want: Vec<f32> = (0..24)
        .flat_map(|i| [10.0 * (i + 1) as f32, (i % 8) as f32])
        .collect();
    for call in 0..3 {
        let got = chain(upload(3, 8)).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(got, want, "call {call}");
    }
}
```

### Expected

Every call returns `(x, index)` pairs, the index running `0..8` in each batch row:

```
[10, 0, 20, 1, 30, 2, 40, 3, 50, 4, 60, 5, 70, 6, 80, 7,      <- batch 0
 90, 0, 100, 1, 110, 2, 120, 3, 130, 4, 140, 5, 150, 6, 160, 7,  <- batch 1
 170, 0, 180, 1, 190, 2, 200, 3, 210, 4, 220, 5, 230, 6, 240, 7] <- batch 2
```

### Actual (Metal)

- Call 0: correct.
- Calls 1 and 2: identical to the expected output except one element. Batch 1, row 2,
  the index channel, gives `0.0` instead of `2.0`:

```
 90, 0, 100, 1, 110, 0, 120, 3, ...   <- batch 1 (`110, 0` should be `110, 2`)
```

`assertion left == right failed: call 1`. This exact code failed in 2 of 2 runs. An
equivalent variant with three prior runs failed in 4 of 4 separate processes, always
first at call 1.

### What it needs

| Change | Result |
|---|---|
| as above | **fails** on call 1 |
| no run at `[2, 21, 1]` first | passes |
| one prior run, or three | **fails** either way |
| drop the single-input `cat` (`let g = g;`) | passes |
| reuse one uploaded tensor (`x.clone()`) instead of a fresh `from_data` per call | passes |
| `.div_scalar((n - 1) as f64)` after `.float()` | **fails** (as in our model) |
| prior shape `[2, 5, 1]`, then `[1, 4, 1]` or `[2, 4, 1]` | passes |
| `burn_cubecl::Cube = CubeBackend` (fusion off), as above | passes |

The pattern looks like a fused execution plan being recorded at one shape and
replayed at another, with the fused `cat`'s per-input offsets or shapes not refreshed
for the new shape. The first call at a new shape runs while the plan is being found;
later calls replay it. This is an inference from the behaviour above; I have not
traced it in `burn-fusion`.

## Confirming it is fusion

`burn-cubecl/src/lib.rs` in a local copy of the fork:

```rust
#[cfg(feature = "fusion")]
pub type Cube = CubeBackend; // was burn_fusion::Fusion<CubeBackend>
```

with every `burn*` crate patched to that copy (`[patch."https://github.com/ax1s-x1zz/burn"]`).
The reproduction above then passes. The downstream test suite (198 tests, including the
ones this broke) passes on Metal in two runs. The fused-`cat` broadcast reproduction in
`../cfft-kernel/` passes too.

## Impact

Any model whose forward contains this pattern gives wrong results on Metal after a
differently shaped forward has run in the same process: a different batch size,
resolution or model size. Training and evaluation loops change shape routinely, for
example a last partial batch, or train and test batch sizes. The error is silent and
changes from one process to the next in a full model: 1e-2 to 1.3e-1 relative in our
1D FNO, against ~1e-7 without fusion.

## Workaround

- Avoid the single-input `cat`: when there is only one part, use it directly instead
  of `Tensor::cat(vec![t], dim)`. In the minimal case this is enough. Whether other
  `cat`s in a fused chain can trigger the same thing is not known.
- Run on a backend without fusion (Flex, NdArray), or build with fusion off as above.
  There is no runtime switch, and `burn-core` enables `burn-wgpu/default`, which
  includes `fusion`.

### What sciml-rs does

`FNO::grid_cl` (`src/neural_operators/models/fno.rs`) builds one coordinate channel
per spatial axis and concatenates them. In 1D there is one channel, which it now
returns directly:

```rust
// One channel (1D) is returned as is: a single-input `cat` of the
// expanded channel, followed by `forward`'s `cat`, gives wrong values
// on the fused GPU backends once the chain has run at another shape.
if channels.len() == 1 {
    return channels.pop().expect("one channel");
}
Tensor::cat(channels, R - 1)
```

The grid values, their order and dtype are unchanged, so saved checkpoints are
unaffected.

Results on Metal:

- A 1D FNO (modes 2, width 4, 1 layer) run after a differently shaped FNO: repeated
  forwards went from 1e-2 to 1.3e-1 relative error (against Flex) on every call after
  the first, to ~1e-7 on every call.
- `learner::tests::inference_step_reports_per_sample_relative_l2` and
  `train_step_gradients_match_hand_written_loop`, which compare two forwards of one
  model, pass in two full-suite runs. Before the fix they failed in most full-suite
  runs (not every one), and only when earlier 1D model tests had run in the same
  process; alone they passed.

### 2D: a replay check, not a fix

In 2D and 3D the grid still concatenates two or three broadcast channels, then
`forward` concatenates that onto the input. That pattern did not trigger the bug in our
tests. To check it under training-like conditions, `replay_2d_poisson` runs:

- the configuration of our Poisson trainer (`FNO<4>`, modes 12 × 12, width 32, 4 layers,
  64² grid), with padding 0, 4, 8 and 16, all in one process;
- eight calls per padding, in the order train 20, train 20, train 7, eval 20, eval 3,
  train 20, train 7, eval 20. Each input is uploaded afresh, and train calls run forward
  plus backward on an autodiff device;
- every call compared against the same weights and inputs on Flex: the output, or the
  loss and the gradient with respect to the input.

Worst relative error over all 32 calls: 2.33e-6, the same in two runs, with no growth
on repeated calls. It did not check the optimiser update, real data loading, or long
runs.

### A related, benign difference

`trainer::tests::device_loaders_train_identically_to_host_loaders` trains the same model
twice: once with device-resident data loaders, once with host loaders. It asserted that
the trained models' outputs were bit-identical. On Metal with fusion on, one value
differed by 2 ulp, after every epoch metric had matched exactly. With fusion off it is
bit-identical. This looks like the fuser compiling the two runs' identical op sequences
into different kernels. That is a rounding difference, not a wrong value. The test now
stays bit-exact on Flex and allows 1e-6 relative on other backends.
