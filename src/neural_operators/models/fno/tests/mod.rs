//! Shared test helpers for `FNO`: the reference forwards, input builders and the grid oracle.

mod checkpoint;
mod grid;
mod padding;
mod reference;

use super::*;
use crate::neural_operators::data::host_float::HostFloat;
use burn::tensor::Distribution;
use ndarray::{Array1, ArrayD, Dimension, IxDyn};

/// max |a - b| relative to max(1, max |b|).
fn rel_err(a: Tensor<1>, b: Tensor<1>) -> f32 {
    let scale = b.clone().abs().max().into_scalar::<f32>().max(1.0);
    (a - b).abs().max().into_scalar::<f32>() / scale
}

fn flat<const R: usize>(t: Tensor<R>) -> Tensor<1> {
    let n: usize = t.dims().iter().product();
    t.reshape([n])
}

/// f32, and each output is a sum of at most 128 O(1) products taken in a
/// different order than the reference, so ~1e-7 per term; 1e-5 leaves
/// headroom without hiding a wrong weight (which gives O(1) errors).
const TOL: f32 = 1e-5;

/// Where and how a reference forward pads the domain.
#[derive(Clone, Copy, Debug)]
enum Pad {
    /// No padding: the pre-#11 forward.
    Off,
    /// `p` zeros at the end of each spatial axis after `fc0`, keep the
    /// leading `n` (Li et al. `fourier_2d.py`; what `FNO::forward` does).
    End(usize),
    /// As `End`, but keep the trailing `n` after the layers: a crop
    /// misaligned with the pad.
    EndCropTail(usize),
    /// Zeros appended to the raw channels-last input before `fc0`, so the
    /// padded cells hold `fc0`'s bias rather than 0.
    BeforeFc0(usize),
    /// `⌊p/2⌋` zeros before and `⌈p/2⌉` after each axis, centre crop.
    Symmetric(usize),
}

/// `x` with `before` zeros prepended and `after` appended on `dim`, built
/// with `cat` so it is independent of `Tensor::pad`.
fn zero_extend<const R: usize>(x: Tensor<R>, dim: usize, before: usize, after: usize) -> Tensor<R> {
    let device = x.device();
    let block = |len: usize| {
        let mut dims = x.dims();
        dims[dim] = len;
        Tensor::<R>::zeros(dims, &device)
    };
    let mut parts = Vec::new();
    if before > 0 {
        parts.push(block(before));
    }
    parts.push(x.clone());
    if after > 0 {
        parts.push(block(after));
    }
    Tensor::cat(parts, dim)
}

/// The pre-4.2 style forward (channels-last `Linear`s, `Conv1d` layers)
/// with domain padding done by `cat`/`narrow` according to `pad`.
fn reference_forward_padded<const R: usize>(m: &FNO<R>, x: Tensor<R>, pad: Pad) -> Tensor<R> {
    let spatial_axes = 1..R - 1; // channels-last: [B, s.., C]
    let n_in: Vec<usize> = x.dims()[spatial_axes.clone()].to_vec();

    let mut x = x;
    if let Pad::BeforeFc0(p) = pad {
        for d in spatial_axes.clone() {
            x = zero_extend(x, d, 0, p);
        }
    }
    let x = m.fc0.forward(x);
    let mut x = to_channels_first(x); // [B, H, s..]
    for d in 2..R {
        x = match pad {
            Pad::End(p) | Pad::EndCropTail(p) => zero_extend(x, d, 0, p),
            Pad::Symmetric(p) => zero_extend(x, d, p / 2, p - p / 2),
            Pad::Off | Pad::BeforeFc0(_) => x,
        };
    }

    let n = m.conv.len();
    for idx in 0..n {
        let x1 = m.conv[idx].forward(x.clone());
        let dims = x.dims();
        let spatial: usize = dims[2..].iter().product();
        let x2 = m.w[idx]
            .forward(x.reshape([dims[0], dims[1], spatial]))
            .reshape(dims);
        x = if idx == n - 1 { x1 + x2 } else { relu(x1 + x2) };
    }

    for d in 2..R {
        let n_d = n_in[d - 2];
        x = match pad {
            Pad::Off => x,
            Pad::End(_) | Pad::BeforeFc0(_) => x.narrow(d, 0, n_d),
            Pad::EndCropTail(p) => x.narrow(d, p, n_d),
            Pad::Symmetric(p) => x.narrow(d, p / 2, n_d),
        };
    }

    let x = relu(m.fc1.forward(to_channels_last(x)));
    m.fc2.forward(x)
}

/// `FNO::forward` exactly as it was before padding existed.
fn pre_padding_forward<const R: usize>(m: &FNO<R>, x: Tensor<R>) -> Tensor<R> {
    let mut x = FNO::<R>::linear_cf(&m.fc0, to_channels_first(x));
    let n = m.conv.len();
    for idx in 0..n {
        let x1 = m.conv[idx].forward(x.clone());
        let x2 = FNO::<R>::conv1x1_cf(&m.w[idx], x);
        x = if idx == n - 1 { x1 + x2 } else { relu(x1 + x2) };
    }
    let x = relu(FNO::<R>::linear_cf(&m.fc1, x));
    let x = FNO::<R>::linear_cf(&m.fc2, x);
    to_channels_last(x)
}

/// Every parameter, by name, as raw f32 values.
fn params<const R: usize>(m: &FNO<R>) -> Vec<(String, Vec<f32>)> {
    use burn::store::{ModuleSnapshot, bridge::to_data};
    let mut out: Vec<_> = m
        .collect(None, None, false)
        .iter()
        .map(|t| {
            (
                t.name.clone(),
                to_data(t).unwrap().try_to_vec::<f32>().unwrap(),
            )
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// `to`'s parameters replaced by `from`'s, through a burnpack checkpoint.
fn copy_weights<const R: usize>(from: &FNO<R>, mut to: FNO<R>) -> FNO<R> {
    use burn::store::{BurnpackStore, ModuleSnapshot};
    let mut save = BurnpackStore::from_bytes(None);
    from.save_into(&mut save).expect("save weights");
    let bytes = save.get_bytes().expect("serialise weights");
    to.load_from(&mut BurnpackStore::from_bytes(Some(bytes)))
        .expect("load weights");
    to
}

fn bits<const R: usize>(t: Tensor<R>) -> Vec<u32> {
    t.into_data()
        .try_to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn test_config(modes: Vec<usize>, data_channels: usize, hidden: usize) -> FNOConfig {
    FNOConfig::new(modes, data_channels, 1)
        .with_hidden_channels(hidden)
        .with_n_layers(2)
}

fn random_input<const R: usize>(spatial: &[usize], channels: usize, device: &Device) -> Tensor<R> {
    let mut shape = vec![2];
    shape.extend_from_slice(spatial);
    shape.push(channels);
    let shape: [usize; R] = shape.try_into().unwrap();
    Tensor::<R>::random(shape, Distribution::Normal(0.0, 1.0), device)
}

/// `m` with `fc0`'s bias set to 1. The lifted field then sits ~1 away
/// from the zero padding, so a wrong convention (padding before `fc0`,
/// which fills the pad with this bias, or no padding at all) changes the
/// output by O(1) relative amounts instead of by the size of Linear's
/// small random default bias, which can fall below the controls' 1e-3.
fn with_unit_lift_bias<const R: usize>(mut m: FNO<R>) -> FNO<R> {
    use burn::module::Param;
    let bias = m.fc0.bias.take().expect("fc0 has a bias");
    let ones = bias.val().ones_like();
    m.fc0.bias = Some(Param::from_tensor(ones));
    m
}

/// 'ij'-indexed grids from per-axis coordinates: grid `a` has shape
/// `[axes[0].len(), ..]` and varies along axis `a` only.
fn ij_grid(axes: &[Array1<f64>]) -> Vec<ArrayD<f64>> {
    let sizes: Vec<usize> = axes.iter().map(|a| a.len()).collect();
    (0..axes.len())
        .map(|a| ArrayD::from_shape_fn(IxDyn(&sizes), |idx| axes[a][idx[a]]))
        .collect()
}

/// `linspace(0, 1, s_a)` per axis (`[0]` when s_a = 1), CONVENTIONS §2.
fn uniform_axes(spatial: &[usize]) -> Vec<Array1<f64>> {
    spatial
        .iter()
        .map(|&n| Array1::linspace(0.0, 1.0, n))
        .collect()
}

/// `axes` as grid channels in the model's order: 'ij' grids in reverse
/// axis order (CONVENTIONS §2; a no-op for a single axis).
fn xy_grid(axes: &[Array1<f64>]) -> Vec<ArrayD<f64>> {
    let mut grid = ij_grid(axes);
    grid.reverse();
    grid
}

/// Kept as the independent f64 oracle: `with_old_grid` feeds the reference forwards with it.
///
/// The grid the loaders used to store, built independently of the model
/// in f64: uniform over `[0, 1]` per axis, in the model's channel order.
fn old_grid_f64(spatial: &[usize]) -> Vec<ArrayD<f64>> {
    xy_grid(&uniform_axes(spatial))
}

/// Channels-last `x` with `grids` appended after the data, in f64, then
/// rounded to f32 once (`HostFloat::from_f64`). `x`'s own f32 values
/// round-trip exactly.
///
/// # Panics
/// If a grid's shape differs from `x`'s spatial shape.
fn with_grid<const R: usize>(x: Tensor<R>, grids: &[ArrayD<f64>]) -> Tensor<R> {
    use burn::tensor::TensorData;
    let dims = x.dims();
    let device = x.device();
    let vals: Vec<f64> = x.into_data().iter::<f32>().map(f64::from).collect();
    let data = ArrayD::from_shape_vec(IxDyn(&dims), vals).unwrap();
    let channels = dims[R - 1];
    for g in grids {
        assert_eq!(g.shape(), &dims[1..R - 1], "grid shape != spatial shape");
    }
    let mut shape = dims.to_vec();
    shape[R - 1] += grids.len();
    let out = ArrayD::from_shape_fn(IxDyn(&shape), |idx| {
        let idx = idx.slice();
        match idx[R - 1].checked_sub(channels) {
            None => data[idx],
            Some(g) => grids[g][&idx[1..R - 1]],
        }
    });
    let out: Vec<f32> = out
        .iter()
        .map(|&v| <f32 as HostFloat>::from_f64(v))
        .collect();
    Tensor::<R>::from_data(TensorData::new(out, shape), &device)
}

/// The pre-#13 model input for data-only `x`.
fn with_old_grid<const R: usize>(x: Tensor<R>) -> Tensor<R> {
    let spatial = x.dims()[1..R - 1].to_vec();
    with_grid(x, &old_grid_f64(&spatial))
}

/// `x` with the model's own grid appended (channels-last), for bitwise
/// comparisons with forwards that expect a grid-augmented input.
fn with_model_grid<const R: usize>(x: Tensor<R>) -> Tensor<R> {
    let grid = FNO::<R>::grid_cl(x.dims(), x.dtype(), &x.device());
    Tensor::cat(vec![x, grid], R - 1)
}

/// Input and parameter gradients of `L = Σ f(x0) · probe`, by name; the spectral
/// weights included.
// The reference differentiates w.r.t. the old grid-augmented input;
// only its data-channel slice is comparable to the new input grad.
fn grads_of(
    model: &FNO<4>,
    probe: &Tensor<4>,
    x0: Tensor<4>,
    f: &dyn Fn(Tensor<4>) -> Tensor<4>,
) -> Vec<(String, Tensor<1>)> {
    let x = x0.detach().require_grad();
    let grads = (f(x.clone()) * probe.clone()).sum().backward();
    let g_in = x.grad(&grads).expect("input grad").narrow(3, 0, 1);
    assert_eq!(g_in.dims(), [2, 8, 6, 1]);
    let mut out: Vec<(String, Tensor<1>)> = vec![("input".into(), flat(g_in))];
    for (name, l) in [
        ("fc0", &model.fc0),
        ("fc1", &model.fc1),
        ("fc2", &model.fc2),
    ] {
        out.push((
            format!("{name}.weight"),
            flat(l.weight.grad(&grads).unwrap()),
        ));
        let b = l.bias.as_ref().unwrap();
        out.push((format!("{name}.bias"), flat(b.grad(&grads).unwrap())));
    }
    for (i, w) in model.w.iter().enumerate() {
        out.push((
            format!("w[{i}].weight"),
            flat(w.weight.grad(&grads).unwrap()),
        ));
        let b = w.bias.as_ref().unwrap();
        out.push((format!("w[{i}].bias"), flat(b.grad(&grads).unwrap())));
    }
    for (i, layer) in model.conv.iter().enumerate() {
        for (k, (re, im)) in layer.corner_weights().into_iter().enumerate() {
            out.push((
                format!("conv[{i}].corner[{k}].re"),
                flat(re.grad(&grads).expect("spectral re grad")),
            ));
            out.push((
                format!("conv[{i}].corner[{k}].im"),
                flat(im.grad(&grads).expect("spectral im grad")),
            ));
        }
    }
    out
}
