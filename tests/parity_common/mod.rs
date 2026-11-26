//! The parity harness (library CM0.5): loads a `neuraloperator` 2.0.0 fixture
//! (`tests/fixtures/parity/`, written by `tools/parity/generate.py`) into a Burn module by a
//! hand-written parameter-name map, and compares tensors against the fixture's within D3.
//!
//! A fixture's `.npz` holds `input`, `output` and every tensor of the 2.0.0 model's
//! `state_dict` as `param.<2.0.0 name>`; a complex tensor is stored as a trailing axis of 2,
//! (real, imaginary). Each map entry names one such array, the Burn path it goes to (as
//! `ModuleSnapshot::collect` names it) and which part of it to take.
//!
//! A parameter whose layout differs from the crate's is converted when the fixture is
//! loaded, into `Fixture::replaced` (2.0.0's centred spectral weights: CM1.2's
//! `centred_conv.rs`, D5).
//!
//! The reference models beside this file (`fno1d.rs`, `legacy_conv.rs`) are included only by
//! the test binaries that use them, with `#[path]`, so no binary compiles dead code. The map
//! pieces two of them share (the FNO's block parts, lifting and projection, 2.0.0's centred
//! spectral weights as corners) are here, with the harness (library CM1.5a).

use std::{collections::BTreeMap, path::Path};

use burn::{
    prelude::*,
    store::{ModuleSnapshot, PathFilter, bridge},
    tensor::TensorData,
};
use ndarray::{ArrayD, Axis, Slice, concatenate};
use ndarray_npy::NpzReader;
use sciml_rs::neural_operators::{
    data::io::readers::npz::NpzFileReader,
    layers::{norm::NormKind, skip::SkipKind},
    models::fno::FNOConfig,
};

/// D3's bound in f32: relative max error ≤ 1e-5 (roadmap, signed off 2026-11-25).
pub const D3_F32: f64 = 1e-5;
/// D3's bound in f64: relative max error ≤ 1e-12.
pub const D3_F64: f64 = 1e-12;

/// The prefix of a parameter array's name in a fixture.
const PARAM_PREFIX: &str = "param.";

/// Which part of a fixture array a map entry takes.
#[derive(Clone, Copy, Debug)]
#[allow(
    dead_code,
    reason = "every parity binary includes this harness; each maps with only some variants"
)]
pub enum Take {
    /// The array as stored.
    Whole,
    /// The real half of a complex array (index 0 of its trailing axis of 2).
    Re,
    /// The imaginary half (index 1 of its trailing axis of 2).
    Im,
    /// A matrix transposed: torch's `Linear` weight `[out, in]` to Burn's `[in, out]`; also a
    /// pointwise convolution's weight `[out, in, 1]` (2.0.0's `ChannelMLP`) to a `Linear`'s.
    Transposed,
}

/// One map entry: the 2.0.0 parameter name (without `param.`), the Burn path, the part.
pub type MapEntry = (&'static str, &'static str, Take);

/// The floating-point type a fixture was written in, from its name's `_f32` / `_f64`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Precision {
    /// `f32`, bound [`D3_F32`].
    F32,
    /// `f64`, bound [`D3_F64`].
    F64,
}

impl Precision {
    /// D3's bound for this precision.
    pub fn d3(self) -> f64 {
        match self {
            Precision::F32 => D3_F32,
            Precision::F64 => D3_F64,
        }
    }

    /// The fixture name's suffix, `f32` or `f64`.
    pub fn suffix(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F64 => "f64",
        }
    }
}

/// An open fixture.
pub struct Fixture {
    /// The fixture's file stem, `<config>_<dtype>`.
    pub(crate) name: String,
    reader: NpzFileReader,
    /// The 2.0.0 parameter names, `param.` stripped (as replaced, if any).
    pub(crate) params: Vec<String>,
    /// The fixture's precision.
    pub precision: Precision,
    /// Converted arrays, by full key (`param.…`), read before the file's: a layout
    /// converted at load (D5) is put here, with `params` updated to match.
    pub(crate) replaced: BTreeMap<String, ArrayD<f64>>,
}

impl Fixture {
    /// Opens `tests/fixtures/parity/<name>.npz`.
    ///
    /// # Panics
    /// If the file is missing or unreadable, or `name` does not end in `_f32` or `_f64`.
    pub fn open(name: &str) -> Self {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/parity")
            .join(format!("{name}.npz"));
        Self::open_path(&path)
    }

    /// Opens `<stem>_f32` or `<stem>_f64`, by `precision`.
    ///
    /// # Panics
    /// As [`open`](Self::open).
    #[allow(
        dead_code,
        reason = "every parity binary includes this harness; some open fixtures by full name"
    )]
    pub fn open_as(stem: &str, precision: Precision) -> Self {
        Self::open(&format!("{stem}_{}", precision.suffix()))
    }

    /// Replaces the parameter `name` by `parts` (each a new parameter name and its array),
    /// for this load only, so that [`load_into`]'s checks see the converted parameters.
    ///
    /// # Panics
    /// If the fixture has no parameter `name`.
    #[allow(
        dead_code,
        reason = "every parity binary includes this harness; only layouts converted at load use it"
    )]
    pub fn replace(&mut self, name: &str, parts: Vec<(String, ArrayD<f64>)>) {
        let i = self
            .params
            .iter()
            .position(|p| p == name)
            .unwrap_or_else(|| panic!("fixture: no parameter `{name}`"));
        self.params.remove(i);
        self.replaced.remove(&format!("{PARAM_PREFIX}{name}"));
        for (part, a) in parts {
            self.replaced.insert(format!("{PARAM_PREFIX}{part}"), a);
            self.params.push(part);
        }
    }

    /// Opens the fixture at `path`; its precision comes from the file stem's suffix.
    ///
    /// # Panics
    /// As [`open`](Self::open).
    fn open_path(path: &Path) -> Self {
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let precision = if name.ends_with("_f32") {
            Precision::F32
        } else if name.ends_with("_f64") {
            Precision::F64
        } else {
            panic!("fixture {name}: name must end in _f32 or _f64");
        };
        let reader = NpzFileReader::new(path).unwrap_or_else(|e| panic!("fixture {name}: {e}"));
        // NpzFileReader keeps its names private; the index is cheap to read twice.
        let params = std::fs::File::open(path)
            .map_err(|e| e.to_string())
            .and_then(|f| NpzReader::new(f).map_err(|e| e.to_string()))
            .and_then(|mut npz| npz.names().map_err(|e| e.to_string()))
            .unwrap_or_else(|e| panic!("fixture {name}: {e}"))
            .iter()
            .filter_map(|n| n.strip_prefix(PARAM_PREFIX).map(str::to_owned))
            .collect();
        Self {
            name,
            reader,
            params,
            precision,
            replaced: BTreeMap::new(),
        }
    }

    /// The array `key` (`input`, `output`, `param.…`), cast up to f64 (exact from f32).
    ///
    /// # Panics
    /// If the fixture has no such array.
    pub fn array(&mut self, key: &str) -> ArrayD<f64> {
        if let Some(a) = self.replaced.get(key) {
            return a.clone();
        }
        self.reader
            .read_field(key)
            .unwrap_or_else(|e| panic!("fixture {}: {e}", self.name))
    }

    /// The fixture's input, `(batch, channels, *grid)`, as a tensor on `device` in the
    /// fixture's precision.
    pub fn input<const R: usize>(&mut self, device: &Device) -> Tensor<R> {
        self.tensor("input", device)
    }

    /// The fixture's output, `(batch, channels, *grid)`, on the host.
    pub fn output(&mut self) -> ArrayD<f64> {
        self.array("output")
    }

    /// The array `key` as a rank-`R` tensor on `device`, in the fixture's precision.
    ///
    /// # Panics
    /// If the fixture has no such array, or it is not of rank `R`.
    pub fn tensor<const R: usize>(&mut self, key: &str, device: &Device) -> Tensor<R> {
        let a = self.array(key);
        assert_eq!(
            a.ndim(),
            R,
            "fixture {}: rank {} ≠ {R}",
            self.name,
            a.ndim()
        );
        Tensor::from_data(self.data(&a), device)
    }

    fn data(&self, a: &ArrayD<f64>) -> TensorData {
        let shape = a.shape().to_vec();
        let values = a.iter().copied();
        match self.precision {
            Precision::F32 => TensorData::new(values.map(|v| v as f32).collect(), shape),
            Precision::F64 => TensorData::new(values.collect(), shape),
        }
    }
}

/// The `take` part of `a`: `a` itself, its transpose, or one half of its trailing (re, im)
/// axis.
///
/// # Panics
/// For `Transposed`, if `a` is neither a matrix nor `[out, in, 1]`; for `Re` / `Im`, if the
/// trailing axis is not of length 2.
pub fn take(a: ArrayD<f64>, take: Take) -> ArrayD<f64> {
    let part = match take {
        Take::Whole => return a,
        Take::Transposed => {
            let a = match a.shape() {
                &[o, i, 1] => a.into_shape_with_order((o, i)).unwrap().into_dyn(),
                _ => a,
            };
            assert_eq!(a.ndim(), 2, "Transposed needs a matrix or [out, in, 1]");
            return a.reversed_axes().as_standard_layout().to_owned();
        }
        Take::Re => 0,
        Take::Im => 1,
    };
    let last = a.ndim() - 1;
    assert_eq!(
        a.shape()[last],
        2,
        "{take:?} needs a trailing (re, im) axis of 2"
    );
    a.index_axis(Axis(last), part).to_owned()
}

/// Loads `fixture`'s parameters into `module` through `map`.
///
/// # Panics
/// Unless the load is complete in both directions: every fixture parameter is taken by the
/// map (a complex one by its `Re` and `Im` both, any other once, `Whole` or `Transposed`), every map entry names a
/// fixture parameter and is applied without error (shapes included), and no module
/// parameter is left out.
#[allow(
    dead_code,
    reason = "every parity binary includes this harness; some load only part of a module"
)]
pub fn load_into<M: ModuleSnapshot>(module: &mut M, fixture: &mut Fixture, map: &[MapEntry]) {
    load_into_part(module, fixture, map, None);
}

/// As [`load_into`], but only the module parameters `filter` selects must be loaded (all of
/// them when `None`): for a fixture of part of a module (library CM1.3b, the FNO's Fourier
/// layers without its lifting and projection).
///
/// # Panics
/// As [`load_into`], for the selected parameters.
pub fn load_into_part<M: ModuleSnapshot>(
    module: &mut M,
    fixture: &mut Fixture,
    map: &[MapEntry],
    filter: Option<PathFilter>,
) {
    let name = fixture.name.clone();
    let mut parts: BTreeMap<&str, Vec<Take>> = BTreeMap::new();
    for &(from, _, part) in map {
        parts.entry(from).or_default().push(part);
    }
    let unmapped: Vec<&str> = fixture
        .params
        .iter()
        .map(String::as_str)
        .filter(|p| !parts.contains_key(p))
        .collect();
    assert!(
        unmapped.is_empty(),
        "fixture {name}: parameters the map leaves out: {unmapped:?}"
    );
    for (from, parts) in &parts {
        let complete = matches!(
            parts.as_slice(),
            [Take::Whole] | [Take::Transposed] | [Take::Re, Take::Im] | [Take::Im, Take::Re]
        );
        assert!(
            complete,
            "fixture {name}: `{from}` must be mapped once (Whole or Transposed) or once each \
             Re and Im, got {parts:?}"
        );
    }

    let tensors: Vec<_> = map
        .iter()
        .map(|&(from, to, part)| {
            let a = take(fixture.array(&format!("{PARAM_PREFIX}{from}")), part);
            bridge::from_data(fixture.data(&a), to.to_owned(), None)
        })
        .collect();
    let result = module.apply(tensors, filter, None, false);
    assert!(
        result.errors.is_empty()
            && result.missing.is_empty()
            && result.unused.is_empty()
            && result.applied.len() == map.len(),
        "fixture {name}: incomplete load: {} applied of {}; errors {:?}; module parameters \
         not loaded {:?}; map paths not in the module {:?}",
        result.applied.len(),
        map.len(),
        result.errors,
        result.missing,
        result.unused
    );
}

/// max |a − e| / max |e| over every element: D3's relative max error, relative to the
/// reference's scale so that elements near zero do not dominate.
///
/// # Panics
/// If the shapes differ, either side has a NaN or infinity, or `expected` is all zeros.
pub fn rel_max_error<const R: usize>(actual: Tensor<R>, expected: &ArrayD<f64>) -> f64 {
    assert_eq!(actual.dims().as_slice(), expected.shape(), "shape");
    let actual = actual
        .into_data()
        .convert::<f64>()
        .try_into_vec::<f64>()
        .unwrap();
    // f64::max skips NaN, so a non-finite value would otherwise pass unseen.
    let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
    assert!(finite(&actual), "the output has a non-finite value");
    assert!(
        finite(expected.as_slice().expect("standard layout")),
        "the reference has a non-finite value"
    );
    let scale = expected.iter().fold(0.0_f64, |m, e| m.max(e.abs()));
    assert!(scale > 0.0, "the reference is all zeros");
    let err = actual
        .iter()
        .zip(expected.iter())
        .fold(0.0_f64, |m, (a, e)| m.max((a - e).abs()));
    err / scale
}

/// Asserts `actual` is within `tol` of `expected` by [`rel_max_error`]; returns the error.
/// For outputs now; for gradients when a fixture carries them.
///
/// # Panics
/// If the error exceeds `tol`, or as [`rel_max_error`].
pub fn assert_parity<const R: usize>(
    actual: Tensor<R>,
    expected: &ArrayD<f64>,
    tol: f64,
    what: &str,
) -> f64 {
    let err = rel_max_error(actual, expected);
    assert!(
        err <= tol,
        "{what}: relative max error {err:.3e} > {tol:.0e}"
    );
    err
}

/// Asserts `actual` within D3 of `fixture`'s output for its precision, prints the error for
/// the record and returns it.
///
/// # Panics
/// As [`assert_parity`].
#[allow(
    dead_code,
    reason = "every parity binary includes this harness; some compare parts of an output"
)]
pub fn check_d3<const R: usize>(actual: Tensor<R>, fixture: &mut Fixture) -> f64 {
    let name = fixture.name.clone();
    let err = assert_parity(actual, &fixture.output(), fixture.precision.d3(), &name);
    println!("{name}: relative max error {err:.3e}");
    err
}

/// `[B, C, s..]` → `[B, s.., C]`: a channels-first fixture tensor for the FNO.
#[allow(
    dead_code,
    reason = "every parity binary includes this harness; only the FNO's binaries permute"
)]
pub fn to_channels_last<const R: usize>(x: Tensor<R>) -> Tensor<R> {
    x.permute(core::array::from_fn(|i| match i {
        0 => 0,
        i if i == R - 1 => 1,
        i => i + 1,
    }))
}

/// `[B, s.., C]` → `[B, C, s..]`, the inverse of [`to_channels_last`].
#[allow(
    dead_code,
    reason = "every parity binary includes this harness; only the FNO's binaries permute"
)]
pub fn to_channels_first<const R: usize>(x: Tensor<R>) -> Tensor<R> {
    x.permute(core::array::from_fn(|i| match i {
        0 => 0,
        1 => R - 1,
        i => i - 1,
    }))
}

/// 2.0.0's centred spectral weight `a`, `(lead.., K_0, .., K_{d−1}, 2)` with a trailing
/// (re, im) axis, as the crate's `2^(d−1)` mode corners, in corner order (bit `j` of the
/// index is axis `j`, CONVENTIONS §5), each `(lead.., m_0, .., m_{d−2}, K_{d−1}, 2)`
/// (library CM1.2, D5).
///
/// On a non-last axis 2.0.0 keeps `K` modes around frequency 0 after `fftshift`: index `i`
/// is frequency `i − K/2` (integer division), so `K/2` negative and `K − K/2` non-negative
/// frequencies. Each corner gets `m = K − K/2`: the low corner indices `K/2..K` (frequencies
/// `0..m`), the high corner indices `K/2 − m..K/2` (frequencies `−m..−1`); for odd `K` that
/// starts one before 0, and the row is zero, a frequency 2.0.0 does not keep. The last axis
/// (the real FFT's, frequencies `0..K`) is taken whole.
///
/// The corners reproduce 2.0.0 only where the grid's extent is even on every non-last
/// axis: on an odd extent 2.0.0 applies `fftshift` (not `ifftshift`) again before the
/// inverse, which moves its output spectrum by one bin.
///
/// # Panics
/// If `a` has fewer than `d + 1` axes or no trailing axis of 2.
#[allow(
    dead_code,
    reason = "every parity binary includes this harness; only the centred layer's use it"
)]
pub fn centred_to_corners(a: ArrayD<f64>, d: usize) -> Vec<ArrayD<f64>> {
    assert!(
        a.ndim() > d,
        "centred_to_corners: {} axes for d = {d}",
        a.ndim()
    );
    assert_eq!(a.shape()[a.ndim() - 1], 2, "no trailing (re, im) axis");
    let lead = a.ndim() - 1 - d;
    (0..1usize << (d - 1))
        .map(|corner| {
            let mut c = a.clone();
            for j in 0..d - 1 {
                let axis = Axis(lead + j);
                let k = c.shape()[lead + j];
                let half = k / 2;
                c = if corner >> j & 1 == 0 {
                    c.slice_axis(axis, Slice::from(half..k)).to_owned()
                } else if k.is_multiple_of(2) {
                    c.slice_axis(axis, Slice::from(0..half)).to_owned()
                } else {
                    let mut zero_shape = c.shape().to_vec();
                    zero_shape[lead + j] = 1;
                    let zero = ArrayD::zeros(zero_shape);
                    let neg = c.slice_axis(axis, Slice::from(0..half));
                    concatenate(axis, &[zero.view(), neg]).unwrap()
                };
            }
            c
        })
        .collect()
}

/// A map path built at run time. Test-only: leaked, so the harness's `&'static str`
/// entries can be built in a loop.
#[allow(
    dead_code,
    reason = "every parity binary includes this harness; only generated maps use it"
)]
pub fn leak(path: String) -> &'static str {
    path.leak()
}

/// 2.0.0's `FNO` lifting and projection (`lifting.fcs.*`, `projection.fcs.*`) to the
/// crate's `fc0`, `lifting_out.0` (with a two-layer lifting), `fc1`, `fc2`, under the Burn
/// path prefix `to` (library CM1.4).
#[allow(
    dead_code,
    reason = "every parity binary includes this harness; only the FNO's binaries map it"
)]
pub fn lift_project(two_layer_lifting: bool, to: &str) -> Vec<MapEntry> {
    let mut layers = vec![
        ("lifting.fcs.0", "fc0"),
        ("projection.fcs.0", "fc1"),
        ("projection.fcs.1", "fc2"),
    ];
    if two_layer_lifting {
        layers.push(("lifting.fcs.1", "lifting_out.0"));
    }
    layers
        .into_iter()
        .flat_map(|(from, at)| {
            [
                (
                    leak(format!("{from}.weight")),
                    leak(format!("{to}{at}.weight")),
                    Take::Transposed,
                ),
                (
                    leak(format!("{from}.bias")),
                    leak(format!("{to}{at}.bias")),
                    Take::Whole,
                ),
            ]
        })
        .collect()
}

/// The parts of 2.0.0's `FNOBlocks` (`fno_blocks.*`) other than the spectral convolutions,
/// for the block options of `options`, to the crate's `FNO` under the Burn path prefix `to`:
/// the FNO skips, channel MLPs and their skips, and the norms (library CM1.3b). Burn names
/// an enum module's parameters after the variant (`Linear`, `SoftGating`, `Group`, `AdaIn`).
#[allow(
    dead_code,
    reason = "every parity binary includes this harness; only the FNO's binaries map it"
)]
pub fn block_parts(options: &FNOConfig, to: &str) -> Vec<MapEntry> {
    let mut m = Vec::new();
    let skip = |m: &mut Vec<MapEntry>, from: String, to: String, kind: &Option<SkipKind>| {
        match kind {
            // 2.0.0's linear skip is a `Flattened1dConv`, its weight `conv.weight`.
            Some(SkipKind::Linear) => m.push((
                leak(format!("{from}.conv.weight")),
                leak(format!("{to}.Linear.weight")),
                Take::Whole,
            )),
            Some(SkipKind::SoftGating) => m.push((
                leak(format!("{from}.weight")),
                leak(format!("{to}.SoftGating")),
                Take::Whole,
            )),
            Some(SkipKind::Identity) | None => {}
        }
    };
    for l in 0..options.n_layers {
        skip(
            &mut m,
            format!("fno_blocks.fno_skips.{l}"),
            format!("{to}fno_skips.{l}"),
            &options.fno_skip,
        );
        if let Some(mlp) = &options.channel_mlp {
            for k in 0..2 {
                for p in ["weight", "bias"] {
                    let path = format!("channel_mlp.{l}.fcs.{k}.{p}");
                    m.push((
                        leak(format!("fno_blocks.{path}")),
                        leak(format!("{to}{path}")),
                        Take::Whole,
                    ));
                }
            }
            skip(
                &mut m,
                format!("fno_blocks.channel_mlp_skips.{l}"),
                format!("{to}channel_mlp_skips.{l}"),
                &mlp.skip,
            );
        }
    }
    for k in 0..2 * options.n_layers {
        let (from, to) = (format!("fno_blocks.norm.{k}"), format!("{to}norms.{k}"));
        match options.norm {
            Some(NormKind::Group) => {
                m.push((
                    leak(format!("{from}.weight")),
                    leak(format!("{to}.Group.gamma")),
                    Take::Whole,
                ));
                m.push((
                    leak(format!("{from}.bias")),
                    leak(format!("{to}.Group.beta")),
                    Take::Whole,
                ));
            }
            Some(NormKind::AdaIn { .. }) => {
                for (t, b) in [(0, "fc_hidden"), (2, "fc_out")] {
                    let (f, t) = (format!("{from}.mlp.{t}"), format!("{to}.AdaIn.{b}"));
                    m.push((
                        leak(format!("{f}.weight")),
                        leak(format!("{t}.weight")),
                        Take::Transposed,
                    ));
                    m.push((
                        leak(format!("{f}.bias")),
                        leak(format!("{t}.bias")),
                        Take::Whole,
                    ));
                }
            }
            Some(NormKind::Instance) | None => {}
        }
    }
    m
}
