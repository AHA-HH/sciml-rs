//! The parity harness (library CM0.5): loads a `neuraloperator` 2.0.0 fixture
//! (`tests/fixtures/parity/`, written by `tools/parity/generate.py`) into a Burn module by a
//! hand-written parameter-name map, and compares tensors against the fixture's within D3.
//!
//! A fixture's `.npz` holds `input`, `output` and every tensor of the 2.0.0 model's
//! `state_dict` as `param.<2.0.0 name>`; a complex tensor is stored as a trailing axis of 2,
//! (real, imaginary). Each map entry names one such array, the Burn path it goes to (as
//! `ModuleSnapshot::collect` names it) and which part of it to take.
//!
//! A parameter whose layout differs from the crate's is converted for one load: the caller
//! passes [`load_into`] the [`Conversion`]s, the fixture itself is never changed (2.0.0's
//! centred spectral weights: CM1.2's `centred_conv.rs`, D5).
//!
//! The reference layers beside this file (`biased_conv.rs`, `legacy_conv.rs`,
//! `centred_conv.rs`) are included only by the test binaries that use them, with `#[path]`,
//! so no binary compiles dead code; the FNO family's shared map pieces are `fno_map.rs`,
//! included the same way (library CM1.5a).
//! 2.0.0's centred spectral weights as corners (`centred_to_corners`) are here.

use std::{collections::BTreeMap, path::Path};

use burn::{
    prelude::*,
    store::{ModuleSnapshot, PathFilter, bridge},
    tensor::TensorData,
};
use ndarray::{ArrayD, Axis, Slice, concatenate};
use sciml_rs::neural_operators::data::io::readers::npz::NpzFileReader;

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

/// A fixture parameter converted for one load (D5): its 2.0.0 name (without `param.`) and
/// the parts that replace it, each a name the map can take and its array.
pub type Conversion = (String, Vec<(String, ArrayD<f64>)>);

/// An open fixture.
pub struct Fixture {
    /// The fixture's file stem, `<config>_<dtype>`.
    name: String,
    reader: NpzFileReader,
    /// The 2.0.0 parameter names, `param.` stripped.
    params: Vec<String>,
    /// The fixture's precision.
    pub precision: Precision,
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
        let params = reader
            .names()
            .iter()
            .filter_map(|n| n.strip_prefix(PARAM_PREFIX).map(str::to_owned))
            .collect();
        Self {
            name,
            reader,
            params,
            precision,
        }
    }

    /// The array `key` (`input`, `output`, `param.…`), cast up to f64 (exact from f32).
    ///
    /// # Panics
    /// If the fixture has no such array.
    pub fn array(&mut self, key: &str) -> ArrayD<f64> {
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

    /// The complex array `key` (a trailing (re, im) axis of 2) as its real and imaginary
    /// parts, rank-`R` tensors on `device` in the fixture's precision.
    ///
    /// # Panics
    /// As [`take`], or if the parts are not of rank `R`.
    #[allow(
        dead_code,
        reason = "every parity binary includes this harness; only the GaLore ones read complex arrays"
    )]
    pub fn complex_tensor<const R: usize>(
        &mut self,
        key: &str,
        device: &Device,
    ) -> (Tensor<R>, Tensor<R>) {
        let a = self.array(key);
        let part = |t| {
            let p = take(a.clone(), t);
            assert_eq!(
                p.ndim(),
                R,
                "fixture {}: {key} rank {} ≠ {R}",
                self.name,
                p.ndim()
            );
            Tensor::from_data(self.data(&p), device)
        };
        (part(Take::Re), part(Take::Im))
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

/// Loads `fixture`'s parameters into `module` through `map`, each parameter `converted`
/// names replaced by its parts for this load.
///
/// # Panics
/// Unless the load is complete in both directions: every fixture parameter (after
/// conversion) is taken by the map (a complex one by its `Re` and `Im` both, any other
/// once, `Whole` or `Transposed`), every map entry names such a parameter and is applied
/// without error (shapes included), and no module parameter is left out; or if a
/// conversion names a parameter the fixture lacks.
#[allow(
    dead_code,
    reason = "every parity binary includes this harness; some load only part of a module"
)]
pub fn load_into<M: ModuleSnapshot>(
    module: &mut M,
    fixture: &mut Fixture,
    map: &[MapEntry],
    converted: &[Conversion],
) {
    load_into_part(module, fixture, map, converted, None);
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
    converted: &[Conversion],
    filter: Option<PathFilter>,
) {
    let name = fixture.name.clone();
    let mut params = fixture.params.clone();
    let mut arrays: BTreeMap<&str, &ArrayD<f64>> = BTreeMap::new();
    for (from, parts) in converted {
        let i = params
            .iter()
            .position(|p| p == from)
            .unwrap_or_else(|| panic!("fixture {name}: no parameter `{from}` to convert"));
        params.remove(i);
        for (part, a) in parts {
            arrays.insert(part, a);
            params.push(part.clone());
        }
    }

    let mut parts: BTreeMap<&str, Vec<Take>> = BTreeMap::new();
    for &(from, _, part) in map {
        parts.entry(from).or_default().push(part);
    }
    let unmapped: Vec<&str> = params
        .iter()
        .map(String::as_str)
        .filter(|p| !parts.contains_key(p))
        .collect();
    assert!(
        unmapped.is_empty(),
        "fixture {name}: parameters the map leaves out: {unmapped:?}"
    );
    let unknown: Vec<&str> = parts
        .keys()
        .copied()
        .filter(|from| !params.iter().any(|p| p == from))
        .collect();
    assert!(
        unknown.is_empty(),
        "fixture {name}: map names that are not fixture parameters: {unknown:?}"
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
            let a = match arrays.get(from) {
                Some(&a) => a.clone(),
                None => fixture.array(&format!("{PARAM_PREFIX}{from}")),
            };
            let a = take(a, part);
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
