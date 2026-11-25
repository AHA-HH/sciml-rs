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
//! the test binaries that use them, with `#[path]`, so no binary compiles dead code.

use std::{collections::BTreeMap, path::Path};

use burn::{
    prelude::*,
    store::{ModuleSnapshot, bridge},
    tensor::TensorData,
};
use ndarray::{ArrayD, Axis};
use ndarray_npy::NpzReader;
use sciml_rs::neural_operators::data::io::readers::npz::NpzFileReader;

/// D3's bound in f32: relative max error ≤ 1e-5 (roadmap, signed off 2026-11-25).
pub const D3_F32: f64 = 1e-5;
/// D3's bound in f64: relative max error ≤ 1e-12.
pub const D3_F64: f64 = 1e-12;

/// The prefix of a parameter array's name in a fixture.
const PARAM_PREFIX: &str = "param.";

/// Which part of a fixture array a map entry takes.
#[derive(Clone, Copy, Debug)]
pub enum Take {
    /// The array as stored.
    Whole,
    /// The real half of a complex array (index 0 of its trailing axis of 2).
    Re,
    /// The imaginary half (index 1 of its trailing axis of 2).
    Im,
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
}

/// An open fixture.
pub struct Fixture {
    name: String,
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
        let a = self.array("input");
        self.tensor(&a, device)
    }

    /// The fixture's output, `(batch, channels, *grid)`, on the host.
    pub fn output(&mut self) -> ArrayD<f64> {
        self.array("output")
    }

    /// `a` as a rank-`R` tensor on `device`, in the fixture's precision.
    ///
    /// # Panics
    /// If `a` is not of rank `R`.
    fn tensor<const R: usize>(&self, a: &ArrayD<f64>, device: &Device) -> Tensor<R> {
        assert_eq!(
            a.ndim(),
            R,
            "fixture {}: rank {} ≠ {R}",
            self.name,
            a.ndim()
        );
        Tensor::from_data(self.data(a), device)
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

/// The `take` part of `a`: `a` itself, or one half of its trailing (re, im) axis.
///
/// # Panics
/// For `Re` / `Im`, if the trailing axis is not of length 2.
pub fn take(a: ArrayD<f64>, take: Take) -> ArrayD<f64> {
    let part = match take {
        Take::Whole => return a,
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
/// map (a complex one by its `Re` and `Im` both, or `Whole`), every map entry names a
/// fixture parameter and is applied without error (shapes included), and no module
/// parameter is left out.
pub fn load_into<M: ModuleSnapshot>(module: &mut M, fixture: &mut Fixture, map: &[MapEntry]) {
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
            [Take::Whole] | [Take::Re, Take::Im] | [Take::Im, Take::Re]
        );
        assert!(
            complete,
            "fixture {name}: `{from}` must be mapped once Whole or once each Re and Im, got {parts:?}"
        );
    }

    let tensors: Vec<_> = map
        .iter()
        .map(|&(from, to, part)| {
            let a = take(fixture.array(&format!("{PARAM_PREFIX}{from}")), part);
            bridge::from_data(fixture.data(&a), to.to_owned(), None)
        })
        .collect();
    let result = module.apply(tensors, None, None, false);
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
