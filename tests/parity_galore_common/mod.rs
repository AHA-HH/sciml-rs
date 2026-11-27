//! The Tensor GaLore parity checks (library CM3.6a, complex CM3.6b), shared by the f32 and
//! f64 binaries: each case of `tools/parity/generate_galore.py` against 2.0.0's
//! `TensorGaLoreProjector`, on real (`galore_*`) or complex (`galore_complex_*`) gradients.

use burn::prelude::*;
use sciml_rs::neural_operators::optim::galore::{
    ComplexTensor, GaLoreRank, GaLoreTensor, TensorGaLoreConfig,
};

use crate::parity_common::{Fixture, Take, assert_parity, take};

/// A tensor the cases compare: read from a fixture, and checked against one.
pub trait Field<const R: usize>: Sized {
    /// The array `key` of `fx` on `device`.
    fn load(fx: &mut Fixture, key: &str, device: &Device) -> Self;
    /// Asserts `self` within `tol` of the array `key`; returns the error.
    fn check(self, fx: &mut Fixture, key: &str, tol: f64) -> f64;
}

impl<const R: usize> Field<R> for Tensor<R> {
    fn load(fx: &mut Fixture, key: &str, device: &Device) -> Self {
        fx.tensor(key, device)
    }

    fn check(self, fx: &mut Fixture, key: &str, tol: f64) -> f64 {
        assert_parity(self, &fx.array(key), tol, key)
    }
}

/// Real and imaginary parts each within `tol`, relative to their own scale.
impl<const R: usize> Field<R> for ComplexTensor<R> {
    fn load(fx: &mut Fixture, key: &str, device: &Device) -> Self {
        let (re, im) = fx.complex_tensor(key, device);
        ComplexTensor::new(re, im)
    }

    fn check(self, fx: &mut Fixture, key: &str, tol: f64) -> f64 {
        let a = fx.array(key);
        let re = assert_parity(
            self.re,
            &take(a.clone(), Take::Re),
            tol,
            &format!("{key}.re"),
        );
        let im = assert_parity(self.im, &take(a, Take::Im), tol, &format!("{key}.im"));
        re.max(im)
    }
}

/// The cases' configurations, as `generate_galore.py`'s `CASES`.
pub fn config(case: &str) -> TensorGaLoreConfig {
    let (rank, gap, scale) = match case {
        "matrix" => (GaLoreRank::Uniform(3), 5, 1.0),
        "tensor4" => (GaLoreRank::PerMode(vec![3, 2, 3, 2]), 200, 0.25),
        "fraction" => (GaLoreRank::Fraction(0.5), 200, 1.0),
        _ => panic!("no case {case}"),
    };
    TensorGaLoreConfig {
        rank,
        update_proj_gap: gap,
        scale,
    }
}

/// Case `case` of `fixture` (rank-`D` gradients of type `T`): the validated ranks, each
/// factor, `project(grad, 0)`, `project_back` of it and `project(grad2, gap)` within D3;
/// prints the largest error.
pub fn run<const D: usize, T>(fixture: &str, case: &str, device: &Device)
where
    T: GaLoreTensor<D> + Field<D>,
    T::Factor: Field<2>,
{
    let mut fx = Fixture::open(fixture);
    let tol = fx.precision.d3();
    let key = |k: &str| format!("{case}.{k}");
    let config = config(case);
    let grad = T::load(&mut fx, &key("grad"), device);

    let ranks: Vec<usize> = fx
        .array(&key("ranks"))
        .iter()
        .map(|&r| r as usize)
        .collect();
    assert_eq!(
        config.rank.ranks(grad.shape()).as_slice(),
        ranks,
        "{case}: ranks"
    );

    let mut projector = config.init::<D, T>();
    let low = projector.project(grad, 0);
    let mut worst = 0.0_f64;
    let factors = projector.factors().expect("factors after project").clone();
    for (k, f) in factors.into_iter().enumerate() {
        worst = worst.max(f.check(&mut fx, &key(&format!("factor{k}")), tol));
    }
    let back = projector.project_back(low.clone());
    worst = worst.max(low.check(&mut fx, &key("low"), tol));
    worst = worst.max(back.check(&mut fx, &key("back"), tol));
    let grad2 = T::load(&mut fx, &key("grad2"), device);
    let low2 = projector.project(grad2, config.update_proj_gap);
    worst = worst.max(low2.check(&mut fx, &key("low2"), tol));
    println!("{fixture} {case}: relative max error {worst:.3e}");
}

/// Case `case` of a complex fixture where a sweep's unfolding is tall, so 2.0.0's factors'
/// phases are ill-conditioned (`optim::galore`'s documentation): what is well defined,
/// within D3: the ranks' subspaces Fₖ Fₖᴴ, `back`, and `low2` taken back to the full space
/// (`project_back(project(grad2, gap))` against the fixture's factors applied to its `low2`);
/// prints the largest error.
pub fn run_up_to_phase<const D: usize>(fixture: &str, case: &str, device: &Device) {
    let mut fx = Fixture::open(fixture);
    let tol = fx.precision.d3();
    let key = |k: &str| format!("{case}.{k}");
    let config = config(case);
    let mut projector = config.init::<D, ComplexTensor<D>>();
    let low = projector.project(ComplexTensor::load(&mut fx, &key("grad"), device), 0);
    let ours = projector.factors().expect("factors after project").clone();
    let theirs: [ComplexTensor<2>; D] =
        std::array::from_fn(|k| ComplexTensor::load(&mut fx, &key(&format!("factor{k}")), device));
    let mut worst = 0.0_f64;
    for (k, (f, g)) in ours.iter().zip(&theirs).enumerate() {
        let what = format!("{fixture} {case} subspace{k}");
        worst = worst.max(close(subspace(f), subspace(g), tol, &what));
    }
    worst = worst.max(
        projector
            .project_back(low)
            .check(&mut fx, &key("back"), tol),
    );
    let grad2 = ComplexTensor::load(&mut fx, &key("grad2"), device);
    let low2 = projector.project(grad2, config.update_proj_gap);
    let ours2 = projector.project_back(low2);
    let theirs2 = ComplexTensor::<D>::load(&mut fx, &key("low2"), device)
        .mode_products(&theirs, None, false)
        .scaled(config.scale);
    worst = worst.max(close(ours2, theirs2, tol, "low2 back"));
    println!("{fixture} {case} (up to phase): relative max error {worst:.3e}");
}

/// F Fᴴ, the orthogonal projector onto a factor's columns.
fn subspace(f: &ComplexTensor<2>) -> ComplexTensor<2> {
    let (a, b) = (f.re.clone(), f.im.clone());
    ComplexTensor::new(
        a.clone().matmul(a.clone().transpose()) + b.clone().matmul(b.clone().transpose()),
        b.clone().matmul(a.clone().transpose()) - a.matmul(b.transpose()),
    )
}

/// Asserts max |a − b| / max |b| (moduli) ≤ `tol`; returns it.
fn close<const R: usize>(a: ComplexTensor<R>, b: ComplexTensor<R>, tol: f64, what: &str) -> f64 {
    let modulus = |re: Tensor<R>, im: Tensor<R>| -> f64 {
        (re.square() + im.square())
            .max()
            .sqrt()
            .into_scalar::<f64>()
    };
    let scale = modulus(b.re.clone(), b.im.clone());
    let err = modulus(a.re - b.re, a.im - b.im) / scale;
    assert!(
        err <= tol,
        "{what}: relative max error {err:.3e} > {tol:.0e}"
    );
    err
}
