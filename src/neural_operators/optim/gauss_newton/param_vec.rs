//! [`ParamVec`]: a model's float parameters as one vector, with the arithmetic CG needs;
//! [`params`] reads it from any [`Module`] and [`with_params`] writes it back.

use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId},
    prelude::*,
    tensor::DType,
};

/// Every float parameter of a model, each flattened to a `Tensor<1>` and keyed by its
/// [`ParamId`], in the order [`Module::visit`] walks them.
///
/// The tensors live on the model's device **without autodiff** (`Tensor::inner`), so the
/// arithmetic below never records a graph. Two vectors combine only if they hold the same
/// ids in the same order and the same lengths, i.e. came from the same model structure;
/// every binary operation panics otherwise.
///
/// Reductions ([`dot`](Self::dot), [`norm`](Self::norm)) accumulate on the device in the
/// tensors' dtype (f32 unless the device is configured otherwise) and return the result
/// on the host as `f64`.
#[derive(Clone, Debug)]
pub struct ParamVec {
    parts: Vec<(ParamId, Tensor<1>)>,
}

impl ParamVec {
    /// A vector from its parts, already flattened, without autodiff and in visit order.
    pub(super) fn from_parts(parts: Vec<(ParamId, Tensor<1>)>) -> Self {
        Self { parts }
    }

    /// The parameter ids, in order.
    pub fn ids(&self) -> impl Iterator<Item = ParamId> + '_ {
        self.parts.iter().map(|(id, _)| *id)
    }

    /// The dtype of the first parameter (the model's float dtype); `None` for a model
    /// without float parameters.
    pub fn dtype(&self) -> Option<DType> {
        self.parts.first().map(|(_, t)| t.dtype())
    }

    /// The device of the parameters (that of the first; all share it).
    ///
    /// # Panics
    /// For a vector with no parameters.
    pub fn device(&self) -> Device {
        self.parts
            .first()
            .map(|(_, t)| t.device())
            .expect("ParamVec::device: no parameters")
    }

    /// The total number of scalars over all parameters.
    pub fn numel(&self) -> usize {
        self.parts.iter().map(|(_, t)| t.dims()[0]).sum()
    }

    /// ⟨self, other⟩.
    ///
    /// # Panics
    /// If the two vectors have different structures.
    pub fn dot(&self, other: &Self) -> f64 {
        self.check_same_structure(other);
        self.parts
            .iter()
            .zip(&other.parts)
            .map(|((_, a), (_, b))| (a.clone() * b.clone()).sum())
            .reduce(|acc, part| acc + part)
            .map_or(0.0, |total| total.into_scalar::<f64>())
    }

    /// The Euclidean norm ‖self‖.
    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    /// self + a · x.
    ///
    /// # Panics
    /// If the two vectors have different structures.
    pub fn axpy(&self, a: f64, x: &Self) -> Self {
        self.zip_with(x, |s, x| s + x.mul_scalar(a))
    }

    /// a · self.
    pub fn scale(&self, a: f64) -> Self {
        self.map(|t| t.mul_scalar(a))
    }

    /// A vector of zeros with the same structure.
    pub fn zeros_like(&self) -> Self {
        self.map(|t| t.zeros_like())
    }

    /// The elementwise product, for diagonal preconditioners.
    ///
    /// # Panics
    /// If the two vectors have different structures.
    pub fn mul(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a * b)
    }

    /// The elementwise quotient self / other, for diagonal preconditioners.
    ///
    /// # Panics
    /// If the two vectors have different structures.
    pub fn div(&self, other: &Self) -> Self {
        self.zip_with(other, |a, b| a / b)
    }

    /// All parameters concatenated into one `[numel]` tensor, in order.
    pub fn to_flat(&self) -> Tensor<1> {
        Tensor::cat(self.parts.iter().map(|(_, t)| t.clone()).collect(), 0)
    }

    /// A vector with `self`'s structure holding the values of `flat` (`[numel]`), the
    /// inverse of [`to_flat`](Self::to_flat). `flat` is moved to `self`'s device and loses
    /// any autodiff association.
    ///
    /// # Panics
    /// If `flat` does not have `self.numel()` elements.
    pub fn from_flat_like(&self, flat: Tensor<1>) -> Self {
        assert_eq!(
            flat.dims()[0],
            self.numel(),
            "from_flat_like: flat vector has the wrong length"
        );
        let flat = flat.inner();
        let mut start = 0;
        let parts = self
            .parts
            .iter()
            .map(|(id, t)| {
                let n = t.dims()[0];
                let part = flat.clone().narrow(0, start, n).to_device(&t.device());
                start += n;
                (*id, part)
            })
            .collect();
        Self { parts }
    }

    /// The parts, flattened, without autodiff and in visit order.
    pub(super) fn parts(&self) -> &[(ParamId, Tensor<1>)] {
        &self.parts
    }

    /// Applies `f` to every part.
    pub(super) fn map(&self, f: impl Fn(Tensor<1>) -> Tensor<1>) -> Self {
        Self {
            parts: self
                .parts
                .iter()
                .map(|(id, t)| (*id, f(t.clone())))
                .collect(),
        }
    }

    fn zip_with(&self, other: &Self, f: impl Fn(Tensor<1>, Tensor<1>) -> Tensor<1>) -> Self {
        self.check_same_structure(other);
        Self {
            parts: self
                .parts
                .iter()
                .zip(&other.parts)
                .map(|((id, a), (_, b))| (*id, f(a.clone(), b.clone())))
                .collect(),
        }
    }

    fn check_same_structure(&self, other: &Self) {
        assert_eq!(
            self.parts.len(),
            other.parts.len(),
            "ParamVec: different number of parameters"
        );
        for ((ia, a), (ib, b)) in self.parts.iter().zip(&other.parts) {
            assert_eq!(ia, ib, "ParamVec: parameter ids differ");
            assert_eq!(a.dims(), b.dims(), "ParamVec: parameter lengths differ");
        }
    }
}

/// Visits every float parameter and collects it, flattened and without autodiff.
struct Collect {
    parts: Vec<(ParamId, Tensor<1>)>,
}

impl ModuleVisitor for Collect {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let value = param.val().inner();
        let n: usize = value.dims().iter().product();
        self.parts.push((param.id, value.reshape([n])));
    }
}

/// Every float parameter of `model` as a [`ParamVec`], in [`Module::visit`] order, on the
/// model's device without autodiff. Values are copied bit for bit.
pub fn params<M: Module>(model: &M) -> ParamVec {
    let mut collect = Collect { parts: Vec::new() };
    model.visit(&mut collect);
    ParamVec {
        parts: collect.parts,
    }
}

/// Replaces every float parameter, in [`Module::map`] order, by the matching part of a
/// [`ParamVec`], reshaped to the parameter's shape.
struct Replace<'a> {
    parts: &'a [(ParamId, Tensor<1>)],
    next: usize,
    /// Keep each parameter's autodiff association and `require_grad` flag; if false, the new
    /// parameters are plain tensors (no graph is recorded through them).
    keep_autodiff: bool,
}

impl ModuleMapper for Replace<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, flat) = self
            .parts
            .get(self.next)
            .expect("with_params: the ParamVec has fewer parameters than the model");
        assert_eq!(
            *id, param.id,
            "with_params: parameter ids differ (the ParamVec is from another model)"
        );
        self.next += 1;
        let keep_autodiff = self.keep_autodiff;
        param.map(|old| {
            let dims = old.dims();
            assert_eq!(
                flat.dims()[0],
                dims.iter().product::<usize>(),
                "with_params: parameter length differs"
            );
            // `flat` is on the inner device; `to_device` does not attach autodiff, so
            // `autodiff()` restores it where the old parameter had it.
            let new = flat.clone().reshape(dims).to_device(&old.device().inner());
            match (keep_autodiff && old.is_autodiff(), old.is_require_grad()) {
                (true, true) => new.autodiff().require_grad(),
                (true, false) => new.autodiff(),
                (false, _) => new,
            }
        })
    }
}

/// `model` with its float parameters replaced by `values`, which must come from
/// [`params`] on a model of the same structure (same [`ParamId`]s, same lengths).
///
/// Each parameter keeps its id, shape, device and `require_grad` flag: on an autodiff
/// device the new parameters are fresh leaves that collect gradients, as the old ones did.
///
/// # Panics
/// If `values` does not match the model's parameters in number, ids or lengths.
pub fn with_params<M: Module>(model: M, values: &ParamVec) -> M {
    replace(model, values, true)
}

/// [`with_params`], optionally dropping autodiff from the new parameters.
pub(super) fn replace<M: Module>(model: M, values: &ParamVec, keep_autodiff: bool) -> M {
    let mut mapper = Replace {
        parts: &values.parts,
        next: 0,
        keep_autodiff,
    };
    let model = model.map(&mut mapper);
    assert_eq!(
        mapper.next,
        values.parts.len(),
        "with_params: the ParamVec has more parameters than the model"
    );
    model
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::nn::{Linear, LinearConfig};

    fn linear(device: &Device) -> Linear {
        LinearConfig::new(3, 2).init(device)
    }

    /// Bits of every parameter, in visit order.
    fn bits(v: &ParamVec) -> Vec<u32> {
        v.to_flat()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    #[test]
    fn params_then_with_params_round_trips_bit_for_bit() {
        let device = Device::default();
        let model = linear(&device);
        let p = params(&model);
        assert_eq!(p.numel(), 3 * 2 + 2);
        let back = params(&with_params(model.clone(), &p));
        assert_eq!(bits(&back), bits(&p));
        assert_eq!(back.ids().collect::<Vec<_>>(), p.ids().collect::<Vec<_>>());
    }

    #[test]
    fn with_params_writes_new_values_and_keeps_shapes() {
        let device = Device::default();
        let model = linear(&device);
        let p = params(&model);
        let doubled = with_params(model.clone(), &p.scale(2.0));
        assert_eq!(doubled.weight.val().dims(), model.weight.val().dims());
        let expected = model.weight.val().mul_scalar(2.0).into_data();
        doubled
            .weight
            .val()
            .into_data()
            .assert_approx_eq::<f32>(&expected, Default::default());
    }

    #[test]
    fn with_params_keeps_require_grad_on_autodiff() {
        let device = Device::default().autodiff();
        let model = linear(&device);
        let p = params(&model);
        let new = with_params(model, &p.scale(2.0));
        let weight = new.weight.val();
        assert!(weight.is_autodiff() && weight.is_require_grad());
        let plain = replace(new, &p, false);
        assert!(!plain.weight.val().is_autodiff());
    }

    #[test]
    #[should_panic(expected = "parameter ids differ")]
    fn with_params_rejects_another_models_vector() {
        let device = Device::default();
        let _ = with_params(linear(&device), &params(&linear(&device)));
    }

    #[test]
    fn arithmetic_matches_the_flat_vectors() {
        let device = Device::default();
        let a = params(&linear(&device));
        let flat_a = a.to_flat().into_data().try_to_vec::<f32>().unwrap();
        let b = a.from_flat_like(Tensor::from_floats(
            (0..a.numel())
                .map(|i| 0.5 + i as f32)
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        ));
        let flat_b = b.to_flat().into_data().try_to_vec::<f32>().unwrap();
        let dot: f64 = flat_a
            .iter()
            .zip(&flat_b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        assert!((a.dot(&b) - dot).abs() <= 1e-5 * dot.abs().max(1.0));
        assert!(
            (b.norm() - flat_b.iter().map(|y| f64::from(y * y)).sum::<f64>().sqrt()).abs() < 1e-4
        );

        let axpy = a
            .axpy(-2.0, &b)
            .to_flat()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let quot = a.div(&b).to_flat().into_data().try_to_vec::<f32>().unwrap();
        let prod = a.mul(&b).to_flat().into_data().try_to_vec::<f32>().unwrap();
        for i in 0..a.numel() {
            assert!((axpy[i] - (flat_a[i] - 2.0 * flat_b[i])).abs() < 1e-6);
            assert!((quot[i] - flat_a[i] / flat_b[i]).abs() < 1e-6);
            assert!((prod[i] - flat_a[i] * flat_b[i]).abs() < 1e-6);
        }
        assert_eq!(a.zeros_like().norm(), 0.0);
    }
}
