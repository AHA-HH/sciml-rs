//! The Jacobian products of a least-squares problem: [`vjp`] (Jᵀu), [`jvp`] (J v) and
//! [`adjoint_error`], which checks one against the other.

use burn::{
    module::{Module, ModuleVisitor, Param, ParamId},
    optim::GradientsParams,
    prelude::*,
    tensor::DType,
};

use super::param_vec::{ParamVec, params, replace};

/// A least-squares problem Φ(θ) = ½‖r(θ)‖² over the parameters θ of a model `M`.
///
/// The problem owns the residual's definition (weights, normalisation, which outputs); the
/// optimiser sees only the flattened residual and the products built on it.
pub trait LeastSquares<M: Module> {
    /// One batch of data, in whatever form the residual needs.
    type Batch;

    /// The weighted residual r(θ) on `batch`, flattened to `[m]`.
    ///
    /// When `model` is on an autodiff device with parameters that require gradients, the
    /// result must be tracked, so that one backward gives Jᵀu ([`vjp`]).
    fn residual(&self, model: &M, batch: &Self::Batch) -> Tensor<1>;

    /// An exact J v, `[m]`, if the problem can compute one (e.g. a tangent pass through
    /// the model). The default is `None`, and [`jvp`] then takes central finite
    /// differences of [`residual`](Self::residual).
    fn jvp(&self, _model: &M, _batch: &Self::Batch, _v: &ParamVec) -> Option<Tensor<1>> {
        None
    }
}

/// Collects each parameter's gradient, flattened and without autodiff; zeros for a
/// parameter the residual does not depend on.
struct CollectGrads<'a> {
    grads: &'a GradientsParams,
    parts: Vec<(ParamId, Tensor<1>)>,
}

impl ModuleVisitor for CollectGrads<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let value = param.val().inner();
        let n: usize = value.dims().iter().product();
        let grad = match self.grads.get::<D>(param.id) {
            Some(g) => g.inner().reshape([n]),
            None => value.reshape([n]).zeros_like(),
        };
        self.parts.push((param.id, grad));
    }
}

/// Jᵀu: one fresh forward of `problem.residual` and one backward of Σ r ⊙ u, with u held
/// constant. The gradient of Φ, Jᵀr, is the case u = r.
///
/// `model` must be on an autodiff device (`Device::autodiff`) with parameters that require
/// gradients; `u` is `[m]`, the residual's length, on any device. The result has
/// [`params`]'s structure and is on the model's device without autodiff.
///
/// Each call costs a forward and a backward: Burn's backward consumes the graph, so the
/// forward cannot be reused (phase README, "Facts verified").
///
/// # Panics
/// If the residual is not tracked (the model is not on an autodiff device, or its
/// parameters do not require gradients), or if `u` is not `[m]`.
pub fn vjp<M: Module, P: LeastSquares<M>>(
    problem: &P,
    model: &M,
    batch: &P::Batch,
    u: &Tensor<1>,
) -> ParamVec {
    let r = problem.residual(model, batch);
    assert!(
        r.is_tracked(),
        "vjp: the residual is not tracked; put the model on an autodiff device with parameters that require gradients"
    );
    assert_eq!(r.dims(), u.dims(), "vjp: u must have the residual's length");
    let u = u.clone().inner().to_device(&r.device());
    let grads = GradientsParams::from_grads((r * u).sum().backward(), model);
    let mut collect = CollectGrads {
        grads: &grads,
        parts: Vec::new(),
    };
    model.visit(&mut collect);
    ParamVec::from_parts(collect.parts)
}

/// The default relative step of [`jvp`]'s central difference for parameters of `dtype`:
/// the cube root of the unit roundoff, the step that balances the O(ε²) truncation error
/// against the O(u/ε) rounding error: about 4.9e-3 in f32, 6.1e-6 in f64.
pub fn default_rel_step(dtype: DType) -> f64 {
    match dtype {
        DType::F64 => f64::EPSILON.cbrt(),
        _ => f64::from(f32::EPSILON).cbrt(),
    }
}

/// J v, `[m]`, for a parameter direction `v` with [`params`]'s structure.
///
/// If `problem.jvp` returns `Some`, that is the result. Otherwise the central difference
///
///   J v ≈ [r(θ + εv) − r(θ − εv)] / 2ε,  ε = h · max(‖θ‖, 1) / ‖v‖,
///
/// evaluated with parameters that carry no autodiff (no graph is recorded), so `model` may
/// be on an autodiff device or not. The step scales with the parameters, so the
/// perturbation εv has norm h · max(‖θ‖, 1) whatever ‖v‖ is. `rel_step` is h; `None` takes
/// [`default_rel_step`] for the parameters' dtype. The result has no autodiff association.
/// For v = 0 the result is zeros, from one forward.
///
/// Costs two forwards. Its error is O(h²) from truncation plus O(u/h) from rounding,
/// which [`adjoint_error`] measures; ReLU kinks between θ ± εv add to it.
///
/// # Panics
/// If `v` does not match the model's parameters.
pub fn jvp<M: Module, P: LeastSquares<M>>(
    problem: &P,
    model: &M,
    batch: &P::Batch,
    v: &ParamVec,
    rel_step: Option<f64>,
) -> Tensor<1> {
    if let Some(jv) = problem.jvp(model, batch, v) {
        return jv.inner();
    }
    let theta = params(model);
    let v_norm = v.norm();
    if v_norm == 0.0 {
        let r = problem.residual(&replace(model.clone(), &theta, false), batch);
        return r.inner().zeros_like();
    }
    let h = rel_step.unwrap_or_else(|| default_rel_step(theta.dtype().unwrap_or(DType::F32)));
    let eps = h * theta.norm().max(1.0) / v_norm;
    let at = |step: f64| {
        let shifted = replace(model.clone(), &theta.axpy(step, v), false);
        problem.residual(&shifted, batch).inner()
    };
    (at(eps) - at(-eps)).div_scalar(2.0 * eps)
}

/// |⟨u, J v⟩ − ⟨Jᵀu, v⟩| / (‖u‖ ‖J v‖): how far [`jvp`] (default step) is from the adjoint
/// of [`vjp`]. Zero up to rounding for an exact J v; for a finite-difference J v it
/// measures the difference noise along v. Returns 0 when both inner products vanish
/// and ‖u‖ ‖J v‖ = 0, and infinity when only the denominator does.
///
/// Costs a [`jvp`] and a [`vjp`]; `model` must be on an autodiff device, as for [`vjp`].
///
/// # Panics
/// As [`vjp`] and [`jvp`].
pub fn adjoint_error<M: Module, P: LeastSquares<M>>(
    problem: &P,
    model: &M,
    batch: &P::Batch,
    u: &Tensor<1>,
    v: &ParamVec,
) -> f64 {
    let jv = jvp(problem, model, batch, v, None);
    let jtu = vjp(problem, model, batch, u);
    let u = u.clone().inner();
    let lhs = (u.clone() * jv.clone()).sum().into_scalar::<f64>();
    let rhs = jtu.dot(v);
    let denom = u.powf_scalar(2.0).sum().into_scalar::<f64>().sqrt()
        * jv.powf_scalar(2.0).sum().into_scalar::<f64>().sqrt();
    let diff = (lhs - rhs).abs();
    if denom > 0.0 {
        diff / denom
    } else if diff == 0.0 {
        0.0
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::optim::gauss_newton::with_params;
    use burn::nn::{Linear, LinearConfig};

    /// r(θ) = tanh(W x + b) − y on a fixed batch: nonlinear in θ, so a finite-difference
    /// J v is not exact.
    struct TanhFit;

    /// Inputs `[n, 3]` and targets `[n, 2]`.
    struct Data {
        x: Tensor<2>,
        y: Tensor<2>,
    }

    impl LeastSquares<Linear> for TanhFit {
        type Batch = Data;
        fn residual(&self, model: &Linear, batch: &Data) -> Tensor<1> {
            let out = model.forward(batch.x.clone()).tanh() - batch.y.clone();
            let [n, c] = out.dims();
            out.reshape([n * c])
        }
    }

    /// TanhFit with a deliberately wrong (sign-flipped) "exact" J v.
    struct FlippedJvp;

    impl LeastSquares<Linear> for FlippedJvp {
        type Batch = Data;
        fn residual(&self, model: &Linear, batch: &Data) -> Tensor<1> {
            TanhFit.residual(model, batch)
        }
        fn jvp(&self, model: &Linear, batch: &Data, v: &ParamVec) -> Option<Tensor<1>> {
            Some(jvp(&TanhFit, model, batch, v, None).neg())
        }
    }

    fn deterministic(n: usize, phase: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.37 + phase).sin()).collect()
    }

    fn setup(device: &Device) -> (Linear, Data) {
        let model = LinearConfig::new(3, 2).init(device);
        let model = with_params(
            model.clone(),
            &params(&model).from_flat_like(Tensor::from_floats(
                deterministic(8, 0.1).as_slice(),
                device,
            )),
        );
        let x = Tensor::<1>::from_floats(deterministic(12, 1.3).as_slice(), device).reshape([4, 3]);
        let y = Tensor::<1>::from_floats(deterministic(8, 2.1).as_slice(), device)
            .reshape([4, 2])
            .mul_scalar(0.5);
        (model, Data { x, y })
    }

    /// J by rows: row i is Jᵀ e_i, one VJP per output.
    fn explicit_jacobian(model: &Linear, data: &Data, m: usize) -> Vec<Vec<f32>> {
        let device = model.weight.val().device();
        (0..m)
            .map(|i| {
                let mut e = vec![0.0f32; m];
                e[i] = 1.0;
                let e = Tensor::from_floats(e.as_slice(), &device.clone().inner());
                vjp(&TanhFit, model, data, &e)
                    .to_flat()
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap()
            })
            .collect()
    }

    fn to_vec(t: Tensor<1>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    fn max_rel(a: &[f32], b: &[f32]) -> f32 {
        let scale = b.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        a.iter()
            .zip(b)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
            / scale
    }

    #[test]
    fn vjp_and_jvp_match_the_explicit_jacobian() {
        let device = Device::default().autodiff();
        let (model, data) = setup(&device);
        let m = 8;
        let j = explicit_jacobian(&model, &data, m);
        let p = j[0].len();
        assert_eq!(p, 8);

        let u = deterministic(m, 0.7);
        let jtu_oracle: Vec<f32> = (0..p)
            .map(|k| (0..m).map(|i| j[i][k] * u[i]).sum())
            .collect();
        let jtu = vjp(
            &TanhFit,
            &model,
            &data,
            &Tensor::from_floats(u.as_slice(), &device),
        );
        let jtu = to_vec(jtu.to_flat());
        assert!(
            max_rel(&jtu, &jtu_oracle) < 1e-5,
            "{jtu:?} vs {jtu_oracle:?}"
        );

        let theta = params(&model);
        let v_flat = deterministic(p, 1.9);
        let v = theta.from_flat_like(Tensor::from_floats(v_flat.as_slice(), &device));
        let jv_oracle: Vec<f32> = (0..m)
            .map(|i| (0..p).map(|k| j[i][k] * v_flat[k]).sum())
            .collect();
        let jv = to_vec(jvp(&TanhFit, &model, &data, &v, None));
        assert!(max_rel(&jv, &jv_oracle) < 1e-3, "{jv:?} vs {jv_oracle:?}");
    }

    #[test]
    fn jvp_of_zero_is_zero_and_exact_jvp_is_used() {
        let device = Device::default();
        let (model, data) = setup(&device);
        let zero = params(&model).zeros_like();
        let jv = jvp(&TanhFit, &model, &data, &zero, None);
        assert_eq!(jv.dims(), [8]);
        assert!(to_vec(jv).iter().all(|x| *x == 0.0));

        let v = params(&model).scale(0.3);
        let fd = to_vec(jvp(&TanhFit, &model, &data, &v, None));
        let flipped = to_vec(jvp(&FlippedJvp, &model, &data, &v, None));
        assert!(fd.iter().zip(&flipped).all(|(a, b)| *a == -*b));
    }

    #[test]
    fn adjoint_error_is_small_and_catches_a_wrong_jvp() {
        let device = Device::default().autodiff();
        let (model, data) = setup(&device);
        let u = Tensor::from_floats(deterministic(8, 0.4).as_slice(), &device);
        let theta = params(&model);
        let v = theta.from_flat_like(Tensor::from_floats(
            deterministic(8, 2.6).as_slice(),
            &device,
        ));
        let good = adjoint_error(&TanhFit, &model, &data, &u, &v);
        assert!(good < 1e-3, "adjoint error {good}");
        let bad = adjoint_error(&FlippedJvp, &model, &data, &u, &v);
        assert!(bad > 0.5, "a sign-flipped J v must be caught, got {bad}");
    }

    #[test]
    #[should_panic(expected = "not tracked")]
    fn vjp_needs_an_autodiff_model() {
        let device = Device::default();
        let (model, data) = setup(&device);
        let u = Tensor::from_floats(deterministic(8, 0.4).as_slice(), &device);
        let _ = vjp(&TanhFit, &model, &data, &u);
    }
}
