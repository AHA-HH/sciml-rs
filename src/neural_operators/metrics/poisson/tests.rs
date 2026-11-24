use super::*;
use crate::neural_operators::chebyshev::nodes;
use crate::neural_operators::chebyshev::{cheb_to_uniform, uniform_to_cheb};
use crate::neural_operators::losses::data_losses::{LpLoss, Reduction};
use burn::prelude::*;
use ndarray::{Array3, Array4};

/// `count` smooth fields on the n × n CGL grid that vanish on ∂Ω; sample k is scaled
/// and shifted so the samples differ.
fn vanishing_u(count: usize, n: usize) -> Array3<f64> {
    let x = nodes(n);
    Array3::from_shape_fn((count, n, n), |(k, i, j)| {
        let (a, b) = (x[i], x[j]);
        (1.0 + k as f64) * (1.0 - a * a) * (1.0 - b * b) * (a + 0.3 * k as f64).exp() * b.cos()
    })
}

/// Deterministic noise in [−1, 1), a hash of the index.
fn noise(k: usize, i: usize, j: usize) -> f64 {
    let t = (k as f64 * 3.17 + i as f64 * 12.9898 + j as f64 * 78.233).sin() * 43758.5453;
    2.0 * (t - t.floor()) - 1.0
}

/// T_cu applied to every sample of `u: [N, n, n]`, giving `[N, n − 1, n − 1]`.
fn to_uniform(u: &Array3<f64>) -> Array3<f64> {
    Transfers::new(u.dim().1).to_uniform_all(u.view())
}

// the floor identity of design §13.3 at s = n - 1 and at coarser learning grids: a
// prediction equal to R u has error 3 = 0 and error 4 = error 2
#[test]
fn exact_prediction_has_only_transfer_error() {
    let n = 17;
    let u = vanishing_u(3, n);
    for s in [n - 1, 8, 5] {
        let transfers = Transfers::with_s(n, s);
        let pred = transfers.to_uniform_all(u.view());
        let e = poisson_errors(pred.view(), u.view(), &transfers);
        assert_exact_prediction(&e, s);
    }
}

fn assert_exact_prediction(e: &PoissonErrors, s: usize) {
    for k in 0..3 {
        assert!(e.model[k] <= 1e-14, "s = {s}: e3[{k}] = {:e}", e.model[k]);
        assert!(
            (e.total[k] - e.transfer[k]).abs() <= 1e-14,
            "e4 {:e} vs e2 {:e}",
            e.total[k],
            e.transfer[k]
        );
        assert!(
            e.transfer[k] > 0.0 && e.transfer[k] < 1e-1,
            "s = {s}: e2 = {:e}",
            e.transfer[k]
        );
        assert!(
            e.boundary_max_ratio[k] <= 1e-14,
            "{:e}",
            e.boundary_max_ratio[k]
        );
        assert!(
            e.boundary_rel_rms[k] <= 1e-14,
            "{:e}",
            e.boundary_rel_rms[k]
        );
    }
}

#[test]
fn boundary_offset_is_measured() {
    // u_k = (1 + k)(1 − x²)(1 − y²): max |u_k| = 1 + k at the origin (a CGL node for
    // odd n), and ∫∫ u_k² dA = (1 + k)² (16/15)², which Clenshaw–Curtis integrates
    // exactly (degree 4 per axis ≤ n − 1)
    let n = 17;
    let c = 0.037;
    let x = nodes(n);
    let u = Array3::from_shape_fn((2, n, n), |(k, i, j)| {
        (1.0 + k as f64) * (1.0 - x[i] * x[i]) * (1.0 - x[j] * x[j])
    });
    let pred = to_uniform(&u) + c;
    let e = poisson_errors(pred.view(), u.view(), &Transfers::new(n));
    for k in 0..2 {
        let scale = 1.0 + k as f64;
        // û = c on ∂Ω: T_uc reproduces constants and both grids share the endpoints.
        // ∮ û² ds = 8c² with each corner weighted by both its edges, so the RMS is c
        let ratio = c / scale;
        let rms = c / (scale * 16.0 / 15.0 / 2.0);
        assert!(
            (e.boundary_max_ratio[k] - ratio).abs() <= 1e-12 * ratio,
            "ratio {} vs {ratio}",
            e.boundary_max_ratio[k]
        );
        assert!(
            (e.boundary_rel_rms[k] - rms).abs() <= 1e-12 * rms,
            "rms {} vs {rms}",
            e.boundary_rel_rms[k]
        );
    }
}

#[test]
fn error3_matches_lploss() {
    let (count, n) = (4, 17);
    let s = n - 1;
    let u = vanishing_u(count, n);
    let target = to_uniform(&u);
    let pred = Array3::from_shape_fn((count, s, s), |(k, i, j)| {
        target[[k, i, j]] + 0.05 * noise(k, i, j)
    });
    let e = poisson_errors(pred.view(), u.view(), &Transfers::new(n));
    let mean = e.model.iter().sum::<f64>() / count as f64;

    let device: Device = Default::default();
    let flat = |a: &Array3<f64>| {
        let data: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        Tensor::<2>::from_data(TensorData::new(data, vec![count, s * s]), &device)
    };
    let loss = LpLoss::new(2, Reduction::Sum).rel(flat(&pred), flat(&target));
    let lp = f64::from(loss.into_scalar::<f32>()) / count as f64;
    assert!((mean - lp).abs() <= 1e-6, "e3 mean {mean} vs LpLoss {lp}");
}

#[test]
#[should_panic(expected = "pred has 2 samples but u has 3")]
fn sample_count_mismatch_panics() {
    let u = vanishing_u(3, 9);
    let pred = Array3::<f64>::zeros((2, 8, 8));
    poisson_errors(pred.view(), u.view(), &Transfers::new(9));
}

#[test]
#[should_panic(expected = "must be on the transfers' grids (s, n)")]
fn wrong_uniform_size_panics() {
    let u = vanishing_u(1, 9);
    let pred = Array3::<f64>::zeros((1, 9, 9));
    poisson_errors(pred.view(), u.view(), &Transfers::new(9));
}

#[test]
#[should_panic(expected = "must be on the transfers' grids (s, n)")]
fn wrong_cgl_size_panics() {
    let u = vanishing_u(1, 9);
    let pred = Array3::<f64>::zeros((1, 4, 4));
    poisson_errors(pred.view(), u.view(), &Transfers::with_s(17, 4));
}

#[test]
fn u_norm_is_the_cc_norm() {
    // u = (1 - x²)(1 - y²): ∫∫ u² dA = (16/15)², exact in Clenshaw–Curtis
    let n = 9;
    let x = nodes(n);
    let u = Array3::from_shape_fn((1, n, n), |(_, i, j)| {
        (1.0 - x[i] * x[i]) * (1.0 - x[j] * x[j])
    });
    let transfers = Transfers::with_s(n, 4);
    let pred = transfers.to_uniform_all(u.view());
    let e = poisson_errors(pred.view(), u.view(), &transfers);
    assert!(
        (e.u_norm[0] - 16.0 / 15.0).abs() <= 1e-14,
        "{}",
        e.u_norm[0]
    );
}

#[test]
#[should_panic(expected = "pred must be [N, s, s]")]
fn non_square_pred_panics() {
    let u = vanishing_u(1, 9);
    let pred = Array3::<f64>::zeros((1, 8, 7));
    poisson_errors(pred.view(), u.view(), &Transfers::new(9));
}

#[test]
#[should_panic(expected = "u must be [N, n, n]")]
fn non_square_u_panics() {
    let u = Array3::<f64>::zeros((1, 9, 8));
    let pred = Array3::<f64>::zeros((1, 8, 8));
    poisson_errors(pred.view(), u.view(), &Transfers::new(9));
}

#[test]
#[should_panic(expected = "n must be >= 4")]
fn too_small_n_panics() {
    let u = Array3::<f64>::zeros((1, 3, 3));
    let pred = Array3::<f64>::zeros((1, 2, 2));
    poisson_errors(pred.view(), u.view(), &Transfers::new(3));
}

/// One metrics row per epoch with the given `test_l2`.
fn metrics(test_l2: &[f32]) -> Vec<EpochMetrics> {
    test_l2
        .iter()
        .enumerate()
        .map(|(epoch, &test_l2)| EpochMetrics {
            epoch,
            train_mse: 0.0,
            train_l2: 0.0,
            test_l2,
            current_lr: 1e-3,
            epoch_time_s: None,
        })
        .collect()
}

/// Synthetic per-sample errors: error 4 is `total`, error 2 `transfer`, error 3 0.25
/// (the test_l2 of [`last_epoch`]) and the relative boundary RMS `rms`.
fn synthetic(total: &[f64], transfer: f64, rms: f64) -> PoissonErrors {
    let k = total.len();
    PoissonErrors {
        transfer: vec![transfer; k],
        model: vec![0.25; k],
        total: total.to_vec(),
        u_norm: vec![1.0; k],
        boundary_max_ratio: vec![0.0; k],
        boundary_rel_rms: vec![rms; k],
    }
}

fn last_epoch() -> EpochMetrics {
    metrics(&[0.5, 0.25]).pop().unwrap()
}

/// The fixture sidecar (n = 65), claiming `count` samples with error 2 `transfer`.
fn sidecar(count: usize, transfer: f64) -> SidecarRecord {
    let mut sidecar = SidecarRecord::load_binary(
        include_str!("../../../../tests/fixtures/poisson_n65_K32_test.json").as_bytes(),
    )
    .unwrap();
    sidecar.count = count;
    (sidecar.error2.mean, sidecar.error2.max) = (transfer, transfer);
    sidecar
}

fn eval(errors: &PoissonErrors, sidecar: &SidecarRecord) -> PoissonEvaluation {
    eval_at(errors, sidecar, 64)
}

fn eval_at(errors: &PoissonErrors, sidecar: &SidecarRecord, s: usize) -> PoissonEvaluation {
    evaluate(
        errors,
        &Transfers::with_s(65, s),
        sidecar,
        &last_epoch(),
        "Cpu".into(),
        vec![],
        Path::new("runs/x"),
    )
}

#[test]
fn evaluate_picks_the_first_worst_sample() {
    let e = synthetic(&[0.1, 0.3, 0.2, 0.3], 1e-5, 0.0);
    let ev = eval(&e, &sidecar(4, 1e-5));
    assert_eq!(ev.record.error4_worst_sample, 1);
    assert!(ev.record.cross_checks_passed, "{ev}");
    assert!(ev.to_string().contains("worst sample (max error 4): 1;"));

    // a NaN is the max, as ErrorStats reports it
    let e = synthetic(&[0.1, 0.3, f64::NAN, 0.2], 1e-5, 0.0);
    assert_eq!(eval(&e, &sidecar(4, 1e-5)).record.error4_worst_sample, 2);
}

#[test]
fn evaluate_boundary_verdict_at_the_threshold() {
    // the ratio is the mean RMS over the mean error 4, here 0.5
    let total = [0.4, 0.6];
    let below = eval(&synthetic(&total, 1e-5, 0.0499), &sidecar(2, 1e-5));
    assert!(below.record.boundary_to_total < BOUNDARY_THRESHOLD);
    assert!(!below.record.ablation_warranted);
    assert!(below.to_string().ends_with("ablation is not warranted"));
    let above = eval(&synthetic(&total, 1e-5, 0.0501), &sidecar(2, 1e-5));
    assert!(above.record.ablation_warranted);
    assert!(above.to_string().ends_with("ablation is warranted"));
}

#[test]
fn evaluate_flags_a_cross_check_mismatch() {
    let e = synthetic(&[0.1, 0.2], 1e-5, 0.0);
    let ok = eval(&e, &sidecar(2, 1e-5));
    assert!(
        ok.record.error2_compared && ok.record.cross_checks_passed,
        "{ok}"
    );

    // error 2 off the sidecar by 1e-9, relative
    let bad = eval(&e, &sidecar(2, 1e-5 * (1.0 + 1e-9)));
    assert!(!bad.error2.passed && !bad.record.cross_checks_passed);
    assert!(bad.to_string().contains("(tol 1e-12) FAILED"), "{bad}");

    // the same mismatch is skipped when the sidecar covers other samples
    let skipped = eval(&e, &sidecar(200, 1e-5 * (1.0 + 1e-9)));
    assert!(!skipped.record.error2_compared && skipped.record.cross_checks_passed);
    assert!(
        skipped
            .to_string()
            .contains("covers 200 samples but 2 were evaluated")
    );

    // error 3 off the run's last test_l2
    let mut e3_off = e.clone();
    e3_off.model = vec![0.3; 2];
    let bad = eval(&e3_off, &sidecar(2, 1e-5));
    assert!(!bad.record.cross_checks_passed);
    assert!(!bad.error3.passed);
}

#[test]
fn evaluate_skips_the_sidecar_at_another_s() {
    // error 2 at s = 16 is not the sidecar's (s = 64), so it is not compared
    let e = synthetic(&[0.1, 0.2], 3e-3, 0.0);
    let ev = eval_at(&e, &sidecar(2, 1e-5), 16);
    assert_eq!(ev.record.s, 16);
    assert!(!ev.record.error2_compared && ev.record.cross_checks_passed);
    assert!(
        ev.to_string()
            .contains("the sidecar's error 2 is at s = 64, this run's at s = 16"),
        "{ev}"
    );
    assert!(ev.to_string().contains("(n = 65, s = 16, 2 samples)"));
}

#[test]
fn log_correlation_of_error_with_norm() {
    // error 4 = c / ||u||: ln e = ln c - ln ||u||, correlation -1
    let norm = [0.5, 1.0, 2.0, 4.0];
    let err: Vec<f64> = norm.iter().map(|v| 0.01 / v).collect();
    assert!((log_correlation(&err, &norm) + 1.0).abs() <= 1e-12);
    assert!((log_correlation(&norm, &norm) - 1.0).abs() <= 1e-12);
    assert!(log_correlation(&[0.1], &[1.0]).is_nan());

    let mut e = synthetic(&[0.1, 0.05, 0.025], 1e-5, 0.0);
    e.u_norm = vec![1.0, 2.0, 4.0];
    let ev = eval(&e, &sidecar(3, 1e-5));
    assert!((ev.record.error4_u_norm_log_correlation + 1.0).abs() <= 1e-12);
    assert_eq!(ev.record.u_norm_per_sample, [1.0, 2.0, 4.0]);
    assert_eq!(ev.record.error4_per_sample, [0.1, 0.05, 0.025]);
}

#[test]
#[should_panic(expected = "sidecar n, d differ")]
fn evaluate_rejects_another_grid() {
    let e = synthetic(&[0.1], 1e-5, 0.0);
    evaluate(
        &e,
        &Transfers::new(33),
        &sidecar(1, 1e-5),
        &last_epoch(),
        "Cpu".into(),
        vec![],
        Path::new("x"),
    );
}

#[test]
fn error3_check_uses_the_last_epoch() {
    let rows = metrics(&[0.5, 0.25]);
    let last = rows.last().unwrap();
    let check = check_error3(last, 0.25 * (1.0 + 5e-6));
    assert_eq!((check.epoch, check.test_l2), (1, 0.25));
    assert!(check.passed, "{check}");
    let check = check_error3(last, 0.25 * (1.0 + 2e-5));
    assert!(!check.passed, "{check}");
    assert!(check.to_string().ends_with("FAILED"));
}

// ---- any rank (3D CL1.3) ----

/// Deterministic noise in [−1, 1) for a 3D sample, a hash of the index.
fn noise3(k: usize, i: usize, j: usize, l: usize) -> f64 {
    let t = (k as f64 * 3.17 + i as f64 * 12.9898 + j as f64 * 78.233 + l as f64 * 37.719).sin()
        * 43758.5453;
    2.0 * (t - t.floor()) - 1.0
}

/// `count` smooth 3D fields on the n³ CGL grid that vanish on ∂Ω, stored 'ijk'.
fn vanishing_u3(count: usize, n: usize) -> Array4<f64> {
    let x = nodes(n);
    Array4::from_shape_fn((count, n, n, n), |(k, i, j, l)| {
        let (a, b, c) = (x[i], x[j], x[l]);
        (1.0 + k as f64)
            * (1.0 - a * a)
            * (1.0 - b * b)
            * (1.0 - c * c)
            * (a + 0.3 * k as f64).exp()
            * b.cos()
            * (1.0 + 0.5 * c)
    })
}

/// Σ_abc t[i, a] t[j, b] t[l, c] g[a, b, c]: the 1D transfer `t` along each axis,
/// summed term by term.
fn transfer3_direct(t: &Array2<f64>, g: ArrayView3<f64>) -> Array3<f64> {
    let (m, k) = t.dim();
    Array3::from_shape_fn((m, m, m), |(i, j, l)| {
        let mut sum = 0.0;
        for a in 0..k {
            for b in 0..k {
                for c in 0..k {
                    sum += t[[i, a]] * t[[j, b]] * t[[l, c]] * g[[a, b, c]];
                }
            }
        }
        sum
    })
}

fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol * want.abs(),
        "{what}: {got:e} vs {want:e}"
    );
}

#[test]
fn nd_errors_match_direct_computation() {
    let (count, n) = (2, 9);
    // a boundary that does not vanish, so error 5 has both terms
    let u = vanishing_u3(count, n)
        + Array4::from_shape_fn((count, n, n, n), |(k, i, j, l)| 0.1 * noise3(k, i, j, l));
    let w = clencurt(n);
    for s in [n - 1, 5] {
        let transfers = Transfers::with_s(n, s);
        let pred = transfers.to_uniform_all_nd(u.view())
            + Array4::from_shape_fn((count, s, s, s), |(k, i, j, l)| {
                0.05 * noise3(k + 7, i, j, l)
            });
        let e = poisson_errors_nd(pred.view(), u.view(), &transfers);

        let r = cheb_to_uniform(n, s);
        let p = uniform_to_cheb(s, n, TRANSFER_DEGREE);
        for k in 0..count {
            let ui = u.index_axis(Axis(0), k);
            let ru = transfer3_direct(&r, ui);
            let round_trip = transfer3_direct(&p, ru.view());
            let u_hat = transfer3_direct(&p, pred.index_axis(Axis(0), k));

            // Σ_ijl w_i w_j w_l v², term by term
            let cc = |v: &dyn Fn(usize, usize, usize) -> f64| {
                let mut sum = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        for l in 0..n {
                            sum += w[i] * w[j] * w[l] * v(i, j, l).powi(2);
                        }
                    }
                }
                sum
            };
            let u_sq = cc(&|i, j, l| ui[[i, j, l]]);
            let e2 = (cc(&|i, j, l| round_trip[[i, j, l]] - ui[[i, j, l]]) / u_sq).sqrt();
            let e4 = (cc(&|i, j, l| u_hat[[i, j, l]] - ui[[i, j, l]]) / u_sq).sqrt();
            let pk = pred.index_axis(Axis(0), k);
            let num: f64 = pk.iter().zip(&ru).map(|(a, b)| (a - b).powi(2)).sum();
            let den: f64 = ru.iter().map(|b| b * b).sum();
            let e3 = (num / den).sqrt();

            // the six faces, one by one; edge and corner nodes once per face
            let last = n - 1;
            let (mut surface, mut max_face) = (0.0, 0.0_f64);
            for a in 0..n {
                for b in 0..n {
                    for v in [
                        u_hat[[0, a, b]],
                        u_hat[[last, a, b]],
                        u_hat[[a, 0, b]],
                        u_hat[[a, last, b]],
                        u_hat[[a, b, 0]],
                        u_hat[[a, b, last]],
                    ] {
                        surface += w[a] * w[b] * v * v;
                        max_face = max_face.max(v.abs());
                    }
                }
            }
            let max_u = ui.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let rms = (surface / 24.0).sqrt() / (u_sq / 8.0).sqrt();

            let tol = 1e-13;
            assert_rel(e.transfer[k], e2, tol, "error 2");
            assert_rel(e.model[k], e3, tol, "error 3");
            assert_rel(e.total[k], e4, tol, "error 4");
            assert_rel(e.u_norm[k], u_sq.sqrt(), tol, "||u||");
            assert_rel(e.boundary_max_ratio[k], max_face / max_u, tol, "max ratio");
            assert_rel(e.boundary_rel_rms[k], rms, tol, "boundary RMS");
        }
    }
}

// the floor identity of design §13.3 in 3D: a prediction equal to R u has error 3 = 0
// and error 4 = error 2
#[test]
fn nd_exact_prediction_has_only_transfer_error() {
    let n = 11;
    let u = vanishing_u3(3, n);
    for s in [n - 1, 6] {
        let transfers = Transfers::with_s(n, s);
        let pred = transfers.to_uniform_all_nd(u.view());
        let e = poisson_errors_nd(pred.view(), u.view(), &transfers);
        assert_exact_prediction(&e, s);
    }
}

#[test]
fn nd_boundary_offset_is_measured() {
    // u_k = c_k (1 − x²)(1 − y²)(1 − z²): max |u_k| = c_k at the origin (a CGL node for
    // odd n), and ∭ u_k² dV = c_k² (16/15)³, exact in Clenshaw–Curtis (degree 4 per
    // axis ≤ n − 1)
    let n = 9;
    let c = 0.037;
    let x = nodes(n);
    let u = Array4::from_shape_fn((2, n, n, n), |(k, i, j, l)| {
        (1.0 + k as f64) * (1.0 - x[i] * x[i]) * (1.0 - x[j] * x[j]) * (1.0 - x[l] * x[l])
    });
    let transfers = Transfers::new(n);
    let pred = transfers.to_uniform_all_nd(u.view()) + c;
    let e = poisson_errors_nd(pred.view(), u.view(), &transfers);
    for k in 0..2 {
        let scale = 1.0 + k as f64;
        // û = c on ∂Ω (T_uc reproduces constants, both grids share the endpoints), so
        // ∯ û² dS = 24 c² with each edge and corner node weighted by every face it is
        // on, and the RMS is c over √(∭ u² dV / 8)
        let ratio = c / scale;
        let rms = c / (scale * (16.0_f64 / 15.0).powf(1.5) / 8.0_f64.sqrt());
        assert_rel(e.boundary_max_ratio[k], ratio, 1e-12, "ratio");
        assert_rel(e.boundary_rel_rms[k], rms, 1e-12, "rms");
    }
}

#[test]
fn nd_u_norm_is_the_cc_norm() {
    // u = (1 − x²)(1 − y²)(1 − z²): ∭ u² dV = (16/15)³, exact in Clenshaw–Curtis
    let n = 9;
    let x = nodes(n);
    let u = Array4::from_shape_fn((1, n, n, n), |(_, i, j, l)| {
        (1.0 - x[i] * x[i]) * (1.0 - x[j] * x[j]) * (1.0 - x[l] * x[l])
    });
    let transfers = Transfers::with_s(n, 4);
    let pred = transfers.to_uniform_all_nd(u.view());
    let e = poisson_errors_nd(pred.view(), u.view(), &transfers);
    assert_rel(e.u_norm[0], (16.0_f64 / 15.0).powf(1.5), 1e-14, "||u||");
}

// the generic face code against the trusted 2D path (four edges, 8 against 4)
#[test]
fn nd_boundary_at_rank_two_matches_2d() {
    let n = 17;
    let w = clencurt(n);
    let u = vanishing_u(2, n);
    for k in 0..2 {
        let ui = u.index_axis(Axis(0), k);
        let u_hat = Array2::from_shape_fn((n, n), |(i, j)| ui[[i, j]] + 0.02 * noise(k, i, j));
        let (ratio, rms) = boundary_errors(&u_hat, ui, &w);
        let (ratio_nd, rms_nd) = boundary_errors_nd(u_hat.view().into_dyn(), ui.into_dyn(), &w);
        assert_rel(ratio_nd, ratio, 1e-14, "ratio");
        assert_rel(rms_nd, rms, 1e-14, "rms");
    }
}

/// The bits of every vector of `e`, to compare two results exactly.
fn bits(e: &PoissonErrors) -> Vec<Vec<u64>> {
    [
        &e.transfer,
        &e.model,
        &e.total,
        &e.u_norm,
        &e.boundary_max_ratio,
        &e.boundary_rel_rms,
    ]
    .iter()
    .map(|v| v.iter().map(|x| x.to_bits()).collect())
    .collect()
}

// D2: 2D fields through poisson_errors_nd are poisson_errors bit for bit, at s = n − 1
// and a coarser s, in standard layout, transposed, and through IxDyn
#[test]
fn nd_rank_two_is_bit_identical_to_2d() {
    let (count, n) = (3, 17);
    let u = vanishing_u(count, n)
        + Array3::from_shape_fn((count, n, n), |(k, i, j)| 0.01 * noise(k + 5, i, j));
    for s in [n - 1, 8] {
        let transfers = Transfers::with_s(n, s);
        let pred = transfers.to_uniform_all(u.view())
            + Array3::from_shape_fn((count, s, s), |(k, i, j)| 0.05 * noise(k, i, j));
        let old = bits(&poisson_errors(pred.view(), u.view(), &transfers));
        assert_eq!(
            bits(&poisson_errors_nd(pred.view(), u.view(), &transfers)),
            old
        );
        assert_eq!(
            bits(&poisson_errors_nd(
                pred.view().into_dyn(),
                u.view().into_dyn(),
                &transfers
            )),
            old
        );
        let (pt, ut) = (
            pred.view().permuted_axes([0, 2, 1]),
            u.view().permuted_axes([0, 2, 1]),
        );
        assert_eq!(
            bits(&poisson_errors_nd(pt, ut, &transfers)),
            bits(&poisson_errors(pt, ut, &transfers))
        );
    }
}

#[test]
#[should_panic(expected = "pred has 2 samples but u has 3")]
fn nd_sample_count_mismatch_panics() {
    let u = vanishing_u3(3, 9);
    let pred = Array4::<f64>::zeros((2, 8, 8, 8));
    poisson_errors_nd(pred.view(), u.view(), &Transfers::new(9));
}

#[test]
#[should_panic(expected = "pred has rank 3 but u has rank 4")]
fn nd_rank_mismatch_panics() {
    let u = vanishing_u3(1, 9).into_dyn();
    let pred = Array3::<f64>::zeros((1, 8, 8)).into_dyn();
    poisson_errors_nd(pred.view(), u.view(), &Transfers::new(9));
}

#[test]
#[should_panic(expected = "on the transfers' grids (s, n) = (8, 9)")]
fn nd_non_cubic_pred_panics() {
    let u = vanishing_u3(1, 9);
    let pred = Array4::<f64>::zeros((1, 8, 8, 7));
    poisson_errors_nd(pred.view(), u.view(), &Transfers::new(9));
}

#[test]
#[should_panic(expected = "on the transfers' grids (s, n) = (4, 9)")]
fn nd_wrong_grid_panics() {
    let u = vanishing_u3(1, 9);
    let pred = Array4::<f64>::zeros((1, 8, 8, 8));
    poisson_errors_nd(pred.view(), u.view(), &Transfers::with_s(9, 4));
}

#[test]
#[should_panic(expected = "at least two spatial axes")]
fn nd_one_spatial_axis_panics() {
    let u = Array2::<f64>::zeros((1, 9));
    let pred = Array2::<f64>::zeros((1, 8));
    poisson_errors_nd(pred.view(), u.view(), &Transfers::new(9));
}
