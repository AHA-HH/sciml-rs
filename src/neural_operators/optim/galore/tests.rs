//! Unit tests of the Tensor GaLore projector (library CM3.6a, complex CM3.6b); its parity with
//! 2.0.0 is `tests/parity_galore*.rs`.

use super::*;

fn device() -> Device {
    Default::default()
}

/// A deterministic tensor of `dims` with entries in [−1, 1) (xorshift64*).
fn draw<const D: usize>(dims: [usize; D], seed: u64) -> Tensor<D> {
    let mut s = seed;
    let n = dims.iter().product();
    let values: Vec<f32> = (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let x = s.wrapping_mul(0x2545_f491_4f6c_dd1d);
            ((x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
        })
        .collect();
    Tensor::from_data(TensorData::new(values, dims), &device())
}

fn max_abs<const D: usize>(x: Tensor<D>) -> f64 {
    x.abs().max().into_scalar::<f64>()
}

#[test]
fn uniform_and_per_mode_ranks_are_as_given() {
    assert_eq!(GaLoreRank::Uniform(2).ranks([4, 5, 6]), [2, 2, 2]);
    assert_eq!(GaLoreRank::PerMode(vec![1, 3]).ranks([4, 5]), [1, 3]);
}

/// tensorly's equation solved by hand. The fixtures' (8, 6, 5) case, fraction 0.5, is checked
/// against 2.0.0's (5, 3, 3) in `tests/parity_galore.rs`.
#[test]
fn fraction_ranks_solve_tensorly_equation() {
    // Matrix 10 × 10, fraction 1: 100 c² + 200 c = 100, c = √2 − 1, ranks round(4.14) = 4.
    assert_eq!(GaLoreRank::Fraction(1.0).ranks([10, 10]), [4, 4]);
    // A tiny fraction keeps rank 1.
    assert_eq!(GaLoreRank::Fraction(1e-6).ranks([10, 10]), [1, 1]);
}

#[test]
#[should_panic(expected = "must be positive")]
fn fraction_must_be_positive() {
    GaLoreRank::Fraction(0.0).ranks([3, 3]);
}

#[test]
#[should_panic(expected = "one rank per mode")]
fn per_mode_needs_one_rank_per_mode() {
    GaLoreRank::PerMode(vec![1]).ranks([3, 3]);
}

#[test]
fn factors_are_orthonormal_with_positive_largest_entries() {
    let factors = tucker_factors(draw([6, 5, 4], 1), [3, 2, 2]);
    for (f, (n, r)) in factors.iter().zip([(6, 3), (5, 2), (4, 2)]) {
        assert_eq!(f.dims(), [n, r]);
        let gram = f.clone().transpose().matmul(f.clone());
        let eye = Tensor::<2>::eye(r, &device());
        assert!(max_abs(gram - eye) < 1e-5);
        let largest = f.clone().abs().argmax(0);
        let at = f.clone().gather(0, largest);
        assert!(at.greater_elem(0.0).all().into_scalar::<bool>());
    }
}

/// Full ranks: the factors are square orthogonal, so `project_back ∘ project` is the
/// identity (times `scale`).
#[test]
fn full_rank_round_trip_is_scale_times_identity() {
    let grad = draw([4, 3, 5], 2);
    let mut p = TensorGaLoreConfig {
        rank: GaLoreRank::PerMode(vec![4, 3, 5]),
        update_proj_gap: 1,
        scale: 0.5,
    }
    .init::<3, Tensor<3>>();
    let low = p.project(grad.clone(), 7);
    let back = p.project_back(low);
    assert!(max_abs(back - grad.mul_scalar(0.5)) < 1e-5);
}

/// A tensor of exact Tucker rank (2, 3, 2) is reproduced from its projection.
#[test]
fn exact_tucker_tensor_is_reproduced() {
    let core = draw([2, 3, 2], 3);
    let a = [draw([5, 2], 4), draw([6, 3], 5), draw([4, 2], 6)];
    let x = core.from_low_rank(&a);
    let mut p = TensorGaLoreConfig::new(GaLoreRank::PerMode(vec![2, 3, 2])).init::<3, Tensor<3>>();
    let low = p.project(x.clone(), 0);
    assert_eq!(low.dims(), [2, 3, 2]);
    assert!(max_abs(p.project_back(low) - x.clone()) < 1e-5 * max_abs(x));
}

/// As 2.0.0, later calls keep the first factors, even at the next gap.
#[test]
fn factors_are_not_refreshed() {
    let mut p = TensorGaLoreConfig {
        rank: GaLoreRank::Uniform(2),
        update_proj_gap: 3,
        scale: 1.0,
    }
    .init::<2, Tensor<2>>();
    p.project(draw([5, 4], 7), 0);
    let first = p.factors().unwrap()[0].clone();
    p.project(draw([5, 4], 8), 3);
    assert_eq!(max_abs(p.factors().unwrap()[0].clone() - first), 0.0);
}

#[test]
#[should_panic(expected = "multiple of update_proj_gap")]
fn first_project_needs_the_gap() {
    TensorGaLoreConfig::new(GaLoreRank::Uniform(1))
        .init::<2, Tensor<2>>()
        .project(draw([3, 3], 9), 1);
}

#[test]
#[should_panic(expected = "project_back before project")]
fn project_back_needs_factors() {
    TensorGaLoreConfig::new(GaLoreRank::Uniform(1))
        .init::<2, Tensor<2>>()
        .project_back(draw([1, 1], 10));
}

#[test]
#[should_panic(expected = "rank 4 of mode 0 outside")]
fn rank_above_its_dimension_panics() {
    tucker_factors(draw([3, 5], 11), [4, 2]);
}

/// Mode 0's sweep unfolding is [12, r_1] = [12, 1]: rank 3 cannot fit.
#[test]
#[should_panic(expected = "rank 3 of mode 0 outside 1..=min(12, 1)")]
fn rank_above_the_other_ranks_product_panics() {
    tucker_factors(draw([12, 7], 12), [3, 1]);
}

#[test]
#[should_panic(expected = "update_proj_gap must be positive")]
fn zero_gap_panics() {
    let mut config = TensorGaLoreConfig::new(GaLoreRank::Uniform(1));
    config.update_proj_gap = 0;
    config.init::<2, Tensor<2>>().project(draw([3, 3], 13), 0);
}

// Complex tensors (CM3.6b).

fn complex_draw<const D: usize>(dims: [usize; D], seed: u64) -> ComplexTensor<D> {
    ComplexTensor::new(draw(dims, seed), draw(dims, seed + 1000))
}

/// Aᴴ C.
fn conj_t_matmul(a: &ComplexTensor<2>, c: &ComplexTensor<2>) -> ComplexTensor<2> {
    a.clone().conj().transpose().matmul(c.clone())
}

fn complex_max_abs<const D: usize>(x: ComplexTensor<D>) -> f64 {
    max_abs(x.re.square() + x.im.square()).sqrt()
}

/// The Jacobi SVD gives orthonormal left singular vectors spanning the range: full rank
/// r = m on a wide matrix, so U Uᴴ M = M; on a tall one Uᴴ U = I.
#[test]
fn complex_leading_vectors_are_orthonormal_and_span_the_range() {
    for (m, n) in [(4, 7), (7, 4), (5, 5)] {
        let a = complex_draw([m, n], 20 + m as u64);
        let r = m.min(n);
        let u = complex_leading_vectors(a.clone(), r);
        assert_eq!(u.re.dims(), [m, r]);
        let gram = conj_t_matmul(&u, &u);
        let eye = ComplexTensor::new(Tensor::eye(r, &device()), Tensor::zeros([r, r], &device()));
        assert!(complex_max_abs(gram - eye) < 1e-5, "{m} × {n}");
        if m <= n {
            // U (Uᴴ A) = A.
            let coeffs = conj_t_matmul(&u, &a);
            let back = u.clone().matmul(coeffs);
            assert!(complex_max_abs(back - a) < 1e-5, "{m} × {n}");
        }
    }
}

/// tensorly's rule on LAPACK's phase: with w the reference entry (LAPACK makes it real,
/// sign unknown) and u_max the entry of largest modulus, the rule leaves
/// phase(u_max) = 2 phase(w) (mod 2π), whatever phase the SVD returned.
#[test]
fn complex_phase_is_twice_the_reference_phase() {
    for (m, n) in [(4, 7), (7, 4)] {
        let a = complex_draw([m, n], 30 + m as u64);
        let r = 3;
        let u = complex_leading_vectors(a.clone(), r);
        let (w_re, w_im) = if m >= n {
            let first = ComplexTensor::new(a.re.slice([0..m, 0..1]), a.im.slice([0..m, 0..1]));
            let w = conj_t_matmul(&first, &u); // aᴴ u = (Aᴴ u)₀
            (w.re, w.im)
        } else {
            (
                u.re.clone().slice([0..1, 0..r]),
                u.im.clone().slice([0..1, 0..r]),
            )
        };
        let largest = (u.re.clone().square() + u.im.clone().square()).argmax(0);
        let (p, q) = (
            u.re.clone().gather(0, largest.clone()),
            u.im.clone().gather(0, largest),
        );
        // (w / |w|)² against u_max / |u_max|.
        let w_mod_sq = w_re.clone().square() + w_im.clone().square();
        let sq_re = (w_re.clone().square() - w_im.clone().square()) / w_mod_sq.clone();
        let sq_im = (w_re * w_im).mul_scalar(2.0) / w_mod_sq;
        let modulus = (p.clone().square() + q.clone().square()).sqrt();
        assert!(max_abs(sq_re - p / modulus.clone()) < 1e-4, "{m} × {n}");
        assert!(max_abs(sq_im - q / modulus) < 1e-4, "{m} × {n}");
    }
}

#[test]
fn complex_factors_are_orthonormal() {
    let factors = tucker_factors(complex_draw([6, 5, 4], 40), [3, 2, 2]);
    for (f, r) in factors.iter().zip([3, 2, 2]) {
        let eye = ComplexTensor::new(Tensor::eye(r, &device()), Tensor::zeros([r, r], &device()));
        assert!(complex_max_abs(conj_t_matmul(f, f) - eye) < 1e-5);
    }
}

#[test]
fn complex_full_rank_round_trip_is_scale_times_identity() {
    let grad = complex_draw([4, 3, 5], 41);
    let mut p = TensorGaLoreConfig {
        rank: GaLoreRank::PerMode(vec![4, 3, 5]),
        update_proj_gap: 1,
        scale: 0.5,
    }
    .init::<3, ComplexTensor<3>>();
    let low = p.project(grad.clone(), 7);
    let back = p.project_back(low);
    assert!(complex_max_abs(back - grad.scaled(0.5)) < 1e-5);
}

/// A complex tensor of exact Tucker rank (2, 3, 2) is reproduced from its projection.
#[test]
fn complex_exact_tucker_tensor_is_reproduced() {
    let core = complex_draw([2, 3, 2], 42);
    let a = [
        complex_draw([5, 2], 43),
        complex_draw([6, 3], 44),
        complex_draw([4, 2], 45),
    ];
    let x = core.from_low_rank(&a);
    let mut p =
        TensorGaLoreConfig::new(GaLoreRank::PerMode(vec![2, 3, 2])).init::<3, ComplexTensor<3>>();
    let low = p.project(x.clone(), 0);
    assert_eq!(low.re.dims(), [2, 3, 2]);
    let scale = complex_max_abs(x.clone());
    assert!(complex_max_abs(p.project_back(low) - x) < 1e-5 * scale);
}

/// A real tensor taken as complex gets the real path's factors: LAPACK's phase makes each
/// vector real up to a sign, and the complex rule then matches `svd_flip`'s.
#[test]
fn complex_path_on_a_real_tensor_gives_the_real_factors() {
    let x = draw([6, 5, 4], 46);
    let ranks = [3, 2, 2];
    let real = tucker_factors(x.clone(), ranks);
    let complex = tucker_factors(ComplexTensor::new(x.clone(), x.zeros_like()), ranks);
    for (f, c) in real.into_iter().zip(complex) {
        assert!(max_abs(c.re - f) < 1e-5);
        assert!(max_abs(c.im) < 1e-5);
    }
}

/// Repeated singular values (here (1, 1, 0.5): (1 + i)/√2 · R diag(1, 1, 0.5) Rᵀ, R two
/// plane rotations, so no phase reference entry is zero) stall a QR SVD of the real embedding in f64; the Jacobi SVD gives orthonormal
/// vectors spanning the range.
#[test]
fn complex_leading_vectors_handle_repeated_singular_values() {
    let (c, s) = (0.8_f32, 0.6_f32);
    let rot = Tensor::<2>::from_data([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], &device());
    let diag = Tensor::<2>::from_data(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]],
        &device(),
    );
    let rot =
        Tensor::<2>::from_data([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], &device()).matmul(rot);
    let re = rot
        .clone()
        .matmul(diag)
        .matmul(rot.transpose())
        .mul_scalar(std::f32::consts::FRAC_1_SQRT_2);
    let a = ComplexTensor::new(re.clone(), re);
    let u = complex_leading_vectors(a.clone(), 3);
    let eye = ComplexTensor::new(Tensor::eye(3, &device()), Tensor::zeros([3, 3], &device()));
    assert!(complex_max_abs(conj_t_matmul(&u, &u) - eye) < 1e-6);
    let back = u.clone().matmul(conj_t_matmul(&u, &a));
    assert!(complex_max_abs(back - a) < 1e-6);
}
