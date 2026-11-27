//! Unit tests of the Tensor GaLore projector (library CM3.6a); its parity with 2.0.0 is
//! `tests/parity_galore*.rs`.

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
    .init::<3>();
    let low = p.project(grad.clone(), 7);
    let back = p.project_back(low);
    assert!(max_abs(back - grad.mul_scalar(0.5)) < 1e-5);
}

/// A tensor of exact Tucker rank (2, 3, 2) is reproduced from its projection.
#[test]
fn exact_tucker_tensor_is_reproduced() {
    let core = draw([2, 3, 2], 3);
    let a = [draw([5, 2], 4), draw([6, 3], 5), draw([4, 2], 6)];
    let x = from_low_rank(core, &a);
    let mut p = TensorGaLoreConfig::new(GaLoreRank::PerMode(vec![2, 3, 2])).init::<3>();
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
    .init::<2>();
    p.project(draw([5, 4], 7), 0);
    let first = p.factors().unwrap()[0].clone();
    p.project(draw([5, 4], 8), 3);
    assert_eq!(max_abs(p.factors().unwrap()[0].clone() - first), 0.0);
}

#[test]
#[should_panic(expected = "multiple of update_proj_gap")]
fn first_project_needs_the_gap() {
    TensorGaLoreConfig::new(GaLoreRank::Uniform(1))
        .init::<2>()
        .project(draw([3, 3], 9), 1);
}

#[test]
#[should_panic(expected = "project_back before project")]
fn project_back_needs_factors() {
    TensorGaLoreConfig::new(GaLoreRank::Uniform(1))
        .init::<2>()
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
    config.init::<2>().project(draw([3, 3], 13), 0);
}
