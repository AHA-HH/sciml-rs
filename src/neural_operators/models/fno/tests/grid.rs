//! Grid channels generated on the device (CONVENTIONS §2).

use super::*;

/// T1: `grid_cl` against the old loader grid, per element and channel.
/// Returns the max |Δ| against the old grid rounded to f32.
///
/// Contract (B2): |Δ| ≤ 1 ulp at 1.0 (`f32::EPSILON`); bit for bit is
/// expected under correctly rounded division. Endpoints are exact, a
/// single point is 0 (not 0/0), and every batch row is the same.
fn check_grid_values<const R: usize>(spatial: &[usize]) -> f32 {
    let device = Device::default();
    let d = R - 2;
    let mut dims = [2; R];
    dims[1..R - 1].copy_from_slice(spatial);
    dims[R - 1] = 1;
    let grid = FNO::<R>::grid_cl(dims, DType::F32, &device);
    let mut want = dims;
    want[R - 1] = d;
    assert_eq!(grid.dims(), want, "grid shape for {spatial:?}");

    let got: Vec<f32> = grid.into_data().try_to_vec::<f32>().unwrap();
    let per_row = got.len() / 2;
    assert_eq!(
        got[..per_row]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        got[per_row..]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "batch rows differ for {spatial:?}"
    );

    let old = old_grid_f64(spatial);
    let mut max_delta = 0f32;
    for (at, idx) in ndarray::indices(spatial).into_iter().enumerate() {
        for (k, old_k) in old.iter().enumerate() {
            let g = got[at * d + k];
            let want = <f32 as HostFloat>::from_f64(old_k[idx.slice()]);
            assert!(g.is_finite(), "{spatial:?} channel {k} at {idx:?}: {g}");
            max_delta = max_delta.max((g - want).abs());
            // channel k varies along spatial axis d - 1 - k only
            let axis = d - 1 - k;
            let (i, n) = (idx[axis], spatial[axis]);
            if i == 0 {
                assert_eq!(g, 0.0, "{spatial:?} channel {k} start at {idx:?}");
            } else if i == n - 1 {
                assert_eq!(g, 1.0, "{spatial:?} channel {k} end at {idx:?}");
            }
        }
    }
    println!("grid {spatial:?}: max |grid_new - grid_old| = {max_delta:e}");
    assert!(
        max_delta <= f32::EPSILON,
        "grid {spatial:?} deviates from the old f64 grid by {max_delta:e} > 1 ulp"
    );
    max_delta
}

#[test]
fn grid_matches_old_loader_grid() {
    let mut worst = 0f32;
    for s in [1, 2, 7, 256, 8192] {
        worst = worst.max(check_grid_values::<3>(&[s]));
    }
    // Non-square: a transposed grid or swapped channels would differ.
    worst = worst.max(check_grid_values::<4>(&[5, 7]));
    worst = worst.max(check_grid_values::<4>(&[7, 5]));
    worst = worst.max(check_grid_values::<4>(&[1, 3]));
    worst = worst.max(check_grid_values::<5>(&[4, 6, 5]));
    println!("grid: max |grid_new - grid_old| over all shapes = {worst:e}");
}

/// Hand-pinned 2 x 3 grid: channel 0 = x along axis 2 (length 3),
/// channel 1 = y along axis 1 (length 2), as `np.meshgrid` 'xy'.
#[test]
fn grid_2d_order_is_pinned() {
    let device = Device::default();
    let grid = FNO::<4>::grid_cl([1, 2, 3, 1], DType::F32, &device);
    let got = grid.into_data().try_to_vec::<f32>().unwrap();
    // [1, 2, 3, 2] channels-last: (x, y) per point, row-major points.
    #[rustfmt::skip]
    let want = [
        0.0, 0.0,   0.5, 0.0,   1.0, 0.0, // row 0: y = 0
        0.0, 1.0,   0.5, 1.0,   1.0, 1.0, // row 1: y = 1
    ];
    assert_eq!(got, want);
}

/// T2 controls: with the grid wrong, the same comparison as
/// `check_forward` / `check_padded_forward` must fail by ≥ 1e-3, so a
/// model that dropped, transposed or mis-scaled its grid is caught.
///
/// The `[0, 1)` grid (`i / n`) is off from `i / (n - 1)` by at most
/// `1 / n`, so its control needs a coarse axis to clear 1e-3 reliably;
/// hence the small 1D grid below.
fn check_grid_controls<const R: usize>(
    modes: Vec<usize>,
    data_channels: usize,
    spatial: &[usize],
    padding: Option<usize>,
) {
    let device = Device::default();
    device.seed(13);
    let model = with_unit_lift_bias(
        test_config(modes, data_channels, 6)
            .with_spectral_init(Some(SpectralInit::SymmetricUniform))
            .with_padding(padding)
            .init::<R>(&device),
    );
    let x = random_input::<R>(spatial, data_channels, &device);
    let out = flat(model.forward(x.clone()));
    let reference = |grids: &[ArrayD<f64>]| {
        let x = with_grid(x.clone(), grids);
        flat(match padding {
            Some(p) if p > 0 => reference_forward_padded(&model, x, Pad::End(p)),
            _ => reference_forward_padded(&model, x, Pad::Off),
        })
    };

    let err = rel_err(out.clone(), reference(&old_grid_f64(spatial)));
    println!("FNO<{R}> {spatial:?} pad {padding:?}: rel err vs old grid {err:e}");
    assert!(
        err <= TOL,
        "FNO<{R}> differs from the old-grid forward: {err:e}"
    );

    let zeros: Vec<ArrayD<f64>> = old_grid_f64(spatial)
        .iter()
        .map(|g| ArrayD::zeros(g.raw_dim()))
        .collect();
    let half_open_axes: Vec<Array1<f64>> = spatial
        .iter()
        .map(|&n| Array1::from_shape_fn(n, |i| i as f64 / n as f64))
        .collect();
    let half_open = xy_grid(&half_open_axes);
    let mut wrong = vec![("no grid", zeros), ("[0, 1) grid", half_open)];
    if spatial.len() > 1 {
        wrong.push(("'ij' order", ij_grid(&uniform_axes(spatial))));
    }
    for (name, grids) in wrong {
        let err = rel_err(out.clone(), reference(&grids));
        println!("FNO<{R}> {spatial:?} pad {padding:?} vs {name}: rel err {err:e}");
        assert!(
            err >= 1e-3,
            "FNO<{R}> forward indistinguishable from {name}: {err:e}"
        );
    }
}

#[test]
fn grid_controls_1d() {
    check_grid_controls::<3>(vec![4], 1, &[6], None);
    check_grid_controls::<3>(vec![4], 1, &[6], Some(5));
}

#[test]
fn grid_controls_2d() {
    check_grid_controls::<4>(vec![3, 2], 1, &[8, 6], None);
    check_grid_controls::<4>(vec![3, 2], 3, &[8, 6], Some(3));
}

#[test]
fn grid_controls_3d() {
    check_grid_controls::<5>(vec![2, 2, 2], 3, &[4, 6, 5], None);
}

/// Separate from the unpadded 3D case so that a failure of the padded
/// 3D path (as `padded_forward_matches_reference_3d`) is reported apart
/// from the grid itself.
#[test]
fn grid_controls_3d_padded() {
    check_grid_controls::<5>(vec![2, 2, 2], 1, &[4, 6, 5], Some(2));
}

/// T3: one set of weights at two resolutions; each forward matches the
/// old path with the grid built for that resolution.
fn check_resolutions<const R: usize>(modes: Vec<usize>, resolutions: &[&[usize]]) {
    let device = Device::default();
    let model: FNO<R> = test_config(modes, 1, 6).init::<R>(&device);
    for &spatial in resolutions {
        let x = random_input::<R>(spatial, 1, &device);
        let out = model.forward(x.clone());
        assert_eq!(&out.dims()[1..R - 1], spatial);
        let err = rel_err(
            flat(out),
            flat(reference_forward_padded(&model, with_old_grid(x), Pad::Off)),
        );
        println!("FNO<{R}> at {spatial:?}: rel err {err:e}");
        assert!(err <= TOL, "FNO<{R}> at {spatial:?} differs: {err:e}");
    }
}

#[test]
fn resolution_change_builds_that_grid() {
    check_resolutions::<3>(vec![4], &[&[16], &[48]]);
    check_resolutions::<4>(vec![3, 2], &[&[8, 6], &[16, 12]]);
}

/// Regression (review R1): on Metal the expanded grid came out different
/// per batch row after earlier device work, silently corrupting some
/// samples' coordinates. Two identical samples must give bit-identical
/// outputs, match the old host-grid forward, and leave `grid_cl`'s rows
/// equal to each other and to the f64 grid, all after real model work.
/// The input is itself a `cat` of one sample, the case in which the old
/// channels-first concatenation lost grid entries.
fn check_batch_rows_identical<const R: usize>(modes: Vec<usize>, spatial: &[usize]) {
    let device = Device::default();
    device.seed(13);
    let model = with_unit_lift_bias(
        test_config(modes, 1, 6)
            .with_spectral_init(Some(SpectralInit::SymmetricUniform))
            .init::<R>(&device),
    );
    let mut one = vec![1];
    one.extend_from_slice(spatial);
    one.push(1);
    let one: [usize; R] = one.try_into().unwrap();
    let sample = Tensor::<R>::random(one, Distribution::Normal(0.0, 1.0), &device);
    let x = Tensor::cat(vec![sample.clone(), sample], 0);

    let out = model.forward(x.clone());
    let rows = bits(out.clone());
    let half = rows.len() / 2;
    assert_eq!(
        rows[..half],
        rows[half..],
        "FNO<{R}> {spatial:?}: batch rows differ"
    );
    let err = rel_err(
        flat(out),
        flat(reference_forward_padded(&model, with_old_grid(x), Pad::Off)),
    );
    println!("FNO<{R}> {spatial:?} batch-row check: rel err vs old grid {err:e}");
    assert!(
        err <= TOL,
        "FNO<{R}> {spatial:?} differs from the old-grid forward: {err:e}"
    );

    let _ = check_grid_values::<R>(spatial);
}

#[test]
fn batch_rows_see_identical_grids() {
    check_batch_rows_identical::<3>(vec![4], &[6]);
    check_batch_rows_identical::<3>(vec![4], &[48]);
    check_batch_rows_identical::<4>(vec![3, 2], &[8, 6]);
    check_batch_rows_identical::<5>(vec![2, 2, 2], &[4, 6, 5]);
}

/// T5: an old-style input that still carries the grid is rejected.
#[test]
#[should_panic(expected = "data_channels = 1 input channels, got 3")]
fn forward_rejects_grid_augmented_input() {
    let device = Device::default();
    let model: FNO<4> = test_config(vec![3, 2], 1, 6).init::<4>(&device);
    let _ = model.forward(random_input::<4>(&[8, 6], 3, &device));
}
