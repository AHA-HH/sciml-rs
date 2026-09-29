//! Uniform coordinate grid generation for N-dimensional spatial data.

use ndarray::{Array1, ArrayD, Axis, IxDyn, concatenate, s};

/// Where [`append_grid`] places the coordinate channels relative to the data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridPlacement {
    /// `[grids.., data..]` - Li's `fourier_3d.py` order.
    BeforeData,
    /// `[data.., grids..]` - Li's 1D and 2D scripts.
    AfterData,
}

/// Broadcast per-axis coordinates into full 'ij'-indexed grids.
///
/// `axes[i]` holds the coordinates along spatial axis `i`; grid `i` has shape
/// `[axes[0].len(), .., axes[n-1].len()]` and varies along axis `i` only.
/// Use this directly for non-`linspace` axes, e.g. a time axis over `(0, 1]`.
///
/// # Panics
/// If `axes` is empty.
pub fn grid_from_axes(axes: &[Array1<f64>]) -> Vec<ArrayD<f64>> {
    assert!(!axes.is_empty(), "grid_from_axes: need at least one axis");
    let ndim = axes.len();
    let sizes: Vec<usize> = axes.iter().map(|a| a.len()).collect();
    let full_shape = IxDyn(&sizes);

    axes.iter()
        .enumerate()
        .map(|(axis, coords)| {
            // shape with this axis live, all others singleton, e.g. [1, s, 1] for axis=1 in 3D
            let mut axis_shape = vec![1; ndim];
            axis_shape[axis] = sizes[axis];

            // reshape then broadcast out to the full spatial shape
            coords
                .clone()
                .into_shape_with_order(IxDyn(&axis_shape))
                .unwrap()
                .broadcast(full_shape.clone())
                .unwrap()
                .to_owned()
        })
        .collect()
}

/// Uniform 'ij'-indexed grids: axis `i` has `sizes[i]` evenly spaced points
/// over the closed interval `bounds[i] = (start, end)`.
///
/// For Li's 2D 'xy' (`np.meshgrid`) channel order, reverse the result.
///
/// # Panics
/// If `bounds` and `sizes` differ in length or are empty.
pub fn uniform_grid(bounds: &[(f64, f64)], sizes: &[usize]) -> Vec<ArrayD<f64>> {
    assert_eq!(
        bounds.len(),
        sizes.len(),
        "uniform_grid: one (start, end) bound per axis size"
    );
    let axes: Vec<Array1<f64>> = bounds
        .iter()
        .zip(sizes)
        .map(|(&(start, end), &n)| Array1::linspace(start, end, n))
        .collect();
    grid_from_axes(&axes)
}

/// Append coordinate channels to `data: [batch, *spatial, channels]` along
/// the last axis, broadcasting each grid across the batch.
///
/// Every grid must have shape `spatial`; the output has
/// `channels + grids.len()` channels, ordered by `placement`.
///
/// # Panics
/// If `data` has no spatial axis or a grid's shape differs from `spatial`.
pub fn append_grid(
    data: ArrayD<f64>,
    grids: &[ArrayD<f64>],
    placement: GridPlacement,
) -> ArrayD<f64> {
    assert!(
        data.ndim() >= 3,
        "append_grid: data must be [batch, *spatial, channels], got {:?}",
        data.shape()
    );
    let batch_size = data.shape()[0];
    let last_axis = data.ndim() - 1;
    let spatial = &data.shape()[1..last_axis];

    // expand each spatial grid [*spatial] -> [batch, *spatial, 1] via broadcast
    let unit_shape: Vec<usize> = std::iter::once(1)
        .chain(spatial.iter().copied())
        .chain(std::iter::once(1))
        .collect();
    let full_shape: Vec<usize> = std::iter::once(batch_size)
        .chain(spatial.iter().copied())
        .chain(std::iter::once(1))
        .collect();
    let expanded: Vec<ArrayD<f64>> = grids
        .iter()
        .enumerate()
        .map(|(i, g)| {
            assert_eq!(
                g.shape(),
                spatial,
                "append_grid: grid {i} shape does not match data spatial shape"
            );
            g.to_shape(IxDyn(&unit_shape))
                .unwrap()
                .broadcast(IxDyn(&full_shape))
                .unwrap()
                .to_owned()
        })
        .collect();

    let mut views = Vec::with_capacity(expanded.len() + 1);
    match placement {
        GridPlacement::BeforeData => {
            views.extend(expanded.iter().map(|a| a.view()));
            views.push(data.view());
        }
        GridPlacement::AfterData => {
            views.push(data.view());
            views.extend(expanded.iter().map(|a| a.view()));
        }
    }

    concatenate(Axis(last_axis), &views).unwrap()
}

/// `n` evenly spaced points over the half-open interval `(start, end]`:
/// `linspace(start, end, n + 1)` without its first point. This is the time
/// axis of Li's `fourier_3d.py` (`np.linspace(0, 1, T + 1)[1:]`); pass it to
/// [`grid_from_axes`]. Not bitwise equal to `linspace(start + h, end, n)`.
pub fn linspace_excluding_start(start: f64, end: f64, n: usize) -> Array1<f64> {
    Array1::linspace(start, end, n + 1)
        .slice(s![1..])
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Distinct value in every cell, so misplaced channels can't coincide.
    fn distinct_data(shape: &[usize]) -> ArrayD<f64> {
        let mut k = 0.0;
        ArrayD::from_shape_simple_fn(IxDyn(shape), || {
            k += 1.0;
            k * 0.37
        })
    }

    fn line(start: f64, end: f64, n: usize) -> ArrayD<f64> {
        uniform_grid(&[(start, end)], &[n]).remove(0)
    }

    // --- 1D (Burgers: [data, x]) ---

    #[test]
    fn uniform_grid_1d_length_endpoints_and_spacing() {
        let grid = line(0.0, 1.0, 10);
        assert_eq!(grid.shape(), &[10]);
        assert_relative_eq!(grid[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(grid[9], 1.0, epsilon = 1e-10);

        let grid = line(0.0, 1.0, 5);
        for i in 0..4 {
            assert_relative_eq!(grid[i + 1] - grid[i], 0.25, epsilon = 1e-10);
        }

        let grid = line(0.0, 2.0 * std::f64::consts::PI, 5);
        assert_relative_eq!(grid[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(grid[4], 2.0 * std::f64::consts::PI, epsilon = 1e-10);
    }

    #[test]
    fn append_grid_1d_places_grid_after_data() {
        // [batch=2, s=4, 1] + grid[4] -> [2, 4, 2]
        let data = ArrayD::from_shape_vec(
            IxDyn(&[2, 4, 1]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let result = append_grid(
            data,
            &uniform_grid(&[(0.0, 1.0)], &[4]),
            GridPlacement::AfterData,
        );
        assert_eq!(result.shape(), &[2, 4, 2]);

        // channel 0 is the original data
        assert_eq!(result[[0, 0, 0]], 1.0);
        assert_eq!(result[[0, 1, 0]], 2.0);
        assert_eq!(result[[1, 0, 0]], 5.0);
        // channel 1 is the grid, identical for every batch example
        assert_relative_eq!(result[[0, 0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[[0, 3, 1]], 1.0, epsilon = 1e-10);
        assert_eq!(result[[0, 1, 1]], result[[1, 1, 1]]);
    }

    // --- 2D (Darcy: [data, x, y] in 'xy' order = reversed 'ij') ---

    #[test]
    fn reversed_uniform_grid_2d_matches_numpy_meshgrid() {
        let mut grid = uniform_grid(&[(0.0, 1.0); 2], &[3, 3]);
        grid.reverse();

        // Pinned against: np.meshgrid(np.linspace(0,1,3), np.linspace(0,1,3))
        let expected_xx = [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]];
        let expected_yy = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]];
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (grid[0][[i, j]] - expected_xx[i][j]).abs() < 1e-9,
                    "xx[{i},{j}]"
                );
                assert!(
                    (grid[1][[i, j]] - expected_yy[i][j]).abs() < 1e-9,
                    "yy[{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn append_grid_2d_channel_placement() {
        // data all 9.0 so it's distinguishable from grid values
        let data = ArrayD::from_elem(IxDyn(&[1, 2, 2, 1]), 9.0);
        let mut grid = uniform_grid(&[(0.0, 1.0); 2], &[2, 2]);
        grid.reverse();
        let result = append_grid(data, &grid, GridPlacement::AfterData);

        assert_eq!(result.shape(), &[1, 2, 2, 3]);
        assert_eq!(result[[0, 0, 0, 0]], 9.0); // channel 0: data
        assert_eq!(result[[0, 0, 1, 1]], 1.0); // channel 1: x, varies along axis 2
        assert_eq!(result[[0, 1, 0, 1]], 0.0);
        assert_eq!(result[[0, 1, 0, 2]], 1.0); // channel 2: y, varies along axis 1
        assert_eq!(result[[0, 0, 1, 2]], 0.0);
    }

    #[test]
    fn append_grid_2d_non_square_axes_follow_their_extents() {
        // [1, 2, 3, 1]: square grids hide a transpose, this doesn't.
        let data = distinct_data(&[1, 2, 3, 1]);
        let mut grid = uniform_grid(&[(0.0, 1.0); 2], &[2, 3]);
        assert_eq!(grid[0].shape(), &[2, 3]);
        grid.reverse();
        let out = append_grid(data.clone(), &grid, GridPlacement::AfterData);
        assert_eq!(out.shape(), &[1, 2, 3, 3]);

        let along_axis2 = [0.0, 0.5, 1.0];
        let along_axis1 = [0.0, 1.0];
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(out[[0, i, j, 0]], data[[0, i, j, 0]]);
                assert_eq!(out[[0, i, j, 1]], along_axis2[j]);
                assert_eq!(out[[0, i, j, 2]], along_axis1[i]);
            }
        }
    }

    // --- 3D (NS: [x, y, t, data], 'ij', t over (0, 1]) ---

    fn ns_grid(s: usize, t_out: usize) -> Vec<ArrayD<f64>> {
        let x = Array1::linspace(0.0, 1.0, s);
        grid_from_axes(&[x.clone(), x, linspace_excluding_start(0.0, 1.0, t_out)])
    }

    #[test]
    fn grid_from_axes_3d_axis_assignment() {
        let (s, t_out) = (4, 3);
        let grid = ns_grid(s, t_out);
        let (xx, yy, tt) = (&grid[0], &grid[1], &grid[2]);
        for g in &grid {
            assert_eq!(g.shape(), &[s, s, t_out]);
        }

        // each grid varies along its own axis only
        for i in 0..s {
            for j in 0..s {
                for k in 0..t_out {
                    assert_eq!(xx[[i, j, k]], xx[[i, 0, 0]], "xx must depend on i only");
                    assert_eq!(yy[[i, j, k]], yy[[0, j, 0]], "yy must depend on j only");
                    assert_eq!(tt[[i, j, k]], tt[[0, 0, k]], "tt must depend on k only");
                }
            }
        }

        // no xy-swap: xx[i] is coords[i], yy[j] is coords[j]
        assert!((xx[[1, 0, 0]] - 1.0 / 3.0).abs() < 1e-9);
        assert!((yy[[0, 2, 0]] - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn ns_time_axis_excludes_zero() {
        let (s, t_out) = (4, 4);
        let grid = ns_grid(s, t_out);
        let (xx, tt) = (&grid[0], &grid[2]);

        // x: closed [0, 1]
        assert!((xx[[0, 0, 0]] - 0.0).abs() < 1e-9);
        assert!((xx[[s - 1, 0, 0]] - 1.0).abs() < 1e-9);

        // t: half-open (0, 1] - first point is 1/t_out, not 0; last is 1
        assert!((tt[[0, 0, 0]] - 1.0 / t_out as f64).abs() < 1e-9);
        assert!((tt[[0, 0, t_out - 1]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn append_grid_3d_channel_order_and_broadcast() {
        let (s, t_out, batch_size) = (2, 2, 2);
        // batch index baked into the value, to catch a broadcast bug
        let data = ArrayD::from_shape_fn(IxDyn(&[batch_size, s, s, t_out, 1]), |idx| {
            (idx[0] * 1000) as f64
        });
        let grid = ns_grid(s, t_out);
        let out = append_grid(data.clone(), &grid, GridPlacement::BeforeData);

        assert_eq!(out.shape(), &[batch_size, s, s, t_out, 4]);
        for b in 0..batch_size {
            for i in 0..s {
                for j in 0..s {
                    for k in 0..t_out {
                        for (c, g) in grid.iter().enumerate() {
                            assert_eq!(out[[b, i, j, k, c]], g[[i, j, k]], "channel {c}");
                        }
                        assert_eq!(out[[b, i, j, k, 3]], data[[b, i, j, k, 0]], "batch {b}");
                    }
                }
            }
        }
    }

    // --- validation ---

    #[test]
    #[should_panic(expected = "one (start, end) bound per axis size")]
    fn uniform_grid_rejects_bounds_size_mismatch() {
        let _ = uniform_grid(&[(0.0, 1.0)], &[3, 3]);
    }

    #[test]
    #[should_panic(expected = "need at least one axis")]
    fn uniform_grid_rejects_zero_axes() {
        let _ = uniform_grid(&[], &[]);
    }

    #[test]
    #[should_panic(expected = "grid 0 shape does not match data spatial shape")]
    fn append_grid_rejects_mismatched_grid() {
        let data = ArrayD::zeros(IxDyn(&[1, 2, 3, 1]));
        let grid = uniform_grid(&[(0.0, 1.0); 2], &[3, 2]); // transposed extents
        let _ = append_grid(data, &grid, GridPlacement::AfterData);
    }
}
