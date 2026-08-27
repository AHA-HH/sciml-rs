//! Uniform coordinate grid generation for N-dimensional spatial data.

// TODO: confirm if generic nd uniform grid breaks for 2d uniform_grid, append_grid
use ndarray::{Array1, Array2, Array3, ArrayD, Axis, IxDyn, concatenate, s};

/// Generate n independent coordinate axes as broadcast-expanded ArrayD grids.
/// Each grid varies along its own axis only ('ij'-style indexing, no swap).
/// `bounds[i]` = (start, end) for axis i, `sizes[i]` = number of points on axis i.
pub fn uniform_grid(bounds: &[(f64, f64)], sizes: &[usize]) -> Vec<ArrayD<f64>> {
    let ndim = sizes.len();
    let full_shape = IxDyn(sizes);

    (0..ndim)
        .map(|axis| {
            // 1D coords for this axis
            let coords = Array1::linspace(bounds[axis].0, bounds[axis].1, sizes[axis]);

            // shape with this axis live, all others singleton, e.g. [1, s, 1] for axis=1 in 3D
            let mut axis_shape = vec![1; ndim];
            axis_shape[axis] = sizes[axis];

            // reshape then broadcast out to the full spatial shape
            coords
                .into_shape_with_order(IxDyn(&axis_shape))
                .unwrap()
                .broadcast(full_shape.clone())
                .unwrap()
                .to_owned()
        })
        .collect()
}

/// Append n grid coordinate channels to `data` along the last axis.
/// `data` is [batch, *spatial, channels]. `grid_first` controls channel order
/// (grids-then-data vs data-then-grids) — TEMPORARY until the 2D/3D convention
/// mismatch in the current codebase is resolved; do not leave this public.
pub fn append_grid(
    data: ArrayD<f64>,
    grids: Vec<ArrayD<f64>>,
    grid_first: bool,
) -> ArrayD<f64> {
    let batch_size = data.shape()[0];
    let last_axis = data.ndim() - 1;

    // expand each spatial grid [*spatial] -> [batch, *spatial, 1] via broadcast
    let expanded: Vec<ArrayD<f64>> = grids
        .into_iter()
        .map(|g| {
            let spatial: Vec<usize> = g.shape().to_vec();
            let unit_shape: Vec<usize> = std::iter::once(1)
                .chain(spatial.iter().copied())
                .chain(std::iter::once(1))
                .collect();
            let full_shape: Vec<usize> = std::iter::once(batch_size)
                .chain(spatial)
                .chain(std::iter::once(1))
                .collect();

            g.into_shape_with_order(IxDyn(&unit_shape))
                .unwrap()
                .broadcast(IxDyn(&full_shape))
                .unwrap()
                .to_owned()
        })
        .collect();

    // assemble concat order per grid_first flag
    let mut views = Vec::with_capacity(expanded.len() + 1);
    if grid_first {
        views.extend(expanded.iter().map(|a| a.view()));
        views.push(data.view());
    } else {
        views.push(data.view());
        views.extend(expanded.iter().map(|a| a.view()));
    }

    concatenate(Axis(last_axis), &views).unwrap()
}

// Generate a uniform 1D grid of n points
pub fn uniform_grid_1d(start: f64, end: f64, n_points: usize) -> Array1<f64> {
    Array1::linspace(start, end, n_points)
}

// Append spatial grid coordinates as an extra channel to the data
pub fn append_grid_1d(data: ArrayD<f64>, grid: Array1<f64>) -> ArrayD<f64> {
    let batch_size = data.shape()[0];
    let n_points = data.shape()[1];

    // Reshape grid from [s] to [batch, s, 1] repeating it for each example in the batch
    let grid_expanded = grid
        .into_shape_with_order(IxDyn(&[1, n_points, 1])) // [1, s, 1]
        .unwrap()
        .broadcast(IxDyn(&[batch_size, n_points, 1])) // [batch, s, 1]
        .unwrap()
        .to_owned();

    // Concatenate along last dimension [batch, s, 1] + [batch, s, 1] -> [batch, s, 2]
    concatenate(Axis(2), &[data.view(), grid_expanded.view()]).unwrap()
}

// Generate a uniform 2D meshgrid of (s x s) points, returned as two coordinate arrays
pub fn uniform_grid_2d(start: f64, end: f64, s: usize) -> (Array2<f64>, Array2<f64>) {
    let coords = Array1::linspace(start, end, s);

    // x varies along columns, constant along rows; y varies along rows, constant along columns
    // — matches np.meshgrid's default 'xy' indexing, same convention Li's Python uses.
    let mut xx = Array2::<f64>::zeros((s, s));
    let mut yy = Array2::<f64>::zeros((s, s));
    for i in 0..s {
        for j in 0..s {
            xx[[i, j]] = coords[j];
            yy[[i, j]] = coords[i];
        }
    }

    (xx, yy)
}

// Append 2D spatial grid coordinates (x, y) as two extra channels to the data
pub fn append_grid_2d(data: ArrayD<f64>, xx: Array2<f64>, yy: Array2<f64>) -> ArrayD<f64> {
    let batch_size = data.shape()[0];
    let s1 = data.shape()[1];
    let s2 = data.shape()[2];

    let xx_expanded = xx
        .into_shape_with_order(IxDyn(&[1, s1, s2, 1]))
        .unwrap()
        .broadcast(IxDyn(&[batch_size, s1, s2, 1]))
        .unwrap()
        .to_owned();

    let yy_expanded = yy
        .into_shape_with_order(IxDyn(&[1, s1, s2, 1]))
        .unwrap()
        .broadcast(IxDyn(&[batch_size, s1, s2, 1]))
        .unwrap()
        .to_owned();

    // [batch, s, s, 1] + [batch, s, s, 1] + [batch, s, s, 1] -> [batch, s, s, 3]
    concatenate(Axis(3), &[data.view(), xx_expanded.view(), yy_expanded.view()]).unwrap()
}

// Generate a uniform 3d grid over (x, y, t) for an (s x s x t_out) volume 
pub fn uniform_grid_3d(start: f64, end: f64, s: usize, t_out: usize) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let coords = Array1::linspace(start, end, s);
    // T+1 points over [start,end], drop the first - leaves T points over (start,end]
    let t_coords_full = Array1::linspace(start, end, t_out + 1);
    let t_coords: Array1<f64> = t_coords_full.slice(s![1..]).to_owned();

    let mut xx = Array3::<f64>::zeros((s, s, t_out));
    let mut yy = Array3::<f64>::zeros((s, s, t_out));
    let mut tt = Array3::<f64>::zeros((s, s, t_out));

    for i in 0..s {
        for j in 0..s {
            for k in 0..t_out {
                // No xy-swap: each grid varies along its own matching axis as fourier_3d.py direct reshape/repeat (not a meshgrid call)
                xx[[i, j, k]] = coords[i];
                yy[[i, j, k]] = coords[j];
                tt[[i, j, k]] = t_coords[k];
            }
        }
    }

    (xx, yy, tt)
}

// Append 3d grid coordinates (x, y, t) as three extra channels
pub fn append_grid_3d(data: ArrayD<f64>, xx: Array3<f64>, yy: Array3<f64>, tt: Array3<f64>) -> ArrayD<f64> {
    let batch_size = data.shape()[0];
    let s1 = data.shape()[1];
    let s2 = data.shape()[2];
    let t_out = data.shape()[3];

    let expand = |g: Array3<f64>| -> ArrayD<f64> {
        g.into_shape_with_order(IxDyn(&[1, s1, s2, t_out, 1])).unwrap().broadcast(IxDyn(&[batch_size, s1, s2, t_out, 1])).unwrap().to_owned()
    };

    let xx_expanded = expand(xx);
    let yy_expanded = expand(yy);
    let tt_expanded = expand(tt);

    // grid channels first, data last match Li's cat order
    concatenate(
        Axis(4), 
        &[xx_expanded.view(), yy_expanded.view(), tt_expanded.view(), data.view()],
    ).unwrap()
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{Array, IxDyn};

    #[test]
    fn test_uniform_grid_1d_correct_length() {
        let grid = uniform_grid_1d(0.0, 1.0, 10);
        assert_eq!(grid.len(), 10);
    }

    #[test]
    fn test_uniform_grid_1d_start_end() {
        let grid = uniform_grid_1d(0.0, 1.0, 10);
        assert_relative_eq!(grid[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(grid[9], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_uniform_grid_1d_uniform_spacing() {
        let grid = uniform_grid_1d(0.0, 1.0, 5);
        // spacing should be 0.25
        assert_relative_eq!(grid[1] - grid[0], 0.25, epsilon = 1e-10);
        assert_relative_eq!(grid[2] - grid[1], 0.25, epsilon = 1e-10);
        assert_relative_eq!(grid[3] - grid[2], 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_uniform_grid_1d_custom_range() {
        let grid = uniform_grid_1d(0.0, 2.0 * std::f64::consts::PI, 5);
        assert_relative_eq!(grid[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(grid[4], 2.0 * std::f64::consts::PI, epsilon = 1e-10);
    }

    #[test]
    fn test_append_grid_output_shape() {
        // [batch=2, s=4, 1] + grid[4] -> [2, 4, 2]
        let data = Array::from_shape_vec(
            IxDyn(&[2, 4, 1]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let grid = uniform_grid_1d(0.0, 1.0, 4);
        let result = append_grid_1d(data, grid);
        assert_eq!(result.shape(), &[2, 4, 2]);
    }

    #[test]
    fn test_append_grid_preserves_data() {
        // original data values should be unchanged in first channel
        let data = Array::from_shape_vec(
            IxDyn(&[2, 4, 1]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let grid = uniform_grid_1d(0.0, 1.0, 4);
        let result = append_grid_1d(data, grid);
        // first channel should be original data
        assert_relative_eq!(result[[0, 0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[[0, 1, 0]], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[[1, 0, 0]], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_append_grid_correct_grid_values() {
        // grid values should appear in second channel, same for all batch examples
        let data = Array::from_shape_vec(
            IxDyn(&[2, 4, 1]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let grid = uniform_grid_1d(0.0, 1.0, 4);
        let result = append_grid_1d(data, grid);
        // second channel should be grid values
        assert_relative_eq!(result[[0, 0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result[[0, 3, 1]], 1.0, epsilon = 1e-10);
        // grid is same for all batch examples
        assert_relative_eq!(result[[0, 1, 1]], result[[1, 1, 1]], epsilon = 1e-10);
    }

    #[test]
    fn test_uniform_grid_2d_matches_numpy_meshgrid() {
        let (xx, yy) = uniform_grid_2d(0.0, 1.0, 3);

        // Pinned against: np.meshgrid(np.linspace(0,1,3), np.linspace(0,1,3))
        let expected_xx = [
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        ];
        let expected_yy = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ];

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (xx[[i, j]] - expected_xx[i][j]).abs() < 1e-9,
                    "xx mismatch at [{},{}]: {} vs {}",
                    i, j, xx[[i, j]], expected_xx[i][j]
                );
                assert!(
                    (yy[[i, j]] - expected_yy[i][j]).abs() < 1e-9,
                    "yy mismatch at [{},{}]: {} vs {}",
                    i, j, yy[[i, j]], expected_yy[i][j]
                );
            }
        }
    }

    #[test]
    fn test_append_grid_2d_channel_placement() {
        // data: [batch=1, s=2, s=2, channels=1], all 9.0 so it's distinguishable from grid values
        let data = ArrayD::from_elem(IxDyn(&[1, 2, 2, 1]), 9.0);

        let (xx, yy) = uniform_grid_2d(0.0, 1.0, 2);

        let result = append_grid_2d(data, xx, yy);

        assert_eq!(result.shape(), &[1, 2, 2, 3]);

        // channel 0 should still be the original data (9.0 everywhere)
        assert_eq!(result[[0, 0, 0, 0]], 9.0);
        // channel 1 should be xx, channel 2 should be yy — spot check one position
        assert_eq!(result[[0, 1, 1, 1]], 1.0); // xx[1,1] = 1.0
        assert_eq!(result[[0, 1, 1, 2]], 1.0); // yy[1,1] = 1.0
    }

    #[test]
    fn test_uniform_grid_3d_axis_assignment() {
        let s = 4;
        let t_out = 3;
        let (xx, yy, tt) = uniform_grid_3d(0.0, 1.0, s, t_out);

        assert_eq!(xx.shape(), &[s, s, t_out]);
        assert_eq!(yy.shape(), &[s, s, t_out]);
        assert_eq!(tt.shape(), &[s, s, t_out]);

        // xx varies with i (dim 0), constant across j, k
        for i in 0..s {
            for j in 0..s {
                for k in 0..t_out {
                    assert!(
                        (xx[[i, j, k]] - xx[[i, 0, 0]]).abs() < 1e-9,
                        "xx should be constant across j,k at fixed i"
                    );
                }
            }
        }
        // yy varies with j (dim 1), constant across i, k
        for j in 0..s {
            for i in 0..s {
                for k in 0..t_out {
                    assert!(
                        (yy[[i, j, k]] - yy[[0, j, 0]]).abs() < 1e-9,
                        "yy should be constant across i,k at fixed j"
                    );
                }
            }
        }
        // tt varies with k (dim 2), constant across i, j
        for k in 0..t_out {
            for i in 0..s {
                for j in 0..s {
                    assert!(
                        (tt[[i, j, k]] - tt[[0, 0, k]]).abs() < 1e-9,
                        "tt should be constant across i,j at fixed k"
                    );
                }
            }
        }

        // no xy-swap: xx[i,*,*] should equal the i-th linspace value directly,
        // not the j-th — this is the specific bug a copy-pasted meshgrid-swap
        // convention from uniform_grid_2d would introduce.
        assert!((xx[[1, 0, 0]] - (1.0 / 3.0)).abs() < 1e-9, "xx[1] should be coords[1], not coords[j]");
        assert!((yy[[0, 2, 0]] - (2.0 / 3.0)).abs() < 1e-9, "yy[2] should be coords[2], not coords[i]");
    }

    #[test]
    fn test_uniform_grid_3d_time_excludes_zero() {
        let s = 4;
        let t_out = 4;
        let (xx, _yy, tt) = uniform_grid_3d(0.0, 1.0, s, t_out);

        // x/y: closed interval [0,1] — first point IS 0
        assert!((xx[[0, 0, 0]] - 0.0).abs() < 1e-9, "x should include 0 as its first point");
        assert!((xx[[s - 1, 0, 0]] - 1.0).abs() < 1e-9, "x should include 1 as its last point");

        // t: half-open (0,1] — first point is 1/t_out, NOT 0; last point IS 1
        let expected_first_t = 1.0 / t_out as f64;
        assert!(
            (tt[[0, 0, 0]] - expected_first_t).abs() < 1e-9,
            "t's first point should be 1/t_out ({}), got {} — check for an accidental switch to plain linspace(0,1,t_out)",
            expected_first_t, tt[[0, 0, 0]]
        );
        assert!((tt[[0, 0, t_out - 1]] - 1.0).abs() < 1e-9, "t's last point should be 1.0");
    }

    #[test]
    fn test_append_grid_3d_channel_order_and_broadcast() {
        let s = 2;
        let t_out = 2;
        let batch_size = 2;
        let c = 1; // single data channel, distinct values per batch to catch a broadcast bug

        let data = Array::from_shape_fn(IxDyn(&[batch_size, s, s, t_out, c]), |idx| {
            (idx[0] * 1000) as f64 // batch index baked into the value
        });

        let (xx, yy, tt) = uniform_grid_3d(0.0, 1.0, s, t_out);
        let out = append_grid_3d(data.clone(), xx.clone(), yy.clone(), tt.clone());

        assert_eq!(out.shape(), &[batch_size, s, s, t_out, 3 + c]);

        for b in 0..batch_size {
            for i in 0..s {
                for j in 0..s {
                    for k in 0..t_out {
                        // grid channels first (0,1,2) — same value regardless of batch
                        assert!((out[[b, i, j, k, 0]] - xx[[i, j, k]]).abs() < 1e-9, "channel 0 should be xx");
                        assert!((out[[b, i, j, k, 1]] - yy[[i, j, k]]).abs() < 1e-9, "channel 1 should be yy");
                        assert!((out[[b, i, j, k, 2]] - tt[[i, j, k]]).abs() < 1e-9, "channel 2 should be tt");
                        // data channel last (3) and correctly broadcast per-batch
                        assert!(
                            (out[[b, i, j, k, 3]] - data[[b, i, j, k, 0]]).abs() < 1e-9,
                            "channel 3 should be original data, batch {}",
                            b
                        );
                    }
                }
            }
        }
    }
}
