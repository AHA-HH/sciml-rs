//! Training-curve plots (loss + learning rate) via gnuplot.

use crate::neural_operators::training::metrics::EpochMetrics;
use gnuplot::{AxesCommon, Caption, Color, Figure, GnuplotInitError, LineWidth};
use std::path::Path;

/// Two-panel plot: relative L2 loss (train/test) and learning rate schedule.
///
/// # Errors
/// If gnuplot can't be started (e.g. it isn't installed or not on `PATH`).
pub fn plot_metrics(
    path: impl AsRef<Path>,
    metrics: &[EpochMetrics],
    title: &str,
) -> Result<(), GnuplotInitError> {
    let epochs: Vec<f32> = metrics.iter().map(|m| m.epoch as f32).collect();
    let train_l2: Vec<f32> = metrics.iter().map(|m| m.train_l2).collect();
    let test_l2: Vec<f32> = metrics.iter().map(|m| m.test_l2).collect();
    let lr: Vec<f64> = metrics.iter().map(|m| m.current_lr).collect();

    let mut fig = Figure::new();

    // Plot 1: loss curves
    fig.set_multiplot_layout(2, 1).set_title(title);

    fig.axes2d()
        .set_title("Relative L2 Loss", &[])
        .set_x_label("Epoch", &[])
        .set_y_label("L2", &[])
        .lines(
            &epochs,
            &train_l2,
            &[
                Caption("train_l2"),
                Color(gnuplot::RGBString("blue")),
                LineWidth(1.5),
            ],
        )
        .lines(
            &epochs,
            &test_l2,
            &[
                Caption("test_l2"),
                Color(gnuplot::RGBString("red")),
                LineWidth(1.5),
            ],
        );

    // Plot 2: LR schedule
    fig.axes2d()
        .set_title("Learning Rate", &[])
        .set_x_label("Epoch", &[])
        .set_y_label("LR", &[])
        .set_y_log(Some(10.0))
        .lines(
            &epochs,
            &lr,
            &[
                Caption("lr"),
                Color(gnuplot::RGBString("green")),
                LineWidth(1.5),
            ],
        );

    fig.save_to_png(path, 1200, 800)
}
