//! Builds the Phase 5 comparison table (docs/phase5/T6-evaluate.md), or with `--by-n` the
//! Phase 7 one (docs/phase7/T3-error-vs-n.md), and their plots from evaluated Poisson runs.
//!
//! Run with: cargo run --release --example poisson_matrix -- docs/phase5/results.md runs/<run_dir> ...
//! or: cargo run --release --example poisson_matrix -- --by-n docs/phase7/results.md runs/<run_dir> ...
//!
//! Every run must have been evaluated by `predict_poisson` first (its `errors.json`, with
//! the run's s). Reads each run's configs (s, padding, seed), `metrics.csv` (final train /
//! test relative L², time per epoch), `errors.json` (errors 2–4, error 4 and ‖u‖ per
//! sample) and `costs.json` when present (runs since Phase 5 T4). Runs are grouped into
//! cells by (padding, s): the main arm is p = 0, any other p a padding control. Writes the
//! Markdown table to the first argument: one row per run, then one per cell with the mean
//! and the min–max over seeds of each error, and the costs. Draws four plots with gnuplot
//! beside it, `<stem>_<plot>.png` (skipped with a warning if gnuplot is missing): mean
//! fine-grid error (error 4) and its transfer-only floor (error 2) against s; time per
//! epoch against s; error 4 against total training time; and error 4 against ‖u‖ per test
//! sample for the s = 128 runs (design §12, decision 22). Loads no model and no dataset.
//!
//! With `--by-n` (Phase 7, design §13.5) the runs must share one (padding, s) and one
//! backend (costs compare only within one, design §13.14), and are
//! grouped by training-set size N, `base.n_train`: a subset run's N, the whole split
//! otherwise, so the Phase 5 runs at that s are the N = 1000 row. Per N, the mean and
//! min–max over all its runs of errors 3 and 4, the final losses, steps and costs; whether
//! the mean error 4 meets D7 ([`D7_THRESHOLD`], design §13.15) and whether every run does;
//! the smallest N whose mean meets it; error 4 averaged per subset seed and per seed; and
//! the extra training time and samples per fall in mean error 4 between successive N.
//! Plots: error 4 against N (per run, mean, min–max band and the D7 line) and training
//! time against N, `<stem>_error_vs_n.png` and `<stem>_time_vs_n.png`.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use burn::config::Config;
use gnuplot::{AutoOption::Fix, AxesCommon, Caption, Figure, FillAlpha, PointSymbol};
use sciml_rs::neural_operators::{
    data::loaders::poisson::PoissonConfig,
    metrics::poisson::{ERRORS_JSON, PoissonErrorsRecord},
    runs::{EpochMetrics, RunCosts, RunDir},
};

/// The learning grid of the error 4 against ‖u‖ scatter: the coupled grid at n = 129.
const SCATTER_S: usize = 128;

/// The s axis of the plots against s, wide enough that s = 16 and 128 sit inside it.
const S_RANGE: (f64, f64) = (12.0, 170.0);

/// D7 (design §13.15, decided 2026-10-29): the fine-grid test relative L² a training-set
/// size must reach in the mean over its runs.
const D7_THRESHOLD: f64 = 2e-2;

/// The N axis of the plots against N, wide enough that N = 25 and 1000 sit inside it.
const N_RANGE: (f64, f64) = (15.0, 1600.0);

/// One evaluated run.
struct Run {
    name: String,
    padding: usize,
    s: usize,
    /// Training-set size N, `base.n_train`.
    n: usize,
    /// The subset seed; `None` for a run on the whole split.
    subset_seed: Option<u64>,
    seed: u64,
    /// Optimisation steps, `TrainingConfig::total_steps(n)`.
    steps: usize,
    last: EpochMetrics,
    /// Median time per epoch over the epochs after the first (which includes warm-up);
    /// `None` for a run before Phase 5 T4 or with a single epoch.
    epoch_time_s: Option<f64>,
    errors: PoissonErrorsRecord,
    costs: Option<RunCosts>,
}

impl Run {
    fn load(dir: &Path) -> Self {
        let fail = |what: &str, e: &dyn std::fmt::Display| -> ! {
            panic!("{}: {what}: {e}", dir.display())
        };
        let run = RunDir::open(dir).unwrap_or_else(|e| fail("open", &e));
        let data = run
            .data_config::<PoissonConfig>()
            .unwrap_or_else(|e| fail("data config", &e));
        let padding = run
            .model_config()
            .unwrap_or_else(|e| fail("model config", &e))
            .padding
            .unwrap_or(0);
        let train_cfg = run
            .training_config()
            .unwrap_or_else(|e| fail("training config", &e));
        let (n, subset_seed) = sample_size(&data);
        let metrics = run.metrics().unwrap_or_else(|e| fail("metrics.csv", &e));
        let mut times: Vec<f64> = metrics
            .iter()
            .skip(1)
            .filter_map(|m| m.epoch_time_s)
            .collect();
        times.sort_by(f64::total_cmp);
        let errors = PoissonErrorsRecord::load(dir.join(ERRORS_JSON))
            .unwrap_or_else(|e| fail("errors.json (run predict_poisson first)", &e));
        assert_eq!(
            errors.s,
            data.s(),
            "{}: errors.json is from another s",
            dir.display()
        );
        Self {
            name: dir.file_name().map_or_else(
                || dir.display().to_string(),
                |f| f.to_string_lossy().into_owned(),
            ),
            padding,
            s: data.s(),
            n,
            subset_seed,
            seed: train_cfg.seed,
            steps: train_cfg.total_steps(n),
            last: metrics
                .last()
                .cloned()
                .unwrap_or_else(|| fail("metrics.csv", &"no epochs")),
            epoch_time_s: (!times.is_empty()).then(|| times[times.len() / 2]),
            errors,
            costs: run.costs().unwrap_or_else(|e| fail("costs.json", &e)),
        }
    }

    fn arm(&self) -> String {
        if self.padding == 0 {
            "main (p = 0)".into()
        } else {
            format!("padding (p = {})", self.padding)
        }
    }
}

/// A run's training-set size and subset seed: `base.n_train` whether or not a subset was
/// drawn, so a run on the whole split counts as N = its size with no subset seed.
fn sample_size(data: &PoissonConfig) -> (usize, Option<u64>) {
    (data.base.n_train, data.subset.as_ref().map(|s| s.seed))
}

/// The mean of `v`; NaN if empty.
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// The mean of the values per key, keys ascending.
fn means_by<K: Ord>(items: impl IntoIterator<Item = (K, f64)>) -> BTreeMap<K, f64> {
    let mut groups: BTreeMap<K, Vec<f64>> = BTreeMap::new();
    for (k, v) in items {
        groups.entry(k).or_default().push(v);
    }
    groups.into_iter().map(|(k, v)| (k, mean(&v))).collect()
}

/// The smallest key whose mean is at most `threshold`.
fn smallest_meeting(means: &BTreeMap<usize, f64>, threshold: f64) -> Option<usize> {
    means
        .iter()
        .find(|&(_, &m)| m <= threshold)
        .map(|(&n, _)| n)
}

/// From one training-set size to the next larger one.
#[derive(Debug, PartialEq)]
struct Step {
    from: usize,
    to: usize,
    /// The fall in mean error 4, positive when the larger N is better.
    fall: f64,
    /// The extra mean training time in seconds, when both sizes have one.
    extra_time: Option<f64>,
}

/// The steps between successive sizes of `points`, `(N, mean error 4, mean training
/// time)` in ascending N.
fn steps_between(points: &[(usize, f64, Option<f64>)]) -> Vec<Step> {
    points
        .windows(2)
        .map(|w| Step {
            from: w[0].0,
            to: w[1].0,
            fall: w[0].1 - w[1].1,
            extra_time: w[0].2.zip(w[1].2).map(|(a, b)| b - a),
        })
        .collect()
}

/// `mean [min–max]` of `v`, or "–" if empty.
fn spread(v: &[f64], fmt: fn(f64) -> String) -> String {
    if v.is_empty() {
        return "–".into();
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let (lo, hi) = v
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        });
    format!("{} [{}–{}]", fmt(mean), fmt(lo), fmt(hi))
}

fn sci(x: f64) -> String {
    format!("{x:.3e}")
}

fn secs(x: f64) -> String {
    format!("{x:.1}")
}

fn opt(x: Option<f64>, fmt: fn(f64) -> String) -> String {
    x.map_or("–".into(), fmt)
}

fn table(runs: &[Run], cells: &BTreeMap<(usize, usize), Vec<&Run>>) -> String {
    let mut md = String::new();
    let w = &mut md;
    writeln!(w, "# Phase 5 results\n").unwrap();
    writeln!(
        w,
        "Generated by `poisson_matrix` from {} runs, each evaluated by `predict_poisson` on the \
         test split (n = 129 CGL nodes). Error 2 is the transfer-only floor ‖P R u − u‖ / ‖u‖, \
         error 3 the coarse error on the learning grid, error 4 the fine-grid error on the \
         CGL grid (design §13.3). Times are wall-clock seconds; s/epoch is the median over \
         epochs after the first.\n",
        runs.len()
    )
    .unwrap();

    writeln!(w, "## Per arm and learning grid\n").unwrap();
    writeln!(w, "Mean [min–max] over the seeds of each cell.\n").unwrap();
    writeln!(
        w,
        "| arm | s | seeds | error 2 (floor) | error 3 (coarse) | error 4 (fine) | final test L² | s/epoch | training time | inference time | peak GPU memory | host-to-device bytes (train) | parameters | backend |"
    )
    .unwrap();
    writeln!(w, "|{}", " --- |".repeat(14)).unwrap();
    for ((_, s), cell) in cells {
        let col = |f: fn(&Run) -> Option<f64>| cell.iter().filter_map(|r| f(r)).collect::<Vec<_>>();
        let seeds: Vec<String> = cell.iter().map(|r| r.seed.to_string()).collect();
        let costs = cell.iter().find_map(|r| r.costs.as_ref());
        let mut peaks: Vec<&str> = cell
            .iter()
            .filter_map(|r| r.costs.as_ref().map(|c| c.peak_gpu_memory.as_str()))
            .collect();
        peaks.sort_unstable();
        peaks.dedup();
        writeln!(
            w,
            "| {} | {s} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            cell[0].arm(),
            seeds.join(", "),
            spread(&col(|r| Some(r.errors.error2.mean)), sci),
            spread(&col(|r| Some(r.errors.error3.mean)), sci),
            spread(&col(|r| Some(r.errors.error4.mean)), sci),
            spread(&col(|r| Some(f64::from(r.last.test_l2))), sci),
            spread(&col(|r| r.epoch_time_s), secs),
            spread(&col(|r| r.costs.as_ref().map(|c| c.training_time_s)), secs),
            spread(
                &col(|r| r.costs.as_ref().and_then(|c| c.inference_time_s)),
                secs
            ),
            if peaks.is_empty() {
                "–".into()
            } else {
                peaks.join(", ")
            },
            costs.map_or("–".into(), |c| c.host_to_device_bytes_train.to_string()),
            costs.map_or("–".into(), |c| c.num_params.to_string()),
            costs.map_or("–".into(), |c| c.backend_features.join(", ")),
        )
        .unwrap();
    }

    writeln!(w, "\n## Per run\n").unwrap();
    writeln!(
        w,
        "| run | p | s | seed | final train L² | final test L² | error 2 mean | error 3 mean | error 4 mean / max | s/epoch | training time | corr(ln error 4, ln ‖u‖) |"
    )
    .unwrap();
    writeln!(w, "|{}", " --- |".repeat(12)).unwrap();
    for r in cells.values().flatten() {
        writeln!(
            w,
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} / {} | {} | {} | {:.3} |",
            r.name,
            r.padding,
            r.s,
            r.seed,
            sci(f64::from(r.last.train_l2)),
            sci(f64::from(r.last.test_l2)),
            sci(r.errors.error2.mean),
            sci(r.errors.error3.mean),
            sci(r.errors.error4.mean),
            sci(r.errors.error4.max),
            opt(r.epoch_time_s, secs),
            opt(r.costs.as_ref().map(|c| c.training_time_s), secs),
            r.errors.error4_u_norm_log_correlation,
        )
        .unwrap();
    }
    md
}

/// Draws the four plots beside `out`; returns the files written.
fn plots(out: &Path, cells: &BTreeMap<(usize, usize), Vec<&Run>>) -> Result<Vec<PathBuf>, String> {
    let stem = out.file_stem().unwrap_or_default().to_string_lossy();
    let path = |name: &str| out.with_file_name(format!("{stem}_{name}.png"));
    let mut written = Vec::new();
    let mut arms: BTreeMap<usize, Vec<&Run>> = BTreeMap::new();
    for ((p, _), cell) in cells {
        arms.entry(*p).or_default().extend(cell);
    }
    let label = |p: usize| {
        if p == 0 {
            "main (p = 0)".to_string()
        } else {
            format!("p = {p}")
        }
    };
    let save = |fig: &mut Figure, file: PathBuf, written: &mut Vec<PathBuf>| {
        fig.save_to_png(&file, 1000, 700)
            .map_err(|e| format!("{e} (is gnuplot installed?)"))?;
        written.push(file);
        Ok::<_, String>(())
    };

    // 1. fine-grid error per run and the floor per s, against s
    let mut fig = Figure::new();
    let ax = fig
        .axes2d()
        .set_title(
            "Fine-grid error (error 4) and its transfer-only floor (error 2)",
            &[],
        )
        .set_x_label("learning grid s", &[])
        .set_y_label("relative L² on the CGL grid", &[])
        .set_x_log(Some(2.0))
        .set_x_range(Fix(S_RANGE.0), Fix(S_RANGE.1))
        .set_y_log(Some(10.0));
    for (p, runs) in &arms {
        let s: Vec<f64> = runs.iter().map(|r| r.s as f64).collect();
        let e4: Vec<f64> = runs.iter().map(|r| r.errors.error4.mean).collect();
        ax.points(
            &s,
            &e4,
            &[
                Caption(&format!("error 4, {}", label(*p))),
                PointSymbol('O'),
            ],
        );
    }
    let (fs, floor): (Vec<f64>, Vec<f64>) = cells
        .iter()
        .filter(|((p, _), _)| *p == 0)
        .map(|((_, s), c)| (*s as f64, c[0].errors.error2.mean))
        .unzip();
    ax.lines_points(&fs, &floor, &[Caption("error 2 (floor)"), PointSymbol('x')]);
    save(&mut fig, path("error_vs_s"), &mut written)?;

    // 2. time per epoch against s
    let mut fig = Figure::new();
    let ax = fig
        .axes2d()
        .set_title("Time per epoch", &[])
        .set_x_label("learning grid s", &[])
        .set_y_label("s/epoch (median)", &[])
        .set_x_log(Some(2.0))
        .set_x_range(Fix(S_RANGE.0), Fix(S_RANGE.1))
        .set_y_log(Some(10.0));
    for (p, runs) in &arms {
        let (s, t): (Vec<f64>, Vec<f64>) = runs
            .iter()
            .filter_map(|r| r.epoch_time_s.map(|t| (r.s as f64, t)))
            .unzip();
        ax.points(&s, &t, &[Caption(&label(*p)), PointSymbol('O')]);
    }
    save(&mut fig, path("epoch_time_vs_s"), &mut written)?;

    // 3. fine-grid error against total training time, one series per s
    let mut fig = Figure::new();
    let ax = fig
        .axes2d()
        .set_title("Fine-grid error against training time", &[])
        .set_x_label("training time (s)", &[])
        .set_y_label("error 4 mean", &[])
        .set_x_log(Some(10.0))
        .set_y_log(Some(10.0));
    for ((p, s), cell) in cells {
        let (t, e4): (Vec<f64>, Vec<f64>) = cell
            .iter()
            .filter_map(|r| {
                r.costs
                    .as_ref()
                    .map(|c| (c.training_time_s, r.errors.error4.mean))
            })
            .unzip();
        ax.points(
            &t,
            &e4,
            &[
                Caption(&format!("s = {s}, {}", label(*p))),
                PointSymbol('O'),
            ],
        );
    }
    save(&mut fig, path("error_vs_time"), &mut written)?;

    // 4. error 4 against ||u|| per test sample, s = 128 runs
    let mut fig = Figure::new();
    let ax = fig
        .axes2d()
        .set_title(
            &format!("Error 4 against ‖u‖, test samples, s = {SCATTER_S}"),
            &[],
        )
        .set_x_label("‖u‖ (Clenshaw–Curtis)", &[])
        .set_y_label("error 4", &[])
        .set_x_log(Some(10.0))
        .set_y_log(Some(10.0));
    for r in cells
        .iter()
        .filter(|((_, s), _)| *s == SCATTER_S)
        .flat_map(|(_, c)| c)
    {
        ax.points(
            &r.errors.u_norm_per_sample,
            &r.errors.error4_per_sample,
            &[
                Caption(&format!(
                    "{}, seed {} (corr {:.2})",
                    label(r.padding),
                    r.seed,
                    r.errors.error4_u_norm_log_correlation
                )),
                PointSymbol('o'),
            ],
        );
    }
    save(&mut fig, path("error4_vs_unorm"), &mut written)?;
    Ok(written)
}

/// The Phase 7 table: one row per N, its D7 verdict, the spread by seed, the cost per fall
/// in error between successive N, then one row per run.
fn table_by_n(runs: &[Run], sizes: &BTreeMap<usize, Vec<&Run>>) -> String {
    let mut md = String::new();
    let w = &mut md;
    let e4_means = means_by(runs.iter().map(|r| (r.n, r.errors.error4.mean)));
    writeln!(w, "# Phase 7 results\n").unwrap();
    writeln!(
        w,
        "Generated by `poisson_matrix --by-n` from {} runs at s = {}, p = {}, each evaluated by \
         `predict_poisson` on the same 200 test samples (n = 129 CGL nodes). N is the \
         training-set size; the N = 1000 runs are on the whole split, with no subset seed. \
         Error 3 is the coarse error on the learning grid, error 4 the fine-grid error on the \
         CGL grid (design §13.3). D7: mean error 4 over an N's runs ≤ {D7_THRESHOLD:e} \
         (design §13.15). Every N trains for the same epochs, so steps grow with N (D6). Times \
         are wall-clock seconds; s/epoch is the median over epochs after the first.\n",
        runs.len(),
        runs[0].s,
        runs[0].padding,
    )
    .unwrap();

    writeln!(w, "## Per training-set size\n").unwrap();
    writeln!(w, "Mean [min–max] over the runs of each N.\n").unwrap();
    writeln!(
        w,
        "| N | runs | subset seeds | seeds | steps | error 3 (coarse) | error 4 (fine) | D7, mean | D7, every run | final train L² | final test L² | s/epoch | training time |"
    )
    .unwrap();
    writeln!(w, "|{}", " --- |".repeat(13)).unwrap();
    for (n, group) in sizes {
        let col =
            |f: fn(&Run) -> Option<f64>| group.iter().filter_map(|r| f(r)).collect::<Vec<_>>();
        let distinct = |f: fn(&Run) -> String| {
            let mut v: Vec<String> = group.iter().map(|r| f(r)).collect();
            v.sort_unstable();
            v.dedup();
            v.join(", ")
        };
        let yes = |b: bool| if b { "yes" } else { "no" };
        writeln!(
            w,
            "| {n} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            group.len(),
            distinct(|r| r.subset_seed.map_or("–".into(), |s| s.to_string())),
            distinct(|r| r.seed.to_string()),
            distinct(|r| r.steps.to_string()),
            spread(&col(|r| Some(r.errors.error3.mean)), sci),
            spread(&col(|r| Some(r.errors.error4.mean)), sci),
            yes(e4_means[n] <= D7_THRESHOLD),
            yes(group.iter().all(|r| r.errors.error4.mean <= D7_THRESHOLD)),
            spread(&col(|r| Some(f64::from(r.last.train_l2))), sci),
            spread(&col(|r| Some(f64::from(r.last.test_l2))), sci),
            spread(&col(|r| r.epoch_time_s), secs),
            spread(&col(|r| r.costs.as_ref().map(|c| c.training_time_s)), secs),
        )
        .unwrap();
    }
    match smallest_meeting(&e4_means, D7_THRESHOLD) {
        Some(n) => writeln!(
            w,
            "\n**Smallest N meeting D7: {n}**, mean error 4 {} over {} runs; every run meets it: {}.",
            spread(
                &sizes[&n].iter().map(|r| r.errors.error4.mean).collect::<Vec<_>>(),
                sci
            ),
            sizes[&n].len(),
            if sizes[&n].iter().all(|r| r.errors.error4.mean <= D7_THRESHOLD) {
                "yes"
            } else {
                "no"
            },
        ),
        None => writeln!(w, "\n**No N meets D7.**"),
    }
    .unwrap();

    writeln!(w, "\n## Spread by seed\n").unwrap();
    writeln!(
        w,
        "Mean error 4 per subset seed (over its seeds) and per seed (over its subsets).\n"
    )
    .unwrap();
    writeln!(w, "| N | by subset seed | by seed |").unwrap();
    writeln!(w, "|{}", " --- |".repeat(3)).unwrap();
    for (n, group) in sizes {
        let list = |m: BTreeMap<String, f64>| {
            m.iter()
                .map(|(k, v)| format!("{k}: {}", sci(*v)))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let by_subset = means_by(group.iter().map(|r| {
            let key = r.subset_seed.map_or("–".into(), |s| s.to_string());
            (key, r.errors.error4.mean)
        }));
        let by_seed = means_by(
            group
                .iter()
                .map(|r| (r.seed.to_string(), r.errors.error4.mean)),
        );
        writeln!(w, "| {n} | {} | {} |", list(by_subset), list(by_seed)).unwrap();
    }

    writeln!(w, "\n## Cost per further reduction in error\n").unwrap();
    writeln!(
        w,
        "Between successive N: the fall in mean error 4, and the extra samples and mean \
         training time it took, per 1e-3 of that fall.\n"
    )
    .unwrap();
    writeln!(
        w,
        "| from N | to N | fall in mean error 4 | extra samples | extra training time | samples per 1e-3 | seconds per 1e-3 |"
    )
    .unwrap();
    writeln!(w, "|{}", " --- |".repeat(7)).unwrap();
    let points: Vec<(usize, f64, Option<f64>)> = sizes
        .iter()
        .map(|(n, group)| {
            let times: Vec<f64> = group
                .iter()
                .filter_map(|r| r.costs.as_ref().map(|c| c.training_time_s))
                .collect();
            (*n, e4_means[n], (!times.is_empty()).then(|| mean(&times)))
        })
        .collect();
    for step in steps_between(&points) {
        let samples = (step.to - step.from) as f64;
        let per = |x: f64| {
            if step.fall > 0.0 {
                format!("{:.1}", x / (step.fall / 1e-3))
            } else {
                "no fall".into()
            }
        };
        writeln!(
            w,
            "| {} | {} | {} | {} | {} | {} | {} |",
            step.from,
            step.to,
            sci(step.fall),
            step.to - step.from,
            opt(step.extra_time, secs),
            per(samples),
            step.extra_time.map_or("–".into(), per),
        )
        .unwrap();
    }

    writeln!(w, "\n## Per run\n").unwrap();
    writeln!(
        w,
        "| run | N | subset seed | seed | steps | final train L² | final test L² | error 3 mean | error 4 mean / max | s/epoch | training time |"
    )
    .unwrap();
    writeln!(w, "|{}", " --- |".repeat(11)).unwrap();
    for r in sizes.values().flatten() {
        writeln!(
            w,
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} / {} | {} | {} |",
            r.name,
            r.n,
            r.subset_seed.map_or("–".into(), |s| s.to_string()),
            r.seed,
            r.steps,
            sci(f64::from(r.last.train_l2)),
            sci(f64::from(r.last.test_l2)),
            sci(r.errors.error3.mean),
            sci(r.errors.error4.mean),
            sci(r.errors.error4.max),
            opt(r.epoch_time_s, secs),
            opt(r.costs.as_ref().map(|c| c.training_time_s), secs),
        )
        .unwrap();
    }
    md
}

/// Draws the two Phase 7 plots beside `out`; returns the files written.
fn plots_by_n(out: &Path, sizes: &BTreeMap<usize, Vec<&Run>>) -> Result<Vec<PathBuf>, String> {
    let stem = out.file_stem().unwrap_or_default().to_string_lossy();
    let path = |name: &str| out.with_file_name(format!("{stem}_{name}.png"));
    let mut written = Vec::new();
    let ns: Vec<f64> = sizes.keys().map(|&n| n as f64).collect();
    let per_n = |f: fn(&Run) -> Option<f64>| -> Vec<Vec<f64>> {
        sizes
            .values()
            .map(|g| g.iter().filter_map(|r| f(r)).collect())
            .collect()
    };
    let extremes = |v: &[Vec<f64>]| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let lo = v
            .iter()
            .map(|g| g.iter().copied().fold(f64::INFINITY, f64::min));
        let hi = v
            .iter()
            .map(|g| g.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        (
            lo.collect(),
            v.iter().map(|g| mean(g)).collect(),
            hi.collect(),
        )
    };
    let scatter = |f: fn(&Run) -> Option<f64>| -> (Vec<f64>, Vec<f64>) {
        sizes
            .values()
            .flatten()
            .filter_map(|r| f(r).map(|y| (r.n as f64, y)))
            .unzip()
    };

    // 1. fine-grid error against N, with the D7 line
    let e4 = per_n(|r| Some(r.errors.error4.mean));
    let (lo, m, hi) = extremes(&e4);
    let (x, y) = scatter(|r| Some(r.errors.error4.mean));
    let mut fig = Figure::new();
    fig.axes2d()
        .set_title("Fine-grid error (error 4) against training-set size", &[])
        .set_x_label("training samples N", &[])
        .set_y_label("relative L² on the CGL grid", &[])
        .set_x_log(Some(10.0))
        .set_x_range(Fix(N_RANGE.0), Fix(N_RANGE.1))
        .set_y_log(Some(10.0))
        .fill_between(&ns, &lo, &hi, &[Caption("min–max"), FillAlpha(0.2)])
        .lines_points(&ns, &m, &[Caption("mean"), PointSymbol('O')])
        .points(&x, &y, &[Caption("runs"), PointSymbol('x')])
        .lines(
            [N_RANGE.0, N_RANGE.1],
            [D7_THRESHOLD; 2],
            &[Caption("D7 threshold")],
        );
    fig.save_to_png(path("error_vs_n"), 1000, 700)
        .map_err(|e| format!("{e} (is gnuplot installed?)"))?;
    written.push(path("error_vs_n"));

    // 2. training time against N
    let times = per_n(|r| r.costs.as_ref().map(|c| c.training_time_s));
    let (_, m, _) = extremes(&times);
    let (x, y) = scatter(|r| r.costs.as_ref().map(|c| c.training_time_s));
    let mut fig = Figure::new();
    fig.axes2d()
        .set_title("Training time against training-set size", &[])
        .set_x_label("training samples N", &[])
        .set_y_label("training time (s)", &[])
        .set_x_log(Some(10.0))
        .set_x_range(Fix(N_RANGE.0), Fix(N_RANGE.1))
        .set_y_log(Some(10.0))
        .lines_points(&ns, &m, &[Caption("mean"), PointSymbol('O')])
        .points(&x, &y, &[Caption("runs"), PointSymbol('x')]);
    fig.save_to_png(path("time_vs_n"), 1000, 700)
        .map_err(|e| format!("{e} (is gnuplot installed?)"))?;
    written.push(path("time_vs_n"));
    Ok(written)
}

/// The Phase 7 output (`--by-n`): every run at one (padding, s), grouped by N.
fn by_n(out: &Path, runs: &[Run]) {
    let grids: BTreeMap<(usize, usize), usize> = runs.iter().fold(BTreeMap::new(), |mut m, r| {
        *m.entry((r.padding, r.s)).or_default() += 1;
        m
    });
    assert!(
        grids.len() == 1,
        "--by-n compares one (padding, s); got runs at (p, s) = {:?}",
        grids.keys().collect::<Vec<_>>()
    );
    // costs compare only within one backend (design §13.14)
    let mut backends: Vec<&Vec<String>> = runs
        .iter()
        .filter_map(|r| r.costs.as_ref().map(|c| &c.backend_features))
        .collect();
    backends.sort_unstable();
    backends.dedup();
    assert!(
        backends.len() <= 1,
        "--by-n compares costs on one backend; got runs on {backends:?}"
    );
    let mut sizes: BTreeMap<usize, Vec<&Run>> = BTreeMap::new();
    for r in runs {
        sizes.entry(r.n).or_default().push(r);
    }
    for group in sizes.values_mut() {
        group.sort_by_key(|r| (r.subset_seed, r.seed));
    }
    std::fs::write(out, table_by_n(runs, &sizes)).expect("write the table");
    println!(
        "wrote {} ({} runs, {} sizes)",
        out.display(),
        runs.len(),
        sizes.len()
    );
    match plots_by_n(out, &sizes) {
        Ok(files) => files.iter().for_each(|f| println!("wrote {}", f.display())),
        Err(e) => eprintln!("plots skipped: {e}"),
    }
}

fn main() {
    let mut args = std::env::args().skip(1).peekable();
    let usage = "usage: poisson_matrix [--by-n] <out.md> <run_dir> ...";
    let by_size = args.next_if(|a| a == "--by-n").is_some();
    let out = PathBuf::from(args.next().expect(usage));
    let runs: Vec<Run> = args.map(|d| Run::load(Path::new(&d))).collect();
    assert!(!runs.is_empty(), "{usage}");
    if by_size {
        return by_n(&out, &runs);
    }

    let mut cells: BTreeMap<(usize, usize), Vec<&Run>> = BTreeMap::new();
    for r in &runs {
        cells.entry((r.padding, r.s)).or_default().push(r);
    }
    for cell in cells.values_mut() {
        cell.sort_by_key(|r| r.seed);
    }

    std::fs::write(&out, table(&runs, &cells)).expect("write the table");
    println!(
        "wrote {} ({} runs, {} cells)",
        out.display(),
        runs.len(),
        cells.len()
    );
    match plots(&out, &cells) {
        Ok(files) => files.iter().for_each(|f| println!("wrote {}", f.display())),
        Err(e) => eprintln!("plots skipped: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sciml_rs::neural_operators::data::loaders::{
        base_dataset::DatasetConfig, poisson::SubsetConfig,
    };

    #[test]
    fn a_run_without_subset_is_the_whole_split() {
        let whole = PoissonConfig::new(
            DatasetConfig {
                n_train: 1000,
                n_test: 200,
            },
            129,
        );
        assert_eq!(sample_size(&whole), (1000, None));
        let mut sub = whole.clone();
        sub.base.n_train = 25;
        let sub = sub.with_subset(Some(SubsetConfig::draw(2, 1000, 25)));
        assert_eq!(sample_size(&sub), (25, Some(2)));
    }

    #[test]
    fn groups_by_n_and_finds_the_smallest_meeting_d7() {
        // (N, error 4) of nine runs: N = 1000 has no subset, as the Phase 5 runs
        let runs = [
            (25, 0.30),
            (25, 0.20),
            (100, 0.03),
            (100, 0.01),
            (250, 0.015),
            (250, 0.025),
            (1000, 0.011),
            (1000, 0.010),
            (1000, 0.012),
        ];
        let means = means_by(runs);
        assert_eq!(
            means.keys().copied().collect::<Vec<_>>(),
            [25, 100, 250, 1000]
        );
        assert!((means[&25] - 0.25).abs() < 1e-15);
        // N = 100 meets it in the mean (0.02) although one run does not
        assert_eq!(smallest_meeting(&means, 0.02), Some(100));
        assert_eq!(smallest_meeting(&means, 0.005), None);
    }

    #[test]
    fn steps_between_successive_sizes() {
        let steps = steps_between(&[
            (25, 0.25, Some(80.0)),
            (50, 0.15, Some(95.0)),
            (100, 0.16, None),
        ]);
        assert_eq!(steps.len(), 2);
        assert_eq!((steps[0].from, steps[0].to), (25, 50));
        assert!((steps[0].fall - 0.10).abs() < 1e-15);
        assert_eq!(steps[0].extra_time, Some(15.0));
        // error rose: a negative fall, and no time without both means
        assert!(steps[1].fall < 0.0);
        assert_eq!(steps[1].extra_time, None);
        assert!(steps_between(&[(25, 0.1, None)]).is_empty());
    }
}
