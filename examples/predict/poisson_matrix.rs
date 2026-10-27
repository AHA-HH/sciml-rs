//! Builds the Phase 5 comparison table (docs/phase5/T6-evaluate.md) and its plots from
//! evaluated Poisson runs.
//!
//! Run with: cargo run --release --example poisson_matrix -- docs/phase5/results.md runs/<run_dir> ...
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

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use burn::config::Config;
use gnuplot::{AutoOption::Fix, AxesCommon, Caption, Figure, PointSymbol};
use sciml_rs::neural_operators::{
    data::loaders::poisson::PoissonConfig,
    metrics::poisson::{ERRORS_JSON, PoissonErrorsRecord},
    runs::{EpochMetrics, RunCosts, RunDir},
};

/// The learning grid of the error 4 against ‖u‖ scatter: the coupled grid at n = 129.
const SCATTER_S: usize = 128;

/// The s axis of the plots against s, wide enough that s = 16 and 128 sit inside it.
const S_RANGE: (f64, f64) = (12.0, 170.0);

/// One evaluated run.
struct Run {
    name: String,
    padding: usize,
    s: usize,
    seed: u64,
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
        let seed = run
            .training_config()
            .unwrap_or_else(|e| fail("training config", &e))
            .seed;
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
            seed,
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

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: poisson_matrix <out.md> <run_dir> ...";
    let out = PathBuf::from(args.next().expect(usage));
    let runs: Vec<Run> = args.map(|d| Run::load(Path::new(&d))).collect();
    assert!(!runs.is_empty(), "{usage}");

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
