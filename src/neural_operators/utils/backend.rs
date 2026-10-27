//! Facts about this build, recorded with runs and datasets: the backends compiled in and
//! the git commit.

use std::process::Command;

/// The backend cargo features compiled into this build, in `Device::default()`'s order of
/// preference. Metal runs through wgpu, so a device's debug name alone does not say which
/// backend it is; recorded together with it, the two do.
pub fn backend_features() -> Vec<&'static str> {
    [
        ("cuda", cfg!(feature = "cuda")),
        ("metal", cfg!(feature = "metal")),
        ("wgpu", cfg!(feature = "wgpu")),
        ("flex", cfg!(feature = "flex")),
    ]
    .into_iter()
    .filter(|&(_, on)| on)
    .map(|(name, _)| name)
    .collect()
}

/// `git rev-parse HEAD` in this crate's checkout, or `"unknown"` if git is unavailable or
/// this is not a checkout.
pub fn git_commit() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map_or_else(|| "unknown".to_string(), |s| s.trim().to_string())
}
