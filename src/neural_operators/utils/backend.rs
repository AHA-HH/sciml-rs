//! The backends compiled into this build.

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
