//! Library CM1.6a in f64: `parity_losses_common` on the f64 fixtures (D3: 1e-12). Its own
//! test binary because it configures Flex's process-wide float dtype to f64. No FFT is
//! involved, so unlike the FNO's f64 parity these are not held back by Flex's f32 twiddles.

// Each test binary uses part of the shared harness.
#[allow(dead_code)]
mod parity_common;
// The f32 binary uses `tensor` too.
#[allow(dead_code)]
mod parity_losses_common;

#[cfg(feature = "flex")]
use std::sync::OnceLock;

#[cfg(feature = "flex")]
use burn::{prelude::*, tensor::FloatDType};
#[cfg(feature = "flex")]
use parity_losses_common::run;

/// Flex configured for f64, once per process.
#[cfg(feature = "flex")]
fn flex_f64() -> Device {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            let mut device = Device::flex();
            device
                .configure(FloatDType::F64)
                .expect("flex accepts f64 as its default float dtype");
            device
        })
        .clone()
}

#[cfg(feature = "flex")]
#[test]
fn losses1d_f64_match_2_0_0() {
    run::<1, 3, 4, 2>("losses1d_f64", &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn losses2d_f64_match_2_0_0() {
    run::<2, 4, 5, 3>("losses2d_f64", &flex_f64());
}

#[cfg(feature = "flex")]
#[test]
fn losses3d_f64_match_2_0_0() {
    run::<3, 5, 6, 4>("losses3d_f64", &flex_f64());
}
