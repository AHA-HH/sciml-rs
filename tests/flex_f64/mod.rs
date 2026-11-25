//! Flex configured for f64, shared by the f64 test binaries (`gauss_newton_f64`,
//! `parity_*_f64`). A device's float dtype can be configured only once per process, so each
//! such binary is its own process; include this module with `#[cfg(feature = "flex")]`.

use std::sync::OnceLock;

use burn::{prelude::*, tensor::FloatDType};

/// Flex configured for f64, once per process.
pub fn flex_f64() -> Device {
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
