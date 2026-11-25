//! 3D CL0.1 (`docs/laplace3d/tasks/CL0.1.md`): does the npz path hold a 3D n = 65 train
//! split above 4 GiB? Writes f and u `[1000, 65, 65, 65]` f64 (4.39 GB) the way
//! `pde::poisson::dataset::write_split` does (`NpzWriter::new`, one `add_array` per field),
//! reads them back with the crate's `NpzFileReader`, and checks every value bit for bit.
//!
//! Usage: `cargo run --release -p fno3d-cost-spike --bin zip64_roundtrip -- <file.npz>`;
//! the file is removed on success.

use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use ndarray::{Array4, ArrayD};
use ndarray_npy::NpzWriter;
use sciml_rs::neural_operators::data::io::readers::npz::NpzFileReader;

const SHAPE: (usize, usize, usize, usize) = (1000, 65, 65, 65);

/// `sign × i` at flat index `i`: exact in f64, so a misplaced or truncated value shows.
fn field(sign: f64) -> Array4<f64> {
    let len = SHAPE.0 * SHAPE.1 * SHAPE.2 * SHAPE.3;
    Array4::from_shape_vec(SHAPE, (0..len).map(|i| sign * i as f64).collect()).unwrap()
}

fn check(name: &str, got: &ArrayD<f64>, sign: f64) {
    assert_eq!(
        got.shape(),
        [SHAPE.0, SHAPE.1, SHAPE.2, SHAPE.3],
        "{name} shape"
    );
    let bad = got
        .iter()
        .enumerate()
        .filter(|&(i, &v)| v.to_bits() != (sign * i as f64).to_bits())
        .count();
    assert_eq!(bad, 0, "{name}: {bad} values differ");
}

fn main() {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: zip64_roundtrip <file.npz>"),
    );
    let t = Instant::now();
    {
        let mut npz = NpzWriter::new(File::create(&path).unwrap());
        for (name, sign) in [("f", 1.0), ("u", -1.0)] {
            npz.add_array(name, &field(sign))
                .unwrap_or_else(|e| panic!("writing {name}: {e}"));
        }
        npz.finish().expect("finishing the archive");
    }
    let bytes = std::fs::metadata(&path).unwrap().len();
    println!(
        "wrote {bytes} bytes ({:.3} GiB, above 4 GiB: {}) in {:.1} s",
        bytes as f64 / (1u64 << 30) as f64,
        bytes > 1 << 32,
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    let mut reader = NpzFileReader::new(&path).expect("opening the archive");
    for (name, sign) in [("f", 1.0), ("u", -1.0)] {
        let got = reader
            .read_field(name)
            .unwrap_or_else(|e| panic!("reading {name}: {e}"));
        check(name, &got, sign);
    }
    println!(
        "read and checked f and u bit for bit in {:.1} s",
        t.elapsed().as_secs_f64()
    );
    std::fs::remove_file(&path).unwrap();
}
