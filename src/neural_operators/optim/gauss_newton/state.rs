//! [`GaussNewtonState`]: everything a [`GaussNewton`](super::GaussNewton) carries between
//! steps, and its bytes for a checkpoint.
//!
//! Little-endian: the magic `SCIMLGN1`; λ (f64) and the step count (u64); the warm start
//! (a vector); the diagonal estimate (u64 step, then a vector). A vector is a presence
//! byte (0 absent, 1 present), then the dtype (0 f32, 1 f64), the number of parameters
//! (u64) and per parameter its `ParamId` (u64), its length (u64) and its values as f64,
//! which round-trips f32 bit for bit. Read through [`LeReader`], so a short or overlong
//! input is an error, never a panic.

use std::io;

use burn::{
    module::ParamId,
    prelude::*,
    tensor::{DType, TensorData},
};

use super::param_vec::ParamVec;
use crate::neural_operators::utils::bytes::{LeReader, invalid};

const MAGIC: &[u8; 8] = b"SCIMLGN1";

/// The state of a [`GaussNewton`](super::GaussNewton) between steps: the optimiser owns one
/// and checkpoints it as it is ([`GaussNewton::state`](super::GaussNewton::state),
/// [`GaussNewton::from_state`](super::GaussNewton::from_state)).
#[derive(Clone, Debug)]
pub struct GaussNewtonState {
    /// The damping the next step will use.
    pub lambda: f64,
    /// Steps taken.
    pub step: u64,
    /// The previous CG solution (undecayed), if any.
    pub warm_start: Option<ParamVec>,
    /// The current diag(G) estimate and the step it was made at, if any.
    pub diagonal: Option<(u64, ParamVec)>,
}

impl GaussNewtonState {
    /// The state's bytes (module docs for the layout), for a checkpoint.
    ///
    /// # Panics
    /// If a vector is neither f32 nor f64.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&self.lambda.to_le_bytes());
        out.extend_from_slice(&self.step.to_le_bytes());
        write_vector(&mut out, self.warm_start.as_ref());
        match &self.diagonal {
            Some((at, d)) => {
                out.extend_from_slice(&at.to_le_bytes());
                write_vector(&mut out, Some(d));
            }
            None => {
                out.extend_from_slice(&0u64.to_le_bytes());
                write_vector(&mut out, None);
            }
        }
        out
    }

    /// Decodes [`to_bytes`](Self::to_bytes), putting the vectors on `device`.
    ///
    /// # Errors
    /// [`io::ErrorKind::InvalidData`] for a wrong magic, an unknown tag, a truncated input
    /// or trailing bytes.
    pub fn from_bytes(bytes: &[u8], device: &Device) -> io::Result<Self> {
        let mut r = LeReader::new(bytes, "Gauss-Newton state");
        if r.take(8)? != MAGIC {
            return Err(invalid("not a Gauss-Newton state"));
        }
        let lambda = r.f64()?;
        let step = r.u64()?;
        let warm_start = read_vector(&mut r, device)?;
        let at = r.u64()?;
        let diagonal = read_vector(&mut r, device)?.map(|d| (at, d));
        r.finish()?;
        Ok(Self {
            lambda,
            step,
            warm_start,
            diagonal,
        })
    }
}

fn write_vector(out: &mut Vec<u8>, v: Option<&ParamVec>) {
    let Some(v) = v else {
        out.push(0);
        return;
    };
    out.push(1);
    out.push(match v.dtype() {
        Some(DType::F64) => 1,
        Some(DType::F32) | None => 0,
        Some(other) => panic!("GaussNewtonState: only f32 and f64 vectors, got {other:?}"),
    });
    out.extend_from_slice(&(v.parts().len() as u64).to_le_bytes());
    for (id, t) in v.parts() {
        let values = t
            .clone()
            .into_data()
            .convert::<f64>()
            .try_to_vec::<f64>()
            .expect("a float tensor converts to f64");
        out.extend_from_slice(&id.val().to_le_bytes());
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for x in values {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
}

fn read_vector(r: &mut LeReader, device: &Device) -> io::Result<Option<ParamVec>> {
    match r.u8()? {
        0 => return Ok(None),
        1 => {}
        _ => return Err(invalid("bad presence byte in the Gauss-Newton state")),
    }
    let dtype = match r.u8()? {
        0 => DType::F32,
        1 => DType::F64,
        _ => return Err(invalid("bad dtype in the Gauss-Newton state")),
    };
    let n_parts = r.u64()? as usize;
    let mut parts = Vec::new();
    for _ in 0..n_parts {
        let id = ParamId::from(r.u64()?);
        let len = r.u64()? as usize;
        let data = TensorData::new(r.f64s(len)?, [len]).convert_dtype(dtype);
        parts.push((id, Tensor::<1>::from_data(data, device)));
    }
    Ok(Some(ParamVec::from_parts(parts)))
}
