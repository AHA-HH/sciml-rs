//! `checkpoint.bin`: the state a training run resumes from (Phase 9 T2).
//!
//! One file, little-endian: the magic `SCIMLCK1`, then `next_epoch`, `resumes` (u64) and
//! `training_time_s` (f64), then three length-prefixed (u64) Burnpack blobs: the model
//! weights, the optimiser record and the scheduler record. A short or overlong file is an
//! error, never a panic.

use std::error::Error;
use std::io;

use burn::{
    lr_scheduler::LrSchedulerRecord,
    optim::OptimizerRecord,
    store::{BurnpackStore, ModuleSnapshot},
    tensor::{Bytes, Device},
};

use crate::neural_operators::models::fno::{FNO, FNOConfig};

const MAGIC: &[u8; 8] = b"SCIMLCK1";

/// Where a run stands at a checkpoint: everything but the tensors.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CheckpointState {
    /// The first epoch still to train (zero-based): the epochs before it are done.
    pub next_epoch: usize,
    /// How many times the run has been resumed; 0 for a run trained in one session.
    pub resumes: usize,
    /// Wall-clock seconds of training up to the checkpoint, summed over sessions. Work
    /// done after a checkpoint and lost to an interruption is not counted.
    pub training_time_s: f64,
}

/// A checkpoint read back by [`RunDir::checkpoint`](super::RunDir::checkpoint): the state,
/// the model, and the optimiser and scheduler records to load into freshly built ones
/// (`ModuleOptimizer::load_record`, `ModuleLrScheduler::load_record`).
pub struct Checkpoint<const R: usize> {
    /// Where the run stands.
    pub state: CheckpointState,
    /// The model after epoch `state.next_epoch - 1`, on the device it was read onto.
    pub model: FNO<R>,
    /// Adam's per-parameter moments and step counts, keyed by the model's `ParamId`s.
    pub optim: OptimizerRecord,
    /// The cosine schedule's position (`current_iter`).
    pub scheduler: LrSchedulerRecord,
}

/// Encodes a checkpoint as the bytes of `checkpoint.bin`.
pub(super) fn encode<const R: usize>(
    state: &CheckpointState,
    model: &FNO<R>,
    optim: OptimizerRecord,
    scheduler: LrSchedulerRecord,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut store = BurnpackStore::from_bytes(None);
    model.save_into(&mut store)?;
    let blobs = [
        store.get_bytes()?,
        optim.into_bytes()?,
        scheduler.into_bytes()?,
    ];

    let mut out = MAGIC.to_vec();
    out.extend_from_slice(&(state.next_epoch as u64).to_le_bytes());
    out.extend_from_slice(&(state.resumes as u64).to_le_bytes());
    out.extend_from_slice(&state.training_time_s.to_le_bytes());
    for blob in &blobs {
        out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
        out.extend_from_slice(blob);
    }
    Ok(out)
}

/// Decodes `checkpoint.bin`, building the model from `model_cfg` on `device`.
///
/// # Errors
/// [`io::ErrorKind::InvalidData`] for a wrong magic, a truncated file or trailing bytes;
/// Burn's errors for a blob that doesn't parse or doesn't fit the architecture.
pub(super) fn decode<const R: usize>(
    bytes: &[u8],
    model_cfg: &FNOConfig,
    device: &Device,
) -> Result<Checkpoint<R>, Box<dyn Error>> {
    let mut rest = bytes;
    if take(&mut rest, 8)? != MAGIC {
        return Err(invalid("not a checkpoint file").into());
    }
    let state = CheckpointState {
        next_epoch: read_u64(&mut rest)? as usize,
        resumes: read_u64(&mut rest)? as usize,
        training_time_s: f64::from_bits(read_u64(&mut rest)?),
    };
    let mut blob = || -> io::Result<Bytes> {
        let len = read_u64(&mut rest)? as usize;
        Ok(Bytes::from_bytes_vec(take(&mut rest, len)?.to_vec()))
    };
    let (weights, optim, scheduler) = (blob()?, blob()?, blob()?);
    if !rest.is_empty() {
        return Err(invalid("trailing bytes after the checkpoint").into());
    }

    let mut model = model_cfg.init::<R>(device);
    model.load_from(&mut BurnpackStore::from_bytes(Some(weights)))?;
    Ok(Checkpoint {
        state,
        model,
        optim: OptimizerRecord::from_bytes(optim)?,
        scheduler: LrSchedulerRecord::from_bytes(scheduler)?,
    })
}

fn invalid(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what)
}

/// The next `n` bytes of `rest`, advancing it.
fn take<'a>(rest: &mut &'a [u8], n: usize) -> io::Result<&'a [u8]> {
    if rest.len() < n {
        return Err(invalid("checkpoint is truncated"));
    }
    let (head, tail) = rest.split_at(n);
    *rest = tail;
    Ok(head)
}

fn read_u64(rest: &mut &[u8]) -> io::Result<u64> {
    let bytes = take(rest, 8)?;
    Ok(u64::from_le_bytes(bytes.try_into().expect("8 bytes")))
}
