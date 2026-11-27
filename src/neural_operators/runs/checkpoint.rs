//! `checkpoint.bin`: the state a training run resumes from (Phase 9 T1).
//!
//! One file, little-endian: the magic `SCIMLCK1`, then `next_epoch`, `resumes` (u64) and
//! `training_time_s` (f64), then three length-prefixed (u64) Burnpack blobs: the model
//! weights, the optimiser record and the scheduler record. A Gauss–Newton run (Phase 10
//! T3) appends a fourth length-prefixed blob, its optimiser state
//! ([`GaussNewtonState::to_bytes`]); an Adam checkpoint has none, so files written before
//! T3 decode unchanged. An L-BFGS run (optimisers CO2.1) writes that fourth blob empty and a
//! fifth, its optimiser record (`LBFGS::to_record`'s Burnpack bytes); a Gauss–Newton state
//! is never empty, so the two stay apart. A short or overlong file is an error, never a
//! panic.

use std::error::Error;
use std::io;

use burn::{
    lr_scheduler::LrSchedulerRecord,
    module::Module,
    optim::{LBFGS, OptimizerRecord},
    store::{BurnpackStore, ModuleSnapshot},
    tensor::{Bytes, Device},
};

use crate::neural_operators::{
    models::{fno::FNO, operator::Operator},
    optim::gauss_newton::GaussNewtonState,
};

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

/// An FNO's checkpoint, read back by [`RunDir::checkpoint`](super::RunDir::checkpoint).
pub type Checkpoint<const R: usize> = OperatorCheckpoint<FNO<R>>;

/// A checkpoint of a run of the operator `M`, read back by
/// [`RunDir::checkpoint_of`](super::RunDir::checkpoint_of): the state, the model, and the
/// optimiser and scheduler records to load into freshly built ones
/// (`ModuleOptimizer::load_record`, `ModuleLrScheduler::load_record`).
pub struct OperatorCheckpoint<M> {
    /// Where the run stands.
    pub state: CheckpointState,
    /// The model after epoch `state.next_epoch - 1`, on the device it was read onto.
    pub model: M,
    /// Adam's per-parameter moments and step counts, keyed by the model's `ParamId`s.
    pub optim: OptimizerRecord,
    /// The cosine schedule's position (`current_iter`).
    pub scheduler: LrSchedulerRecord,
    /// The Gauss–Newton state of a Gauss–Newton run, `None` for an Adam run.
    pub gn: Option<GaussNewtonState>,
    /// The L-BFGS record of an L-BFGS run (`LBFGS::load_record` takes it), else `None`.
    pub lbfgs: Option<OptimizerRecord>,
}

/// Encodes a checkpoint as the bytes of `checkpoint.bin`.
pub(super) fn encode(
    state: &CheckpointState,
    model: &impl Module,
    optim: OptimizerRecord,
    scheduler: LrSchedulerRecord,
    gn: Option<&GaussNewtonState>,
    lbfgs: Option<&LBFGS>,
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
    let gn = gn.map(GaussNewtonState::to_bytes);
    let lbfgs = lbfgs.map(LBFGS::into_bytes).transpose()?;
    // an L-BFGS run's fourth (Gauss-Newton) blob is empty
    let gn = gn.or_else(|| lbfgs.as_ref().map(|_| Vec::new()));
    let blobs = blobs
        .iter()
        .map(|b| &b[..])
        .chain(gn.as_deref())
        .chain(lbfgs.as_deref());
    for blob in blobs {
        out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
        out.extend_from_slice(blob);
    }
    Ok(out)
}

/// Decodes `checkpoint.bin`, building the model from `model_cfg` on `device` (`M::init`,
/// whose weights the checkpoint's then replace).
///
/// # Errors
/// [`io::ErrorKind::InvalidData`] for a wrong magic, a truncated file or trailing bytes;
/// Burn's errors for a blob that doesn't parse or doesn't fit the architecture.
pub(super) fn decode<M: Operator>(
    bytes: &[u8],
    model_cfg: &M::Config,
    device: &Device,
) -> Result<OperatorCheckpoint<M>, Box<dyn Error>> {
    let mut rest = bytes;
    if take(&mut rest, 8)? != MAGIC {
        return Err(invalid("not a checkpoint file").into());
    }
    let state = CheckpointState {
        next_epoch: read_u64(&mut rest)? as usize,
        resumes: read_u64(&mut rest)? as usize,
        training_time_s: f64::from_bits(read_u64(&mut rest)?),
    };
    let blob = |rest: &mut &[u8]| -> io::Result<Bytes> {
        let len = read_u64(rest)? as usize;
        Ok(Bytes::from_bytes_vec(take(rest, len)?.to_vec()))
    };
    let (weights, optim, scheduler) = (blob(&mut rest)?, blob(&mut rest)?, blob(&mut rest)?);
    let (mut gn, mut lbfgs) = (None, None);
    if !rest.is_empty() {
        let len = read_u64(&mut rest)? as usize;
        let state = take(&mut rest, len)?;
        if len == 0 {
            lbfgs = Some(OptimizerRecord::from_bytes(blob(&mut rest)?)?);
        } else {
            gn = Some(GaussNewtonState::from_bytes(state, device)?);
        }
        if !rest.is_empty() {
            return Err(invalid("trailing bytes after the checkpoint").into());
        }
    }

    let mut model = M::init(model_cfg, device);
    model.load_from(&mut BurnpackStore::from_bytes(Some(weights)))?;
    Ok(OperatorCheckpoint {
        state,
        model,
        optim: OptimizerRecord::from_bytes(optim)?,
        scheduler: LrSchedulerRecord::from_bytes(scheduler)?,
        gn,
        lbfgs,
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
