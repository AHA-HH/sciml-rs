//! Device-resident batching: each split is uploaded once and batches are
//! gathered on the device, so a training step does no host-side copying.
//!
//! Burn's own `DataLoader` still drives the iteration, over an
//! [`IndexDataset`] of sample indices. The shuffle (seeded `StdRng`, forked
//! per epoch), the partial last batch and `to_device` / `slice` (used by the
//! `Learner`) are therefore Burn's code, and the batch order matches a loader
//! built over the [`OperatorDataset`] itself with the same settings.

use crate::neural_operators::data::{
    batcher::Batch, dataitem::HostFloat, dataset::OperatorDataset,
};
use burn::{
    Tensor,
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::{Dataset, DatasetError},
    },
    prelude::*,
};
use ndarray::ArrayD;
use std::{
    io::{Error, ErrorKind},
    sync::Arc,
};

/// The indices `0..len`, as a dataset. Item `i` is `i`.
pub struct IndexDataset(pub usize);

impl Dataset<usize> for IndexDataset {
    fn get(&self, index: usize) -> Result<usize, DatasetError> {
        if index < self.0 {
            Ok(index)
        } else {
            Err(DatasetError::new(Error::new(
                ErrorKind::InvalidInput,
                format!("index {index} out of bounds (len {})", self.0),
            )))
        }
    }

    fn len(&self) -> usize {
        self.0
    }
}

/// Holds a whole split on the device and builds each [`Batch`] by selecting
/// rows with a small index tensor.
///
/// The split is stored without autodiff. Batches follow the device the
/// DataLoader supplies: moved there if it is different hardware, and given an
/// autodiff association when it is an autodiff device. The stored tensors
/// never require gradients, so no gradient flows back into the data.
pub struct DeviceBatcher<const R: usize, const RM1: usize> {
    inputs: Tensor<R>,
    targets: Tensor<RM1>,
}

impl<const R: usize, const RM1: usize> DeviceBatcher<R, RM1> {
    const RANK_OK: () = assert!(
        RM1 + 1 == R,
        "targets rank must be inputs rank minus one (no channel dim)"
    );

    /// Uploads `data` to `device` (its non-autodiff form) in one transfer per
    /// tensor, converting to the device's default float dtype.
    ///
    /// `RM1` must be `R - 1`:
    ///
    /// ```compile_fail,E0080
    /// use burn::tensor::Device;
    /// use ndarray::{ArrayD, IxDyn};
    /// use sciml_rs::neural_operators::data::{
    ///     dataset::OperatorDataset, device_batcher::DeviceBatcher,
    /// };
    ///
    /// let data = OperatorDataset::<f32>::new(
    ///     ArrayD::zeros(IxDyn(&[1, 4, 2])),
    ///     ArrayD::zeros(IxDyn(&[1, 4])),
    /// );
    /// let _ = DeviceBatcher::<3, 3>::new(&data, &Device::default());
    /// ```
    ///
    /// # Panics
    ///
    /// If the inputs are not rank `R` or the targets not rank `RM1`.
    pub fn new<T: HostFloat>(data: &OperatorDataset<T>, device: &Device) -> Self {
        let () = Self::RANK_OK;
        let device = device.clone().inner();
        Self {
            inputs: upload::<R, T>(data.inputs(), &device, "inputs"),
            targets: upload::<RM1, T>(data.targets(), &device, "targets"),
        }
    }
}

fn upload<const D: usize, T: HostFloat>(a: &ArrayD<T>, device: &Device, name: &str) -> Tensor<D> {
    assert_eq!(
        a.ndim(),
        D,
        "{name} must have rank {D}, got shape {:?}",
        a.shape()
    );
    let values: Vec<T> = a.iter().copied().collect(); // logical (row-major) order
    Tensor::from_data(TensorData::new(values, a.shape().to_vec()), device)
}

impl<const R: usize, const RM1: usize> Batcher<usize, Batch<R, RM1>> for DeviceBatcher<R, RM1> {
    fn batch(&self, items: Vec<usize>, device: &Device) -> Batch<R, RM1> {
        assert!(!items.is_empty(), "cannot construct an empty batch");

        let home = self.inputs.device();
        let idx: Vec<i64> = items.iter().map(|&i| i as i64).collect();
        let n = idx.len();
        let idx = Tensor::<1, Int>::from_data(TensorData::new(idx, [n]), &home);

        Batch {
            inputs: to_batch_device(self.inputs.clone().select(0, idx.clone()), device),
            targets: to_batch_device(self.targets.clone().select(0, idx), device),
        }
    }
}

/// Moves `t` to `device`'s hardware if needed. `to_device` keeps the autodiff
/// association and `Device` equality ignores it, so autodiff is set separately.
fn to_batch_device<const D: usize>(t: Tensor<D>, device: &Device) -> Tensor<D> {
    let t = if t.device() == *device {
        t
    } else {
        t.to_device(device)
    };
    if device.is_autodiff() {
        t.autodiff()
    } else {
        t
    }
}

/// A DataLoader over `data` held on `device`, with the same batching and
/// shuffling as `DataLoaderBuilder` over the [`OperatorDataset`] itself.
///
/// Batches land on `device` (use the autodiff device for training). `shuffle`
/// is the seed, or `None` for sequential order.
pub fn device_loader<const R: usize, const RM1: usize, T: HostFloat>(
    data: &OperatorDataset<T>,
    device: &Device,
    batch_size: usize,
    shuffle: Option<u64>,
) -> Arc<dyn DataLoader<Batch<R, RM1>>> {
    let builder = DataLoaderBuilder::new(DeviceBatcher::<R, RM1>::new(data, device))
        .set_device(device.clone())
        .batch_size(batch_size);
    let builder = match shuffle {
        Some(seed) => builder.shuffle(seed),
        None => builder,
    };
    builder.build(IndexDataset(data.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::data::batcher::OperatorBatcher;
    use ndarray::IxDyn;

    /// `n` samples, inputs `[n, 4, 2]`, targets `[n, 4]`; every value encodes
    /// its sample, so a wrong row or order changes the data.
    fn dataset(n: usize) -> OperatorDataset<f32> {
        let inputs = ArrayD::from_shape_fn(IxDyn(&[n, 4, 2]), |i| {
            (i[0] * 100 + i[1] * 10 + i[2]) as f64 + 0.25
        });
        let targets = ArrayD::from_shape_fn(IxDyn(&[n, 4]), |i| (i[0] * 100 + i[1]) as f64 - 0.5);
        OperatorDataset::from_f64(inputs, targets)
    }

    /// Burn's loader over the `OperatorDataset` itself: the pre-4.1 path.
    fn host_loader(
        n: usize,
        device: &Device,
        batch_size: usize,
        shuffle: Option<u64>,
    ) -> Arc<dyn DataLoader<Batch<3, 2>>> {
        let b = DataLoaderBuilder::new(OperatorBatcher::<3, 2>::new())
            .set_device(device.clone())
            .batch_size(batch_size);
        let b = match shuffle {
            Some(seed) => b.shuffle(seed),
            None => b,
        };
        b.build(dataset(n))
    }

    /// Every batch of `epochs` passes, as raw f32 bits plus dims and autodiff flag.
    type Snapshot = Vec<(Vec<u32>, Vec<u32>, [usize; 3], bool)>;

    fn snapshot(loader: &Arc<dyn DataLoader<Batch<3, 2>>>, epochs: usize) -> Snapshot {
        let bits = |t: TensorData| t.iter::<f32>().map(f32::to_bits).collect::<Vec<_>>();
        let mut out = Vec::new();
        for _ in 0..epochs {
            for batch in loader.iter() {
                let batch = batch.expect("in-range indices");
                let (dims, ad) = (batch.inputs.dims(), batch.inputs.is_autodiff());
                assert_eq!(ad, batch.targets.is_autodiff());
                out.push((
                    bits(batch.inputs.into_data()),
                    bits(batch.targets.into_data()),
                    dims,
                    ad,
                ));
            }
        }
        out
    }

    /// REVIEW.md 4.1: same batches, same order, same device as the host
    /// loader, over several shuffled epochs with a partial last batch.
    #[test]
    fn matches_host_loader_bit_for_bit() {
        let device = Device::default().autodiff();
        let (n, bs) = (11, 3); // 3 full batches + 1 of 2
        for shuffle in [Some(42), Some(7), None] {
            let old = snapshot(&host_loader(n, &device, bs, shuffle), 3);
            let new = snapshot(
                &device_loader::<3, 2, _>(&dataset(n), &device, bs, shuffle),
                3,
            );
            assert_eq!(old.len(), 12);
            assert_eq!(new, old, "shuffle = {shuffle:?}");
        }
    }

    /// Sanity check on the comparison above: epochs really are reshuffled,
    /// and a different seed gives a different order.
    #[test]
    fn epochs_are_reshuffled_and_seed_matters() {
        let device = Device::default();
        let a = snapshot(
            &device_loader::<3, 2, _>(&dataset(11), &device, 3, Some(42)),
            2,
        );
        let b = snapshot(
            &device_loader::<3, 2, _>(&dataset(11), &device, 3, Some(43)),
            2,
        );
        assert_ne!(a[..4], a[4..], "second epoch repeated the first");
        assert_ne!(a, b);
    }

    /// The `Learner` calls `to_device(inner)` for validation and may `slice`;
    /// both must behave as for the host loader.
    #[test]
    fn to_device_and_slice_match_host_loader() {
        let device = Device::default().autodiff();
        let old = host_loader(11, &device, 3, Some(42));
        let new = device_loader::<3, 2, _>(&dataset(11), &device, 3, Some(42));
        assert_eq!(new.num_items(), old.num_items());

        let inner = device.clone().inner();
        let (old_in, new_in) = (old.to_device(&inner), new.to_device(&inner));
        let s = snapshot(&new_in, 2);
        assert!(
            s.iter().all(|b| !b.3),
            "inner loader produced autodiff batches"
        );
        assert_eq!(s, snapshot(&old_in, 2));

        let (old_sl, new_sl) = (old.slice(2, 9), new.slice(2, 9));
        assert_eq!(new_sl.num_items(), 7);
        assert_eq!(snapshot(&new_sl, 2), snapshot(&old_sl, 2));
    }

    /// Training batches are autodiff but untracked: the stored split is a
    /// constant, not a graph leaf.
    #[test]
    fn stored_split_does_not_require_grad() {
        let device = Device::default().autodiff();
        let loader = device_loader::<3, 2, _>(&dataset(4), &device, 2, None);
        let batch = loader.iter().next().unwrap().unwrap();
        assert!(batch.inputs.is_autodiff());
        assert!(!batch.inputs.is_require_grad());
        assert!(!batch.targets.is_require_grad());
    }

    #[test]
    fn index_dataset_rejects_out_of_range() {
        let d = IndexDataset(3);
        assert_eq!(d.get(2).unwrap(), 2);
        assert!(d.get(3).is_err());
    }

    #[test]
    #[should_panic(expected = "inputs must have rank 3")]
    fn wrong_input_rank_is_rejected() {
        let data = OperatorDataset::<f32>::new(
            ArrayD::zeros(IxDyn(&[2, 4])),
            ArrayD::zeros(IxDyn(&[2, 4])),
        );
        let _ = DeviceBatcher::<3, 2>::new(&data, &Device::default());
    }
}
