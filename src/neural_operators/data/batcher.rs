//! Batching for operator-learning data: stacks individual [`DataItem`]s into
//! rank-`R`/`R-1` tensors for training.

use crate::neural_operators::data::dataitem::{DataItem, HostFloat};
use burn::{Tensor, data::dataloader::batcher::Batcher, prelude::*};

/// A batch of stacked input/target tensor pairs.
///
/// `R` = input tensor rank (batch + spatial dims + channel axis).
/// `RM1` = target tensor rank, constrained to `R - 1` (targets have no
/// channel axis) - enforced at compile time by [`OperatorBatcher::new`].
#[derive(Clone, Debug)]
pub struct Batch<const R: usize, const RM1: usize> {
    pub inputs: Tensor<R>,
    pub targets: Tensor<RM1>,
}

/// Stacks [`DataItem`]s into a [`Batch`] for a fixed input/target rank pair.
///
/// Batches are placed on whatever device the DataLoader supplies to
/// [`Batcher::batch`]. In a hand-written loop, choose it with
/// `DataLoaderBuilder::set_device`; Burn's `Learner` overrides it via
/// `DataLoader::to_device` (autodiff device for training, inner device for
/// validation).
///
/// Zero-sized; the private field forces construction through
/// [`OperatorBatcher::new`] so the rank guard always runs.
pub struct OperatorBatcher<const R: usize, const RM1: usize>(());

impl<const R: usize, const RM1: usize> OperatorBatcher<R, RM1> {
    /// Compile-time guard: evaluated in `new`, so a wrong rank pairing fails
    /// to build instead of panicking at runtime.
    const RANK_OK: () = assert!(
        RM1 + 1 == R,
        "targets rank must be inputs rank minus one (no channel dim)"
    );

    /// Constructs a batcher.
    ///
    /// The batcher does not choose a device. If the DataLoader is built
    /// without `DataLoaderBuilder::set_device`, batches land on
    /// `Device::default()`, which is **not** an autodiff device, so training
    /// on them fails or produces no gradients. Pass
    /// `device.clone().autodiff()` for training loaders.
    ///
    /// `RM1` must be `R - 1`, the only rank pairing this batcher supports:
    ///
    /// ```
    /// use sciml_rs::neural_operators::data::batcher::OperatorBatcher;
    ///
    /// let _batcher = OperatorBatcher::<3, 2>::new();
    /// ```
    ///
    /// Any other pairing is rejected at compile time:
    ///
    /// ```compile_fail,E0080
    /// use sciml_rs::neural_operators::data::batcher::OperatorBatcher;
    ///
    /// let _batcher = OperatorBatcher::<3, 3>::new();
    /// ```
    #[allow(clippy::new_without_default)] // a derived `Default` would skip `RANK_OK`
    pub fn new() -> Self {
        let () = Self::RANK_OK;
        Self(())
    }
}

impl<const R: usize, const RM1: usize, T: HostFloat> Batcher<DataItem<T>, Batch<R, RM1>>
    for OperatorBatcher<R, RM1>
{
    /// Places the batch on `device`, as supplied by the DataLoader (set with
    /// `DataLoaderBuilder::set_device`, or by the `Learner` via `to_device`),
    /// converting from the host dtype `T` to the device's default float dtype.
    fn batch(&self, items: Vec<DataItem<T>>, device: &Device) -> Batch<R, RM1> {
        let n = items.len();

        assert!(!items.is_empty(), "cannot construct an empty batch");

        let input_item_shape = items[0].input.shape();

        let input_shape: [usize; R] =
            core::array::from_fn(|i| if i == 0 { n } else { input_item_shape[i - 1] });

        let target_item_shape = items[0].target.shape();

        let target_shape: [usize; RM1] =
            core::array::from_fn(|i| if i == 0 { n } else { target_item_shape[i - 1] });

        let input_item_len = items[0].input.len();
        let target_item_len = items[0].target.len();

        let mut input_data = Vec::with_capacity(n * input_item_len);

        let mut target_data = Vec::with_capacity(n * target_item_len);

        // Fill both buffers in a single traversal of the batch.
        for item in &items {
            input_data.extend(item.input.iter().copied());
            target_data.extend(item.target.iter().copied());
        }

        let inputs = Tensor::<R>::from_data(
            burn::tensor::TensorData::new(input_data, input_shape),
            device,
        );

        let targets = Tensor::<RM1>::from_data(
            burn::tensor::TensorData::new(target_data, target_shape),
            device,
        );

        Batch { inputs, targets }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn stacks_items_in_order_with_batch_dim_prepended() {
        let device = Device::default();
        let batcher = OperatorBatcher::<3, 2>::new();

        // Two items: input [4, 2], target [4]. Values encode their origin.
        let items: Vec<DataItem<f64>> = (0..2)
            .map(|k| DataItem {
                input: ArrayD::from_shape_fn(IxDyn(&[4, 2]), |i| {
                    (k * 100 + i[0] * 10 + i[1]) as f64
                }),
                target: ArrayD::from_shape_fn(IxDyn(&[4]), |i| (k * 100 + i[0]) as f64),
            })
            .collect();

        let batch = batcher.batch(items, &device);

        assert_eq!(batch.inputs.dims(), [2, 4, 2]);
        assert_eq!(batch.targets.dims(), [2, 4]);

        let inputs: Vec<f64> = batch.inputs.into_data().iter::<f64>().collect();
        assert_eq!(inputs[0], 0.0); // item 0, [0,0]
        assert_eq!(inputs[8], 100.0); // item 1, [0,0] — 4*2 elements per item
    }

    #[test]
    fn batch_lands_on_the_supplied_device() {
        let batcher = OperatorBatcher::<3, 2>::new();
        let items = || {
            vec![DataItem {
                input: ArrayD::<f64>::zeros(IxDyn(&[4, 2])),
                target: ArrayD::zeros(IxDyn(&[4])),
            }]
        };

        let ad = batcher.batch(items(), &Device::default().autodiff());
        assert!(ad.inputs.is_autodiff());
        assert!(ad.targets.is_autodiff());

        let inner = batcher.batch(items(), &Device::default().autodiff().inner());
        assert!(!inner.inputs.is_autodiff());
        assert!(!inner.targets.is_autodiff());
    }

    /// REVIEW.md 4.1: storing the host data as f32 (one cast at load time)
    /// must give the same device tensors as f64 host data converted per
    /// batch. Values are not exactly representable in f32, so a double
    /// rounding or a truncating cast would change some of them.
    #[test]
    fn f32_host_items_upload_bit_identical_to_f64() {
        let device = Device::default();
        let batcher = OperatorBatcher::<3, 2>::new();

        let items_f64: Vec<DataItem<f64>> = (0..3)
            .map(|k| DataItem {
                input: ArrayD::from_shape_fn(IxDyn(&[5, 2]), |i| {
                    ((k * 10 + i[0] * 2 + i[1]) as f64 * 0.731).sin() / 3.0
                }),
                target: ArrayD::from_shape_fn(IxDyn(&[5]), |i| 1e-3 + (k + i[0]) as f64 / 7.0),
            })
            .collect();
        let items_f32: Vec<DataItem<f32>> = items_f64
            .iter()
            .map(|it| DataItem {
                input: it.input.mapv(f32::from_f64),
                target: it.target.mapv(f32::from_f64),
            })
            .collect();

        let a = batcher.batch(items_f64, &device);
        let b = batcher.batch(items_f32, &device);

        // The device default is f32; this is what makes the claim hold.
        assert_eq!(a.inputs.dtype(), burn::tensor::DType::F32);
        assert_eq!(b.inputs.dtype(), burn::tensor::DType::F32);

        let bits = |t: TensorData| t.iter::<f32>().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(bits(a.inputs.into_data()), bits(b.inputs.into_data()));
        assert_eq!(bits(a.targets.into_data()), bits(b.targets.into_data()));
    }
}
