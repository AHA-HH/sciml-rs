//! Shared data-pipeline type used by both `OperatorDataset` and `OperatorBatcher`.

use ndarray::ArrayD;

/// A single training example: one input/target pair seen by the model during training.
#[derive(Clone, Debug)]
pub struct DataItem {
    pub input: ArrayD<f64>, 
    pub target: ArrayD<f64>, 
}