//! Generic template for FNO for n dimensions

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Linear, LinearConfig,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{Device, activation::relu},
};

use crate::neural_operators::layers::spectral_convolution::SpectralConv;

// FNO Architecture
#[derive(Config, Debug)]
pub struct FNOConfig {
    pub modes: Vec<usize>, // length D = R - 2
    #[config(default = 32)]
    pub hidden_channels: usize,
    pub data_channels: usize, // problem specific, 1 (coefficients), 10 (3d problem, stacked timesteps)
    pub out_channels: usize,
    #[config(default = 4)]
    pub n_layers: usize,
}

#[derive(Module, Debug)]
pub struct FNO<const R: usize> {
    fc0: Linear,
    conv: Vec<SpectralConv<R>>,
    w: Vec<Conv1d>,
    fc1: Linear,
    fc2: Linear,
}

impl FNOConfig {
    pub fn init<const R: usize>(&self, device: &Device) -> FNO<R> {
        assert_eq!(
            self.modes.len() + 2,
            R,
            "FNO<{R}> needs {} modes, config has {}",
            R - 2,
            self.modes.len()
        );
        assert_eq!(
            self.out_channels, 1,
            "only scalar-output operators are supported (see flatten_pair)"
        );

        let coord_channels = self.modes.len(); // D = R - 2, need to derive and input into layer

        FNO {
            fc0: LinearConfig::new(self.data_channels + coord_channels, self.hidden_channels)
                .init(device),

            conv: (0..self.n_layers)
                .map(|_| {
                    SpectralConv::<R>::new(
                        device,
                        self.hidden_channels,
                        self.hidden_channels,
                        &self.modes,
                    )
                })
                .collect(),

            w: (0..self.n_layers)
                .map(|_| {
                    Conv1dConfig::new(self.hidden_channels, self.hidden_channels, 1).init(device)
                })
                .collect(),

            fc1: LinearConfig::new(self.hidden_channels, 128).init(device),
            fc2: LinearConfig::new(128, self.out_channels).init(device),
        }
    }
}

impl<const R: usize> FNO<R> {
    fn apply_pointwise(conv: &Conv1d, x: Tensor<R>) -> Tensor<R> {
        let dims = x.dims();
        let (b, hidden_channels) = (dims[0], dims[1]);
        let spatial: usize = dims[2..].iter().product();
        conv.forward(x.reshape([b, hidden_channels, spatial]))
            .reshape(dims)
    }

    pub fn forward(&self, x: Tensor<R>) -> Tensor<R> {
        let x = self.fc0.forward(x);

        let perm_in: [usize; R] = core::array::from_fn(|i| match i {
            0 => 0,
            1 => R - 1,
            i => i - 1,
        });
        let mut x = x.permute(perm_in);

        let n = self.conv.len();
        for idx in 0..n {
            let x1 = self.conv[idx].forward(x.clone());
            let x2 = Self::apply_pointwise(&self.w[idx], x);
            x = if idx == n - 1 { x1 + x2 } else { relu(x1 + x2) };
        }

        let perm_out: [usize; R] = core::array::from_fn(|i| match i {
            0 => 0,
            i if i == R - 1 => 1,
            i => i + 1,
        });
        let x = x.permute(perm_out);
        let x = self.fc1.forward(x);
        let x = relu(x);
        self.fc2.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    fn fno_forward_shape_check<const R: usize>(
        modes: Vec<usize>,
        hidden_channels: usize,
        data_channels: usize,
        spatial: &[usize],
    ) {
        let device = Device::default();
        let coord_channels = modes.len();
        assert_eq!(spatial.len(), R - 2, "spatial dims must match modes count");

        let config = FNOConfig {
            modes,
            hidden_channels,
            data_channels,
            out_channels: 1,
            n_layers: 4,
        };
        let model: FNO<R> = config.init::<R>(&device);

        // Channels-last on input, as fc0 expects.
        let mut x_shape = vec![1];
        x_shape.extend_from_slice(spatial);
        x_shape.push(data_channels + coord_channels);
        let x_shape: [usize; R] = x_shape.try_into().unwrap();

        let out = model.forward(Tensor::<R>::random(x_shape, Distribution::Default, &device));

        let mut expected = vec![1];
        expected.extend_from_slice(spatial);
        expected.push(1);
        let expected: [usize; R] = expected.try_into().unwrap();

        assert_eq!(
            out.dims(),
            expected,
            "expected [batch, spatial.., out_channels]"
        );
    }

    #[test]
    fn forward_shape_1d() {
        fno_forward_shape_check::<3>(vec![16], 32, 1, &[64]);
    }

    #[test]
    fn forward_shape_2d() {
        fno_forward_shape_check::<4>(vec![2, 2], 4, 1, &[4, 4]);
    }

    #[test]
    fn forward_shape_3d() {
        // 10 stacked timesteps + x + y + t channels.
        fno_forward_shape_check::<5>(vec![2, 2, 2], 4, 10, &[4, 4, 4]);
    }

    #[test]
    #[should_panic(expected = "needs 2 modes")]
    fn init_rejects_rank_mismatch() {
        let device = Device::default();
        let config = FNOConfig {
            modes: vec![16], // 1 entry → R must be 3
            hidden_channels: 32,
            data_channels: 1,
            out_channels: 1,
            n_layers: 4,
        };
        let _: FNO<4> = config.init::<4>(&device); // asking for rank 4
    }

    // --- REVIEW.md 2.7: inference without autodiff ---

    #[test]
    fn autodiff_trained_weights_give_identical_inference_on_plain_device() {
        // Mirrors train -> save -> predict: weights from a model built on the
        // autodiff device, loaded into one on the plain device (what the
        // predict examples now use).
        use burn::store::{BurnpackStore, ModuleSnapshot};
        use burn::tensor::TensorData;

        let ad = Device::default().autodiff();
        let plain = Device::default();
        let config = FNOConfig {
            modes: vec![4],
            hidden_channels: 8,
            data_channels: 1,
            out_channels: 1,
            n_layers: 2,
        };

        let trained: FNO<3> = config.init::<3>(&ad);
        let mut save = BurnpackStore::from_bytes(None);
        trained.save_into(&mut save).expect("save weights");
        let bytes = save.get_bytes().expect("serialise weights");

        let mut loaded: FNO<3> = config.init::<3>(&plain);
        loaded
            .load_from(&mut BurnpackStore::from_bytes(Some(bytes)))
            .expect("load weights");

        // [batch, s, data + coord channels], channels-last.
        let shape = vec![2, 16, 2];
        let vals: Vec<f32> = (0..64).map(|i| (i as f32 * 0.29).sin()).collect();
        let x_ad = Tensor::<3>::from_data(TensorData::new(vals.clone(), shape.clone()), &ad);
        let x_plain = Tensor::<3>::from_data(TensorData::new(vals, shape), &plain);

        let out_ad = trained.forward(x_ad);
        let out_plain = loaded.forward(x_plain);

        // The old example path tracked a backward graph; the new one doesn't.
        assert!(out_ad.is_autodiff());
        assert!(!out_plain.is_autodiff());

        // Same backend kernels either way, so the values must match exactly.
        assert_eq!(
            out_ad.inner().into_data().try_to_vec::<f32>().unwrap(),
            out_plain.into_data().try_to_vec::<f32>().unwrap()
        );
    }
}
