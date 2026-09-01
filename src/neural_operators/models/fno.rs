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
    pub width: usize,
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
        let coord_channels = self.modes.len(); // D = R - 2, need to derive and input into layer

        FNO {
            fc0: LinearConfig::new(self.data_channels + coord_channels, self.width).init(device),

            conv: (0..self.n_layers)
                .map(|_| SpectralConv::<R>::new(device, self.width, self.width, &self.modes))
                .collect(),

            w: (0..self.n_layers)
                .map(|_| Conv1dConfig::new(self.width, self.width, 1).init(device))
                .collect(),

            fc1: LinearConfig::new(self.width, 128).init(device),
            fc2: LinearConfig::new(128, self.out_channels).init(device),
        }
    }
}

impl<const R: usize> FNO<R> {
    fn apply_pointwise(conv: &Conv1d, x: Tensor<R>) -> Tensor<R> {
        let dims = x.dims();
        let (b, width) = (dims[0], dims[1]);
        let spatial: usize = dims[2..].iter().product();
        conv.forward(x.reshape([b, width, spatial])).reshape(dims)
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
