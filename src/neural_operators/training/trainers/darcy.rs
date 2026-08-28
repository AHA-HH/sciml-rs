// Tensor-native decode: (x * (std + eps)) + mean — mirrors
// UnitGaussianNormaliser::decode's math exactly, but built from
// differentiable Tensor ops so autodiff can trace gradients through it.
// mean/std have rank D (spatial dims, batch axis reduced away during fit);
// x has rank D2 (spatial dims + batch). unsqueeze broadcasts mean/std
// across the batch axis.
fn decode_tensor<const D: usize, const D2: usize>(
    x: Tensor<D2>,
    mean: &Tensor<D>,
    std: &Tensor<D>,
    eps: f64,
) -> Tensor<D2> {
    let mean_b = mean.clone().unsqueeze::<D2>();
    let std_b = std.clone().unsqueeze::<D2>();
    x * (std_b + eps) + mean_b
}

// Converts a fitted UnitGaussianNormaliser's mean/std into rank-D Tensors,
// once, before training starts. Never called per-batch — mean/std are
// fixed after fit(), so repeated ndarray<->Tensor conversion in the loop
// would be pure waste.
fn normaliser_to_tensors<const D: usize>(
    normaliser: &UnitGaussianNormaliser,
    device: &Device,
) -> (Tensor<D>, Tensor<D>) {
    let mean_data: Vec<f64> = normaliser.mean_ref().iter().copied().collect();
    let mean_shape = normaliser.mean_ref().shape().to_vec();
    let std_data: Vec<f64> = normaliser.std_ref().iter().copied().collect();

    let mean = Tensor::<D>::from_data(
        TensorData::new(mean_data, mean_shape.clone()),
        device,
    );
    let std = Tensor::<D>::from_data(
        TensorData::new(std_data, mean_shape),
        device,
    );
    (mean, std)
}

pub fn train_darcy_flow_fno(
    train_data: OperatorDataset,
    test_data: OperatorDataset,
    y_normalizer: &UnitGaussianNormaliser,
    config: &FNOConfig,
    device: &Device,
) -> (FNOnd<4>, Vec<EpochMetricsNd>) {
    let model_config = FNOndConfig {
        modes: config.modes.clone(),
        width: config.width,
        data_channels: config.data_channels,
        out_channels: config.out_channels,
        n_layers: config.n_layers,
    };
    let mut model: FNOnd<4> = model_config.init(device);

    let optim_config = AdamConfig::new().with_epsilon(1e-8).with_weight_decay(Some(
        burn::optim::decay::WeightDecayConfig::new(config.weight_decay),
    ));
    let mut optim = optim_config.init();

    let steps_per_epoch = (config.n_train + config.batch_size - 1) / config.batch_size;
    let total_steps = config.epochs * steps_per_epoch;

    let mut scheduler = CosineAnnealingLrSchedulerConfig::new(config.learning_rate, total_steps)
        .with_min_lr(config.min_lr)
        .init()
        .expect("valid cosine scheduler config");

    let loss_fn = LpLoss::new(1, 2, false, true);

    // Fitted statistics, converted to Tensor ONCE, outside the loop.
    let (y_mean, y_std) = normaliser_to_tensors::<2>(y_normalizer, device);
    let eps = y_normalizer.eps_val();

    let train_loader =
        DataLoaderBuilder::new(OperatorBatcherNd::new(device.clone()))
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .build(train_data);

    let test_loader =
        DataLoaderBuilder::new(OperatorBatcherNd::<4, 3>::new(device.clone()))
            .batch_size(config.batch_size)
            .build(test_data);

    let mut metrics: Vec<EpochMetricsNd> = Vec::with_capacity(config.epochs);

    for epoch in 0..config.epochs {
        let mut train_mse_accum = 0.0_f32;
        let mut train_l2_accum = 0.0_f32;
        let mut last_lr = 0.0_f64;

        for batch in train_loader.iter() {
            let out = model.forward(batch.inputs.clone());
            let [b, s1, s2, _] = out.dims();
            let out_flat = out.reshape([b, s1, s2]);

            // Decode BOTH sides before computing loss — matches Li's
            // out = y_normalizer.decode(out); y = y_normalizer.decode(y).
            // Tensor-native, so autodiff traces through this into the model.
            let out_decoded = decode_tensor::<2, 3>(out_flat, &y_mean, &y_std, eps);
            let y_decoded = decode_tensor::<2, 3>(batch.targets.clone(), &y_mean, &y_std, eps);

            let out_view = out_decoded.clone().reshape([b, s1 * s2]);
            let y_view = y_decoded.clone().reshape([b, s1 * s2]);

            let mse = (out_decoded - y_decoded).powf_scalar(2.0).mean();
            let l2 = loss_fn.rel(out_view, y_view);

            train_mse_accum += mse.into_scalar();
            train_l2_accum += l2.clone().into_scalar();

            let grads = l2.backward();
            let grad_params = GradientsParams::from_grads(grads, &model);
            let lr = scheduler.step();
            model = optim.step(lr, model, grad_params);
            last_lr = lr;
        }

        let mut test_l2_accum = 0.0_f32;

        for batch in test_loader.iter() {
            let out = model.forward(batch.inputs);
            let [b, s1, s2, _] = out.dims();
            let out_flat = out.reshape([b, s1, s2]);

            // y (batch.targets) is NOT decoded here — matches Li's reference,
            // where y_test is never encoded during loading in the first
            // place, so only predictions need decoding.
            let out_decoded = decode_tensor::<2, 3>(out_flat, &y_mean, &y_std, eps);

            let out_view = out_decoded.reshape([b, s1 * s2]);
            let y_view = batch.targets.reshape([b, s1 * s2]);

            let test_l2 = loss_fn.rel(out_view, y_view);
            test_l2_accum += test_l2.into_scalar();
        }

        let m = EpochMetricsNd {
            epoch,
            train_mse: train_mse_accum / steps_per_epoch as f32,
            train_l2: train_l2_accum / config.n_train as f32,
            test_l2: test_l2_accum / config.n_test as f32,
            current_lr: last_lr,
        };

        println!("{:?}", m);
        metrics.push(m);
    }

    (model, metrics)
}