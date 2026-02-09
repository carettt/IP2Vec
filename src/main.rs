use std::path::PathBuf;

use anyhow::{Context, Result};

use burn::{backend::{libtorch::LibTorchDevice, Autodiff}, optim::SgdConfig};
use clap::Parser;

use ip2vec::{
  dataset::Ip2VecDataset, interface::{Arguments}, model::Ip2VecConfig, train::{TrainingConfig, ApplyOption}, Tch
};

fn main() -> Result<()> {
  let args = Arguments::parse();

  let device = LibTorchDevice::Cuda(0);

  let trainer = TrainingConfig::new(
    args.output.unwrap_or_else(|| PathBuf::from("./model")),
    Ip2VecConfig::new(),
    SgdConfig::new()
  );

  let trainer = trainer
    .apply_opt(TrainingConfig::with_seed, args.params.seed)
    .apply_opt(TrainingConfig::with_epochs, args.params.epochs)
    .apply_opt(TrainingConfig::with_split_ratio, args.params.split_ratio)
    .apply_opt(TrainingConfig::with_batch_size, args.params.batch_size)
    .apply_opt(TrainingConfig::with_threads, args.params.threads)
    .apply_opt(TrainingConfig::with_learning_rate, args.params.learning_rate)
    .apply_opt(TrainingConfig::with_context_window, args.params.context_window);

  let dataset = Ip2VecDataset::import_dataset(&args.dataset)
    .context("failed to import dataset")?;

  trainer.init::<Tch>(&device)
    .context("failed to initialize trainer")?;
  trainer.train::<Autodiff<Tch>>(dataset, &device);

  Ok(())
}
