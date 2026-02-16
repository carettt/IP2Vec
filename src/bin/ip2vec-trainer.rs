use std::path::PathBuf;

use anyhow::{Context, Result};

use burn::{config::Config, backend::{libtorch::LibTorchDevice, Autodiff}, optim::SgdConfig};
use clap::Parser;

use ip2vec::{
  dataset::Ip2VecDataset, interface::{ColumnFeatures, TrainerArgs}, model::Ip2VecConfig, train::{ApplyOption, TrainingConfig}, Tch
};

fn main() -> Result<()> {
  let args = TrainerArgs::parse();

  let device = LibTorchDevice::Cuda(0);

  let trainer: TrainingConfig;

  let dataset: PathBuf;
  let features: ColumnFeatures;

  if let Some(store_path) = args.store {
    let mut config_path = store_path.clone();
    config_path.push("config.json");

    trainer = TrainingConfig::load(&config_path)?;

    if let Some(dataset_path) = args.dataset {
      dataset = dataset_path;
    } else {
      dataset = trainer.dataset_path.clone();
    }

    if let Some(feature_names) = args.features {
      features = feature_names;
    } else {
      features = trainer.dataset_features.clone();
    }
  } else {
    dataset = args.dataset
      .context("should have either store or positional dataset argument")?;
    features = args.features
      .context("should have either store or dataset feature column name arguments")?;

    trainer = TrainingConfig::new(
      PathBuf::from("./model"),
      dataset.clone(),
      features.clone(),
      Ip2VecConfig::new(),
      SgdConfig::new()
    );
  }

  let trainer = trainer
    .apply_opt(TrainingConfig::with_seed, args.params.seed)
    .apply_opt(TrainingConfig::with_epochs, args.params.epochs)
    .apply_opt(TrainingConfig::with_split_ratio, args.params.split_ratio)
    .apply_opt(TrainingConfig::with_batch_size, args.params.batch_size)
    .apply_opt(TrainingConfig::with_threads, args.params.threads)
    .apply_opt(TrainingConfig::with_learning_rate, args.params.learning_rate)
    .apply_opt(TrainingConfig::with_context_window, args.params.context_window);

  let dataset = Ip2VecDataset::import_dataset(&dataset, features)
    .context("failed to import dataset")?;

  trainer.init::<Tch>(&device)
    .context("failed to initialize trainer")?;
  trainer.train::<Autodiff<Tch>>(dataset, &device);

  Ok(())
}
