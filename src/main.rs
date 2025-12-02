use anyhow::{Context, Result};

use burn::{backend::{libtorch::LibTorchDevice, Autodiff}, optim::SgdConfig};
use ip2vec::{
  dataset::Ip2VecDataset, model::Ip2VecConfig, train::TrainingConfig, Tch
};

fn main() -> Result<()> {
  let device = LibTorchDevice::Cuda(0);

  let trainer = TrainingConfig::new(
    String::from("/home/jovyan/work/model"),
    Ip2VecConfig::new(),
    SgdConfig::new())
      .with_batch_size(4096)
      .with_epochs(20)
      .with_learning_rate(1.0e-3)
      .with_threads(128);

  let dataset = Ip2VecDataset::import_dataset("/home/jovyan/work/NF-UNSW-NB15-v3.csv")
    .context("failed to import dataset")?;

  trainer.init::<Tch>(&device)
    .context("failed to initialize trainer")?;
  trainer.train::<Autodiff<Tch>>(dataset, &device);

  Ok(())
}
