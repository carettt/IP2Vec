use anyhow::Result;

use burn::{backend::{cuda::CudaDevice, Autodiff}, optim::SgdConfig};
use ip2vec::{
  dataset::Ip2VecDataset, model::Ip2VecConfig, train::TrainingConfig, Backend
};

fn main() -> Result<()> {
  let device = CudaDevice::new(0);

  let trainer = TrainingConfig::new(
    String::from("/home/caret/Documents/cmp400/model"),
    Ip2VecConfig::new(),
    SgdConfig::new()
  );

  let dataset = Ip2VecDataset::import_dataset("../NF-UNSW-NB15-v3/data/NF-UNSW-NB15-v3.csv")?;

  trainer.init::<Backend>(&device)?;
  trainer.train::<Autodiff<Backend>>(dataset, &device);

  Ok(())
}
