use anyhow::Result;

use ip2vec::{
  Backend,
  dataset::Ip2VecDataset,
  model::Ip2VecConfig
};

fn main() -> Result<()> {
  let _dataset = Ip2VecDataset::import_dataset("../NF-UNSW-NB15-v3/data/NF-UNSW-NB15-v3.csv")?;

  let device = Default::default();
  let model = Ip2VecConfig::new().init::<Backend>(&device);

  println!("{model}");

  Ok(())
}
