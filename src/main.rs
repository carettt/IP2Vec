use anyhow::Result;

use ip2vec::{
  dataset::Dataset,
  model::Ip2VecConfig
};

type Backend = burn::backend::Cuda<f32, i32>;

fn main() -> Result<()> {
  let _dataset = Dataset::import_dataset("../NF-UNSW-NB15-v3/data/NF-UNSW-NB15-v3.csv")?;

  let device = Default::default();
  let model = Ip2VecConfig::new().init::<Backend>(&device);

  println!("{model}");

  Ok(())
}
