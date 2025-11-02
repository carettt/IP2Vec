use anyhow::Result;

use burn::data::dataloader::DataLoaderBuilder;
use ip2vec::{
  dataset::Ip2VecDataset,
  model::Ip2VecConfig
};

type Backend = burn::backend::Cuda<f32, i32>;

fn main() -> Result<()> {
  let dataset = Ip2VecDataset::import_dataset("../NF-UNSW-NB15-v3/data/NF-UNSW-NB15-v3.csv")?;

  let device = Default::default();
  let model = Ip2VecConfig::new().init::<Backend>(&device);

  //let dataloader: DataLoaderBuilder<Backend, ip2vec::dataset::IpContext, ip2vec::dataset::ContextBatch<_>> = DataLoaderBuilder::new(dataset)
  //  .batch_size(25_000)
  //  .build(dataset);

  println!("{model}");

  Ok(())
}
