use std::path::PathBuf;

use burn::{
  backend::{Autodiff, libtorch::LibTorchDevice},
  optim::SgdConfig,
};
use ip2vec::{
  Tch, dataset::Ip2VecDataset, interface::ColumnFeatures, model::Ip2VecConfig,
  train::TrainingConfig,
};
use itertools::iproduct;
use rand::SeedableRng;
use serde_json::json;

fn config_iter(
  dataset: PathBuf,
  features: ColumnFeatures,
  model: Ip2VecConfig,
  optimizer: SgdConfig,
) -> impl Iterator<Item = TrainingConfig> {
  let batch_sizes = [1024, 2048, 4096];
  let learning_rates = [1.0e-2, 1.0e-3, 1.0e-4];
  let context_windows = [4, 5, 6];
  let neg_multiplers = [2, 4, 5];

  iproduct!(
    batch_sizes,
    learning_rates,
    context_windows,
    neg_multiplers
  )
  .map(move |(bs, lr, cw, nm)| {
    TrainingConfig::new(
      dataset.clone(),
      features.clone(),
      model.clone(),
      optimizer.clone(),
    )
    .with_batch_size(bs)
    .with_learning_rate(lr)
    .with_context_window(cw)
    .with_neg_multiplier(nm)
  })
}

fn save_result(config_id: usize, config: &TrainingConfig, loss: f32) {
  let result_string = json!({
    "batch_size": config.batch_size,
    "learning_rate": config.learning_rate,
    "context_window": config.context_window,
    "neg_multiplier": config.neg_multiplier,
    "loss": loss
  })
  .to_string();

  std::fs::write(
    format!("./experiments/trial_results/trial_{config_id}"),
    result_string,
  )
  .unwrap()
}

#[test]
fn trial() {
  let config_id = std::env::var("CONFIG_ID")
    .expect("config ID not set")
    .parse::<usize>()
    .unwrap();

  let device = LibTorchDevice::Cuda(0);

  let dataset_path = PathBuf::from("../NF-UNSW-NB15-v3.csv");
  let features = ColumnFeatures {
    src_ip: "IPV4_SRC_ADDR".to_string(),
    dst_ip: "IPV4_DST_ADDR".to_string(),
    dst_port: "L4_DST_PORT".to_string(),
    protocol: "PROTOCOL".to_string(),
  };

  let artifact_path = PathBuf::from(format!("./experiments/trial_{config_id}"));

  let config = config_iter(
    dataset_path.clone(),
    features.clone(),
    Ip2VecConfig::new(),
    SgdConfig::new(),
  )
  .nth(config_id)
  .expect("config ID out of range")
  .with_artifact_path(artifact_path.clone());

  let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

  let mut reader = csv::Reader::from_path(&dataset_path).unwrap();
  let mut dataset = Ip2VecDataset::deserialize(&mut reader, features).unwrap();
  dataset
    .preprocess(&mut rng, config.context_window, config.neg_multiplier)
    .unwrap();

  config
    .init::<Tch>(&device)
    .expect("failed to initialize config");

  config.train::<Autodiff<Tch>>(dataset, &device);

  let mut loss_path = artifact_path.clone();
  loss_path.push(format!("valid/epoch-{}/Loss.log", config.epochs));

  let loss = std::fs::read_to_string(&loss_path)
    .unwrap()
    .lines()
    .last()
    .unwrap()
    .split(',')
    .next()
    .unwrap()
    .parse::<f32>()
    .unwrap();

  println!("[CONFIG_{config_id}] finished training! Loss: {loss}");
  save_result(config_id, &config, loss);
}
