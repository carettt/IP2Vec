use anyhow::{Context, Result};
use burn::{backend::libtorch::LibTorchDevice, config::Config, data::dataloader::batcher::Batcher, module::Module, record::{CompactRecorder, DefaultRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Record, Recorder}, Tensor};
use clap::Parser;
use ip2vec::{dataset::{batch::ContextBatcher, ContextItem, Ip2VecDataset, IpContext}, interface::{InferenceArgs, TrainerArgs}, model::Ip2Vec, train::TrainingConfig, Tch};

fn main() -> Result<()> {
  let args = InferenceArgs::parse();
  let device = &burn::backend::libtorch::LibTorchDevice::Cuda(0);

  let artifact_dir = args.config;

  let mut config_path = artifact_dir.clone();
  config_path.push("config.json");
  let mut instance_path = artifact_dir.clone();
  instance_path.push("model.mpk");

  let config = TrainingConfig::load(&config_path)?;
  let instance = DefaultRecorder::new().load(instance_path, device)?;

  let input = IpContext::new(
    args.features.src_ip,
    args.features.dst_ip,
    args.features.dst_port,
    args.features.protocol
  );

  let input_tensor: Tensor<Tch, 2> =
    Tensor::<Tch, 1>::from_data(input.encode().as_slice(), device)
      .reshape([1, 34]);

  let model = config.model.init::<Tch>(&device).load_record(instance);

  let embedding = model.embed(input_tensor);

  println!("{}", embedding) ;

  Ok(())
}
