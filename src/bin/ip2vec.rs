use anyhow::{Context, Result};
use burn::{config::Config, module::Module, record::{DefaultRecorder, Recorder}, Tensor};
use clap::Parser;
use ip2vec::{dataset::{Ip2VecDataset, IpContext}, interface::{Commands, InferenceArgs}, to_array2, train::TrainingConfig, Tch};
use petal_decomposition::{PcaBuilder, RandomizedPcaBuilder};

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

  let input_tensor: Tensor<Tch, 2>;

  match args.command {
    Commands::Single { features } => {
      let input = IpContext::new(
        features.src_ip,
        features.dst_ip,
        features.dst_port,
        features.protocol
      );

      input_tensor =
        Tensor::<Tch, 1>::from_data(input.encode().as_slice(), device)
          .reshape([1, 34]);
    },
    Commands::Batch { file } => {
      let mut reader = csv::Reader::from_path(&file)?;
      let dataset = Ip2VecDataset::import_dataset(&mut reader, config.dataset_features)
        .context("failed to import dataset")?;

      input_tensor = dataset.batch_encode(device);
    }
  };

  let model = config.model.init::<Tch>(&device).load_record(instance);

  let embedding = model.embed(input_tensor);

  if args.pca {
    let mut writer = csv::Writer::from_path("./pca.csv")?;

    let embedding_arr = to_array2(&embedding)?;

    let mut pca = RandomizedPcaBuilder::new(3).seed(config.seed.into()).build();
    let projection = pca.fit_transform(&embedding_arr)?;

    for row in projection.rows() {
      let record: Vec<String> = row.iter().map(|i| i.to_string()).collect();
      writer.write_record(&record)?;
    }

    writer.flush()?;
  }

  println!("{}", embedding) ;
  Ok(())
}
