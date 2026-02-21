use anyhow::{Context, Result};
use burn::{config::Config, module::Module, record::{DefaultRecorder, Recorder}, Tensor};
use clap::Parser;
use ip2vec::{dataset::{Ip2VecDataset, IpContext}, interface::{Commands, InferenceArgs}, to_matrix, train::TrainingConfig, Tch};
use smartcore::{decomposition::pca::{PCAParameters, PCA}, linalg::basic::arrays::{Array, Array2}};

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
    let embedding_matrix = to_matrix(embedding.clone())?;

    let pca =PCA::fit(
      &embedding_matrix,
      PCAParameters::default().with_n_components(2)
    )?;

    let projection = pca.transform(&embedding_matrix)?;

    let (n_rows, n_cols) = projection.shape();

    for i in 0..n_rows {
      let mut row_vec: Vec<String> = Vec::with_capacity(n_cols);
      for j in 0..n_cols {
        row_vec.push(projection.get((i, j)).to_string());
      }

      writer.write_record(row_vec)?;
    }
  }

  println!("{}", embedding) ;
  Ok(())
}
