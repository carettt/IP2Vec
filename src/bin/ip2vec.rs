use anyhow::{Context, Result};
use burn::{config::Config, module::Module, record::{DefaultRecorder, Recorder}, Tensor};
use clap::Parser;
use ip2vec::{dataset::{Ip2VecDataset, Sample}, interface::{Commands, InferenceArgs, Reduction}, to_array2, train::TrainingConfig, Tch};
use petal_decomposition::{RandomizedPca, RandomizedPcaBuilder};

fn main() -> Result<()> {
  let args = InferenceArgs::parse();
  let device = &burn::backend::libtorch::LibTorchDevice::Cuda(0);

  let artifact_dir = args.store;

  let mut config_path = artifact_dir.clone();
  config_path.push("config.json");
  let mut instance_path = artifact_dir.clone();
  instance_path.push("model.mpk");

  let config = TrainingConfig::load(&config_path)?;
  let instance = DefaultRecorder::new().load(instance_path, device)?;

  let subnets: Vec<String>;
  let ports: Vec<String>;
  let protocols: Vec<String>;
  let input_tensor: Tensor<Tch, 2>;

  let reduction = args.command.get_reduction();

  match args.command {
    Commands::Single { features, .. } => {
      let input = Sample::new(
        features.src_ip.into(),
        features.dst_ip.into(),
        features.dst_port.into(),
        features.protocol.into()
      );

      input_tensor =
        Tensor::<Tch, 1>::from_data(input.encode().as_slice(), device)
          .reshape([1, 34]);

      let subnet = features.src_ip.octets().into_iter()
          .take(3)
          .map(|i| i.to_string())
          .collect::<Vec<_>>()
          .join(".");

      subnets = vec![
        format!("{subnet}.0/24")
      ];
      ports = vec![
        features.dst_port.to_string()
      ];
      protocols = vec![
        features.protocol.to_string()
      ];
    },
    Commands::Batch { file, .. } => {
      let mut reader = csv::Reader::from_path(&file)?;
      let dataset = Ip2VecDataset::deserialize(&mut reader, config.dataset_features)
        .context("failed to deserialize batch")?;

      input_tensor = dataset.batch_encode(device);
      [subnets, ports, protocols] = dataset.get_feature_strings();
    }
  };

  let model = config.model.init::<Tch>(&device).load_record(instance);

  let embedding = model.embed(input_tensor);
  println!("{}", embedding);

  if let Some(reduction) = reduction {
    match reduction {
      Reduction::Pca { dim, save_fit, load_fit } => {
        let output_path = "./pca.csv";

        let arr = to_array2(&embedding)?.to_owned();
        let mut writer = csv::Writer::from_path(&output_path)?;

        let mut pca: RandomizedPca<f32, _>;

        if load_fit {
          let mut fit_path = artifact_dir.clone();
          fit_path.push("pca_fit.mpk");

          let fit_file = std::fs::File::open(&fit_path)
            .context("could not open PCA fitting save file")?;

          pca = rmp_serde::decode::from_read(fit_file)?;
        } else {
          let dim = dim
            .context("dim is a required argument if not loading a PCA fitting save file")?;

          pca = RandomizedPcaBuilder::new(dim).seed(config.seed.into()).build();
          println!("starting PCA decomposition");
          pca.fit(&arr)?;
        }

        let mut headers = (1..=pca.n_components())
          .map(|i| format!("pc{}", i)).collect::<Vec<_>>();
        headers.push("subnet_24".to_string());
        headers.push("port".to_string());
        headers.push("protocol".to_string());

        let projection = pca.transform(&arr)?;

        println!("variance: {}",
          pca.explained_variance_ratio().iter().enumerate()
          .map(|(i, v)| format!("PC{}: {} ", i+1, v)).collect::<String>()
        );

        writer.write_record(headers)?;

        for (i, row) in projection.rows().into_iter().enumerate() {
          let mut record: Vec<String> = row.iter().map(|v| v.to_string()).collect();
          record.push(subnets[i].clone());
          record.push(ports[i].clone());
          record.push(protocols[i].clone());

          writer.write_record(&record)?;
        }

        writer.flush()?;
        println!("saved components to {output_path}");

        if save_fit {
          let mut save_path = artifact_dir.clone();
          save_path.push("pca_fit.mpk");

          let mut save_file = std::fs::File::create_new(&save_path)
            .context("PCA fitting save file already exists")?;

          rmp_serde::encode::write(&mut save_file, &pca)
            .context("unable to serialize PCA fit")?;

          println!("saved PCA fit to {}", save_path.display());
        }
      }
    }
  }

  Ok(())
}
