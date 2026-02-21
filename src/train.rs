//! Module containing training functionality such as metrics

use std::path::PathBuf;

use crate::{dataset::{batch::ContextBatcher, ContextItem, Ip2VecDataset}, interface::ColumnFeatures, model::Ip2VecConfig, Tch};

use burn::{
  backend::libtorch::LibTorchDevice, data::{dataloader::DataLoaderBuilder, dataset::{transform::PartialDataset, Dataset}}, optim::SgdConfig, prelude::*, record::DefaultRecorder, tensor::{backend::AutodiffBackend, Transaction}, train::{metric::{
    Adaptor, CpuUse, CudaMetric, ItemLazy, LossInput, LossMetric
  }, LearnerBuilder, LearningStrategy}
};
use anyhow::Result;

/// Trait for applying an `Option` to a `struct` with builder-like config functions
pub trait ApplyOption: Sized {
  /// Apply `val` to `f` if `Some`, otherwise return self unchanged
  fn apply_opt<T>(self, f: impl FnOnce(Self, T) -> Self, val: Option<T>) -> Self {
    match val {
      Some(val) => f(self, val),
      None => self
    }
  }
}

/// Struct containing various training parameters, including optimizer and model
/// configuration objects ([Ip2VecConfig], [SgdConfig])
#[derive(Config, Debug)]
pub struct TrainingConfig {
  #[config(default = "PathBuf::from(\"./model\")")]
  artifact_path: PathBuf,
  
  /// Path to training dataset CSV file
  pub dataset_path: PathBuf,
  /// Names of feature columns in dataset CSV file
  pub dataset_features: ColumnFeatures,

  /// Embedding model configuration
  pub model: Ip2VecConfig,
  optimizer: SgdConfig,

  #[config(default = 1)]
  seed: u64,
  #[config(default = 10)]
  epochs: usize,
  #[config(default = 0.6)]
  split_ratio: f64,
  #[config(default = 64)]
  batch_size: usize,
  #[config(default = 8)]
  threads: usize,
  #[config(default = 1.0e-4)]
  learning_rate: f64,
  
  #[config(default = 5)]
  context_window: usize,
  #[config(default = 5)]
  neg_multiplier: usize
}

impl ApplyOption for TrainingConfig {}

impl TrainingConfig {
  /// Clear previous training data from `artifact_path` and seed RNG
  pub fn init<B: Backend>(&self, device: &B::Device) -> Result<()> {
    match std::fs::remove_dir_all(&self.artifact_path) {
      Ok(_) => {},
      Err(e) if e.kind() == std::io::ErrorKind::NotFound => {},
      Err(e) => return Err(e.into()),
    };

    std::fs::create_dir_all(&self.artifact_path)?;

    let mut config_path = self.artifact_path.clone();
    config_path.push("config.json");

    self.save(&config_path)?;

    B::seed(&device, self.seed);

    Ok(())
  }

  fn split_dataset<B, D>(&self, dataset: D)
  -> (PartialDataset<D, ContextItem>, PartialDataset<D, ContextItem>)
  where
    B: Backend,
    D: Dataset<ContextItem> + Clone,
  {
    let len = dataset.clone().len();
    let split_idx = (len as f64 * self.split_ratio).round() as usize;

    let train: PartialDataset<D, ContextItem> =
      PartialDataset::new(dataset.clone(), 0, split_idx);

    let test: PartialDataset<D, ContextItem> =
      PartialDataset::new(dataset, split_idx, len);

    (train, test)
  }

  /// Train model using parameters configured with [TrainingConfig]
  pub fn train<B>(&self, dataset: Ip2VecDataset, device: &B::Device)
  where 
    B: AutodiffBackend<Device = LibTorchDevice>,
  {
    // Initialize dataset & dataloaders w/ batcher
    let (train, test) = self.split_dataset::<B, _>(dataset);

    let batcher = ContextBatcher::new(
      self.context_window,
      self.neg_multiplier
    );
    let cpu_device = LibTorchDevice::Cpu;

    let dataloader_train =
      DataLoaderBuilder::new(batcher.clone())
        .batch_size(self.batch_size)
        .shuffle(self.seed)
        .num_workers(self.threads)
        .set_device(cpu_device)
        .build(train);

    let dataloader_test =
      DataLoaderBuilder::new(batcher.clone())
        .batch_size(self.batch_size)
        .shuffle(self.seed)
        .num_workers(self.threads)
        .set_device(cpu_device)
        .build(test);

    // Initialize learner with metrics
    let learner = LearnerBuilder::new(&self.artifact_path)
      .metric_train_numeric(LossMetric::new())
      .metric_train(CudaMetric::new())
      .metric_train(CpuUse::new())
      .metric_valid_numeric(LossMetric::new())
      .metric_valid(CudaMetric::new())
      .metric_valid(CpuUse::new())
      .with_file_checkpointer(DefaultRecorder::new())
      .learning_strategy(LearningStrategy::SingleDevice(device.clone()))
      .num_epochs(self.epochs)
      .summary()
      .build(
        self.model.init::<B>(&device),
        self.optimizer.init(),
        self.learning_rate
      );

    // Fit model and save
    learner.fit(dataloader_train, dataloader_test)
      .model
      .save_file(&self.artifact_path, &DefaultRecorder::new())
      .expect("could not fit model");
  }
}

/// Embedding output for adapted to multiple metrics for training
pub struct EmbeddingOutput<B: Backend> {
  /// Embeddings of shape [batch_size, 150]
  pub embeddings: Tensor<B, 2>,

  /// Loss of shape [batch_size]
  pub loss: Tensor<B, 1>
}

impl<B: Backend> EmbeddingOutput<B> {
  /// Create new [EmbeddingOutput] from tensors
  pub fn new(embeddings: Tensor<B, 2>, loss: Tensor<B, 1>) -> Self{
    Self { embeddings, loss }
  }
}

impl<B: Backend> Adaptor<LossInput<B>> for EmbeddingOutput<B> {
  fn adapt(&self) -> LossInput<B> {
    LossInput::new(self.loss.clone())
  }
}

impl<B: Backend> Adaptor<CudaMetric> for EmbeddingOutput<B> {
  fn adapt(&self) -> CudaMetric {
    CudaMetric::new()
  }
}

impl<B: Backend> Adaptor<CpuUse> for EmbeddingOutput<B> {
  fn adapt(&self) -> CpuUse {
    CpuUse::new()
  }
}

impl<B: Backend> ItemLazy for EmbeddingOutput<B> {
  type ItemSync = EmbeddingOutput<Tch>;

  fn sync(self) -> EmbeddingOutput<Tch> {
    let [embeddings, loss] = Transaction::default()
      .register(self.embeddings)
      .register(self.loss)
      .execute()
      .try_into()
      .expect("could not sync embedding output");

    let device = &burn::backend::libtorch::LibTorchDevice::Cuda(1);

    EmbeddingOutput {
      embeddings: Tensor::from_data(embeddings, device),
      loss: Tensor::from_data(loss, device)
    }
  }
}
