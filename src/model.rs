//! Module containing [Ip2Vec] model struct, including input projection and 
//! hidden layer (which will be extracted for embeddings). The model should be
//! constructed with [Ip2VecConfig] which provides default embed dimension params.

use crate::{
  dataset::batch::ContextBatch, loss::NegEmbeddingLossConfig, train::EmbeddingOutput
};

use burn::{
  backend::libtorch::LibTorchDevice, prelude::*, tensor::{backend::AutodiffBackend, Transaction}, train::{TrainOutput, TrainStep, ValidStep}
};

/// [Ip2Vec] builder with default settings of 100-d `src_ip`, and 25-d `dst_port`
/// and `protocol` projection layers; totaling to a 150-d embedding tensor.
#[derive(Config, Debug)]
pub struct Ip2VecConfig {
  #[config(default="128")]
  src_ip_embed_dim: usize,
  #[config(default="16")]
  dst_port_embed_dim: usize,
  #[config(default="4")]
  protocol_embed_dim: usize,
}

impl Ip2VecConfig {
  /// [Ip2Vec] initializer, can be overriden using `with_<CONFIG>` builder functions.
  pub fn init<B: Backend>(&self, device: &B::Device) -> Ip2Vec<B> {
    let combined_dim =
      self.src_ip_embed_dim + self.dst_port_embed_dim + self.protocol_embed_dim;

    Ip2Vec {
      activation: nn::Gelu::new(),

      src_ip_input: nn::LinearConfig::new(32, self.src_ip_embed_dim).init(device),
      dst_port_input: nn::EmbeddingConfig::new(65536, self.dst_port_embed_dim).init(device),
      protocol_input: nn::EmbeddingConfig::new(256, self.protocol_embed_dim).init(device),

      hidden: nn::LinearConfig::new(combined_dim, combined_dim).init(device)
    }
  }
}

/// Neural network with 3 input tensors which are projected into a 150-d latent
/// space, which are embedded into a tensor of shape [batch_size, 150].
#[derive(Module, Debug)]
pub struct Ip2Vec<B: Backend> {
  activation: nn::Gelu,

  src_ip_input: nn::Linear<B>,
  dst_port_input: nn::Embedding<B>,
  protocol_input: nn::Embedding<B>,

  hidden: nn::Linear<B>,
}

impl<B: Backend> Ip2Vec<B> {
  /// Embed tensor of shape [batch_size, 34] to tensor of shape [batch_size, 150]
  pub fn embed(
    &self,
    input: Tensor<B, 2>,
  ) -> Tensor<B, 2> {
    let src_ip = input.clone().slice(s![.., 0..32]);
    let dst_port = (input.clone().slice(s![.., 32]) * 65535.0).int();
    let protocol = (input.clone().slice(s![.., 33]) * 255.0).int();

    let src_ip_proj = self.activation.forward(
      self.src_ip_input.forward(src_ip));
    let dst_port_proj = self.activation.forward(
      self.dst_port_input.forward(dst_port).squeeze());
    let protocol_proj = self.activation.forward(
      self.protocol_input.forward(protocol).squeeze());

    let combined = Tensor::cat(vec![src_ip_proj, dst_port_proj, protocol_proj], 1);

    self.hidden.forward(combined)
  }

  /// Flatten batch fields, perform forward passes and compute loss
  pub fn forward(
    &self,
    targets: Tensor<B, 2>,
    positive_context: Tensor<B, 3>,
    negative_context: Tensor<B, 3>
  ) -> EmbeddingOutput<B> {
    let [batch_size, pos_context_num, feat_dim] = positive_context.dims();
    let [_, neg_context_num, _] = negative_context.dims();

    let embeddings = self.embed(targets.clone()); // [batch_size, 150]
    let embedding_dim = embeddings.dims()[1];

    let pos_context_embeddings = self.embed(positive_context
      .reshape([batch_size * pos_context_num, feat_dim]))
      .reshape([batch_size, pos_context_num, embedding_dim]);
    let neg_context_embeddings = self.embed(negative_context
      .reshape([batch_size * neg_context_num, feat_dim]))
      .reshape([batch_size, neg_context_num, embedding_dim]);

    let loss = NegEmbeddingLossConfig::new()
      .init()
      .forward(
        embeddings.clone(),
        pos_context_embeddings,
        neg_context_embeddings
      );

    EmbeddingOutput::new(embeddings, loss)
  }
}

impl<B> TrainStep<ContextBatch<B>, EmbeddingOutput<B>> for Ip2Vec<B> 
where 
  B: AutodiffBackend<Device = LibTorchDevice>
{
  fn step(&self, batch: ContextBatch<B>) -> TrainOutput<EmbeddingOutput<B>> {
    let gpu_device = LibTorchDevice::Cuda(0);

    let [sample_data, pos_context_data, neg_context_data] = Transaction::default()
      .register(batch.samples)
      .register(batch.positive_context)
      .register(batch.negative_context)
      .execute()
      .try_into()
      .expect("failed to sync batch to GPU");

    let samples: Tensor<B, 2> = Tensor::from_data(sample_data, &gpu_device);
    let positive_context: Tensor<B, 3> = Tensor::from_data(pos_context_data, &gpu_device);
    let negative_context: Tensor<B, 3> = Tensor::from_data(neg_context_data, &gpu_device);

    let output = self.forward(samples, positive_context, negative_context);

    TrainOutput::new(self, output.loss.backward(), output)
  }
}

impl <B> ValidStep<ContextBatch<B>, EmbeddingOutput<B>> for Ip2Vec<B>
where 
  B: Backend<Device = LibTorchDevice>
{
  fn step(&self, batch: ContextBatch<B>) -> EmbeddingOutput<B> {
    let gpu_device = LibTorchDevice::Cuda(0);

    let [sample_data, pos_context_data, neg_context_data] = Transaction::default()
      .register(batch.samples)
      .register(batch.positive_context)
      .register(batch.negative_context)
      .execute()
      .try_into()
      .expect("failed to sync batch to GPU");

    let samples: Tensor<B, 2> = Tensor::from_data(sample_data, &gpu_device);
    let positive_context: Tensor<B, 3> = Tensor::from_data(pos_context_data, &gpu_device);
    let negative_context: Tensor<B, 3> = Tensor::from_data(neg_context_data, &gpu_device);

    self.forward(samples, positive_context, negative_context)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{dataset::{batch::ContextBatch, Ip2VecDataset}, Tch};

  use proptest::prelude::*;
  use burn::backend::libtorch::LibTorchDevice;

  proptest! {
    #[test]
    fn output_shape(dataset in any::<Ip2VecDataset>()) {
      let device = LibTorchDevice::Cuda(0);
      let config = Ip2VecConfig::new();
      let model = config.init::<Tch>(&device);

      let batch_size = dataset.samples.len();
      let embed_dim = config.src_ip_embed_dim + config.dst_port_embed_dim + config.protocol_embed_dim;

      let tensor = dataset.batch_encode(&device);

      let output = model.embed(tensor);

      prop_assert_eq!(output.dims(), [batch_size, embed_dim]);
    }
  }
}
