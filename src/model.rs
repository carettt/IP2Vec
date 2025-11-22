//! Module containing [Ip2Vec] model struct, including input projection and 
//! hidden layer (which will be extracted for embeddings). The model should be
//! constructed with [Ip2VecConfig] which provides default embed dimension params.

use crate::{
  train::EmbeddingOutput,
  dataset::batch::ContextBatch
};

use burn::{
  prelude::*,
  nn::loss::CosineEmbeddingLossConfig,
  tensor::backend::AutodiffBackend,
  train::{TrainStep, TrainOutput, ValidStep}
};

/// [Ip2Vec] builder with default settings of 100-d `src_ip`, and 25-d `dst_port`
/// and `protocol` projection layers; totaling to a 150-d embedding tensor.
#[derive(Config, Debug)]
pub struct Ip2VecConfig {
  #[config(default="100")]
  src_ip_embed_dim: usize,
  #[config(default="25")]
  dst_port_embed_dim: usize,
  #[config(default="25")]
  protocol_embed_dim: usize
}

impl Ip2VecConfig {
  /// [Ip2Vec] initializer, can be overriden using `with_<CONFIG>` builder functions.
  pub fn init<B: Backend>(&self, device: &B::Device) -> Ip2Vec<B> {
    let embed_dim =
      self.src_ip_embed_dim + self.dst_port_embed_dim + self.protocol_embed_dim;

    Ip2Vec {
      activation: nn::Relu::new(),

      src_ip_input: nn::LinearConfig::new(32, self.src_ip_embed_dim).init(device),
      dst_port_input: nn::LinearConfig::new(1, self.dst_port_embed_dim).init(device),
      protocol_input: nn::LinearConfig::new(1, self.protocol_embed_dim).init(device),

      hidden: nn::LinearConfig::new(embed_dim, embed_dim).init(device)
    }
  }
}

/// Neural network with 3 input tensors which are projected into a 150-d latent
/// space, which are embedded into a tensor of shape [batch_size, 150].
#[derive(Module, Debug)]
pub struct Ip2Vec<B: Backend> {
  activation: nn::Relu,

  src_ip_input: nn::Linear<B>,
  dst_port_input: nn::Linear<B>,
  protocol_input: nn::Linear<B>,

  hidden: nn::Linear<B>
}

impl<B: Backend> Ip2Vec<B> {
  fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
    let src_ip_proj = self.activation.forward(
      self.src_ip_input.forward(input.clone().narrow(1, 0, 32)));
    let dst_port_proj = self.activation.forward(
      self.dst_port_input.forward(input.clone().narrow(1, 32, 1)));
    let protocol_proj = self.activation.forward(
      self.protocol_input.forward(input.clone().narrow(1, 33, 1)));

    let combined = Tensor::cat(vec![src_ip_proj, dst_port_proj, protocol_proj], 1);

    self.hidden.forward(combined)
  }

  /// Flatten batch fields, perform forward passes and compute loss
  pub fn embed(
    &self,
    targets: Tensor<B, 2>,
    context: Tensor<B, 3>,
    mask: Tensor<B, 2, Int>
  ) -> EmbeddingOutput<B> {
    let embeddings = self.forward(targets.clone());

    let context_window = mask.dims()[1];

    // Repeat each row contiguously for every context
    let expanded_embeddings: Tensor<B, 2> = embeddings.clone()
      .unsqueeze_dim::<3>(1).repeat(&[1, context_window, 1]).flatten(0, 1);

    let context: Tensor<B, 2> = self.forward(context.flatten(0, 1));

    // Flatten mask and convert to 0 to -1
    let mask: Tensor<B, 1, Int> = (mask.flatten(0, 1) * 2) - 1;

    let loss = CosineEmbeddingLossConfig::new()
      .with_margin(0.5)
      .init()
      .forward(expanded_embeddings, context, mask);

    EmbeddingOutput::new(embeddings, loss)
  }
}

impl<B: AutodiffBackend> TrainStep<ContextBatch<B>, EmbeddingOutput<B>> for Ip2Vec<B> {
  fn step(&self, batch: ContextBatch<B>) -> TrainOutput<EmbeddingOutput<B>> {
    let output = self.embed(batch.samples, batch.contexts, batch.context_mask);

    TrainOutput::new(self, output.loss.backward(), output)
  }
}

impl <B: Backend> ValidStep<ContextBatch<B>, EmbeddingOutput<B>> for Ip2Vec<B> {
  fn step(&self, batch: ContextBatch<B>) -> EmbeddingOutput<B> {
    self.embed(batch.samples, batch.contexts, batch.context_mask)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{Cuda, dataset::batch::ContextBatch};

  use proptest::prelude::*;
  use burn::backend::cuda::CudaDevice;

  fn batch_strategy() -> impl Strategy<Value = ContextBatch<Cuda>> {
    let device = CudaDevice::new(0);

    prop::collection::vec(
      prop::collection::vec(
        0..=1,
        34
      ),
      2..10
    ).prop_map(move |vec: Vec<Vec<i32>>| {
      let batch_size = vec.len();
      let context_window = 2;

      let flattened: Vec<i32> = vec.into_iter().flatten().collect();
      let target_data = TensorData::new(flattened, [batch_size, 34]);

      let target: Tensor<Cuda, 2> =
        Tensor::from_data(target_data, &device);
      let mut contexts: Tensor<Cuda, 3> = 
        Tensor::zeros([batch_size, context_window, target.dims()[1]], &device);
      let mut context_mask = Tensor::zeros([batch_size, context_window], &device);

      let dims = target.dims();

      for i in 0..dims[0] {
        contexts = contexts.slice_assign(s![i], target.clone().slice(s![0..2]).unsqueeze::<3>());
        context_mask = context_mask.slice_fill(s![i, 0], 1);
      }

      ContextBatch::new(target, contexts, context_mask)
    })
  }

  proptest! {
    #[test]
    fn output_shape(batch in batch_strategy()) {
      let device = CudaDevice::new(0);

      let config = Ip2VecConfig::new();

      let batch_size = batch.samples.dims()[0];
      let embed_dim = config.src_ip_embed_dim + config.dst_port_embed_dim + config.protocol_embed_dim;

      let model = config.init::<Cuda>(&device);

      let output = model.embed(batch.samples, batch.contexts, batch.context_mask);

      prop_assert_eq!(output.embeddings.dims(), [batch_size, embed_dim]);
    }
  }
}
