//! Module containing [Ip2Vec] model struct, including input projection and 
//! hidden layer (which will be extracted for embeddings). The model should be
//! constructed with [Ip2VecConfig] which provides default embed dimension params.

use crate::{
  train::EmbeddingOutput,
  dataset::batch::ContextBatch
};

use burn::{
  prelude::*,
  backend::libtorch::LibTorchDevice,
  nn::loss::CosineEmbeddingLossConfig,
  tensor::{backend::AutodiffBackend, Transaction},
  train::{TrainOutput, TrainStep, ValidStep}
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
  pub fn embed(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
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
  pub fn forward(
    &self,
    targets: Tensor<B, 2>,
    context: Vec<Tensor<B, 2>>,
    mask: Tensor<B, 1, Int>
  ) -> EmbeddingOutput<B> {
    let embeddings = self.embed(targets.clone()); // [batch_size, 150]

    // Repeat each row contiguously for every context
    //let expanded_embeddings: Tensor<B, 2> = embeddings.clone()
    //  .unsqueeze_dim::<3>(1).repeat(&[1, context_window, 1]).flatten(0, 1);
    let mut duplicated_embeddings: Vec<Tensor<B, 2>> = Vec::with_capacity(context.len());

    //eprintln!("context_len: {}, embedding_len: {}", context.len(), embeddings.dims()[0]);

    for (i, context) in context.iter().enumerate() {
      duplicated_embeddings.push(embeddings.clone().slice(s![i, ..])
        .repeat_dim(0, context.dims()[0]));
    }

    //let context: Tensor<B, 2> = self.embed(context);
    let expanded_embeddings: Tensor<B, 2> = Tensor::cat(duplicated_embeddings, 0);
    let context: Tensor<B, 2> = self.embed(Tensor::cat(context, 0));

    // L2 Normalize embedding outputs
    //let expanded_embeddings = l2_norm(expanded_embeddings, 1);
    //let context = l2_norm(context, 1);

    // Flatten mask and convert to 0 to -1
    //let mask: Tensor<B, 1, Int> = (mask.flatten(0, 1) * 2) - 1;

    //eprintln!("expanded: {:?}, context: {:?}, mask: {:?}", expanded_embeddings.dims(), context.dims(), mask.dims());

    let loss = CosineEmbeddingLossConfig::new()
      .with_margin(0.5)
      .with_reduction(nn::loss::Reduction::Mean)
      .init()
      .forward(expanded_embeddings, context, mask);

    EmbeddingOutput::new(embeddings, loss)
  }
}

impl<B> TrainStep<ContextBatch<B>, EmbeddingOutput<B>> for Ip2Vec<B> 
where 
  B: AutodiffBackend<Device = LibTorchDevice>
{
  fn step(&self, batch: ContextBatch<B>) -> TrainOutput<EmbeddingOutput<B>> {
    let gpu_device = LibTorchDevice::Cuda(0);

    let [sample_data, context_mask_data] = Transaction::default()
      .register(batch.samples)
      .register(batch.context_mask)
      .execute()
      .try_into()
      .expect("failed to sync batch to GPU");

    let samples = Tensor::from_data(sample_data, &gpu_device);
    let context_mask = Tensor::from_data(context_mask_data, &gpu_device);

    let mut context_transaction = Transaction::default();
    for context in batch.contexts {
      context_transaction = context_transaction.register(context);
    }
    let context_data = context_transaction.execute();

    let mut contexts = Vec::with_capacity(context_data.len());
    for data in context_data {
      contexts.push(Tensor::from_data(data, &gpu_device));
    }

    let output = self.forward(samples, contexts, context_mask);

    TrainOutput::new(self, output.loss.backward(), output)
  }
}

impl <B> ValidStep<ContextBatch<B>, EmbeddingOutput<B>> for Ip2Vec<B>
where 
  B: Backend<Device = LibTorchDevice>
{
  fn step(&self, batch: ContextBatch<B>) -> EmbeddingOutput<B> {
    let gpu_device = LibTorchDevice::Cuda(0);

    let [sample_data, context_mask_data] = Transaction::default()
      .register(batch.samples)
      .register(batch.context_mask)
      .execute()
      .try_into()
      .expect("failed to sync batch to GPU");

    let samples: Tensor<B, 2> = Tensor::from_data(sample_data, &gpu_device);
    let context_mask: Tensor<B, 1, Int> = Tensor::from_data(context_mask_data, &gpu_device);

    let mut context_transaction = Transaction::default();
    for context in batch.contexts {
      context_transaction = context_transaction.register(context);
    }
    let context_data = context_transaction.execute();

    let mut contexts: Vec<Tensor<B, 2>> = Vec::with_capacity(context_data.len());
    for data in context_data {
      contexts.push(Tensor::from_data(data, &gpu_device));
    }

    self.forward(samples, contexts, context_mask)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{Tch, dataset::batch::ContextBatch};

  use proptest::prelude::*;
  use burn::backend::libtorch::LibTorchDevice;

  fn batch_strategy() -> impl Strategy<Value = ContextBatch<Tch>> {
    let device = LibTorchDevice::Cuda(0);

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

      let target: Tensor<Tch, 2> =
        Tensor::from_data(target_data, &device);
      let mut contexts: Tensor<Tch, 3> = 
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
      let device = LibTorchDevice::Cuda(0);

      let config = Ip2VecConfig::new();

      let batch_size = batch.samples.dims()[0];
      let embed_dim = config.src_ip_embed_dim + config.dst_port_embed_dim + config.protocol_embed_dim;

      let model = config.init::<Tch>(&device);

      let output = model.forward(batch.samples, batch.contexts, batch.context_mask);

      prop_assert_eq!(output.embeddings.dims(), [batch_size, embed_dim]);
    }
  }
}
