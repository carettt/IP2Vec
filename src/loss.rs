//! Module containing logic for implementation of NEG loss from the Word2Vec
//! paper (Mikolov et al., 2013)

use burn::{
  config::Config,
  module::{Content, DisplaySettings, Ignored, Module, ModuleDisplay},
  nn::loss::Reduction,
  tensor::{Tensor, activation::log_sigmoid, backend::Backend},
};

/// Configuration for [NegEmbeddingLoss]
#[derive(Config, Debug)]
pub struct NegEmbeddingLossConfig {
  /// Specifies the reduction to be applied to the output tensor
  #[config(default = "Reduction::Mean")]
  pub reduction: Reduction,
}

impl NegEmbeddingLossConfig {
  /// Initialize NEG embedding loss struct
  pub fn init(self) -> NegEmbeddingLoss {
    NegEmbeddingLoss::new(self.reduction)
  }
}

/// Implementation of NEG loss (Mikolov et al., 2013)
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct NegEmbeddingLoss {
  /// Reduction technique applied to loss tensor
  pub reduction: Ignored<Reduction>,
}

impl ModuleDisplay for NegEmbeddingLoss {
  fn custom_settings(&self) -> Option<DisplaySettings> {
    DisplaySettings::new()
      .with_new_line_after_attribute(false)
      .optional()
  }

  fn custom_content(&self, content: Content) -> Option<Content> {
    content
      .add("reduction", format!("{:?}", &self.reduction.0).as_str())
      .optional()
  }
}

impl NegEmbeddingLoss {
  fn new(reduction: Reduction) -> Self {
    NegEmbeddingLoss {
      reduction: Ignored(reduction),
    }
  }

  /// Calculate loss without any Reduction, returns tensor of shape [batch_size]
  pub fn forward_no_reduction<B: Backend>(
    &self,
    target: Tensor<B, 2>,
    positive: Tensor<B, 3>,
    negative: Tensor<B, 3>,
  ) -> Tensor<B, 1> {
    let unsqueezed_target = target.unsqueeze_dim(1);

    let positive_similarity: Tensor<B, 1> = unsqueezed_target
      .clone()
      .matmul(positive.swap_dims(1, 2)) // [batch_size, 1, context_window]
      .squeeze_dim::<2>(1) // [batch_size, context_window]
      .sum_dim(1) // [batch_size, 1]
      .squeeze_dim::<1>(1); // [batch_size]
    let negative_similarity: Tensor<B, 1> = unsqueezed_target
      .matmul(negative.swap_dims(1, 2)) // [batch_size, 1, context_window * neg_mult]
      .squeeze_dim::<2>(1) // [batch_size, context_window * neg_mult]
      .sum_dim(1) // [batch_size, 1]
      .squeeze_dim::<1>(1); // [batch_size]

    (log_sigmoid(positive_similarity) + log_sigmoid(-negative_similarity)).neg()
  }

  /// Calculate loss with Reduction from config (default: Mean), returns tensor of shape [1]
  pub fn forward<B: Backend>(
    &self,
    target: Tensor<B, 2>,
    positive: Tensor<B, 3>,
    negative: Tensor<B, 3>,
  ) -> Tensor<B, 1> {
    let loss = self.forward_no_reduction(target, positive, negative);

    match self.reduction.0 {
      Reduction::Sum => loss.sum(),
      Reduction::Mean | Reduction::Auto => loss.mean(),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::Tch;

  use burn::backend::libtorch::LibTorchDevice;

  use proptest::prelude::*;

  fn input_strategy<B: Backend>(
    device: &B::Device,
  ) -> impl Strategy<Value = (Tensor<B, 2>, Tensor<B, 3>, Tensor<B, 3>)> {
    (
      prop::collection::vec(0f64..=1.0, 150 * 4)
        .prop_map(|data| Tensor::<B, 1>::from_data(data.as_slice(), device).reshape([4, 150])),
      prop::collection::vec(0f64..=1.0, 150 * 5 * 4)
        .prop_map(|data| Tensor::<B, 1>::from_data(data.as_slice(), device).reshape([4, 5, 150])),
      prop::collection::vec(0f64..=1.0, 150 * 25 * 4)
        .prop_map(|data| Tensor::<B, 1>::from_data(data.as_slice(), device).reshape([4, 25, 150])),
    )
  }

  proptest! {
    #[test]
    fn forward_no_reduction(input_tensors in input_strategy::<Tch>(&LibTorchDevice::Cuda(0))) {
      let (target, positive, negative) = input_tensors;

      let loss = NegEmbeddingLossConfig::new()
        .init()
        .forward_no_reduction(target, positive, negative);

      eprintln!("{loss}");

      assert_eq!(loss.dims()[0], 4);
    }

    #[test]
    fn forward(input_tensors in input_strategy::<Tch>(&LibTorchDevice::Cuda(0))) {
      let (target, positive, negative) = input_tensors;

      let loss = NegEmbeddingLossConfig::new()
        .init()
        .forward(target, positive, negative); // Mean Reduction

      eprintln!("{loss}");

      assert_eq!(loss.dims()[0], 1);
    }
  }
}
