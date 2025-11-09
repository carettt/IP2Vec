//! Module containing training functionality such as metrics

use burn::{
  prelude::*,
  train::metric::{
    Adaptor, CudaMetric, LossInput
  }
};

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
