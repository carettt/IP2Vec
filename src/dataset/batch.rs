//! Submodule containing batching functionalities accessible through the 
//! [ContextBatcher] struct, which yields [ContextBatch]es.

use burn::{
  prelude::*,
  data::dataloader::batcher::Batcher,
  tensor::Transaction
};

use crate::dataset::{ContextItem};

/// Batch of samples derived from [IpContext] into [Tensor]s of sizes:
/// samples: `[batch_size, 34]` & context_mask `[batch_size, batch_size]`
#[derive(Debug, Clone)]
pub struct ContextBatch<B: Backend> {
  samples: Tensor<B, 2>,
  contexts: Tensor<B, 3>,
  context_mask: Tensor<B, 2>
}

/// Struct implementing [Batcher] to transform [Dataset] into [ContextBatch]es.
#[derive(Clone)]
pub struct ContextBatcher {
  context_window: usize
}

impl Default for ContextBatcher {
  fn default() -> Self {
    Self {
      context_window: 4
    }
  }
}

impl<B: Backend> Batcher<B, ContextItem, ContextBatch<B>> for ContextBatcher {
  fn batch(&self, items: Vec<ContextItem>, device: &B::Device) -> ContextBatch<B> {
    let batch_size = items.len();

    // Collect sample targets and move to GPU
    let sample_tensors = items.iter().cloned()
      .map(|i| i.target).collect::<Vec<_>>();
    let samples = Tensor::from_data(
      Tensor::stack::<2>(sample_tensors, 0).into_data()
    , device);

    let mut contexts: Tensor<B, 3> =
      Tensor::zeros([batch_size, self.context_window, samples.dims()[1]], device);
    let mut context_mask = Tensor::zeros([batch_size, self.context_window], device);

    // Batch synchronization from CPU to GPU of contexts with a transaction
    let mut transaction = Transaction::default();
    for item in items.into_iter() {
      transaction = transaction.register(item.context);
    }

    // Execute sync
    let context_data = transaction.execute();

    for (i, data) in context_data.into_iter().enumerate() {
      let context_tensor: Tensor<B, 2> = Tensor::from_data(data, device);
      let dims = context_tensor.dims();
      
      // Assign contexts to plane on 3D vector, unsqueezing for compatibility
      contexts = contexts.slice_assign(s![i], context_tensor.unsqueeze::<3>());
      // Toggle mask bits from 0 until number of contexts
      context_mask = context_mask.slice_fill(s![i, 0..dims[0]], 1);
    }

    ContextBatch { samples, contexts, context_mask }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::dataset::{
    Ip2VecDataset
  };

  use proptest::prelude::*;
  use burn::{
    backend::cuda::CudaDevice,
    data::dataset::Dataset,
    data::dataloader::batcher::Batcher,
  };

  proptest! {
    #[test]
    fn batch_shape(
      dataset in any::<Ip2VecDataset>()
    ) {
      type Backend = burn::backend::Cuda<f32, i32>;

      let mut valid = true;

      let device = CudaDevice::new(0);

      let num_batches = (dataset.len() + 1) / 2;

      let batch_size = dataset.len() / num_batches;
      let remainder = dataset.len() % num_batches;

      let mut batch_start = 0;
      let batcher = ContextBatcher::default();

      let mut batches = Vec::with_capacity(num_batches);

      for i in 0..num_batches {
        let batch_end = (batch_start + batch_size)
          + if i < remainder {1} else {0};

        let batch_items = 
          (batch_start..batch_end).map(|i| dataset.get(i))
          .collect::<Option<Vec<_>>>()
          .expect("could not get all batch items");

        let batch_len = batch_items.len();

        let batch =
          <ContextBatcher as Batcher<Backend, ContextItem, ContextBatch<Backend>>>
            ::batch(&batcher, batch_items, &device);

        batch_start = batch_end;

        valid = valid &&
          (batch.samples.dims() == [batch_len, 34]) &&
          (batch.contexts.dims() == [batch_len, 4, 34]) &&
          (batch.context_mask.dims() == [batch_len, 4]);

        batches.push(batch);
      }

      prop_assert!(valid);
    }
  }
}

