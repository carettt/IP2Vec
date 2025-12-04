//! Submodule containing batching functionalities accessible through the 
//! [ContextBatcher] struct, which yields [ContextBatch]es.

use burn::{
  prelude::*,
  data::dataloader::batcher::Batcher,
};

use crate::dataset::ContextItem;

/// Batch of samples derived from [IpContext] into [Tensor]s
#[derive(Debug, Clone)]
pub struct ContextBatch<B: Backend> {
  /// 2D tensor of shape [batch_size, 34]
  pub samples: Tensor<B, 2>,
  /// 3D tensor of shape [batch_size, context_window, 34]
  pub contexts: Tensor<B, 3>,
  /// 2D tensor of shape [batch_size, context_window]
  pub context_mask: Tensor<B, 2, Int>
}

impl<B: Backend> ContextBatch<B> {
  /// Helper function to create a new batch for testing (shouldn't be used)
  pub fn new(
    samples: Tensor<B, 2>,
    contexts: Tensor<B, 3>,
    context_mask: Tensor<B, 2, Int>
  ) -> Self {
    Self { samples, contexts, context_mask }
  }
}

/// Struct implementing [Batcher] to transform [Dataset] into [ContextBatch]es.
#[derive(Clone)]
pub struct ContextBatcher {
  context_window: usize
}

impl ContextBatcher {
  /// Create new [ContextBatcher] with `context_window` for batch generation
  pub fn new(context_window: usize) -> Self {
    Self { context_window }
  }
}

impl Default for ContextBatcher {
  fn default() -> Self {
    Self::new(15)
  }
}

impl<B: Backend> Batcher<B, ContextItem, ContextBatch<B>> for ContextBatcher {
  fn batch(&self, items: Vec<ContextItem>, device: &B::Device) -> ContextBatch<B> {
    let batch_size = items.len();

    let positive_count = self.context_window / 3;
    let mut negative_count = self.context_window - positive_count;

    let mut sample_buffer = Vec::with_capacity(items.len());
    let mut context_buffer = Vec::with_capacity(items.len() * self.context_window);
    let mut mask_buffer = Vec::with_capacity(items.len() * self.context_window);

    for item in items.iter() {
      let sample = item.target.encode();

      let context_num = item.context.len();

      if context_num < positive_count {
        negative_count += positive_count - context_num;
      }

      sample_buffer.extend(sample);


      // Positive samples
      for i in 0..context_num {
        context_buffer.extend(item.context[i].encode());
      }
      mask_buffer.extend(vec![1; context_num]);

      // Negative samples
      let negative_samples = items.iter().cloned()
        .filter(|i| (i != item) && !(item.context.contains(&i.target)))
        .take(negative_count)
        .flat_map(|i| i.target.encode())
        .collect::<Vec<_>>();
      context_buffer.extend(negative_samples);
      mask_buffer.extend(vec![-1; negative_count]);

      //if context_num < self.context_window {
      //  let remainder = self.context_window - context_num;


      //  context_buffer.extend(vec![0.0; remainder * sample_size]);
      //  mask_buffer.extend(vec![0; remainder]);
      //} else if context_num > self.context_window {
      //  let remainder = context_num - self.context_window;

      //  context_buffer.truncate(context_buffer.len() - (remainder * sample_size));
      //  mask_buffer.truncate(mask_buffer.len() - remainder);
      //}
    }

    let sample_dims = [batch_size, sample_buffer.len() / batch_size];

    let samples =
      Tensor::<B, 1>::from_floats(sample_buffer.as_slice(), device)
        .reshape(sample_dims);

    let contexts = Tensor::<B, 1>::from_floats(context_buffer.as_slice(), device)
      .reshape([batch_size, self.context_window, sample_dims[1]]);
    let context_mask = Tensor::<B, 1, Int>::from_ints(mask_buffer.as_slice(), device)
      .reshape([batch_size, self.context_window]);

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
    backend::libtorch::LibTorchDevice,
    data::dataset::Dataset,
    data::dataloader::batcher::Batcher,
  };

  proptest! {
    #[test]
    fn batch_shape(
      dataset in any::<Ip2VecDataset>()
    ) {
      type Backend = burn::backend::LibTorch<f32>;

      let mut valid = true;

      let device = LibTorchDevice::Cuda(0);

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


    #[test]
    fn batch_device(
      dataset in any::<Ip2VecDataset>()
    ) {
      type Backend = burn::backend::LibTorch<f32>;

      let mut valid = true;

      let device = LibTorchDevice::Cuda(0);

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

        let batch =
          <ContextBatcher as Batcher<Backend, ContextItem, ContextBatch<Backend>>>
            ::batch(&batcher, batch_items, &device);

        batch_start = batch_end;

        //eprintln!("{:?}", batch.samples.device());
        //eprintln!("{:?}", batch.contexts.device());
        //eprintln!("{:?}", batch.context_mask.device());

        valid = valid &&
          (batch.samples.device() == device) &&
          (batch.contexts.device() == device) &&
          (batch.context_mask.device() == device);

        batches.push(batch);
      }

      prop_assert!(valid);
    }
  }
}

