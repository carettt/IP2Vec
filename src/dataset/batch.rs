//! Submodule containing batching functionalities accessible through the 
//! [ContextBatcher] struct, which yields [ContextBatch]es.

use burn::{prelude::*, data::dataloader::batcher::Batcher};

use crate::dataset::IpContext;

/// Batch of samples derived from [IpContext] into [Tensor]s of sizes:
/// samples: `[batch_size, 34]` & context_mask `[batch_size, batch_size]`
#[derive(Debug, Clone)]
pub struct ContextBatch<B: Backend> {
  samples: Tensor<B, 2>,
  context_mask: Tensor<B, 2>
}

/// Struct implementing [Batcher] to transform [Dataset] into [ContextBatch]es.
#[derive(Default, Clone)]
pub struct ContextBatcher {}

impl<B: Backend> Batcher<B, IpContext, ContextBatch<B>> for ContextBatcher {
  fn batch(&self, items: Vec<IpContext>, device: &B::Device) -> ContextBatch<B> {
    let batch_size = items.len();

    let mut samples: Vec<f32> = Vec::with_capacity(batch_size * 34);
    let mut context_mask = Tensor::full([batch_size, batch_size], false, device);

    for (i, item) in items.iter().enumerate() {
      for idx in item.context_indices.iter() {
        context_mask = context_mask.slice_fill([i..i+1, *idx..*idx+1], true);
      }

      let src_ip = item.src_ip.to_bits();
      let src_ip_bits = (0..32).map(|j| {
        ((src_ip >> j) & 1) as f32
      }).collect::<Vec<f32>>();

      samples.extend(src_ip_bits);
      samples.push(item.dst_port as f32 / 65535.0);
      samples.push(item.protocol as f32 / 255.0);
    }

    let samples =
      Tensor::from_data(TensorData::new(samples, [batch_size, 34]), device);

    println!("{samples}");
    println!("{context_mask}");

    ContextBatch { samples, context_mask }
  }
}
