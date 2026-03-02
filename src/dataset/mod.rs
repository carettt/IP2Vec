//! Module containing [Dataset] struct and associated functions for manipulating
//! and using the dataset. Currently, only CSV is supported.

use std::{
  collections::{HashMap, HashSet}, net::Ipv4Addr, sync::Arc
};

use derivative::*;
use indexmap::IndexSet;
use anyhow::{bail, anyhow, Context, Result};
use burn::{
  data::dataset::Dataset, prelude::Backend, Tensor
};

#[cfg(test)] use proptest_derive::Arbitrary;

use crate::interface::ColumnFeatures;

pub mod batch;

#[derive(Copy, Clone)]
enum FieldKind {
  SrcIp,
  DstIp,
  DstPort,
  Protocol
}

/// Abstract Struct containing contextual information about an IP from a flow
#[derive(Derivative, Clone, PartialEq, Eq, Hash)]
#[derivative(Debug)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct IpContext {
  #[cfg_attr(test, proptest(value = "Arc::default()"))]
  #[derivative(Debug="ignore")]
  context_indices: Arc<Vec<usize>>,

  /// Source IP of [IpContext] flow
  pub src_ip: Ipv4Addr,

  dst_ip: Ipv4Addr,
  pub dst_port: u16,

  pub protocol: u8
}

impl IpContext {
  /// Helper to create new IpContext from data
  pub fn new(src_ip: Ipv4Addr, dst_ip: Ipv4Addr, dst_port: u16, protocol: u8) -> Self {
    Self {
      context_indices: Arc::default(),
      src_ip,
      dst_ip,
      dst_port,
      protocol
    }
  }

  /// Encode [IpContext] to `Vec<f32>` of size 34 for tensorization
  pub fn encode(&self) -> Vec<f32> {
    let mut data: Vec<f32> = Vec::with_capacity(34);

    let src_ip = self.src_ip.to_bits();

    for i in 0..32 {
      data.push(((src_ip >> i) & 1) as f32);
    }

    data.push(self.dst_port as f32 / 65535.0);
    data.push(self.protocol as f32 / 255.0);

    data
  }
}

/// Encoded IPContext into target tensor of shape [34] and context tensor of
/// shape [context_window, 34]
#[derive(Clone, Debug, Eq)]
pub struct ContextItem {
  target: IpContext,
  context: Vec<IpContext>
}

impl PartialEq for ContextItem {
  fn eq(&self, other: &Self) -> bool {
    self.target == other.target
  }
}

/// Struct containing indexed set of [IpContext] and a conversion map for fetching
/// a sample index for each `src_ip`.
#[derive(Clone)]
#[cfg_attr(test, derive(Debug))]
pub struct Ip2VecDataset {
  /// [IndexSet] containing all [IpContext] samples
  pub samples: IndexSet<IpContext>,
}

impl Ip2VecDataset {
  /// Helper function to generate a validated [Dataset] from a collection of
  /// [IpContext]
  fn new<T>(collection: T) -> Result<Self>
  where 
    T: IntoIterator<Item = IpContext>
  {
    let mut samples: IndexSet<IpContext> = IndexSet::from_iter(collection);

    // Initialize new sample Vec and index map
    let mut validated_samples: Vec<IpContext> =
      Vec::with_capacity(samples.len());
    let mut ip_to_idx: HashMap<Ipv4Addr, Vec<usize>> =
      HashMap::with_capacity(samples.len());

    // Remove samples without context
    let mut valid = false;

    while !valid {
      let src_set: HashSet<Ipv4Addr> = samples.iter().map(|s| s.src_ip).collect();

      validated_samples = samples.iter().cloned()
        .filter(|s| src_set.contains(&s.dst_ip))
        .collect();

      if validated_samples.len() == samples.len() {
        valid = true;
      }

      samples = validated_samples.iter().cloned().collect();
    }

    // Fill `ip_to_idx` with indexes of each sample
    for (i, sample) in samples.iter().enumerate() {
      ip_to_idx.entry(sample.src_ip).or_default().push(i);
    }

    // Wrap indices `Vec`s with an Arc to reduce memory usage
    let shared_contexts: HashMap<Ipv4Addr, Arc<Vec<usize>>> = ip_to_idx.into_iter()
      .map(|(ip, indices)| (ip, Arc::new(indices))).collect();


    // Update samples with `context_indices`
    for sample in validated_samples.iter_mut() {
      sample.context_indices = shared_contexts.get(&sample.dst_ip)
        .context("could not find context after validation")?.clone();
    }

    let samples: IndexSet<IpContext> =
      IndexSet::from_iter(validated_samples.into_iter());

    if samples.len() < 2 {
      bail!("insufficient valid samples!")
    } else {
      Ok(Self { samples })
    }
  }

  fn deserialize<R>(reader: &mut csv::Reader<R>, features: ColumnFeatures)
  -> Result<Vec<IpContext>>
  where 
    R: std::io::Read
  {
    // Construct map of column name to [FieldKind] enum
    let mut field_map: HashMap<String, FieldKind> = HashMap::new();
    field_map.insert(features.src_ip, FieldKind::SrcIp);
    field_map.insert(features.dst_ip, FieldKind::DstIp);
    field_map.insert(features.dst_port, FieldKind::DstPort);
    field_map.insert(features.protocol, FieldKind::Protocol);

    // Map row index to FieldKind
    let mut header_map: Vec<(usize, FieldKind)> = Vec::with_capacity(field_map.len());
    for (i, h) in reader.headers()?.into_iter().enumerate() {
      if let Some(kind) = field_map.get(h) {
        header_map.push((i, *kind));
      }
    }

    // Deserialize for each record
    reader
      .records()
      .map(|res| {
        let record = res?;

        let mut src_ip = None;
        let mut dst_ip = None;
        let mut dst_port = None;
        let mut protocol = None;

        // Hot-loop through each index and assign parsed value
        for &(i, field) in &header_map {
          let val = &record[i];

          match field {
            FieldKind::SrcIp => src_ip = Some(val.parse()?),
            FieldKind::DstIp => dst_ip = Some(val.parse()?),
            FieldKind::DstPort => dst_port = Some(val.parse()?),
            FieldKind::Protocol => protocol = Some(val.parse()?),
          }
        }

        // Convert `Option` to inner value or throw error for missing feature in record
        let src_ip = src_ip.ok_or_else(|| anyhow!("record missing src_ip field"))?;
        let dst_ip = dst_ip.ok_or_else(|| anyhow!("record missing dst_ip field"))?;
        let dst_port = dst_port.ok_or_else(|| anyhow!("record missing dst_port field"))?;
        let protocol = protocol.ok_or_else(|| anyhow!("record missing protocol field"))?;

        // Construct and return deserialized [IpContext]
        Ok(IpContext::new(src_ip, dst_ip, dst_port, protocol))
      })
      .collect::<Result<Vec<_>>>()
  }

  /// Function for importing CSV data set from `path`
  pub fn import_dataset<R>(reader: &mut csv::Reader<R>, features: ColumnFeatures)
  -> Result<Self>
  where 
    R: std::io::Read
  {
    // Deserialize samples
    let samples = Self::deserialize(reader, features)?;

    // Construct new dataset with deserialized samples
    Self::new(samples)
  }

  /// Function for importing CSV batch without validating for training, only for inference
  pub fn import_batch<R>(reader: &mut csv::Reader<R>, features: ColumnFeatures)
  -> Result<Self>
  where 
    R: std::io::Read
  {
    // Deserialize samples
    let samples = Self::deserialize(reader, features)?;

    Ok(Self {
      samples: samples.into_iter().collect()
    })
  }

  /// Fetch index of context pair for sample at `idx`
  pub fn get_context_indices(&self, idx: usize) -> Result<Arc<Vec<usize>>> {
    if let Some(sample) = self.samples.get_index(idx) {
      Ok(sample.context_indices.clone())
    } else {
      bail!("could not find context for sample {idx}")
    }
  }

  /// Encode all dataset samples to a single Tensor, for inference
  pub fn batch_encode<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
    let dim = self.samples[0].encode().len();
    let mut sample_buffer: Vec<f32> = Vec::with_capacity(self.samples.len() * dim);

    for sample in &self.samples {
      sample_buffer.extend(sample.encode());
    }

    Tensor::<B, 1>::from_data(sample_buffer.as_slice(), device)
      .reshape([sample_buffer.len() / dim, dim])
  }
}

impl Dataset<ContextItem> for Ip2VecDataset {
  fn get(&self, idx: usize) -> Option<ContextItem> {
    let target = self.samples.get_index(idx)?;
    let context: Vec<_> =
      self.get_context_indices(idx).ok()?.iter()
        .map(|i| self.samples.get_index(*i)
          .and_then(|s| Some::<_>(s.clone())))
        .collect::<Option<_>>()?;

    Some(ContextItem {
      target: target.clone(),
      context
    })
  }

  fn len(&self) -> usize {
    self.samples.len()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  use proptest::prelude::*;
  use csv::Reader;

  impl Arbitrary for Ip2VecDataset {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
      prop::collection::vec(
        any::<IpContext>()
        , 10..500)
      .prop_filter_map("invalid dataset", |vec| {
        Ip2VecDataset::new(vec).ok()
      })
      .boxed()
    }
  }

  fn csv_data_strategy() -> impl Strategy<Value = String> {
    prop::collection::vec(
      (
        (0u8..=255, 0u8..=255, 0u8..=255, 0u8..=255) // src_ip
          .prop_map(|(a, b, c, d)| format!("{a}.{b}.{c}.{d}")),
        (0u8..=255, 0u8..=255, 0u8..=255, 0u8..=255) // dst_ip
          .prop_map(|(a, b, c, d)| format!("{a}.{b}.{c}.{d}")),
        0u16..65535, // dst_port
        0u8..255 // protocol
      ).prop_map(|(src_ip, dst_ip, dst_port, protocol)|
          format!("{src_ip},{dst_ip},{dst_port},{protocol}")),
      0..500 // 500 rows
    ).prop_map(|rows| {
      let mut data = "IPV4_SRC_ADDR,IPV4_DST_ADDR,L4_DST_PORT,PROTOCOL\n".to_string();
      for row in rows {
        data.push_str(&row);
        data.push_str("\n");
      }

      data
    })
  }

  #[test]
  fn contextless() {
    let dataset = Ip2VecDataset {
      samples: IndexSet::new()
    };

    let result = dataset.get_context_indices(0);

    assert!(result.is_err());
  }

  proptest! {
    #[test]
    fn ipcontext_deserialization(data in csv_data_strategy()) {
      let mut reader = Reader::from_reader(data.as_bytes());
      let features = ColumnFeatures {
        src_ip: String::from("IPV4_SRC_ADDR"),
        dst_ip: String::from("IPV4_DST_ADDR"),
        dst_port: String::from("L4_DST_PORT"),
        protocol: String::from("PROTOCOL")
      };

      let res = Ip2VecDataset::import_dataset(&mut reader, features);

      prop_assert!(res.is_ok());
    }

    #[test]
    fn validation(dataset in any::<Ip2VecDataset>()) {
      for i in 0..dataset.samples.len() {
        let indices = dataset.get_context_indices(i).unwrap();

        for i in indices.iter() {
          let context = dataset.samples.get_index(*i);
          prop_assert!(context.is_some());
        }
      }
    }

    #[test]
    fn encoding(dataset in any::<Ip2VecDataset>()) {
      for i in 0..dataset.samples.len() {
        let item = dataset.get(i).unwrap();

        prop_assert_eq!(item.target.encode().len(), 34);
      }
    }
  }
}
