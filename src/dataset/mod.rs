//! Module containing [Dataset] struct and associated functions for manipulating
//! and using the dataset. Currently, only CSV is supported.

use std::{
  collections::{HashMap, HashSet},
  sync::Arc,
  net::Ipv4Addr,
};

use indexmap::IndexSet;
use anyhow::{Result, Context, bail};
use serde::Deserialize;
use burn::{
  data::dataset::Dataset
};

#[cfg(test)] use proptest_derive::Arbitrary;

pub mod batch;

/// Abstract Struct containing contextual information about an IP from a flow
#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Hash)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct IpContext {
  #[serde(skip, default)]
  #[cfg_attr(test, proptest(value = "Arc::default()"))]
  context_indices: Arc<Vec<usize>>,

  #[serde(rename = "IPV4_SRC_ADDR")]
  src_ip: Ipv4Addr,

  #[serde(rename = "IPV4_DST_ADDR")]
  dst_ip: Ipv4Addr,
  #[serde(rename = "L4_DST_PORT")]
  dst_port: u16,

  #[serde(rename = "PROTOCOL")]
  protocol: u8
}

impl IpContext {
  fn encode(&self) -> Vec<f32> {
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
#[derive(Clone, Debug)]
pub struct ContextItem {
  target: IpContext,
  context: Vec<IpContext>
}

/// Struct containing indexed set of [IpContext] and a conversion map for fetching
/// a sample index for each `src_ip`.
#[derive(Clone)]
#[cfg_attr(test, derive(Debug))]
pub struct Ip2VecDataset {
  samples: IndexSet<IpContext>,
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

  /// Function for importing CSV data set from `path`
  pub fn import_dataset(path: &str) -> Result<Self> {
    let mut reader = csv::Reader::from_path(path)?;
    let samples = reader.deserialize::<IpContext>().enumerate()
      .map(|(i, result)|
        result.map(|sample| sample)
        .context(format!("could not deserialize sample {i}")))
      .collect::<Result<Vec<_>>>()?;

    Self::new(samples)
  }

  /// Fetch index of context pair for sample at `idx`
  pub fn get_context_indices(&self, idx: usize) -> Result<Arc<Vec<usize>>> {
    if let Some(sample) = self.samples.get_index(idx) {
      Ok(sample.context_indices.clone())
    } else {
      bail!("could not find context for sample {idx}")
    }
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

      for row in reader.deserialize::<IpContext>() {
        prop_assert!(row.is_ok());
      }
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

    //#[test]
    //fn target_encoding(dataset in any::<Ip2VecDataset>()) {
    //  for i in 0..dataset.samples.len() {
    //    let item = dataset.get(i).unwrap();

    //    prop_assert_eq!(item.target.shape().dims, vec![34]);
    //  }
    //}

    //#[test]
    //fn context_encoding(dataset in any::<Ip2VecDataset>()) {
    //  for i in 0..dataset.samples.len() {
    //    let sample = dataset.samples.get_index(i).unwrap();
    //    let item = dataset.get(i).unwrap();

    //    prop_assert_eq!(
    //      item.context.shape().dims,
    //      vec![sample.context_indices.len(), 34]
    //    );
    //  }
    //}
  }
}
