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

#[cfg(test)] use proptest_derive::Arbitrary;

pub mod batch;

/// Struct containing contextual information about an IP from a flow
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


/// Struct containing indexed set of [IpContext] and a conversion map for fetching
/// a sample index for each `src_ip`.
#[cfg_attr(test, derive(Debug, Default))]
pub struct Ip2VecDataset {
  samples: IndexSet<IpContext>
}

impl Ip2VecDataset {
  /// Helper function to generate a non-validated [Dataset] from a collection of
  /// [IpContext]
  fn new<T>(collection: T) -> Self
  where 
    T: IntoIterator<Item = IpContext>
  {
    Self {
      samples: IndexSet::from_iter(collection)
    }
  }

  /// Validation function to ensure all samples have context. Context is defined
  /// as a sample with `src` and `dst` fields mirrored.
  fn validate_samples(&mut self) {
    // Get list of all `src_ip`s
    let src_set: HashSet<Ipv4Addr> =
      self.samples.iter().map(|sample| sample.src_ip).collect();

    // Initialize new sample Vec and index map
    let mut validated_samples: Vec<IpContext> =
      Vec::with_capacity(self.samples.len());
    let mut ip_to_idx: HashMap<Ipv4Addr, Vec<usize>> =
      HashMap::with_capacity(self.samples.len());

    // Validate each sample's `dst_ip` has a corresponding `src` and append to
    // index map `Vec`
    for sample in self.samples.iter() {
      if src_set.contains(&sample.dst_ip) {
        let idx = validated_samples.len();
        ip_to_idx.entry(sample.src_ip).or_default().push(idx);
        validated_samples.push(sample.clone());
      }
    }

    // Wrap indices `Vec`s with an Arc to reduce memory usage
    let shared_contexts: HashMap<Ipv4Addr, Arc<Vec<usize>>> = ip_to_idx.into_iter()
      .map(|(ip, indices)| (ip, Arc::new(indices))).collect();

    // Update samples with `context_indices`
    for sample in validated_samples.iter_mut() {
      sample.context_indices = shared_contexts.get(&sample.src_ip)
        .expect("could not find context after validation").clone();
    }

    // Update dataset with validated samples
    self.samples = IndexSet::from_iter(validated_samples.into_iter());
  }

  /// Function for importing CSV data set from `path`
  pub fn import_dataset(path: &str) -> Result<Self> {
    let mut reader = csv::Reader::from_path(path)?;
    let samples = reader.deserialize::<IpContext>().enumerate()
      .map(|(i, result)|
        result.map(|sample| sample)
        .context(format!("could not deserialize sample {i}")))
      .collect::<Result<Vec<_>>>()?;

    let mut dataset = Self::new(samples);
    dataset.validate_samples();

    Ok(dataset)
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
        , 0..500)
      .prop_map(|vec| {
        Ip2VecDataset::new(vec)
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
    fn validation(mut dataset in any::<Ip2VecDataset>()) {
      dataset.validate_samples();

      for i in 0..dataset.samples.len() {
        let result = dataset.get_context_indices(i);

        prop_assert!(result.is_ok());
      }
    }
  }
}
