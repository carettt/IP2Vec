//! Module containing [Dataset] struct and associated functions for manipulating
//! and using the dataset. Currently, only CSV is supported.

use std::{
  collections::HashMap,
  net::Ipv4Addr,
};

use indexmap::IndexSet;
use anyhow::{Result, Context};
use thousands::Separable;
use serde::Deserialize;

#[cfg(test)] use proptest_derive::Arbitrary;


/// Struct containing contextual information about an IP from a flow
#[derive(Debug, Deserialize, PartialEq, Eq, Hash)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct IpContext{
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
#[cfg_attr(test, derive(Debug))]
pub struct Dataset {
  samples: IndexSet<IpContext>,
  ip_to_idx: HashMap<Ipv4Addr, usize>
}

impl Dataset {
  /// Helper function to generate a non-validated [Dataset] from a collection of
  /// [IpContext]
  fn new<T>(collection: T) -> Self
  where 
    T: IntoIterator<Item = IpContext>
  {
    let mut samples = IndexSet::new();
    let mut ip_to_idx: HashMap<Ipv4Addr, usize> = HashMap::new();
    let mut i = 0;

    for sample in collection {
      if samples.insert(sample) {
        ip_to_idx.insert(samples[i].src_ip, i);
        i += 1;
      }
    }

    Self { samples, ip_to_idx }
  }

  /// Validation function to ensure all samples have context. Context is defined
  /// as a sample with `src` and `dst` fields mirrored.
  fn validate_samples(&mut self) {
    let mut contextless: Vec<usize> = Vec::new();

    for i in 0..self.samples.len() {
      if self.get_context_idx(i).is_err() {
        contextless.push(i);
      }
    }

    contextless.reverse();
    for i in &contextless {
      self.samples.swap_remove_index(*i);
    }

    println!("removed {} contextless samples from dataset",
      contextless.len().separate_with_commas());
  }

  /// Function for importing CSV data set from `path`
  pub fn import_dataset(path: &str) -> Result<Self> {
    let mut reader = csv::Reader::from_path(path)?;
    let samples =
      reader.deserialize().collect::<Result<Vec<_>, csv::Error>>()?;

    let mut dataset = Self::new(samples);
    dataset.validate_samples();

    Ok(dataset)
  }

  /// Fetch index of context pair for sample at `idx`
  pub fn get_context_idx(&self, idx: usize) -> Result<&usize> {
    self.ip_to_idx.get(&self.samples[idx].dst_ip)
      .context(format!("could not find context for sample {idx}"))
  }
}

#[cfg(test)]
mod dataset_tests {
  use super::*;

  use proptest::prelude::*;
  use csv::Reader;

  impl Arbitrary for Dataset {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
      prop::collection::vec(
        any::<IpContext>()
        , 0..500)
      .prop_map(|vec| {
        Dataset::new(vec)
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

  proptest! {
    #[test]
    fn ipcontext_deserialization(data in csv_data_strategy()) {
      let mut reader = Reader::from_reader(data.as_bytes());

      for row in reader.deserialize::<IpContext>() {
        prop_assert!(row.is_ok());
      }
    }

    #[test]
    fn contextless(mut dataset in any::<Dataset>()) {
      dataset.validate_samples();

      for i in 0..dataset.samples.len() {
        let result = dataset.get_context_idx(i);

        prop_assert!(result.is_ok());
      }
    }
  }
}
