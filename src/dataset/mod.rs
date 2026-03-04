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
#[cfg(test)] use proptest::prelude::*;
use rand::{distr::{Distribution, StandardUniform}, seq::IteratorRandom, Rng};

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
#[derive(Derivative, Clone)]
#[derivative(Debug, PartialEq, Eq, Hash)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct Sample {
  //#[cfg_attr(test, proptest(value = "Arc::default()"))]
  //#[derivative(Debug="ignore")]
  //context_indices: Arc<Vec<usize>>,
  
  /// Source IP of flow
  src_ip: Arc<Ipv4Addr>,
  /// Destination IP of flow
  dst_ip: Arc<Ipv4Addr>,
  /// Destination port of flow
  dst_port: Arc<u16>,
  /// Internet protocol byte of flow
  protocol: Arc<u8>,
}

impl Sample {
  /// Helper to create new [Sample] from data
  fn new(
    src_ip: Arc<Ipv4Addr>,
		dst_ip: Arc<Ipv4Addr>,
		dst_port: Arc<u16>,
		protocol: Arc<u8>,
  ) -> Self {
    Self {
      src_ip,
      dst_ip,
      dst_port,
      protocol
    }
  }

  /// Encode [Sample] to `Vec<f32>` of size 34 for tensorization
  pub fn encode(&self) -> Vec<f32> {
    let mut data: Vec<f32> = Vec::with_capacity(34);

    let src_ip = self.src_ip.to_bits();

    for i in 0..32 {
      data.push(((src_ip >> i) & 1) as f32);
    }

    data.push(*self.dst_port as f32 / 65535.0);
    data.push(*self.protocol as f32 / 255.0);

    data
  }

  pub fn is_context(&self, other: &Self) -> bool {
    if other.src_ip == self.dst_ip {
      return true;
    }
    if other.dst_port == self.dst_port {
      return true;
    }
    if other.protocol == self.protocol {
      return true;
    }

    false
  }
}

/// Encoded IPContext into target tensor of shape [34] and context tensor of
/// shape [context_window, 34]
#[derive(Clone, Debug)]
pub struct ContextItem {
  target: Arc<Sample>,
  context: SampleContext
}

impl PartialEq for ContextItem {
  fn eq(&self, other: &Self) -> bool {
    self.target == other.target
  }
}

#[derive(Default, Debug, Clone)]
struct SampleContext {
  positive: HashSet<Arc<Sample>>, // [n, n*5]
  negative: HashSet<Arc<Sample>>, // [n, n*15]
}

/// Struct containing indexed set of [Sample] and a conversion map for fetching
/// a sample index for each `src_ip`.
#[derive(Clone)]
#[cfg_attr(test, derive(Debug))]
pub struct Ip2VecDataset {
  /// `IndexSet` containing all [Sample]s
  pub samples: IndexSet<Arc<Sample>>, // [n]

  hosts: HashSet<Arc<Ipv4Addr>>, // [n]
  ports: HashSet<Arc<u16>>, // [n]
  protocols: HashSet<Arc<u8>>, // [n]

  contexts: Option<HashMap<Arc<Sample>, SampleContext>> // n*15 per n
}

impl Ip2VecDataset {
  /// Create new empty dataset & populate data structures
  fn new(data: Vec<(Ipv4Addr, Ipv4Addr, u16, u8)>) -> Self {
    let n = data.len();

    let mut dataset = Self {
      samples: IndexSet::with_capacity(n),

      hosts: HashSet::with_capacity(n),
      ports: HashSet::with_capacity(n),
      protocols: HashSet::with_capacity(n),
      
      contexts: None
    };

    for sample_data in data {
      let (src_ip, dst_ip, port, protocol) = sample_data;

      // Allocate pointers
      let src_ip = Arc::new(src_ip);
      let dst_ip = Arc::new(dst_ip);
      let port = Arc::new(port);
      let protocol = Arc::new(protocol);

      let sample = Arc::new(Sample::new(
          Arc::clone(&src_ip),
          Arc::clone(&dst_ip),
          Arc::clone(&port),
          Arc::clone(&protocol)
      ));

      // Populate data structures
      dataset.hosts.insert(Arc::clone(&src_ip));
      dataset.hosts.insert(Arc::clone(&dst_ip));
      dataset.ports.insert(Arc::clone(&port));
      dataset.protocols.insert(Arc::clone(&protocol));

      dataset.samples.insert(Arc::clone(&sample));
    }

    dataset
  }

  /// Function to derive and populate `contexts`
  fn preprocess<R>(&mut self, rng: &mut R, context_window: usize, neg_multiplier: usize) 
  -> Result<()>
  where 
    R: Rng
  {
    // Allocate `HashMap`
    let mut contexts = HashMap::with_capacity(self.samples.len());

    // Loop through samples
    for sample in &self.samples.clone() {
      // Take `context_window` positive samples from dataset
      let positive: HashSet<Arc<Sample>> = self.samples.iter()
        .filter(|other| sample.is_context(other))
        .cloned()
        .take(context_window)
        .collect();

      // Randomly sample `context_window * neg_multiplier` negative samples from dataset
      let negative: HashSet<Arc<Sample>> = self.samples.iter()
        .filter(|other| !sample.is_context(other))
        .cloned()
        .sample(rng, context_window * neg_multiplier)
        .into_iter().collect();

      if positive.len() != context_window || negative.len() != (context_window * neg_multiplier) {
        // Prune samples with not enough context
        self.samples.shift_remove(sample);
      } else {
        // Populate contexts
        contexts.insert(Arc::clone(sample), SampleContext { positive, negative });
      }
    }

    if self.samples.len() < 1 {
      bail!("no samples left after preprocessing");
    }

    self.contexts = Some(contexts);

    Ok(())
  }

  fn deserialize<R>(reader: &mut csv::Reader<R>, features: ColumnFeatures)
  -> Result<Ip2VecDataset>
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
    let sample_data = reader
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

        // Construct and return deserialized [Sample]
        Ok((src_ip, dst_ip, dst_port, protocol))
      })
      .collect::<Result<_>>()?;

    Ok(Self::new(sample_data))
  }

  /// Fetch context for sample
  fn get_context(&self, sample: &Arc<Sample>) -> Option<SampleContext> {
    if let Some(contexts) = &self.contexts {
      contexts.get(sample).cloned()
    } else {
      None
    }
  }

  /// Encode all dataset samples to a single Tensor, for inference
  pub fn batch_encode<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> {
    let dim = self.samples.iter().next().unwrap().encode().len();
    let mut sample_buffer: Vec<f32> = Vec::with_capacity(self.samples.len() * dim);

    for sample in &self.samples {
      sample_buffer.extend(sample.encode());
    }

    Tensor::<B, 1>::from_data(sample_buffer.as_slice(), device)
      .reshape([sample_buffer.len() / dim, dim])
  }

  /// Get all features from dataset as `String`s in format [subnets, ports, protocols]
  pub fn get_feature_strings(&self) -> [Vec<String>; 3] {
    let dataset_size = self.samples.len();

    let mut subnets = Vec::with_capacity(dataset_size);
    let mut ports = Vec::with_capacity(dataset_size);
    let mut protocols = Vec::with_capacity(dataset_size);

    for sample in &self.samples {
      let subnet = sample.src_ip.octets().into_iter()
        .take(3)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(".");
      
      subnets.push(format!("{subnet}.0/24"));
      ports.push(sample.dst_port.to_string());
      protocols.push(sample.protocol.to_string());
    }
    
    [subnets, ports, protocols]
  }
}

impl Dataset<ContextItem> for Ip2VecDataset {
  fn get(&self, idx: usize) -> Option<ContextItem> {
    let target = self.samples.get_index(idx)?;
    let context = self.get_context(target)?;

    Some(ContextItem {
      target: Arc::clone(target),
      context
    })
  }

  fn len(&self) -> usize {
    self.samples.len()
  }
}

#[cfg(test)]
impl Arbitrary for Ip2VecDataset {
  type Parameters = bool;
  type Strategy = BoxedStrategy<Self>;

  fn arbitrary_with(preprocessing: Self::Parameters) -> Self::Strategy {
    prop::collection::vec((
      any::<Ipv4Addr>(),
      any::<Ipv4Addr>(),
      any::<u16>(),
      any::<u8>()
    ), 10..500)
    .prop_filter_map("failed to generate dataset", move |vec| {
      let mut dataset = Ip2VecDataset::new(vec);

      if preprocessing {
        let mut rng = rand::rng();

        if let Ok(_) = dataset.preprocess(&mut rng, 5, 3) {
          Some(dataset)
        } else {
          None
        }
      } else {
        Some(dataset)
      }
    })
    .boxed()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::Tch;

  use burn::backend::libtorch::LibTorchDevice;
  use csv::Reader;

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
    fn contextless(dataset in any::<Ip2VecDataset>()) {
      let sample = dataset.samples.iter().next().unwrap();
      let context = dataset.get_context(&sample);

      prop_assert!(context.is_none());
    }

    #[test]
    fn sample_deserialization(data in csv_data_strategy()) {
      let mut reader = Reader::from_reader(data.as_bytes());
      let features = ColumnFeatures {
        src_ip: String::from("IPV4_SRC_ADDR"),
        dst_ip: String::from("IPV4_DST_ADDR"),
        dst_port: String::from("L4_DST_PORT"),
        protocol: String::from("PROTOCOL")
      };

      let res = Ip2VecDataset::deserialize(&mut reader, features);

      prop_assert!(res.is_ok());
    }

    #[test]
    fn validation(dataset in any_with::<Ip2VecDataset>(true)) {
      let opt = dataset.samples.iter()
        .map(|s| dataset.get_context(s))
        .collect::<Option<Vec<_>>>();

      prop_assert!(opt.is_some());
    }

    #[test]
    fn encoding(dataset in any::<Ip2VecDataset>()) {
      let device = LibTorchDevice::Cuda(0);

      let t = dataset.batch_encode::<Tch>(&device);

      prop_assert_eq!(t.dims(), [dataset.samples.len(), 34]);
    }
  }
}
