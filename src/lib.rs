#![warn(missing_docs)]
#![allow(clippy::all)]

//! `ip2vec` is a Rust library that allows for embedding IP addresses
//! into vectors, using an adaptation of the well-known Word2Vec
//! (Mikolov et al., 2013) embedding algorithm

use anyhow::{Result, anyhow};

use burn::{prelude::Backend, Tensor};
use ndarray::Array2;

/// Type alias for 32-bit Cuda backend used for most tensor operations
pub type Tch = burn::backend::LibTorch<f32>;

pub mod dataset;
pub mod model;
pub mod train;
pub mod interface;
pub mod loss;

/// Trait for applying an `Option` to a `struct` with builder-like config functions
pub trait ApplyOption: Sized {
  /// Apply `val` to `f` if `Some`, otherwise return self unchanged
  fn apply_opt<T>(self, f: impl FnOnce(Self, T) -> Self, val: Option<T>) -> Self {
    match val {
      Some(val) => f(self, val),
      None => self
    }
  }

  /// Same as [apply_opt] using mutable references instead
  fn apply_opt_mut<T>(&mut self, f: impl FnOnce(&mut Self, T) -> &mut Self, val: Option<T>) -> &mut Self {
    match val {
      Some(val) => f(self, val),
      None => self
    }
  }
}

impl<'a> ApplyOption for bhtsne::tSNE<'a, f32, Vec<f32>> {}

/// Convert [Tensor] to [DenseMatrix]
//pub fn to_matrix<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Result<DenseMatrix<f32>> {
//  let dims = tensor.dims();
//
//  Ok(DenseMatrix::new(dims[0], dims[1], vec, false)?)
//}

pub fn to_array2<B: Backend>(tensor: &Tensor<B, 2>) -> Result<Array2<f32>> {
  let [n_rows, n_cols] = tensor.dims();
  let vec = tensor.to_data().to_vec()
    .map_err(|_| anyhow!("could not get tensor data"))?;

  Ok(Array2::from_shape_vec((n_rows, n_cols), vec)?)
}

/// Save output matrix to CSV file
pub fn save_output(
  data: Vec<Vec<f32>>,
	output_path: &str,
	prefix: &str,
  features: Option<Vec<Vec<String>>>,
) -> Result<()> {
  let dim = data[0].len();
  let mut writer = csv::Writer::from_path(&output_path)?;

  let mut headers = (1..=dim)
    .map(|i| format!("{prefix}{}", i)).collect::<Vec<_>>();
  headers.push("subnet_24".to_string());
  headers.push("port".to_string());
  headers.push("protocol".to_string());

  writer.write_record(headers)?;

  for (i, row) in data.iter().enumerate() {
    let mut record: Vec<String> = row.iter().map(|v| v.to_string()).collect();

    if let Some(features) = &features {
      for feature in features {
        record.push(feature[i].clone());
      }
    }

    writer.write_record(&record)?;
  }

  writer.flush()?;
  println!("saved components to {output_path}");

  Ok(())
}

