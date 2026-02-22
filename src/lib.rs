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
