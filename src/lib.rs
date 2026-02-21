#![warn(missing_docs)]
#![allow(clippy::all)]

//! `ip2vec` is a Rust library that allows for embedding IP addresses
//! into vectors, using an adaptation of the well-known Word2Vec
//! (Mikolov et al., 2013) embedding algorithm

use anyhow::{Result, anyhow};

use burn::{prelude::Backend, Tensor};
use smartcore::linalg::basic::matrix::DenseMatrix;

/// Type alias for 32-bit Cuda backend used for most tensor operations
pub type Tch = burn::backend::LibTorch<f32>;

pub mod dataset;
pub mod model;
pub mod train;
pub mod interface;
pub mod loss;

/// Convert [Tensor] to [DenseMatrix]
pub fn to_matrix<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Result<DenseMatrix<f32>> {
  let dims = tensor.dims();
  let vec = tensor.into_data().to_vec()
    .map_err(|_| anyhow!("could not get tensor data"))?;

  Ok(DenseMatrix::new(dims[0], dims[1], vec, false)?)
}
