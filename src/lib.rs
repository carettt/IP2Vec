#![warn(missing_docs)]
#![allow(clippy::all)]

//! `ip2vec` is a Rust library that allows for embedding IP addresses
//! into vectors, using an adaptation of the well-known Word2Vec
//! (Mikolov et al., 2013) embedding algorithm

/// Type alias for 32-bit Cuda backend used for most tensor operations
pub type Backend = burn::backend::Cuda<f32, i32>;

pub mod dataset;
pub mod model;
