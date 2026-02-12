//! Module containing functionality for CLI and configuration saving/loading

use std::{net::Ipv4Addr, path::PathBuf};

use clap::{Parser, Args};

/// ip2vec - IP embedding neural network
#[derive(Parser, Debug)]
#[command(version, about)]
pub struct InferenceArgs {
  /// Path to configuration folder containing model.mpk and config.json
  #[arg(short, long)]
  pub config: PathBuf,

  /// Input data features
  #[command(flatten)]
  pub features: DataFeatures
}

/// ip2vec-trainer - IP embedding neural network trainer
#[derive(Parser, Debug)]
#[command(version, about)]
pub struct TrainerArgs {
  /// CSV dataset filepath
  pub dataset: PathBuf,

  /// Required feature column names
  #[command(flatten)]
  pub features: ColumnFeatures,
  /// Optional parameters for trainer
  #[command(flatten)]
  pub params: TrainingParams,

  /// Config file path
  #[arg(long)]
  pub config: Option<PathBuf>,
  /// Whether to save configuration to path
  #[arg(long)]
  pub save: Option<PathBuf>,
  /// Folder path to save configuration and experiment logs
  #[arg(short, long)]
  pub output: Option<PathBuf>
}

/// Struct to contain input flow features for inference embedding
#[derive(Args, Debug)]
pub struct DataFeatures {
  /// Source IP
  #[arg(long)]
  pub src_ip: Ipv4Addr,
  /// Destination IP
  #[arg(long)]
  pub dst_ip: Ipv4Addr,
  /// Destination Port
  #[arg(long)]
  pub dst_port: u16,
  /// Protocol
  #[arg(long)]
  pub protocol: u8
}

/// Struct to contain dataset feature column names
#[derive(Args, Debug)]
pub struct ColumnFeatures {
  /// Column name for source IP
  #[arg(long)]
  pub src_ip: String,
  /// Column name for destination IP
  #[arg(long)]
  pub dst_ip: String,
  /// Column name for destination port
  #[arg(long)]
  pub dst_port: String,
  /// Column name for protocol
  #[arg(long)]
  pub protocol: String,
}

/// Struct to contain training configuration parameters
#[derive(Args, Debug)]
pub struct TrainingParams {
  /// RNG seed
  #[arg(short, long)]
  pub seed: Option<u64>,
  /// Epochs to train for
  #[arg(short, long)]
  pub epochs: Option<usize>,
  /// Percentage of dataset to use for training vs. validation
  #[arg(short='r', long="ratio")]
  pub split_ratio: Option<f64>,
  /// Batch size
  #[arg(short, long)]
  pub batch_size: Option<usize>,
  /// CPU threads
  #[arg(short, long)]
  pub threads: Option<usize>,
  /// Learning rate for gradient adjustment
  #[arg(short, long)]
  pub learning_rate: Option<f64>,

  /// Amount of context examples per sample
  #[arg(short, long)]
  pub context_window: Option<usize>
}
