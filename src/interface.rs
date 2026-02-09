//! Module containing functionality for CLI and configuration saving/loading

use std::path::PathBuf;

use clap::{Parser, Args};

/// IP2Vec - IP embedding neural network trainer
#[derive(Parser, Debug)]
#[command(version, about)]
pub struct Arguments {
  /// CSV dataset filepath
  pub dataset: PathBuf,

  /// Required feature column names
  #[command(flatten)]
  pub features: Features,
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

/// Struct to contain dataset feature column names
#[derive(Args, Debug)]
pub struct Features {
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
