//! Module containing functionality for CLI and configuration saving/loading

use std::{net::Ipv4Addr, path::PathBuf};

use clap::{Parser, Args, Subcommand};
use serde::{Deserialize, Serialize};

/// ip2vec - IP embedding neural network
#[derive(Parser, Debug)]
#[command(version, about)]
pub struct InferenceArgs {
  /// Path to configuration folder containing model.mpk and config.json
  #[arg(short, long)]
  pub store: PathBuf,

  /// Whether to save PCA projection to CSV file
  #[arg(long)]
  pub pca: bool,

  /// Input subcommand
  #[command(subcommand)]
  pub command: Commands
}

/// ip2vec-trainer - IP embedding neural network trainer
#[derive(Parser, Debug)]
#[command(version, about)]
pub struct TrainerArgs {
  /// CSV dataset filepath
  #[arg(required_unless_present = "store")]
  pub dataset: Option<PathBuf>,

  /// Required feature column names
  #[command(flatten)]
  pub features: Option<ColumnFeatures>,
  /// Optional parameters for trainer
  #[command(flatten)]
  pub params: TrainingParams,

  /// Folder path to save configuration and experiment logs
  #[arg(long)]
  pub store: Option<PathBuf>
}

/// Enum containing inference subcommands
#[derive(Subcommand, Debug)]
pub enum Commands {
  /// Single sample inference
  Single {
    /// Input data features
    #[command(flatten)]
    features: DataFeatures
  },

  /// File batch inference
  Batch {
    /// Path to batch
    file: PathBuf,
  }
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
#[derive(Args, Debug, Clone, Serialize, Deserialize)]
#[group(requires="src_ip", requires="dst_ip", requires="dst_port", requires="protocol")]
pub struct ColumnFeatures {
  /// Column name for source IP
  #[arg(long, required=false)]
  pub src_ip: String,
  /// Column name for destination IP
  #[arg(long, required=false)]
  pub dst_ip: String,
  /// Column name for destination port
  #[arg(long, required=false)]
  pub dst_port: String,
  /// Column name for protocol
  #[arg(long, required=false)]
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
  pub context_window: Option<usize>,
  /// Amount of negative context examples per postive context
  #[arg(short, long)]
  pub neg_multiplier: Option<usize>
}
