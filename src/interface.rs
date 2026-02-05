use std::path::PathBuf;

use clap::{Parser, Args};

#[derive(Parser, Debug)]
#[command(version, about)]
pub struct Arguments {
  pub dataset: PathBuf,

  #[command(flatten)]
  pub features: Features,
  #[command(flatten)]
  pub params: TrainingParams,

  #[arg(long)]
  pub config: Option<PathBuf>,
  #[arg(long)]
  pub save: Option<PathBuf>,
  #[arg(short, long)]
  pub output: Option<PathBuf>
}

#[derive(Args, Debug)]
pub struct Features {
  #[arg(long)]
  pub src_ip: String,
  #[arg(long)]
  pub dst_ip: String,
  #[arg(long)]
  pub dst_port: String,
  #[arg(long)]
  pub protocol: String,
}

#[derive(Args, Debug)]
pub struct TrainingParams {
  #[arg(short, long)]
  pub seed: Option<u64>,
  #[arg(short, long)]
  pub epochs: Option<usize>,
  #[arg(short='r', long="ratio")]
  pub split_ratio: Option<f64>,
  #[arg(short, long)]
  pub batch_size: Option<usize>,
  #[arg(short, long)]
  pub threads: Option<usize>,
  #[arg(short, long)]
  pub learning_rate: Option<f64>,

  #[arg(short, long)]
  pub context_window: Option<usize>
}
