//! Module containing [Ip2Vec] model struct, including input projection and 
//! hidden layer (which will be extracted for embeddings). The model should be
//! constructed with [Ip2VecConfig] which provides default embed dimension params.

use burn::prelude::*;

/// [Ip2Vec] builder with default settings of 100-d `src_ip`, and 25-d `dst_port`
/// and `protocol` projection layers; totaling to a 150-d embedding tensor.
#[derive(Config, Debug)]
pub struct Ip2VecConfig {
  #[config(default="100")]
  src_ip_embed_dim: usize,
  #[config(default="25")]
  dst_port_embed_dim: usize,
  #[config(default="25")]
  protocol_embed_dim: usize
}

impl Ip2VecConfig {
  /// [Ip2Vec] initializer, can be overriden using `with_<CONFIG>` builder functions.
  pub fn init<B: Backend>(&self, device: &B::Device) -> Ip2Vec<B> {
    let embed_dim =
      self.src_ip_embed_dim + self.dst_port_embed_dim + self.protocol_embed_dim;

    Ip2Vec {
      activation: nn::Relu::new(),

      src_ip_input: nn::LinearConfig::new(32, self.src_ip_embed_dim).init(device),
      dst_port_input: nn::LinearConfig::new(1, self.dst_port_embed_dim).init(device),
      protocol_input: nn::LinearConfig::new(1, self.protocol_embed_dim).init(device),

      hidden: nn::LinearConfig::new(embed_dim, embed_dim).init(device)
    }
  }
}

/// Neural network with 3 input tensors which are projected into a 150-d latent
/// space, which are embedded into a tensor of shape [batch_size, 150].
#[derive(Module, Debug)]
pub struct Ip2Vec<B: Backend> {
  activation: nn::Relu,

  src_ip_input: nn::Linear<B>,
  dst_port_input: nn::Linear<B>,
  protocol_input: nn::Linear<B>,

  hidden: nn::Linear<B>
}

impl<B: Backend> Ip2Vec<B> {
  fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
    let src_ip_proj = self.activation.forward(
      self.src_ip_input.forward(input.clone().narrow(1, 0, 32)));
    let dst_port_proj = self.activation.forward(
      self.dst_port_input.forward(input.clone().narrow(1, 32, 1)));
    let protocol_proj = self.activation.forward(
      self.protocol_input.forward(input.clone().narrow(1, 33, 1)));

    let combined = Tensor::cat(vec![src_ip_proj, dst_port_proj, protocol_proj], 1);

    self.hidden.forward(combined)
  }
}
