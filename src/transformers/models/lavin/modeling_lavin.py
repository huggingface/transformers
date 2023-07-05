""" PyTorch laVIN model."""

from typing import Optional

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import BackboneMixin
from .configuration_resnet import ResNetConfig


class transformer_block(nn.Module):
  def __init__(self, layer, params:arguments):
    super().__init__()
    self.params = params
    self.head_dim = self.params.dim // self.params.n_heads
    self.attention = Attention(self.params)
    self.feed_forward = feed_forward(dim = self.params.dim, hidden_dim=4*self.params.dim, multiple_of=self.params.multiple_of)
    self.layer = layer
    self.attention_norm = RMSNorm(self.params.dim, eps=self.params.norm_eps)
    self.drop_path = drop_path(self.params.drop_path) if self.params.drop_path>0. else nn.Identity()

  def drop_path(self, x, drop_prob: float = 0., training: bool = False):
      keep_prob = 1 - drop_prob
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
      random_tensor.floor_()  # binarize
      output = x.div(keep_prob) * random_tensor
      return output

  def forward(self, x:torch.Tensor, start_pos:int, freq_cis:torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    x = x+self.drop_path(self,attention.forward (self.attention_norm(x), start_pos, freq_cis, mask,adapter))
    out = 

## make all the arguments in a Union maybe and never write all the arguments manually just get the dict or Union working something like that ! for every code block that exists here do it for every single one here


class transformer(nn.Module):
  def __init__(self, params: arguments):
    super().__init__()
    self.params = params
    self.vocab = params.vocab
    self.n_layers = params.n_layers
    self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
    self.criterion = CrossEntropyLoss(ignore_index=0)

    self.layers = nn.Modulelist()
## experimental (might not include in release)
    self.layers += transformer_block(layer, params) for layer in range(self.params.n_layers)
    #for layer_id in range(params.n_layers):
    #  self.layers.append(transformer_block(layer, params)) for layer in range(self.params.n_layers)
    self.norm = RMSNorm(self.params.dim, self.params.vocab_size, bias=False)
    self.output = nn.Linear(self.params.dim, self.params.vocab_size, bias=False)
    self.freq_cis = precompute_freq(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)

    self.backbone = clip.load('ViT-L/14')[0]

    self.adapter_proj = AdapterMLP(1024,self.params.hidden_proj, params.dim).float()
    self.adapter_modality_embedding = nn.Embedding(2,self.params.dim).float()





