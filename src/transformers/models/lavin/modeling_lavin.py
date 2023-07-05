""" PyTorch laVIN model."""

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ....transformers import CLIPModel as clip

#from ...modeling_outputs import (
#    BackboneOutput,
#    BaseModelOutputWithNoAttention,
#    BaseModelOutputWithPoolingAndNoAttention,
#    ImageClassifierOutputWithNoAttention,
#)
#from ...modeling_utils import PreTrainedModel
#from ...utils import (
#    add_code_sample_docstrings,
#    add_start_docstrings,
#    add_start_docstrings_to_model_forward,
#    logging,
#    replace_return_docstrings,
#)
#from ...utils.backbone_utils import BackboneMixin
#from .configuration_resnet import ResNetConfig


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    hidden_proj: int=128

    max_batch_size: int = 32
    max_seq_len: int = 2048
    drop_path: float=0.

class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))
  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)

  def forward(self, x):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight

def precomute_freqs_cis(sim: int, end: int, theta: float = 10000.0):
  freqs = 1. / (theta ** (torch.arange(0,dim,2)[: (dim // 2)].float()/dim))
  t = torch.arange(end, device=freqs.device)
  freqs = torch.outer(t, freqs).float()
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
  return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
  ndim = x.ndim
  assert 0 <= 1 < ndim
  assert freqs_cis.sape == (x.shape[1], x.shape[-1])
  shape = [d if i == 1 or i == ndim - 1 else 1 for i,d in enumerate(x.shape)]
  return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, frews_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
  return xq_out.type_as(xq), xk_out.type_as(xk)


class attention_block(nn.Module):
  def __init__(self, params: ModelArgs):
    super().__init__()
    self.params = params
    self.head_dim = self.params.dim // self.params.n_heads

    self.wq = Linear(self.params.dim, self.params.n_heads * self.params.head_dim, bias=False)
    self.wk = Linear(self.params.dim, self.params.n_heads * self.params.head_dim, bias=False)
    self.wv = Linear(self.params.dim, self.params.n_heads * self.params.head_dim, bias=False)
    self.wo = Linear(self.params.n_heads * self.params.head_dim, self.params.dim,  bias=False)

  def forward(self, x:torch.Tensor, start_pos: int, freq_cis:torch.Tensor, mask: Optional[torch.Tensor], adaptor=None):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    
    xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xv = xqv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freq_cis=freq_cis)
    keys = xk
    values = xv

    xq = xq.transpose(1,2)
    keys, values = keys.transpose(1,2), values.transpose(1,2)
    scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
    if mask is not None:
      scores += mask
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = torch.matmul(scores, values)
    output = output.transpose(1,2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)

    

class feed_forward(nn.Module):
  def __init__(self, dim:int, hidden_dim: int, multiple_of: int):
    super().__init__()
    hidden_dim = int(2*hidden_dim / 3)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    self.w1 = Linear(dim, hidden_dim, bias=False)
    self.w2 = Linear(hidden_dim, dim, bias=False)
    self.w3 = Linear(dim, hidden_dim, bias=False)

  def forward(self, x):
    return self.w2(F.silu(self.w1(x), inplace=False) * self.w3(x))


class transformer_block(nn.Module):
  def __init__(self, layer, params:ModelArgs):
    super().__init__()
    self.params = params
    self.head_dim = self.params.dim // self.params.n_heads
    self.attention = Attention(self.params)
    self.feed_forward = feed_forward(dim = self.params.dim, hidden_dim=4*self.params.dim, multiple_of=self.params.multiple_of)
    self.layer = layer
    self.attention_norm = RMSNorm(self.params.dim, eps=self.params.norm_eps)
    self.ffn_norm = RMSNorm(self.params.dim, eps=self.params.norm_eps)
    self.drop_path = drop_path(self.params.drop_path) if self.params.drop_path>0. else nn.Identity()

  def drop_path(self, x, drop_prob: float = 0., training: bool = False):
      keep_prob = 1 - drop_prob
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
      random_tensor.floor_()  # binarize
      output = x.div(keep_prob) * random_tensor
      return output

  def forward(self, x:torch.Tensor, start_pos:int, freq_cis:torch.Tensor, mask: Optional[torch.Tensor], adapter=None):
    x = x+self.drop_path(self.attention.forward(self.attention_norm(x), start_pos, freq_cis, mask,adapter))
    x = x + self.drop_path(self.feed_forward.forward(self.ffn_norm(x)))
    return x

## make all the arguments in a Union maybe and never write all the arguments manually just get the dict or Union working something like that ! for every code block that exists here do it for every single one here

class AdapterMLP(nn.Module):
  def __init__(self, in_features=768, hidden_dim=128, out_features=4096):
    super().__init__()
    self.conv_A = nn.Linear(in_features, hidden_dim)
    slef.conv_B = nn.Linear(hidden_dim, out_features)

    nn.init.xavier_uniform_(self.conv_A.weight)
    nn.init.zeros_(self.conv_A.bias)
    nn.init.xavier_uniform_(self.conv_B.weight)
    nn.init.zeros_(self.conv_B.bias)

  def forward(self,x):
    with autocast():
      x = self.conv_B(F.silu(self.conv_A(x)))
    return x

class transformer(nn.Module):
  def __init__(self, params: ModelArgs):
    super().__init__()
    self.params = params
    self.vocab = params.vocab
    self.n_layers = params.n_layers
    self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
    self.criterion = CrossEntropyLoss(ignore_index=0)

    self.layers = nn.Modulelist()
## experimental (might not include in release)
    #self.layers += transformer_block(layer, params) for layer in range(self.params.n_layers)
    for layer_id in range(params.n_layers):
      self.layers.append(transformer_block(layer_id, params))
    self.norm = RMSNorm(self.params.dim, self.params.vocab_size, bias=False)
    self.output = nn.Linear(self.params.dim, self.params.vocab_size, bias=False)
    self.freq_cis = precompute_freq(self.params.dim // self.params.n_heads, self.params.max_seq_len * 2)

    self.backbone = clip.load('ViT-L/14')[0]

    self.adapter_proj = AdapterMLP(1024,self.params.hidden_proj, params.dim).float()
    self.adapter_modality_embedding = nn.Embedding(2,self.params.dim).float()

  def insert_image_embeds(self,examples,labels,image_embeds,prefix_img,prefix_nonimg,img_indicators):
    _bsz, seqlen,_ = examples.shape
    new_examples=[]
    new_labels=[]
    for i, (example,label) in enumerate(zip(examples,labels)):
      if img_indicators[i]>0.:
        new_example=torch.cat([example[:1],prefix_img,image_embeds[i],example[1:]],0)
        new_label=torch.cat([label[:1],
                             torch.zeros(prefix_img.shape[0]+image_embeds.shape[1]).to(examples.device).type_as(labels),
                             label[1:]])
        new_example = new_example[:seqlen]
        new_label = new_label[:seqlen]
      else:
        new_example=torch.cat([example[:1],prefix_nonimg,example[1:]],0)
        new_label=torch.cat([label[:1],
                             torch.zeros(prefix_nonimg.shape[0]).to(examples.device).type_as(labels),
                             label[1:]])
        new_example = new_example[:seqlen]
        new_label = new_label[:seqlen]
      new_examples.append(new_example.unsqueeze(0))
      new_labels.append(new_label.unsqueeze(0))
    new_examples = torch.cat(new_examples, 0)
    new_labels = torch.cat(new_labels, 0)
    return new_examples,new_labels

  def forward(self, examples, labels,images=None, prefix_img=None, prefix_nonimg=None,img_indicators=None):

    # print(images.dtype)
    image_embeds = self.backbone.encode_image(images).half()

    # print(img_indicators)
    if isinstance(img_indicators,list):
      img_indicators = torch.Tensor(img_indicators).to(image_embeds.device).long()
    modality_embed=self.adapter_modality_embedding(img_indicators.unsqueeze(1))

    # with autocast():
    image_embeds=self.adapter_proj(image_embeds)

    # print(image_embeds.shape)

    _bsz, seqlen = examples.shape

    examples = self.tok_embeddings(examples)
    prefix_img=self.tok_embeddings(prefix_img.unsqueeze(0)).squeeze(0)
    prefix_nonimg=self.tok_embeddings(prefix_nonimg.unsqueeze(0)).squeeze(0)


    h,labels=self.insert_image_embeds(examples,labels,image_embeds,prefix_img,prefix_nonimg,img_indicators)

    h=torch.cat([modality_embed.half(),h],1)[:,:seqlen]
    modality_labels=torch.zeros(_bsz,1).to(labels.device).type_as(labels)
    labels=torch.cat([modality_labels,labels],1)[:,:seqlen]


    freqs_cis = self.freqs_cis.to(h.device)
    freqs_cis = freqs_cis[:seqlen]
    mask = None
    mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
    mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

    #mask decision token
    mask[:,:,1:,0]=float("-inf")

    start_pos = 0
    for layer in self.layers:
      h = layer(h, start_pos, freqs_cis, mask)

    h = self.norm(h)
    output = self.output(h)
    output = output[:, :-1, :].reshape(-1, self.vocab_size)
    labels = labels[:, 1:].flatten()


    c_loss = self.criterion(output, labels)
    return c_loss


model = transformers(ModelArgs)
model()
