# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Idefics model."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ... import PreTrainedModel
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PretrainedConfig
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_idefics import IdeficsConfig
from .perceiver import IdeficsPerceiverResampler
from .vision import IdeficsVisionTransformer


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "IdeficsConfig"


from ..deprecated._archive_maps import IDEFICS_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402


@dataclass
class IdeficsBaseModelOutputWithPast(ModelOutput):
    """
    Base class for Idefics model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class IdeficsCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Idefics causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


def expand_inputs_for_generation(
    input_ids,
    expand_size=1,
    is_encoder_decoder=False,
    attention_mask=None,
    encoder_outputs=None,
    **model_kwargs,
):
    expanded_return_idx = (
        torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
    )
    input_ids = input_ids.index_select(0, expanded_return_idx)
    model_kwargs["pixel_values"] = model_kwargs.get("pixel_values", None)
    model_kwargs["image_encoder_embeddings"] = model_kwargs.get("image_encoder_embeddings", None)
    model_kwargs["perceiver_embeddings"] = model_kwargs.get("perceiver_embeddings", None)
    model_kwargs["image_attention_mask"] = model_kwargs.get("image_attention_mask", None)

    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

    if model_kwargs["image_attention_mask"] is not None:
        model_kwargs["image_attention_mask"] = model_kwargs["image_attention_mask"].index_select(
            0, expanded_return_idx
        )

    if model_kwargs["pixel_values"] is not None:
        model_kwargs["pixel_values"] = model_kwargs["pixel_values"].index_select(0, expanded_return_idx)

    elif model_kwargs["image_encoder_embeddings"] is not None:
        model_kwargs["image_encoder_embeddings"] = model_kwargs["image_encoder_embeddings"].index_select(
            0, expanded_return_idx
        )

    elif model_kwargs["perceiver_embeddings"] is not None:
        model_kwargs["perceiver_embeddings"] = model_kwargs["perceiver_embeddings"].index_select(
            0, expanded_return_idx
        )

    return input_ids, model_kwargs


def prepare_inputs_for_generation(input_ids, past_key_values=None, **kwargs):
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    if past_key_values:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -1].unsqueeze(-1)

    pixel_values = kwargs.get("pixel_values", None)
    image_encoder_embeddings = kwargs.get("image_encoder_embeddings", None)
    perceiver_embeddings = kwargs.get("perceiver_embeddings", None)
    image_attention_mask = kwargs.get("image_attention_mask", None)
    interpolate_pos_encoding = kwargs.get("interpolate_pos_encoding", False)

    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "pixel_values": pixel_values,
        "image_encoder_embeddings": image_encoder_embeddings,
        "perceiver_embeddings": perceiver_embeddings,
        "image_attention_mask": image_attention_mask,
        "interpolate_pos_encoding": interpolate_pos_encoding,
    }


def freeze_model(model, module_exceptions=[]):
    mapping = {
        "LayerNorm": nn.LayerNorm,
        "Linear": nn.Linear,
        "Embedding": nn.Embedding,
    }
    module_exceptions_mapped = [mapping[m] for m in module_exceptions]
    for module in model.modules():
        if module_exceptions and any(isinstance(module, t) for t in module_exceptions_mapped):
            module.requires_grad_(True)  # Explicitely setting it to true to avoid any mistakes
        else:
            module.requires_grad_(False)
    return model


class IdeficsDecoupledEmbedding(nn.Embedding):
    # Derived from https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#Embedding
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the embeddings. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `num_additional_embeddings` > 0,
    then it will create `num_additional_embeddings` additional parameters that are always trained. If
    `num_additional_embeddings=0`, then the module defaults back to the regular behavior of `nn.Embedding`.
    """

    def __init__(
        self,
        num_embeddings,
        num_additional_embeddings,
        embedding_dim,
        partially_freeze: Optional[bool] = False,
        device=None,
        dtype=None,
        padding_idx=None,
        **kwargs,
    ) -> None:
        """
        Args:
            num_embeddings (`int`):
                Size of the dictionary of embeddings
            num_additional_embeddings (`int`):
                Number of additional embeddings. Only useful when you `partially_freeze=True`.
            embedding_dim (`int`):
                The size of each embedding vector
            partially_freeze: (`bool`, *optional*, defaults to `False`):
                If `True`, the regular `weight` will be frozen. `additional_weight` is never frozen.
            padding_idx (`int`, *optional*):
                The padding index (needs to be less than num_embeddings)

        Note: there are a lot of other parameters to initialize a standard `nn.Embedding` such as `padding_idx`,
        `max_norm` or `norm_type`. We are not supporting these.
        """
        if padding_idx is not None and padding_idx > num_embeddings:
            raise ValueError(f"padding_idx must be within num_embeddings. Got {padding_idx} and {num_embeddings}")
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
            padding_idx=padding_idx,
            **kwargs,
        )
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.num_additional_embeddings = num_additional_embeddings
        self.partially_freeze = partially_freeze

        if partially_freeze:
            self.weight.requires_grad_(False)

        if self.num_additional_embeddings > 0:
            self.additional_embedding = nn.Embedding(
                num_embeddings=self.num_additional_embeddings,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            )

    def forward(self, input_ids):
        """
        we have 2 embeddings, with different indices - one pretrained self.weight and another
        self.additional_embedding.weight that is being trained.

        in order to make a lookup of the input ids, we:
        1. find out the indices of the entries belonging to the 2nd embedding
        2. extract those values while subtracting the size of the first embedding (num_embeddings), since the 2nd
           embedding starts from 0 and not num_embeddings
        3. perform the 2nd embedding lookup
        4. now we handle the 1st embedding, we overwrite indices belonging to the 2nd embedding with a padding index
        5. perform the 1st embedding lookup
        6. now we overwrite the values in the 1st embedding lookup with the values of the 2nd embedding lookup

        note: for the 1st embedding lookup we could have looked up only the low indices and not do the padding, but
        then we have to create a new tensor and populate it with 2 tensors that are spread out across various indices -
        i.e. not a simple concat - I haven't benchmarked the complex case if it's any faster, given that seqlens are
        usually relatively short it's probably not faster or if faster not by much - but might be a good idea to
        measure.

        """
        if self.num_additional_embeddings == 0:
            return F.embedding(input_ids, self.weight)

        # Clone so that we don't modify the original input_ids later on
        input_ids = input_ids.clone()
        additional_vocab_indices = torch.where(input_ids >= self.num_embeddings)
        input_ids_additional_vocab = input_ids[additional_vocab_indices]
        additional_embeddings = self.additional_embedding(input_ids_additional_vocab - self.num_embeddings)

        # for successful lookup replace input_ids with 0, the results of these will be discarded anyway
        input_ids[additional_vocab_indices] = 0
        full_vector = F.embedding(input_ids, self.weight)

        # overwrite the records with high indices
        full_vector[additional_vocab_indices] = additional_embeddings

        return full_vector

    def extra_repr(self) -> str:
        return "num_embeddings={}, num_additional_embeddings={}, embedding_dim={}, partially_freeze={}".format(
            self.num_embeddings,
            self.num_additional_embeddings,
            self.embedding_dim,
            self.partially_freeze,
        )


class IdeficsDecoupledLinear(nn.Linear):
    # Derived from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the parameters. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `out_additional_features` > 0,
    then it will create `out_additional_features * in_features` additional parameters that are always trained. If
    `out_additional_features=0`, then the module defaults back to the regular behavior of `nn.Linear`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        out_additional_features: int = 0,
        bias: bool = True,
        partially_freeze: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """
        out_additional_features: int. Number of additional trainable dimensions. Only makes sense when
        `partially_freeze=True`. partially_freeze: bool. If True, the regular `weight` will be frozen and extra
        parameters (if any) will be trainable. If False, default to the regular behavior of nn.Linear.
        """
        super().__init__(in_features, out_features, bias, device, dtype)
        self.out_additional_features = out_additional_features
        self.partially_freeze = partially_freeze

        self.in_features = in_features
        self.out_features = out_features

        if partially_freeze:
            self.weight.requires_grad_(False)
            if bias:
                self.bias.requires_grad_(False)

        if out_additional_features > 0:
            self.additional_fc = nn.Linear(
                in_features=in_features,
                out_features=out_additional_features,
                bias=bias,
                device=device,
                dtype=dtype,
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.linear(input, self.weight, self.bias)

        if self.out_additional_features > 0:
            additional_features = self.additional_fc(input)
            output = torch.cat((output, additional_features), -1)

        return output

    def extra_repr(self) -> str:
        """Overwriting `nn.Linear.extra_repr` to include new parameters."""
        return "in_features={}, out_features={}, out_additional_features={}, bias={}, partially_freeze={}".format(
            self.in_features,
            self.out_features,
            self.out_additional_features,
            self.bias is not None,
            self.partially_freeze,
        )


# this was adapted from LlamaRMSNorm
class IdeficsRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        IdeficsRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


ALL_LAYERNORM_LAYERS.append(IdeficsRMSNorm)


# this was adapted from LlamaRotaryEmbedding
class IdeficsEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# this was adapted from LlamaMLP
class IdeficsMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# this was adapted from LlamaAttention
class IdeficsAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        is_cross_attention: bool = False,
        config: PretrainedConfig = None,
        qk_layer_norms: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.is_causal = True

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {num_heads})."
            )

        self.is_cross_attention = is_cross_attention

        if not hasattr(nn.functional, "scaled_dot_product_attention"):
            raise ValueError("this model requires pytorch 2.0 or higher")

        if self.is_cross_attention:
            kv_input_dim = (
                self.hidden_size if not hasattr(config.vision_config, "embed_dim") else config.vision_config.embed_dim
            )
            self.q_proj = nn.Linear(
                self.hidden_size,
                num_heads * self.head_dim,
                bias=False,
            )
            self.k_proj = nn.Linear(kv_input_dim, num_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(
                kv_input_dim,
                num_heads * self.head_dim,
                bias=False,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size,
                num_heads * self.head_dim,
                bias=False,
            )
            self.k_proj = nn.Linear(
                self.hidden_size,
                num_heads * self.head_dim,
                bias=False,
            )
            self.v_proj = nn.Linear(
                self.hidden_size,
                num_heads * self.head_dim,
                bias=False,
            )
        self.o_proj = nn.Linear(
            num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = IdeficsEmbedding(self.head_dim)

        self.qk_layer_norms = qk_layer_norms
        if self.qk_layer_norms:
            self.q_layer_norm = IdeficsRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_layer_norm = IdeficsRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # if key_value_states are provided this layer is used as a cross-attention layer
        is_cross_attention = self.is_cross_attention or key_value_states is not None

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if not is_cross_attention:
            key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            _, kv_len, _ = key_value_states.size()  # Note that, in this case, `kv_len` == `kv_seq_len`
            key_states = self.k_proj(key_value_states).view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = (
                self.v_proj(key_value_states).view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
            )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        if not is_cross_attention:
            cos, sin = self.rotary_emb(value_states, seq_len=max(kv_seq_len, q_len))
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if self.qk_layer_norms:
            query_states = self.q_layer_norm(query_states)
            key_states = self.k_layer_norm(key_states)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        attn_weights = None
        if output_attentions:
            logger.warning_once(
                "attn_weights are not extracted in scaled_dot_product_attention. The model returns None instead"
            )

        return attn_output, attn_weights, past_key_value


# this was adapted from LlamaDecoderLayer
class IdeficsDecoderLayer(nn.Module):
    def __init__(self, config: IdeficsConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = IdeficsAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            config=config,
        )
        self.mlp = IdeficsMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = IdeficsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = IdeficsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = config.dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class IdeficsGatedCrossAttentionLayer(nn.Module):
    def __init__(self, config: IdeficsConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.cross_attn = IdeficsAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            is_cross_attention=True,
            dropout=config.dropout,
            config=config,
            qk_layer_norms=config.qk_layer_norms,
        )
        self.mlp = IdeficsMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = IdeficsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = IdeficsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config.dropout

        self.act_cross_attn = nn.Tanh()
        self.act_dense = nn.Tanh()

        if config.alpha_initializer == "zeros":
            if config.alpha_type == "vector":
                self.alpha_cross_attn = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
                self.alpha_dense = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
            elif config.alpha_type == "float":
                self.alpha_cross_attn = nn.Parameter(torch.zeros(1))
                self.alpha_dense = nn.Parameter(torch.zeros(1))
            else:
                raise ValueError(f"Unknown value for `alpha_type` ({config.alpha_type})")

        elif config.alpha_initializer == "ones":
            if config.alpha_type == "vector":
                self.alpha_cross_attn = nn.Parameter(torch.ones(1, 1, self.hidden_size))
                self.alpha_dense = nn.Parameter(torch.ones(1, 1, self.hidden_size))
            elif config.alpha_type == "float":
                self.alpha_cross_attn = nn.Parameter(torch.ones(1))
                self.alpha_dense = nn.Parameter(torch.ones(1))
            else:
                raise ValueError(f"Unknown value for `alpha_type` ({config.alpha_type})")

        elif config.alpha_initializer in {"normal", "gaussian", "random"}:
            if config.alpha_type == "vector":
                self.alpha_cross_attn = nn.Parameter(
                    torch.normal(mean=0.0, std=config.alphas_initializer_range, size=(1, 1, self.hidden_size))
                )
                self.alpha_dense = nn.Parameter(
                    torch.normal(mean=0.0, std=config.alphas_initializer_range, size=(1, 1, self.hidden_size))
                )
            elif config.alpha_type == "float":
                self.alpha_cross_attn = nn.Parameter(
                    torch.normal(mean=0.0, std=config.alphas_initializer_range, size=(1))
                )
                self.alpha_dense = nn.Parameter(torch.normal(mean=0.0, std=config.alphas_initializer_range, size=(1)))
            else:
                raise ValueError(f"Unknown value for `alpha_type` ({config.alpha_type})")

        else:
            raise NotImplementedError(f"Alpha initialization scheme {config.alpha_initializer} not yet implemented!")

        if not (hasattr(self, "alpha_cross_attn") and hasattr(self, "alpha_dense")):
            raise ValueError("Alpha parameters not initialized correctly!")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_hidden_states: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_gate: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            image_attention_mask (`torch.FloatTensor`, *optional*): image attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            cross_attention_gate (`torch.FloatTensor`, *optional*):
                gate of size `(batch, seq_len)` used to zero-out cross-attention output for tokens attending no images.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if image_hidden_states is None:
            raise ValueError(
                "`image_hidden_states` is required for Idefics cross attention module which are visual features to be"
                " conditioned on."
            )

        if cross_attention_gate is None:
            raise ValueError(
                "`cross_attention_gate` is required for Idefics cross attention module to zero-out the cross-attention hidden_states attending to no images."
            )

        if past_key_value is not None:
            raise NotImplementedError("Past key value states are not implemented for Idefics cross attention module.")

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.cross_attn(
            hidden_states=hidden_states,
            key_value_states=image_hidden_states,
            attention_mask=image_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.config, training=self.training)
        # Fill in zeros for cross_attention hidden_states of tokens attending to no images
        hidden_states[cross_attention_gate == 0] = hidden_states[cross_attention_gate == 0].fill_(0)
        hidden_states = residual + self.act_cross_attn(self.alpha_cross_attn) * hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.config, training=self.training)
        hidden_states = residual + self.act_dense(self.alpha_dense) * hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`IdeficsConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class IdeficsPreTrainedModel(PreTrainedModel):
    config_class = IdeficsConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["IdeficsDecoderLayer", "IdeficsGatedCrossAttentionLayer"]
    _supports_sdpa = True

    def _init_weights(self, module):
        # important: this ported version of Idefics isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the m4 code
        # base should be used for training from scratch and it contains the correct code.
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # Adapted from transformers.modeling_utils.PreTrainedModel._check_and_enable_sdpa
    @classmethod
    def _check_and_enable_sdpa(cls, config, hard_check_only: bool = False) -> PretrainedConfig:
        # We remove the checks on `is_torch_sdpa_available()` and `cls._supports_sdpa` as Falcon supports SDPA from torch==2.0.0 (no requirement on 2.1).
        _is_bettertransformer = getattr(cls, "use_bettertransformer", False)
        if _is_bettertransformer:
            return config

        if not hard_check_only:
            config._attn_implementation = "sdpa"
        return config


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class IdeficsModel(IdeficsPreTrainedModel):
    """
    Transformer decoder consisting of `config.num_hidden_layers` layers. Each layer is a [`IdeficsDecoderLayer`]

    Args:
        config: IdeficsConfig
    """

    def __init__(self, config: IdeficsConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = IdeficsDecoupledEmbedding(
            num_embeddings=config.vocab_size,
            num_additional_embeddings=config.additional_vocab_size,
            embedding_dim=config.hidden_size,
            partially_freeze=config.freeze_text_layers,
            padding_idx=self.padding_idx,
        )

        self.image_size = config.vision_config.image_size
        self.vision_config = config.vision_config
        self.vision_model = IdeficsVisionTransformer(config.vision_config)

        # Perceiver Resampler
        if config.use_resampler:
            perceiver_config = config.perceiver_config
            self.perceiver_resampler = IdeficsPerceiverResampler(
                config,
                config.vision_config.embed_dim,
                perceiver_config.resampler_depth,
                perceiver_config.resampler_n_heads,
                perceiver_config.resampler_head_dim,
                perceiver_config.resampler_n_latents,
            )

        self.layers = nn.ModuleList([IdeficsDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.cross_layer_interval = config.cross_layer_interval
        num_cross_layers = config.num_hidden_layers // self.cross_layer_interval
        self.gated_cross_attn_layers = nn.ModuleList(
            [IdeficsGatedCrossAttentionLayer(config) for _ in range(num_cross_layers)]
        )
        self.gradient_checkpointing = False

        self.norm = IdeficsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

        self.freeze_relevant_params(config)

    def freeze_relevant_params(self, config=None):
        if config is None:
            config = self.config

        if config.freeze_text_layers:
            self.freeze_text_layers(config.freeze_text_module_exceptions)

        if config.freeze_vision_layers:
            freeze_model(self.vision_model, module_exceptions=config.freeze_vision_module_exceptions)

    def freeze_text_layers(self, module_exceptions=[]):
        for module in [self.layers, self.norm]:
            freeze_model(module, module_exceptions=module_exceptions)

    def freeze_vision_layers(self, module_exceptions=[]):
        freeze_model(self.vision_model, module_exceptions=module_exceptions)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_encoder_embeddings: Optional[torch.FloatTensor] = None,
        perceiver_embeddings: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, IdeficsBaseModelOutputWithPast]:
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        elif position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if (pixel_values, image_encoder_embeddings, perceiver_embeddings).count(None) != 2:
            raise ValueError(
                "Exactly 1 of pixel_values, image_encoder_embeddings or perceiver_embeddings has to be not-None."
            )

        elif pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.dtype, device=device)  # fp16 compatibility
            batch_size, num_images = pixel_values.shape[:2]
            pixel_values = pixel_values.contiguous().view(batch_size * num_images, *pixel_values.shape[2:])

            # Get sequence from the vision encoder
            image_hidden_states = self.vision_model(
                pixel_values=pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
            ).last_hidden_state

        elif image_encoder_embeddings is not None:
            batch_size, num_images, image_seq_len, image_hidden_size = image_encoder_embeddings.size()
            image_hidden_states = image_encoder_embeddings.to(dtype=self.dtype, device=device)
            image_hidden_states = image_hidden_states.view(batch_size * num_images, image_seq_len, image_hidden_size)

        if self.config.use_resampler:
            if perceiver_embeddings is None:
                perceiver_embeddings = self.perceiver_resampler(image_hidden_states)
                image_seq_len, image_hidden_size = perceiver_embeddings.size(1), perceiver_embeddings.size(2)
            else:
                batch_size, num_images, image_seq_len, image_hidden_size = perceiver_embeddings.size()
            image_hidden_states = perceiver_embeddings
        elif perceiver_embeddings is None:
            image_seq_len, image_hidden_size = image_hidden_states.size(1), image_hidden_states.size(2)
        else:
            raise ValueError("If `perceiver_embeddings` are passed, use_resampler should be True")

        image_hidden_states = image_hidden_states.view(batch_size, num_images * image_seq_len, image_hidden_size)
        # # Hack to use the model in full language modeling mode
        # image_attention_mask = torch.zeros(batch_size, seq_length, 1, dtype=torch.long, device=image_hidden_states.device)
        # Make image_attention_mask compatible with hidden states
        text_seq_len = image_attention_mask.size(1)
        image_attention_mask = image_attention_mask.unsqueeze(-1)
        image_attention_mask = image_attention_mask.repeat(1, 1, 1, image_seq_len)
        image_attention_mask = image_attention_mask.view(batch_size, text_seq_len, num_images * image_seq_len)

        if image_hidden_states is not None:
            image_batch_size, image_sequence_length, _ = image_hidden_states.size()
            image_hidden_shape = (image_batch_size, image_sequence_length)
            if image_attention_mask is None:
                image_attention_mask = torch.ones(image_hidden_shape, device=device)
            image_attention_mask = self.invert_attention_mask(image_attention_mask)
        else:
            image_attention_mask = None

        # cross_attention_gate:
        # For any tokens attending to no images, the hidden_states comming out of the cross-attention should be zeroed-out.
        # `image_attention_mask` has shape [bsz, 1, num_images, hidden_size] with elements equal to either 0.0 or a very negative number.
        # If any of the elements are 0.0, then the token is attending to at least one image and the gate value is 1. Otherwise the gate value is 0.
        # `cross_attention_gate` has shape [bsz, seq_len] with elements equal to either 0.0 or 1.0.
        cross_attention_gate = ((((image_attention_mask == 0.0).any(dim=-1)).to(dtype=self.dtype)).squeeze(dim=1)).to(
            device
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            def vblock(
                main_block,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                image_hidden_states,
                image_attention_mask,
                cross_attention_gate,
                output_attentions,
                use_cache,
                layer_idx,
                cross_layer_interval,
                gated_cross_attn_layers,
            ):
                # TODO(ls): Add cross attention values to respective lists
                if layer_idx % cross_layer_interval == 0:
                    xblock = gated_cross_attn_layers[layer_idx // cross_layer_interval]
                    outputs = xblock(
                        hidden_states,
                        attention_mask=attention_mask,
                        image_hidden_states=image_hidden_states,
                        image_attention_mask=image_attention_mask,
                        cross_attention_gate=cross_attention_gate,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        past_key_value=None,  # not implemented
                    )
                    hidden_states = outputs[0]

                layer_outputs = main_block(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

                return layer_outputs

            if self.gradient_checkpointing and self.training:
                past_key_value = None
                if use_cache:
                    logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                layer_outputs = self._gradient_checkpointing_func(
                    vblock,
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    image_hidden_states,
                    image_attention_mask,
                    cross_attention_gate,
                    output_attentions,
                    use_cache,
                    idx,
                    self.cross_layer_interval,
                    self.gated_cross_attn_layers,
                )
            else:
                layer_outputs = vblock(
                    decoder_layer,
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    image_hidden_states=image_hidden_states,
                    image_attention_mask=image_attention_mask,
                    cross_attention_gate=cross_attention_gate,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    layer_idx=idx,
                    cross_layer_interval=self.cross_layer_interval,
                    gated_cross_attn_layers=self.gated_cross_attn_layers,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        image_hidden_states = image_hidden_states.view(batch_size, num_images, image_seq_len, image_hidden_size)
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, image_hidden_states]
                if v is not None
            )
        return IdeficsBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            image_hidden_states=image_hidden_states,
        )


class IdeficsForVisionText2Text(IdeficsPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _tied_weights_keys = ["model.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config, vision_model=None):
        super().__init__(config)
        self.model = IdeficsModel(config)

        self.lm_head = IdeficsDecoupledLinear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            out_additional_features=config.additional_vocab_size,
            bias=False,
            partially_freeze=config.freeze_lm_head,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def tie_weights(self):
        """
        Overwrite `transformers.modeling_utils.PreTrainedModel.tie_weights` to handle the case of
        IdeficsDecoupledLinear and IdeficsDecoupledEmbedding.
        """
        output_embeddings = self.get_output_embeddings()
        input_embeddings = self.get_input_embeddings()

        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings.weight = input_embeddings.weight
            if input_embeddings.num_additional_embeddings > 0:
                assert output_embeddings.out_additional_features == input_embeddings.num_additional_embeddings
                output_embeddings.additional_fc.weight = input_embeddings.additional_embedding.weight

        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings
            if hasattr(output_embeddings, "out_additional_features") and hasattr(
                input_embeddings, "num_additional_embeddings"
            ):
                output_embeddings.out_additional_features = input_embeddings.num_additional_embeddings

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=IdeficsCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_encoder_embeddings: Optional[torch.FloatTensor] = None,
        perceiver_embeddings: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, IdeficsCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, IdeficsForVisionText2Text

        >>> model = IdeficsForVisionText2Text.from_pretrained("HuggingFaceM4/idefics-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("HuggingFaceM4/idefics-9b")

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_encoder_embeddings=image_encoder_embeddings,
            perceiver_embeddings=perceiver_embeddings,
            image_attention_mask=image_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return IdeficsCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        image_hidden_states = kwargs.pop("image_hidden_states", None)
        if image_hidden_states is not None:
            if self.config.use_resampler:
                kwargs["perceiver_embeddings"] = image_hidden_states
            else:
                kwargs["image_encoder_embeddings"] = image_hidden_states
            kwargs["pixel_values"] = None
        inputs = prepare_inputs_for_generation(input_ids, past=past, **kwargs)
        unwanted_kwargs = ["token_type_ids"]
        for kwarg in unwanted_kwargs:
            inputs.pop(kwarg, None)
        return inputs

    @staticmethod
    def _expand_inputs_for_generation(
        *args,
        **model_kwargs,
    ):
        return expand_inputs_for_generation(*args, **model_kwargs)

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder,
            standardize_cache_format,
        )

        if "image_attention_mask" in model_kwargs:
            image_attention_mask = model_kwargs["image_attention_mask"]
            last_mask = image_attention_mask[:, -1, :].unsqueeze(1)
            model_kwargs["image_attention_mask"] = last_mask

        # Get the precomputed image_hidden_states
        model_kwargs["image_hidden_states"] = outputs.image_hidden_states
        return model_kwargs

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
