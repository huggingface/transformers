# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
""" PyTorch RecurrentGemma model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import einops
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_utils import ModelOutput, PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
    replace_return_docstrings,
)
from .configuration_recurrent_gemma import RecurrentGemmaConfig


if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "RecurrentGemmaConfig"

_MAX_SQRT_GRADIENT = 1000.0


class HybridCache(Cache):
    """
    Hybrid Sliding window Cache class to be used with `torch.compile(model)`.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the `window_length`, `hidden_size` and `num_attention_heads`
            required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(self, config, max_batch_size, window_length: int, device, dtype=None) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = window_length
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )

        cache_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        self.key_cache: torch.Tensor = []
        self.value_cache: torch.Tensor = []
        self.rg_lru_state: torch.Tensor = []
        self.conv1d_state: torch.Tensor = []

        for i in range(config.num_hidden_layers):
            if config.block_types[i] == "recurrent":
                self.rg_lru_state.append(torch.zeros((max_batch_size, config.lru_width), dtype=config.torch_dtype,device=device))
                self.conv1d_state.append(torch.zeros((max_batch_size, config.hidden_size, config.conv1d_width), dtype=config.torch_dtype,device=device))
                self.value_cache.append([])
                self.key_cache.append([])
            else:
                self.value_cache.append(torch.zeros(cache_shape, dtype=self.dtype, device=device))
                self.key_cache.append(torch.zeros(cache_shape, dtype=self.dtype, device=device))
                self.rg_lru_state.append([])
                self.conv1d_state.append([])

    def update(self, key_states, value_states, layer_idx, **cache_kwargs):
        new_cache_positions = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        # Shift the cache
        if new_cache_positions[-1] >= self.max_cache_len:
            k_out = torch.roll(k_out, shifts=-1, dim=2)
            v_out = torch.roll(v_out, shifts=-1, dim=2)

        k_out[:, :, new_cache_positions] = key_states
        v_out[:, :, new_cache_positions] = value_states

        return k_out, v_out

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))


# Copied from transformers.models.gemma.modeling_gemma.GemmaRMSNorm with Gemma->RecurrentGemma
class RecurrentGemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst RecurrentGemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


ALL_LAYERNORM_LAYERS.append(RecurrentGemmaRMSNorm)


# Copied from transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding with Gemma->RecurrentGemma
class RecurrentGemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).float() / self.dim)
            )
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
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
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RecurrentGemmaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Ignore copy
    def __init__(self, config: RecurrentGemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.rope_theta = config.rope_theta

        self.partial_rotary_factor = 0.5
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_attention_heads`: {self.num_attention_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = RecurrentGemmaRotaryEmbedding(
            int(self.partial_rotary_factor * self.head_dim),
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)

        # Partial rotary embedding
        query_rot, query_pass = torch.chunk(query_states, int(1/self.partial_rotary_factor), dim=-1)
        key_rot, key_pass = torch.chunk(key_states, int(1/self.partial_rotary_factor), dim=-1)
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, **cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_attention_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_attention_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, past_key_value


class RecurrentGemmaRecurrentBlock(nn.Module):
    """Griffin and Hawk's recurrent block."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.linear_y = nn.Linear(in_features=config.hidden_size, out_features=config.lru_width)
        self.linear_x = nn.Linear(in_features=config.hidden_size, out_features=config.lru_width)
        self.linear_out = nn.Linear(in_features=config.lru_width, out_features=config.hidden_size)
        self.conv1d_width = config.conv1d_width
        self.conv_1d = nn.Conv1d(
            config.lru_width,
            config.lru_width,
            kernel_size=config.conv1d_width,
            groups=config.lru_width,
            padding=config.conv1d_width - 1,
        )
        self.rg_lru = RecurrentGemmaRglru(config)
        self.act_fn = ACT2FN[config.hidden_activation]
        self.layer_idx = layer_idx

    def forward(
        self,
        input_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache: Union[dict[str, torch.Tensor], None],
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Calls the recurrent block.

        Args:
          hidden_states: Sequence of input activations.
          position_ids: Position of each token in the sequence.
          attention_mask: Unused attention mask.
          cache: Optional cache with the previous state of the RG-LRU and Conv1D.

        Returns:
          Output of the block together with the updated cache. If `cache` is None
          than the returned updated cache is empty initialized and filled in from
          the input sequence.
        """
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        y_branch = self.linear_y(input_states)
        y_branch = self.act_fn(y_branch)

        x_branch = self.linear_x(input_states)

        hidden_states = self.conv_1d(x_branch.transpose(1,2))

        if cache is not None:
            if cache_position[0] == 0:
                conv_state = nn.functional.pad( 
                    hidden_states, (self.conv1d_width - hidden_states.shape[-1], 0)
                )
                hidden_states = self.conv_1d(hidden_states)[..., :seq_len]
            else:
                conv_state = cache.conv1d_state[self.layer_idx]                # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                hidden_states = torch.sum(conv_state * self.conv_1d.weight[:, 0, :], dim=-1) + self.conv_1d.bias

            cache.conv1d_state[self.layer_idx].copy_(conv_state)
        
            rg_lru_state = cache.rg_lru_state[self.layer_idx]
        else:
            hidden_states = self.act(self.conv_1d(hidden_states)[..., :seq_len])  
        
            rg_lru_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size), device=hidden_states.device, dtype=dtype
            )


        hidden_states, rg_lru_state = self.rg_lru(
            activations=y_branch,
            position_ids=position_ids,
            prev_h=rg_lru_state,
        )

        # Join branches.
        hidden_states = hidden_states * y_branch
        hidden_states = self.linear_out(hidden_states)

        cache.rg_lru_state = rg_lru_state

        return hidden_states, cache


TEMPORAL_BLOCK_CLASSES = {"recurrent": RecurrentGemmaRecurrentBlock, "attention": RecurrentGemmaAttention}


# TODO copied from mistral or whatever
class RecurrentGemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))



# TODO remove einops from this one
class RecurrentGemmaBlockDiagonalLinear(nn.Module):
    """Block-diagonal linear layer."""

    def __init__(self, config):
        """Initializes the RecurrentGemmaBlockDiagonalLinear.

        Args:
          width: The number of dimensions of the input and output.
          num_blocks: The number of diagonal blocks in the layer.
          w_init_variance_scale: A parameters that scales the variance of the
            initialization of the weights.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialization.
          dtype: What dtype to use for initialization.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_blocks = config.num_blocks
        self.w_init_variance_scale = config.w_init_variance_scale
        self.block_width = self.hidden_size // self.num_blocks

        # Parameters.
        self.weight = nn.Parameter(torch.empty([self.num_blocks, self.block_width, self.block_width]))
        self.bias = nn.Parameter(torch.empty([self.num_blocks, self.block_width]))

        # Initialization.
        self.reset_parameters()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Calls the RecurrentGemmaBlockDiagonalLinear."""
        # TODO REMOVE EINSUMS
        # Split x to blocks.
        hidden_states = einops.rearrange(hidden_states, "... (h i) -> ... h i", h=self.num_blocks)

        # Linear layer over each block + bias.
        y = torch.einsum("... h i, h i j -> ... h j", hidden_states, self.weight) + self.bias

        # Flatten the output.
        return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)


# TODO to refactor
def rnn_scan(
    x: torch.Tensor,
    a: torch.Tensor,
    reset: torch.Tensor,
    h0: Union[torch.Tensor, None],
    acc_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs the recurrence of a linear RNN.

    Args:
      x: The input sequence.
      a: The diagonal of the recurrence matrix `A`.
      reset: Indicator of document boundaries, e.g. when to reset the hidden state
        of the RNN.
      h0: The initial hidden state.
      acc_dtype: The data type for the accumulation.

    Returns:
      The output of the linear recurrence.
    """
    assert x.ndim == 3
    assert a.shape == x.shape[-a.ndim :]
    assert a.dtype == x.dtype
    assert type(a) is type(x)
    assert h0 is None or h0.dtype == acc_dtype

    # Multiply `a` by the reset.
    a = a * ~reset

    if x.shape[1] == 1:
        # Using scan in sampling mode.
        if h0 is None:
            return x, x[:, 0].type(acc_dtype)

        else:
            y = a.type(acc_dtype) * h0[:, None] + x.type(acc_dtype)
            return y.type(x.dtype), y[:, -1]

    else:
        # Using scan in linear mode.
        if h0 is not None:
            h_t = h0
        else:
            h_t = torch.zeros(x[:, 0].shape, dtype=acc_dtype, device=x.device)

        y = torch.zeros_like(x)
        for t in range(x.shape[1]):
            h_t = a[:, t].type(acc_dtype) * h_t + x[:, t].type(acc_dtype)
            y[:, t] = h_t.type(x.dtype)

    return y, h_t


class SqrtBoundDerivative(torch.autograd.Function):
    """Computes a square root with a gradient clipped at `_MAX_SQRT_GRADIENT`."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """The forward pass, which is a normal `sqrt`."""
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """The backward pass, which clips the `sqrt` gradient."""
        (x,) = ctx.saved_tensors
        clipped_x_times_4 = torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT**2))
        return grad_output / torch.sqrt(clipped_x_times_4)


class RecurrentGemmaRglru(nn.Module):
    """A Real-Gated Linear Recurrent Unit (RG-LRU) layer."""

    def __init__(self, config):
        """Initializes the RG-LRU.

        Args:
          width: The number of dimensions of the input and output.
          num_attention_heads: The number of diagonal blocks in the input and A gate layers.
          w_init_variance_scale: Initialization parameter for the
            RecurrentGemmaBlockDiagonalLinear layers of the gates. See the `RecurrentGemmaBlockDiagonalLinear`
            layer for details.
          device: On what device to initialize parameters. Needed to allow for
            initializing the module without parameter initialization.
          dtype: What dtype to use for initialization.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.block_width = self.hidden_size // self.num_attention_heads
        # Parameters and layers.
        self.a_param = nn.Parameter(torch.empty([self.hidden_size]))
        self.input_gate = nn.Linear(self.num_attention_heads * self.block_width, self.block_width) 
        self.a_gate =  nn.Linear(self.num_attention_heads * self.block_width, self.block_width) 

    def __call__(
        self,
        activations: torch.Tensor,
        position_ids: torch.Tensor,
        prev_h: Union[torch.Tensor, None] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calls the RG-LRU.

        Args:
          activations: Sequence of input activations.
          position_ids: Position of each token in the sequence.
          prev_h: The previous hidden state of the RG-LRU.

        Returns:
          Output of the block together with the updated hidden state.
        """

        batch_size, seq_len, hidden_size = activations.shape
        reset = position_ids[:, :, None] == 0

        x = activations.reshape(batch_size, seq_len, self.num_attention_heads, hidden_size//self.num_attention_heads)
        y = torch.einsum("... h i, h i j -> ... h j", x, self.input_gate.weight.reshape(self.num_attention_heads, self.block_width, self.block_width).transpose(1,2))
        y += self.input_gate.bias
        gate_x = torch.sigmoid(einops.rearrange(y, "... h j -> ... (h j)", h=self.num_attention_heads))

        y = torch.einsum("... h i, h i j -> ... h j", x, self.a_gate.weight.reshape(self.num_attention_heads, self.block_width, self.block_width).transpose(1,2))
        y += self.a_gate.bias
        gate_a = torch.sigmoid(einops.rearrange(y, "... h j -> ... (h j)", h=self.num_attention_heads))

        # vs
        # gate_x = torch.sigmoid(self.input_gate(activations)) # TODO fix me
        # gate_a = torch.sigmoid(self.a_gate(activations))

        # Compute the parameter `A` of the recurrence.
        log_a = -8.0 * gate_a * nn.functional.softplus(self.a_param)
        a_matrix = torch.exp(log_a)
        a_square = torch.exp(2 * log_a)

        # Gate the input.
        gated_x = activations * gate_x

        # Apply gamma normalization to the input. We need to clip the derivatives of
        # `sqrt` in order to prevent NaNs during training in bfloat16.
        multiplier = SqrtBoundDerivative.apply(1 - a_square)
        multiplier = reset + ~reset * multiplier
        normalized_x = gated_x * multiplier.type(activations.dtype)

        y, last_h = rnn_scan(
            x=normalized_x,
            a=a_matrix,
            reset=reset,
            h0=prev_h,
        )
        return y, last_h


class RecurrentGemmaDecoderLayer(nn.Module):
    """Griffin and Hawk's residual block."""

    def __init__(self, config, layer_idx):
        super().__init__()
        # Sub-blocks and layers.
        self.temporal_pre_norm = RecurrentGemmaRMSNorm(config.hidden_size)
        self.temporal_block = TEMPORAL_BLOCK_CLASSES[config.block_types[layer_idx]](config, layer_idx)
        self.channel_pre_norm = RecurrentGemmaRMSNorm(config.hidden_size)
        self.mlp_block = RecurrentGemmaMLP(config)

    def forward(
        self,
        activations: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor = None,
        cache: Union[dict[str, torch.Tensor], None] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw_activations = activations

        inputs_normalized = self.temporal_pre_norm(raw_activations)
        hidden_states, cache = self.temporal_block(
            inputs_normalized,
            position_ids,
            attention_mask,
            cache,
            cache_position=cache_position,
        )

        residual = hidden_states + raw_activations

        hidden_states = self.channel_pre_norm(residual)
        hidden_states = self.mlp_block(hidden_states)

        hidden_states = hidden_states + residual

        return hidden_states


@dataclass
class GriffinOutput(ModelOutput):
    """
    Class for the Griffin model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache (`HybridCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache: Optional[HybridCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class GriffinCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache (`HybridCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache: Optional[HybridCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Union[tuple, None] = None


# TODO(lberrada, botev): adapt all doctsrings.

RECURRENTGEMMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RecurrentGemmaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare RecurrentGemma Model outputting raw hidden-states without any specific head on top.",
    RECURRENTGEMMA_START_DOCSTRING,
)
class RecurrentGemmaPreTrainedModel(PreTrainedModel):
    config_class = RecurrentGemmaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RecurrentGemmaDecoderLayer"]
    _skip_keys_device_placement = ["cache"]
    # TODO(lberrada, botev): decide whether we want to support the various implementations of attention
    # in first version.
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = False

    def _init_weights(self, module):
        pass
        # TODO add the missing init schemes


RECURRENTGEMMA_INPUTS_DOCSTRING = r"""
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

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        cache (`HybridCache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_attention_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention  See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all  See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare RecurrentGemma Model outputting raw hidden-states without any specific head on top.",
    RECURRENTGEMMA_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaModel with LLAMA->RECURRENTGEMMA,Llama->RecurrentGemma
class RecurrentGemmaModel(RecurrentGemmaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`RecurrentGemmaDecoderLayer`]

    Args:
        config: RecurrentGemmaConfig
    """

    def __init__(self, config: RecurrentGemmaConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [RecurrentGemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_norm = RecurrentGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(RECURRENTGEMMA_INPUTS_DOCSTRING)
    # Ignore copy
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cache: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, GriffinOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache is None:
            cache = HybridCache(self.config, inputs_embeds.shape[0], self.config.attention_window_size, inputs_embeds.device)

        hidden_states = inputs_embeds

        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        # normalized
        if self.config.embeddings_scale_by_sqrt_dim:
            normalizer = torch.tensor(self.config.hidden_size**0.5)
            hidden_states = hidden_states * normalizer.type(torch.bfloat16)

        all_hidden_states = () if output_hidden_states else None
        for residual_block in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    residual_block.__call__, hidden_states, position_ids, causal_mask, cache_position, cache
                )
            else:
                hidden_states = residual_block(hidden_states, position_ids, causal_mask, cache_position, cache)

        hidden_states = self.final_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache, all_hidden_states] if v is not None)

        return GriffinOutput(
            last_hidden_state=hidden_states,
            cache=cache,
            hidden_states=all_hidden_states,
        )

    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
    # Ignore copy
    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = self.config.attention_window_size

        diagonal = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        offset = cache_position[-1] - self.config.attention_window_size - 1
        causal_mask = torch.tril(diagonal, diagonal=offset)
        if sequence_length != 1:
            causal_mask += torch.triu(diagonal, diagonal=1)

        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

        # Sliding window:
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


# Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM with LLAMA->RECURRENTGEMMA,Llama->RecurrentGemma,llama->gemma
class RecurrentGemmaForCausalLM(RecurrentGemmaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = RecurrentGemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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

    # Ignore copy
    @add_start_docstrings_to_model_forward(RECURRENTGEMMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GriffinCausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, GriffinCausalLMOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RecurrentGemmaForCausalLM

        >>> model = RecurrentGemmaForCausalLM.from_pretrained("google/gemma-7b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            cache_position=cache_position,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # Soft-cap the logits
        if self.config.logits_soft_cap is not None:
            c = self.config.logits_soft_cap
            logits = nn.functional.tanh(logits / c) * c

        logits = logits.float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return GriffinCausalLMOutput(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=(),
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        past_length = 0
        if past_key_values is not None:
            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            max_cache_length = self.config.attention_window_size
            cache_length = torch.min(max_cache_length, past_length)

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
