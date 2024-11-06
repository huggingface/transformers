# coding=utf-8
# Copyright 2024 Zyphra Technologies and the HuggingFace Inc. team. All rights reserved.
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
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from itertools import cycle

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
from ...modeling_flash_attention_utils import _flash_attention_forward
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from ...utils.import_utils import (
    is_causal_conv1d_available,
    is_mamba_ssm_available,
    is_torchdynamo_compiling,
)


if is_mamba_ssm_available():
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
else:
    selective_state_update, mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined = None, None, None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all((selective_state_update, causal_conv1d_fn, causal_conv1d_update))

from ..zamba.modeling_zamba import (
    ZambaMambaDecoderLayer,
    ZambaAttention,
    ZambaForCausalLM,
    ZambaForSequenceClassification,
    ZambaModel,
    ZambaPreTrainedModel,
    ZambaRMSNorm,
    repeat_kv,
)
from ...configuration_utils import PretrainedConfig


_CONFIG_FOR_DOC = "Zyphra/Zamba2-2.7B"

logger = logging.get_logger(__name__)


class Zamba2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Zamba2Model`]. It is used to instantiate a
    Zamba2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Zamba2 model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Zamba2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Zamba2Model`]
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        hidden_size (`int`, *optional*, defaults to 2560):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 54):
            Number of hidden layers in the model.
        mamba_d_state (`int`, *optional*, defaults to 64): shape of the state space latents.
        mamba_d_conv (`int`, *optional*, defaults to 4): Size of the convolution kernel.
        mamba_expand (`int`, *optional*, defaults to 2): Expanding factor used to determine the intermediate size.
        mamba_headdim (`int`, *optional*, defaults to 64):
            Dimension of each mamba head.
        mamba_ngroups (`int`, *optional*, defaults to 8):
            Number of groups for the evolution matrices of mamba 2.
        time_step_min (`float`, *optional*, defaults to 0.001):
            Minimum `time_step` used to bound `dt_proj.bias`.
        time_step_max (`float`, *optional*, defaults to 0.1):
            Maximum `time_step` used to bound `dt_proj.bias`.
        time_step_floor (`float`, *optional*, defaults to 0.0001):
            Minimum clamping value of the `dt_proj.bias` layer initialization.
        time_step_limit (`tuple`, *optional*, defaults to `(0.0, inf)`):
            Accepted range of time step values.
        mamba_dt_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
            Rank of the discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
        n_mamba_heads (`int`, *optional*, defaults to 1):
            Number of heads for the evolution matrices of mamba 2.
        use_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias in the convolution layer of the mixer block.
        mamba_proj_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in ["in_proj", "out_proj"] of the mixer block
        hidden_mamba_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) adjacent to the mamba conv.
        chunk_size (`int`, *optional*, defaults to 256):
            Size of the chunks that will comprise the sequence.
        add_bias_linear (`bool`, *optional*, defaults to `False`):
            Flag indicating whether or not to use bias in various layers
        intermediate_size (`int`, *optional*, defaults to 4 * hidden_size):
            Dimension of the MLP representations.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the MLP.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=None`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf).
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_mem_blocks (`int`, *optional*, defaults to 1):
            Number of unshared transformer blocks.
        use_shared_block_lora (`bool`, *optional*, defaults to `True`):
            If True, unshared LoRA's will be added to the shared MLP's.
        use_shared_attention_lora (`bool`, *optional*, defaults to `False`):
            If True, unshared LoRA's will be added to the q, k, v projectors in the shared attention layers.
        lora_rank (`int`, *optional*, defaults to 128):
            Rank of the LoRA in the shared MLP and shared attention layers.
        use_mem_rope (`bool`, *optional*, defaults to `False`):
            If True, includes RoPE in the shared attention layers.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
            Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
            integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
            logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
            sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
            significantly.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
    """

    model_type = "zamba2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        hidden_size=2560,
        num_hidden_layers=54,
        
        mamba_d_state=64,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_headdim=64,
        mamba_ngroups=1,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=1e-4,
        time_step_limit=(0.0, float("inf")),
        
        mamba_dt_rank="auto",
        n_mamba_heads=1,
        mamba_proj_bias=False,
        hidden_mamba_act="silu",
        use_conv_bias=True,
        chunk_size=256,

        add_bias_linear=False,       
        intermediate_size=None,
        hidden_act="gelu",
        num_attention_heads=32,
        num_key_value_heads=None,
        attention_dropout=0.0,
        
        num_mem_blocks=1,
        use_shared_block_lora=True,
        use_shared_attention_lora=False,
        lora_rank=128,
        use_mem_rope=False,
        rope_theta=10000,
        
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        num_logits_to_keep=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,

        use_long_context=False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        if intermediate_size is None:
            self.intermediate_size = 4 * hidden_size
        else:
            self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_mem_blocks = num_mem_blocks
        self.use_mem_rope = use_mem_rope
        self.rope_theta = rope_theta
        self.attention_hidden_size = 2 * hidden_size
        self.attention_head_dim = 2 * self.hidden_size // self.num_attention_heads
        self.attention_dropout = attention_dropout
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = math.ceil(self.hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        self.add_bias_linear = add_bias_linear
        self.mamba_headdim = mamba_headdim
        self.mamba_ngroups = mamba_ngroups
        self.n_mamba_heads = n_mamba_heads
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.hidden_mamba_act = hidden_mamba_act
        self.use_conv_bias = use_conv_bias
        self.chunk_size = chunk_size
        self.time_step_limit = time_step_limit
        
        self.use_shared_block_lora = use_shared_block_lora
        self.use_shared_attention_lora = use_shared_attention_lora
        self.lora_rank = lora_rank
        self.use_long_context=use_long_context
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        if use_long_context:
            self.max_position_embeddings = 16384
        
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.num_attention_heads = num_attention_heads
        self.kv_channels = self.hidden_size // self.num_attention_heads
        self.num_query_groups = self.num_attention_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        if intermediate_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep


        # Below, "mamba" stands for mamba layer, "hybrid" stands for hybrid layer (composed by a shared transformer followed by mamba layer)
        self.layers_block_type = ['mamba'] + (['mamba'] * 5 + ['hybrid']) * 7 + ['mamba'] * 4 + ['hybrid'] + ['mamba'] * 3 + ['hybrid'] + ['mamba'] * 2


def count_mem_blocks_in_config(config: Zamba2Config):
    """
    Count number of shared blocks
    """
    num_gs = 0
    for val in config.layers_block_type:
        if val == 'hybrid':
            num_gs +=1
    return num_gs


def layer_type_list(config: Zamba2Config):
    """
    Returns list of layer ids containing hybrid layers
    """
    ll = []
    i = 0    
    for val in config.layers_block_type:
        if val == 'hybrid':
            ll.append(i)
        i += 1
    return ll


# Helper methods for segment sum computation


def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    # pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if input_tensor.ndim == 4 else (0, 0, 0, pad_size, 0, 0)

    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)

    # if len(input_tensor.shape) == 3:
    if input_tensor.ndim == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )


def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


class Zamba2DynamicCache(DynamicCache):
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
    (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for mamba cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
    For mamba layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
    and `ssm_states` represents the ssm state and has a shape of `(batch_size, d_inner, d_state)`.
    """

    def __init__(
        self, config: Zamba2Config, batch_size: int, dtype: torch.dtype = torch.float16, device: Optional[str] = None
    ):
        self.layers_block_type = config.layers_block_type
        self.transformer_layers = []
        
        self.has_previous_state = False
        self.dtype = dtype
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = int(config.mamba_expand * config.hidden_size)

        self.conv_states = {
            i: torch.zeros(
                batch_size,
                self.intermediate_size + 2 * config.mamba_ngroups * config.mamba_d_state,
                self.conv_kernel_size,
                device=device,
                dtype=dtype,
            )
            for i in range(config.num_hidden_layers)
        }
        num_heads = self.intermediate_size // config.mamba_headdim
        self.ssm_states = {
            i: torch.zeros(
                batch_size, num_heads, config.mamba_headdim, config.mamba_d_state, device=device, dtype=dtype
            )
            for i in range(config.num_hidden_layers)
        }
        
        for i in range(config.num_hidden_layers):
            if self.layers_block_type[i] == "hybrid":
                self.transformer_layers.append(i)

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_position: torch.LongTensor
    ) -> torch.Tensor:
        conv_state = self.conv_states[layer_idx]
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = new_conv_state.to(conv_state.device)
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()
    
    # Copied from transformers.models.jamba.modeling_jamba.Zamba2DynamicCache.update
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # Copied from transformers.models.jamba.modeling_jamba.Zamba2DynamicCache.reorder_cache
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

            device = self.conv_states[layer_idx].device
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(device))
            device = self.ssm_states[layer_idx].device
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(0, beam_idx.to(device))

    # Copied from transformers.models.jamba.modeling_jamba.Zamba2DynamicCache.get_seq_length
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # take any layer that contains cache and not empty tensor
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    # Copied from transformers.models.jamba.modeling_jamba.Zamba2DynamicCache.to_legacy_cache
    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("Zamba2DynamicCache does not have a legacy cache equivalent.")

    @classmethod
    # Copied from transformers.models.jamba.modeling_jamba.Zamba2DynamicCache.from_legacy_cache
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        raise NotImplementedError("Zamba2DynamicCache does not have a legacy cache equivalent.")


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Zamba
class Zamba2RMSNorm(ZambaRMSNorm):
    pass


class MambaRMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)
    

ALL_LAYERNORM_LAYERS.append(Zamba2RMSNorm)


class Zamba2RotaryEmbedding(nn.Module):
    def __init__(self, config, dim, max_position_embeddings=4096, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        if config.use_long_context:
            a = 8
            base = base * a ** (dim / (dim-2))
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        

    @torch.no_grad()
    # Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
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


class Zamba2Attention(ZambaAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. 

    Adapted from transformers.models.mistral.modeling_mistral.MistralAttention:
    The input dimension here is attention_hidden_size = 2 * hidden_size, and head_dim = attention_hidden_size // num_heads.
    The extra factor of 2 comes from the input being the concatenation of original_hidden_states with the output of the previous (mamba) layer
    (see fig. 2 in https://arxiv.org/pdf/2405.16712).
    Additionally, replaced
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim) with
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim/2)
    Finally, this attention layer contributes to tied transformer blocks aimed to increasing compute without increasing model size. Because this 
    layer is tied, un-tied LoRA modules are added to the q, k, v projectors to increase expressivity with a small memory overhead.
    """

    def __init__(self, config: Zamba2Config, layer_idx: Optional[int] = None, num_fwd_mem_blocks = None, block_id: int = None):
        super().__init__(config, layer_idx)
        self.num_fwd_mem_blocks = num_fwd_mem_blocks
        self.rope_theta = config.rope_theta
        self.layer_block_map = layer_type_list(config)
        self.block_id = block_id
        self.is_causal = True

        if config.use_shared_attention_lora:
            self.linear_q_lora_A_list = nn.ModuleList([])
            self.linear_q_lora_B_list = nn.ModuleList([])
            self.linear_k_lora_A_list = nn.ModuleList([])
            self.linear_k_lora_B_list = nn.ModuleList([])
            self.linear_v_lora_A_list = nn.ModuleList([])
            self.linear_v_lora_B_list = nn.ModuleList([])
            
            for i in range(self.num_fwd_mem_blocks):
                if i % config.num_mem_blocks == block_id:
                    linear_q_lora_A = nn.Linear(self.attention_hidden_size, self.config.lora_rank, bias = False)
                    linear_q_lora_B = nn.Linear(self.config.lora_rank, self.attention_hidden_size, bias = False)
                    linear_k_lora_A = nn.Linear(self.attention_hidden_size, self.config.lora_rank, bias = False)
                    linear_k_lora_B = nn.Linear(self.config.lora_rank, self.attention_hidden_size, bias = False)
                    linear_v_lora_A = nn.Linear(self.attention_hidden_size, self.config.lora_rank, bias = False)
                    linear_v_lora_B = nn.Linear(self.config.lora_rank, self.attention_hidden_size, bias = False)
                else:
                    linear_q_lora_A = nn.Identity()
                    linear_q_lora_B = nn.Identity()
                    linear_k_lora_A = nn.Identity()
                    linear_k_lora_B = nn.Identity()
                    linear_v_lora_A = nn.Identity()
                    linear_v_lora_B = nn.Identity()
                self.linear_q_lora_A_list.append(linear_q_lora_A)
                self.linear_q_lora_B_list.append(linear_q_lora_B)
                self.linear_k_lora_A_list.append(linear_k_lora_A)
                self.linear_k_lora_B_list.append(linear_k_lora_B)
                self.linear_v_lora_A_list.append(linear_v_lora_A)
                self.linear_v_lora_B_list.append(linear_v_lora_B)

        if config.use_mem_rope:
            self.rotary_emb = Zamba2RotaryEmbedding(
                config,
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=self.rope_theta,
            )

        self.layer_dic = {value: index for index, value in enumerate(self.layer_block_map)}

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Zamba2DynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.use_shared_attention_lora:
            lora_layer_idx = self.layer_dic[layer_idx]
            linear_q_lora_A = self.linear_q_lora_A_list[lora_layer_idx]
            linear_q_lora_B = self.linear_q_lora_B_list[lora_layer_idx]
            q_lora_output = linear_q_lora_A(hidden_states)
            q_lora_output = linear_q_lora_B(q_lora_output)
            query_states = self.q_proj(hidden_states)
            query_states = query_states + q_lora_output
            linear_k_lora_A = self.linear_k_lora_A_list[lora_layer_idx]
            linear_k_lora_B = self.linear_k_lora_B_list[lora_layer_idx]
            k_lora_output = linear_k_lora_A(hidden_states)
            k_lora_output = linear_k_lora_B(k_lora_output)
            key_states = self.k_proj(hidden_states)
            key_states = key_states + k_lora_output
            linear_v_lora_A = self.linear_v_lora_A_list[lora_layer_idx]
            linear_v_lora_B = self.linear_v_lora_B_list[lora_layer_idx]
            v_lora_output = linear_v_lora_A(hidden_states)
            v_lora_output = linear_v_lora_B(v_lora_output)
            value_states = self.v_proj(hidden_states)
            value_states = value_states + v_lora_output
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.config.use_mem_rope:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim / 2)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.attention_hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Adapted from transformers.models.mistral.modeling_mistral.MistralAttention:
# Added softmax_scale = 1 / (query_states.shape[-1]/2)**0.5 to the arguments of self._flash_attention_forward
# dropped use_sliding_windows from the arguments of self._flash_attention_forward
class Zamba2FlashAttention2(Zamba2Attention):
    """
    Zamba2 flash attention module. This module inherits from `Zamba2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Zamba2DynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        if self.config.use_shared_attention_lora:
            linear_q_lora_A = self.linear_q_lora_A_list[layer_idx]
            linear_q_lora_B = self.linear_q_lora_B_list[layer_idx]
            q_lora_output = linear_q_lora_A(hidden_states)
            q_lora_output = linear_q_lora_B(q_lora_output)
            query_states = self.q_proj(hidden_states)
            query_states = query_states + q_lora_output
            linear_k_lora_A = self.linear_k_lora_A_list[layer_idx]
            linear_k_lora_B = self.linear_k_lora_B_list[layer_idx]
            k_lora_output = linear_k_lora_A(hidden_states)
            k_lora_output = linear_k_lora_B(k_lora_output)
            key_states = self.k_proj(hidden_states)
            key_states = key_states + k_lora_output
            linear_v_lora_A = self.linear_v_lora_A_list[layer_idx]
            linear_v_lora_B = self.linear_v_lora_B_list[layer_idx]
            v_lora_output = linear_v_lora_A(hidden_states)
            v_lora_output = linear_v_lora_B(v_lora_output)
            value_states = self.v_proj(hidden_states)
            value_states = value_states + v_lora_output
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.config.use_mem_rope:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)        

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        softmax_scale = 1 / math.sqrt(self.head_dim / 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=softmax_scale,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.attention_hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Adapted from transformers.models.mistral.modeling_mistral.MistralAttention:
# added scale = 1 / (query_states.shape[-1]/2)**0.5 to the arguments of torch.nn.functional.scaled_dot_product_attention
class Zamba2SdpaAttention(Zamba2Attention):
    """
    Zamba2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Zamba2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Zamba2DynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Zamba2Model is using Zamba2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.use_shared_attention_lora:
            linear_q_lora_A = self.linear_q_lora_A_list[layer_idx]
            linear_q_lora_B = self.linear_q_lora_B_list[layer_idx]
            q_lora_output = linear_q_lora_A(hidden_states)
            q_lora_output = linear_q_lora_B(q_lora_output)
            query_states = self.q_proj(hidden_states)
            query_states = query_states + q_lora_output
            linear_k_lora_A = self.linear_k_lora_A_list[layer_idx]
            linear_k_lora_B = self.linear_k_lora_B_list[layer_idx]
            k_lora_output = linear_k_lora_A(hidden_states)
            k_lora_output = linear_k_lora_B(k_lora_output)
            key_states = self.k_proj(hidden_states)
            key_states = key_states + k_lora_output
            linear_v_lora_A = self.linear_v_lora_A_list[layer_idx]
            linear_v_lora_B = self.linear_v_lora_B_list[layer_idx]
            v_lora_output = linear_v_lora_A(hidden_states)
            v_lora_output = linear_v_lora_B(v_lora_output)
            value_states = self.v_proj(hidden_states)
            value_states = value_states + v_lora_output
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.config.use_mem_rope:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        softmax_scale = 1 / math.sqrt(self.head_dim / 2)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
            scale=softmax_scale,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.attention_hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


ZAMBA2_ATTENTION_CLASSES = {
    "eager": Zamba2Attention,
    "flash_attention_2": Zamba2FlashAttention2,
    "sdpa": Zamba2SdpaAttention,
}


class Zamba2MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: Zamba2Config, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = int(config.mamba_expand * self.hidden_size)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias # add this with default True
        self.activation = "silu"
        self.act = nn.SiLU()

        self.n_groups = config.mamba_ngroups
        self.head_dim = config.mamba_headdim
        self.num_heads = self.intermediate_size // self.head_dim
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit # add this with default (0.0, float("inf"))
        self.time_step_min = config.time_step_min # add this, with same default as zamba1
        self.time_step_max = config.time_step_max # add this, with same default as zamba1

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=True,
            kernel_size=config.mamba_d_conv,
            groups=self.conv_dim,
            padding=config.mamba_d_conv - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.add_bias_linear,
        )
        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = MambaRMSNormGated(self.intermediate_size, eps=1e-5)
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.add_bias_linear)

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, causal_conv1d_fn, causal_conv1d_update)`"
                " is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Zamba2DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # set up dimensions for reshapes later

        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_to_remove = 2 * self.intermediate_size + 2 * self.n_groups * self.ssm_state_size + self.num_heads

        # getting projected states from cache if it exists
        if cache_params is not None and cache_params.has_previous_state:
            in_projected_states = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
            d_mlp = (in_projected_states.shape[-1] - d_to_remove) // 2
            split_projection_dim = [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads]
            _, _, gate, hidden_states_B_C, dt = torch.split(in_projected_states, split_projection_dim, dim=-1)

            hidden_states_B_C = causal_conv1d_update(
                hidden_states_B_C,
                cache_params.conv_states[self.layer_idx],
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )

            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                dim=-1,
            )
            A = -torch.exp(self.A_log.float())  # (nheads,)

            A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
            hidden_states_reshaped = hidden_states.view(batch_size, self.num_heads, self.head_dim)
            hidden_states = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            hidden_states = hidden_states.view(batch_size, self.num_heads * self.head_dim)
            hidden_states = self.norm(hidden_states, gate)
            out = self.out_proj(hidden_states)[:, None, ...]
        # if no cache is found, calling the kernel
        else:
            if attention_mask is not None and not torch.all(attention_mask==1):
                # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                dtype = hidden_states.dtype
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
            # 1. Gated MLP's linear projection
            projected_states = self.in_proj(hidden_states)
            A = -torch.exp(self.A_log.float())  # (num_heads) or (intermediate_size, state_size)
            dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}
            if attention_mask is not None:
                input_not_masked = torch.all(attention_mask==1)
            else:
                input_not_masked = True

            if self.training and cache_params is None and input_not_masked:
                out, ssm_state = mamba_split_conv1d_scan_combined(
                    projected_states,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=None,  # was seq_idx
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight,
                    rmsnorm_eps=self.norm.variance_epsilon,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=self.head_dim,
                    ngroups=self.n_groups,
                    norm_before_gate=False,
                    return_final_states=True,
                    **dt_limit_kwargs,
                )

            else:
                gate, hidden_states_B_C, time_step = torch.split(
                    projected_states,
                    [self.intermediate_size, self.conv_dim, self.num_heads],
                    dim=-1,
                )

                # 1D Convolution
                hidden_states_B_C_t = hidden_states_B_C.transpose(1,2)
                conv_state = nn.functional.pad(
                    hidden_states_B_C_t,
                    (self.conv_kernel_size - hidden_states_B_C_t.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                    hidden_states_B_C = self.act(
                        self.conv1d(hidden_states_B_C.transpose(1, 2)).transpose(1, 2)[:, :seq_len]
                    )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                else:
                    hidden_states_B_C = causal_conv1d_fn(
                        x=hidden_states_B_C.transpose(1, 2),
                        weight=self.conv1d.weight.squeeze(1),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    ).transpose(1, 2)[:, :seq_len]
                hidden_states, B, C = torch.split(
                    hidden_states_B_C,
                    [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                    dim=-1,
                )
                if attention_mask is not None and not torch.all(attention_mask==1):
                    # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                    dtype = hidden_states.dtype
                    hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
                scan_output, ssm_state = mamba_chunk_scan_combined(
                    hidden_states.view(batch_size, seq_len, -1, self.head_dim),
                    time_step,
                    A,
                    B.view(batch_size, seq_len, self.n_groups, -1),
                    C.view(batch_size, seq_len, self.n_groups, -1),
                    chunk_size=self.chunk_size,
                    D=self.D,
                    z=None,
                    seq_idx=None,
                    return_final_states=True,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    **dt_limit_kwargs,
                )
                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
                scan_output = scan_output.view(batch_size, seq_len, -1)
                # Multiply "gate" branch and apply extra normalization layer
                scan_output = self.norm(scan_output, gate)
                out = self.out_proj(scan_output)
        return out

    # fmt: off
    def torch_forward(self, input_states, cache_params: Optional[Zamba2DynamicCache]=None, cache_position:Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # Gated MLP's linear projection
        projected_states =  self.in_proj(input_states.squeeze(1))
        d_mlp = (projected_states.shape[-1] - 2 * self.intermediate_size -  2 * self.n_groups * self.ssm_state_size- self.num_heads) // 2
        _, _, gate, hidden_states, dt = projected_states.split(
                [d_mlp, d_mlp, self.intermediate_size,  self.conv_dim, self.num_heads], dim=-1
        )

        # Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            ssm_state = ssm_state.to(hidden_states.device)
            if cache_params.has_previous_state:
                conv_state = cache_params.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                # handle batched generation - states are copied through
                conv_state[:, :, -1] = hidden_states[:, 0, :] if hidden_states.ndim == 3 else hidden_states
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = torch.sum(conv_state.to(projected_states.device) * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype)[:, None, ...]         # [batch, 1, intermediate_size] : decoding
            else:
                hidden_states = hidden_states.transpose(1,2)
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = self.act(self.conv1d(hidden_states).transpose(1,2))[:, :seq_len, :]     # [batch, intermediate_size, seq_len]
                if attention_mask is not None and not torch.all(attention_mask==1):
                    dtype = hidden_states.dtype
                    # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
                    hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
        else:
            ssm_state = torch.zeros(
                (batch_size, self.num_heads, self.head_dim, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        hidden_states, B, C = torch.split(hidden_states, [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size], dim=-1)
        A = -torch.exp(self.A_log.float())                            # [num_heads]
        if cache_params is not None and cache_params.has_previous_state:
            # Note: there is no need to pad parameter matrices here, as there is just one new token
            # for batched generation
            dt = dt[:, None, ...] if dt.ndim == 2 else dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            # [num_heads] -> [num_heads, head_dim]
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

            dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_min) #, self.time_step_max)
            A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            # [bsz, num_heads, head_dim, state_size]
            dA = torch.exp(dt[..., None] * A)

            # Discretize B
            # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
            # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            # [bsz, num_heads, head_dim, state_size]
            dB = dt[..., None] * B[..., None, :]

            # Discretize x into dB
            # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = dB * hidden_states[..., None]

            # State calculation
            cache_params.ssm_states[self.layer_idx].copy_(
                cache_params.ssm_states[self.layer_idx] * dA + dBx
            )

            # Subsequent output
            # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])
            # [bsz, num_heads, head_dim]

            ssm_states = cache_params.ssm_states[self.layer_idx].to(C.dtype)  # Shape: [b, h, d, n]
            # Reshape ssm_states to merge the first two dimensions
            ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)  # Shape: [b*h, d, n]
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)  # Shape: [b*h, n, 1]
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)

            # D skip connection
            # [num_heads] -> [num_heads, head_dim]
            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(y.dtype)

            # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            # begin ssd naive implementation without einsums
            dt = nn.functional.softplus(dt + self.dt_bias)
            dt = torch.clamp(dt, self.time_step_min)
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            B = B.reshape(batch_size, seq_len,  -1, self.ssm_state_size).float()
            C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            B = B.repeat(1, 1, self.num_heads // self.n_groups, 1)
            C = C.repeat(1, 1, self.num_heads // self.n_groups, 1)
            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

            # Discretize x and A
            hidden_states = hidden_states * dt[..., None]
            A = A.to(hidden_states.dtype) * dt

            # Rearrange into blocks/chunks
            hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)]


            # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
            A = A.permute(0, 3, 1, 2)
            A_cumsum = torch.cumsum(A, dim=-1)

            # 1. Compute the output for each intra-chunk (diagonal blocks)
            # This is the analog of a causal mask
            L = torch.exp(segment_sum(A))

            # First, contraction of C and B to get G (attention-weights like)
            G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, : ,:]  # shape: (b, c, l, s, h, n)
            G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)


            # Step 2: Compute M, equivalent to applying attention mask to weights
            M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_intermediate.sum(dim=-1)

            # Step 3: Compute Y_diag (apply to values)
            Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(3)

            # (right term of low-rank factorization of off-diagonal blocks; B terms)

            decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
            B_decay_contraction = B * decay_states.permute(0, 2, 3, 1)[..., None]
            # permute back B * decay states
            states = (B_decay_contraction.permute(0, 1, 3, 2, 4)[..., None]  * hidden_states.permute(0, 1, 3, 2, 4)[..., None, :]).sum(dim=3).permute(0, 1, 2, 4, 3)
            if cache_params is not None and cache_params.has_previous_state:
                previous_states = cache_params.ssm_states[self.layer_idx][:, None, ...]
            else:
                previous_states = torch.zeros_like(states[:, :1])
            states = torch.cat([previous_states, states], dim=1)
            decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))

            states_permuted = states.permute(0, 2, 1, 3, 4)
            result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=2)
            new_states = result.permute(0, 2, 1, 3, 4)
            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            # Compute state -> output conversion per chunk
            # (left term of low-rank factorization of off-diagonal blocks; C terms)
            state_decay_out = torch.exp(A_cumsum)
            # compute Yoff
            C_times_states = (C[..., None, :] * states[:, :, None, ...])
            state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])
            # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)

            y = Y_diag + Y_off
            # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
            y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)

            y = y + D_residual
            # Cutting off padded chunks
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape(batch_size, seq_len, -1)
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        scan_output = self.norm(y, gate)

        # end ssd naive

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def forward(
        self,
        hidden_states,
        cache_params: Optional[Zamba2DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if is_fast_path_available and "cuda" in self.in_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)
        dtype = hidden_states.dtype
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        return self.torch_forward(hidden_states, cache_params, cache_position, attention_mask)


class Zamba2MLP(nn.Module):
    def __init__(self, config: Zamba2Config, num_fwd_mem_blocks = None, block_id: int = None):
        """
        This MLP layer contributes to tied transformer blocks aimed to increasing compute without increasing model size. Because this layer 
        is tied, un-tied LoRA modules are added to the up and gate projectors to increase expressivity with a small memory overhead.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_fwd_mem_blocks = num_fwd_mem_blocks
        self.block_id = block_id
        self.ffn_intermediate_size = config.intermediate_size
        
        self.act_fn = ACT2FN[config.hidden_act]
        def gated_act_fn(x):
            x = torch.chunk(x, 2, dim=-1)

            return self.act_fn(x[0]) * x[1]
        self.gated_act_fn = gated_act_fn

        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.ffn_intermediate_size, bias = config.add_bias_linear)
        self.down_proj = nn.Linear(self.ffn_intermediate_size, self.hidden_size, bias=config.add_bias_linear)
        
        if self.config.use_shared_block_lora:
            self.gate_up_proj_lora_A_list = nn.ModuleList([])
            self.gate_up_proj_lora_B_list = nn.ModuleList([])
            for i in range(self.num_fwd_mem_blocks):
                if i % config.num_mem_blocks == block_id:
                    gate_up_proj_lora_A = nn.Linear(self.config.hidden_size, self.config.lora_rank, bias = False)
                    gate_up_proj_lora_B = nn.Linear(self.config.lora_rank, 2 * self.ffn_intermediate_size, bias = False)
                else:
                    gate_up_proj_lora_A = nn.Identity()
                    gate_up_proj_lora_B = nn.Identity()
                self.gate_up_proj_lora_A_list.append(gate_up_proj_lora_A)
                self.gate_up_proj_lora_B_list.append(gate_up_proj_lora_B)
        
        layer_block_map = layer_type_list(config)
        self.layer_dic = {value: index for index, value in enumerate(layer_block_map)}

    def forward(self, hidden_state, layer_idx = None):
        if self.config.use_shared_block_lora:
            layer_idx = self.layer_dic[layer_idx]
            gate_up_proj_lora_A = self.gate_up_proj_lora_A_list[layer_idx]
            gate_up_proj_lora_B = self.gate_up_proj_lora_B_list[layer_idx]
            lora_output = gate_up_proj_lora_A(hidden_state)
            lora_output = gate_up_proj_lora_B(lora_output)
            intermediate_state = self.gate_up_proj(hidden_state)
            hidden_state = intermediate_state + lora_output
        else:
            hidden_state = self.gate_up_proj(hidden_state)

        hidden_state = self.gated_act_fn(hidden_state)
        output = self.down_proj(hidden_state)
        return output


class Zamba2AttentionDecoderLayer(nn.Module):
    def __init__(self, config: Zamba2Config, block_id: int = None, layer_idx: Optional[int] = None):
        super().__init__()
        num_gs = count_mem_blocks_in_config(config)
        self.block_id = block_id
        self.self_attn = ZAMBA2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=-1, num_fwd_mem_blocks=num_gs, block_id=block_id)
        self.feed_forward = Zamba2MLP(config, num_fwd_mem_blocks=num_gs, block_id=block_id)
        self.input_layernorm = Zamba2RMSNorm(config.attention_hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = Zamba2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Zamba2DynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): output of previous Mamba layer of shape `(batch, seq_len, embed_dim)`
            original_hidden_states (`torch.FloatTensor`): word embedding output of shape `(batch, seq_len, embed_dim)`.
                This is concatenated with `hidden_states` (which is the output of the previous (mamba) layer). The
                concatenated tensor is then used as input of the pre-attention RMSNorm
                (see fig. 2 in https://arxiv.org/pdf/2405.16712).
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Zamba2DynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
        """
        hidden_states = torch.concatenate([hidden_states, original_hidden_states], dim=-1)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            layer_idx=layer_idx,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states, layer_idx)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Zamba2MambaDecoderLayer(ZambaMambaDecoderLayer):
    def __init__(self, config: Zamba2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mamba = Zamba2MambaMixer(config=config, layer_idx=layer_idx)
        self.input_layernorm = Zamba2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Zamba2HybridLayer(nn.Module):
    def __init__(self, shared_transf: Zamba2AttentionDecoderLayer, linear: nn.Linear, mamba: Zamba2MambaDecoderLayer):
        super().__init__()
        self.shared_transf = shared_transf
        self.linear = linear
        self.mamba_decoder = mamba

    def forward(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: Optional[torch.Tensor] = None,
        layer_idx: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Zamba2DynamicCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            original_hidden_states (`torch.FloatTensor`): word embedding output that will be concatenated with
            hidden activations to form the input of the shared transformer layer.
            layer_idx (`int`): layer number.
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Zamba2DynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
        """

        layer_outputs = self.shared_transf(
            hidden_states,
            original_hidden_states=original_hidden_states,
            layer_idx=layer_idx,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        transformer_hidden_states = layer_outputs[0]

        if output_attentions:
            self_attn_weights = layer_outputs[1]

        transformer_hidden_states = self.linear(transformer_hidden_states)

        layer_outputs = self.mamba_decoder(
            hidden_states,
            transformer_hidden_states=transformer_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        if output_attentions:
            layer_outputs = (layer_outputs[0], self_attn_weights) + layer_outputs[2:]

        return layer_outputs

ZAMBA2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Zamba2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Zamba2 Model outputting raw hidden-states without any specific head on top.",
    ZAMBA2_START_DOCSTRING,
)
class Zamba2PreTrainedModel(PreTrainedModel):
    config_class = Zamba2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Zamba2AttentionDecoderLayer", "Zamba2MambaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True  # Note: only supports Zamba2DynamicCache
    _is_stateful = True
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Zamba2MambaMixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            num_heads = int(self.config.mamba_expand * self.config.hidden_size) // self.config.mamba_headdim
            dt = torch.exp(
                torch.rand(num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                module.dt_bias.copy_(inv_dt)
            module.dt_bias._no_reinit = True


ZAMBA2_INPUTS_DOCSTRING = r"""
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
        past_key_values (`Zamba2DynamicCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            A Zamba2DynamicCache object containing pre-computed hidden-states (keys and values in the
            self-attention blocks and convolution and ssm states in the mamba blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
            Key and value cache tensors have shape `(batch_size, num_heads, seq_len, head_dim)`.
            Convolution and ssm states tensors have shape `(batch_size, d_inner, d_conv)` and
            `(batch_size, d_inner, d_state)` respectively.
            See the `Zamba2DynamicCache` class for more details.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `input_ids` of shape `(batch_size, sequence_length)`.
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
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Zamba2 Model outputting raw hidden-states without any specific head on top.",
    ZAMBA2_START_DOCSTRING,
)
class Zamba2Model(ZambaModel, Zamba2PreTrainedModel):
    """
    Model consisting of *config.num_hidden_layers* layers.

    Args:
        config: Zamba2Config
    """

    def __init__(self, config: Zamba2Config):
        Zamba2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        blocks = [Zamba2AttentionDecoderLayer(config, block_id=k) for k in range(config.num_mem_blocks)]
        mamba_layers = []
        linear_layers = []
        self.layers_block_type = config.layers_block_type
        for i in range(config.num_hidden_layers):
            if config.layers_block_type[i] == "mamba":
                mamba_layers.append(Zamba2MambaDecoderLayer(config, layer_idx=i))
            elif config.layers_block_type[i] == "hybrid":
                linear_layers.append(nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False))
                mamba_layers.append(Zamba2MambaDecoderLayer(config, layer_idx=i))
        mamba_layers = iter(mamba_layers)
        linear_layers = iter(linear_layers)
        blocks = cycle(blocks)
        layers = []
        self._tied_weights_keys = []
        for layer_id, layer_type in enumerate(self.layers_block_type):
            if layer_type == "hybrid":
                prefix_name = f"layers.{layer_id}."
                tied_keys = [
                    "shared_transf.self_attn.q_proj.weight",
                    "shared_transf.self_attn.k_proj.weight",
                    "shared_transf.self_attn.v_proj.weight",
                    "shared_transf.self_attn.o_proj.weight",
                    "shared_transf.feed_forward.gate_proj.weight",
                    "shared_transf.feed_forward.up_proj.weight",
                    "shared_transf.feed_forward.down_proj.weight",
                    "shared_transf.input_layernorm.weight",
                    "shared_transf.pre_ff_layernorm.weight",
                ]
                self._tied_weights_keys = [*self._tied_weights_keys, *[prefix_name + key for key in tied_keys]]
                layers.append(Zamba2HybridLayer(next(blocks), next(linear_layers), next(mamba_layers)))
            else:
                layers.append(next(mamba_layers))
        self.layers = nn.ModuleList(layers)

        self._attn_implementation = config._attn_implementation
        self.final_layernorm = Zamba2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        
        # Initialize weights and apply final processing
        self.post_init()


# Adapted from transformers.models.jamba.modeling_jamba.JambaForCausalLM with Jamba->Zamba, JAMBA->ZAMBA
class Zamba2ForCausalLM(ZambaForCausalLM, Zamba2PreTrainedModel, GenerationMixin):
    def __init__(self, config: Zamba2Config):
        super().__init__(config)
        self.model = Zamba2Model(config)
        self._tied_weights_keys = ["lm_head.weight", *self.model._tied_weights_keys]

        # Initialize weights and apply final processing
        self.post_init()
    
    # Adapted from transformers.models.zamba.modeling_zamba.ZambaForCausalLM.prepare_inputs_for_generation
    # with `Zamba2DynamicCache` -> `Zamba2DynamicCache`
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # Overwitten -- has a unique cache type, `Zamba2DynamicCache`

        empty_past_kv = past_key_values is None

        # Omit tokens covered by past_key_values
        if not empty_past_kv:
            # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
            # Exception 1: when passing input_embeds, input_ids may be missing entries
            # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        else:
            past_key_values = Zamba2DynamicCache(
                self.config, input_ids.shape[0], dtype=self.dtype, device=self.device
            )

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty_past_kv:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "num_logits_to_keep": self.config.num_logits_to_keep,
                "cache_position": cache_position,
            }
        )
        return model_inputs


@add_start_docstrings(
    """
    The Zamba2 Model with a sequence classification head on top (linear layer).

    [`Zamba2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    ZAMBA2_START_DOCSTRING,
)
class Zamba2ForSequenceClassification(ZambaForSequenceClassification, Zamba2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Zamba2Model(config)
        self._tied_weights_keys = self.model._tied_weights_keys

        # Initialize weights and apply final processing
        self.post_init()