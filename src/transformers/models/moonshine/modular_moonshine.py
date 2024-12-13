from ...configuration_utils import PretrainedConfig
from ..phi.modeling_phi import PhiAttention, PhiFlashAttention2, PhiSdpaAttention, PhiMLP, PhiRotaryEmbedding
from ..llama.modeling_llama import LlamaDecoderLayer, LlamaModel
from ..mistral.modeling_mistral import MistralMLP
from ..whisper.modeling_whisper import WhisperModel

from typing import List, Optional, Tuple, Union
from ...processing_utils import Unpack

from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache

from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs, _flash_attention_forward

from ...modeling_utils import PreTrainedModel

import torch.nn as nn
import torch

from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from typing import Optional, Tuple

from ...activations import ACT2FN

import copy
import math

logger = logging.get_logger(__name__)


class MoonshineConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MoonshineModel`]. It is used to instantiate a Moonshine
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Moonshine
    [UsefulSensors/moonshine](https://huggingface.co/UsefulSensors/moonshine).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32768):
            Vocabulary size of the Moonshine model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MoonshineModel`].
        hidden_size (`int`, *optional*, defaults to 288):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder and decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder and decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        encoder_hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder.
        decoder_hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. TODO: check this
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings. TODO: check this
        partial_rotary_factor (`float`, *optional*, defaults to 0.5):
            Percentage of the query and keys which will have rotary embedding. TODO: check this
        ff_mult (`int`, *optional*, defaults to 4):
            Factor by which to scale the intermediate size.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        qk_layernorm (`bool`, *optional*, defaults to `False`):
            Whether or not to normalize the Queries and Keys after projecting the hidden states.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        bos_token_id (`int`, *optional*, defaults to 1):
            Denotes beginning of sequences token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            Denotes end of sequences token id.

    Example:

    ```python
    >>> from transformers import MoonshineModel, MoonshineConfig

    >>> # Initializing a Moonshine style configuration
    >>> configuration = MoonshineConfig().from_pretrained("UsefulSensors/moonshine")

    >>> # Initializing a model from the configuration
    >>> model = MoonshineModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "moonshine"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32768,
        hidden_size=288,
        intermediate_size=None,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=None,
        encoder_hidden_act="gelu",
        decoder_hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        attention_bias=False,
        attention_dropout=0.0,
        qk_layernorm=False,
        rope_scaling=None,
        ff_mult=4,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.encoder_hidden_act = encoder_hidden_act
        self.decoder_hidden_act = decoder_hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor

        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.qk_layernorm = qk_layernorm
        self.rope_scaling = rope_scaling
        self.ff_mult = ff_mult

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

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

def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


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

    q_embed = (q * cos) + (rotate_every_two(q) * sin)
    k_embed = (k * cos) + (rotate_every_two(k) * sin)
    return q_embed, k_embed


class MoonshineRotaryEmbedding(PhiRotaryEmbedding):
    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.stack((freqs, freqs), dim=-1)
            emb = emb.flatten(-2) # in einsum notation: rearrange(x, '... d j -> ... (d j)')
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MoonshineNonGatedMLP(PhiMLP):
    def __init__(self, config: MoonshineConfig, hidden_act: str):
        config = copy.deepcopy(config)
        config.hidden_act = hidden_act
        if config.intermediate_size is None:
            config.intermediate_size = config.hidden_size * config.ff_mult
        super().__init__(config)


class MoonshineGatedMLP(nn.Module):
    def __init__(self, config: MoonshineConfig, hidden_act: str):
        super().__init__()
        config = copy.deepcopy(config)
        config.hidden_act = hidden_act
        if config.intermediate_size is None:
            config.intermediate_size = config.hidden_size * config.ff_mult * 2
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size // 2, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        hidden_state = self.up_proj(hidden_state)
        hidden_state, gate = hidden_state.chunk(2, dim=-1)
        hidden_state = self.act_fn(gate) * hidden_state
        return self.down_proj(hidden_state)
    

class MoonshineMLP:
    def __new__(cls, config: MoonshineConfig, hidden_act: str):
        if hidden_act == "gelu":  
            return MoonshineNonGatedMLP(config, hidden_act)
        elif hidden_act == "silu":
            return MoonshineGatedMLP(config, hidden_act)
        else:
            raise ValueError(f"Unsupported activation function: {hidden_act}, please use 'gelu' or 'silu'")


class MoonshineAttention(PhiAttention):
    def __init__(self, config: MoonshineConfig, layer_idx: Optional[int] = None, is_causal: bool = False):
        super().__init__(config, layer_idx)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_ndims = max(config.hidden_size // config.num_attention_heads // 2, 32)

        self.rotary_emb = MoonshineRotaryEmbedding(
            dim=self.rotary_ndims, 
            max_position_embeddings=config.max_position_embeddings,
        )

        self.is_causal = is_causal
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        key_value_states: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k_proj(current_states)
            value_states = self.v_proj(current_states)

        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if not is_cross_attention: 
            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings

            # Partial rotary embedding
            query_rot, query_pass = (
                query_states[..., : self.rotary_ndims],
                query_states[..., self.rotary_ndims :],
            )
            key_rot, key_pass = (
                key_states[..., : self.rotary_ndims],
                key_states[..., self.rotary_ndims :],
            )
            # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
            query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)

            # [batch_size, seq_length, num_heads, head_dim]
            query_states = torch.cat((query_rot, query_pass), dim=-1)
            key_states = torch.cat((key_rot, key_pass), dim=-1)

        if past_key_value is not None:
            if not is_cross_attention:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "partial_rotation_size": self.rotary_ndims,
                    "cache_position": cache_position,
                }
            else:
                cache_kwargs = {
                    "cache_position": cache_position,
                }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Queries and keys upcast to fp32 is required by Moonshine-2 to avoid overflow
        attn_weights = torch.matmul(
            query_states.to(torch.float32), key_states.to(torch.float32).transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights += causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MoonshineFlashAttention2(PhiFlashAttention2):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        key_value_states: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k_proj(current_states)
            value_states = self.v_proj(current_states)

        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if not is_cross_attention: 
            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings

            # Partial rotary embedding
            query_rot, query_pass = (
                query_states[..., : self.rotary_ndims],
                query_states[..., self.rotary_ndims :],
            )
            key_rot, key_pass = (
                key_states[..., : self.rotary_ndims],
                key_states[..., self.rotary_ndims :],
            )
            # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
            query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)

            # [batch_size, seq_length, num_heads, head_dim]
            query_states = torch.cat((query_rot, query_pass), dim=-1)
            key_states = torch.cat((key_rot, key_pass), dim=-1)

        if past_key_value is not None:
            if not is_cross_attention:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "partial_rotation_size": self.rotary_ndims,
                    "cache_position": cache_position,
                }
            else:
                cache_kwargs = {
                    "cache_position": cache_position,
                }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_dropout = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32.

        if query_states.dtype == torch.float32:
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

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=attn_dropout,
            softmax_scale=None,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MoonshineSdpaAttention(PhiSdpaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        key_value_states: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k_proj(current_states)
            value_states = self.v_proj(current_states)

        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if not is_cross_attention: 
            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings

            # Partial rotary embedding
            query_rot, query_pass = (
                query_states[..., : self.rotary_ndims],
                query_states[..., self.rotary_ndims :],
            )
            key_rot, key_pass = (
                key_states[..., : self.rotary_ndims],
                key_states[..., self.rotary_ndims :],
            )
            # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
            query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)

            # [batch_size, seq_length, num_heads, head_dim]
            query_states = torch.cat((query_rot, query_pass), dim=-1)
            key_states = torch.cat((key_rot, key_pass), dim=-1)

        if past_key_value is not None:
            if not is_cross_attention:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "partial_rotation_size": self.rotary_ndims,
                    "cache_position": cache_position,
                }
            else:
                cache_kwargs = {
                    "cache_position": cache_position,
                }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()` here. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        if self.require_contiguous_qkv and query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if self.is_causal and causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.dense(attn_output)

        return attn_output, None, past_key_value


MOONSHINE_ATTENTION_CLASSES = {
    "eager": MoonshineAttention,
    "flash_attention_2": MoonshineFlashAttention2,
    "sdpa": MoonshineSdpaAttention,
}


class MoonshineEncoderLayer(LlamaDecoderLayer):
    def __init__(self, config: MoonshineConfig, layer_idx: int):
        super().__init__(config, layer_idx)
       
        self.mlp = MoonshineMLP(config, config.encoder_hidden_act)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)


class MoonshineDecoderLayer(nn.Module):
    def __init__(self, config: MoonshineConfig, layer_idx: int = None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MOONSHINE_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx, is_causal=True)
        self.encoder_attn = MOONSHINE_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx, is_causal=True)

        self.mlp = MoonshineMLP(config, config.decoder_hidden_act) 
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        encoder_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            encoder_position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, encoder_seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
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
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=encoder_position_embeddings,
            )
            hidden_states = residual + hidden_states

            # add cross-attn to positions 1 of present_key_value tuple
            present_key_value = (present_key_value, cross_attn_present_key_value)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    

MOONSHINE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MoonshineConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Moonshine Model outputting raw hidden-states without any specific head on top.",
    MOONSHINE_START_DOCSTRING,
)
class MoonshinePreTrainedModel(PreTrainedModel):
    config_class = MoonshineConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MoonshineDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MoonshineEncoder(MoonshinePreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`MoonshineEncoderLayer`].

    Args:
        config: MoonshineConfig
    """

    def __init__(self, config: MoonshineConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.conv1 = nn.Conv1d(1, embed_dim, kernel_size=127, stride=64, bias=False)
        self.conv2 = nn.Conv1d(embed_dim, 2 * embed_dim, kernel_size=7, stride=3)
        self.conv3 = nn.Conv1d(2 * embed_dim, embed_dim, kernel_size=3, stride=2)
        self.groupnorm = nn.GroupNorm(num_groups=1, num_channels=embed_dim, eps=1e-5)

        self.rotary_emb = MoonshineRotaryEmbedding(
            dim=max(config.hidden_size // config.num_attention_heads // 2, 32), 
            max_position_embeddings=config.max_position_embeddings, 
        ) 

        self.layers = nn.ModuleList([MoonshineEncoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps, bias=False)

        self.gradient_checkpointing = False
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, 1, sequence_length)`):
                Float values of the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`).
            attention_mask (`torch.Tensor`)`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1, 
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs_embeds = nn.functional.tanh(self.conv1(input_features))
        inputs_embeds = self.groupnorm(inputs_embeds)
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = nn.functional.gelu(self.conv3(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1) 
        hidden_states = inputs_embeds
        
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        embed_pos = self.rotary_emb(hidden_states, position_ids)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
       
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    None,
                    output_attentions,
                    position_embeddings=embed_pos,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    None,
                    output_attentions=output_attentions,
                    position_embeddings=embed_pos,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
    

class MoonshineDecoder(LlamaModel):
    def __init__(self, config: MoonshineConfig):
        super().__init__(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        self.rotary_emb = MoonshineRotaryEmbedding(
            dim= max(config.hidden_size // config.num_attention_heads // 2, 32), 
            max_position_embeddings=config.max_position_embeddings,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        return_self_attention_cache = False
        if use_cache or past_key_values is not None:
            if isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
                return_self_attention_cache = True
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            elif not isinstance(past_key_values, EncoderDecoderCache):
                return_legacy_cache = True
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.43.0. "
                    "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                    "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
                )
                past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if encoder_position_ids is None:
            encoder_position_ids = torch.arange(
                encoder_hidden_states.shape[1], device=encoder_hidden_states.device
            ).unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        encoder_position_embeddings = self.rotary_emb(encoder_hidden_states, encoder_position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    encoder_hidden_states,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    position_ids=position_ids,
                    encoder_position_ids=encoder_position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    encoder_position_embeddings=encoder_position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None
        if return_self_attention_cache:
            next_cache = past_key_values.self_attention_cache
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
    

class MoonshineModel(WhisperModel):
    def __init__(self, config: MoonshineConfig):
        super().__init__(config)
        self.encoder = MoonshineEncoder(config)
        self.decoder = MoonshineDecoder(config)


