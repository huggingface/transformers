from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers.utils.generic import TransformersKwargs

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import layer_type_validation
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ..olmo2.configuration_olmo2 import Olmo2Config
from ..olmo2.modeling_olmo2 import (
    Olmo2Attention,
    Olmo2DecoderLayer,
    Olmo2ForCausalLM,
    Olmo2Model,
    Olmo2PreTrainedModel,
    Olmo2RMSNorm,
    Olmo2RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask


def get_flex_attn_causal_block_mask(
    seq_len: int,
    device: torch.device,
    num_sink_tokens: int = 0,
    window_size: Optional[tuple[int, int]] = None,
    doc_lens: Optional[tuple[int, ...]] = None,
) -> BlockMask:
    if device is None:
        raise ValueError("Device is required")

    has_window = window_size is not None and window_size != (-1, -1)
    has_docs = doc_lens is not None

    if has_docs:
        document_ids = torch.cat(
            [torch.full((int(doc_len),), i, device=device, dtype=torch.long) for i, doc_len in enumerate(doc_lens)]
        )

    def total_mask_mod(B: torch.Tensor, H: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        is_sink = kv_idx < num_sink_tokens
        adjusted_kv_idx = kv_idx - num_sink_tokens
        causal_mask = q_idx >= adjusted_kv_idx
        if has_window:
            window_mask = (q_idx - adjusted_kv_idx <= window_size[0]) & (adjusted_kv_idx - q_idx <= window_size[1])
            mask = causal_mask & window_mask
        else:
            mask = causal_mask
        if has_docs:
            clamped_idx = torch.clamp(adjusted_kv_idx, min=0, max=len(document_ids) - 1)
            doc_mask = document_ids[q_idx] == document_ids[clamped_idx]
            mask = mask & doc_mask
        return is_sink | mask

    kv_len = seq_len + num_sink_tokens
    return create_block_mask(
        total_mask_mod,
        B=None, H=None,
        Q_LEN=seq_len,
        KV_LEN=kv_len,
        device=device,
        BLOCK_SIZE=128, 
    )


def flex_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    sliding_window: Optional[int] = None,
    doc_lens: Optional[tuple] = None,
    **kwargs,
):
    if dropout != 0.0:
        raise NotImplementedError("Flex attention with dropout not supported yet.")

    B, H, Q, D = query.shape
    _, _, K, _ = key.shape  # K == seq_len

    # Apply GQA by repeating key/value for compatibility
    key = repeat_kv(key, module.num_key_value_groups)  
    value = repeat_kv(value, module.num_key_value_groups)

    s_param = module.sinks
    if s_param is not None:
        if s_param.ndim == 1:
            num_sink_tokens = 1
        elif s_param.ndim == 2:
            num_sink_tokens = s_param.size(1)
        else:
            raise ValueError("module.sinks must have shape [H] or [H, S]")

        sink_k = key.new_zeros(B, H, num_sink_tokens, D)
        sink_v = value.new_zeros(B, H, num_sink_tokens, D)
        key = torch.cat([sink_k, key], dim=2)
        value = torch.cat([sink_v, value], dim=2)

        
        local_sinks = s_param  # Use s_param.to_local() if DTensor

        if local_sinks.ndim == 1:
            def score_mod_fn(score, batch_idx, head_idx, q_idx, kv_idx):
                is_sink = kv_idx < num_sink_tokens
                sink_logit = local_sinks[head_idx].to(score.dtype)
                return torch.where(is_sink, sink_logit, score)
        elif local_sinks.ndim == 2:
            def score_mod_fn(score, batch_idx, head_idx, q_idx, kv_idx):
                is_sink = kv_idx < num_sink_tokens
                safe_kv_idx = torch.clamp(kv_idx, 0, num_sink_tokens - 1)
                sink_logit = local_sinks[head_idx, safe_kv_idx].to(score.dtype)
                return torch.where(is_sink, sink_logit, score)
    else:
        num_sink_tokens = 0
        score_mod_fn = None

    # For sliding window attention support in flex attention
    window_size = None
    if sliding_window is not None:
        # sliding_window parameter comes from the function signature
        window_size = (sliding_window - 1, 0)
    
    block_mask = get_flex_attn_causal_block_mask(
        seq_len=Q,
        device=query.device,
        num_sink_tokens=num_sink_tokens,
        window_size=window_size,
        doc_lens=doc_lens # if intra-doc masking needed
    )

    cast_to_bf16 = query.device.type == 'cuda'
    og_dtype = query.dtype
    if cast_to_bf16:
        query, key, value = query.bfloat16(), key.bfloat16(), value.bfloat16()

    with torch.autocast(enabled=False, device_type=query.device.type):
        attn_out = flex_attention(
            query, key, value,
            block_mask=block_mask,
            scale=scaling,
            score_mod=score_mod_fn,
            enable_gqa=True,
        )

    if cast_to_bf16:
        attn_out = attn_out.to(og_dtype)

    return attn_out.transpose(1, 2).contiguous(), None


def olmo3_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Forward pass using eager attention that matches flex attention logic."""
    B, H, Q, D = query.shape
    _, _, K, _ = key.shape

    # Apply GQA by repeating key/value for compatibility  
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    s_param = module.sinks
    if s_param is not None:
        if s_param.ndim == 1:
            num_sink_tokens = 1
        elif s_param.ndim == 2:
            num_sink_tokens = s_param.size(1)
        else:
            raise ValueError("module.sinks must have shape [H] or [H, S]")

        # Add dummy sink K/V tensors like in flex attention (zeros are fine since we'll override scores)
        sink_k = key.new_zeros(B, H, num_sink_tokens, D)
        sink_v = value.new_zeros(B, H, num_sink_tokens, D)
        key = torch.cat([sink_k, key], dim=2)
        value = torch.cat([sink_v, value], dim=2)
        
        local_sinks = s_param  # Use s_param.to_local() if DTensor

    # Standard attention computation
    attn_logits = torch.matmul(query, key.transpose(2, 3)) * scaling  # [B, H, Q, K+S]
    
    if attention_mask is not None:
        # causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        causal_mask = attention_mask[:, :, :, :K]
        # If we have sinks, we need to pad the mask with zeros for sink positions
        if s_param is not None:
            # Pad causal mask with zeros for sink tokens (sinks can attend to everything)
            sink_mask = torch.zeros(
                causal_mask.shape[:-1] + (num_sink_tokens,),
                device=causal_mask.device,
                dtype=causal_mask.dtype
            )
            causal_mask = torch.cat([sink_mask, causal_mask], dim=-1)
        attn_logits = attn_logits + causal_mask

    # Apply score_mod_fn logic exactly like flex attention
    if s_param is not None:
        if local_sinks.ndim == 1:
            # For 1D sinks, each head gets its own sink value (like score_mod_fn)
            for h in range(H):
                sink_logit = local_sinks[h].to(attn_logits.dtype)
                # Apply to all query positions for sink kv positions (first num_sink_tokens)
                attn_logits[:, h, :, :num_sink_tokens] = sink_logit
        elif local_sinks.ndim == 2:
            # For 2D sinks, each head gets different values per sink position
            for h in range(H):
                for s in range(num_sink_tokens):
                    sink_logit = local_sinks[h, s].to(attn_logits.dtype)
                    attn_logits[:, h, :, s] = sink_logit

    # Apply precision casting like flex attention AFTER sink logits
    cast_to_bf16 = query.device.type == 'cuda'
    og_dtype = query.dtype
    if cast_to_bf16:
        query, key, value = query.bfloat16(), key.bfloat16(), value.bfloat16()
        attn_logits = attn_logits.bfloat16()

    # Use autocast context like flex attention
    with torch.autocast(enabled=False, device_type=query.device.type):
        attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(attn_logits.dtype)
        probs = F.dropout(attn_probs, p=dropout, training=module.training)
        attn_out = torch.matmul(probs, value)  # [B, H, Q, D]
    
    # Cast back to original dtype
    if cast_to_bf16:
        attn_out = attn_out.to(og_dtype)
    
    # Return only attn_out to match OLMo-core format (no weights)
    return attn_out.transpose(1, 2).contiguous()



# Register flex attention if available
try:
    ALL_ATTENTION_FUNCTIONS["flex_attention"] = flex_attention_forward
except Exception:
    pass  # flex attention not available


class Olmo3Config(Olmo2Config):
    r"""
    This is the configuration class to store the configuration of a [`Olmo3Model`]. It is used to instantiate an OLMo3
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the [allenai/OLMo-3-0725-1B](https://huggingface.co/allenai/OLMo-3-0725-1B).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50304):
            Vocabulary size of the Olmo3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Olmo3Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 50279):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        sliding_window (`int`, *optional*, defaults to 4097):
            Size of the sliding window for sliding window attention.
        use_sinks (`bool`, *optional*, defaults to `False`):
            Whether to use attention sinks for improved long context performance.
        layer_types (`list`, *optional*):
            Attention pattern for each layer. Defaults to full attention in each layer.

    ```python
    >>> from transformers import Olmo3Model, Olmo3Config

    >>> # Initializing a Olmo3 7B style configuration
    >>> configuration = Olmo3Config()

    >>> # Initializing a model from the Olmo3 7B style configuration
    >>> model = Olmo3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "olmo3"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.k_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.v_proj": "colwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.o_proj": "rowwise_rep",  # we need to replicate here due to the added norm on q and k
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
        use_sinks=False,
        sliding_window=4097,
        layer_types=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rms_norm_eps=rms_norm_eps,
            use_sinks=use_sinks,
            **kwargs,
        )

        self.sliding_window = sliding_window
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if i % 4 != 0 else "full_attention" for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)


class Olmo3RMSNorm(Olmo2RMSNorm):
    pass


# Olmo3 attention is identical to OLMo 2 attention except:
# - Norm is applied headwise to attention queries and keys.
# - Sliding window attention is used for 3 out of 4 layers.
class Olmo3Attention(Olmo2Attention):
    def __init__(self, config: Olmo3Config, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        assert config.layer_types is not None
        self.attention_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window if self.attention_type == "sliding_attention" else None
        self.q_norm = Olmo3RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Olmo3RMSNorm(self.head_dim, config.rms_norm_eps)
        if config.use_sinks:
            self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))
        else:
            self.sinks = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = olmo3_eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Olmo3DecoderLayer(Olmo2DecoderLayer):
    def __init__(self, config: Olmo3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Olmo3Attention(config=config, layer_idx=layer_idx)


class Olmo3RotaryEmbedding(Olmo2RotaryEmbedding):
    pass


class Olmo3PreTrainedModel(Olmo2PreTrainedModel):
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, Olmo3Attention) and module.sinks is not None:
            module.sinks.data.normal_(mean=0.0, std=std)


# The OLMo 3 model is identical to the OLMo 2 model, except:
# - Sliding window attention is used for 3 out of 4 layers.
class Olmo3Model(Olmo2Model):
    def __init__(self, config: Olmo3Config):
        super().__init__(config)
        self.norm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [Olmo3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.self_attn.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


# The heads now only need to redefine the model inside to the correct `RobertaModel`
class Olmo3ForCausalLM(Olmo2ForCausalLM):
    def __init__(self, config: Olmo3Config):
        super().__init__(config)
        self.model = Olmo3Model(config)


__all__ = [
    "Olmo3Config",
    "Olmo3ForCausalLM",
    "Olmo3Model",
    "Olmo3PreTrainedModel",  # noqa: F822
]