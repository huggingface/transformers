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
from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...activations import ACT2FN
from ...cache_utils import Cache, HybridCache, StaticCache
from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import logging
from ..gemma.modeling_gemma import (
    GemmaAttention,
    GemmaForCausalLM,
    GemmaForSequenceClassification,
    GemmaForTokenClassification,
    GemmaMLP,
    GemmaModel,
    GemmaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)


_CHECKPOINT_FOR_DOC = "google/gemma2-7b"

logger = logging.get_logger(__name__)


class Gemma2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Gemma2Model`]. It is used to instantiate an Gemma2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Gemma2-7B.
    e.g. [google/gemma2-7b](https://huggingface.co/google/gemma2-7b)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Gemma2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Gemma2Model`]
        hidden_size (`int`, *optional*, defaults to 2304):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 9216):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 26):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 256):
            The attention head dimension.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the decoder. Will default to `"gelu_pytorch_tanh"`
            if not specified. `"gelu_pytorch_tanh"` uses an approximation of the `"gelu"` activation function.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        query_pre_attn_scalar (`float`, *optional*, defaults to 256): scaling factor used on the attention scores
        sliding_window (`int`, *optional*, defaults to 4096): in Gemma2, every other layer uses sliding window attention. This is the
            size of the sliding window.
        final_logit_softcapping (`float`, *optional*, defaults to 30.0): scaling factor when applying tanh softcapping on the logits.
        attn_logit_softcapping (`float`, *optional*, defaults to 50.0): scaling factor when applying tanh softcapping on the attention scores.
        cache_implementation (`str`, *optional*, defaults to `"hybrid"`): the cache type to be used with `generate`.

    ```python
    >>> from transformers import Gemma2Model, Gemma2Config
    >>> # Initializing a Gemma2 gemma2-7b style configuration
    >>> configuration = Gemma2Config()
    >>> # Initializing a model from the gemma2-7b style configuration
    >>> model = Gemma2Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gemma2"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
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
        vocab_size=256000,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=4096,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        cache_implementation="hybrid",
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
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.cache_implementation = cache_implementation


class Gemma2RMSNorm(GemmaRMSNorm):
    pass


class Gemma2MLP(GemmaMLP):
    def __init__(self, config):
        super().__init__()
        self.act_fn = ACT2FN[config.hidden_activation]


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Gemma2Attention(GemmaAttention):
    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.sliding_window = config.sliding_window if not bool(layer_idx % 2) else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # Here we need to slice as we use a static cache by default, but FA2 does not support it
            if attention_mask is not None and self.config._attn_implementation == "flash_attention_2":
                seq_len = attention_mask.shape[-1]
                key_states, value_states = key_states[:, :, :seq_len, :], value_states[:, :, :seq_len, :]

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            softcap=self.attn_logit_softcapping,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma2DecoderLayer(nn.Module):
    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.is_sliding = not bool(layer_idx % 2)
        self.self_attn = Gemma2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma2MLP(config)
        self.input_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.pre_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: int = 0,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if self.is_sliding and attention_mask is not None:  # efficient SDPA and no padding
            # In prefill, we may be larger than sliding window
            effective_seq_len = max(cache_position.shape[0], self.sliding_window)
            # For FA2, the mask is 2D and is of shape [bs, processed_tokens] (not [bs, max_cache_len]),
            # thus we must slice from the right (at most `effective_seq_len` elements)
            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask[:, -effective_seq_len:]
            # Otherwise, the mask is 4D of shape [bs, 1, query_len, max_cache_len] thus we must slice
            # from the left, with an offset if we are beyond the sliding window
            else:
                min_dtype = torch.finfo(attention_mask.dtype).min
                sliding_window_mask = torch.tril(
                    torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-self.sliding_window
                )
                attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
                # In case we are beyond the sliding window, we need to correctly offset the mask slicing
                # `last_cache_position` is equivalent to `cache_position[-1]` but without breaking dynamo
                offset = last_cache_position - effective_seq_len
                # Should only be used when beyond the sliding window (i.e. offset > 0)
                offset = max(0, offset)
                attention_mask = attention_mask[:, :, :, offset : offset + effective_seq_len]

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Gemma2Model(GemmaModel):
    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Gemma2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: Optional[int] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            batch_size, seq_len, _ = inputs_embeds.shape
            # NOTE: ideally, `HybridCache` should be initialized outside the model with `layer_device_map`
            past_key_values = HybridCache(
                self.config,
                max_batch_size=batch_size,
                max_cache_len=seq_len,
                dtype=inputs_embeds.dtype,
                device=self.device,
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # This is needed to correctly slice the mask without data-dependent slicing later on if using dynamo tracing
        # (retrieving the same value from `cache_position` later on would crash dynamo)
        if last_cache_position is None:
            last_cache_position = 0
            if attention_mask is not None:
                # In case a 4d mask is passed directly without using `generate`, we have to rely on cache_position
                # It will break dynamo tracing but there are no way around it (and it should never happen in practice)
                last_cache_position = (
                    attention_mask.shape[-1] if attention_mask.dim() == 2 else cache_position[-1].item()
                )
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # normalized
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    position_embeddings,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    last_cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    last_cache_position=last_cache_position,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    @torch.no_grad()
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: HybridCache,
        output_attentions: bool,
    ):
        # Flash Attention currently doesn't support static cache but Gemma2 work only with static cache.
        # So we will pass in attention mask as is in any case, not only when ther's padding. Then we'll use its shape
        # to cut out keys/values trailing 0 used in static cache. This workaround should be compile compatible
        # as it doesn't cause dynamic control issues.
        if self.config._attn_implementation == "flash_attention_2":
            return attention_mask

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if isinstance(past_key_values, (HybridCache, StaticCache)):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = attention_mask.shape[-1] if attention_mask is not None else input_tensor.shape[1]

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )
        return causal_mask


class Gemma2ForCausalLM(GemmaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Gemma2Model(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Gemma2ForCausalLM

        >>> model = Gemma2ForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""

        if self.training and self.config._attn_implementation != "eager":
            logger.warning_once(
                "It is strongly recommended to train Gemma2 models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **loss_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten: has a special cache type, `HybridCache`

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # This is needed to correctly slice the mask without data-dependent slicing later on if using dynamo tracing
        # (retrieving the same value from `cache_position` later on would crash dynamo)
        model_inputs["last_cache_position"] = attention_mask.shape[-1] if attention_mask is not None else 0
        if logits_to_keep is None:
            _ = model_inputs.pop("logits_to_keep", None)

        if (
            isinstance(past_key_values, HybridCache)
            and attention_mask.ndim == 2
            and not self.config._attn_implementation == "flash_attention_2"
        ):
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
            )
            model_inputs["attention_mask"] = attention_mask

        return model_inputs


class Gemma2ForSequenceClassification(GemmaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.model = Gemma2Model(config)
        self.post_init()


class Gemma2ForTokenClassification(GemmaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.model = Gemma2Model(config)
        self.post_init()


__all__ = [
    "Gemma2Config",
    "Gemma2ForCausalLM",
    "Gemma2Model",
    "Gemma2PreTrainedModel",  # noqa: F822
    "Gemma2ForSequenceClassification",
    "Gemma2ForTokenClassification",
]
