# Copyright 2025 Johns Hopkins University, LightOn, and the HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Callable
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs
from ..modernbert.modeling_modernbert import (
    ModernBertEmbeddings,
    ModernBertMLP,
    ModernBertPredictionHead,
    ModernBertPreTrainedModel,
    ModernBertRotaryEmbedding,
    apply_rotary_pos_emb,
)


logger = logging.get_logger(__name__)


class ModernBertDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ModernBertDecoderModel`]. It is used to instantiate a ModernBert
    decoder model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ModernBERT-base decoder.
    e.g. [blab-jhu/test-32m-dec](https://huggingface.co/blab-jhu/test-32m-dec)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50368):
            Vocabulary size of the ModernBert decoder model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ModernBertDecoderModel`]
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 1152):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 22):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer decoder.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder. Will default to `"gelu"`
            if not specified.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_cutoff_factor (`float`, *optional*, defaults to 2.0):
            The cutoff factor for the truncated_normal_initializer for initializing all weight matrices.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        norm_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the normalization layers.
        pad_token_id (`int`, *optional*, defaults to 50283):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 50282):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 50281):
            Beginning of stream token id.
        cls_token_id (`int`, *optional*, defaults to 50281):
            Classification token id.
        sep_token_id (`int`, *optional*, defaults to 50282):
            Separation token id.
        global_rope_theta (`float`, *optional*, defaults to 160000.0):
            The base period of the global RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        embedding_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the embeddings.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the MLP layers.
        mlp_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the MLP layers.
        decoder_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the decoder layers.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the classifier.
        classifier_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the classifier.
        classifier_activation (`str`, *optional*, defaults to `"gelu"`):
            The activation function for the classifier.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        local_attention (`int`, *optional*, defaults to 128):
            The sliding window size for local attention. Only used for layers that use local attention. Note that for
            the decoder to match ModernBERT this is actually half of the sliding window size, so 128 => 64.
        global_attn_every_n_layers (`int`, *optional*, defaults to 3):
            Every `global_attn_every_n_layers` layers will use global attention instead of local attention.
        local_rope_theta (`float`, *optional*, defaults to 160000.0):
            The base period of the local RoPE embeddings. If not specified, defaults to 160000.0
        layer_types (`list`, *optional*):
            List of layer types, one for each layer. If not specified, will be automatically generated based on
            `global_attn_every_n_layers`. Should contain "full_attention" or "sliding_attention".

    Examples:

    ```python
    >>> from transformers import ModernBertDecoderModel, ModernBertDecoderConfig

    >>> # Initializing a ModernBert decoder style configuration
    >>> configuration = ModernBertDecoderConfig()

    >>> # Initializing a model from the modernbert-base decoder style configuration
    >>> model = ModernBertDecoderModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "modernbert-decoder"
    attribute_map = {"rope_theta": "global_rope_theta"}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50368,
        hidden_size=768,
        intermediate_size=1152,
        num_hidden_layers=22,
        num_attention_heads=12,
        hidden_activation="gelu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        initializer_cutoff_factor=2.0,
        norm_eps=1e-5,
        norm_bias=False,
        pad_token_id=50283,
        eos_token_id=50282,
        bos_token_id=50281,
        cls_token_id=50281,
        sep_token_id=50282,
        global_rope_theta=160000.0,
        attention_bias=False,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        mlp_bias=False,
        mlp_dropout=0.0,
        decoder_bias=True,
        classifier_dropout=0.0,
        classifier_bias=False,
        classifier_activation="gelu",
        use_cache=True,
        local_attention=128,
        global_attn_every_n_layers=3,
        local_rope_theta=160000.0,
        layer_types=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.initializer_cutoff_factor = initializer_cutoff_factor
        self.norm_eps = norm_eps
        self.norm_bias = norm_bias
        self.global_rope_theta = global_rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.embedding_dropout = embedding_dropout
        self.mlp_bias = mlp_bias
        self.mlp_dropout = mlp_dropout
        self.decoder_bias = decoder_bias
        self.classifier_dropout = classifier_dropout
        self.classifier_bias = classifier_bias
        self.classifier_activation = classifier_activation
        self.use_cache = use_cache
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.local_rope_theta = local_rope_theta
        # for consistency with ModernBert
        self.reference_compile = False

        # Set up layer_types for standardized layer type detection
        self.layer_types = layer_types
        if self.layer_types is None:
            # Create layer_types based on the alternating pattern
            self.layer_types = []
            for layer_id in range(num_hidden_layers):
                if layer_id % global_attn_every_n_layers != 0:
                    self.layer_types.append("sliding_attention")
                else:
                    self.layer_types.append("full_attention")

        # NOTE: sliding window numbers matches ModernBERT but is only half of it
        self.sliding_window = local_attention // 2 if local_attention else -1


class ModernBertDecoderEmbeddings(ModernBertEmbeddings):
    pass


class ModernBertDecoderMLP(ModernBertMLP):
    pass


class ModernBertDecoderRotaryEmbedding(ModernBertRotaryEmbedding):
    pass


def eager_attention_forward(
    module: "ModernBertDecoderAttention",
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """A simple eager attention implementation for ModernBERT decoder."""
    if scaling is None:
        scaling = module.head_dim**-0.5

    # Compute attention scores
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    # Use the pre-computed attention mask
    causal_mask = attention_mask[:, :, :, : key.shape[-2]]
    attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class ModernBertDecoderAttention(nn.Module):
    """Performs causal multi-headed self attention for ModernBERT decoder.

    It supports both local attention (sliding window) and global attention patterns.
    """

    def __init__(self, config: ModernBertDecoderConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )

        # NOTE: this is different than ModernBERT (separated QKV) so be sure to adapt to this
        self.q_proj = nn.Linear(self.config.hidden_size, self.all_head_size, bias=self.config.attention_bias)
        self.k_proj = nn.Linear(self.config.hidden_size, self.all_head_size, bias=self.config.attention_bias)
        self.v_proj = nn.Linear(self.config.hidden_size, self.all_head_size, bias=self.config.attention_bias)

        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(config.attention_dropout)

        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
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
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_drop(self.Wo(attn_output))
        return attn_output, attn_weights


class ModernBertDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: ModernBertDecoderConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.attn_norm = (
            nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
            if layer_idx != 0
            else nn.Identity()
        )
        self.attn = ModernBertDecoderAttention(config=config, layer_idx=layer_idx)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ModernBertDecoderMLP(config)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        # apply global RoPE to non-sliding layer only
        if self.attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        # Self Attention
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = attn_outputs[0]

        # Add residual connection
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        return hidden_states


class ModernBertDecoderPredictionHead(ModernBertPredictionHead):
    pass


@auto_docstring
class ModernBertDecoderPreTrainedModel(ModernBertPreTrainedModel):
    _skip_keys_device_placement = ["past_key_values"]
    _no_split_modules = ["ModernBertDecoderLayer"]
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": ModernBertDecoderLayer,
        "attentions": ModernBertDecoderAttention,
    }

    def _init_weights(self, module: nn.Module):
        cutoff_factor = self.config.initializer_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        def init_weight(module: nn.Module, std: float):
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )

            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        stds = {
            "in": self.config.initializer_range,
            "out": self.config.initializer_range / math.sqrt(2.0 * self.config.num_hidden_layers),
            "embedding": self.config.initializer_range,
            "final_out": self.config.hidden_size**-0.5,
        }

        if isinstance(module, ModernBertDecoderEmbeddings):
            init_weight(module.tok_embeddings, stds["embedding"])
        elif isinstance(module, ModernBertDecoderMLP):
            init_weight(module.Wi, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, ModernBertDecoderAttention):
            init_weight(module.q_proj, stds["in"])
            init_weight(module.k_proj, stds["in"])
            init_weight(module.v_proj, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, ModernBertDecoderPredictionHead):
            init_weight(module.dense, stds["out"])
        elif isinstance(module, ModernBertDecoderForSequenceClassification):
            init_weight(module.classifier, stds["final_out"])
        elif isinstance(module, ModernBertDecoderForCausalLM):
            init_weight(module.decoder, stds["out"])
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check):
        raise AttributeError("No need to inherit!")

    def _maybe_set_compile(self):
        raise AttributeError("No need to inherit!")

    def resize_token_embeddings(self, *args, **kwargs):
        raise AttributeError("No need to inherit!")


@auto_docstring
class ModernBertDecoderModel(ModernBertDecoderPreTrainedModel):
    def __init__(self, config: ModernBertDecoderConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = ModernBertDecoderEmbeddings(config)
        self.layers = nn.ModuleList(
            [ModernBertDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.gradient_checkpointing = False

        self.global_rotary_emb = ModernBertDecoderRotaryEmbedding(config=config)
        self.local_rotary_emb = ModernBertDecoderRotaryEmbedding(config=config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor, ...], BaseModelOutputWithPast]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            batch_size, seq_length = input_ids.shape[:2]
        else:
            batch_size, seq_length = inputs_embeds.shape[:2]

        # Handle past_key_values and cache setup
        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + seq_length,
                device=input_ids.device if input_ids is not None else inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        # Calculate embeddings
        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": hidden_states,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # create position embeddings to be shared across the decoder layers
        position_embeddings_global = self.global_rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.local_rotary_emb(hidden_states, position_ids)

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.final_norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring(
    custom_intro="""
    The ModernBert Decoder Model with a language modeling head on top for causal language modeling (CLM).
    """
)
class ModernBertDecoderForCausalLM(ModernBertDecoderPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["decoder.weight"]

    def __init__(self, config: ModernBertDecoderConfig):
        super().__init__(config)
        self.config = config
        self.model = ModernBertDecoderModel(config)
        self.lm_head = ModernBertDecoderPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings):
        self.decoder = new_embeddings

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            [`~modeling_outputs.CausalLMOutputWithPast`]
            comprising various elements depending on the configuration and inputs.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, ModernBertDecoderForCausalLM

        >>> model = ModernBertDecoderForCausalLM.from_pretrained("blab-jhu/test-32m-dec")
        >>> tokenizer = AutoTokenizer.from_pretrained("blab-jhu/test-32m-dec")

        >>> prompt = "The capital of France is"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=1)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The capital of France is Paris"
        ```
        """
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.decoder(self.lm_head(hidden_states))

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

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The ModernBert Decoder Model with a sequence classification head on top (linear layer).

    [`ModernBertDecoderForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1, GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """
)
class ModernBertDecoderForSequenceClassification(ModernBertDecoderPreTrainedModel):
    def __init__(self, config: ModernBertDecoderConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = ModernBertDecoderModel(config)

        self.head = ModernBertDecoderPredictionHead(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=config.classifier_bias)
        self.drop = torch.nn.Dropout(config.classifier_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring(checkpoint="blab-jhu/test-32m-dec")
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = transformer_outputs[0]
        hidden_states = self.drop(self.head(hidden_states))
        logits = self.classifier(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


__all__ = [
    "ModernBertDecoderConfig",
    "ModernBertDecoderModel",
    "ModernBertDecoderPreTrainedModel",
    "ModernBertDecoderForCausalLM",
    "ModernBertDecoderForSequenceClassification",
]
