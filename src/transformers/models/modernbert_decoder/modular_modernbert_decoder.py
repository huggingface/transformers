# Copyright 2024 Answer.AI, LightOn, and contributors, and the HuggingFace Inc. team. All rights reserved.
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

from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...cache_utils import HybridCache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...models.modernbert.modeling_modernbert import (
    ModernBertEmbeddings,
    ModernBertMLP,
    ModernBertPredictionHead,
    ModernBertPreTrainedModel,
    ModernBertRotaryEmbedding,
    apply_rotary_pos_emb,
)
from ...utils import auto_docstring, logging


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
        deterministic_flash_attn (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic flash attention. If `False`, inference will be faster but not deterministic.
        reference_compile (`bool`, *optional*):
            Whether to compile the layers of the model which were compiled during pretraining. If `None`, then parts of
            the model will be compiled if 1) `triton` is installed, 2) the model is not on MPS, 3) the model is not
            shared between devices, and 4) the model is not resized after initialization. If `True`, then the model may
            be faster in some scenarios.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        local_attention (`int`, *optional*, defaults to 128):
            The sliding window size for local attention. Only used for layers that use local attention.
        global_attn_every_n_layers (`int`, *optional*, defaults to 3):
            Every `global_attn_every_n_layers` layers will use global attention instead of local attention.
        local_rope_theta (`float`, *optional*):
            The base period of the local RoPE embeddings. If not specified, uses the same value as `global_rope_theta`.
        num_labels (`int`, *optional*, defaults to 2):
            Number of labels for sequence classification.

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
        global_rope_theta=160000.0,
        attention_bias=False,
        attention_dropout=0.0,
        embedding_dropout=0.0,
        mlp_bias=False,
        mlp_dropout=0.0,
        decoder_bias=True,
        deterministic_flash_attn=False,
        reference_compile=None,
        use_cache=True,
        local_attention=128,
        global_attn_every_n_layers=3,
        local_rope_theta=None,
        num_labels=2,
        **kwargs,
    ):
        # Set default attention implementation
        if "_attn_implementation" not in kwargs:
            kwargs["_attn_implementation"] = "flash_attention_2"

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
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
        self.deterministic_flash_attn = deterministic_flash_attn
        self.reference_compile = reference_compile
        self.use_cache = use_cache
        self.local_attention = local_attention
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.local_rope_theta = local_rope_theta
        self.num_labels = num_labels

    def to_dict(self):
        output = super().to_dict()
        output.pop("reference_compile", None)
        return output


def eager_attention_forward(
    module: "ModernBertDecoderAttention",
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    is_causal: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Eager attention implementation for ModernBERT decoder with causal masking.

    Args:
        module: The attention module
        query: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
        key: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
        value: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
        attention_mask: Attention mask tensor
        dropout: Dropout probability
        scaling: Attention scaling factor (will use module.scaling if None)
        sliding_window: Sliding window size (for local attention)
        is_causal: Whether to use causal attention

    Returns:
        Tuple of (attention_output, attention_weights)
    """
    batch_size, num_heads, seq_len, head_dim = query.shape

    # Use module's scaling if not provided
    if scaling is None:
        scaling = module.scaling

    # Compute attention scores
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    # Apply attention mask if provided
    if attention_mask is not None:
        # For causal attention, mask should already be properly shaped
        if attention_mask.dim() == 4:
            # Already 4D mask: [batch_size, num_heads, seq_len, seq_len]
            attn_weights = attn_weights + attention_mask
        elif attention_mask.dim() == 3:
            # 3D mask: [batch_size, seq_len, seq_len] - add head dimension
            attn_weights = attn_weights + attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 2:
            # 2D mask: [batch_size, seq_len] - expand to causal mask
            # This case shouldn't happen with proper causal masking, but handle it
            causal_mask = torch.full(
                (seq_len, seq_len),
                torch.finfo(attn_weights.dtype).min,
                device=attn_weights.device,
                dtype=attn_weights.dtype,
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            attn_weights = attn_weights + causal_mask

    # Apply sliding window mask if specified
    if sliding_window is not None and sliding_window > 0:
        # Create sliding window mask
        mask = torch.full((seq_len, seq_len), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        for i in range(seq_len):
            start = max(0, i - sliding_window + 1)
            end = i + 1  # Causal: can only attend to current and previous tokens
            mask[i, start:end] = 0
        attn_weights = attn_weights + mask.unsqueeze(0).unsqueeze(0)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class ModernBertDecoderAttention(nn.Module):
    """Performs causal multi-headed self attention for ModernBERT decoder.

    This module implements causal (unidirectional) attention, suitable for language modeling tasks.
    It supports both local attention (sliding window) and global attention patterns.
    """

    def __init__(self, config: ModernBertDecoderConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )

        self.attention_dropout = config.attention_dropout
        self.deterministic_flash_attn = config.deterministic_flash_attn
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True  # used sometimes for FA2

        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.attention_bias)

        # Determine if this layer uses local or global attention
        if layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (config.local_attention // 2, config.local_attention // 2)
        else:
            self.local_attention = (-1, -1)

        rope_theta = config.global_rope_theta
        if self.local_attention != (-1, -1):
            if config.local_rope_theta is not None:
                rope_theta = config.local_rope_theta

        self.rotary_emb = ModernBertRotaryEmbedding(config=config, dim=self.head_dim, base=rope_theta)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.Wqkv(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        # Create position_ids if None
        if position_ids is None:
            device = hidden_states.device
            if past_key_value is not None:
                # For incremental decoding, start from cached length
                cache_length = past_key_value[0].shape[-2]
                position_ids = torch.arange(cache_length, cache_length + seq_len, dtype=torch.long, device=device)
            else:
                # For initial forward pass, start from 0
                position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # Apply rotary embeddings
        if past_key_value is not None:
            # Extract query, key, value first for caching case
            query, key, value = qkv.transpose(3, 1).unbind(dim=2)
            # query, key, value: [batch_size, num_heads, seq_len, head_dim]

            # create QKV format for RoPE computation
            qkv_for_rope = torch.stack([query, key, value], dim=2).transpose(1, 2)
            cos, sin = self.rotary_emb(qkv_for_rope, position_ids=position_ids)

            # Apply RoPE only to the new query and key
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

            # Concatenate with past keys and values
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        else:
            # For initial forward pass, apply RoPE to full sequence
            # Call rotary embedding on qkv tensor before extracting query, key, value
            cos, sin = self.rotary_emb(qkv, position_ids=position_ids)
            query, key, value = qkv.transpose(3, 1).unbind(dim=2)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Set up cache BEFORE calling attention (attention might modify tensors)
        past_key_value_out = (key, value) if use_cache else None

        # Use the appropriate attention interface
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        is_causal = True
        if self.config._attn_implementation == "sdpa":  # it can always be True for eager or FA2, but not SDPA
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal = True if attention_mask is None and seq_len > 1 else False

        # FlashAttention only supports fp16 and bf16 data types
        if self.config._attn_implementation == "flash_attention_2":
            original_dtype = query.dtype
            if original_dtype not in (torch.float16, torch.bfloat16):
                # Cast to bfloat16 if available, otherwise fp16
                target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                query = query.to(target_dtype)
                key = key.to(target_dtype)
                value = value.to(target_dtype)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(target_dtype)
        else:
            original_dtype = None

        attn_outputs = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            sliding_window=self.local_attention[0] if self.local_attention[0] != -1 else None,
            is_causal=is_causal,
            **kwargs,
        )

        attn_output = attn_outputs[0]
        attn_weights = attn_outputs[1] if output_attentions and len(attn_outputs) > 1 else None

        # Cast back to original dtype if we changed it for FlashAttention
        if self.config._attn_implementation == "flash_attention_2" and original_dtype is not None:
            if original_dtype not in (torch.float16, torch.bfloat16):
                attn_output = attn_output.to(original_dtype)
                if attn_weights is not None:
                    attn_weights = attn_weights.to(original_dtype)

        # Ensure attn_output has the correct shape: (batch_size, seq_len, hidden_size)
        if attn_output.dim() == 4:  # (batch_size, seq_len, num_heads, head_dim)
            attn_output = attn_output.view(batch_size, seq_len, self.all_head_size)

        # Apply output projection
        hidden_states = self.out_drop(self.Wo(attn_output))

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (past_key_value_out,)

        return outputs


class ModernBertDecoderLayer(nn.Module):
    def __init__(self, config: ModernBertDecoderConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        # Determine if this layer uses sliding window attention
        self.is_sliding_window = layer_id % config.global_attn_every_n_layers != 0

        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = ModernBertDecoderAttention(config=config, layer_id=layer_id)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ModernBertMLP(config)

    @torch.compile(dynamic=True)
    def compiled_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.mlp_norm(hidden_states))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        **kwargs,
    ) -> torch.Tensor:
        attn_outputs = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = hidden_states + attn_outputs[0]
        mlp_output = (
            self.compiled_mlp(hidden_states)
            if self.config.reference_compile
            else self.mlp(self.mlp_norm(hidden_states))
        )
        hidden_states = hidden_states + mlp_output

        outputs = (hidden_states,)
        if len(attn_outputs) > 1:
            outputs += attn_outputs[1:]

        return outputs


@auto_docstring
class ModernBertDecoderPreTrainedModel(ModernBertPreTrainedModel):
    config_class = ModernBertDecoderConfig
    base_model_prefix = "model"


@auto_docstring
class ModernBertDecoderModel(ModernBertDecoderPreTrainedModel):
    def __init__(self, config: ModernBertDecoderConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [ModernBertDecoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor, ...], BaseModelOutputWithPast]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.num_hidden_layers`, with each tuple having 2 tensors
            of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            batch_size, seq_length = input_ids.shape[:2]
        else:
            batch_size, seq_length = inputs_embeds.shape[:2]

        # Create cache_position - this is the sequence positions for the current tokens
        if past_key_values is None:
            past_seen_tokens = 0
        else:
            past_seen_tokens = past_key_values[0][0].shape[-2] if past_key_values[0] is not None else 0

        # Create cache_position using position_ids if available (to respect padding)
        if position_ids is not None:
            # Use the actual positions from position_ids for cache_position
            cache_position = (
                position_ids[0] if position_ids.shape[0] > 0 else torch.arange(seq_length, device=position_ids.device)
            )
        else:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + seq_length,
                device=input_ids.device if input_ids is not None else inputs_embeds.device,
            )

        # Create position_ids that respect padding tokens if not provided
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if past_key_values is None:
                # For initial forward pass, create position_ids that respect padding
                if attention_mask is not None:
                    # Create cumulative sum of attention_mask to get proper positions
                    # This ensures padding tokens don't increment position
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 0)
                else:
                    # Fallback: sequential positions
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            else:
                # For cached generation, continue from where we left off
                past_seen_tokens = past_key_values[0][0].shape[-2] if past_key_values[0] is not None else 0

                if attention_mask is not None and attention_mask.shape[-1] > seq_length:
                    # Full attention mask provided - use it to calculate proper positions
                    # Count real tokens in the past to get proper starting position
                    past_attention = attention_mask[..., :-seq_length]
                    past_real_tokens = past_attention.sum(dim=-1, keepdim=True)
                    current_positions = past_real_tokens + torch.arange(seq_length, device=device)
                    # Only increment for non-padding tokens in current sequence
                    current_mask = attention_mask[..., -seq_length:]
                    position_ids = current_positions.masked_fill(current_mask == 0, 0)
                else:
                    # Fallback: continue sequentially
                    position_ids = torch.arange(
                        past_seen_tokens, past_seen_tokens + seq_length, dtype=torch.long, device=device
                    ).unsqueeze(0)

        # Calculate embeddings first, as we need them for mask creation
        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        # Convert tuple-based cache to HybridCache for masking functions
        cache_for_masking = None
        if past_key_values is not None and past_key_values[0] is not None:
            # Set up config attributes needed for HybridCache
            if not hasattr(self.config, "sliding_window"):
                self.config.sliding_window = self.config.local_attention

            if not hasattr(self.config, "layer_types"):
                # Create layer_types based on the alternating pattern
                self.config.layer_types = []
                for layer_id in range(self.config.num_hidden_layers):
                    if layer_id % self.config.global_attn_every_n_layers != 0:
                        self.config.layer_types.append("sliding_window")
                    else:
                        self.config.layer_types.append("full_attention")

            # Create HybridCache with appropriate parameters
            cache_for_masking = HybridCache(
                config=self.config,
                max_batch_size=batch_size,
                max_cache_len=past_key_values[0][0].shape[-2] + seq_length,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

            # Populate the cache with existing data to ensure proper masking
            for layer_idx, (key, value) in enumerate(past_key_values):
                if key is not None and value is not None:
                    # Need to populate cache properly for masking utilities to work
                    cache_position_past = torch.arange(key.shape[-2], device=key.device)
                    cache_for_masking.update(key, value, layer_idx, {"cache_position": cache_position_past})
        else:
            # Even without past_key_values, we may need HybridCache for mask creation
            # Set up config attributes for consistent masking
            if not hasattr(self.config, "sliding_window"):
                self.config.sliding_window = self.config.local_attention

            if not hasattr(self.config, "layer_types"):
                # Create layer_types based on the alternating pattern
                self.config.layer_types = []
                for layer_id in range(self.config.num_hidden_layers):
                    if layer_id % self.config.global_attn_every_n_layers != 0:
                        self.config.layer_types.append("sliding_window")
                    else:
                        self.config.layer_types.append("full_attention")

            # Create empty HybridCache for masking consistency
            cache_for_masking = HybridCache(
                config=self.config,
                max_batch_size=batch_size,
                max_cache_len=seq_length,  # Just the current sequence length
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        if self.config._attn_implementation != "flash_attention_2":  # FA2 generates it's own mask
            # It may already have been prepared by e.g. `generate`
            if isinstance(attention_mask, dict):
                # attention_mask is already a mask mapping, use it directly
                causal_mask_mapping = attention_mask
            else:
                # Prepare mask arguments
                mask_kwargs = {
                    "config": self.config,
                    "input_embeds": hidden_states,
                    "attention_mask": attention_mask,
                    "cache_position": cache_position,
                    "past_key_values": cache_for_masking,
                }

                # Create the masks - always need causal mask for global attention layers
                causal_mask_mapping = {
                    "global_attention": create_causal_mask(**mask_kwargs),
                }

                # Only create sliding window mask if we have layers that need it
                if self.config.global_attn_every_n_layers != 1:
                    # Temporarily add sliding_window attribute to config for mask creation
                    sliding_mask_kwargs = mask_kwargs.copy()
                    self.config.sliding_window = self.config.local_attention
                    causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**sliding_mask_kwargs)
                    del self.config.sliding_window  # Clean up
        else:
            causal_mask_mapping = None

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        self._maybe_set_compile()

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # Get the appropriate mask for this layer
            if causal_mask_mapping is not None:
                mask_type = "sliding_attention" if decoder_layer.is_sliding_window else "global_attention"
                current_mask = causal_mask_mapping[mask_type]
            else:
                current_mask = None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    current_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    **kwargs,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=current_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.final_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
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
        self.lm_head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings.tok_embeddings = value

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings):
        self.decoder = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @torch.compile(dynamic=True)
    def compiled_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.lm_head(hidden_states))

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, ModernBertDecoderForCausalLM

        >>> model = ModernBertDecoderForCausalLM.from_pretrained("your-username/your-model-name")
        >>> tokenizer = AutoTokenizer.from_pretrained("your-username/your-model-name")

        >>> prompt = "Hello, I'm a language model,"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hello, I'm a language model, and I'm here to help you with your questions."
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self._maybe_set_compile()

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = (
            self.compiled_head(hidden_states)
            if self.config.reference_compile
            else self.decoder(self.lm_head(hidden_states))
        )

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

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


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
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings.tok_embeddings = value

    def get_output_embeddings(self):
        return None

    def set_output_embeddings(self, new_embeddings):
        pass

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @auto_docstring(checkpoint="blab-jhu/test-32m-dec")
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

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

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

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
