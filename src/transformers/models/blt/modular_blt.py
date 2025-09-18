# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
"""Blt modular model, inheriting from Mllama where appropriate."""

from typing import Callable, Optional

import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, is_torch_flex_attn_available, logging
from ...utils.generic import OutputRecorder, check_model_inputs
from .configuration_blt import (
    BltConfig,
    BltGlobalTransformerConfig,
    BltLocalDecoderConfig,
    BltLocalEncoderConfig,
    BltPatcherConfig,
)


if is_torch_flex_attn_available():
    pass


from ..mllama.modeling_mllama import (
    MllamaForCausalLM,
    MllamaPreTrainedModel,
    MllamaRotaryEmbedding,
    MllamaSelfAttentionDecoderLayer,
    MllamaTextCrossAttention,
    MllamaTextMLP,
    MllamaTextRMSNorm,
    MllamaTextSelfAttention,
    eager_attention_forward,
)


logger = logging.get_logger(__name__)


def rolling_polynomial_hash(token_tensor, hash_func_nb: int = 0):
    primes = [
        1000000007,
        5915587277,
        1500450271,
        3267000013,
        5754853343,
        4093082899,
        9576890767,
        3628273133,
        2860486313,
        5463458053,
        3367900313,
    ]
    prime = torch.tensor(primes[hash_func_nb], dtype=torch.int64, device=token_tensor.device)
    powers = torch.arange(token_tensor.shape[-1], device=token_tensor.device)
    prime_powers = prime**powers
    return torch.sum(token_tensor * prime_powers, dim=-1)


def byte_group_hash_function(
    token_ids: torch.Tensor, group_size: int = 2, hash_func_nb: int = 0, max_hash: int = 30000
):
    """Hash token groups and map to range [0, max_hash]."""
    with torch.no_grad():
        batch_size, seq_len = token_ids.shape
        # Add padding for sliding window
        padding = torch.zeros(batch_size, group_size - 1, dtype=torch.int64, device=token_ids.device)
        padded_tokens = torch.cat([padding, token_ids], dim=1)

        # Create sliding windows and compute hashes
        windows = padded_tokens.unfold(1, group_size, 1)
        hashes = rolling_polynomial_hash(windows, hash_func_nb)
        hash_values = hashes % max_hash

    return hash_values


def compute_hash_embeddings(
    local_encoder_tokens: torch.Tensor,
    local_encoder,
    encoder_hash_tok_embedding: nn.ModuleList,
    encoder_hash_byte_group_nb_functions: int,
    encoder_hash_byte_group_size: list,
    encoder_hash_byte_group_vocab: int,
) -> torch.Tensor:
    """Compute token embeddings enhanced with hash-based embeddings."""
    embeddings = local_encoder.embed_tokens(local_encoder_tokens)
    embedding_idx = 0
    for func_nb in range(encoder_hash_byte_group_nb_functions):
        for group_size in encoder_hash_byte_group_size:
            hash_ids = byte_group_hash_function(
                local_encoder_tokens, group_size, func_nb, encoder_hash_byte_group_vocab
            )
            embeddings += encoder_hash_tok_embedding[embedding_idx](hash_ids)
            embedding_idx += 1

    return embeddings


def _prepare_patch_cross_attention_mask(
    patch_ids: torch.Tensor,
    num_patches: int,
    sequence_length: int,
    patches_as_queries: bool = False,
    cross_attn_k: int = 1,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare cross-attention mask for patch-based attention, following mllama's robust approach.

    This function creates masks that control which patches can attend to which other patches,
    with support for query/key role swapping and cross-attention multipliers.

    Args:
        patch_ids (torch.Tensor): Tensor of shape [batch_size, seq_len] containing patch ids.
        num_patches (int): Total number of patches.
        sequence_length (int): Length of the sequence.
        patches_as_queries (bool): If True, patches are used as queries, otherwise as keys.
        cross_attn_k (int): Cross-attention multiplier for repeating patches.
        dtype (torch.dtype): Data type for the output mask.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - cross_attention_mask: 4D tensor [batch_size, 1, q_len, kv_len]
            - full_text_row_masked_out_mask: 4D tensor indicating fully masked rows
    """
    batch_size, seq_len = patch_ids.shape
    device = patch_ids.device

    # Determine query and key lengths based on configuration
    if patches_as_queries:
        q_len = num_patches * cross_attn_k
        kv_len = sequence_length
        # Create patch-to-sequence mapping
        q_patch_ids = (
            torch.arange(num_patches, device=device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, num_patches, seq_len)
        )
        kv_patch_ids = patch_ids.unsqueeze(1).expand(batch_size, num_patches, seq_len)
    else:
        q_len = sequence_length
        kv_len = num_patches * cross_attn_k
        # Create sequence-to-patch mapping
        q_patch_ids = patch_ids.unsqueeze(-1).expand(batch_size, seq_len, num_patches)
        kv_patch_ids = (
            torch.arange(num_patches, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, num_patches)
        )

    # Create base attention mask - boolean mask where True means "should attend"
    # Exact patch matching
    cross_attention_mask = q_patch_ids == kv_patch_ids

    # Handle cross_attn_k multiplier by repeating along appropriate dimension
    repeat_dim = 1 if patches_as_queries else -1
    cross_attention_mask = cross_attention_mask.repeat_interleave(cross_attn_k, dim=repeat_dim)

    # Validate dimensions
    expected_shape = (batch_size, q_len, kv_len)
    if cross_attention_mask.shape != expected_shape:
        raise ValueError(
            f"Cross attention mask shape {cross_attention_mask.shape} doesn't match expected {expected_shape}"
        )

    # Reshape so it can be used by attn module - add head dimension
    cross_attention_mask = cross_attention_mask.unsqueeze(1)  # [batch_size, 1, q_len, kv_len]

    # Invert the mask (following mllama pattern exactly)
    # True -> 0.0 (attend), False -> 1.0 (will become -inf)
    inverted_cross_attn_mask = 1.0 - cross_attention_mask.to(dtype)
    cross_attention_mask = inverted_cross_attn_mask.masked_fill(
        inverted_cross_attn_mask.to(torch.bool), torch.finfo(dtype).min
    )

    # Apply full-row bias (following mllama pattern exactly)
    # Return 4D tensor of shape [B, H, S1, 1] where value is 0 if a full row in cross attn mask's
    # last dimension contains negative infinity values, otherwise it's 1
    negative_inf_value = torch.finfo(dtype).min
    full_text_row_masked_out_mask = (
        (cross_attention_mask != negative_inf_value).any(dim=-1).type_as(cross_attention_mask)[..., None]
    )
    cross_attention_mask *= full_text_row_masked_out_mask

    return cross_attention_mask


def process_patch_lengths(patch_lengths: torch.Tensor, max_patch_length: Optional[int]) -> torch.Tensor:
    """
    Splits patch lengths into smaller segments if they exceed `max_patch_length`.
    Pads the result to uniform length across the batch.

    Args:
        patch_lengths (torch.Tensor): [batch_size, num_patches] tensor of patch lengths.
        max_patch_length (int, optional): Maximum allowed length per patch.

    Returns:
        torch.Tensor: [batch_size, max_len] tensor of split and padded patch lengths.
    """
    if max_patch_length is None:
        return patch_lengths

    batch_size = patch_lengths.size(0)
    processed = []

    for seq in patch_lengths:
        splits = []
        for length in seq[seq > 0]:
            length = length.item()
            full_chunks, remainder = divmod(length, max_patch_length)
            splits.extend([max_patch_length] * full_chunks)
            if remainder:
                splits.append(remainder)
        processed.append(splits)

    # Find max length to pad to
    max_len = max(len(splits) for splits in processed)
    padded = torch.zeros((batch_size, max_len), dtype=patch_lengths.dtype, device=patch_lengths.device)

    for i, splits in enumerate(processed):
        if splits:
            padded[i, : len(splits)] = torch.tensor(splits, dtype=patch_lengths.dtype, device=patch_lengths.device)

    # Trim zero columns
    if (padded != 0).any(dim=0).sum() < padded.shape[1]:
        last_nonzero = (padded != 0).any(dim=0).nonzero().max().item() + 1
        padded = padded[:, :last_nonzero]

    return padded


class BltMLP(MllamaTextMLP):
    pass


class BltRMSNorm(MllamaTextRMSNorm):
    pass


class BltRotaryEmbedding(MllamaRotaryEmbedding):
    def __init__(self, config: BltConfig, device=None):
        super().__init__(config=config, device=device)
        # BC: "rope_type" was originally "type"
        self.rope_type = (
            config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            if config.rope_scaling is not None
            else "default"
        )

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # Copied from Cohere2RotaryEmbedding.forward
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.repeat_interleave(freqs, 2, dim=-1)  # Use Cohere2 pattern for compatibility
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class BltTransformerLayer(MllamaSelfAttentionDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.self_attn = BltSelfAttention(config=config, layer_idx=layer_idx)
        self.mlp = BltMLP(config)
        self.input_layernorm = BltRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BltRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class BltSelfAttention(MllamaTextSelfAttention):
    def __init__(self, config: BltConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        use_cache: bool = False,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )


class BltCrossAttention(MllamaTextCrossAttention):
    """Cross-attention module for Blt, following transformers style"""

    def __init__(self, config: BltConfig, layer_idx: int, hidden_size: Optional[int] = None):
        super().__init__()
        self.is_causal = False
        self.q_norm = BltRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.k_norm = BltRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        full_text_row_masked_out_mask: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_norm(hidden_states)
        query_states = self.q_proj(query_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if cross_attention_states is not None:
            cross_attention_states = self.k_norm(cross_attention_states)
            key_states = self.k_proj(cross_attention_states)
            value_states = self.v_proj(cross_attention_states)
            key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            if past_key_value is not None:
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
        elif cache_position[0] != 0:
            key_states, value_states = (
                past_key_value.layers[self.layer_idx].keys,
                past_key_value.layers[self.layer_idx].values,
            )
        else:
            raise ValueError(
                "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
            )
        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        attn_output = attn_output + hidden_states
        return attn_output, attn_weights


class BltLocalEncoder(nn.Module):
    def __init__(self, config: BltLocalEncoderConfig):
        super().__init__()
        self.gradient_checkpointing = False
        self.config = config
        self.layers = nn.ModuleList(
            [BltTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = BltRotaryEmbedding(config=config)
        self.patch_embedding_projection = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size * config.cross_attn_k,
            bias=False,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.cross_attn_layers = nn.ModuleList()
        layers_to_add = config.num_hidden_layers if config.cross_attn_all_layers else 1
        for layer_idx in range(layers_to_add):
            self.cross_attn_layers.append(
                BltCrossAttention(config=config, layer_idx=layer_idx, hidden_size=config.hidden_size)
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        patch_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        num_patches: Optional[int] = None,
        patch_ids: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size = inputs_embeds.shape[0]
        hidden_states = F.dropout(inputs_embeds, p=self.config.dropout, training=self.training)

        if position_ids is None:
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
            )

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)

        for idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
            if idx == len(self.layers) - 1 or self.config.cross_attn_all_layers:
                patch_embeds = self.patch_reduce(hidden_states, num_patches, "amax", patch_ids)
                patch_embeds = self.patch_embedding_projection(patch_embeds)
                patch_embeds = patch_embeds.reshape(
                    batch_size, patch_embeds.shape[1] * self.config.cross_attn_k, self.config.hidden_size
                )
                layer_idx = idx if self.config.cross_attn_all_layers else 0
                # Remove cross_attention_states from kwargs if present to avoid multiple values error
                kwargs.pop("cross_attention_states", None)
                cross_attention_output, _ = self.cross_attn_layers[layer_idx](
                    hidden_states=patch_embeds,
                    cross_attention_states=hidden_states,
                    attention_mask=cross_attention_mask,
                    full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                    **kwargs,
                )
                patch_embeds = patch_embeds + cross_attention_output
        encoder_cross_states = patch_embeds
        return hidden_states, encoder_cross_states

    def patch_reduce(self, hidden_states, max_num_patches, reduction, patch_ids):
        """
        Reduce variable length patches to single embedding per patch
        Note: this works with variable number of patches for different sequences in the batch
        It handles variable length patches by assuming that patch_lengths will be 0 for any
        extra patches on the *right*. Since there can be a variable number of patches
        this function also return the number of patches for each sequence in the batch.
        Any embeddings on the right that are not allocated to a patch
        (i.e. if the sum(patch_lengths[i]) < seq_len for any i)
        will be sent to a dummy patch, which is trimmed before returning.
        """
        batch_size = hidden_states.shape[0]
        embedding_dim = hidden_states.shape[-1]

        patch_ids = patch_ids.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])

        reduced_embeddings = torch.zeros(
            (batch_size, max_num_patches, embedding_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        reduced_embeddings = reduced_embeddings.scatter_reduce(
            src=hidden_states,
            dim=1,
            index=patch_ids,
            reduce=reduction,
            include_self=False,
        )
        reduced_embeddings = reduced_embeddings[:, :max_num_patches, :]

        return reduced_embeddings


class BltLocalDecoder(nn.Module):
    def __init__(self, config: BltLocalDecoderConfig):
        super().__init__()
        self.gradient_checkpointing = False
        self.config = config
        self.cross_attn_decoder = True
        self.layers = nn.ModuleList(
            [BltTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = BltRotaryEmbedding(config=config)
        self.patch_embedding_projection = nn.Linear(
            in_features=config.hidden_size_global,
            out_features=config.hidden_size * config.cross_attn_k,
            bias=False,
        )
        self.norm = BltRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_layers = nn.ModuleList()
        layers_to_add = config.num_hidden_layers if config.cross_attn_all_layers else 1
        for layer_idx in range(layers_to_add):
            self.cross_attn_layers.append(
                BltCrossAttention(config=config, layer_idx=layer_idx, hidden_size=config.hidden_size)
            )

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        patch_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        batch_size = inputs_embeds.shape[0]
        hidden_states = inputs_embeds
        patch_embeds = self.patch_embedding_projection(patch_embeds)
        patch_embeds = patch_embeds.reshape(
            batch_size, patch_embeds.shape[1] * self.config.cross_attn_k, self.config.hidden_size
        )

        if patch_embeds is not None and not self.cross_attn_decoder:
            hidden_states = hidden_states + patch_embeds

        if position_ids is None:
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
            )

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)

        for i, layer in enumerate(self.layers):
            if i == 0 or self.config.cross_attn_all_layers:
                # Remove cross_attention_states from kwargs if present to avoid multiple values error
                kwargs.pop("cross_attention_states", None)
                cross_attention_output, _ = self.cross_attn_layers[i](
                    hidden_states=hidden_states,
                    cross_attention_states=patch_embeds,
                    attention_mask=cross_attention_mask,
                    full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                    **kwargs,
                )
                hidden_states = hidden_states + cross_attention_output
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        logits = self.norm(hidden_states)
        return logits


class BltGlobalTransformer(nn.Module):
    def __init__(self, config: BltGlobalTransformerConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            self.layers.append(BltTransformerLayer(config, layer_idx))
        self.rotary_emb = BltRotaryEmbedding(config=config)

    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        batch_size, seq_len, _ = input_embeds.shape
        hidden_states = input_embeds
        hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)
        if position_ids is None:
            position_ids = (
                torch.arange(input_embeds.shape[1], device=input_embeds.device).unsqueeze(0).expand(batch_size, -1)
            )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        return hidden_states


@auto_docstring
class BltPreTrainedModel(MllamaPreTrainedModel):
    config: BltConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    _supports_static_cache = False  # static cache cannot have different shapes for each layer
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": OutputRecorder(BltTransformerLayer, index=0, layer_name="local_decoder"),
        "attentions": OutputRecorder(BltSelfAttention, index=1, layer_name="local_decoder"),
        "encoder_attentions": OutputRecorder(BltSelfAttention, index=1, layer_name="local_encoder"),
        "global_attentions": OutputRecorder(BltSelfAttention, index=1, layer_name="global_transformer"),
    }

    def _init_weights(self, module):
        raise AttributeError("No need to inherit it!")


class BltModel(BltPreTrainedModel):
    def __init__(self, config: BltConfig):
        super().__init__(config)
        self.gradient_checkpointing = False
        config.patcher_config._attn_implementation = config._attn_implementation
        config.encoder_config._attn_implementation = config._attn_implementation
        config.decoder_config._attn_implementation = config._attn_implementation
        config.global_config._attn_implementation = config._attn_implementation

        self.config = config
        self.local_encoder = BltLocalEncoder(config.encoder_config)
        self.global_transformer = BltGlobalTransformer(config.global_config)
        self.local_decoder = BltLocalDecoder(config.decoder_config)
        num_embeddings = config.encoder_hash_byte_group_nb_functions * len(config.encoder_hash_byte_group_size)
        embeddings = [
            nn.Embedding(config.encoder_hash_byte_group_vocab, config.encoder_config.hidden_size)
            for _ in range(num_embeddings)
        ]
        self.encoder_hash_tok_embedding = nn.ModuleList(embeddings)
        if self.config.patch_in_forward:
            self.patcher = BltPatcher(config.patcher_config)
            self.patcher.eval()
            for param in self.patcher.parameters():
                param.requires_grad = False
        else:
            self.patcher = None
        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        patch_lengths: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape
        else:
            batch_size, sequence_length, _ = inputs_embeds.shape
        if patch_lengths is None:
            if self.config.patching_mode == "entropy" and self.patcher is not None:
                if input_ids is None:
                    raise ValueError("input_ids is required for entropy-based patching")
                _, patch_lengths, _ = self.patcher(
                    input_ids,
                    patch_size=self.config.patch_size,
                    threshold=self.config.patching_threshold,
                    max_patch_length=self.config.max_patch_length,
                    patching_batch_size=self.config.patching_batch_size,
                    device=input_ids.device,
                )
            else:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                dtype = input_ids.dtype if input_ids is not None else inputs_embeds.dtype
                patch_lengths = process_patch_lengths(
                    torch.ones((batch_size, sequence_length + 1), dtype=dtype, device=device),
                    self.config.max_patch_length,
                )
        patch_ids = self._patch_ids_from_lengths(patch_lengths, sequence_length)
        if inputs_embeds is not None:
            encoder_embeds = inputs_embeds
        else:
            encoder_embeds = compute_hash_embeddings(
                input_ids,
                self.local_encoder,
                self.encoder_hash_tok_embedding,
                self.config.encoder_hash_byte_group_nb_functions,
                self.config.encoder_hash_byte_group_size,
                self.config.encoder_hash_byte_group_vocab,
            )
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + encoder_embeds.shape[1], device=encoder_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=encoder_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        cross_attn_mask_enc = _prepare_patch_cross_attention_mask(
            patch_ids, patch_lengths.shape[1], sequence_length, True, self.config.cross_attn_k, encoder_embeds.dtype
        )
        # Remove full_text_row_masked_out_mask from kwargs if present to avoid multiple values error
        kwargs.pop("full_text_row_masked_out_mask", None)
        encoder_hidden_states, encoder_cross_states = self.local_encoder(
            input_ids=input_ids,
            inputs_embeds=encoder_embeds,
            patch_embeds=None,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            cache_position=None,
            cross_attention_mask=cross_attn_mask_enc,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
        )
        global_hidden_states = encoder_cross_states.view(batch_size, patch_lengths.shape[1], -1)
        global_cache_position = torch.arange(0, global_hidden_states.shape[1], device=global_hidden_states.device)
        global_position_ids = global_cache_position.unsqueeze(0)
        global_causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=global_hidden_states,
            attention_mask=None,
            cache_position=global_cache_position,
            past_key_values=None,
            position_ids=None,
        )

        global_hidden_states = self.global_transformer(
            input_embeds=global_hidden_states,
            attention_mask=global_causal_mask,
            position_ids=global_position_ids,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        )
        decoder_patch_ids = self._patch_ids_from_lengths(patch_lengths[:, 1:], sequence_length)
        cross_attn_mask_dec = _prepare_patch_cross_attention_mask(
            decoder_patch_ids,
            patch_lengths.shape[1],
            sequence_length,
            False,
            self.config.cross_attn_k,
            encoder_embeds.dtype,
        )
        output = self.local_decoder(
            input_ids=input_ids,
            inputs_embeds=encoder_hidden_states,
            patch_embeds=global_hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            cross_attention_mask=cross_attn_mask_dec,
        )
        return BaseModelOutputWithPast(
            last_hidden_state=output,
            past_key_values=past_key_values,
        )

    def get_input_embeddings(self):
        return self.local_encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.local_encoder.embed_tokens = value

    def _patch_ids_from_lengths(self, patch_lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch_size = patch_lengths.shape[0]
        patch_starts = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=patch_lengths.dtype, device=patch_lengths.device),
                patch_lengths.cumsum(dim=-1)[:, :-1],
            ],
            dim=-1,
        )
        token_positions = torch.arange(seq_len, device=patch_lengths.device)
        return (patch_starts.unsqueeze(1) <= token_positions.unsqueeze(0).unsqueeze(-1)).sum(dim=-1) - 1


class BltPatcher(BltPreTrainedModel):
    def __init__(self, config: BltPatcherConfig):
        super().__init__(config)
        self.rotary_emb = BltRotaryEmbedding(config=self.config)
        self.layers = nn.ModuleList()
        for layer_idx in range(self.config.num_hidden_layers):
            self.layers.append(BltTransformerLayer(self.config, layer_idx))
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.norm = BltRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        token_values: torch.Tensor,
        patch_size: Optional[int] = None,
        threshold: Optional[float] = None,
        max_patch_length: Optional[int] = None,
        patching_batch_size: int = 1,
        device: Optional[str] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        entropies = []
        predictions = []
        max_length = self.config.max_position_embeddings
        batch_numel = max_length * patching_batch_size
        splits = torch.split(token_values.flatten(), batch_numel)
        for split in splits:
            pad_size = (max_length - (split.numel() % max_length)) % max_length
            pad = torch.zeros(pad_size, dtype=split.dtype, device=split.device, requires_grad=False)
            split = torch.cat((split, pad), dim=0)
            split = split.reshape(-1, max_length)
            if device is not None:
                split = split.to(device)
            batch_size, sequence_length = split.shape
            input_embeds = self.embed_tokens(split)
            hidden_states = input_embeds
            batch_size = input_embeds.shape[0]
            position_ids = torch.arange(split.shape[1], device=input_embeds.device).unsqueeze(0).expand(batch_size, -1)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            cache_position = torch.arange(sequence_length, device=input_embeds.device)

            causal_mask = create_causal_mask(
                config=self.config,
                input_embeds=input_embeds,
                attention_mask=None,
                cache_position=cache_position,
                past_key_values=None,
                position_ids=None,
            )

            for i, layer in enumerate(self.layers):
                layer_outputs = layer(
                    hidden_states, position_embeddings=position_embeddings, attention_mask=causal_mask
                )
                hidden_states = layer_outputs[0]

            logits = self.lm_head(self.norm(hidden_states))
            logits = logits.reshape(-1, logits.shape[-1])[: split.numel() - pad_size, :]
            predictions.append(logits)
            prediction_entropies = torch.distributions.Categorical(logits=logits).entropy()
            entropies.append(prediction_entropies)
        concat_entropies = torch.cat(entropies, dim=0).reshape(token_values.shape)
        concat_predictions = torch.cat(predictions, dim=0).reshape(token_values.shape[0], -1)
        batch_size, sequence_length = token_values.shape
        if patch_size is not None:
            patch_lengths = self.patch_lengths_from_entropies(
                entropies=concat_entropies,
                sequence_length=sequence_length,
                patch_size=patch_size,
                threshold=threshold,
            )
        else:
            patch_lengths = torch.ones(
                (batch_size, sequence_length), dtype=token_values.dtype, device=token_values.device
            )
        patch_lengths = process_patch_lengths(patch_lengths, max_patch_length)
        return concat_entropies, patch_lengths, concat_predictions

    @staticmethod
    def patch_lengths_from_entropies(
        entropies,
        sequence_length,
        patch_size=None,
        threshold=None,
    ):
        """
        Computes patch lengths from token entropies.

        Depending on whether a threshold is provided, the function uses either:
        - Top-k selection based on entropy (when `threshold` is None), or
        - Thresholding the entropy values (when `threshold` is set).
        """

        batch_size = entropies.shape[0]

        # Always include token 0 and 1 as starting tokens
        init_tokens = (
            torch.tensor([0, 1], dtype=torch.long, device=entropies.device).unsqueeze(0).repeat(batch_size, 1)
        )
        offset = init_tokens.shape[1]

        # Ignore first token entropy (BOS)
        entropies = entropies[:, 1:]

        if threshold is None:
            # Use top-k entropy values to define patch start points
            num_patches = sequence_length // patch_size
            topk_indices = entropies.topk(num_patches - 2, dim=1).indices
            patch_starts = topk_indices.sort(dim=1).values
        else:
            # Threshold the entropy values to define patch start points
            patch_mask = entropies > threshold

            seq_len = patch_mask.shape[1]

            # Create patch IDs (token indices), and add a sentinel to ensure alignment
            token_indices = torch.arange(seq_len, device=entropies.device).unsqueeze(0).expand(batch_size, -1)
            sentinel = torch.full_like(token_indices, seq_len)
            padded_indices = torch.cat([token_indices, sentinel], dim=1)

            # Pad mask with inverse to align sentinel correctly
            padded_mask = torch.cat([patch_mask, ~patch_mask], dim=1)

            # Select indices where mask is True
            patch_starts = padded_indices[padded_mask].reshape(batch_size, seq_len)
            max_valid_patches = patch_mask.sum(dim=1).max()
            patch_starts = patch_starts[:, :max_valid_patches]

        # Offset patch starts to account for the two initial tokens
        patch_start_ids = torch.cat((init_tokens, patch_starts + offset), dim=1)

        # Compute patch end positions by shifting start positions
        last_token = torch.full_like(patch_start_ids[:, :1], sequence_length - 1)
        patch_ends = torch.cat((patch_start_ids[:, 1:] - 1, last_token), dim=1)

        patch_lengths = patch_ends - patch_start_ids + 1

        return patch_lengths


class BltForCausalLM(MllamaForCausalLM):
    config: BltConfig
    base_model_prefix = "model"
    _can_compile_fullgraph = False
    _tied_weights_keys = ["lm_head.weight"]
    _no_split_modules = ["BltTransformerLayer", "BltLocalEncoder", "BltLocalDecoder", "BltGlobalTransformer"]

    def __init__(self, config: BltConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.model = BltModel(config)
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, config.vocab_size, bias=False)

        self.post_init()


__all__ = [
    "BltPreTrainedModel",
    "BltModel",
    "BltPatcher",
    "BltForCausalLM",
]
