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

from collections.abc import Callable

import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import maybe_autocast, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..cohere2.modeling_cohere2 import rotate_half  # noqa: F401
from ..llama.modeling_llama import LlamaRotaryEmbedding
from ..mllama.modeling_mllama import (
    MllamaPreTrainedModel,
    MllamaSelfAttentionDecoderLayer,
    MllamaTextCrossAttention,
    MllamaTextMLP,
    MllamaTextRMSNorm,
    MllamaTextSelfAttention,
    eager_attention_forward,
)
from .configuration_blt import (
    BltConfig,
    BltGlobalTransformerConfig,
    BltLocalDecoderConfig,
    BltLocalEncoderConfig,
    BltPatcherConfig,
)


logger = logging.get_logger(__name__)


def rolling_polynomial_hash(token_tensor, prime: int = 1000000007):
    """
    A polynomial rolling hash algorithm that converts sequences
    of tokens into hash values. The hash is computed as:
        hash = (token_0 * prime^0 + token_1 * prime^1 + ... + token_n * prime^n)

    The rolling hash allows the model to efficiently
    identify and encode recurring byte-level patterns in the input text.

    Args:
        token_tensor (torch.Tensor): [batch_size, seq_len, group_size] containing token IDs to hash
        prime (int): Prime number used as the base for the polynomial hash.

    Returns:
        torch.Tensor: Hash values of shape [batch_size, seq_len] where each value
                     represents the hash of the corresponding token group

    Example:
        >>> tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> hashes = rolling_polynomial_hash(tokens, prime=31)
        >>> # hash[0] = 1*31^0 + 2*31^1 + 3*31^2
        >>> # hash[1] = 4*31^0 + 5*31^1 + 6*31^2
    """
    prime_tensor = torch.tensor(prime, dtype=torch.int64, device=token_tensor.device)
    powers = torch.arange(token_tensor.shape[-1], device=token_tensor.device)
    prime_powers = prime_tensor**powers
    return torch.sum(token_tensor * prime_powers, dim=-1)


def byte_group_hash_function(
    token_ids: torch.Tensor, group_size: int = 2, prime: int = 1000000007, max_hash: int = 30000
):
    """Hash token groups and map to range [0, max_hash]."""
    with torch.no_grad():
        batch_size, seq_len = token_ids.shape
        # Add padding for sliding window
        padding = torch.zeros(batch_size, group_size - 1, dtype=torch.int64, device=token_ids.device)
        padded_tokens = torch.cat([padding, token_ids], dim=1)

        # Create sliding windows and compute hashes
        windows = padded_tokens.unfold(1, group_size, 1)
        hashes = rolling_polynomial_hash(windows, prime)
        hash_values = hashes % max_hash

    return hash_values


def compute_hash_embeddings(
    local_encoder_tokens: torch.Tensor,
    local_encoder,
    encoder_hash_tok_embedding: nn.Embedding,
    encoder_hash_byte_group_nb_functions: int,
    encoder_hash_byte_group_size: list,
    encoder_hash_byte_group_vocab: int,
) -> torch.Tensor:
    """Compute token embeddings enhanced with hash-based embeddings."""
    # Available primes for hash functions
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

    embeddings = local_encoder.embed_tokens(local_encoder_tokens)
    embedding_idx = 0
    for func_nb in range(encoder_hash_byte_group_nb_functions):
        prime = primes[func_nb % len(primes)]  # Cycle through primes if more functions than primes
        for group_size in encoder_hash_byte_group_size:
            hash_ids = byte_group_hash_function(local_encoder_tokens, group_size, prime, encoder_hash_byte_group_vocab)
            # Apply offset to get the correct slice of the fused embedding
            offset_hash_ids = hash_ids + embedding_idx * encoder_hash_byte_group_vocab
            embeddings += encoder_hash_tok_embedding(offset_hash_ids).to(embeddings.device)
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

    return cross_attention_mask


def process_patch_lengths(patch_lengths: torch.Tensor, max_patch_length: int | None) -> torch.Tensor:
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


class BltRotaryEmbedding(LlamaRotaryEmbedding):
    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.repeat_interleave(freqs, 2, dim=-1)  # diff from Llama: we interleave() instead of cat()
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


class BltCrossAttention(MllamaTextCrossAttention):
    """Cross-attention module for Blt, following transformers style"""

    def __init__(self, config: BltConfig, layer_idx: int, hidden_size: int | None = None):
        super().__init__()
        self.is_causal = False
        self.q_norm = BltRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.k_norm = BltRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_norm(hidden_states)
        query_states = self.q_proj(query_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        cross_attention_states = self.k_norm(cross_attention_states)
        key_states = self.k_proj(cross_attention_states)
        value_states = self.v_proj(cross_attention_states)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

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


@auto_docstring
class BltPreTrainedModel(MllamaPreTrainedModel):
    config: BltConfig
    _supports_attention_backend = False
    _supports_flash_attn = False
    _supports_flex_attn = False
    _no_split_modules = ["BltTransformerLayer"]
    _can_record_outputs = {
        "hidden_states": OutputRecorder(BltTransformerLayer, index=0),
        "attentions": OutputRecorder(BltSelfAttention, index=1),
    }

    # Weight initialization is adapted from:
    # - https://github.com/facebookresearch/blt/blob/main/bytelatent/model/blt.py
    # - https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/transformers_modeling_backend/model/model.py
    #
    # Both implementations use truncated normal initialization with std ~ 1 / sqrt(d_model)
    # (or 1 / sqrt(hidden_dim) for FFN outputs), and unit initialization for normalization layers.
    # We follow the same scheme here, but expressed in the Transformers APIs.

    @torch.no_grad()
    def _init_weights(self, module):
        """
        Initialize BLT weights following the original ByteLatentTransformer:

        - Most weights are drawn from a truncated normal.
        - Scale is ~ 1 / sqrt(model_dim) (or 1 / sqrt(hidden_dim) for FFN outputs).
        - Norm layers are set to weight = 1, bias = 0.
        """
        class_name = module.__class__.__name__

        # Norms: RMSNorm / LayerNorm
        if isinstance(module, (BltRMSNorm, nn.LayerNorm)) or "RMSNorm" in class_name or "LayerNorm" in class_name:
            if getattr(module, "weight", None) is not None:
                init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)
            return

        # Embeddings (encoder / patcher / hash embeddings)
        if isinstance(module, nn.Embedding):
            hidden_size = getattr(self.config, "hidden_size", None)
            if hidden_size is None and hasattr(self.config, "encoder_config"):
                hidden_size = getattr(self.config.encoder_config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = module.embedding_dim

            std = hidden_size**-0.5
            init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
            if module.padding_idx is not None:
                init.zeros_(module.weight[module.padding_idx])
            return

        # Self-attention / cross-attention projections
        if isinstance(module, (BltSelfAttention, BltCrossAttention)) or class_name in (
            "MllamaTextSelfAttention",
            "MllamaTextCrossAttention",
        ):
            dim = getattr(self.config, "hidden_size", None)
            if dim is None and hasattr(module, "hidden_size"):
                dim = module.hidden_size
            if dim is None:
                for name in ("q_proj", "k_proj", "v_proj", "o_proj", "dense"):
                    proj = getattr(module, name, None)
                    if proj is not None and hasattr(proj, "weight"):
                        dim = proj.weight.shape[-1]
                        break
            if dim is None:
                return

            std = dim**-0.5

            # Input projections (q, k, v)
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(module, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    init.trunc_normal_(
                        proj.weight,
                        mean=0.0,
                        std=std,
                        a=-3 * std,
                        b=3 * std,
                    )
                    if getattr(proj, "bias", None) is not None:
                        init.zeros_(proj.bias)

            # Output projection: o_proj or dense
            o_proj = getattr(module, "o_proj", getattr(module, "dense", None))
            if o_proj is not None and hasattr(o_proj, "weight"):
                init.trunc_normal_(
                    o_proj.weight,
                    mean=0.0,
                    std=std,
                    a=-3 * std,
                    b=3 * std,
                )
                if getattr(o_proj, "bias", None) is not None:
                    init.zeros_(o_proj.bias)
            return

        # MLP / FFN blocks
        if isinstance(module, BltMLP) or class_name == "MllamaTextMLP":
            hidden_size = getattr(self.config, "hidden_size", None)
            if hidden_size is None and hasattr(self.config, "decoder_config"):
                hidden_size = getattr(self.config.decoder_config, "hidden_size", None)
            if hidden_size is None and hasattr(self.config, "encoder_config"):
                hidden_size = getattr(self.config.encoder_config, "hidden_size", None)

            # Input-side std
            in_std = None
            if hidden_size is not None:
                in_std = hidden_size**-0.5

            gate_proj = getattr(module, "gate_proj", getattr(module, "fc1", None))
            up_proj = getattr(module, "up_proj", None)
            down_proj = getattr(module, "down_proj", getattr(module, "fc2", None))

            # gate / input projections
            for proj in (gate_proj, up_proj):
                if proj is not None and hasattr(proj, "weight"):
                    std = in_std or (proj.weight.shape[1] ** -0.5)
                    init.trunc_normal_(
                        proj.weight,
                        mean=0.0,
                        std=std,
                        a=-3 * std,
                        b=3 * std,
                    )
                    if getattr(proj, "bias", None) is not None:
                        init.zeros_(proj.bias)

            # output/ down projections
            if down_proj is not None and hasattr(down_proj, "weight"):
                hidden_dim = down_proj.weight.shape[1]
                out_std = hidden_dim**-0.5
                init.trunc_normal_(
                    down_proj.weight,
                    mean=0.0,
                    std=out_std,
                    a=-3 * out_std,
                    b=3 * out_std,
                )
                if getattr(down_proj, "bias", None) is not None:
                    init.zeros_(down_proj.bias)
            return

        # Generic Linear layers (projections, lm_head, etc.)
        if isinstance(module, nn.Linear):
            fan_in = module.in_features
            std = fan_in**-0.5
            init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
            if module.bias is not None:
                init.zeros_(module.bias)
            return

        if isinstance(module, BltRotaryEmbedding):
            rope_fn = (
                ROPE_INIT_FUNCTIONS[module.rope_type]
                if module.rope_type != "default"
                else module.compute_default_rope_parameters
            )
            buffer_value, _ = rope_fn(module.config)
            init.copy_(module.inv_freq, buffer_value)
            init.copy_(module.original_inv_freq, buffer_value)


class BltLocalEncoder(BltPreTrainedModel):
    config: BltLocalEncoderConfig
    _can_record_outputs = {
        "encoder_attentions": OutputRecorder(BltSelfAttention, index=1, layer_name="local_encoder"),
    }

    def __init__(self, config: BltLocalEncoderConfig):
        super().__init__(config)
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

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        patch_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        num_patches: int | None = None,
        patch_ids: torch.Tensor | None = None,
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
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
            if idx == len(self.layers) - 1 or self.config.cross_attn_all_layers:
                patch_embeds = self.patch_reduce(hidden_states, num_patches, patch_ids)
                patch_embeds = self.patch_embedding_projection(patch_embeds)
                patch_embeds = patch_embeds.reshape(
                    batch_size, patch_embeds.shape[1] * self.config.cross_attn_k, self.config.hidden_size
                )
                layer_idx = idx if self.config.cross_attn_all_layers else 0
                cross_attention_output, _ = self.cross_attn_layers[layer_idx](
                    hidden_states=patch_embeds,
                    cross_attention_states=hidden_states,
                    attention_mask=encoder_attention_mask,
                    **kwargs,
                )
                patch_embeds = patch_embeds + cross_attention_output
        encoder_cross_states = patch_embeds
        return hidden_states, encoder_cross_states

    def patch_reduce(self, hidden_states, max_num_patches, patch_ids):
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
            reduce="amax",
            include_self=False,
        )
        reduced_embeddings = reduced_embeddings[:, :max_num_patches, :]

        return reduced_embeddings


class BltLocalDecoder(BltPreTrainedModel):
    config: BltLocalDecoderConfig

    def __init__(self, config: BltLocalDecoderConfig):
        super().__init__(config)
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

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        patch_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
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
                cross_attention_output, _ = self.cross_attn_layers[i](
                    hidden_states=hidden_states,
                    cross_attention_states=patch_embeds,
                    attention_mask=encoder_attention_mask,
                    **kwargs,
                )
                hidden_states = hidden_states + cross_attention_output
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        logits = self.norm(hidden_states)
        return logits


class BltGlobalTransformer(BltPreTrainedModel):
    config: BltGlobalTransformerConfig
    _can_record_outputs = {
        "global_attentions": OutputRecorder(BltSelfAttention, index=1, layer_name="global_transformer"),
    }

    def __init__(self, config: BltGlobalTransformerConfig):
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            self.layers.append(BltTransformerLayer(config, layer_idx))
        self.rotary_emb = BltRotaryEmbedding(config=config)

        # Create token embedding projection (use nn.Identity() when no projection needed)
        if getattr(config, "encoder_cross_output_size", None) is not None:
            self.token_embedding_projection = nn.Linear(
                config.encoder_cross_output_size, config.hidden_size, bias=False
            )
        else:
            self.token_embedding_projection = nn.Identity()

        self.post_init()

    @deprecate_kwarg("input_embeds", version="5.6.0", new_name="inputs_embeds")
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        batch_size, seq_len, _ = inputs_embeds.shape
        hidden_states = self.token_embedding_projection(inputs_embeds)
        hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)
        if position_ids is None:
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
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


class BltPatcher(BltPreTrainedModel):
    config: BltPatcherConfig

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

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        patch_size: int | None = None,
        threshold: float | None = None,
        max_patch_length: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings, attention_mask=causal_mask)

        logits = self.lm_head(self.norm(hidden_states))
        prediction_entropies = torch.distributions.Categorical(logits=logits).entropy()

        batch_size, sequence_length = inputs_embeds.shape[:2]
        if patch_size is not None:
            patch_lengths = self.patch_lengths_from_entropies(
                entropies=prediction_entropies,
                sequence_length=sequence_length,
                patch_size=patch_size,
                threshold=threshold,
            )
        else:
            patch_lengths = torch.ones(
                (batch_size, sequence_length), dtype=inputs_embeds.dtype, device=inputs_embeds.device
            )
        patch_lengths = process_patch_lengths(patch_lengths, max_patch_length)
        return prediction_entropies, patch_lengths, logits

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


class BltModel(BltPreTrainedModel):
    def __init__(self, config: BltConfig):
        super().__init__(config)
        self.gradient_checkpointing = False

        self.config = config
        self.local_encoder = BltLocalEncoder(config.encoder_config)
        self.global_transformer = BltGlobalTransformer(config.global_config)
        self.local_decoder = BltLocalDecoder(config.decoder_config)
        num_embeddings = config.encoder_hash_byte_group_nb_functions * len(config.encoder_hash_byte_group_size)
        total_vocab_size = config.encoder_hash_byte_group_vocab * num_embeddings
        self.encoder_hash_tok_embedding = nn.Embedding(total_vocab_size, config.encoder_config.hidden_size)
        if self.config.patch_in_forward:
            self.patcher = BltPatcher(config.patcher_config)
            self.patcher.eval()
            for param in self.patcher.parameters():
                param.requires_grad = False
        else:
            self.patcher = None
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        patch_lengths: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache:
            if past_key_values is None:
                past_key_values = EncoderDecoderCache(
                    DynamicCache(config=self.config), DynamicCache(config=self.config)
                )
            elif not isinstance(past_key_values, EncoderDecoderCache):
                # BLT uses an encoder-decoder cache even though it is not en encoder-decoder model. Create a cross-cache
                # if not yet created by the user
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache(config=self.config))

        # Extract input embeddings as early as possible
        if inputs_embeds is not None:
            encoder_embeds = inputs_embeds
            batch_size, sequence_length, _ = inputs_embeds.shape
        else:
            batch_size, sequence_length = input_ids.shape
            encoder_embeds = compute_hash_embeddings(
                input_ids,
                self.local_encoder,
                self.encoder_hash_tok_embedding,
                self.config.encoder_hash_byte_group_nb_functions,
                self.config.encoder_hash_byte_group_size,
                self.config.encoder_hash_byte_group_vocab,
            )

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
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + encoder_embeds.shape[1], device=encoder_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=encoder_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values.self_attention_cache if past_key_values is not None else None,
            position_ids=position_ids,
        )

        cross_attn_mask_enc = _prepare_patch_cross_attention_mask(
            patch_ids=patch_ids,
            num_patches=patch_lengths.shape[1],
            sequence_length=sequence_length,
            patches_as_queries=True,
            cross_attn_k=self.config.cross_attn_k,
            dtype=encoder_embeds.dtype,
        )
        encoder_hidden_states, encoder_cross_states = self.local_encoder(
            input_ids=input_ids,
            inputs_embeds=encoder_embeds,
            attention_mask=causal_mask,
            position_ids=position_ids,
            encoder_attention_mask=cross_attn_mask_enc,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
            past_key_values=past_key_values.self_attention_cache if past_key_values is not None else None,
            **kwargs,
        )
        encoder_cross_states = encoder_cross_states.view(batch_size, patch_lengths.shape[1], -1)
        global_cache_position = torch.arange(0, encoder_cross_states.shape[1], device=encoder_cross_states.device)
        global_position_ids = global_cache_position.unsqueeze(0)
        global_causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=encoder_cross_states,
            attention_mask=None,
            cache_position=global_cache_position,
            past_key_values=None,
            position_ids=None,
        )

        global_hidden_states = self.global_transformer(
            inputs_embeds=encoder_cross_states,
            attention_mask=global_causal_mask,
            position_ids=global_position_ids,
            **kwargs,
        )
        decoder_patch_ids = self._patch_ids_from_lengths(patch_lengths[:, 1:], sequence_length)
        cross_attn_mask_dec = _prepare_patch_cross_attention_mask(
            patch_ids=decoder_patch_ids,
            num_patches=patch_lengths.shape[1],
            sequence_length=sequence_length,
            patches_as_queries=False,
            cross_attn_k=self.config.cross_attn_k,
            dtype=encoder_embeds.dtype,
        )
        output = self.local_decoder(
            input_ids=input_ids,
            inputs_embeds=encoder_hidden_states,
            patch_embeds=global_hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values.cross_attention_cache if past_key_values is not None else None,
            cache_position=cache_position,
            encoder_attention_mask=cross_attn_mask_dec,
            **kwargs,
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


@auto_docstring(
    custom_intro="""
    The Blt Text Model with a language modeling head on top.
    """
)
class BltForCausalLM(BltPreTrainedModel, GenerationMixin):
    config: BltConfig
    _can_compile_fullgraph = False
    base_model_prefix = "model"
    _tied_weights_keys = {"model.local_encoder.embed_tokens.weight": "lm_head.weight"}

    def __init__(self, config: BltConfig):
        super().__init__(config)
        self.text_config = config.get_text_config()
        self.vocab_size = config.vocab_size
        self.model = BltModel(config)
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cross_attention_states: torch.LongTensor | None = None,  # Keep for compatibility
        cross_attention_mask: torch.LongTensor | None = None,
        full_text_row_masked_out_mask: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        cross_attention_states (`torch.FloatTensor`, *optional*):
            Output of the vision model, used for cross-attention. This tensor contains the processed image features that
            the language model will attend to.
        cross_attention_mask (`torch.Tensor` of shape `(batch_size, seq_length, max_num_images, max_num_tiles)`, *optional*):
            Cross-attention mask to control the interaction between text tokens and image tiles.
            This 4D tensor defines which image tiles each text token should attend to.

            For each text token (in seq_length):
            - 1 indicates the token **should attend** to the corresponding image tile
            - 0 indicates the token **should not attend** to the corresponding image tile
        full_text_row_masked_out_mask (`tuple[torch.Tensor, torch.Tensor]`, *optional*):
            A tuple containing two tensors that mask out rows in the cross-attention mechanism:
            - The first tensor has shape `(batch_size, 1, seq_length, 1)` and contains values of 0 or 1.
              A value of 0 indicates that the corresponding text token's entire row in the cross-attention
              matrix should be masked out (all image tokens ignored).
            - The second tensor has the same shape and is used internally to apply the masking during
              the forward pass of cross-attention layers.
            This mask is derived from the cross_attention_mask and is used to handle cases where a text token
            should not attend to any image token.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BltForCausalLM

        >>> model = BltForCausalLM.from_pretrained("itazap/blt-1b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("itazap/blt-1b-hf")

        >>> prompt = "If I had to write a haiku, it would be:"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=40, do_sample=True, temperature=0.6)
        >>> result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(result)
        If I had to write a haiku, it would be: "Snowflakes gently fall" - simple, yet peaceful.
        I love the idea of snowflakes gently falling, each one
        ```
        """
        # Call parent forward but exclude cross_attention_states from model call
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]).float()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "BltPreTrainedModel",
    "BltModel",
    "BltPatcher",
    "BltForCausalLM",
]
