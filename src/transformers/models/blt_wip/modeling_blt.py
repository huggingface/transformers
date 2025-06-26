# coding=utf-8
# Copyright 2025 the Facebook Research and HuggingFace Inc. team. All rights reserved.
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
"""BLT model."""

from ...utils import is_torch_flex_attn_available, logging
from typing import Callable, List, Optional, Tuple, Union

from ...cache_utils import Cache
from ...activations import ACT2FN

import torch
import torch.distributions
import torch.nn
import torch.nn as nn
from torch.nn import functional as F

from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from .configuration_blt import (
    BLTConfig,
    BLTLocalEncoderConfig,
    BLTLocalDecoderConfig,
    BLTGlobalTransformerConfig,
    BLTPatcherConfig,
    PatchingModeEnum,
)

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
    from ...integrations.flex_attention import make_flex_block_causal_mask


logger = logging.get_logger(__name__)


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


def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # TODO: not exactly equivalent to other transformers implementations,, need feedback
    # Extract first head_dim//2 elements which correspond to the unique frequencies
    # This matches the original BLT approach which uses head_dim//2 frequency pairs
    head_dim = q.shape[-1]
    cos_freqs = cos[..., :head_dim//2]  # [B, S, D/2]
    sin_freqs = sin[..., :head_dim//2]  # [B, S, D/2]
    
    # Expand cos/sin to match query/key tensor format [B, H, S, D/2]
    cos_freqs = cos_freqs.unsqueeze(1).expand(-1, q.shape[1], -1, -1)  # [B, 1, S, D/2] -> [B, H, S, D/2]
    sin_freqs = sin_freqs.unsqueeze(1).expand(-1, q.shape[1], -1, -1)  # [B, 1, S, D/2] -> [B, H, S, D/2]
    
    # Split q and k into pairs for rotation: (d0, d1), (d2, d3), ...
    q_pairs = q.view(*q.shape[:-1], head_dim//2, 2)  # [B, H, S, D/2, 2]
    k_pairs = k.view(*k.shape[:-1], head_dim//2, 2)  # [B, H, S, D/2, 2]
    
    # Extract real and i parts
    q_real, q_imag = q_pairs[..., 0], q_pairs[..., 1]  # [B, H, S, D/2]
    k_real, k_imag = k_pairs[..., 0], k_pairs[..., 1]  # [B, H, S, D/2]
    
    # Apply rotation: [real', imag'] = [cos*real - sin*imag, sin*real + cos*imag]
    q_real_rot = cos_freqs * q_real - sin_freqs * q_imag
    q_imag_rot = sin_freqs * q_real + cos_freqs * q_imag
    k_real_rot = cos_freqs * k_real - sin_freqs * k_imag
    k_imag_rot = sin_freqs * k_real + cos_freqs * k_imag
    
    # Recombine pairs and reshape back to original format
    q_rot = torch.stack([q_real_rot, q_imag_rot], dim=-1).view(*q.shape)  # [B, H, S, D]
    k_rot = torch.stack([k_real_rot, k_imag_rot], dim=-1).view(*k.shape)  # [B, H, S, D]
    
    return q_rot.type_as(q), k_rot.type_as(k)


class BLTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    

class BLTRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BLTRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class BLTTransformerLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = BLTSelfAttention(config=config, layer_idx=layer_idx)
        self.mlp = BLTMLP(config)
        self.input_layernorm = BLTRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = BLTRMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.LongTensor`, *optional*):
                Position indices of tokens in the sequence for RoPE computation.
            past_key_value (`Cache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

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

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BLTSelfAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = config.rope_theta
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            position_embeddings: torch.Tensor,
            output_attentions: bool = False,
            use_cache: bool = False,
            past_key_value=None,
            cache_position=None,
            **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        output_attentions = False
        self.config._attn_implementation = "sdpa"
        self.scaling = None
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
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
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    

def rolling_polynomial_hash(token_tensor, hash_func_nb: int = 0):
    primes = [
        1000000007, 5915587277, 1500450271, 3267000013, 5754853343,
        4093082899, 9576890767, 3628273133, 2860486313, 5463458053, 3367900313,
    ]
    prime = torch.tensor(primes[hash_func_nb], dtype=torch.int64, device=token_tensor.device)
    powers = torch.arange(token_tensor.shape[-1], device=token_tensor.device)
    prime_powers = prime ** powers
    return torch.sum(token_tensor * prime_powers, dim=-1)


def byte_group_hash_function(token_ids: torch.Tensor, group_size: int = 2, hash_func_nb: int = 0, max_hash: int = 30000):
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
        
    hash_values.requires_grad = False
    return hash_values


def init_hash_embeddings(config, local_encoder_dim: int, encoder_hash_byte_group_size: list):
    """Initialize hash-based token embeddings for the BLT encoder."""
    num_embeddings = config.encoder_hash_byte_group_nb_functions * len(encoder_hash_byte_group_size)
    embeddings = [
        nn.Embedding(config.encoder_hash_byte_group_vocab, local_encoder_dim)
        for _ in range(num_embeddings)
    ]
    return nn.ModuleList(embeddings)


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
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        q_patch_ids = torch.arange(num_patches, device=device).unsqueeze(0).unsqueeze(-1).expand(
            batch_size, num_patches, seq_len
        )
        kv_patch_ids = patch_ids.unsqueeze(1).expand(batch_size, num_patches, seq_len)
    else:
        q_len = sequence_length
        kv_len = num_patches * cross_attn_k
        # Create sequence-to-patch mapping
        q_patch_ids = patch_ids.unsqueeze(-1).expand(batch_size, seq_len, num_patches)
        kv_patch_ids = torch.arange(num_patches, device=device).unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, num_patches
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
        raise ValueError(f"Cross attention mask shape {cross_attention_mask.shape} doesn't match expected {expected_shape}")
    
    # Reshape so it can be used by attn module - add head dimension
    cross_attention_mask = cross_attention_mask.unsqueeze(1)  # [batch_size, 1, q_len, kv_len]
    
    # Invert the mask (following mllama pattern exactly)
    # True -> 0.0 (attend), False -> 1.0 (will become -inf)
    inverted_cross_attn_mask = (1.0 - cross_attention_mask.to(dtype))
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
    
    return cross_attention_mask, full_text_row_masked_out_mask


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
            padded[i, :len(splits)] = torch.tensor(splits, dtype=patch_lengths.dtype, device=patch_lengths.device)

    # Trim zero columns
    if (padded != 0).any(dim=0).sum() < padded.shape[1]:
        last_nonzero = (padded != 0).any(dim=0).nonzero().max().item() + 1
        padded = padded[:, :last_nonzero]

    return padded


class BLTRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.rope_type = config.rope_scaling["rope_type"]
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class BLTLocalEncoder(nn.Module):
    def __init__(self, config: BLTLocalEncoderConfig):
        super().__init__()
    
        self.hidden_size = config.hidden_size
        self.vocab_size=config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.dropout = config.dropout
        self.cross_attn_all_layers = config.cross_attn_all_layers
        self.cross_attn_k = config.cross_attn_k
        
        self.layers = nn.ModuleList([BLTTransformerLayer(config, layer_idx) for layer_idx in range(self.num_hidden_layers)])

        self.rotary_emb = BLTRotaryEmbedding(config=config)

        self.patch_embedding_projection = nn.Linear(
            in_features=config.encoder_dim_patch_emb,
            out_features=config.encoder_dim_token_emb * config.cross_attn_k,
            bias=False,
        )

        self.embed_tokens = nn.Embedding(self.vocab_size + config.pm_size, self.hidden_size)

        self.cross_attn_layers = torch.nn.ModuleList()
        layers_to_add = self.num_hidden_layers if self.cross_attn_all_layers else 1
        for layer_idx in range(layers_to_add):
            self.cross_attn_layers.append(
                BLTCrossAttention(config=config, layer_idx=layer_idx, hidden_size=self.hidden_size)
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        input_embeds: Optional[torch.Tensor] = None,
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union["BlockMask", torch.Tensor, str]] = None,
        cross_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_patches: Optional[int] = None,
        patch_ids: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        """ """
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)

        batch_size, _, _ = input_embeds.shape

        hidden_states = nn.functional.dropout(input_embeds, p=self.dropout, training=self.training) 

        position_ids = torch.arange(input_ids.shape[1], device=input_embeds.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)  

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        for idx, layer in enumerate(self.layers):
            layer_outputs = layer(hidden_states, position_embeddings=position_embeddings, attention_mask=None)
            hidden_states = layer_outputs[0]

            if idx == len(self.layers) - 1 or self.cross_attn_all_layers:
                patch_embeds = self.patch_reduce(hidden_states, num_patches, "amax", patch_ids)
                patch_embeds = self.patch_embedding_projection(patch_embeds)
                patch_embeds = patch_embeds.reshape(batch_size, patch_embeds.shape[1] * self.cross_attn_k, self.hidden_size)

                layer_idx = idx if self.cross_attn_all_layers else 0
                cross_attention_output, _, _ = self.cross_attn_layers[layer_idx](
                    hidden_states=patch_embeds,
                    cross_attention_states=hidden_states,
                    attention_mask=cross_mask,
                    full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=None,
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
        batch_size, _, embedding_dim = hidden_states.shape

        patch_ids = patch_ids.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])

        reduced_embeddings = torch.zeros((batch_size, max_num_patches, embedding_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        reduced_embeddings = reduced_embeddings.scatter_reduce(
            src=hidden_states,
            dim=1,
            index=patch_ids,
            reduce=reduction,
            include_self=False,
        )
        reduced_embeddings = reduced_embeddings[:, :max_num_patches, :]

        return reduced_embeddings


class BLTLocalDecoder(nn.Module):
    def __init__(self, config: BLTLocalDecoderConfig):
        super().__init__()

        # Extract config values to instance attributes
        self.hidden_size = config.hidden_size
        self.vocab_size=config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.dropout = config.dropout
        self.cross_attn_decoder = True #config.cross_attn_decoder #TODO: maybe remove
        self.cross_attn_all_layers = config.cross_attn_all_layers
        self.cross_attn_k = config.cross_attn_k

        self.layers = nn.ModuleList([BLTTransformerLayer(config, layer_idx) for layer_idx in range(self.num_hidden_layers)])

        self.rotary_emb = BLTRotaryEmbedding(config=config)

        self.patch_embedding_projection = nn.Linear(
            in_features=config.hidden_size_global,
            out_features=config.decoder_dim_token_emb * config.cross_attn_k,
            bias=False,
        )

        self.norm = BLTRMSNorm(self.hidden_size, eps=config.norm_eps)

        self.cross_attn_layers = torch.nn.ModuleList()
        layers_to_add = self.num_hidden_layers if self.cross_attn_all_layers else 1
        for layer_idx in range(layers_to_add):
            self.cross_attn_layers.append(
                BLTCrossAttention(config=config, layer_idx=layer_idx, hidden_size=self.hidden_size)
            )

        self.lm_head = nn.Linear(
            self.hidden_size,
            self.vocab_size,
            bias=False,
        )


    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor],
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union["BlockMask", torch.Tensor, str]] = None,
        cross_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        batch_size, _, _ = embeds.shape

        hidden_states = embeds

        patch_embeds = self.patch_embedding_projection(patch_embeds)
        patch_embeds = patch_embeds.reshape(batch_size, patch_embeds.shape[1] * self.cross_attn_k, self.hidden_size)

        if patch_embeds is not None and not self.cross_attn_decoder:
            hidden_states = hidden_states + patch_embeds

        position_ids = torch.arange(tokens.shape[1], device=embeds.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)  

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        for i, layer in enumerate(self.layers):
            if i == 0 or self.cross_attn_all_layers:
                # Use cross attention to extract info from patch_embeds into hidden_states
                cross_attention_output, _, _ = self.cross_attn_layers[i](
                    hidden_states=hidden_states,
                    cross_attention_states=patch_embeds,
                    attention_mask=cross_mask,
                    full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=None,
                )
                hidden_states = hidden_states + cross_attention_output

            layer_outputs = layer(hidden_states, position_embeddings=position_embeddings, attention_mask=None)
            hidden_states = layer_outputs[0]

        logits = self.lm_head(self.norm(hidden_states))
        return logits, cache


class BLTCrossAttention(nn.Module):
    """Cross-attention module for BLT, following transformers style"""

    def __init__(self, config: BLTConfig, layer_idx: int, hidden_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # Use provided hidden_size or fallback to encoder dimension
        self.hidden_size = hidden_size or config.hidden_size_local_encoder
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads  # Assuming same for cross attention
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = None #self.head_dim ** -0.5
        self.dropout = config.dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.hidden_size, eps=config.norm_eps)
        self.k_norm = nn.RMSNorm(self.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_norm(hidden_states) # BLT normalizes first
        query_states = self.q_proj(query_states)

        if cross_attention_states is not None:
            cross_attention_states = self.k_norm(cross_attention_states)  # BLT normalizes first
            key_states = self.k_proj(cross_attention_states)
            value_states = self.v_proj(cross_attention_states)
            if past_key_value is not None:
                # if we have a new cross attention states + new tokens, we only computed key_states on that new cross attention states
                # we still update the cross key states, past_cross_states, new_cross_states. And use it!
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
        elif cache_position is not None and cache_position[0] != 0:
            key_states, value_states = (
                past_key_value.key_cache[self.layer_idx],
                past_key_value.value_cache[self.layer_idx],
            )
        else:
            if cross_attention_states is None:
                raise ValueError(
                    "Cross attention layer can't find neither `cross_attention_states` nor cached values for key/values!"
                )

        attention_interface: Callable = eager_attention_forward

        self.config._attn_implementation = "sdpa" 
        attn = "sdpa"
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0, #if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        # Apply full row masking if provided (following mllama pattern)
        if full_text_row_masked_out_mask is not None:
            attn_output = full_text_row_masked_out_mask[:, 0] * attn_output

        attn_output = attn_output + hidden_states

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class BLTGlobalTransformer(nn.Module):
    def __init__(self, config: BLTGlobalTransformerConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.dropout = config.dropout

        self.layers = nn.ModuleList()
        for layer_idx in range(self.num_hidden_layers):
            self.layers.append(BLTTransformerLayer(config, layer_idx))

        self.rotary_emb = BLTRotaryEmbedding(config=config)


    def forward(
        self,
        input_embeds: torch.Tensor,
        mask: Optional[Union[BlockMask, torch.Tensor, str]] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        batch_size, seq_len, _ = input_embeds.shape

        hidden_states = input_embeds

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        position_ids = torch.arange(seq_len, device=input_embeds.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)  

        for i, layer in enumerate(self.layers):
            layer_outputs = layer(hidden_states, position_embeddings=position_embeddings, attention_mask=None)
            hidden_states = layer_outputs[0]

        return hidden_states, cache




class BLTPreTrainedModel(PreTrainedModel):
    config_class = BLTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BLTTransformerLayer", "BLTLocalEncoder", "BLTLocalDecoder", "BLTGlobalTransformer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = False  # BLT uses its own attention implementation
    _supports_sdpa = True
    _supports_cache_class = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = getattr(module, '_custom_std', module.in_features ** (-0.5))
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            std = getattr(module, '_custom_std', module.embedding_dim ** (-0.5))
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
             
        elif isinstance(module, BLTModel):
            if module.encoder_hash_tok_embedding is not None:
                emb_std = module.config.hidden_size_local_encoder ** (-0.5)
                for emb in module.encoder_hash_tok_embedding:
                    emb._custom_std = emb_std
                    
        elif isinstance(module, BLTLocalEncoder):
            if module.patch_embedding_projection is not None:
                module.patch_embedding_projection._custom_std = module.config.encoder_dim_patch_emb ** (-0.5)
                
        elif isinstance(module, BLTLocalDecoder):
            if module.patch_embedding_projection is not None:
                module.patch_embedding_projection._custom_std = module.config.hidden_size_global ** (-0.5)
                
        elif isinstance(module, BLTPatcher):
            emb_std = module.config.hidden_size ** (-0.5)
            module.embed_tokens._custom_std = emb_std
            module.lm_head._custom_std = emb_std


class BLTModel(BLTPreTrainedModel):
    def __init__(self, config: BLTConfig):
        super().__init__(config)

        self.config = config

        self.local_encoder = BLTLocalEncoder(config.encoder_config)
        self.global_transformer = BLTGlobalTransformer(config.global_config)
        self.local_decoder = BLTLocalDecoder(config.decoder_config)

        self.encoder_hash_tok_embedding = init_hash_embeddings(
            config,
            local_encoder_dim=config.hidden_size_local_encoder,
            encoder_hash_byte_group_size=config.encoder_hash_byte_group_size,
        )

        if self.config.patch_in_forward:
            self.patcher = BLTPatcher(config.patcher_config)
            self.patcher.eval()
            for param in self.patcher.parameters():
                param.requires_grad = False
        else:
            self.patcher = None

    def forward(self, tokens: torch.Tensor, patch_lengths: Optional[torch.Tensor] = None):
        batch_size, sequence_length = tokens.shape

        # Handle patching
        if patch_lengths is None:
            if self.config.patching_mode == PatchingModeEnum.entropy:
                _, patch_lengths, _ = self.patcher(
                    tokens,
                    patch_size=self.config.patch_size,
                    threshold=self.config.patching_threshold,
                    max_patch_length=self.config.max_patch_length,
                    patching_batch_size=self.config.patching_batch_size,
                    device=self.config.patching_device,
                )
            else:
                # Default to byte-level patching
                patch_lengths = process_patch_lengths(
                    torch.ones((batch_size, sequence_length + 1), dtype=tokens.dtype, device=tokens.device),
                    self.config.max_patch_length
                )

        patch_ids = self._patch_ids_from_lengths(patch_lengths, sequence_length)
        cross_attn_mask_enc, full_text_row_masked_out_mask_enc = _prepare_patch_cross_attention_mask(
            patch_ids, patch_lengths.shape[1], sequence_length, True, self.config.cross_attn_k, torch.float32
        )

        encoder_embeds = compute_hash_embeddings(
            tokens, self.local_encoder, self.encoder_hash_tok_embedding,
            self.config.encoder_hash_byte_group_nb_functions,
            self.config.encoder_hash_byte_group_size,
            self.config.encoder_hash_byte_group_vocab,
        )

        encoder_hidden_states, encoder_cross_states = self.local_encoder(
            input_ids=tokens,
            input_embeds=encoder_embeds,
            patch_embeds=None,
            cross_mask=cross_attn_mask_enc,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask_enc,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
        )

        global_hidden_states = encoder_cross_states.view(batch_size, patch_lengths.shape[1], -1)

        global_hidden_states, _ = self.global_transformer(
            input_embeds=global_hidden_states,
        )

        decoder_patch_ids = self._patch_ids_from_lengths(patch_lengths[:, 1:], sequence_length)
        cross_attn_mask_dec, full_text_row_masked_out_mask_dec = _prepare_patch_cross_attention_mask(
            decoder_patch_ids, patch_lengths.shape[1], sequence_length, False, self.config.cross_attn_k, torch.float32
        )

        output, _ = self.local_decoder(
            tokens=tokens,
            embeds=encoder_hidden_states,
            patch_embeds=global_hidden_states,
            mask=None,
            cross_mask=cross_attn_mask_dec,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask_dec,
        )
        
        return output
    
    def _patch_ids_from_lengths(self, patch_lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Convert patch lengths to patch IDs for each token position."""
        batch_size = patch_lengths.shape[0]
        patch_starts = torch.cat([
            torch.zeros(batch_size, 1, dtype=patch_lengths.dtype, device=patch_lengths.device),
            patch_lengths.cumsum(dim=-1)[:, :-1]
        ], dim=-1)
        
        token_positions = torch.arange(seq_len, device=patch_lengths.device)
        return (patch_starts.unsqueeze(1) <= token_positions.unsqueeze(0).unsqueeze(-1)).sum(dim=-1) - 1
    

class BLTPatcher(BLTPreTrainedModel):
    def __init__(self, config: BLTPatcherConfig):
        super().__init__(config)

        self.rotary_emb = BLTRotaryEmbedding(config=self.config)

        self.layers = nn.ModuleList()
        # Create transformer layers using the patcher config
        for layer_idx in range(self.config.num_hidden_layers):
            self.layers.append(BLTTransformerLayer(self.config, layer_idx))


        self.embed_tokens = torch.nn.Embedding(self.config.vocab_size, self.config.hidden_size)

        self.norm = BLTRMSNorm(self.config.hidden_size, eps=self.config.norm_eps)

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
    ):

        # Handle chunked processing for entropy calculation
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

            # Process chunk: embeddings -> layers -> output
            batch_size, sequence_length = split.shape
            input_embeds = self.embed_tokens(split)

            hidden_states = input_embeds

            batch_size, _, _ = input_embeds.shape

            position_ids = torch.arange(split.shape[1], device=input_embeds.device).unsqueeze(0).expand(batch_size, -1)
            
            position_embeddings = self.rotary_emb(hidden_states, position_ids)  # = BLT self.rope
            
            for i, layer in enumerate(self.layers):
                layer_outputs = layer(hidden_states, position_embeddings=position_embeddings, attention_mask=None) #, attn_impl=self.config.patcher_attn_impl )
                hidden_states = layer_outputs[0]

            logits = self.lm_head(self.norm(hidden_states))
            logits = logits.reshape(-1, logits.shape[-1])[: split.numel() - pad_size, :]  # [batch_size * seq_len, vocab]
            predictions.append(logits)
            prediction_entropies = torch.distributions.Categorical(logits=logits).entropy()
            entropies.append(prediction_entropies)

        concat_entropies = torch.cat(entropies, dim=0).reshape(token_values.shape)
        concat_predictions = torch.cat(predictions, dim=0).reshape(token_values.shape[0], -1)

        # Always compute patch lengths from concatenated entropies
        batch_size, sequence_length = token_values.shape

        # Find patch start IDs based on entropy
        if patch_size is not None:
            patch_lengths = self.patch_lengths_from_entropies(
                entropies=concat_entropies,
                sequence_length=sequence_length,
                patch_size=patch_size,
                threshold=threshold,
            )
        else:
            # Default to byte-level patching
            patch_lengths = torch.ones((batch_size, sequence_length), dtype=token_values.dtype, device=token_values.device)
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
        init_tokens = torch.tensor([0, 1], dtype=torch.long, device=entropies.device).unsqueeze(0).repeat(batch_size, 1)
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

__all__ = [
    "BLTPreTrainedModel",
    "BLTModel",
    "BLTPatcher",
    "BLTLocalEncoder",
    "BLTLocalDecoder", 
    "BLTGlobalTransformer",
    "BLTTransformerLayer",
]