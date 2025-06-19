# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
from typing import List, Optional, Tuple, Union
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from .configuration_blt import (
    BLTConfig,
    PatchingModeEnum,
)

RMSNorm = nn.RMSNorm

logger = logging.getLogger()

flex_attention_comp = flex_attention

def cross_entropy(pred, target, **kwargs):
    return F.nll_loss(
        F.log_softmax(pred.flatten(end_dim=-2).float(), -1),
        target.flatten(end_dim=-1),
        **kwargs,
    )


def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    assert dim == 2, "Only dim=2 is supported. Check the implementation for other dims."
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class BLTMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mp_size: int = 1,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B S D
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)
        return output


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


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



class BLTSelfAttention(nn.Module):
    def __init__(self, config: BLTConfig, layer_idx: int):
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

        self.wq = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

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

        query_states = self.wq(hidden_states)
        key_states = self.wk(hidden_states)
        value_states = self.wv(hidden_states)
        
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
        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class BLTTransformerLayer(nn.Module):
    def __init__(self, dim, n_heads, config, layer_idx=0):
        super().__init__()

        # Extract parameters from dictionary
        dim = dim
        n_heads = n_heads
        head_dim = getattr(config, "head_dim", None)
        n_kv_heads = getattr(config, "n_kv_heads", None)
        rope_theta = getattr(config, "rope_theta", None)
        multiple_of = getattr(config, "multiple_of", 256)
        ffn_dim_multiplier = getattr(config, "ffn_dim_multiplier", None)
        norm_eps = getattr(config, "norm_eps", None)

        self.head_dim = head_dim or dim // n_heads
        self.n_heads = n_heads or dim // head_dim
        self.n_kv_heads = n_kv_heads or self.n_heads

        config.hidden_size = dim

        self.attention = BLTSelfAttention(config=config, layer_idx=layer_idx)

        self.feed_forward = BLTMLP(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[bool] = None,
        position_embeddings: Optional[torch.Tensor] = None,

        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,

    ) -> torch.Tensor:

        residual = hidden_states
        norm_hidden_states = self.attention_norm(hidden_states)


        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=norm_hidden_states,
            # TODO: = BLT, attn_out = self.attention(self.attention_norm(x), in TransformerBlock.forward,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings
        )

        h = residual + hidden_states
        h_norm = self.ffn_norm(h)
        out = h + self.feed_forward(h_norm)
        return out

def check_non_zero_after_zero(tensor):
    zero_mask = tensor == 0
    shifted_mask = torch.cat(
        [
            torch.zeros(tensor.shape[0], 1, dtype=torch.bool, device=tensor.device),
            zero_mask[:, :-1],
        ],
        dim=1,
    )
    non_zero_after_zero = (tensor != 0) & shifted_mask
    return non_zero_after_zero.any()

def rolling_polynomial_hash(t, hash_func_nb: int = 0):
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
    prime = torch.tensor(primes[hash_func_nb], dtype=torch.int64, device=t.device)
    prime_powers = torch.stack([prime**i for i in range(t.shape[-1])])
    return torch.sum(t * prime_powers, dim=-1)


def byte_group_hash_function(x: torch.Tensor, group_size: int = 2, hash_func_nb: int = 0, max_hash: int = 30000):
    """
    Returns a hash of the input x and maps it to a value in the range [0, max_hash].

    expects: x of shape (batch_size, seq_len) with values as ids in the token vocab.
    returns a tensor  of shape (batch_size, seq_len) with values in the range [0, max_hash].

    Note: max hash can make a big difference on the number of collisions.
    """
    with torch.no_grad():
        bs, seq_len = x.shape
        prefix = torch.zeros(bs, group_size - 1, dtype=torch.int64, device=x.device)
        x = torch.cat([prefix, x], dim=1)
        windows = x.unfold(1, group_size, 1)
        # hashes = get_rolling_polynomial_hash_fn(hash_func_nb, group_size)(windows)
        hashes = rolling_polynomial_hash(windows, hash_func_nb)
        hash_values_range = hashes % max_hash
    hash_values_range.requires_grad = False
    return hash_values_range


def create_patch_mask_from_ids(patch_ids, num_patches, window=None, patches_as_queries=False):
    """
    Creates a tensor of shape [bs, seq_len, num_patches] where each element at position (i, j, k)
    is True if the patch id at position (i, j) is less than or equal to k.
    Args:
        patch_ids (torch.Tensor): Tensor of shape [bs, seq_len] containing patch ids.
        num_patches (int): Total number of patches.
        window (int): If not None, only considers patches within a window of size window.
        patches_as_queries (bool): If True, the patches are used as queries
    Returns:
        torch.Tensor: Tensor of shape [bs, q_len, kv_len] with the desired mask.
    """
    bs, seq_len = patch_ids.shape
    if not patches_as_queries:
        q_ids = patch_ids.unsqueeze(-1).expand(bs, seq_len, num_patches)
        kv_ids = (
            torch.arange(num_patches, device=patch_ids.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(bs, seq_len, num_patches)
        )
    else:
        kv_ids = patch_ids.unsqueeze(1).expand(bs, num_patches, seq_len)
        q_ids = (
            torch.arange(num_patches, device=patch_ids.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(bs, num_patches, seq_len)
        )
    if window is None:
        mask = q_ids == kv_ids
    else:
        mask = (kv_ids <= q_ids) & (q_ids < kv_ids + window)
    return mask


def cross_attn_mask(
    patch_ids,
    patch_lengths,
    N,
    patches_as_queries=False,
    cross_attn_k=1,
    window=None,
    block_mask=True,
):
    bs = patch_ids.shape[0]
    with torch.no_grad():
        # Create the patch mask
        cross_mask = create_patch_mask_from_ids(
            patch_ids,
            patch_lengths.shape[1],
            window=window,
            patches_as_queries=patches_as_queries,
        ).repeat_interleave(cross_attn_k, dim=1 if patches_as_queries else -1)
        q_len = patch_lengths.shape[1] * cross_attn_k if patches_as_queries else N
        kv_len = N if patches_as_queries else patch_lengths.shape[1] * cross_attn_k
        assert cross_mask.shape == (
            bs,
            q_len,
            kv_len,
        ), f"{cross_mask.shape} != {(bs, q_len, kv_len)}"
        block_mask = None
        if block_mask:

            def patch_mask(b, h, q_idx, kv_idx):
                return cross_mask[b, q_idx, kv_idx]

            block_mask = create_block_mask(
                patch_mask,
                B=bs,
                H=None,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                _compile=True,
            )
            return block_mask
        else:
            return torch.where(cross_mask, torch.tensor(0.0), torch.tensor(float("-inf"))).unsqueeze(
                1
            )  # [bs, 1, q_len, kv_len]


def process_patch_lengths(patch_lengths: torch.Tensor, max_patch_length: int) -> torch.Tensor:
    if max_patch_length is None:
        return patch_lengths

    batch_size = patch_lengths.size(0)
    split_all = []
    max_len = 0

    for seq in patch_lengths:
        splits = []
        for length in seq[seq > 0]:
            # Split long patches into max_patch_length chunks
            full, rem = divmod(length.item(), max_patch_length)
            splits.extend([max_patch_length] * full + ([rem] if rem else []))
        split_all.append(splits)
        max_len = max(max_len, len(splits))

    # Pad sequences to the maximum length
    padded = torch.zeros((batch_size, max_len), dtype=patch_lengths.dtype, device=patch_lengths.device)
    for i, splits in enumerate(split_all):
        if splits:
            padded[i, :len(splits)] = torch.tensor(splits, dtype=patch_lengths.dtype, device=patch_lengths.device)

    # Trim trailing columns that are all zeros
    last_non_zero = (padded != 0).flip(1).int().argmax(1).min()
    if last_non_zero < padded.shape[1]:
        padded = padded[:, :padded.shape[1] - last_non_zero]

    return padded


def create_causal_mask_for_blt(
    seqlen: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """
    Creates a causal mask for BLT local encoder.
    """
    min_value = torch.finfo(dtype).min
    mask = torch.full(
        (batch_size, 1, seqlen, seqlen),  # Note: using seqlen, not total_seqlen
        min_value,
        dtype=dtype,
        device=device,
    )
    
    if sliding_window is not None:
        # Create local causal mask with sliding window
        for i in range(seqlen):
            start_idx = max(0, i - sliding_window + 1)
            mask[:, :, i, start_idx:i + 1] = 0
    else:
        # Create full causal mask
        mask = torch.triu(mask, diagonal=0)
        mask = mask.masked_fill(mask == 0, min_value)
    
    return mask


class BLTRotaryEmbedding(nn.Module):
    def __init__(self, config: BLTConfig, device=None):
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
    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config #TODO: rm this

        self.dropout = config.dropout
        
        self.layers = nn.ModuleList([BLTTransformerLayer(config.dim_local_encoder, config.n_heads_local_encoder, config) for _ in range(config.n_layers_local_encoder)])

        self.rotary_emb = BLTRotaryEmbedding(config=config)
        self.pos_embeddings = None

        self.token_embedding_projection = (
            nn.Linear(config.encoder_dim_token_emb, config.dim_local_encoder, bias=False)
            if config.encoder_dim_token_emb is not None and config.encoder_dim_token_emb != config.dim_local_encoder
            else None
        )

        self.patch_embedding_projection = self._create_patch_projection(config)

        self.tok_embeddings = nn.Embedding(config.vocab_size + config.pm_size, config.dim_local_encoder)

        # Initialize cross attention layers only if cross attention is enabled
        self.cross_attn_layers = None
        if getattr(config, "cross_attn_encoder", False) and config.cross_attn_nheads is not None:
            self.cross_attn_layers = torch.nn.ModuleList()
            layers_to_add = config.n_layers_local_encoder if config.cross_attn_all_layers_encoder else 1
            for _ in range(layers_to_add):
                self.cross_attn_layers.append(
                    BLTCrossAttention(
                        dim=config.dim_local_encoder,
                        head_dim=config.dim_local_encoder // config.cross_attn_nheads,
                        n_heads=config.cross_attn_nheads,
                        n_kv_heads=config.cross_attn_nheads,
                        norm_eps=config.norm_eps,
                    )
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        input_embeds: Optional[torch.Tensor] = None,
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union["BlockMask", torch.Tensor, str]] = None,
        cross_mask: Optional[torch.Tensor] = None,
        num_patches: Optional[int] = None,
        patch_ids: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        """ """
        bs, seqlen = input_ids.shape
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length, _ = input_embeds.shape

        if mask is None:
            attention_mask = create_causal_mask_for_blt(
                seqlen=seq_length,
                batch_size=batch_size,
                device=input_embeds.device,
                dtype=input_embeds.dtype,
                sliding_window=self.config.sliding_window,
            )

        h = input_embeds

        h_residual = input_embeds
        h = nn.functional.dropout(h, p=self.dropout, training=self.training) 

        position_ids = torch.arange(input_ids.shape[1], device=input_embeds.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(h, position_ids)  

        h = F.dropout(h, p=self.config.dropout, training=self.training)

        for idx, layer in enumerate(self.layers):
            h = layer(h, position_embeddings=position_embeddings, attention_mask=None)

            if getattr(self.config, "cross_attn_encoder", None) and (idx == len(self.layers) - 1 or self.config.cross_attn_all_layers_encoder):
                if self.config.cross_attn_init_by_pooling and patch_embeds is None:
                    patch_embeds = self.patch_reduce(h, num_patches, "amax", patch_ids)
                    if self.patch_embedding_projection is not None:
                        patch_embeds = self.patch_embedding_projection(patch_embeds)
                        patch_embeds = patch_embeds.reshape(bs, patch_embeds.shape[1] * getattr(self.config, "cross_attn_k", 1), self.config.dim_local_encoder)

                layer_idx = idx if self.config.cross_attn_all_layers_encoder else 0
                patch_embeds_cross = self.cross_attn_layers[layer_idx](
                    x=patch_embeds,
                    kv=h,
                    mask=cross_mask,
                )
                patch_embeds = patch_embeds + patch_embeds_cross

        h_residual = patch_embeds if getattr(self.config, "cross_attn_encoder", None) else None
        return (h, h_residual), cache
    
    def _create_patch_projection(self, config):
        dimension_mismatch = config.encoder_dim_patch_emb is not None and config.encoder_dim_patch_emb != config.dim_local_encoder

        cross_attn_conditions = (config.cross_attn_encoder and config.cross_attn_init_by_pooling) or (
            config.cross_attn_decoder and config.cross_attn_init_by_pooling
        )

        if not (dimension_mismatch or cross_attn_conditions):
            return None

        output_dim = config.encoder_dim_token_emb * (getattr(config, "cross_attn_k", None) or 1)

        return nn.Linear(
            in_features=config.encoder_dim_patch_emb,
            out_features=output_dim,
            bias=False,
        )

    def embed_tokens(self, tokens, embeds):
        if embeds is not None:
            assert self.config.encoder_hash_byte_group_size is not None, "Not expecting embeddings to be passed."
            return embeds
        else:
            return self.tok_embeddings(tokens)

    def patch_reduce(self, h, max_num_patches, reduction, patch_ids):
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
        bs, seq_len, emb_dim = h.shape

        patch_ids = patch_ids.unsqueeze(-1).expand(-1, -1, h.shape[-1])

        reduced_embs = torch.zeros((bs, max_num_patches, emb_dim), dtype=h.dtype, device=h.device)
        reduced_embs = reduced_embs.scatter_reduce(
            src=h,
            dim=1,
            index=patch_ids,
            reduce=reduction,
            include_self=False,
        )
        reduced_embs = reduced_embs[:, :max_num_patches, :]

        return reduced_embs


class BLTLocalDecoder(nn.Module):
    def __init__(self, config: BLTConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([BLTTransformerLayer(config.dim_local_decoder, config.n_heads_local_decoder, config) for _ in range(config.n_layers_local_decoder)])

        decoder_config = config
        decoder_config.head_dim = config.dim_local_decoder // config.n_heads_local_decoder
        decoder_config.max_position_embeddings = config.max_encoder_seq_length or config.max_seqlen

        self.rotary_emb = BLTRotaryEmbedding(config=decoder_config)

        self.pos_embeddings = None

        self.token_embedding_projection = (
            nn.Linear(config.decoder_dim_token_emb, config.dim_local_decoder, bias=False)
            if config.decoder_dim_token_emb is not None and config.decoder_dim_token_emb != config.dim_local_decoder
            else None
        )

        self.patch_embedding_projection = self._create_patch_projection(config)

        self.norm = RMSNorm(config.dim_local_decoder, eps=config.norm_eps)

        # Initialize cross attention layers only if cross attention is enabled
        self.cross_attn_layers = None
        if getattr(config, "cross_attn_decoder", False) and config.cross_attn_nheads is not None:
            self.cross_attn_layers = torch.nn.ModuleList()
            layers_to_add = config.n_layers_local_decoder if config.cross_attn_all_layers_decoder else 1
            for _ in range(layers_to_add):
                self.cross_attn_layers.append(
                    BLTCrossAttention(
                        dim=config.dim_local_decoder,
                        head_dim=config.dim_local_decoder // config.cross_attn_nheads,
                        n_heads=config.cross_attn_nheads,
                        n_kv_heads=config.cross_attn_nheads,
                        norm_eps=config.norm_eps,
                    )
                )

        self.output = nn.Linear(config.dim_local_decoder, config.vocab_size, bias=False)

    def _create_patch_projection(self, config):
        dimension_mismatch = config.dim_global is not None and config.dim_global != config.dim_local_decoder

        # Check cross attention conditions
        cross_attn_conditions = (config.cross_attn_encoder and config.cross_attn_init_by_pooling) or (
            config.cross_attn_decoder and config.cross_attn_init_by_pooling
        )

        if not (dimension_mismatch or cross_attn_conditions):
            return None

        output_dim = config.decoder_dim_token_emb * (getattr(config, "cross_attn_k", None) or 1)

        return nn.Linear(
            in_features=config.dim_global,
            out_features=output_dim,
            bias=False,
        )

    def apply_embedding(self, tokens, embeds):
        if embeds is not None:
            return embeds
        else:
            return self.tok_embeddings(tokens)

    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor],
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union["BlockMask", torch.Tensor, str]] = None,
        cross_mask: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        bs, seqlen = tokens.shape
        batch_size, seq_length, _ = embeds.shape

        assert embeds is not None, "Embeddings must be provided"

        if mask is None:
            attention_mask = create_causal_mask_for_blt(
                seqlen=seq_length,
                batch_size=batch_size,
                device=embeds.device,
                dtype=embeds.dtype,
                sliding_window=self.config.sliding_window,
            )

        h = embeds

        if self.patch_embedding_projection is not None:
            assert patch_embeds is not None, "Patch embeddings must be passed."
            patch_embeds = self.patch_embedding_projection(patch_embeds)
            if getattr(self.config, "cross_attn_k", None) is not None:
                patch_embeds = patch_embeds.reshape(bs, patch_embeds.shape[1] * self.config.cross_attn_k, self.config.dim_local_decoder)

        if patch_embeds is not None and not getattr(self.config, "cross_attn_decoder", None):
            h = h + patch_embeds

        position_ids = torch.arange(tokens.shape[1], device=embeds.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(h, position_ids)  

        h = F.dropout(h, p=self.config.dropout, training=self.training)
        for i, layer in enumerate(self.layers):
            if getattr(self.config, "cross_attn_decoder", None) and (i == 0 or self.config.cross_attn_all_layers_decoder):
                # Use cross attention to extract info from patch_embeds into h
                h_cross = self.cross_attn_layers[i](
                    x=h,
                    kv=patch_embeds,
                    mask=cross_mask,
                )
                h = h + h_cross

            h = layer(h, position_embeddings=position_embeddings, attention_mask=None)

        h_preds = self.norm(h)
        h_preds = F.dropout(h_preds, p=self.config.dropout, training=self.training)
        h_preds = self.output(h_preds)
        h_preds = h_preds.float()
        return h_preds, cache


class BLTCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.cross_attn_norm_q = nn.RMSNorm(dim, eps=norm_eps)
        self.cross_attn_norm_kv = RMSNorm(dim, eps=norm_eps)

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            n_kv_heads * head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        mask: Optional[Union[BlockMask, str]] = None,
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, _ = x.shape
        _, slen_kv, _ = kv.shape
        x_norm = self.cross_attn_norm_q(x)
        kv = self.cross_attn_norm_kv(kv)

        xq = self.wq(x_norm)
        xk = self.wk(kv)
        xv = self.wv(kv)

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)

        # assert mask is None or isinstance(mask, BlockMask)
        xq, xk, xv = (e.transpose(1, 2) for e in (xq, xk, xv))
        # output = flex_attention_comp(xq, xk, xv, block_mask=mask)
        is_causal = (mask == "causal") if isinstance(mask, str) else False
        mask = mask if isinstance(mask, torch.Tensor) else None
        mask = mask.to(dtype=xq.dtype).to(xq.device)
        output = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            is_causal=is_causal,
            attn_mask=mask,
        )
        output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        output = self.wo(output.reshape(output_shape))

        return x + output


class BLTGlobalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList()
        old = config.n_kv_heads 
        config.n_kv_heads = config.n_kv_heads_global
        for _ in range(config.n_layers_global):
            self.layers.append(BLTTransformerLayer(self.config.dim_global, self.config.n_heads_global, config))
        config.n_kv_heads = old

        global_config = config
        global_config.head_dim = config.dim_global // config.n_heads_global

        self.rotary_emb = BLTRotaryEmbedding(config=global_config)

        self.token_embedding_projection = None
        if config.global_dim_patch_emb is not None and config.global_dim_patch_emb != config.dim_global:
            self.token_embedding_projection = nn.Linear(
                config.global_dim_patch_emb,
                config.dim_global,
                bias=False,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, torch.Tensor, str]] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        batch_size, seq_length, _ = input_embeds.shape

        h = input_embeds

        if self.token_embedding_projection is not None and h.shape[-1] != self.config.dim_global:
            h = self.token_embedding_projection(h)

        h = F.dropout(h, p=self.config.dropout, training=self.training)

        position_ids = torch.arange(input_ids.shape[1], device=input_embeds.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(h, position_ids)  

        for i, layer in enumerate(self.layers):
            h = layer(h, position_embeddings=position_embeddings, attention_mask=None)

        return h, cache


def compute_hash_embeddings(
    local_encoder_tokens: torch.Tensor,
    local_encoder,
    encoder_hash_tok_embedding: nn.ModuleList,
    encoder_hash_byte_group_nb_functions: int,
    encoder_hash_byte_group_size: list,
    encoder_hash_byte_group_vocab: int,
) -> torch.Tensor:
    """
    Compute embeddings using hash token embeddings.

    Args:
        local_encoder_tokens: Input tokens tensor
        local_encoder: Encoder object with tok_embeddings method
        encoder_hash_tok_embedding: ModuleList of hash token embeddings
        encoder_hash_byte_group_nb_functions: Number of hash functions
        encoder_hash_byte_group_size: List of byte group sizes
        encoder_hash_byte_group_vocab: Vocabulary size for hash embeddings

    Returns:
        torch.Tensor: Combined embeddings
    """
    if encoder_hash_tok_embedding is None:
        return None

    local_encoder_embeds = local_encoder.tok_embeddings(local_encoder_tokens)

    i = 0
    for func_nb in range(encoder_hash_byte_group_nb_functions):
        for byte_group_size in encoder_hash_byte_group_size:
            hash_ids = byte_group_hash_function(
                local_encoder_tokens,
                byte_group_size,
                hash_func_nb=func_nb,
                max_hash=encoder_hash_byte_group_vocab,
            )
            hash_tok_embedding = encoder_hash_tok_embedding[i]
            local_encoder_embeds = local_encoder_embeds + hash_tok_embedding(hash_ids)
            i += 1

    assert i == len(encoder_hash_tok_embedding)
    return local_encoder_embeds


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
            
        # elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
        #     nn.init.ones_(module.weight)
        #     if module.bias is not None:
        #         nn.init.zeros_(module.bias)
             
        elif isinstance(module, BLTModel):
            if module.encoder_hash_tok_embedding is not None:
                emb_std = module.config.dim_local_encoder ** (-0.5)
                for emb in module.encoder_hash_tok_embedding:
                    emb._custom_std = emb_std
                    
        elif isinstance(module, BLTLocalEncoder):
            if module.token_embedding_projection is not None:
                module.token_embedding_projection._custom_std = module.config.dim_local_encoder ** (-0.5)
                
            if module.patch_embedding_projection is not None:
                module.patch_embedding_projection._custom_std = module.config.encoder_dim_patch_emb ** (-0.5)
                
        elif isinstance(module, BLTLocalDecoder):
            if module.token_embedding_projection is not None:
                module.token_embedding_projection._custom_std = module.config.dim_local_decoder ** (-0.5)
                
            if module.patch_embedding_projection is not None:
                module.patch_embedding_projection._custom_std = module.config.dim_global ** (-0.5)
                
        elif isinstance(module, BLTGlobalTransformer):
            if module.token_embedding_projection is not None:
                module.token_embedding_projection._custom_std = module.dim_token_emb ** (-0.5)
                
        elif isinstance(module, BLTPatcher):
            emb_std = module.config.dim ** (-0.5)
            module.tok_embeddings._custom_std = emb_std
            module.output._custom_std = emb_std


class BLTModel(BLTPreTrainedModel):
    def __init__(self, config: BLTConfig):
        super().__init__(config)

        self.config = config
        self.local_encoder = BLTLocalEncoder(config)
        self.global_transformer = BLTGlobalTransformer(config)
        self.local_decoder = BLTLocalDecoder(config)

        self.encoder_hash_tok_embedding = init_hash_embeddings(
            config,
            local_encoder_dim=config.dim_local_encoder,
            encoder_hash_byte_group_size=config.encoder_hash_byte_group_size,
        )

        if config.patch_in_forward:
            self.patcher = BLTPatcher(config)
            self.patcher.eval()
            for param in self.patcher.parameters():
                param.requires_grad = False
        else:
            self.patcher = None

    def forward(
        self,
        tokens: torch.Tensor,
        patch_lengths: Optional[torch.Tensor] = None,
    ):
        # NOTE: ngram_ids parameter removed since frequency-based n-gram embeddings
        # are no longer used in the final BLT model

        bs, N = tokens.shape  # Batch size and sequence length

        local_encoder_tokens, local_decoder_tokens = tokens, tokens

        # Patching
        if patch_lengths is None:
            # assert (
            #     getattr(self.config, "patch_in_forward", None) is not None and self.config.patch_in_forward
            # ), "Patch in forward not enabled and no patch_lengths passed."

            # PATCHER MODEL DEFINED
            if self.config.patching_mode == PatchingModeEnum.entropy:
                _, patch_lengths, _ = self.patcher(
                    local_encoder_tokens,
                    patch_size=self.config.patch_size,
                    include_next_token=True,
                    threshold=self.config.patching_threshold,
                    max_patch_length=self.config.max_patch_length,
                    patching_batch_size=self.config.patching_batch_size,
                    device=self.config.patching_device,
                )
            else:
                # self.config.patching_mode == PatchingModeEnum.byte
                bs, seq_len = local_encoder_tokens.shape
                seq_len_next_tok = seq_len + 1  # include_next_token=True
                patch_lengths = torch.ones(
                    (bs, seq_len_next_tok), dtype=local_encoder_tokens.dtype, device=local_encoder_tokens.device
                )

                patch_lengths = process_patch_lengths(patch_lengths, self.config.max_patch_length)

        #assert torch.min(patch_lengths) >= 0
        # Generate patch IDs from patch_lengths
        patch_ids = self._patch_ids_from_lengths(patch_lengths, local_encoder_tokens.shape[-1])
        # assert torch.max(patch_ids) + 1 <= torch.max((patch_lengths != 0).sum(dim=-1)), (
        #     f"{torch.max(patch_ids) + 1} > {torch.max((patch_lengths != 0).sum(dim=-1))}"
        # )

        cross_attn_mask_enc = None
        # Cross-attention encoder
        if self.config.cross_attn_encoder:
            cross_attn_mask_enc = cross_attn_mask(
                patch_ids,
                patch_lengths,
                N,
                patches_as_queries=True,
                cross_attn_k=self.config.cross_attn_k,
                window=self.config.cross_attn_window_encoder,
                block_mask=self.config.cross_attn_use_flex_attention,
            )

        # Hashing and embedding
        local_encoder_embeds = compute_hash_embeddings(
            local_encoder_tokens=local_encoder_tokens,
            local_encoder=self.local_encoder,
            encoder_hash_tok_embedding=self.encoder_hash_tok_embedding,
            encoder_hash_byte_group_nb_functions=self.config.encoder_hash_byte_group_nb_functions,
            encoder_hash_byte_group_size=self.config.encoder_hash_byte_group_size,
            encoder_hash_byte_group_vocab=self.config.encoder_hash_byte_group_vocab,
        )

        # NOTE: Frequency-based n-gram embeddings removed as per paper
        # The final BLT model uses only hash-based n-gram embeddings

        # Local encoder
        (h_encoder, h_cross), cache_encoder = self.local_encoder(
            input_ids=local_encoder_tokens,
            input_embeds=local_encoder_embeds,
            patch_embeds=None,
            cross_mask=cross_attn_mask_enc,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
        )

        # Downsampling
        h = h_cross.view(bs, patch_lengths.shape[1], -1)

        # Global transformer
        global_tokens = tokens.new(h.shape[0], h.shape[1]).fill_(self.config.boe_id)
        rows, cols = torch.where(local_encoder_tokens == self.config.eos_token_id)
        eos_patch_ids = patch_ids[rows, cols]
        global_tokens[rows, eos_patch_ids] = self.config.eos_token_id

        h, _ = self.global_transformer(
            input_embeds=h,
            input_ids=global_tokens,
        )

        # Unpatching

        dec_embeds = h_encoder

        # Decoder uses patches 1,2,3,... (skipping patch 0 which contains BOE tokens), so we need to map decoder positions to the remaining patches.
        decoder_patch_ids = self._patch_ids_from_lengths(patch_lengths[:, 1:], local_decoder_tokens.shape[-1])
        # assert torch.max(decoder_patch_ids) + 1 <= h.shape[1], f"{torch.max(decoder_patch_ids) + 1} > {h.shape[1]}"
        # assert decoder_patch_ids.shape[1] == dec_embeds.shape[1], (
        #     f"{decoder_patch_ids.shape[1]} != {dec_embeds.shape[1]}"
        # )

        # Cross-attention decoder
        if not self.config.cross_attn_decoder:
            h = torch.gather(h, 1, decoder_patch_ids.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
            cross_attn_mask_dec = None
            # assert local_decoder_tokens.shape == h.shape[:-1]
        else:
            cross_attn_mask_dec = cross_attn_mask(
                decoder_patch_ids,
                patch_lengths,
                N,
                patches_as_queries=False,
                cross_attn_k=self.config.cross_attn_k,
                window=self.config.cross_attn_window_decoder,
                block_mask=self.config.cross_attn_use_flex_attention,
            )

        # Local decoder
        output, _ = self.local_decoder(
            embeds=dec_embeds,
            patch_embeds=h,
            tokens=local_decoder_tokens,
            cross_mask=cross_attn_mask_dec,
        )
        return output
    
    def _patch_ids_from_lengths(self, patch_lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Convert patch lengths to patch IDs for each token position.
        For each token position in the sequence, determines which patch it belongs to.

        Args:
            patch_lengths: [batch_size, num_patches] - length of each patch
            seq_len: total sequence length

        Returns:
            patch_ids: [batch_size, seq_len] - patch index for each token position

        Example:
            patch_lengths = [[3, 2, 4, 1]]  # 4 patches of lengths 3,2,4,1
            seq_len = 10
            Returns: [[0, 0, 0, 1, 1, 2, 2, 2, 2, 3]]
                     # pos 0-2→patch 0, pos 3-4→patch 1, pos 5-8→patch 2, pos 9→patch 3
        """
        batch_size, num_patches = patch_lengths.shape

        # Create patch start positions: [0, 3, 5, 9] for the example above
        patch_starts = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=patch_lengths.dtype, device=patch_lengths.device),
                patch_lengths.cumsum(dim=-1)[:, :-1],  # cumsum without the final total
            ],
            dim=-1,
        )

        # For each token position, find which patch it belongs to
        # by finding the rightmost patch start that's <= the position
        token_positions = torch.arange(seq_len, device=patch_lengths.device)  # [0, 1, 2, ..., seq_len-1]

        # Broadcasting: patch_starts[batch, patch] <= token_positions[position]
        # Result: [batch, seq_len, num_patches] where result[b,t,p] = True if patch p starts <= position t
        position_ge_patch_start = patch_starts.unsqueeze(1) <= token_positions.unsqueeze(0).unsqueeze(-1)

        # Count how many patch starts are <= each position, then subtract 1 to get patch index
        patch_ids = position_ge_patch_start.sum(dim=-1) - 1

        return patch_ids
    

class BLTPatcher(BLTPreTrainedModel):
    def __init__(self, config):
        # Store reference to main config for accessing non-patcher settings
        self.main_config = config
        
        # Initialize with patcher config directly
        super().__init__(config.patcher_config)

        # Initialize rotary embeddings with patcher config
        self.rotary_emb = BLTRotaryEmbedding(config=self.config)

        self.layers = nn.ModuleList()
        for _ in range(self.config.n_layers):
            self.layers.append(BLTTransformerLayer(self.config.dim, self.config.n_heads, self.config))

        #assert self.config.vocab_size > 0

        self.tok_embeddings = torch.nn.Embedding(self.config.vocab_size, self.config.dim)

        self.norm = RMSNorm(self.config.dim, eps=self.config.norm_eps)

        self.output = nn.Linear(
            self.config.dim,
            self.config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        token_values: torch.Tensor,
        patch_size: Optional[int] = None,
        include_next_token: bool = True,
        threshold: Optional[float] = None,
        max_patch_length: Optional[int] = None,
        patching_batch_size: int = 1,
        device: Optional[str] = None,
    ):

        # Handle chunked processing for entropy calculation
        entropies = []
        preds = []
        max_length = self.config.max_seqlen
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
            bsz, seqlen = split.shape
            input_embeds = self.tok_embeddings(split)

            hidden_states = input_embeds


            batch_size, seq_length, _ = input_embeds.shape

            position_ids = torch.arange(split.shape[1], device=input_embeds.device).unsqueeze(0).expand(batch_size, -1)
            
            position_embeddings = self.rotary_emb(hidden_states, position_ids)  # = BLT self.rope
            
            for i, layer in enumerate(self.layers):
                hidden_states = layer(hidden_states, position_embeddings=position_embeddings, attention_mask=None) #, attn_impl=self.config.patcher_attn_impl )

            pred = self.output(self.norm(hidden_states))
            pred = pred.reshape(-1, pred.shape[-1])[: split.numel() - pad_size, :]  # [batch_size * seq_len, vocab]
            preds.append(pred)
            pred_entropies = self.entropy(pred)
            entropies.append(pred_entropies)

        concat_entropies = torch.cat(entropies, dim=0).reshape(token_values.shape)
        concat_preds = torch.cat(preds, dim=0).reshape(token_values.shape[0], -1)

        # Always compute patch lengths from concatenated entropies
        bs, seq_len = token_values.shape
        seq_len_next_tok = seq_len + 1 if include_next_token else seq_len

        # Find patch start IDs based on entropy
        if patch_size is not None:
            patch_start_ids = self.find_entropy_patch_start_ids(
                concat_entropies,
                patch_size,
                include_next_token=include_next_token,
                threshold=threshold
            )
            patch_lengths = self.patch_lengths_from_start_ids(patch_start_ids, seq_len_next_tok)
        else:
            # Default to byte-level patching
            patch_lengths = torch.ones((bs, seq_len_next_tok), dtype=token_values.dtype, device=token_values.device)

        patch_lengths = process_patch_lengths(patch_lengths, max_patch_length)
        return concat_entropies, patch_lengths, concat_preds


    @staticmethod
    def entropy(scores):
        """
        scores: [bs, seq_len, vocab]
        returns [bs, seq_len]

        Computes the entropy for each token in the batch.
        Note: uses natural log.
        """
        log_probs = F.log_softmax(scores, dim=-1)
        probs = torch.exp(log_probs)
        p_log_p = log_probs * probs
        entropy = -p_log_p.sum(dim=-1)
        return entropy

    @staticmethod
    def patch_start_ids_from_patch_start_mask(patch_start_mask):
        bs, trunc_seq_len = patch_start_mask.shape
        max_patches = patch_start_mask.sum(dim=1).max()
        if max_patches == 0:
            patch_start_ids = torch.full(
                (bs, trunc_seq_len),
                trunc_seq_len,
                dtype=torch.long,
                device=patch_start_mask.device,
            )
        else:
            patch_ids = torch.arange(trunc_seq_len, device=patch_start_mask.device).unsqueeze(0).repeat(bs, 1)
            extra_patch_ids = torch.full(
                (bs, trunc_seq_len),
                trunc_seq_len,
                dtype=torch.long,
                device=patch_start_mask.device,
            )
            all_patch_ids = torch.cat((patch_ids, extra_patch_ids), dim=1)
            patch_start_mask_padded = torch.cat((patch_start_mask, ~patch_start_mask), dim=1)
            patch_start_ids = all_patch_ids[patch_start_mask_padded].reshape(bs, trunc_seq_len)[:, :max_patches]
        return patch_start_ids

    @staticmethod
    def patch_lengths_from_start_ids(patch_start_ids, seq_len):
        """
        Calculate patch lengths from start ids.
        start ids: ex: [0, 1, 7, 7, 7, 7, 7], it has the start ids of the patches (here 0, 1), and then
            the rest are filled to the seq len.
        seq_len: ex: 7 length of the sequence

        returns the patch lengths:
        [1, 6] for the above example.
        """
        last_ids = torch.full_like(patch_start_ids[:, :1], seq_len - 1)
        patch_end_ids = torch.cat((patch_start_ids[:, 1:] - 1, last_ids), dim=1)
        patch_lengths = patch_end_ids - patch_start_ids + 1
        assert torch.all(patch_lengths >= 0), f"{patch_lengths}"
        assert not check_non_zero_after_zero(patch_lengths), f"{patch_lengths}"
        return patch_lengths

    @staticmethod
    def find_entropy_patch_start_ids(
        entropies,
        patch_size=None,
        threshold=None,
        include_next_token=True,
    ):
        """
        Use entropies to find the start ids of each patch.
        Use patch_size or threshold to figure out the total number of patches to allocate.

        When threshold is not None the number of patches is not constant between
        different sequences, but patches can be identified incrementally rather than
        decided globally using the entire sequence.
        """
        bs, seq_len = entropies.shape[:2]

        first_ids = torch.tensor([0, 1], dtype=torch.long, device=entropies.device).unsqueeze(0).repeat(bs, 1)
        preds_truncation_len = first_ids.shape[1]  # remove the first preds because they will be start of patches.
        entropies = entropies[:, 1:]
        if threshold is None:
            num_patches = seq_len // patch_size
            patch_start_ids = entropies.topk(num_patches - 2, dim=1).indices
            patch_start_ids = patch_start_ids.sort(dim=1).values
        else:
            patch_start_mask = entropies > threshold
            if not include_next_token:
                patch_start_mask = patch_start_mask[:, :-1]
            # patch_start_mask[1:] |= tokens[:-1] < OFFSET
            patch_start_ids = BLTPatcher.patch_start_ids_from_patch_start_mask(patch_start_mask)

        patch_start_ids = torch.cat((first_ids, patch_start_ids + preds_truncation_len), dim=1)
        return patch_start_ids

def init_hash_embeddings(
    config,
    local_encoder_dim: int,
    encoder_hash_byte_group_size: list,
):
    """Initialize hash-based token embeddings for the BLT encoder."""
    if config.encoder_hash_byte_group_size is None:
        return None

    embeddings = []
    emb_dim = local_encoder_dim
    encoder_hash_byte_group_vocab = config.encoder_hash_byte_group_vocab

    for _ in range(config.encoder_hash_byte_group_nb_functions):
        for _ in encoder_hash_byte_group_size:
            embeddings.append(
                nn.Embedding(
                    encoder_hash_byte_group_vocab,
                    emb_dim,
                )
            )

    return nn.ModuleList(embeddings)


__all__ = [
    "BLTPreTrainedModel",
    "BLTModel",
    "BLTPatcher",
    "BLTLocalEncoder",
    "BLTLocalDecoder",
    "BLTGlobalTransformer",
]