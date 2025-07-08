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
"""BLT modular model, inheriting from Mllama where appropriate."""

from typing import Callable, List, Optional, Tuple, Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F

from ...cache_utils import Cache
from ...activations import ACT2FN
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...modeling_outputs import CausalLMOutputWithPast
from ...generation.utils import GenerationMixin
from ...utils import logging, is_torch_flex_attn_available
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

# Import configuration classes
from .configuration_blt import (
    BLTConfig,
    BLTLocalEncoderConfig,
    BLTLocalDecoderConfig,
    BLTGlobalTransformerConfig,
    BLTPatcherConfig,
)

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
    from ...integrations.flex_attention import make_flex_block_causal_mask

# Import from mllama for inheritance
from ..mllama.modeling_mllama import (
    MllamaTextMLP,
    MllamaTextRMSNorm,
    MllamaRotaryEmbedding,
    MllamaTextCrossAttention,
    MllamaSelfAttentionDecoderLayer,
    MllamaPreTrainedModel,
    eager_attention_forward,
    repeat_kv,
    apply_rotary_pos_emb as mllama_apply_rotary_pos_emb,
)

# Import other utility functions and classes from original BLT
from .modeling_blt import (
    PatchingModeEnum,
    byte_group_hash_function,
    rolling_polynomial_hash,
    init_hash_embeddings,
    compute_hash_embeddings,
    _prepare_patch_cross_attention_mask,
    process_patch_lengths,
    apply_rotary_pos_emb,
)

logger = logging.get_logger(__name__)


# ==============================================================================
# INHERITED COMPONENTS (minimal changes from Mllama)
# ==============================================================================

class BLTMLP(MllamaTextMLP):
    pass


class BLTRMSNorm(MllamaTextRMSNorm):
    pass


class BLTRotaryEmbedding(MllamaRotaryEmbedding):
    pass


# ==============================================================================
# INHERITED BUT CUSTOMIZED COMPONENTS
# ==============================================================================

class BLTPreTrainedModel(MllamaPreTrainedModel):
    """BLT PreTrainedModel inheriting from Mllama but with BLT-specific init."""
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
                emb_std = module.config.encoder_config.hidden_size ** (-0.5)
                for emb in module.encoder_hash_tok_embedding:
                    emb._custom_std = emb_std
                    
        elif isinstance(module, BLTLocalEncoder):
            if module.patch_embedding_projection is not None:
                module.patch_embedding_projection._custom_std = module.config.hidden_size ** (-0.5)
                
        elif isinstance(module, BLTLocalDecoder):
            if module.patch_embedding_projection is not None:
                module.patch_embedding_projection._custom_std = module.config.hidden_size ** (-0.5)
                
        elif isinstance(module, BLTPatcher):
            emb_std = module.config.hidden_size ** (-0.5)
            module.embed_tokens._custom_std = emb_std
            module.lm_head._custom_std = emb_std
            
        elif isinstance(module, BLTForCausalLM):
            if module.lm_head is not None:
                module.lm_head._custom_std = module.config.decoder_config.hidden_size ** (-0.5)


class BLTSelfAttention(nn.Module):
    """BLT Self Attention that could inherit from Mllama but has some BLT-specific patterns."""
    
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


class BLTTransformerLayer(MllamaSelfAttentionDecoderLayer):
    pass


# ==============================================================================
# BLT-SPECIFIC COMPONENTS (no Mllama equivalent)  
# ==============================================================================

class BLTLocalEncoder(nn.Module):
    def __init__(self, config: BLTLocalEncoderConfig):
        super().__init__()
    
        self.config = config
        
        self.layers = nn.ModuleList([BLTTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

        self.rotary_emb = BLTRotaryEmbedding(config=config)

        self.patch_embedding_projection = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size * config.cross_attn_k,
            bias=False,
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.cross_attn_layers = torch.nn.ModuleList()
        layers_to_add = config.num_hidden_layers if config.cross_attn_all_layers else 1
        for layer_idx in range(layers_to_add):
            self.cross_attn_layers.append(
                BLTCrossAttention(config=config, layer_idx=layer_idx, hidden_size=config.hidden_size)
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

        hidden_states = F.dropout(input_embeds, p=self.config.dropout, training=self.training) 

        position_ids = torch.arange(input_ids.shape[1], device=input_embeds.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)  

        hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)

        for idx, layer in enumerate(self.layers):
            layer_outputs = layer(hidden_states, position_embeddings=position_embeddings, attention_mask=None)
            hidden_states = layer_outputs[0]

            if idx == len(self.layers) - 1 or self.config.cross_attn_all_layers:
                patch_embeds = self.patch_reduce(hidden_states, num_patches, "amax", patch_ids)
                patch_embeds = self.patch_embedding_projection(patch_embeds)
                patch_embeds = patch_embeds.reshape(batch_size, patch_embeds.shape[1] * self.config.cross_attn_k, self.config.hidden_size)

                layer_idx = idx if self.config.cross_attn_all_layers else 0
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
        self.config = config
        self.cross_attn_decoder = True #config.cross_attn_decoder #TODO: maybe remove

        self.layers = nn.ModuleList([BLTTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

        self.rotary_emb = BLTRotaryEmbedding(config=config)

        self.patch_embedding_projection = nn.Linear(
            in_features=config.hidden_size_global,
            out_features=config.hidden_size * config.cross_attn_k,
            bias=False,
        )

        self.norm = BLTRMSNorm(config.hidden_size, eps=config.norm_eps)

        self.cross_attn_layers = torch.nn.ModuleList()
        layers_to_add = config.num_hidden_layers if config.cross_attn_all_layers else 1
        for layer_idx in range(layers_to_add):
            self.cross_attn_layers.append(
                BLTCrossAttention(config=config, layer_idx=layer_idx, hidden_size=config.hidden_size)
            )

        # self.lm_head = nn.Linear(
        #     config.hidden_size,
        #     config.vocab_size,
        #     bias=False,
        # )


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
        patch_embeds = patch_embeds.reshape(batch_size, patch_embeds.shape[1] * self.config.cross_attn_k, self.config.hidden_size)

        if patch_embeds is not None and not self.cross_attn_decoder:
            hidden_states = hidden_states + patch_embeds

        position_ids = torch.arange(tokens.shape[1], device=embeds.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)  

        hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)
        for i, layer in enumerate(self.layers):
            if i == 0 or self.config.cross_attn_all_layers:
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

        logits = self.norm(hidden_states)
      #  logits = self.lm_head(logits)
        return logits, cache


class BLTCrossAttention(nn.Module):
    """Cross-attention module for BLT, following transformers style"""

    def __init__(self, config: BLTConfig, layer_idx: int, hidden_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # Use provided hidden_size or fallback to encoder dimension
        self.hidden_size = hidden_size or config.encoder_config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads  # Assuming same for cross attention
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
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
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if full_text_row_masked_out_mask is not None:
            attn_output = full_text_row_masked_out_mask[:, 0] * attn_output

        attn_output = attn_output + hidden_states

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class BLTGlobalTransformer(nn.Module):
    def __init__(self, config: BLTGlobalTransformerConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
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

        hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)

        position_ids = torch.arange(seq_len, device=input_embeds.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)  

        for i, layer in enumerate(self.layers):
            layer_outputs = layer(hidden_states, position_embeddings=position_embeddings, attention_mask=None)
            hidden_states = layer_outputs[0]

        return hidden_states, cache


class BLTModel(BLTPreTrainedModel):
    def __init__(self, config: BLTConfig):
        super().__init__(config)
        self.config = config
        self.local_encoder = BLTLocalEncoder(config.encoder_config)
        self.global_transformer = BLTGlobalTransformer(config.global_config)
        self.local_decoder = BLTLocalDecoder(config.decoder_config)
        self.encoder_hash_tok_embedding = init_hash_embeddings(
            config,
            local_encoder_dim=config.encoder_config.hidden_size,
            encoder_hash_byte_group_size=config.encoder_hash_byte_group_size,
        )
        if self.config.patch_in_forward:
            self.patcher = BLTPatcher(config.patcher_config)
            self.patcher.eval()
            for param in self.patcher.parameters():
                param.requires_grad = False
        else:
            self.patcher = None

    def forward(
        self,
        tokens: torch.Tensor,
        patch_lengths: Optional[torch.Tensor] = None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        """
        Args:
            tokens (torch.Tensor): Input token ids.
            patch_lengths (Optional[torch.Tensor]): Patch lengths for patching.
            attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, **kwargs: Ignored, for compatibility.
        Returns:
            torch.Tensor: Final hidden states (as before).
        """
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
                    device=tokens.device,
                )
            else:
                patch_lengths = process_patch_lengths(
                    torch.ones((batch_size, sequence_length + 1), dtype=tokens.dtype, device=tokens.device),
                    self.config.max_patch_length
                )
        patch_ids = self._patch_ids_from_lengths(patch_lengths, sequence_length)
        encoder_embeds = compute_hash_embeddings(
            tokens, self.local_encoder, self.encoder_hash_tok_embedding,
            self.config.encoder_hash_byte_group_nb_functions,
            self.config.encoder_hash_byte_group_size,
            self.config.encoder_hash_byte_group_vocab,
        )
        cross_attn_mask_enc, full_text_row_masked_out_mask_enc = _prepare_patch_cross_attention_mask(
            patch_ids, patch_lengths.shape[1], sequence_length, True, self.config.cross_attn_k, encoder_embeds.dtype
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
            decoder_patch_ids, patch_lengths.shape[1], sequence_length, False, self.config.cross_attn_k, encoder_embeds.dtype
        )
        output, _ = self.local_decoder(
            tokens=tokens,
            embeds=encoder_hidden_states,
            patch_embeds=global_hidden_states,
            mask=None,
            cross_mask=cross_attn_mask_dec,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask_dec,
        )
        if output_hidden_states or output_attentions:
            if return_dict:
                return {"last_hidden_state": output, "hidden_states": None, "attentions": None}
            else:
                return (output, None, None)
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
        
        for layer_idx in range(self.config.num_hidden_layers):
            self.layers.append(BLTTransformerLayer(self.config, layer_idx))


        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)

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
            
            position_embeddings = self.rotary_emb(hidden_states, position_ids) 
            
            for i, layer in enumerate(self.layers):
                layer_outputs = layer(hidden_states, position_embeddings=position_embeddings, attention_mask=None)
                hidden_states = layer_outputs[0]

            logits = self.lm_head(self.norm(hidden_states))
            logits = logits.reshape(-1, logits.shape[-1])[: split.numel() - pad_size, :] 
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


class BLTForCausalLM(BLTPreTrainedModel, GenerationMixin):
    config_class = BLTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BLTTransformerLayer", "BLTLocalEncoder", "BLTLocalDecoder", "BLTGlobalTransformer"]

    def __init__(self, config):
        super().__init__(config)
        self.model = BLTModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.local_encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.local_encoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        """
        Args:
            input_ids (torch.LongTensor): Input token ids.
            attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict, cache_position, **kwargs: Standard transformers arguments.
            labels (torch.LongTensor, optional): Labels for language modeling loss.
        Returns:
            CausalLMOutputWithPast or tuple: Standard transformers output.
        """
        # Route only input_ids to BLTModel (as tokens)
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        if isinstance(hidden_states, dict):
            sequence_output = hidden_states["last_hidden_state"]
        elif isinstance(hidden_states, tuple):
            sequence_output = hidden_states[0]
        else:
            sequence_output = hidden_states
        logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            return output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


__all__ = [
    "BLTPreTrainedModel",
    "BLTModel",
    "BLTPatcher",
    "BLTLocalEncoder",
    "BLTLocalDecoder", 
    "BLTGlobalTransformer",
    "BLTTransformerLayer",
    "BLTForCausalLM",
    "BLTMLP",
    "BLTRMSNorm",
    "BLTRotaryEmbedding",
    "BLTSelfAttention",
    "BLTCrossAttention",
] 