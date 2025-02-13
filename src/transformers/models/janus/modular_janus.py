# coding=utf-8
# Copyright 2024 Google AI and The HuggingFace Team. All rights reserved.
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
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.init import _calculate_fan_in_and_fan_out

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
    torch_int,
)

from .configuration_janus import JanusVisionConfig,JanusConfig,JanusTextConfig
from ..vit.modeling_vit import ViTPatchEmbeddings
from ..dinov2_with_registers.modeling_dinov2_with_registers  import Dinov2WithRegistersLayerScale, Dinov2WithRegistersDropPath
from ..siglip.modeling_siglip import SiglipEncoder, SiglipVisionTransformer, SiglipVisionModel, SiglipMultiheadAttentionPoolingHead
from ..llama.modeling_llama import LlamaForCausalLM

if is_flash_attn_2_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "JanusConfig"
_CHECKPOINT_FOR_DOC = ""


class JanusVisionPatchEmbeddings(ViTPatchEmbeddings):
    pass

# ToDO: Is interpolate pos embeddings required for this model?

class JanusVisionEmbeddings(nn.Module):
    def __init__(self, config:JanusVisionConfig):
        super().__init__()

        self.use_special_tokens = config.use_special_tokens
        if self.use_special_tokens:
            self.cls_token = nn.Parameter(torch.rand(1,1,config.hidden_size))
            self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))

        # Currently using hidden_drop_rate instead of positional_dropout_rate, is it necessary?
        self.dropout = nn.Dropout(config.hidden_dropout_rate)
        self.patch_embeddings = JanusVisionPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches

        num_prefix_tokens = config.num_register_tokens + 1
        pos_embed_len = self.num_patches + num_prefix_tokens if self.use_special_tokens else self.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, pos_embed_len, config.hidden_size) * 0.02)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        # Add CLS and Register token embeddings if .
        special_token_embeddings = []
        if self.use_special_tokens:
            cls_token_embeddings = self.cls_token.expand((batch_size, -1,-1))
            special_token_embeddings.append(cls_token_embeddings)

            if self.register_tokens.shape[1]:
                register_token_embeddings = self.register_tokens.expand((batch_size, -1,-1))
                special_token_embeddings.append(register_token_embeddings)

        if self.use_special_tokens:
            embeddings = embeddings + self.position_embeddings
            embeddings = torch.cat(special_token_embeddings+[embeddings], dim=1)
        else:
            # embeddings = torch.cat(special_token_embeddings+[embeddings], dim=1)
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings

# Todo: introduce compatiability for cache
class JanusVisionAttention(nn.Module):
    """Attention Class for Janus Vision Encoder """
    def __init__(self, config: JanusVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        proj_dropout = config.projection_dropout
        qk_norm = config.use_qk_norm

        # Split the weights manually and checkif getting correct output or not
        self.qkv = nn.Linear(self.embed_dim, 3*self.embed_dim, bias=config.qkv_bias)
        self.projection_layer = nn.Linear(self.embed_dim, self.embed_dim)
        self.projection_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

        self.query_norm = nn.LayerNorm(self.embed_dim) if qk_norm else nn.Identity()
        self.key_norm = nn.LayerNorm(self.embed_dim) if qk_norm else nn.Identity()

    def forward(self,hidden_states:torch.Tensor,attention_mask: Optional[torch.Tensor] = None, output_attentions: Optional[torch.Tensor] = None):
        batch_size , seq_len, _ = hidden_states.size()

        # Batched computation of query, key, value states.
        qkv = self.qkv(hidden_states).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        #  Permute the dims of qkv vector and unravel it into query, key, value states.
        query_states, key_states, value_states = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query_states = self.query_norm(query_states)
        key_states = self.key_norm(key_states)

        # Is it a bug or deliberate change?
        query_states = query_states * self.scale
        # b,numhead,seqlen,hed dim -> b, num head, head dim, seq len
        attn_weights = torch.matmul(query_states, key_states.transpose(2,3))

        print(attn_weights.shape)
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_weights.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (batch_size,1, seq_len, self.head_dim):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, seq_len, self.head_dim)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1 ,dtype=torch.float32).to(query_states.dtype)
        # Only apply attention dropout during training.
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.reshape( batch_size, seq_len, self.embed_dim)

        output = self.projection_layer(attn_output)
        output = self.projection_dropout(output)

        outputs = (output, attn_weights) if output_attentions else (output, None)
        return outputs

class JanusVisionFlashAttention2(JanusVisionAttention):
    """
    JanusVision flash attention module. This module inherits from `JanusVisionAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    is_causal = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    # Adapted from transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        batch_size , seq_len, _ = hidden_states.size()

        # Batched computation of query, key, value states.
        qkv = self.qkv(hidden_states).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states, key_states, value_states = qkv.permute.unbind(2)
        query_states = self.query_norm(query_states)
        key_states = self.key_norm(key_states)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (Idefics2VisionRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
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
            seq_len,
            dropout=dropout_rate,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim).contiguous()
        output = self.projection_layer(attn_output)
        output = self.projection_dropout(output)

        # In `Flash Attenition` we don't return Attention weights, hence return None.
        return output, None

class JanusVisionSdpaAttention(JanusVisionAttention):
    """
    Janusvision attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `JanusVisionAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    is_causal = False

    # Adapted from transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "SiglipModel is using SiglipSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
        
        batch_size , seq_len, _ = hidden_states.size()

        qkv = self.qkv(hidden_states).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        query_states, key_states, value_states = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query_states = self.query_norm(query_states)
        key_states = self.key_norm(key_states)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if self.is_causal and seq_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        output = self.projection_layer(attn_output)
        output = self.projection_dropout(output)
        return output, None
    
JANUS_VISION_ATTENTION_CLASSES = {
    "eager": JanusVisionAttention,
    "sdpa": JanusVisionSdpaAttention,
    "flash_attention_2": JanusVisionFlashAttention2,
}


class JanusVisionLayerScale(Dinov2WithRegistersLayerScale):
    pass
class JanusVisionDropPath(Dinov2WithRegistersDropPath):
    pass

class JanusVisionMLP(nn.Module):
    def __init__(self, config:JanusVisionConfig):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act] # Gelu act
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_rate)
        self.dropout2 = nn.Dropout(config.hidden_dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        return hidden_states

class JanusVisionEncoderLayer(nn.Module):
    def __init__(self,config: JanusVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = JANUS_VISION_ATTENTION_CLASSES[config._attn_implementation](config=config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.layer_scale1 = JanusVisionLayerScale(config) if config.layerscale_value else nn.Identity()
        self.layer_scale2 = JanusVisionLayerScale(config) if config.layerscale_value else nn.Identity()
        self.drop_path1 = (
            JanusVisionDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        )
        self.drop_path2 = (
            JanusVisionDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        )
        self.mlp = JanusVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # Pre-Norm before attention .
        norm_hidden_states = self.layer_norm1(hidden_states)
        attn_output, attn_weights = self.self_attn(
            norm_hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )

        scaled_attn_output = self.layer_scale1(attn_output)
        dropped_attn_output = self.drop_path1(scaled_attn_output)
        hidden_states = hidden_states + dropped_attn_output

        norm_hidden_states = self.layer_norm2(hidden_states)

        mlp_output = self.mlp(norm_hidden_states)

        scaled_mlp_output = self.layer_scale2(mlp_output)
        dropped_mlp_output = self.drop_path2(scaled_mlp_output)
        hidden_states = hidden_states + dropped_mlp_output

        return (hidden_states, attn_weights if output_attentions else None)

class JanusVisionAttentionPoolLatent(nn.Module):
    def __init__(self, config: JanusVisionConfig):
        super().__init__()

        self.latent_len = getattr(config, "latent_len", 1)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.mlp_ratio = getattr(config, "mlp_ratio", 4.0)
        self.scale = self.head_dim ** -0.5

        # Learnable latent query (probe)
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, self.hidden_size))

        # Linear layers for QKV projection
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.kv_proj = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Normalization & MLP
        self.norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.mlp = JanusVisionMLP(config)

        self.proj_drop = nn.Dropout(getattr(config, "dropout", 0.0))

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, _ = hidden_states.shape

        # Expand learnable latent tokens for batch
        q_latent = self.latent.expand(batch_size, -1, -1)  # (B, latent_len, hidden_size)

        # Compute Q projection from latent tokens
        query_states = self.q_proj(q_latent)  # (B, latent_len, hidden_size)

        # Compute combined KV projection
        kv = self.kv_proj(hidden_states)
        key_states, value_states = kv.view(batch_size, seq_len, 2, self.num_heads, self.head_dim).unbind(2)

        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        query_states = query_states.view(batch_size, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)  # (B, num_heads, latent_len, head_dim)

        # Validate shape
        if attn_output.size() != (batch_size, self.num_heads, self.latent_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, self.latent_len, self.head_dim)},"
                f" but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, self.latent_len, self.hidden_size)

        output = self.proj(attn_output)
        output = self.proj_drop(output)

        output = output + self.mlp(self.norm(output))

        return output[:, 0]


# class JanusVisionEncoder(SiglipEncoder):
#     def __init__(self,config:JanusVisionConfig):
#         nn.Module.__init__(config)
#         self.config = config
#         self.layers = nn.ModuleList([JanusVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
#         self.gradient_checkpointing = False


class JanusVisionEncoder(nn.Module):
    def __init__(self,config:JanusVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([JanusVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
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

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class JanusPreTrainedModel():
     """An abstract class to load pretrained weigths"""
     pass

# Copied from siglip vision transformer
# class JanusVisionTransformer(SiglipVisionTransformer,nn.Module):
#     def __init__(self, config: JanusVisionConfig):
#         nn.Module.__init__(config)
#         self.config = config
#         self.post_layenorm = nn.LayerNorm(config.hidden_size)
#         self.embeddings = JanusVisionEmbeddings(config,use_special_tokens=False)
#         self.encoder = JanusVisionEncoder(config)
#         self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
#         if self.use_head:
#             self.head = JanusVisionAttentionPoolLatent(config)


class JanusVisionTransformer(nn.Module):
    def __init__(self, config: JanusVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = JanusVisionEmbeddings(config)
        self.encoder = JanusVisionEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = JanusVisionAttentionPoolLatent(config)
            # Won't be using as a standalone classifier head hence no num classes

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.layernorm(last_hidden_state)
        pooler_output = self.head(hidden_states)


        if not return_dict:
            return (last_hidden_state) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class JanusVisionAlignerMLP(nn.Module):
    def __init__(self, config:JanusVisionConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.hidden_size, config.aligner_projection_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(config.aligner_projection_size, config.aligner_projection_size) for _ in range(1, config.num_aligner_hidden_states)
        ])
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        for layer in self.hidden_layers:
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = layer(hidden_states)
        return hidden_states

class JanusTextForCausalLM(LlamaForCausalLM):
    pass

# class JanusVLM(nn.Module):
#     def __init__(self,config:JanusConfig):
#         super().__init__()

#         self.vision_model = JanusVisionTransformer(config.vision_config)
#         self.aligner = JanusVisionAlignerMLP(config.vision_config)
#         self.language_model = JanusTextForCausalLM(config.text_config)

#     def forward(self,
#         input_ids: torch.LongTensor,
#         pixel_values: torch.FloatTensor,
#         images_seq_mask: torch.LongTensor,
#         **kwargs,
#     ):
#         image_embeds = self.vision_model(pixel_values)
#         image_embeds = self.aligner(image_embeds.last_hidden_state)

#         text_embeds = self.language_model.get_input_embeddings()(input_ids)
#         text_embeds[images_seq_mask] = image_embeds

#         return text_embeds


