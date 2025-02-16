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
from functools import cached_property
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
from ...generation import GenerationMixin
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torch_available,
    is_vision_available,
    logging,
    replace_return_docstrings,
    torch_int,
)

from ..chameleon.modeling_chameleon import (
    ChameleonForConditionalGeneration,
    ChameleonImageVocabularyMapping,
    ChameleonModel,
    ChameleonPreTrainedModel,
    ChameleonVQVAE,
    ChameleonVQVAEEncoderAttnBlock,
    ChameleonVQVAEEncoderResnetBlock,
    ChameleonVQVAEVectorQuantizer,
    ChameleonVQVAEEncoder,
    ChameleonVQVAEEncoderConvDownsample
)
from .configuration_janus import JanusVisionConfig,JanusConfig,JanusTextConfig, JanusVQVAEConfig
from ..vit.modeling_vit import ViTPatchEmbeddings
from ..dinov2_with_registers.modeling_dinov2_with_registers  import Dinov2WithRegistersLayerScale, Dinov2WithRegistersDropPath
from ..siglip.modeling_siglip import SiglipEncoder, SiglipVisionTransformer, SiglipVisionModel, SiglipMultiheadAttentionPoolingHead
from ..llama.modeling_llama import LlamaModel, LlamaForCausalLM

if is_flash_attn_2_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward

if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint

logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "JanusConfig"
_CHECKPOINT_FOR_DOC = ""


class JanusVisionPatchEmbeddings(ViTPatchEmbeddings):
    pass

# ToDO: Is interpolate pos embeddings required for this model as of now passing?
@dataclass
class JanusVQVAEOutput:
    pass

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

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype), interpolate_pos_encoding)

        # Add CLS and Register token embeddings
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

        # Permute the dims of qkv vector and unravel it into query, key, value states.
        query_states, key_states, value_states = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query_states = self.query_norm(query_states)
        key_states = self.key_norm(key_states)

        # Is it a bug or deliberate change?
        query_states = query_states * self.scale

        attn_weights = torch.matmul(query_states, key_states.transpose(2,3))

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
                "JanusVisionModel is using JanusVisionSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
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


class JanusVisionEncoder(SiglipEncoder):
    def __init__(self,config:JanusVisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([JanusVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


class JanusPreTrainedModel():
     """An abstract class to load pretrained weigths"""
     pass

class JanusVisionTransformer(SiglipVisionTransformer,nn.Module):
    def __init__(self, config: JanusVisionConfig):
        nn.Module.__init__()
        self.config = config
        self.post_layernorm = nn.LayerNorm(config.hidden_size)
        self.embeddings = JanusVisionEmbeddings(config)
        self.encoder = JanusVisionEncoder(config)
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = JanusVisionAttentionPoolLatent(config)

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

class JanusVQVAEVectorQuantizer(ChameleonVQVAEVectorQuantizer):
    def __init__(self, config):
        super().__init__(config)
        self.quant_state_dims = [config.resolution // 2 ** (len(config.channel_multiplier) - 1)] * 2

    def get_codebook_entry(self, image_tokens: torch.LongTensor) -> torch.FloatTensor:
        batch_size = image_tokens.shape[0]
        emb_dim: int = self.embedding.weight.shape[-1]
        # get quantized latent vectors
        hidden_state_quant = self.embedding(image_tokens)

        # reshape back to match original input shape
        hidden_state_quant = hidden_state_quant.view((batch_size, *self.quant_state_dims, emb_dim))
        hidden_state_quant = hidden_state_quant.permute(0, 3, 1, 2).contiguous()

        return hidden_state_quant


class JanusVQVAEResnetBlock(ChameleonVQVAEEncoderResnetBlock):
    pass


class JanusVQVAEAttnBlock(ChameleonVQVAEEncoderAttnBlock):
    pass

class JanusVQVAEConvDownsample(ChameleonVQVAEEncoderConvDownsample):
    pass


class JanusVQVAEConvUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states):
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states

class JanusVQVAEEncoder(ChameleonVQVAEEncoder,nn.Module):
    def __init__(self, config):
        nn.Module.__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        base_channels = config.base_channels
        in_channels = config.in_channels
        double_latent = config.double_latent
        latent_channels = config.latent_channels
        channel_multiplier = config.channel_multiplier

        self.conv_in = torch.nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        in_channel_multiplier = (1,) + tuple(channel_multiplier)
        self.in_channel_multiplier = in_channel_multiplier
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = base_channels * in_channel_multiplier[i_level]
            block_out = base_channels * channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    JanusVQVAEResnetBlock(
                        config=config,
                        in_channels=block_in,
                        out_channels=block_out,
                    )
                )
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn.append(JanusVQVAEAttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = JanusVQVAEConvDownsample(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = JanusVQVAEResnetBlock(
            config=config,
            in_channels=block_in,
            out_channels=block_in,
        )
        self.mid.attn_1 = JanusVQVAEAttnBlock(block_in) if config.attn_type == "vanilla" else nn.Identity()
        self.mid.block_2 = JanusVQVAEResnetBlock(
            config=config,
            in_channels=block_in,
            out_channels=block_in,
        )

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * latent_channels if double_latent else latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

class JanusVQVAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        base_channels = config.base_channels
        resolution = config.resolution
        latent_channels = config.latent_channels
        out_channels = config.out_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = base_channels * config.channel_multiplier[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, latent_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(latent_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = JanusVQVAEResnetBlock(
            config=config,
            in_channels=block_in,
            out_channels=block_in,
        )
        self.mid.attn_1 = JanusVQVAEAttnBlock(block_in) if config.attn_type == "vanilla" else nn.Identity()
        self.mid.block_2 = JanusVQVAEResnetBlock(
            config=config,
            in_channels=block_in,
            out_channels=block_in,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = base_channels * config.channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    JanusVQVAEResnetBlock(
                        config=config,
                        in_channels=block_in,
                        out_channels=block_out,
                    )
                )
                block_in = block_out
                if i_level == self.num_resolutions-1:
                    attn.append(JanusVQVAEAttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = JanusVQVAEConvUpsample(block_in)
            self.up.append(up)

        # end
        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_state: torch.FloatTensor) -> torch.FloatTensor:
        hidden_state = self.conv_in(hidden_state)

        # middle
        hidden_state = self.mid.block_1(hidden_state)
        hidden_state = self.mid.attn_1(hidden_state)
        hidden_state = self.mid.block_2(hidden_state)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                hidden_state = self.up[i_level].block[i_block](hidden_state)
                if len(self.up[i_level].attn) > 0:
                    hidden_state = self.up[i_level].attn[i_block](hidden_state)
            if i_level != 0:
                hidden_state = self.up[i_level].upsample(hidden_state)

        hidden_state = self.norm_out(hidden_state)
        hidden_state *= torch.sigmoid(hidden_state)
        hidden_state = self.conv_out(hidden_state)
        return hidden_state


class JanusPreTrainedModel(ChameleonPreTrainedModel):
    pass


class JanusVQVAE(ChameleonVQVAE):
    main_input_name = "pixel_values"

    def __init__(self, config: JanusVQVAEConfig):
        super().__init__(config)
        self.decoder = JanusVQVAEDecoder(config)
        self.gradient_checkpointing = False
        # self.post_init()

    def decode(self, image_tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Decodes quantized token IDs into pixel values.
        Args:
            image_tokens (`torch.LongTensor` of shape `(batch_size, quantize.quant_state_dims[0] * quantize.quant_state_dims[1])`):
                Batch of token IDs.
        Returns:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                Pixel values decoded from the token IDs.
        """
        if image_tokens.shape[1] != self.quantize.quant_state_dims[0] * self.quantize.quant_state_dims[1]:
            raise ValueError(
                f"Expected `image_tokens` to have shape `(batch_size, {self.quantize.quant_state_dims[0] * self.quantize.quant_state_dims[1]})`, "
                f"but got shape `{image_tokens.shape}`."
            )
        codebook_entry = self.quantize.get_codebook_entry(image_tokens)
        hidden_states = self.post_quant_conv(codebook_entry)
        pixel_values = self.decoder(hidden_states)
        return pixel_values

    def forward(
        self, pixel_values: torch.FloatTensor, return_dict: bool = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Encodes pixel values into quantized tokens and decodes them back.
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
                The tensors corresponding to the input images.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        Returns:
            decoded_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                Reconstructed pixel values after encoding and decoding the input.
            emb_loss (`torch.FloatTensor`):
                Embedding loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = pixel_values.shape[0]
        quant, emb_loss, indices = self.encode(pixel_values)
        decoded_pixel_values = self.decode(indices.view(batch_size, -1))
        if not return_dict:
            return (decoded_pixel_values, emb_loss)
        return JanusVQVAEOutput(decoded_pixel_values, emb_loss)


class JanusImageVocabularyMapping(ChameleonImageVocabularyMapping):
    @cached_property
    def bpe2img_mapping_tensor(self):
        mapping = torch.zeros(max(self.bpe2img.keys()) + 1, dtype=torch.int)
        for k, v in self.bpe2img.items():
            mapping[k] = v
        return mapping

class JanusTextModel(LlamaModel):
    pass
class JanusTextForCausalLM(LlamaForCausalLM, JanusPreTrainedModel, GenerationMixin):
    config_class = JanusTextConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = JanusTextModel(config)

class JanusForConditionalGeneration(JanusPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["language_model.lm_head.weight"]
    _supports_static_cache = False  # `get_image_tokens()`, called when `pixel_values` is passed, is not compilable.

    def __init__(self, config):
        super().__init__(config)
        self.vision_model = JanusVisionTransformer(config.vision_config)
        self.language_model = JanusTextForCausalLM(config.text_config)
        self.aligner = JanusVisionAlignerMLP(config.vision_config)
        self.vqmodel = JanusVQVAE(config.vq_config)
        # self.vocabulary_mapping = JanusImageVocabularyMapping(config.vocabulary_map)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self,input_ids):
        return self.language_model.get_input_embeddings()(input_ids)

    def get_image_embeddings(self,pixel_values):
        image_embeds = self.vision_model(pixel_values)
        image_embeds = self.aligner(image_embeds.last_hidden_state)
        return image_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None:
            image_embeds = self.get_image_embeddings(pixel_values)
            special_image_mask = input_ids == 100581
            image_embeds_flat = image_embeds.reshape(-1, 2048)
            special_image_mask = special_image_mask.unsqueeze(-1).expand(-1, -1, 2048)

            text_embeds = self.get_input_embeddings(input_ids)
            image_embeds = image_embeds.to(text_embeds.device, text_embeds.dtype)
            inputs_embeds = text_embeds.masked_scatter(special_image_mask, image_embeds_flat)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
        )

        return outputs

