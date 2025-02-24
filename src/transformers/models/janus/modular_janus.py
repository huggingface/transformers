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

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image
from torch import nn
from tqdm import tqdm

from transformers.models.blip.image_processing_blip import BlipImageProcessor

from ...activations import ACT2FN
from ...cache_utils import Cache, StaticCache
from ...generation import ClassifierFreeGuidanceLogitsProcessor, GenerationMixin, GenerationMode
from ...image_transforms import (
    resize,
)
from ...image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from ...modeling_outputs import (
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torch_available,
    is_torchdynamo_compiling,
    logging,
    torch_int,
)
from ..auto import AutoModel
from ..chameleon.modeling_chameleon import (
    ChameleonVQVAE,
    ChameleonVQVAEEncoder,
    ChameleonVQVAEEncoderAttnBlock,
    ChameleonVQVAEEncoderConvDownsample,
    ChameleonVQVAEEncoderResnetBlock,
    ChameleonVQVAEVectorQuantizer,
)
from ..dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersDropPath,
    Dinov2WithRegistersLayerScale,
)
from ..siglip.modeling_siglip import SiglipEncoder, SiglipVisionTransformer
from ..vit.modeling_vit import ViTPatchEmbeddings
from .configuration_janus import JanusConfig, JanusVisionConfig, JanusVQVAEConfig


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
    """
    Base class for Janus VQ-VAE mode model outputs.
    Args:
        decoded_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            Reconstructed pixel values after encoding and decoding the input.
        emb_loss (`torch.FloatTensor`):
            Embedding loss.
    """

    decoded_pixel_values: Optional[torch.FloatTensor] = None
    emb_loss: torch.FloatTensor = None


class JanusVisionEmbeddings(nn.Module):
    def __init__(self, config: JanusVisionConfig):
        super().__init__()

        self.use_special_tokens = config.use_special_tokens
        if self.use_special_tokens:
            self.cls_token = nn.Parameter(torch.rand(1, 1, config.hidden_size))
            self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))

        # Currently using hidden_drop_rate instead of positional_dropout_rate, is it necessary?
        self.dropout = nn.Dropout(config.hidden_dropout_rate)
        self.patch_embeddings = JanusVisionPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches

        num_prefix_tokens = config.num_register_tokens + 1
        num_positions = self.num_patches + num_prefix_tokens if self.use_special_tokens else self.num_patches
        self.position_embeddings = nn.Embedding(num_positions, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(num_positions).expand((1, -1)), persistent=False)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing and no class embeddings.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        if interpolate_pos_encoding:
            pos_embeds = self.interpolate_pos_encoding(embeddings, height, width)
        else:
            pos_embeds = self.position_embeddings(self.position_ids)

        # Add CLS and Register token embeddings.
        special_token_embeddings = []
        if self.use_special_tokens:
            cls_token_embeddings = self.cls_token.expand((batch_size, -1, -1))
            special_token_embeddings.append(cls_token_embeddings)

            if self.register_tokens.shape[1]:
                register_token_embeddings = self.register_tokens.expand((batch_size, -1, -1))
                special_token_embeddings.append(register_token_embeddings)

        if self.use_special_tokens:
            embeddings = embeddings + pos_embeds
            embeddings = torch.cat(special_token_embeddings + [embeddings], dim=1)
        else:
            embeddings = embeddings + pos_embeds

        embeddings = self.dropout(embeddings)
        return embeddings


class JanusVisionAttention(nn.Module):
    """Attention Class for Janus Vision Encoder"""

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
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.projection_layer = nn.Linear(self.embed_dim, self.embed_dim)
        self.projection_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

        self.query_norm = nn.LayerNorm(self.embed_dim) if qk_norm else nn.Identity()
        self.key_norm = nn.LayerNorm(self.embed_dim) if qk_norm else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len, _ = hidden_states.size()

        # Batched computation of query, key, value states.
        qkv = self.qkv(hidden_states).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        # Permute the dims of qkv vector and unravel it into query, key, value states.
        query_states, key_states, value_states = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query_states = self.query_norm(query_states)
        key_states = self.key_norm(key_states)

        # Is it a bug or deliberate change?
        query_states = query_states * self.scale

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_weights.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, seq_len, self.head_dim):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, seq_len, self.head_dim)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Only apply attention dropout during training.
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

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

        batch_size, seq_len, _ = hidden_states.size()

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

        batch_size, seq_len, _ = hidden_states.size()

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
    def __init__(self, config: JanusVisionConfig):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]  # Gelu act
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
    def __init__(self, config: JanusVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        # self.attn = JANUS_VISION_ATTENTION_CLASSES[config._attn_implementation](config=config)
        self.attn = JanusVisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.layer_scale1 = JanusVisionLayerScale(config) if config.layerscale_value else nn.Identity()
        self.layer_scale2 = JanusVisionLayerScale(config) if config.layerscale_value else nn.Identity()
        self.drop_path1 = JanusVisionDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.drop_path2 = JanusVisionDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
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
        attn_output, attn_weights = self.attn(
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
        self.scale = self.head_dim**-0.5

        # Learnable latent query (probe)
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, self.hidden_size))

        # Linear layers for QKV projection
        self.q = nn.Linear(self.hidden_size, self.hidden_size)
        self.kv = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.projection_layer = nn.Linear(self.hidden_size, self.hidden_size)

        # Normalization & MLP
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.mlp = JanusVisionMLP(config)

        self.proj_drop = nn.Dropout(getattr(config, "dropout", 0.0))

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, _ = hidden_states.shape

        # Expand learnable latent tokens for batch
        q_latent = self.latent.expand(batch_size, -1, -1)  # (B, latent_len, hidden_size)

        # Compute Q projection from latent tokens
        query_states = self.q(q_latent)  # (B, latent_len, hidden_size)

        # Compute combined KV projection
        kv = self.kv(hidden_states)
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

        output = self.projection_layer(attn_output)
        output = self.proj_drop(output)

        output = output + self.mlp(self.layer_norm(output))

        return output[:, 0]


class JanusVisionEncoder(SiglipEncoder):
    def __init__(self, config: JanusVisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([JanusVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])


class JanusVisionTransformer(SiglipVisionTransformer, nn.Module):
    config_class = JanusVisionConfig
    _supports_sdpa = False

    def __init__(self, config: JanusVisionConfig):
        nn.Module.__init__()
        self.config = config
        self.embeddings = JanusVisionEmbeddings(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)
        self.encoder = JanusVisionEncoder(config)
        self.use_head = True if not hasattr(config, "use_vision_head") else config.use_vision_head
        if self.use_head:
            self.head = JanusVisionAttentionPoolLatent(config)


class JanusVisionAlignerMLP(nn.Module):
    def __init__(self, config: JanusVisionConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.hidden_size, config.aligner_projection_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(config.aligner_projection_size, config.aligner_projection_size) for _ in range(1, config.depth)]
        )
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        for layer in self.hidden_layers:
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = layer(hidden_states)
        return hidden_states


class JanusVQVAEVectorQuantizer(ChameleonVQVAEVectorQuantizer):
    def __init__(self, config: JanusVQVAEConfig):
        super().__init__(config)
        self.quant_state_dims = [config.num_patches] * 2

    def get_codebook_entry(self, image_tokens: torch.LongTensor) -> torch.FloatTensor:
        batch_size = image_tokens.shape[0]
        emb_dim: int = self.embedding.weight.shape[-1]

        # get quantized latent vectors
        hidden_state_quant = self.embedding(image_tokens)
        hidden_state_quant = F.normalize(hidden_state_quant, p=2, dim=-1)

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


class JanusVQVAEEncoder(ChameleonVQVAEEncoder, nn.Module):
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
        latent_channels = config.latent_channels
        out_channels = config.out_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = base_channels * config.channel_multiplier[self.num_resolutions - 1]

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
                if i_level == self.num_resolutions - 1:
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
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks + 1):
                hidden_state = self.up[i_level].block[i_block](hidden_state)
                if len(self.up[i_level].attn) > 0:
                    hidden_state = self.up[i_level].attn[i_block](hidden_state)
            if i_level != self.num_resolutions - 1:
                hidden_state = self.up[i_level].upsample(hidden_state)

        hidden_state = self.norm_out(hidden_state)
        hidden_state *= torch.sigmoid(hidden_state)
        hidden_state = self.conv_out(hidden_state)
        return hidden_state


class JanusPreTrainedModel(PreTrainedModel):
    config_class = JanusConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # _no_split_modules = None # Should we pass Llama Decoder Layer?
    _skip_keys_device_placement = ["past_key_values", "causal_mask"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_param_buffer_assignment = False

    def _init_weights(self, module):
        std = self.config.vision_config.initializer_range
        if isinstance(module, JanusVQVAE):
            module.apply(module._init_weights)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class JanusVQVAE(ChameleonVQVAE):
    def __init__(self, config: JanusVQVAEConfig):
        super().__init__(config)
        self.decoder = JanusVQVAEDecoder(config)
        self.gradient_checkpointing = False

        # Initilaize the
        self.post_init()

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


class JanusVQVAEAligner(nn.Module):
    def __init__(self, config: JanusVQVAEConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.embed_dim, config.aligner_projection_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(config.aligner_projection_size, config.aligner_projection_size) for _ in range(1, config.depth)]
        )
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        for layer in self.hidden_layers:
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = layer(hidden_states)
        return hidden_states


class JanusVQVAEHead(nn.Module):
    def __init__(self, config: JanusVQVAEConfig):
        super().__init__()
        self.proj_out = nn.Linear(config.image_token_embed_size, config.aligner_projection_size)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.vision_head = nn.Linear(config.aligner_projection_size, config.num_embeddings)

    def forward(self, hidden_states: torch.Tensor) -> torch.tensor:
        hidden_states = self.proj_out(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.vision_head(hidden_states)
        return hidden_states


class JanusModel(JanusPreTrainedModel):
    def __init__(self, config: JanusConfig):
        super().__init__(config)
        self.config = config
        self.vision_model = JanusVisionTransformer(config.vision_config)
        self.aligner = JanusVisionAlignerMLP(config.vision_config)

        self.vqmodel = JanusVQVAE(config.vq_config)
        self.gen_embed = nn.Embedding(config.vq_config.num_embeddings, config.vq_config.embed_dim)
        self.gen_aligner = JanusVQVAEAligner(config.vq_config)
        self.gen_head = JanusVQVAEHead(config.vq_config)

        self.language_model = AutoModel.from_config(config=config.text_config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_embeddings(self, pixel_values):
        image_embeds = self.vision_model(pixel_values)
        image_embeds = self.aligner(image_embeds.last_hidden_state)
        return image_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
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

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )
        # Generate input embeds which will be used while decoding
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_embeddings(pixel_values)
            image_attention_mask = input_ids == 100581

            # Flatten the image embeddings and mask. Refactor it to use input embeds info better
            embed_dim = inputs_embeds.shape[-1]
            image_embeds = image_embeds.reshape(-1, embed_dim)
            image_attention_mask = image_attention_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_attention_mask, image_embeds)

        output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            labels=labels,
            **kwargs,
        )

        return output if return_dict else output.to_tuple()


class JanusForConditionalGeneration(JanusPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["model.language_model.embed_tokens.weight", "lm_head.weight"]
    _supports_static_cache = False  # `get_image_tokens()`, called when `pixel_values` is passed, is not compilable.

    def __init__(self, config: JanusConfig):
        super().__init__(config)
        self.config = config
        self.model = JanusModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # Initialize weights and apply final processing.
        self.post_init()

    def get_input_embeddings(self):
        return self.model.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.language_model.set_input_embeddings(value)

    def prepare_emeddings_for_image_generation(self, inputs):
        hidden_state = self.model.gen_embed(inputs)
        hidden_state = self.model.gen_aligner(hidden_state)
        return hidden_state

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
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
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
        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

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

    # ToDo: Make it compatible with super class (add contine gen using embeds)
    def prepare_inputs_for_generation(
        self,
        input_ids,
        pixel_values=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        # (we can't check exception 3 while compiling)
        # Excpetion 4: If input_embeds are passed then slice it through `cache_position`, to keep only the unprocessed tokens and
        # generate the first token for each sequence. Later use the generated Input ids for continuation.
        if past_key_values is not None:
            if inputs_embeds is not None and input_ids.shape[1] == 0:  # Exception 4
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
            elif (
                inputs_embeds is not None  # Exception 1
                or (is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1])  # Exception 3
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if cache_position[0] != 0:
            pixel_values = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "cache_position": cache_position,
            }
        )
        return model_inputs

    @torch.no_grad
    def generate(self, input_ids: torch.Tensor = None, **kwargs):

        generation_mode = kwargs.pop("generation_mode", None)

        if generation_mode is None:
            logger.info("Generation mode argument is not passed. Defaulting to `Text`generation.")
            generation_mode = "text"

        # Perform usual auto-regressive text generation.
        if generation_mode == "text":
            return super().generate(input_ids=input_ids, **kwargs)

        generation_config = kwargs.pop("generation_config",None)
        if generation_config is None:
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs

        # Fix me: Better way to pass generation config params rather than hardcoding.
        generation_config.temperature = 1
        generation_config.pad_token_id = 100002
        generation_config.guidance_scale = 5

        generation_mode = generation_config.get_generation_mode()
        if generation_mode not in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            raise ValueError(
                "Got incompatible mode for generation, should be one of greedy or sampling. "
                "Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`."
            )

        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # If image is passed then perform custom pre and post processing on generated tokens.
        batch_size, seq_len = input_ids.shape

        input_tokens = input_ids.repeat(2, 1)  # Now (batch_size * 2, seq_len)

        # Set the second half of tokens to pad_token_id. These would be used
        # for generating unconditional logits
        input_tokens[batch_size:, 1:-1] = generation_config.pad_token_id

        # Generate input_embeddings from the given prompt.
        inputs_embeds = self.get_input_embeddings()(input_tokens) # (B,seq_len,embed_dim)

        # Placeholder to store generated tokens
        generated_tokens = torch.zeros((batch_size, self.config.vision_config.num_image_tokens),dtype=input_tokens.dtype)
        outputs = None

        logit_processor = ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale)

        # Loop through the number of image tokens that need to be generated
        for i in tqdm(range(self.config.vision_config.num_image_tokens)):
            # Pass the input embeddings through the language model
            outputs = self.model.language_model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=outputs.past_key_values if i != 0 else None,
            )

            # Extract the last hidden state (corresponding to the most recent token)
            hidden_state = outputs.last_hidden_state[:, -1, :]

            # Generate scores using the generation head.
            scores = self.model.gen_head(hidden_state)

            # Apply logit processor for controlled generation.
            logits = logit_processor(input_ids, scores)
            probs = torch.softmax(logits / generation_config.temperature, dim=-1)

            # Sample the next token from the probability distribution.
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token

            # Prepare embeddings for image generation.
            next_token = next_token
            next_token = torch.cat([next_token, next_token], dim=1).view(-1)

            img_embeds = self.prepare_emeddings_for_image_generation(next_token)

            inputs_embeds = img_embeds.unsqueeze(dim=1)

        # Decode generated tokens using Janus VQ model.
        decoded_image = self.model.vqmodel.decode(generated_tokens)

        # Convert tensor to numpy array and then normalize, reshape the decoded image.
        decoded_image = decoded_image.to(torch.float32).detach().numpy().transpose(0, 2, 3, 1)
        decoded_image = np.clip((decoded_image + 1) / 2 * 255, 0, 255)
        decoded_image = decoded_image.astype(np.uint8)

        return decoded_image


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class JanusImageProcessor(BlipImageProcessor):
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        min_size: int = 14,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.min_size = min_size
        if image_mean is None:
            self.background_color = (127, 127, 127)
        else:
            self.background_color = tuple([int(x * 255) for x in image_mean])

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to dynamically calculated size.

        Args:
            image (`np.ndarray`):
                Image to resize.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BICUBIC`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        height, width, _ = image.shape
        max_size = max(height, width)

        if isinstance(size, dict):
            if "height" not in size or "width" not in size:
                raise ValueError(
                    f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}"
                )
            image_size = size["height"]  # Assign height as image_size assuming height=width
        else:
            image_size = size

        output_size = [
            max(int(height / max_size * image_size), self.min_size),
            max(int(width / max_size * image_size), self.min_size),
        ]

        image = resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            return_numpy=False,
            **kwargs,
        )
        # expand and pad the images
        image = expand2square(image, self.background_color)
        image = to_numpy_array(image)
        return image


__all__ = [
    "JanusImageProcessor",
    "JanusPreTrainedModel",
    "JanusForConditionalGeneration",
    "JanusModel",
    "JanusVQVAE",
]
