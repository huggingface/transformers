# coding=utf-8
# Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.
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
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from tqdm import tqdm

from transformers.models.blip.image_processing_blip import BlipImageProcessor

from ...activations import ACT2FN
from ...cache_utils import Cache, StaticCache
from ...generation import (
    ClassifierFreeGuidanceLogitsProcessor,
    GenerationMixin,
    GenerationMode,
    LogitsProcessorList,
)
from ...generation.utils import GenerateDecoderOnlyOutput
from ...image_processing_utils import BatchFeature
from ...image_transforms import (
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
)
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torch_available,
    is_torchdynamo_compiling,
    is_vision_available,
    logging,
    replace_return_docstrings,
    torch_int,
)
from ...utils.deprecation import deprecate_kwarg
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
from ..siglip.modeling_siglip import SiglipEncoder, SiglipVisionModel, SiglipVisionTransformer
from ..vit.modeling_vit import ViTPatchEmbeddings
from .configuration_janus import JanusConfig, JanusVisionConfig, JanusVQVAEConfig


if is_flash_attn_2_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward

if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint

if is_vision_available():
    import PIL

logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "JanusConfig"

JANUS_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`JanusConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Janus Model outputting raw hidden-states without any specific head on top.",
    JANUS_START_DOCSTRING,
)
class JanusPreTrainedModel(PreTrainedModel):
    config_class = JanusConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "causal_mask"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_param_buffer_assignment = False

    def _init_weights(self, module):
        std = (
            self.config.vision_config.initializer_range
            if hasattr(self.config, "vision_config")
            else self.config.initializer_range
        )
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



@dataclass
class JanusVQVAEOutput(ModelOutput):
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


@dataclass
class JanusBaseModelOutputWithPast(ModelOutput):
    """
    Base class for Janus model.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size (num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class JanusCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Janus causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size (num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class JanusVisionPatchEmbeddings(ViTPatchEmbeddings):
    pass


class JanusVisionEmbeddings(nn.Module):
    def __init__(self, config: JanusVisionConfig):
        super().__init__()

        self.use_special_tokens = config.use_special_tokens
        if self.use_special_tokens:
            self.cls_token = nn.Parameter(torch.rand(1, 1, config.hidden_size))
            self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.size()

        # Batched computation of query, key, value states.
        qkv = self.qkv(hidden_states).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states, key_states, value_states = qkv.unbind(2)
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
                target_dtype = self.qkv.weight.dtype

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
        self.attn = JANUS_VISION_ATTENTION_CLASSES[config._attn_implementation](config=config)
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
    def __init__(self, config: JanusVisionConfig):
        nn.Module.__init__()
        self.config = config
        self.embeddings = JanusVisionEmbeddings(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = JanusVisionEncoder(config)
        self.use_head = True if not hasattr(config, "use_vision_head") else config.use_vision_head
        if self.use_head:
            self.head = JanusVisionAttentionPoolLatent(config)


class JanusVisionModel(SiglipVisionModel):
    config_class = JanusVisionConfig

    def __init__(self, config: JanusVisionConfig):
        super().__init__(config)
        del self.vision_model
        self.vision_transformer = JanusVisionTransformer(config)
        self.post_init()

    def forward(self, *args, **kwargs):
        return self.vision_transformer(*args, **kwargs)


class JanusVisionAlignerMLP(nn.Module):
    def __init__(self, config: JanusVisionConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.hidden_size, config.projection_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(config.projection_dim, config.projection_dim) for _ in range(1, config.depth)]
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
        # l2 normalization on the last dimension
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


class JanusVQVAE(ChameleonVQVAE):
    """Vision Transformer-based VQ-VAE model for encoding and decoding pixel values."""

    _no_split_modules = [
        "JanusVQVAEAttnBlock",
        "JanusVQVAEResnetBlock",
        "JanusVQVAEVectorQuantizer",
    ]
    main_input_name = "pixel_values"

    def __init__(self, config: JanusVQVAEConfig):
        super().__init__(config)
        self.decoder = JanusVQVAEDecoder(config)
        self.gradient_checkpointing = False

        # Initialize the VQVAE model.
        self.post_init()

    def decode(self, image_tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Decodes quantized token IDs into pixel values.
        Args:
            image_tokens (torch.LongTensor): Batch of token IDs.
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
            emb_loss (`torch.FloatTensor`): Embedding loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = pixel_values.shape[0]
        quant, emb_loss, indices = self.encode(pixel_values)
        decoded_pixel_values = self.decode(indices.view(batch_size, -1))

        if not return_dict:
            return (decoded_pixel_values, emb_loss)
        return JanusVQVAEOutput(decoded_pixel_values, emb_loss)


class JanusVQVAEAlignerMLP(nn.Module):
    def __init__(self, config: JanusVQVAEConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.embed_dim, config.projection_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(config.projection_dim, config.projection_dim) for _ in range(1, config.num_hidden_layers)]
        )
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        for layer in self.hidden_layers:
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = layer(hidden_states)
        return hidden_states


class JanusVQVAEHead(nn.Module):
    """Head used for sampling tokens in image generation, replacing the usual lm head."""

    def __init__(self, config: JanusVQVAEConfig):
        super().__init__()
        self.proj_out = nn.Linear(config.image_token_embed_dim, config.projection_dim)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.vision_head = nn.Linear(config.projection_dim, config.num_embeddings)

    def forward(self, hidden_states: torch.Tensor) -> torch.tensor:
        hidden_states = self.proj_out(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.vision_head(hidden_states)
        return hidden_states


JANUS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`].
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    """The Janus model which consists of a siglip vision backbone, a Llama language model and a VQ model.""",
    JANUS_START_DOCSTRING,
)
class JanusModel(JanusPreTrainedModel):
    def __init__(self, config: JanusConfig):
        super().__init__(config)
        self.config = config
        # This is necessary for backward compatibility, see SiglipModel initialization
        self.vision_model = JanusVisionModel._from_config(config.vision_config)
        self.aligner = JanusVisionAlignerMLP(self.vision_model.config)

        self.vqmodel = JanusVQVAE._from_config(config.vq_config)

        # Below gen_* modules are used for image generation.
        # Embeddings used for image generation, instead of Janus vision embeddings.
        self.gen_embed = nn.Embedding(self.vqmodel.config.num_embeddings, self.vqmodel.config.embed_dim)
        self.gen_aligner = JanusVQVAEAlignerMLP(self.vqmodel.config)
        self.gen_head = JanusVQVAEHead(self.vqmodel.config)

        self.language_model = AutoModel.from_config(config=config.text_config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing.
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_embeddings(self, pixel_values):
        image_embeds = self.vision_model(pixel_values)
        image_embeds = self.aligner(image_embeds.last_hidden_state)
        return image_embeds

    def _prepare_4d_causal_attention_mask_with_cache_position(self, *args, **kwargs):
        return self.language_model._prepare_4d_causal_attention_mask_with_cache_position(*args, **kwargs)


    @add_start_docstrings_to_model_forward(JANUS_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
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

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_embeddings(pixel_values)
            image_attention_mask = input_ids == self.config.image_token_index

            embed_dim = inputs_embeds.shape[-1]
            image_features = image_embeds.reshape(-1, embed_dim)
            image_attention_mask = image_attention_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_attention_mask, image_features)

        lm_output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        output = JanusBaseModelOutputWithPast(
            last_hidden_state=lm_output.last_hidden_state,
            past_key_values=lm_output.past_key_values,
            hidden_states=lm_output.hidden_states,
            attentions=lm_output.attentions,
            image_hidden_states=image_embeds if pixel_values is not None else None,
        )

        return output if return_dict else output.to_tuple()


class JanusForConditionalGeneration(JanusPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["model.language_model.embed_tokens.weight", "lm_head.weight"]
    _supports_static_cache = True

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

    def prepare_embeddings_for_image_generation(self, inputs: torch.Tensor) -> torch.Tensor:
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



    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(JANUS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=JanusCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
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
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:
        """
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

        return JanusCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

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

        # if `inputs_embeds` are passed, only use them in the 1st generation step for every prompt.
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

    def decode_image_tokens(self, image_tokens: torch.Tensor):
        """
        Decodes generated image tokens from language model to continuous pixel values
        with VQGAN module via upsampling.
        Args:
            image_tokens (`torch.LongTensor` of shape `(batch_size, num_of_tokens)`):
                The tensors corresponding to the input images.
        """
        decoded_image = self.model.vqmodel.decode(image_tokens)
        decoded_image = decoded_image.permute(0, 2, 3, 1)
        return decoded_image

    @torch.no_grad
    def generate(
        self,
        inputs: torch.Tensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        **kwargs,
    ):
        # 1. Handle generation config and model kwargs
        generation_config = kwargs.pop("generation_config", self.generation_config)
        generation_config = copy.deepcopy(generation_config)

        # Default to "text" generation if mode isn't provided
        generation_mode = kwargs.pop("generation_mode", "text")
        if generation_mode == "text":
            # Set to prevent running UnbatchedCFG processor.
            generation_config.guidance_scale = None
            logger.info("Generation mode argument is not passed. Setting to default `Text` generation.")
            return super().generate(inputs=inputs, generation_config=generation_config, **kwargs)

        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs

        # Validate generation mode
        if generation_config.get_generation_mode() not in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            raise ValueError(
                "Got incompatible mode for Image Generation, should be one of greedy or sampling. "
                "Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`."
            )

        # Validate the configuration and model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Initialize logit processors
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        # Don't require stopping criteria for image generation.

        # Set `use_cache=True` as we will be using input embeds for generation.
        model_kwargs["use_cache"] = True
        model_kwargs["guidance_scale"] = generation_config.guidance_scale

        # 3. Prepare model inputs
        input_ids, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )

        if len(input_ids.shape) != 2:
            raise ValueError(
                f"Expected input ids as input of shape (batch_size, seq_len), but got {input_ids.shape}"
                "Passing `inputs embeds` is not supported currently."
            )

        # Prepare special tokens which will be used generate internally.
        kwargs_has_attention_mask = attention_mask is not None
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=input_ids.device)

        # 4. Add CFG processor along with user passed logit processor.
        if generation_config.guidance_scale and generation_config.guidance_scale > 1:
            logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
            generation_config.guidance_scale = None  # Reset to prevent processor duplication.

        # 5. Prepare logits processor
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids.shape[1],
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
            device=input_ids.device,
        )

        # 6. Expand inputs for multiple image generations per prompt.
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            expand_size=generation_config.num_return_sequences,
            **model_kwargs,
        )

        # 7. Prepare input and model caches
        batch_size, seq_len = input_ids.shape

        num_image_tokens = self.model.vision_model.config.num_image_tokens

        input_tokens = input_ids.repeat(2, 1)  # Double batch size for conditional/unconditional logits
        attention_mask = model_kwargs.pop("attention_mask", None) 
        attention_mask = attention_mask.repeat(2, 1)

        input_tokens[batch_size:, :].masked_fill_(input_tokens[batch_size:, :] != generation_config.bos_token_id,
                                                  generation_config.pad_token_id)

        inputs_embeds = self.get_input_embeddings()(input_tokens)

        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        if model_kwargs.get("past_key_values", None) is None:
            # Prepare cache if not provided
            model_kwargs["past_key_values"] = self._get_cache(
                cache_implementation=generation_config.cache_implementation or "static",
                # batch_size should account for both conditional/unconditional input; hence multiplied by 2.
                batch_size=batch_size * 2,
                # we should have at least a cache len of seq_len + num_image_tokens
                max_cache_len=max(generation_config.max_length, num_image_tokens + seq_len),
                device=input_ids,
                model_kwargs=model_kwargs,
            )

        # Placeholder for generated tokens
        dtype, device = input_ids.dtype, input_ids.device
        generated_tokens = torch.zeros((batch_size, num_image_tokens), dtype=dtype, device=device)

        # 8. init attention / hidden states / scores tuples
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None


        for i in tqdm(range(num_image_tokens)):
            model_inputs = super().prepare_inputs_for_generation(
                inputs_embeds=inputs_embeds,
                input_ids=input_tokens,
                attention_mask=attention_mask,
                **model_kwargs
            )

            # inputs_embeds.device can change on multi-gpu
            model_inputs["attention_mask"] = model_inputs["attention_mask"].to(inputs_embeds.device)
            model_inputs["cache_position"] = model_inputs["cache_position"].to(inputs_embeds.device)

            outputs = self.model.language_model(
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                **model_inputs,
            )

            # Update model_kwargs like cache_position for next generation.
            model_kwargs["attention_mask"] = attention_mask  # needed for the following update
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
            attention_mask = model_kwargs.pop("attention_mask", None) # to avoid future in-place modification
            hidden_state = outputs.last_hidden_state[:, -1, :].clone()

            # Generate scores using the generation head. (not using above defined lm head)
            scores = self.model.gen_head(hidden_state)
            logits = logits_processor(input_ids, scores)
            probs = torch.softmax(logits / generation_config.temperature, dim=-1)

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            generated_tokens[:, i] = next_token

            # Prepare embeddings for the next step.
            next_token = torch.cat([next_token, next_token])
            img_embeds = self.prepare_embeddings_for_image_generation(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

            # similar to GenerationMixin._sample, this is needed in prepare_inputs_for_generation
            input_tokens = torch.cat([input_tokens, next_token[:, None]], dim=-1)

        if return_dict_in_generate:
            if output_scores:
                scores += (scores,)
            if output_logits:
                raw_logits += (hidden_state.float(),)
            if output_attentions:
                decoder_attentions += outputs.attentions
            if output_hidden_states:
                decoder_hidden_states += outputs.hidden_states

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=generated_tokens,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=outputs.past_key_values,
            )
        else:
            return generated_tokens


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = PIL.Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = PIL.Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class JanusImageProcessor(BlipImageProcessor):
    r"""
    Constructs a JANUS image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 384, "width": 384}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        min_size (`int`, *optional*, defaults to 14):
            The minimum allowed size for the resized image. Ensures that neither the height nor width
            falls below this value after resizing.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

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
        size: Union[Dict[str, int], int],
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
                - `None`: will be inferred from input
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        if input_data_format == ChannelDimension.FIRST:
            _, height, width = image.shape
        elif input_data_format == ChannelDimension.LAST:
            height, width, _ = image.shape
        else:
            raise ValueError(
                f"Invalid `input_data_format`. Must be one of {ChannelDimension.FIRST}, {ChannelDimension.LAST}"
            )

        max_size = max(height, width)

        if isinstance(size, dict):
            if "height" not in size or "width" not in size:
                raise ValueError(
                    f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}"
                )
            if size["height"] != size["width"]:
                raise ValueError(
                    f"Output height and width must be the same. Got height={size['height']} and width={size['width']}"
                )
            size = size["height"]
        elif not isinstance(size, int):
            ValueError(f"Expected `size` to be of type `int` or `dict`. Got {type(size)}")

        delta = size / max_size

        # Largest side becomes `size` and the other side is scaled according to the aspect ratio
        output_size_nonpadded = [
            max(int(height * delta), self.min_size),
            max(int(width * delta), self.min_size),
        ]

        image = resize(
            image,
            size=output_size_nonpadded,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            return_numpy=False,
            **kwargs,
        )
        # expand and pad the images to obtain a square image of dimensions `size x size`
        image = expand2square(image, self.background_color)
        # PS: to_numpy_array puts the channel dimension last
        image = to_channel_dimension_format(
            to_numpy_array(image),
            data_format if data_format is not None else input_data_format,
            input_channel_dim=ChannelDimension.LAST,
        )
        return image

    def postprocess(
        self,
        images: ImageInput,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: List[float] = None,
        image_std: List[float] = None,
        input_data_format: str = None,
        return_tensors: str = None,
    ):
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = 1.0 / self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        images = make_list_of_images(images)  # Ensures input is a list

        if isinstance(images[0], PIL.Image.Image):
            return images if len(images) > 1 else images[0]

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])  # Determine format dynamically

        pixel_values = []

        for image in images:
            image = to_numpy_array(image)  # Ensure NumPy format

            if do_normalize:
                image = self.unnormalize(
                    image=image, image_mean=image_mean, image_std=image_std, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)
                image = image.clip(0, 255).astype(np.uint8)

            if do_normalize and do_rescale and return_tensors == "PIL.Image.Image":
                image = to_channel_dimension_format(image, ChannelDimension.LAST, input_channel_dim=input_data_format)
                image = PIL.Image.fromarray(image)

            pixel_values.append(image)

        data = {"pixel_values": pixel_values}
        return_tensors = return_tensors if return_tensors != "PIL.Image.Image" else None

        return BatchFeature(data=data, tensor_type=return_tensors)

    def unnormalize(
        self,
        image: np.array,
        image_mean: Union[float, Iterable[float]],
        image_std: Union[float, Iterable[float]],
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.array:
        """
        Unnormalizes `image` using the mean and standard deviation specified by `mean` and `std`.
        image = (image * image_std) + image_mean
        Args:
            image (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)` or `(num_channels, image_size, image_size)`):
                Batch of pixel values to postprocess.
            image_mean (`float` or `Iterable[float]`):
                The mean to use for unnormalization.
            image_std (`float` or `Iterable[float]`):
                The standard deviation to use for unnormalization.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        num_channels = 3

        if isinstance(image_mean, Iterable):
            if len(image_mean) != num_channels:
                raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(image_mean)}")
        else:
            image_mean = [image_mean] * num_channels

        if isinstance(image_std, Iterable):
            if len(image_std) != num_channels:
                raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {len(image_std)}")
        else:
            image_std = [image_std] * num_channels

        rev_image_mean = tuple(-mean / std for mean, std in zip(image_mean, image_std))
        rev_image_std = tuple(1 / std for std in image_std)
        image = self.normalize(
            image=image, mean=rev_image_mean, std=rev_image_std, input_data_format=input_data_format
        )
        return image


__all__ = [
    "JanusImageProcessor",
    "JanusPreTrainedModel",
    "JanusForConditionalGeneration",
    "JanusModel",
    "JanusVQVAE",
    "JanusVisionModel",
]
