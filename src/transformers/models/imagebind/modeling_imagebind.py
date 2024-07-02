# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""PyTorch ImageBind model."""

import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_imagebind import (
    ImageBindAudioConfig,
    ImageBindConfig,
    ImageBindTextConfig,
    ImageBindVisionConfig,
)


logger = logging.get_logger(__name__)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[batch_size, seq_len]` to `[batch_size, 1, tgt_seq_len, src_seq_len]`.
    """
    batch_size, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->imagebind
def imagebind_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


# BaseModelOutputWithPooling + num_clips field for modalities which have clips (vision, audio)
@dataclass
class ImageBindTransformerOutput(ModelOutput):
    """
    The output class for ImageBind*Transformer models. This is [`BaseModelOutputWithPooling`] with an additional
    `num_clips` field for modalities which are organized into clips as well as batches (vision, audio).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# CLIPTextModelOutput + normalized embeddings
class ImageBindTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        normalized_text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when model is initialized with `with_projection=True`):
            The normalized text embeddings obtained by applying the projection layer to the pooler_output, then
            applying L2 normalization and scaling the logits.
    """

    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    normalized_text_embeds: Optional[torch.FloatTensor] = None


@dataclass
# ClipVisionModelOutput + normalized embeddings
class ImageBindVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        normalized_image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when model is initialized with `with_projection=True`):
            The normalized image embeddings obtained by applying the projection layer to the pooler_output, then
            applying L2 normalization and scaling the logits.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    normalized_image_embeds: Optional[torch.FloatTensor] = None


# CLAPAudioModelOutput + normalized embeddings
@dataclass
class ImageBindAudioModelOutput(ModelOutput):
    """
    ClapAudio model output to mimic the output of the original implementation.

    Args:
        audio_embeds (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            The Audio embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        normalized_audio_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when model is initialized with `with_projection=True`):
            The normalized audio embeddings obtained by applying the projection layer to the pooler_output, then
            applying L2 normalization and scaling the logits.
    """

    audio_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    normalized_audio_embeds: Optional[torch.FloatTensor] = None


@dataclass
class ImageBindOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        logits_per_audio:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `audio_embeds` and `image_embeds`. This represents the audio-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The normalized text embeddings obtained by applying the projection layer to the pooled output of [`ImageBindTextModel`], then applying L2 normalization and logit scaling.
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The normalized image embeddings obtained by applying the projection layer to the pooled output of [`ImageBindVisionModel`], then applying L2 normalization and logit scaling.
        audio_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The normalized audio embeddings obtained by applying the projection layer to the pooled output of [`ImageBindAudioModel`], then applying L2 normalization and logit scaling.
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`ImageBindTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`ImageBindVisionModel`].
        audio_model_output(`BaseModelOutputWithPooling`):
            The output of the [`ImageBindAudioModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    logits_per_audio: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    audio_embeds: torch.FloatTensor = None
    vision_model_output: BaseModelOutputWithPooling = None
    text_model_output: BaseModelOutputWithPooling = None
    audio_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        fields_to_exclude = [
            "text_model_output",
            "vision_model_output",
            "audio_model_output",
        ]
        return tuple(self[k] if k not in fields_to_exclude else getattr(self, k).to_tuple() for k in self.keys())


class ImageBindGenericPatchEmbedding(nn.Module):
    """Generic Patch Embedding class that can be used for Vision (image/video), Audio, Depth, Thermal modalities."""

    def __init__(
        self,
        config: Union[ImageBindVisionConfig, ImageBindAudioConfig],
        projection: nn.Module,
        use_layernorm: bool = False,
    ):
        super().__init__()

        if hasattr(config, "image_size"):
            image_size = config.image_size
        elif hasattr(config, "num_mel_bins") and hasattr(config, "target_len"):
            image_size = (config.num_mel_bins, config.target_len)
        else:
            raise ValueError("Either `image_size` or `num_mel_bins` and `target_len` must be provided in the config.")

        self.image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        self.num_channels = config.num_channels

        self.projection = projection
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if use_layernorm else None

    def forward(self, input_values: torch.FloatTensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        if input_values.ndim not in [4, 5]:
            raise ValueError(f"Input tensor shape should have length 4 or 5 but got {input_values.ndim}.")

        _, num_channels, *spatial_shape = input_values.shape
        height, width = spatial_shape[-2:]

        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )

        embeddings = self.projection(input_values).flatten(2).transpose(1, 2)
        if self.layernorm is not None:
            embeddings = self.layernorm(embeddings)

        return embeddings


class ImageBindVisionEmbeddings(nn.Module):
    def __init__(self, config: ImageBindVisionConfig):
        super().__init__()
        self.config = config
        self.num_frames = config.num_frames
        num_patches = (config.image_size // config.patch_size) ** 2

        projection = nn.Conv3d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=(config.num_frames, config.patch_size, config.patch_size),
            stride=(config.num_frames, config.patch_size, config.patch_size),
            bias=False,
        )
        self.patch_embedding = ImageBindGenericPatchEmbedding(
            config=config, projection=projection, use_layernorm=False
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))

    # Copied from transformers.models.vit.modeling_vit.ViTEmbeddings.interpolate_pos_encoding
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def image_to_video(self, pixel_values: torch.FloatTensor, time_dim: int = 2, ntimes: int = 2):
        """
        Maps 4-dim image tensors of shape (B, C, H, W) to 5-dim video tensors, possibly repeating the image along the
        time dimension. For example, if `time_dim == 1`, RGB images of shape (B, C, H, W) will be transformed to
        video of shape (B, 1, C, H, W), and then the image will be repeated along the time dimension `ntimes` to get
        shape (B, N, C, H, W).
        """
        if pixel_values.ndim not in [4, 5]:
            raise ValueError(
                f"The input `image` tensor should be 4- or 5-dimensional but has {pixel_values.ndim} dimensions."
            )

        # Add time dimension at specified dim index
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(time_dim)

        # Repeat image across the time dimension ntimes.
        if pixel_values.shape[time_dim] == 1:
            new_shape = [1] * len(pixel_values.shape)
            new_shape[time_dim] = ntimes
            pixel_values = pixel_values.repeat(new_shape)

        return pixel_values

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        pixel_values = self.image_to_video(pixel_values, ntimes=self.num_frames)
        batch_size, num_channels, num_frames, height, width = pixel_values.shape

        embeddings = self.patch_embedding(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        return embeddings


class ImageBindAudioEmbeddings(nn.Module):
    def __init__(self, config: ImageBindAudioConfig):
        super().__init__()
        self.config = config

        num_patches_height = int((config.num_mel_bins - config.patch_size) / config.stride + 1)
        num_patches_width = int((config.target_len - config.patch_size) / config.stride + 1)
        num_patches = num_patches_height * num_patches_width

        proj = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.stride,
            bias=False,
        )

        self.patch_embedding = ImageBindGenericPatchEmbedding(config=config, projection=proj, use_layernorm=True)

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))

    def forward(self, input_features: torch.FloatTensor) -> torch.Tensor:
        embeddings = self.patch_embedding(input_features, interpolate_pos_encoding=False)

        cls_tokens = self.cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # Could also add interpolation of position encoding as well
        embeddings = embeddings + self.position_embeddings

        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPTextEmbeddings with CLIP->ImageBind
class ImageBindTextEmbeddings(nn.Module):
    def __init__(self, config: ImageBindTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


# CLIPAttention + key/value biases
class ImageBindAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
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
        self.dropout = config.attention_dropout

        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Create bias parameters for key and value sequences.
        if config.add_kv_bias:
            self.k_bias = nn.Parameter(torch.empty((1, 1, self.embed_dim)))
            self.v_bias = nn.Parameter(torch.empty((1, 1, self.embed_dim)))
        else:
            self.k_bias = None
            self.v_bias = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_len, embed_dim = hidden_states.size()

        qkv = self.qkv_proj(hidden_states).reshape(batch_size, seq_len, 3, -1).permute(2, 0, 1, 3)
        query_states, key_states, value_states = qkv.unbind(0)

        query_states = query_states * self.scale

        # Add key/value biases if necessary
        if self.k_bias is not None and self.v_bias is not None:
            # Repeat bias along batch dimension (first)
            key_states = torch.cat([key_states, self.k_bias.repeat(batch_size, 1, 1)], dim=1)
            value_states = torch.cat([value_states, self.v_bias.repeat(batch_size, 1, 1)], dim=1)

        key_states = self._shape(key_states, -1, batch_size)
        value_states = self._shape(value_states, -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, seq_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (batch_size * self.num_heads, seq_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, seq_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, seq_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, seq_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(batch_size, self.num_heads, seq_len, src_len) + attention_mask
            attn_weights = attn_weights.view(batch_size * self.num_heads, seq_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, seq_len, src_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, seq_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (batch_size * self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(batch_size, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class ImageBindMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        intermediate_size = int(config.hidden_size * config.mlp_ratio)

        self.fc1 = nn.Linear(config.hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->ImageBind
class ImageBindDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# CLIPEncoderLayer with DropPath layer after each residual subblock (attention, feedforward)
class ImageBindEncoderLayer(nn.Module):
    def __init__(
        self,
        config: Union[ImageBindVisionConfig, ImageBindTextConfig, ImageBindAudioConfig],
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = ImageBindAttention(config)
        self.layernorm_before = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = ImageBindMlp(config)
        self.layernorm_after = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        if drop_path_rate > 0.0:
            self.drop_path = ImageBindDropPath(drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layernorm_before(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.drop_path(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.drop_path(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class ImageBindPostProcessor(nn.Module):
    """
    Post-processes ImageBind embeddings by using a normalize layer followed by an optional logit scaling layer.
    """

    def __init__(
        self,
        config,
        dim: int = -1,
        max_logit_scale: float = 100,
    ):
        super().__init__()
        self.dim = dim
        self.scale_logits = config.logit_scale_init_value is not None

        if self.scale_logits:
            self.logit_scale_init = config.logit_scale_init_value
            self.max_logit_scale = max_logit_scale
            self.learnable = config.learnable_logit_scale

            log_logit_scale = torch.ones([]) * np.log(self.logit_scale_init)
            if self.learnable:
                self.log_logit_scale = nn.Parameter(log_logit_scale)
            else:
                self.register_buffer("log_logit_scale", log_logit_scale)

    def forward(self, logits: torch.FloatTensor) -> torch.FloatTensor:
        logits = nn.functional.normalize(logits, dim=self.dim, p=2)
        if self.scale_logits:
            logits = torch.clip(self.log_logit_scale.exp(), max=self.max_logit_scale) * logits
        return logits


class ImageBindPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ImageBindConfig
    base_model_prefix = "imagebind"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, ImageBindTextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, (ImageBindVisionEmbeddings, ImageBindAudioEmbeddings)):
            factor = self.config.initializer_factor
            nn.init.normal_(module.cls_token, std=module.config.hidden_size**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.projection.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embeddings, std=module.config.initializer_range * factor)
        elif isinstance(module, ImageBindAttention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.qkv_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
            if module.k_bias is not None:
                nn.init.normal_(module.k_bias, std=in_proj_std)
            if module.v_bias is not None:
                nn.init.normal_(module.v_bias, std=in_proj_std)
        elif isinstance(module, ImageBindMlp):
            factor = self.config.initializer_factor
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, ImageBindModel):
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.vision_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.audio_projection.weight,
                std=module.audio_embed_dim**-0.5 * self.config.initializer_factor,
            )

            configs = [self.config.text_config, self.config.vision_config, self.config.audio_config]
            modalities = ["text", "vision", "audio"]
            for config, modality in zip(configs, modalities):
                logit_scale_init_value, learnable_logit_scale = (
                    config.logit_scale_init_value,
                    config.learnable_logit_scale,
                )
                if logit_scale_init_value is not None and learnable_logit_scale:
                    logit_scale = torch.ones([]) * np.log(logit_scale_init_value) * factor
                    postprocessor = getattr(module, f"{modality}_postprocessor")
                    postprocessor.log_logit_scale = nn.Parameter(logit_scale)

        elif isinstance(module, ImageBindVisionModelWithProjection):
            nn.init.normal_(
                module.vision_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
            logit_scale_init_value = self.config.logit_scale_init_value
            learnable_logit_scale = self.config.learnable_logit_scale
            if logit_scale_init_value is not None and learnable_logit_scale:
                logit_scale = torch.ones([]) * np.log(logit_scale_init_value) * self.config.initializer_factor
                module.vision_postprocessor.log_logit_scale = nn.Parameter(logit_scale)
        elif isinstance(module, ImageBindTextModelWithProjection):
            nn.init.normal_(
                module.text_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
            logit_scale_init_value = self.config.logit_scale_init_value
            learnable_logit_scale = self.config.learnable_logit_scale
            if logit_scale_init_value is not None and learnable_logit_scale:
                logit_scale = torch.ones([]) * np.log(logit_scale_init_value) * self.config.initializer_factor
                module.text_postprocessor.log_logit_scale = nn.Parameter(logit_scale)
        elif isinstance(module, ImageBindAudioModelWithProjection):
            nn.init.normal_(
                module.audio_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
            logit_scale_init_value = self.config.logit_scale_init_value
            learnable_logit_scale = self.config.learnable_logit_scale
            if logit_scale_init_value is not None and learnable_logit_scale:
                logit_scale = torch.ones([]) * np.log(logit_scale_init_value) * self.config.initializer_factor
                module.audio_postprocessor.log_logit_scale = nn.Parameter(logit_scale)

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ImageBindEncoder):
            module.gradient_checkpointing = value


IMAGEBIND_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ImageBindConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

IMAGEBIND_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

IMAGEBIND_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`ImageBindImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

IMAGEBIND_AUDIO_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor` of shape `(batch_size, num_mel_bins, target_len)`):
            Input features. Padding will be ignored by default should you provide it. Input features can be obtained
            using [`AutoFeatureExtractor`]. See [`ImageBindFeatureExtractor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

IMAGEBIND_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`ImageBindImageProcessor.__call__`] for details.
        input_features (`torch.FloatTensor` of shape `(batch_size, num_mel_bins, target_len)`):
            Input features. Padding will be ignored by default should you provide it. Input features can be obtained
            using [`AutoFeatureExtractor`]. See [`ImageBindFeatureExtractor.__call__`] for details.
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# CLIPEncoder with DropPath support
class ImageBindEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`ImageBindEncoderLayer`].

    Args:
        config: ImageBindConfig
    """

    def __init__(self, config: ImageBindConfig):
        super().__init__()
        self.config = config

        drop_path_rates = [prob.item() for prob in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layers = nn.ModuleList(
            [ImageBindEncoderLayer(config, drop_path_rate) for drop_path_rate in drop_path_rates]
        )
        self.gradient_checkpointing = False

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
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
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


class ImageBindTextTransformer(nn.Module):
    def __init__(self, config: ImageBindTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = ImageBindTextEmbeddings(config)
        self.encoder = ImageBindEncoder(config)
        self.layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(IMAGEBIND_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindTransformerOutput, config_class=ImageBindTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageBindTransformerOutput]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        batch_size, seq_len = input_shape

        attention_mask = self._build_attention_mask(
            attention_mask, batch_size, seq_len, hidden_states.dtype, hidden_states.device
        )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.layernorm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return ImageBindTransformerOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_attention_mask(self, attention_mask, batch_size, seq_len, dtype, device=None):
        # Build causal mask
        mask = torch.empty(batch_size, seq_len, seq_len, dtype=dtype, device=device)
        mask.fill_(torch.finfo(dtype).min)
        mask.triu_(1)
        mask = mask.unsqueeze(1)  # expand mask

        # If attention_mask update causal mask
        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, dtype)
            return mask + attention_mask
        return mask


@add_start_docstrings(
    """The text model from ImageBind without any head or projection on top.""",
    IMAGEBIND_START_DOCSTRING,
)
class ImageBindTextModel(ImageBindPreTrainedModel):
    config_class = ImageBindTextConfig

    _no_split_modules = ["ImageBindEncoderLayer"]

    def __init__(self, config: ImageBindTextConfig):
        super().__init__(config)
        self.text_model = ImageBindTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(IMAGEBIND_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindTransformerOutput, config_class=ImageBindTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageBindTransformerOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ImageBindTextModel

        >>> model = ImageBindTextModel.from_pretrained("EduardoPacheco/imagebind-huge")
        >>> tokenizer = AutoTokenizer.from_pretrained("EduardoPacheco/imagebind-huge")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class ImageBindVisionTransformer(nn.Module):
    def __init__(self, config: ImageBindVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = ImageBindVisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = ImageBindEncoder(config)
        self.layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(IMAGEBIND_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindTransformerOutput, config_class=ImageBindVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageBindTransformerOutput]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # For video inputs we take multiple clips and average the embeddings
        # See https://github.com/facebookresearch/ImageBind/blob/main/imagebind/models/imagebind_model.py#L470
        reduce_clips = pixel_values.ndim >= 5
        if reduce_clips:
            batch_size, num_clips = pixel_values.shape[:2]
            pixel_values = pixel_values.reshape(batch_size * num_clips, *pixel_values.shape[2:])

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layernorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return ImageBindTransformerOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """The vision model from ImageBind without any head or projection on top.""",
    IMAGEBIND_START_DOCSTRING,
)
class ImageBindVisionModel(ImageBindPreTrainedModel):
    config_class = ImageBindVisionConfig
    _no_split_modules = ["ImageBindEncoderLayer"]

    main_input_name = "pixel_values"

    def __init__(self, config: ImageBindVisionConfig):
        super().__init__(config)
        self.vision_model = ImageBindVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(IMAGEBIND_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindTransformerOutput, config_class=ImageBindVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageBindTransformerOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ImageBindVisionModel

        >>> model = ImageBindVisionModel.from_pretrained("EduardoPacheco/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("EduardoPacheco/imagebind-huge")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class ImageBindAudioTransformer(nn.Module):
    def __init__(self, config: ImageBindAudioConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = ImageBindAudioEmbeddings(config)
        self.encoder = ImageBindEncoder(config)
        self.layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(IMAGEBIND_AUDIO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindTransformerOutput, config_class=ImageBindAudioConfig)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageBindTransformerOutput]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_features is None:
            raise ValueError("You have to specify input_features")

        # If audio is chunked (i.e. same audio is split into multiple clips), reduce embedding over clips dimension
        # See https://github.com/facebookresearch/ImageBind/blob/main/imagebind/models/imagebind_model.py#L470
        reduce_clips = input_features.ndim >= 5
        if reduce_clips:
            batch_size, num_clips = input_features.shape[:2]
            input_features = input_features.reshape(batch_size * num_clips, *input_features.shape[2:])

        hidden_states = self.embeddings(input_features)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return ImageBindTransformerOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """The vision model from ImageBind without any head or projection on top.""",
    IMAGEBIND_START_DOCSTRING,
)
class ImageBindAudioModel(ImageBindPreTrainedModel):
    config_class = ImageBindAudioConfig
    _no_split_modules = ["ImageBindEncoderLayer"]

    main_input_name = "input_features"

    def __init__(self, config: ImageBindAudioConfig):
        super().__init__(config)
        self.audio_model = ImageBindAudioTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.audio_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(IMAGEBIND_AUDIO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindTransformerOutput, config_class=ImageBindAudioConfig)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageBindTransformerOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> import torchaudio
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, ImageBindAudioModel

        >>> ds = load_dataset("EduardoPacheco/imagebind-example-data", split="train")
        >>> audio = ds[0]["audio"]
        >>> audio = torchaudio.functional.resample(torch.from_numpy(audio["array"]), orig_freq=audio["sampling_rate"], new_freq=16000).numpy()

        >>> model = ImageBindAudioModel.from_pretrained("EduardoPacheco/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("EduardoPacheco/imagebind-huge")

        >>> inputs = processor(audios=audio, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.audio_model(
            input_features=input_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


@add_start_docstrings(IMAGEBIND_START_DOCSTRING)
class ImageBindModel(ImageBindPreTrainedModel):
    config_class = ImageBindConfig
    main_input_name = "pixel_values"

    def __init__(self, config: ImageBindConfig):
        super().__init__(config)

        if not isinstance(config.text_config, ImageBindTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type ImageBindTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, ImageBindVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type ImageBindVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        if not isinstance(config.audio_config, ImageBindAudioConfig):
            raise ValueError(
                "config.audio_config is expected to be of type ImageBindAudioConfig but is of type"
                f" {type(config.audio_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config
        audio_config = config.audio_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        self.audio_embed_dim = audio_config.hidden_size

        self.text_model = ImageBindTextTransformer(text_config)
        self.vision_model = ImageBindVisionTransformer(vision_config)
        self.audio_model = ImageBindAudioTransformer(audio_config)

        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.vision_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.audio_projection = nn.Linear(self.audio_embed_dim, self.projection_dim, bias=False)

        self.text_postprocessor = ImageBindPostProcessor(text_config)
        self.vision_postprocessor = ImageBindPostProcessor(vision_config)
        self.audio_postprocessor = ImageBindPostProcessor(audio_config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(IMAGEBIND_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`ImageBindTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ImageBindModel

        >>> model = ImageBindModel.from_pretrained("EduardoPacheco/imagebind-huge")
        >>> tokenizer = AutoTokenizer.from_pretrained("EduardoPacheco/imagebind-huge")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use ImageBind model's config for some fields (if specified) instead of those in the text component.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    @add_start_docstrings_to_model_forward(IMAGEBIND_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`ImageBindVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ImageBindModel

        >>> model = ImageBindModel.from_pretrained("EduardoPacheco/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("EduardoPacheco/imagebind-huge")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use ImageBind model's config for some fields (if specified) instead of those in the vision components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = pixel_values.shape[0]

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.vision_projection(pooled_output)

        if pixel_values.ndim >= 5:
            num_clips = pixel_values.shape[1]
            image_features = image_features.reshape(batch_size, num_clips, -1)
            # Take mean over all clips
            image_features = image_features.mean(dim=1)

        return image_features

    @add_start_docstrings_to_model_forward(IMAGEBIND_AUDIO_INPUTS_DOCSTRING)
    def get_audio_features(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            audio_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The audio embeddings obtained by
            applying the projection layer to the pooled output of [`ImageBindAudioModel`].

        Examples:

        ```python
        >>> import torchaudio
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, ImageBindModel

        >>> ds = load_dataset("EduardoPacheco/imagebind-example-data", split="train")
        >>> audio = ds[0]["audio"]
        >>> audio = torchaudio.functional.resample(torch.from_numpy(audio["array"]), orig_freq=audio["sampling_rate"], new_freq=16000).numpy()

        >>> model = ImageBindModel.from_pretrained("EduardoPacheco/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("EduardoPacheco/imagebind-huge")

        >>> inputs = processor(audios=audio, return_tensors="pt")

        >>> audio_features = model.get_audio_features(**inputs)
        ```"""
        # Use ImageBind model's config for some fields (if specified) instead of those in the audio component.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_features.shape[0]

        audio_outputs = self.audio_model(
            input_features=input_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = audio_outputs[1]  # pooled_output
        audio_features = self.audio_projection(pooled_output)

        # If audio is chunked (i.e. same audio is split into multiple clips), reduce embedding over clips dimension
        # See https://github.com/facebookresearch/ImageBind/blob/main/imagebind/models/imagebind_model.py#L470
        if input_features.ndim >= 5:
            num_clips = input_features.shape[1]
            audio_features = audio_features.reshape(batch_size, num_clips, -1)
            # Take mean over all clips
            audio_features = audio_features.mean(dim=1)

        return audio_features

    @add_start_docstrings_to_model_forward(IMAGEBIND_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindOutput, config_class=ImageBindConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_features: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageBindOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ImageBindModel

        >>> model = ImageBindModel.from_pretrained("EduardoPacheco/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("EduardoPacheco/imagebind-huge")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # We expect a combination of pixel_values and one of the other inputs i.e. input_features or input_ids should be provided
        if input_ids is None and input_features is None:
            raise ValueError("At least one of `input_ids` or `input_features` should be provided.")

        # We expect only one of input_features or input_ids to be provided
        if input_ids is not None and input_features is not None:
            raise ValueError("Only one of `input_ids` or `input_features` should be provided.")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # running the vision model
        image_batch_size = pixel_values.shape[0]

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.vision_projection(image_embeds)
        image_embeds = self.vision_postprocessor(image_embeds)

        # For video inputs we take multiple clips and average the embeddings
        # See https://github.com/facebookresearch/ImageBind/blob/main/imagebind/models/imagebind_model.py#L470
        if pixel_values.ndim >= 5:
            image_num_clips = pixel_values.shape[1]
            image_embeds = image_embeds.reshape(image_batch_size, image_num_clips, -1)
            image_embeds = image_embeds.mean(dim=1)

        # running the text model if input_ids is provided
        text_embeds = None
        logits_per_text = None
        text_outputs = None
        if input_ids is not None:
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            text_embeds = text_outputs[1]
            text_embeds = self.text_projection(text_embeds)
            text_embeds = self.text_postprocessor(text_embeds)

            logits_per_text = torch.matmul(text_embeds, image_embeds.t())
            logits_per_image = logits_per_text.t()

        # running the audio model if input_features is provided
        audio_embeds = None
        logits_per_audio = None
        audio_outputs = None
        if input_features is not None:
            audio_batch_size = input_features.shape[0]
            audio_outputs = self.audio_model(
                input_features,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            audio_embeds = audio_outputs[1]
            audio_embeds = self.audio_projection(audio_embeds)
            audio_embeds = self.audio_postprocessor(audio_embeds)

            if input_features.ndim >= 5:
                num_clips = input_features.shape[1]
                audio_embeds = audio_embeds.reshape(audio_batch_size, num_clips, -1)
                audio_embeds = audio_embeds.mean(dim=1)

            logits_per_audio = torch.matmul(audio_embeds, image_embeds.t())
            logits_per_image = logits_per_audio.t()

        loss = None
        if return_loss:
            loss = imagebind_loss(logits_per_text) if logits_per_text is not None else imagebind_loss(logits_per_audio)

        if not return_dict:
            output = (
                logits_per_image,
                logits_per_text,
                logits_per_audio,
                image_embeds,
                text_embeds,
                audio_embeds,
                vision_outputs,
                text_outputs,
                audio_outputs,
            )
            output = tuple([out for out in output if out is not None])
            return ((loss,) + output) if loss is not None else output

        return ImageBindOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            logits_per_audio=logits_per_audio,
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            audio_embeds=audio_embeds,
            vision_model_output=vision_outputs,
            text_model_output=text_outputs,
            audio_model_output=audio_outputs,
        )


@add_start_docstrings(
    """
    ImageBind Text Model with a projection layer on top (a linear layer on top of the pooled output).
    """,
    IMAGEBIND_START_DOCSTRING,
)
class ImageBindTextModelWithProjection(ImageBindPreTrainedModel):
    config_class = ImageBindTextConfig

    _no_split_modules = ["ImageBindEncoderLayer"]

    def __init__(self, config: ImageBindTextConfig):
        super().__init__(config)

        self.text_model = ImageBindTextTransformer(config)

        self.text_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        self.text_postprocessor = ImageBindPostProcessor(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(IMAGEBIND_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindTextModelOutput, config_class=ImageBindTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageBindTextModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ImageBindTextModelWithProjection

        >>> model = ImageBindTextModelWithProjection.from_pretrained("EduardoPacheco/imagebind-huge")
        >>> tokenizer = AutoTokenizer.from_pretrained("EduardoPacheco/imagebind-huge")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> text_embeds = outputs.text_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]

        text_embeds = self.text_projection(pooled_output)
        normalized_text_embeds = self.text_postprocessor(text_embeds)

        if not return_dict:
            outputs = (text_embeds, text_outputs[0]) + text_outputs[2:] + (normalized_text_embeds,)
            return tuple(output for output in outputs if output is not None)

        return ImageBindTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
            normalized_text_embeds=normalized_text_embeds,
        )


@add_start_docstrings(
    """
    ImageBind Vision Model with a projection layer on top (a linear layer on top of the pooled output).
    """,
    IMAGEBIND_START_DOCSTRING,
)
class ImageBindVisionModelWithProjection(ImageBindPreTrainedModel):
    config_class = ImageBindVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: ImageBindVisionConfig):
        super().__init__(config)

        self.vision_model = ImageBindVisionTransformer(config)

        self.vision_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        self.vision_postprocessor = ImageBindPostProcessor(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(IMAGEBIND_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindVisionModelOutput, config_class=ImageBindVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageBindVisionModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ImageBindVisionModelWithProjection

        >>> model = ImageBindVisionModelWithProjection.from_pretrained("EduardoPacheco/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("EduardoPacheco/imagebind-huge")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> image_embeds = outputs.image_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = pixel_values.shape[0]

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output

        image_embeds = self.vision_projection(pooled_output)
        normalized_image_embeds = self.vision_postprocessor(image_embeds)

        if pixel_values.ndim >= 5:
            num_clips = pixel_values.shape[1]
            image_embeds = image_embeds.reshape(batch_size, num_clips, -1)
            # Take mean over all clips
            image_embeds = image_embeds.mean(dim=1)

            normalized_image_embeds = normalized_image_embeds.reshape(batch_size, num_clips, -1)
            normalized_image_embeds = normalized_image_embeds.mean(dim=1)

        if not return_dict:
            outputs = (image_embeds, vision_outputs[0]) + vision_outputs[2:] + (normalized_image_embeds,)
            return tuple(output for output in outputs if output is not None)

        return ImageBindVisionModelOutput(
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
            normalized_image_embeds=normalized_image_embeds,
        )


@add_start_docstrings(
    """
    ImageBind Audio Model with a projection layer on top (a linear layer on top of the pooled output).
    """,
    IMAGEBIND_START_DOCSTRING,
)
class ImageBindAudioModelWithProjection(ImageBindPreTrainedModel):
    config_class = ImageBindAudioConfig
    main_input_name = "input_features"

    def __init__(self, config: ImageBindAudioConfig):
        super().__init__(config)

        self.audio_model = ImageBindAudioTransformer(config)

        self.audio_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        self.audio_postprocessor = ImageBindPostProcessor(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.audio_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(IMAGEBIND_AUDIO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindAudioModelOutput, config_class=ImageBindAudioConfig)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageBindAudioModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> import torch
        >>> import torchaudio
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, ImageBindAudioModelWithProjection

        >>> ds = load_dataset("EduardoPacheco/imagebind-example-data", split="train")
        >>> audio = ds[0]["audio"]
        >>> audio = torchaudio.functional.resample(torch.from_numpy(audio["array"]), orig_freq=audio["sampling_rate"], new_freq=16000).numpy()

        >>> model = ImageBindAudioModelWithProjection.from_pretrained("EduardoPacheco/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("EduardoPacheco/imagebind-huge")

        >>> inputs = processor(audios=audio, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> audio_embeds = outputs.audio_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_features.shape[0]

        audio_outputs = self.audio_model(
            input_features=input_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = audio_outputs[1]  # pooled_output

        audio_embeds = self.audio_projection(pooled_output)
        normalized_audio_embeds = self.audio_postprocessor(audio_embeds)

        if input_features.ndim >= 5:
            num_clips = input_features.shape[1]
            audio_embeds = audio_embeds.reshape(batch_size, num_clips, -1)
            # Take mean over all clips
            audio_embeds = audio_embeds.mean(dim=1)

            normalized_audio_embeds = normalized_audio_embeds.reshape(batch_size, num_clips, -1)
            normalized_audio_embeds = normalized_audio_embeds.mean(dim=1)

        if not return_dict:
            outputs = (audio_embeds, audio_outputs[0]) + audio_outputs[2:] + (normalized_audio_embeds,)
            return tuple(output for output in outputs if output is not None)

        return ImageBindAudioModelOutput(
            audio_embeds=audio_embeds,
            last_hidden_state=audio_outputs.last_hidden_state,
            hidden_states=audio_outputs.hidden_states,
            attentions=audio_outputs.attentions,
            normalized_audio_embeds=normalized_audio_embeds,
        )
