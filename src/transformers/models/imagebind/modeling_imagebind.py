# Copyright 2023 The HuggingFace Team. All rights reserved.
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
""" PyTorch ImageBind model."""


from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from timm.layers import DropPath

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
    ImageBindConfig,
    ImageBindAudioConfig,
    ImageBindDepthConfig,
    ImageBindImuConfig,
    ImageBindTextConfig,
    ImageBindThermalConfig,
    ImageBindVisionConfig,
)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/imagebind-huge"

IMAGEBIND_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/imagebind-huge",
    # See all ImageBind models at https://huggingface.co/models?filter=imagebind
]


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# TODO: can use code already in transformers?
# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/ImageBind.html
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
        num_clips: (`int`, *optional*):
            The number of clips for modalities which have both a batch dimension (dim 0) and clip dimension (dim 1).
            In the original ImageBind model, these modalities are vision (image/video) and audio.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    num_clips: Optional[int] = None


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
class ImageBindDepthModelOutput(ModelOutput):
    """
    Base class for depth model's outputs that also contains a pooling of the last hidden states.

    Args:
        depth_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The depth embeddings obtained by applying the projection layer to the pooler_output.
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
        normalized_depth_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when model is initialized with `with_projection=True`):
            The normalized depth embeddings obtained by applying the projection layer to the pooler_output, then
            applying L2 normalization and scaling the logits.
    """
    
    depth_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    normalized_depth_embeds: Optional[torch.FloatTensor] = None


@dataclass
class ImageBindThermalModelOutput(ModelOutput):
    """
    Base class for thermal model's outputs that also contains a pooling of the last hidden states.

    Args:
        thermal_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The thermal embeddings obtained by applying the projection layer to the pooler_output.
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
        normalized_thermal_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when model is initialized with `with_projection=True`):
            The normalized thermal embeddings obtained by applying the projection layer to the pooler_output, then
            applying L2 normalization and scaling the logits.
    """
    
    thermal_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    normalized_thermal_embeds: Optional[torch.FloatTensor] = None


@dataclass
class ImageBindImuModelOutput(ModelOutput):
    """
    Base class for IMU model's outputs that also contains a pooling of the last hidden states.

    Args:
        imu_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The IMU embeddings obtained by applying the projection layer to the pooler_output.
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
        normalized_imu_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when model is initialized with `with_projection=True`):
            The normalized IMU embeddings obtained by applying the projection layer to the pooler_output, then
            applying L2 normalization and scaling the logits.
    """
    
    imu_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    normalized_imu_embeds: Optional[torch.FloatTensor] = None


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPOutput with CLIP->ImageBind
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
        logits_per_depth:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `depth_embeds` and `image_embeds`. This represents the depth-image
            similarity scores.
        logits_per_thermal:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `thermal_embeds` and `image_embeds`. This represents the thermal-image
            similarity scores.
        logits_per_imu:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `imu_embeds` and `image_embeds`. This represents the IMU-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The normalized text embeddings obtained by applying the projection layer to the pooled output of [`ImageBindTextModel`], then applying L2 normalization and logit scaling.
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The normalized image embeddings obtained by applying the projection layer to the pooled output of [`ImageBindVisionModel`], then applying L2 normalization and logit scaling.
        audio_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The normalized audio embeddings obtained by applying the projection layer to the pooled output of [`ImageBindAudioModel`], then applying L2 normalization and logit scaling.
        depth_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The normalized depth embeddings obtained by applying the projection layer to the pooled output of [`ImageBindDepthModel`], then applying L2 normalization and logit scaling.
        thermal_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The normalized thermal embeddings obtained by applying the projection layer to the pooled output of [`ImageBindThermalModel`], then applying L2 normalization and logit scaling.
        imu_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The normalized IMU embeddings obtained by applying the projection layer to the pooled output of [`ImageBindImuModel`], then applying L2 normalization and logit scaling.
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`ImageBindTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`ImageBindVisionModel`].
        audio_model_output(`BaseModelOutputWithPooling`):
            The output of the [`ImageBindAudioModel`].
        depth_model_output(`BaseModelOutputWithPooling`):
            The output of the [`ImageBindDepthModel`].
        thermal_model_output(`BaseModelOutputWithPooling`):
            The output of the [`ImageBindThermalModel`].
        imu_model_output(`BaseModelOutputWithPooling`):
            The output of the [`ImageBindImuModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    logits_per_audio: torch.FloatTensor = None
    logits_per_depth: torch.FloatTensor = None
    logits_per_thermal: torch.FloatTensor = None
    logits_per_imu: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    audio_embeds: torch.FloatTensor = None
    depth_embeds: torch.FloatTensor = None
    thermal_embeds: torch.FloatTensor = None
    imu_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None
    audio_model_output: BaseModelOutputWithPooling = None
    depth_model_output: BaseModelOutputWithPooling = None
    thermal_model_output: BaseModelOutputWithPooling = None
    imu_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        fields_to_exclude = [
            "text_model_output",
            "vision_model_output",
            "audio_model_output",
            "depth_model_output",
            "thermal_model_output",
            "imu_model_output",
        ]
        return tuple(
            self[k] if k not in fields_to_exclude else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# Copied from transformers.models.clip.modeling_clip.CLIPTextEmbeddings with CLIP->ImageBind
class ImageBindTextEmbeddings(nn.Module):
    def __init__(self, config: ImageBindTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

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


class RGBDTPatchEmbedding(nn.Module):
    """
    Creates patch embeddings for spatiotemporal data (e.g. images, video, depth etc.). This handles patch embeddings
    for all image-like modalities (image/video, depth, thermal).
    """
    def __init__(
        self,
        config: Union[ImageBindAudioConfig, ImageBindDepthConfig, ImageBindThermalConfig, ImageBindVisionConfig],
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.stride = config.stride
        self.num_frames = config.num_frames if hasattr(config, "num_frames") else None
        self.is_temporal = self.num_frames is not None

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        if self.is_temporal:
            patch_embedding_cls = nn.Conv3d
        else:
            patch_embedding_cls = nn.Conv2d
        
        self.patch_embedding = patch_embedding_cls(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride,
            bias=False,
        )
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

        if self.is_temporal:
            self.time_patch_size = self.patch_size[0]
            self.spatial_patch_size = self.patch_size[1]
            self.num_patches = (config.num_frames // self.time_patch_size) * (self.image_size // self.spatial_patch_size) ** 2
        else:
            self.time_patch_size = None
            self.spatial_patch_size = self.patch_size
            self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))
    
    def image_to_video(self, image: torch.FloatTensor, time_dim: int = 2, ntimes: int = 2, pad_type: str = "repeat"):
        """
        Maps 4-dim image tensors of shape (B, C, H, W) to 5-dim video tensors, possibly repeating the image along the
        time dimension. For example, if `time_dim == 1`, RGB images of shape (B, C, H, W) will be transformed to
        video of shape (B, 1, C, H, W), and then the image will be repeated along the time dimension `ntimes` to get
        shape (B, N, C, H, W).
        """
        if image.ndim not in [4, 5]:
            raise ValueError(
                f"The input `image` tensor should be 4- or 5-dimensional but has {image.ndim} dimensions."
            )

        # Add time dimension at specified dim index
        if image.ndim == 4:
            image = image.unsqueeze(time_dim)

        # Repeat image across the time dimension ntimes.
        if image.shape[time_dim] == 1:
            if pad_type == "repeat":
                new_shape = [1] * len(image.shape)
                new_shape[time_dim] = ntimes
                video = image.repeat(new_shape)
            elif pad_type == "zero":
                pad_arg = [0, 0] * len(image.shape)
                pad_arg[2 * time_dim + 1] = self.ntimes - image.shape[time_dim]
                video = nn.functional.pad(image, pad_arg)
        return video

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        if self.is_temporal:
            pixel_values = self.image_to_video(pixel_values, time_dim=1, ntimes=self.num_frames)
        
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        patch_embeds = self.norm_layer(patch_embeds)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class ImageBindVisionEmbeddings(RGBDTPatchEmbedding):
    def __init__(self, config: ImageBindVisionConfig):
        super().__init__(config, norm_layer=None)


class ImageBindAudioEmbeddings(RGBDTPatchEmbedding):
    def __init__(self, config: ImageBindAudioConfig):
        layer_norm = nn.LayerNorm(config.hidden_size)
        super().__init__(config, norm_layer=layer_norm)
    
    def forward(self, audio: torch.FloatTensor) -> torch.Tensor:
        super().forward(pixel_values=audio)


class ImageBindDepthEmbeddings(RGBDTPatchEmbedding):
    def __init__(self, config: ImageBindDepthConfig):
        layer_norm = nn.LayerNorm(config.hidden_size)
        super().__init__(config, norm_layer=layer_norm)
    
    def forward(self, depth: torch.FloatTensor) -> torch.Tensor:
        super().forward(pixel_values=depth)


class ImageBindThermalEmbeddings(RGBDTPatchEmbedding):
    def __init__(self, config: ImageBindThermalConfig):
        layer_norm = nn.LayerNorm(config.hidden_size)
        super().__init__(config, norm_layer=layer_norm)
    
    def forward(self, thermal: torch.FloatTensor) -> torch.Tensor:
        super().forward(pixel_values=thermal)


class ImageBindImuEmbeddings(nn.Module):
    def __init__(self, config: ImageBindImuConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.kernel_size = config.kernel_size
        self.in_features = config.input_shape[0] * self.kernel_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Linear(self.in_features, self.embed_dim, bias=False)
        self.norm_layer = nn.LayerNorm(self.embed_dim)

        self.num_patches = config.input_shape[1] // self.kernel_size
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))
    
    def forward(self, imu: torch.FloatTensor) -> torch.Tensor:
        batch_size = imu.shape[0]

        # Patchify
        # (B, L, D) -> (B, L, D // K, K) -> (B, D // K, L, K)
        patches = imu.unfold(-1, self.kernel_size, self.kernel_size).permute(0, 2, 1, 3)
        patches = patches.reshape(batch_size, patches.shape[1], -1)

        patch_embeds = self.patch_embedding(patches)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        patch_embeds = self.norm_layer(patch_embeds)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
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

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Create bias parameters for key and value sequences.
        if config.add_kv_bias:
            self.k_bias = nn.Parameter(torch.empty((1, 1, self.embed_dim)))
            self.v_bias = nn.Parameter(torch.empty((1, 1, self.embed_dim)))
        else:
            self.k_bias = None
            self.v_bias = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Add key/value biases if necessary
        if self.k_bias is not None and self.v_bias is not None:
            # Repeat bias along batch dimension (first)
            key_states = torch.cat([key_states, self.k_bias.repeat(bsz, 1, 1)])
            value_states = torch.cat([value_states, self.v_bias.repeat(bsz, 1, 1)])
        
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->ImageBind
class ImageBindMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# CLIPEncoderLayer with DropPath layer after each residual subblock (attention, feedforward)
class ImageBindEncoderLayer(nn.Module):
    def __init__(self, config: ImageBindConfig, drop_path_rate: float = 0.0):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = ImageBindAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = ImageBindMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        if drop_path_rate > 0.0:
            self.drop_path = DropPath(drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
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

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.drop_path(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
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
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, ImageBindTextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, RGBDTPatchEmbedding):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, ImageBindImuEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, ImageBindAttention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
            if module.k_bias is not None:
                nn.init.normal_(module.k_bias, std=in_proj_std)
            if module.v_bias is not None:
                nn.init.normal_(module.v_bias, std=in_proj_std)
        elif isinstance(module, ImageBindMLP):
            factor = self.config.initializer_factor
            in_proj_std = (
                (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            )
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, ImageBindModel):
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, ImageBindVisionModelWithProjection):
            nn.init.normal_(
                module.visual_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, ImageBindTextModelWithProjection):
            nn.init.normal_(
                module.text_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )

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

# TODO: add inputs doctrings for remaining modalities (audio, depth, thermal, IMU)
IMAGEBIND_AUDIO_INPUTS_DOCSTRING = r"""
    Args:
        TODO
"""

IMAGEBIND_DEPTH_INPUTS_DOCSTRING = r"""
    Args:
        TODO
"""

IMAGEBIND_THERMAL_INPUTS_DOCSTRING = r"""
    Args:
        TODO
"""

IMAGEBIND_IMU_INPUTS_DOCSTRING = r"""
    Args:
        TODO
"""

# TODO: update inputs docstring with remaining modalities
IMAGEBIND_INPUTS_DOCSTRING = r"""
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
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`ImageBindImageProcessor.__call__`] for details.
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

    def __init__(self, config: ImageBindConfig, drop_path_type: str = "progressive"):
        super().__init__()
        self.config = config

        if drop_path_type == "progressive":
            drop_path_rates = [prob.item() for prob in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        elif drop_path_type == "uniform":
            drop_path_rates = [config.drop_path_rate for _ in range(config.num_hidden_layers)]
        else:
            raise ValueError(
                f"`drop_path_type` is expected to be in `['uniform', 'progressive']` but got {drop_path_type}"
            )
        
        self.layers = nn.ModuleList(
            [ImageBindEncoderLayer(config, drop_path_rate) for drop_path_rate in drop_path_rates]
        )

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
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
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

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
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
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


# TODO: copied from CLIP?
class ImageBindTextTransformer(nn.Module):
    def __init__(self, config: ImageBindTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = ImageBindTextEmbeddings(config)
        self.encoder = ImageBindEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

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

        bsz, seq_len = input_shape
        # ImageBind's text model uses causal mask, prepare it here.
        # https://github.com/facebookresearch/ImageBind/blob/95d27c7fd5a8362f3527e176c3a80ae5a4d880c0/imagebind/models/imagebind_model.py#L172
        causal_attention_mask = self._build_causal_attention_mask(
            bsz, seq_len, hidden_states.dtype, device=hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:] + (None,)

        return ImageBindTransformerOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            num_clips=None,
        )

    def _build_causal_attention_mask(self, bsz, seq_len, dtype, device=None):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype, device=device)
        mask.fill_(torch.finfo(dtype).min)
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask


# TODO: copied from CLIP?
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

        >>> model = ImageBindTextModel.from_pretrained("facebook/imagebind-huge")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/imagebind-huge")

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


# TODO: copied from CLIP?
class ImageBindVisionTransformer(nn.Module):
    def __init__(self, config: ImageBindVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = ImageBindVisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = ImageBindEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

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
        
        num_clips = None
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
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:] + (num_clips,)

        return ImageBindTransformerOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            num_clips=num_clips,
        )


# TODO: copied from CLIP?
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

        >>> model = ImageBindVisionModel.from_pretrained("facebook/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("facebook/imagebind-huge")

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


# TODO: copied from CLIP?
class ImageBindAudioTransformer(nn.Module):
    def __init__(self, config: ImageBindAudioConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = ImageBindAudioEmbeddings(config)
        self.encoder = ImageBindEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

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
        
        num_clips = None
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
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:] + (num_clips,)

        return ImageBindTransformerOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            num_clips=num_clips,
        )


@add_start_docstrings(
    """The vision model from ImageBind without any head or projection on top.""",
    IMAGEBIND_START_DOCSTRING,
)
class ImageBindAudioModel(ImageBindPreTrainedModel):
    config = ImageBindAudioConfig
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
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ImageBindAudioModel

        >>> model = ImageBindAudioModel.from_pretrained("facebook/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("facebook/imagebind-huge")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

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


# TODO: copied from CLIP?
class ImageBindDepthTransformer(nn.Module):
    def __init__(self, config: ImageBindDepthConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = ImageBindDepthEmbeddings(config)
        self.encoder = ImageBindEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(IMAGEBIND_DEPTH_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindTransformerOutput, config_class=ImageBindDepthConfig)
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

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:] + (None,)

        return ImageBindTransformerOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            num_clips=None,
        )


@add_start_docstrings(
    """The depth model from ImageBind without any head or projection on top.""",
    IMAGEBIND_START_DOCSTRING,
)
class ImageBindDepthModel(ImageBindPreTrainedModel):
    config = ImageBindDepthConfig
    _no_split_modules = ["ImageBindEncoderLayer"]

    main_input_name = "pixel_values"  # TODO: rename to something better?

    def __init__(self, config: ImageBindDepthConfig):
        super().__init__(config)
        self.depth_model = ImageBindDepthTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self) -> nn.Module:
        return self.depth_model.embeddings.patch_embedding
    
    @add_start_docstrings_to_model_forward(IMAGEBIND_DEPTH_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindTransformerOutput, config_class=ImageBindDepthConfig)
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
        >>> from transformers import AutoProcessor, ImageBindDepthModel

        >>> model = ImageBindDepthModel.from_pretrained("facebook/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("facebook/imagebind-huge")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.depth_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# TODO: copied from CLIP?
class ImageBindThermalTransformer(nn.Module):
    def __init__(self, config: ImageBindThermalConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = ImageBindThermalEmbeddings(config)
        self.encoder = ImageBindEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(IMAGEBIND_THERMAL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindTransformerOutput, config_class=ImageBindThermalConfig)
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

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:] + (None,)

        return ImageBindTransformerOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            num_clips=None,
        )


@add_start_docstrings(
    """The thermal model from ImageBind without any head or projection on top.""",
    IMAGEBIND_START_DOCSTRING,
)
class ImageBindThermalModel(ImageBindPreTrainedModel):
    config = ImageBindThermalConfig
    _no_split_modules = ["ImageBindEncoderLayer"]

    main_input_name = "pixel_values"  # TODO: rename to something better?

    def __init__(self, config: ImageBindThermalConfig):
        super().__init__(config)
        self.thermal_model = ImageBindThermalTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self) -> nn.Module:
        return self.thermal_model.embeddings.patch_embedding
    
    @add_start_docstrings_to_model_forward(IMAGEBIND_THERMAL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=ImageBindThermalConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ImageBindThermalModel

        >>> model = ImageBindThermalModel.from_pretrained("facebook/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("facebook/imagebind-huge")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.thermal_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# TODO: copied from CLIP?
class ImageBindImuTransformer(nn.Module):
    def __init__(self, config: ImageBindImuConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = ImageBindImuEmbeddings(config)
        self.encoder = ImageBindEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.post_dropout = nn.Dropout(p=config.final_dropout)

    @add_start_docstrings_to_model_forward(IMAGEBIND_IMU_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindTransformerOutput, config_class=ImageBindImuConfig)
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

        hidden_states = self.embeddings(input_features)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)
        pooled_output = self.post_dropout(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:] + (None,)

        return ImageBindTransformerOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            num_clips=None,
        )


@add_start_docstrings(
    """The IMU model from ImageBind without any head or projection on top.""",
    IMAGEBIND_START_DOCSTRING,
)
class ImageBindImuModel(ImageBindPreTrainedModel):
    config = ImageBindImuConfig
    _no_split_modules = ["ImageBindEncoderLayer"]

    main_input_name = "input_features"

    def __init__(self, config: ImageBindImuConfig):
        super().__init__(config)
        self.imu_model = ImageBindImuTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self) -> nn.Module:
        return self.imu_model.embeddings.patch_embedding
    
    @add_start_docstrings_to_model_forward(IMAGEBIND_IMU_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindTransformerOutput, config_class=ImageBindImuConfig)
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
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ImageBindImuModel

        >>> model = ImageBindImuModel.from_pretrained("facebook/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("facebook/imagebind-huge")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.imu_model(
            input_features=input_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


@add_start_docstrings(IMAGEBIND_START_DOCSTRING)
class ImageBindModel(ImageBindPreTrainedModel):
    config_class = ImageBindConfig

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
        
        if not isinstance(config.depth_config, ImageBindDepthConfig):
            raise ValueError(
                "config.depth_config is expected to be of type ImageBindDepthConfig but is of type"
                f" {type(config.depth_config)}."
            )
        
        if not isinstance(config.thermal_config, ImageBindThermalConfig):
            raise ValueError(
                "config.thermal_config is expected to be of type ImageBindThermalConfig but is of type"
                f" {type(config.thermal_config)}."
            )
        
        if not isinstance(config.imu_config, ImageBindImuConfig):
            raise ValueError(
                "config.imu_config is expected to be of type ImageBindImuConfig but is of type"
                f" {type(config.imu_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config
        audio_config = config.audio_config
        depth_config = config.depth_config
        thermal_config = config.thermal_config
        imu_config = config.imu_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        self.audio_embed_dim = audio_config.hidden_size
        self.depth_embed_dim = depth_config.hidden_size
        self.thermal_embed_dim = thermal_config.hidden_size
        self.imu_embed_dim = imu_config.hidden_size

        self.text_model = ImageBindTextTransformer(text_config)
        self.vision_model = ImageBindVisionTransformer(vision_config)
        self.audio_model = ImageBindAudioTransformer(audio_config)
        self.depth_model = ImageBindDepthTransformer(depth_config)
        self.thermal_model = ImageBindThermalTransformer(thermal_config)
        self.imu_model = ImageBindImuTransformer(imu_config)

        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.audio_projection = nn.Linear(self.audio_embed_dim, self.projection_dim, bias=False)
        self.depth_projection = nn.Linear(self.depth_embed_dim, self.projection_dim, bias=False)
        self.thermal_projection = nn.Linear(self.thermal_embed_dim, self.projection_dim, bias=False)
        self.imu_projection = nn.Linear(self.imu_embed_dim, self.projection_dim, bias=False)

        self.text_postprocessor = ImageBindPostProcessor(text_config)
        self.vision_postprocessor = ImageBindPostProcessor(vision_config)
        self.audio_postprocessor = ImageBindPostProcessor(audio_config)
        self.depth_postprocessor = ImageBindPostProcessor(depth_config)
        self.thermal_postprocessor = ImageBindPostProcessor(thermal_config)
        self.imu_postprocessor = ImageBindPostProcessor(imu_config)

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

        >>> model = ImageBindModel.from_pretrained("facebook/imagebind-huge")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/imagebind-huge")

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

        >>> model = ImageBindModel.from_pretrained("facebook/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("facebook/imagebind-huge")

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
        image_features = self.visual_projection(pooled_output)

        num_clips = vision_outputs[-1]
        if num_clips is not None:
            image_features = image_features.reshape(batch_size, num_clips, -1)
            # Take mean over all clips
            image_features = image_features.mean(dim=1)

        return image_features
    
    # TODO: make sure inputs match with ImageBindAudioModel
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
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ImageBindModel

        >>> model = ImageBindModel.from_pretrained("facebook/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("facebook/imagebind-huge")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

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

        num_clips = audio_outputs[-1]
        if num_clips is not None:
            audio_features = audio_features.reshape(batch_size, num_clips, -1)
            # Take mean over all clips
            audio_features = audio_features.mean(dim=1)

        return audio_features

    # TODO: make sure inputs match with ImageBindDepthModel
    @add_start_docstrings_to_model_forward(IMAGEBIND_DEPTH_INPUTS_DOCSTRING)
    def get_depth_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            depth_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The depth embeddings obtained by
            applying the projection layer to the pooled output of [`ImageBindDepthModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ImageBindModel

        >>> model = ImageBindModel.from_pretrained("facebook/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("facebook/imagebind-huge")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> depth_features = model.get_depth_features(**inputs)
        ```"""
        # Use ImageBind model's config for some fields (if specified) instead of those in the depth component.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        depth_outputs = self.depth_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = depth_outputs[1]  # pooled_output
        depth_features = self.depth_projection(pooled_output)

        return depth_features

    # TODO: make sure inputs match with ImageBindThermalModel
    @add_start_docstrings_to_model_forward(IMAGEBIND_THERMAL_INPUTS_DOCSTRING)
    def get_thermal_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            thermal_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The thermal embeddings obtained by
            applying the projection layer to the pooled output of [`ImageBindThermalModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ImageBindModel

        >>> model = ImageBindModel.from_pretrained("facebook/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("facebook/imagebind-huge")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> thermal_features = model.get_thermal_features(**inputs)
        ```"""
        # Use ImageBind model's config for some fields (if specified) instead of those in the thermal component.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        thermal_outputs = self.thermal_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = thermal_outputs[1]  # pooled_output
        thermal_features = self.thermal_projection(pooled_output)

        return thermal_features

    # TODO: make sure inputs match with ImageBindImuModel
    @add_start_docstrings_to_model_forward(IMAGEBIND_IMU_INPUTS_DOCSTRING)
    def get_imu_features(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            imu_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The IMU embeddings obtained by
            applying the projection layer to the pooled output of [`ImageBindImuModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ImageBindModel

        >>> model = ImageBindModel.from_pretrained("facebook/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("facebook/imagebind-huge")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> imu_features = model.get_imu_features(**inputs)
        ```"""
        # Use ImageBind model's config for some fields (if specified) instead of those in the IMU component.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        imu_outputs = self.imu_model(
            input_features=input_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = imu_outputs[1]  # pooled_output
        imu_features = self.imu_projection(pooled_output)

        return imu_features

    @add_start_docstrings_to_model_forward(IMAGEBIND_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindOutput, config_class=ImageBindConfig)
    def forward(
        self,
        input_features: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        modality: Optional[str] = None,
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

        >>> model = ImageBindModel.from_pretrained("facebook/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("facebook/imagebind-huge")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use ImageBind model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_batch_size = pixel_values.shape[0]
        other_batch_size = input_features.shape[0]

        other_model, other_projection, other_postprocessor = self._resolve_modality_models(modality)

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if modality == "text":
            other_outputs = other_model(
                input_ids=input_features,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            other_outputs = other_model(
                input_ids=input_features,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        other_embeds = other_outputs[1]
        other_embeds = other_projection(other_embeds)

        # normalized features: postprocessor performs normalization and logit scaling
        image_embeds = self.vision_postprocessor(image_embeds)
        other_embeds = other_postprocessor(other_embeds)

        # If modality input was batched and clipped, reduce embedding over clips dimension
        image_num_clips = vision_outputs[-1]
        if image_num_clips is not None:
            image_embeds = image_embeds.reshape(image_batch_size, image_num_clips, -1)
            # Take mean over all clips
            image_embeds = image_embeds.mean(dim=1)
        other_num_clips = other_outputs[-1]
        if other_num_clips is not None:
            other_embeds = other_embeds.reshape(other_batch_size, other_num_clips, -1)
            other_embeds = other_embeds.mean(dim=1)

        # cosine similarity as logits
        logits_per_other = torch.matmul(other_embeds, image_embeds.t())
        logits_per_image = logits_per_other.t()

        loss = None
        if return_loss:
            loss = imagebind_loss(logits_per_other)

        if not return_dict:
            output = (logits_per_image, logits_per_other, other_embeds, image_embeds, other_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output
        
        output_kwargs = self._resolve_output_keys(modality, logits_per_other, other_embeds, other_outputs)

        return ImageBindOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            image_embeds=image_embeds,
            vision_model_output=vision_outputs,
            **output_kwargs,
        )
    
    def _resolve_modality_models(self, modality: str):
        if modality == "text":
            model = self.text_model
            projection = self.text_projection
            postprocessor = self.text_postprocessor
        elif modality == "vision":
            model = self.vision_model
            projection = self.visual_projection
            postprocessor = self.vision_postprocessor
        elif modality == "audio":
            model = self.audio_model
            projection = self.audio_projection
            postprocessor = self.audio_postprocessor
        elif modality == "depth":
            model = self.depth_model
            projection = self.depth_projection
            postprocessor = self.depth_postprocessor
        elif modality == "thermal":
            model = self.thermal_model
            projection = self.thermal_projection
            postprocessor = self.thermal_postprocessor
        elif modality == "imu":
            model = self.imu_model
            projection = self.imu_projection
            postprocessor = self.imu_postprocessor
        else:
            raise ValueError(
                f"`modality` is expected to be in `['text', 'vision', 'audio', 'depth', 'thermal', 'imu']` but got"
                f" {modality}"
            )
        return model, projection, postprocessor
    
    def _resolve_output_keys(self, modality: str, logits, embeds, model_outputs):
        output_kwargs = {}
        if modality == "vision":
            # Different naming pattern
            output_kwargs["logits_per_image"] = logits
            output_kwargs["image_embeds"] = embeds
            output_kwargs["vision_model_output"] = model_outputs
        else:
            output_kwargs[f"logits_per_{modality}"] = logits
            output_kwargs[f"{modality}_embeds"] = embeds
            output_kwargs[f"{modality}_model_output"] = model_outputs
        return output_kwargs


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

        >>> model = ImageBindTextModelWithProjection.from_pretrained("facebook/imagebind-huge")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/imagebind-huge")

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
            # Exclude num_clips output
            outputs = (text_embeds, text_outputs[0]) + text_outputs[2:-1] + (normalized_text_embeds,)
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

        self.visual_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

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

        >>> model = ImageBindVisionModelWithProjection.from_pretrained("facebook/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("facebook/imagebind-huge")

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

        image_embeds = self.visual_projection(pooled_output)
        normalized_image_embeds = self.vision_postprocessor(image_embeds)

        num_clips = vision_outputs[-1]
        if num_clips is not None:
            image_embeds = image_embeds.reshape(batch_size, num_clips, -1)
            # Take mean over all clips
            image_embeds = image_embeds.mean(dim=1)

            normalized_image_embeds = normalized_image_embeds.reshape(batch_size, num_clips, -1)
            normalized_image_embeds = normalized_image_embeds.mean(dim=1)

        if not return_dict:
            # Exclude num_clips output
            outputs = (image_embeds, vision_outputs[0]) + vision_outputs[2:-1] + (normalized_image_embeds,)
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
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ImageBindAudioModelWithProjection

        >>> model = ImageBindAudioModelWithProjection.from_pretrained("facebook/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("facebook/imagebind-huge")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")  # TODO

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

        num_clips = audio_outputs[-1]
        if num_clips is not None:
            audio_embeds = audio_embeds.reshape(batch_size, num_clips, -1)
            # Take mean over all clips
            audio_embeds = audio_embeds.mean(dim=1)

            normalized_audio_embeds = normalized_audio_embeds.reshape(batch_size, num_clips, -1)
            normalized_audio_embeds = normalized_audio_embeds.mean(dim=1)

        if not return_dict:
            # Exclude num_clips output
            outputs = (audio_embeds, audio_outputs[0]) + audio_outputs[2:-1] + (normalized_audio_embeds,)
            return tuple(output for output in outputs if output is not None)

        return ImageBindAudioModelOutput(
            audio_embeds=audio_embeds,
            last_hidden_state=audio_outputs.last_hidden_state,
            hidden_states=audio_outputs.hidden_states,
            attentions=audio_outputs.attentions,
            normalized_audio_embeds=normalized_audio_embeds,
        )


@add_start_docstrings(
    """
    ImageBind Depth Model with a projection layer on top (a linear layer on top of the pooled output).
    """,
    IMAGEBIND_START_DOCSTRING,
)
class ImageBindDepthModelWithProjection(ImageBindPreTrainedModel):
    config_class = ImageBindDepthConfig
    main_input_name = "pixel_values"  # TODO: rename to something better?

    def __init__(self, config: ImageBindDepthConfig):
        super().__init__(config)

        self.depth_model = ImageBindDepthTransformer(config)

        self.depth_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        self.depth_postprocessor = ImageBindPostProcessor(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.depth_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(IMAGEBIND_DEPTH_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindDepthModelOutput, config_class=ImageBindDepthConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageBindDepthModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ImageBindDepthModelWithProjection

        >>> model = ImageBindDepthModelWithProjection.from_pretrained("facebook/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("facebook/imagebind-huge")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")  # TODO

        >>> outputs = model(**inputs)
        >>> depth_embeds = outputs.depth_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        depth_outputs = self.depth_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = depth_outputs[1]  # pooled_output

        depth_embeds = self.depth_projection(pooled_output)
        normalized_depth_embeds = self.depth_postprocessor(depth_embeds)

        if not return_dict:
            # Exclude num_clips output
            outputs = (depth_embeds, depth_outputs[0]) + depth_outputs[2:-1] + (normalized_depth_embeds,)
            return tuple(output for output in outputs if output is not None)

        return ImageBindDepthModelOutput(
            depth_embeds=depth_embeds,
            last_hidden_state=depth_outputs.last_hidden_state,
            hidden_states=depth_outputs.hidden_states,
            attentions=depth_outputs.attentions,
            normalized_depth_embeds=normalized_depth_embeds,
        )


@add_start_docstrings(
    """
    ImageBind Thermal Model with a projection layer on top (a linear layer on top of the pooled output).
    """,
    IMAGEBIND_START_DOCSTRING,
)
class ImageBindThermalModelWithProjection(ImageBindPreTrainedModel):
    config_class = ImageBindThermalConfig
    main_input_name = "pixel_values"  # TODO: rename to something better?

    def __init__(self, config: ImageBindThermalConfig):
        super().__init__(config)

        self.thermal_model = ImageBindThermalTransformer(config)

        self.thermal_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        self.thermal_postprocessor = ImageBindPostProcessor(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.thermal_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(IMAGEBIND_THERMAL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindThermalModelOutput, config_class=ImageBindThermalConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageBindThermalModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ImageBindDepthModelWithProjection

        >>> model = ImageBindDepthModelWithProjection.from_pretrained("facebook/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("facebook/imagebind-huge")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")  # TODO

        >>> outputs = model(**inputs)
        >>> depth_embeds = outputs.depth_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        thermal_outputs = self.thermal_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = thermal_outputs[1]  # pooled_output

        thermal_embeds = self.thermal_projection(pooled_output)
        normalized_thermal_embeds = self.thermal_postprocessor(thermal_embeds)

        if not return_dict:
            # Exclude num_clips output
            outputs = (thermal_embeds, thermal_outputs[0]) + thermal_outputs[2:-1] + (normalized_thermal_embeds,)
            return tuple(output for output in outputs if output is not None)

        return ImageBindThermalModelOutput(
            thermal_embeds=thermal_embeds,
            last_hidden_state=thermal_outputs.last_hidden_state,
            hidden_states=thermal_outputs.hidden_states,
            attentions=thermal_outputs.attentions,
            normalized_thermal_embeds=normalized_thermal_embeds,
        )


@add_start_docstrings(
    """
    ImageBind IMU Model with a projection layer on top (a linear layer on top of the pooled output).
    """,
    IMAGEBIND_START_DOCSTRING,
)
class ImageBindImuModelWithProjection(ImageBindPreTrainedModel):
    config_class = ImageBindImuConfig
    main_input_name = "input_features"

    def __init__(self, config: ImageBindImuConfig):
        super().__init__(config)

        self.imu_model = ImageBindImuTransformer(config)

        self.imu_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        self.imu_postprocessor = ImageBindPostProcessor(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.imu_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(IMAGEBIND_IMU_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageBindImuModelOutput, config_class=ImageBindImuConfig)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageBindImuModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ImageBindDepthModelWithProjection

        >>> model = ImageBindDepthModelWithProjection.from_pretrained("facebook/imagebind-huge")
        >>> processor = AutoProcessor.from_pretrained("facebook/imagebind-huge")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")  # TODO

        >>> outputs = model(**inputs)
        >>> depth_embeds = outputs.depth_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        imu_outputs = self.imu_model(
            input_features=input_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = imu_outputs[1]  # pooled_output

        imu_embeds = self.imu_projection(pooled_output)
        normalized_imu_embeds = self.imu_postprocessor(imu_embeds)

        if not return_dict:
            # Exclude num_clips output
            outputs = (imu_embeds, imu_outputs[0]) + imu_outputs[2:-1] + (normalized_imu_embeds,)
            return tuple(output for output in outputs if output is not None)

        return ImageBindImuModelOutput(
            imu_embeds=imu_embeds,
            last_hidden_state=imu_outputs.last_hidden_state,
            hidden_states=imu_outputs.hidden_states,
            attentions=imu_outputs.attentions,
            normalized_imu_embeds=normalized_imu_embeds,
        )
