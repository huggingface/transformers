# Copyright 2026 Xiaomi Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Xiaomi-Lens-0 model."""

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...generation.utils import GenerateDecoderOnlyOutput
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import slice_logits_to_keep
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from ..qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5PreTrainedModel,
    Qwen3_5TextModel,
    Qwen3_5VisionModel,
)
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from ..sam.modeling_sam import SamLayerNorm
from ..siglip.modeling_siglip import SiglipAttention, SiglipEncoderLayer, SiglipMLP


@auto_docstring(checkpoint="SeerRay-Lab/Xiaomi-Lens-0")
@strict
class XiaomiLens0TextConfig(Qwen3_5TextConfig):
    model_type = "xiaomi_lens0_text"

    vocab_size: int = 248078
    hidden_size: int = 2560
    intermediate_size: int = 9216
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = True


@auto_docstring(checkpoint="SeerRay-Lab/Xiaomi-Lens-0")
@strict
class XiaomiLens0VisionConfig(Qwen3_5VisionConfig):
    r"""
    out_hidden_size (`int`, *optional*, defaults to 2560):
        The output hidden size of the vision model.
    num_position_embeddings (`int`, *optional*, defaults to 2304):
        The maximum sequence length that this model might ever be used with
    """

    model_type = "xiaomi_lens0_vision"

    depth: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    out_hidden_size: int = 2560


@auto_docstring(checkpoint="SeerRay-Lab/Xiaomi-Lens-0")
@strict
class XiaomiLens0MaskDecoderConfig(PreTrainedConfig):
    r"""
    vision_hidden_size (`int`, *optional*, defaults to 1024):
        Dimensionality of the vision encoder patch features, taken before the spatial merger.
    upscale_channels (`int`, *optional*, defaults to 256):
        Number of channels the patch features are reduced to before upscaling.
    mask_embed_dim (`int`, *optional*, defaults to 32):
        Dimensionality of the per-pixel embeddings and of the `<|mask|>` token embeddings they are matched against.
    """

    model_type = "xiaomi_lens0_mask_decoder"
    base_config_key = "mask_decoder_config"

    hidden_size: int = 2560
    intermediate_size: int = 9216
    num_attention_heads: int = 16
    vision_hidden_size: int = 1024
    upscale_channels: int = 256
    mask_embed_dim: int = 32
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02


@auto_docstring(checkpoint="SeerRay-Lab/Xiaomi-Lens-0")
@strict
class XiaomiLens0Config(Qwen3_5Config):
    r"""
    mask_decoder_config (`Union[XiaomiLens0MaskDecoderConfig, dict]`, *optional*):
        The config object or dictionary of the mask decoder.
    mask_token_id (`int`, *optional*, defaults to 248077):
        The id of the `<|mask|>` token. Every `<|mask|>` token is decoded into a segmentation mask.
    think_start_token_id (`int`, *optional*, defaults to 248068):
        The ID of `<think>`, used to filter reasoning masks before decoding.
    think_end_token_id (`int`, *optional*, defaults to 248069):
        The ID of `</think>`, used to filter reasoning masks before decoding.
    mask_loss_weight (`float`, *optional*, defaults to 1.0):
        Weight of the combined mask loss.
    mask_dice_loss_weight (`float`, *optional*, defaults to 1.0):
        Weight of the Dice term in the mask loss.
    mask_bce_loss_weight (`float`, *optional*, defaults to 2.0):
        Weight of the pixel-wise binary cross-entropy term in the mask loss.
    mask_loss_chunk_size (`int`, *optional*, defaults to 8):
        Number of masks processed per chunk for each image when computing the mask loss.

    Example:

    ```python
    >>> from transformers import XiaomiLens0ForConditionalGeneration, XiaomiLens0Config

    >>> # Initializing a Xiaomi-Lens-0 style configuration
    >>> configuration = XiaomiLens0Config()

    >>> # Initializing a model from the Xiaomi-Lens-0 style configuration
    >>> model = XiaomiLens0ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    sub_configs_defaults = {
        "vision_config": XiaomiLens0VisionConfig,
        "text_config": XiaomiLens0TextConfig,
        "mask_decoder_config": XiaomiLens0MaskDecoderConfig,
    }

    mask_decoder_config: dict | PreTrainedConfig | None = None
    mask_token_id: int = 248077
    think_start_token_id: int = 248068
    think_end_token_id: int = 248069
    tie_word_embeddings: bool = True
    mask_loss_weight: float = 1.0
    mask_dice_loss_weight: float = 1.0
    mask_bce_loss_weight: float = 2.0
    mask_loss_chunk_size: int = 8

    def __post_init__(self, **kwargs):
        raise AttributeError("Not needed for Xiaomi-Lens-0")


@auto_docstring
class XiaomiLens0PreTrainedModel(Qwen3_5PreTrainedModel):
    _no_split_modules = ["XiaomiLens0DecoderLayer", "XiaomiLens0VisionBlock", "XiaomiLens0MaskDecoder"]


@auto_docstring
class XiaomiLens0VisionModel(Qwen3_5VisionModel):
    pass


@auto_docstring
class XiaomiLens0TextModel(Qwen3_5TextModel):
    pass


@auto_docstring
@dataclass
class XiaomiLens0ModelOutputWithPast(Qwen3_5ModelOutputWithPast):
    r"""
    rope_deltas (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        The rope index difference between sequence length and multimodal rope. Deprecated in favor of
        `model.base_model.rope_deltas`; this field will be removed in v5.20.
    image_hidden_states (`torch.FloatTensor` of shape `(num_patches, vision_hidden_size)`, *optional*):
        Vision patch features before spatial merging, retaining gradients for mask supervision.
    """

    image_hidden_states: torch.FloatTensor | None = None


class XiaomiLens0Model(Qwen3_5Model):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | XiaomiLens0ModelOutputWithPast:
        if (pixel_values is not None or pixel_values_videos is not None) and mm_encoder_outputs is not None:
            raise ValueError(
                "You cannot specify both pixel_values/pixel_values_videos and mm_encoder_outputs at the same time"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **kwargs
            )

        if mm_encoder_outputs.get("video") is None and pixel_values_videos is not None:
            mm_encoder_outputs["video"] = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True, **kwargs
            )

        if mm_encoder_outputs.get("image") is not None:
            image_embeds = torch.cat(mm_encoder_outputs["image"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if mm_encoder_outputs.get("video") is not None:
            video_embeds = torch.cat(mm_encoder_outputs["video"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return XiaomiLens0ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            image_hidden_states=(
                mm_encoder_outputs["image"].last_hidden_state if mm_encoder_outputs.get("image") is not None else None
            ),
        )


class XiaomiLens0MaskDecoderAttention(SiglipAttention):
    def __init__(self, config: XiaomiLens0MaskDecoderConfig):
        super().__init__(config)
        self.num_key_value_groups = 1  # needed for eager attention


class XiaomiLens0MaskDecoderMLP(SiglipMLP):
    pass


class XiaomiLens0MaskDecoderLayer(SiglipEncoderLayer):
    def __init__(self, config: XiaomiLens0MaskDecoderConfig):
        super().__init__(config)
        self.self_attn = XiaomiLens0MaskDecoderAttention(config)
        self.mlp = XiaomiLens0MaskDecoderMLP(config)


class XiaomiLens0MaskEmbedding(SiglipMLP):
    def __init__(self, config: XiaomiLens0MaskDecoderConfig):
        super().__init__(config)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.mask_embed_dim)


class XiaomiLens0LayerNorm(SamLayerNorm):
    pass


class XiaomiLens0MaskUpscaleBlock(nn.Module):
    def __init__(self, config: XiaomiLens0MaskDecoderConfig):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(config.upscale_channels, config.upscale_channels // 2, kernel_size=2, stride=2)
        self.layer_norm = XiaomiLens0LayerNorm(config.upscale_channels // 2, data_format="channels_first")
        self.activation = ACT2FN[config.hidden_act]
        self.conv2 = nn.ConvTranspose2d(config.upscale_channels // 2, config.mask_embed_dim, kernel_size=2, stride=2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.activation(self.layer_norm(hidden_states))
        return self.conv2(hidden_states)


class XiaomiLens0PixelDecoder(nn.Module):
    def __init__(self, config: XiaomiLens0MaskDecoderConfig):
        super().__init__()
        self.patch_proj = nn.Linear(config.vision_hidden_size, config.hidden_size)
        self.patch_layer = XiaomiLens0MaskDecoderLayer(config)
        self.channel_reduce = nn.Conv2d(config.hidden_size, config.upscale_channels, kernel_size=1)
        self.upscale_block = XiaomiLens0MaskUpscaleBlock(config)

    def forward(self, image_features: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        batch_size, grid_height, grid_width, _ = image_features.shape
        patch_embeddings = self.patch_proj(image_features).flatten(1, 2)
        patch_embeddings = self.patch_layer(patch_embeddings, attention_mask=None, **kwargs)
        patch_embeddings = patch_embeddings.transpose(1, 2).reshape(batch_size, -1, grid_height, grid_width)

        pixel_embeddings = self.channel_reduce(patch_embeddings)
        return self.upscale_block(pixel_embeddings)


@auto_docstring
class XiaomiLens0MaskDecoder(XiaomiLens0PreTrainedModel):
    """Decodes `<|mask|>` token hidden states into mask logits."""

    config: XiaomiLens0MaskDecoderConfig

    def __init__(self, config: XiaomiLens0MaskDecoderConfig):
        super().__init__(config)
        self.mask_embed = XiaomiLens0MaskEmbedding(config)
        self.pixel_decoder = XiaomiLens0PixelDecoder(config)

        self.post_init()

    @auto_docstring
    def forward(
        self,
        mask_hidden_states: torch.Tensor,
        image_features: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        mask_hidden_states (`torch.Tensor` of shape `(batch_size, num_masks, hidden_size)`):
            Last hidden states of the `<|mask|>` tokens.
        image_features (`torch.Tensor` of shape `(batch_size, grid_height, grid_width, vision_hidden_size)`):
            Vision encoder patch features before the spatial merger, in row-major spatial layout.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_masks, 16 * grid_height, 16 * grid_width)`: the mask logits.
        """
        batch_size = image_features.shape[0]
        mask_embeddings = self.mask_embed(mask_hidden_states)

        pixel_embeddings = self.pixel_decoder(image_features, **kwargs)

        height, width = pixel_embeddings.shape[-2:]
        mask_logits = mask_embeddings @ pixel_embeddings.flatten(2) / self.config.mask_embed_dim**0.5
        mask_logits = mask_logits.view(batch_size, -1, height, width)
        # The fixed 4x upsampling brings the masks to the resolution of the image fed to the vision encoder
        return F.interpolate(mask_logits, scale_factor=4, mode="bilinear", align_corners=False)


def get_preceding_image_indices(
    input_ids: torch.LongTensor, image_token_id: int
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Map each token to its preceding image.

    Returns:
        `tuple[torch.LongTensor, torch.LongTensor]`: `image_indices` of shape `(batch_size, sequence_length)`
        contains image indices across the batch, or `-1` for tokens before any image in their sequence.
        `images_per_sequence` of shape `(batch_size,)` contains the number of images in each sequence.
    """
    is_image_token = input_ids == image_token_id
    is_image_start = is_image_token & ~F.pad(is_image_token[:, :-1], (1, 0))
    images_per_sequence = is_image_start.sum(-1)
    image_indices = is_image_start.cumsum(-1) - 1
    has_preceding_image = image_indices >= 0
    image_indices += (images_per_sequence.cumsum(0) - images_per_sequence)[:, None]
    image_indices.masked_fill_(~has_preceding_image, -1)
    return image_indices, images_per_sequence


def get_non_thinking_token_mask(
    input_ids: torch.LongTensor, think_start_token_id: int, think_end_token_id: int
) -> torch.BoolTensor:
    """Returns boolean mask where all tokens outside thinking blocks are `True`."""
    token_positions = torch.arange(input_ids.shape[-1], device=input_ids.device) + 1
    latest_thinking_start_positions = ((input_ids == think_start_token_id) * token_positions).cummax(-1).values
    latest_thinking_end_positions = ((input_ids == think_end_token_id) * token_positions).cummax(-1).values
    return (latest_thinking_start_positions <= latest_thinking_end_positions) & (input_ids != think_end_token_id)


@auto_docstring
@dataclass
class XiaomiLens0GenerateOutput(GenerateDecoderOnlyOutput):
    r"""
    scores (`tuple[torch.FloatTensor]`, *optional*):
        Processed prediction scores, one tensor per generated token. Returned when `output_scores=True`.
    logits (`tuple[torch.FloatTensor]`, *optional*):
        Unprocessed language-model logits, one tensor per generated token. Returned when `output_logits=True`.
    image_hidden_states (`torch.FloatTensor` of shape `(num_patches, vision_hidden_size)`, *optional*):
        Vision patch features before spatial merging, in original input-image order. Returned when images are
        provided and `return_dict_in_generate=True`, independently of `output_hidden_states`. These features can
        be passed to `decode_masks`. They are not expanded for multiple returned sequences.
    """

    image_hidden_states: torch.FloatTensor | None = None


@auto_docstring
@dataclass
class XiaomiLens0CausalLMOutputWithPast(Qwen3_5CausalLMOutputWithPast):
    r"""
    rope_deltas (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        The rope index difference between sequence length and multimodal rope. Deprecated in favor of
        `model.base_model.rope_deltas`; this field will be removed in v5.20.
    mask_logits (`list[torch.FloatTensor]`, *optional*):
        Mask logits grouped by image, each of shape `(num_masks, height, width)`.
    mask_token_indices (`list[torch.LongTensor]`, *optional*):
        Batch and sequence indices of the decoded mask tokens, grouped by image.
    mask_loss (`torch.FloatTensor` of shape `()`, *optional*):
        Weighted Dice and binary cross-entropy loss. The total loss is the sum of the language modeling loss and this weighted mask loss.
    """

    mask_logits: list[torch.FloatTensor] | None = None
    mask_token_indices: list[torch.LongTensor] | None = None
    mask_loss: torch.FloatTensor | None = None


@auto_docstring
class XiaomiLens0ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.mask_decoder = XiaomiLens0MaskDecoder(config.mask_decoder_config)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mask_labels: list[torch.Tensor] | None = None,
        mask_labels_valid_mask: list[torch.Tensor] | None = None,
        output_masks: bool = False,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | XiaomiLens0CausalLMOutputWithPast:
        r"""
        mask_labels (`list[torch.Tensor]`, *optional*):
            Binary mask targets grouped by image in input-image order, each of shape
            `(num_masks, height, width)`. Within each image, targets follow the order of its `<|mask|>` tokens,
            including thinking tokens. Spatial dimensions must match the decoder output (16 times the patch grid).
            Use an empty tensor for an image with no mask tokens.
        mask_labels_valid_mask (`list[torch.Tensor]`, *optional*):
            Boolean tensors matching `mask_labels`, selecting supervised pixels. An all-false mask is excluded
            from normalization and can be used for unsupervised mask tokens.
        output_masks (`bool`, *optional*, defaults to `False`):
            Whether to decode masks from a full sequence. Mask labels also enable decoding. Requires `input_ids`,
            image features and `image_grid_thw`; this is not intended for incremental cached decoding.
        """
        if mask_labels_valid_mask is not None and mask_labels is None:
            raise ValueError("mask_labels_valid_mask requires mask_labels")
        if mask_labels is not None or output_masks:
            if input_ids is None or image_grid_thw is None:
                raise ValueError("Mask decoding requires input_ids and image_grid_thw")
            if past_key_values is not None and past_key_values.get_seq_length() > 0:
                raise ValueError("Mask decoding requires a full sequence without a populated cache")

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(slice_logits_to_keep(hidden_states, logits_to_keep))

        mask_logits = mask_token_indices = None
        if mask_labels is not None or output_masks:
            mask_logits, mask_token_indices = self.decode_masks(
                input_ids, hidden_states, outputs.image_hidden_states, image_grid_thw, ignore_thinking=False
            )

        loss = mask_loss = None
        if labels is not None or mask_labels is not None:
            loss, _, mask_loss = self.loss_function(
                logits=logits,
                labels=labels,
                mask_logits=mask_logits,
                mask_labels=mask_labels,
                mask_labels_valid_mask=mask_labels_valid_mask,
                config=self.config,
                **kwargs,
            )

        return XiaomiLens0CausalLMOutputWithPast(
            loss=loss,
            mask_logits=mask_logits,
            mask_token_indices=mask_token_indices,
            mask_loss=mask_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        # Custom generate to return image_hidden_states for mask decoding.
        mm_encoder_outputs = dict(kwargs.pop("mm_encoder_outputs", None) or {})
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is not None:
            if mm_encoder_outputs.get("image") is not None:
                raise ValueError("You cannot pass both: raw pixels and pre-computed embeddings for input image")
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values, kwargs.get("image_grid_thw"), return_dict=True
            )
        if mm_encoder_outputs:
            kwargs["mm_encoder_outputs"] = mm_encoder_outputs

        image_outputs = mm_encoder_outputs.get("image")
        image_hidden_states = image_outputs.last_hidden_state if image_outputs is not None else None
        result = super().generate(*args, **kwargs)
        if isinstance(result, GenerateDecoderOnlyOutput):
            return XiaomiLens0GenerateOutput(**result, image_hidden_states=image_hidden_states)
        return result

    def decode_masks(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.FloatTensor,
        image_hidden_states: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        ignore_thinking: bool = True,
    ) -> tuple[list[torch.FloatTensor], list[torch.LongTensor]]:
        """Decode mask-token states against their preceding images, returning only decoded masks.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Full input sequences, including generated tokens when decoding generation output.
            hidden_states (`torch.FloatTensor` of shape `(batch_size, processed_length, hidden_size)`):
                Final-layer hidden states aligned with `input_ids`.
            image_hidden_states (`torch.FloatTensor` of shape `(num_patches, vision_hidden_size)`):
                Vision patch features before the spatial merger (`get_image_features().last_hidden_state`),
                ordered by image occurrence in `input_ids`.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
                Temporal, height and width dimensions for each image's patch features.
            ignore_thinking (`bool`, *optional*, defaults to `True`):
                Do not decode masks inside thinking blocks.

        Returns:
            `tuple[list[torch.FloatTensor], list[torch.LongTensor]]`: Mask logits and their token indices, with
            one entry per image in image-occurrence order across the batch. Logits have shape
            `(num_masks, height, width)` and indices have shape `(num_masks, 2)`, containing
            `(batch_index, token_position)`. Indices are returned on CPU for postprocessing. Both tensors are empty
            for images without decoded masks.
        """
        image_indices, _ = get_preceding_image_indices(input_ids, self.config.image_token_id)

        is_mask_token = input_ids == self.config.mask_token_id
        # `<|mask|>` tokens without a preceding image in their sequence are not decoded
        mask_tokens_to_decode = is_mask_token & (image_indices >= 0)
        if ignore_thinking:
            mask_tokens_to_decode &= get_non_thinking_token_mask(
                input_ids, self.config.think_start_token_id, self.config.think_end_token_id
            )
        selected_mask_batch_indices, selected_mask_token_indices = mask_tokens_to_decode.nonzero(as_tuple=True)
        selected_mask_image_indices = image_indices[selected_mask_batch_indices, selected_mask_token_indices]

        # Selected tokens are in batch/sequence order, so masks for each preceding image are contiguous.
        masks_per_image = torch.bincount(selected_mask_image_indices, minlength=image_grid_thw.shape[0]).tolist()
        if mask_tokens_to_decode[:, hidden_states.shape[1] :].any():
            # TODO: ignore instead of raise?
            raise ValueError(
                "The final <|mask|> token has no hidden state. Generate further or use a full forward pass."
            )
        # Grid dimensions are needed by Python for splitting features and reshaping each image.
        image_grids = image_grid_thw.tolist()
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        patches_per_image = [
            grid_time * grid_height * grid_width for grid_time, grid_height, grid_width in image_grids
        ]
        patch_features_per_image = torch.split(image_hidden_states, patches_per_image)
        decoded_masks_per_image = []
        mask_offset = 0
        for image_index, (num_image_masks, (_, grid_height, grid_width)) in enumerate(
            zip(masks_per_image, image_grids)
        ):
            if num_image_masks == 0:
                patch_size = self.config.vision_config.patch_size
                decoded_masks_per_image.append(
                    hidden_states.new_empty(0, grid_height * patch_size, grid_width * patch_size)
                )
                continue
            # The vision encoder orders patches by spatial merge window, restore the row-major layout
            spatial_image_features = (
                patch_features_per_image[image_index]
                .view(
                    grid_height // spatial_merge_size,
                    grid_width // spatial_merge_size,
                    spatial_merge_size,
                    spatial_merge_size,
                    -1,
                )
                .permute(0, 2, 1, 3, 4)
                .reshape(1, grid_height, grid_width, -1)
            )
            image_mask_slice = slice(mask_offset, mask_offset + num_image_masks)
            mask_offset += num_image_masks
            mask_batch_indices = selected_mask_batch_indices[image_mask_slice]
            mask_token_indices = selected_mask_token_indices[image_mask_slice]
            mask_hidden_states = hidden_states[mask_batch_indices, mask_token_indices]
            image_mask_logits = self.mask_decoder(
                mask_hidden_states[None], spatial_image_features.to(hidden_states.dtype)
            )
            decoded_masks_per_image.append(image_mask_logits[0])

        mask_token_indices = torch.stack((selected_mask_batch_indices, selected_mask_token_indices), dim=-1).cpu()
        return decoded_masks_per_image, list(mask_token_indices.split(masks_per_image))


class XiaomiLens0Processor(Qwen3VLProcessor):
    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids("<|mask|>")
        self.think_start_token_id = self.tokenizer.convert_tokens_to_ids("<think>")
        self.think_end_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        self.bbox_start_token_id = self.tokenizer.convert_tokens_to_ids("<bbox>")
        self.bbox_end_token_id = self.tokenizer.convert_tokens_to_ids("</bbox>")
        self.box_pattern = re.compile(r"\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*")

    def post_process_grounding(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: torch.LongTensor,
        mask_logits: list[torch.FloatTensor] | None = None,
        mask_token_indices: list[torch.LongTensor] | None = None,
        target_sizes: list[tuple[int, int]] | None = None,
        mask_threshold: float = 0.5,
        ignore_thinking: bool = True,
    ) -> list[dict[str, torch.Tensor]]:
        r"""
        Extract boxes and masks independently, pairing them by occurrence order within each image.
        Boxes use `<bbox>[x1, y1, x2, y2]</bbox>` with coordinates normalized to `[0, 1000]`; masks use
        `<|mask|>`. Either may appear first or without the other. The first box pairs with the first mask,
        and so on; excess boxes or masks are unpaired. Malformed boxes raise `ValueError`.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The full sequences (prompt and generated tokens) the masks were decoded from.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
                The image grids returned by the processor.
            mask_logits (`list[torch.FloatTensor]`, *optional*):
                One `(num_masks, height, width)` tensor per image, in token order as returned by `decode_masks`.
                Omit or pass `None` for box-only output with empty mask tensors.
            mask_token_indices (`list[torch.LongTensor]`, *optional*):
                One CPU `(num_masks, 2)` tensor per image containing `(batch_index, token_position)`, as returned
                by `decode_masks`, in token order matching `mask_logits`. Required when `mask_logits` is provided.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                The `(height, width)` of each image to resize the boxes and masks to. Defaults to the size of the
                resized image fed to the vision encoder.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Probability threshold to binarize the masks.
            ignore_thinking (`bool`, *optional*, defaults to `True`):
                Exclude boxes inside thinking blocks, using the same token filter as `decode_masks`.
                Use the same setting for decoding masks so both occurrence orders refer to the same text.

        Returns:
            `list[dict]`: One dictionary per image with:
                - **boxes** (`torch.FloatTensor` of shape `(num_boxes, 4)`): Boxes in absolute
                  `(top_left_x, top_left_y, bottom_right_x, bottom_right_y)` format.
                - **masks** (`torch.BoolTensor` of shape `(num_masks, height, width)`): Binary masks.
                  Boxes and masks are each in token order. Entries with the same index are paired up to
                  `min(num_boxes, num_masks)`, remaining entries have no counterpart.
        """
        patch_size = self.image_processor.patch_size
        image_grids = image_grid_thw.tolist()
        num_images = len(image_grids)

        if mask_logits is None:
            if mask_token_indices is not None:
                raise ValueError("mask_token_indices cannot be set if mask_logits is None")
            mask_logits = [torch.empty(0, height * patch_size, width * patch_size) for _, height, width in image_grids]
            mask_token_indices = [torch.empty((0, 2), dtype=torch.long) for _ in image_grids]
        if mask_token_indices is None or len(mask_logits) != num_images or len(mask_token_indices) != num_images:
            raise ValueError(
                "Expected one mask tensor and one token-index tensor per image grid. "
                f"But got {len(mask_logits)} masks, {num_images} images, and {len(mask_token_indices)} mask_token_images."
            )
        if target_sizes is not None and len(target_sizes) != num_images:
            raise ValueError(
                f"Expected one target size per image. But got {len(target_sizes)} target sizes and {num_images} images"
            )

        sequences = input_ids.tolist()
        selected_tokens = (
            get_non_thinking_token_mask(input_ids, self.think_start_token_id, self.think_end_token_id).tolist()
            if ignore_thinking
            else None
        )

        image_indices, images_per_sequence = get_preceding_image_indices(input_ids, self.image_token_id)
        image_indices = image_indices.tolist()
        image_counts = images_per_sequence.tolist()

        # Text-only processor inputs can omit image placeholders when there is one image per sequence.
        if sum(image_counts) == 0 and num_images == len(sequences):
            image_indices = [[sequence_index] * len(tokens) for sequence_index, tokens in enumerate(sequences)]
        elif sum(image_counts) != num_images:
            raise ValueError("The number of image placeholders must match the image grids.")

        boxes_per_image = [[] for _ in image_grids]
        for sequence_index, sequence_ids in enumerate(sequences):
            bbox_start_index = None
            for token_index, token_id in enumerate(sequence_ids):
                if selected_tokens is not None and not selected_tokens[sequence_index][token_index]:
                    bbox_start_index = None
                    continue

                if token_id == self.bbox_start_token_id:
                    if bbox_start_index is not None:
                        raise ValueError(f"Unclosed bbox in sequence {sequence_index}.")
                    bbox_start_index = token_index
                elif token_id == self.bbox_end_token_id:
                    if bbox_start_index is None:
                        raise ValueError(f"Missing <bbox> in sequence {sequence_index}.")
                    coordinates = self.tokenizer.decode(
                        sequence_ids[bbox_start_index + 1 : token_index], skip_special_tokens=False
                    )
                    match = self.box_pattern.fullmatch(coordinates)
                    if match is None:
                        raise ValueError(f"Invalid bbox in sequence {sequence_index} at token {bbox_start_index}.")
                    image_index = image_indices[sequence_index][bbox_start_index]
                    if image_index >= 0:
                        boxes_per_image[image_index].append([int(coordinate) for coordinate in match.groups()])
                    bbox_start_index = None
            if bbox_start_index is not None:
                raise ValueError(f"Unclosed bbox in sequence {sequence_index}.")

        results = []
        for image_index, (image_mask_logits, token_indices) in enumerate(zip(mask_logits, mask_token_indices)):
            if len(token_indices) != image_mask_logits.shape[0]:
                raise ValueError(
                    f"Image {image_index} has {len(token_indices)} mask token indices but received {image_mask_logits.shape[0]} masks."
                )
            _, grid_height, grid_width = image_grids[image_index]
            height, width = (
                target_sizes[image_index]
                if target_sizes is not None
                else (grid_height * patch_size, grid_width * patch_size)
            )
            if len(token_indices) == 0:
                image_mask_logits = image_mask_logits.new_empty(0, height, width)
            elif (height, width) != image_mask_logits.shape[-2:]:
                image_mask_logits = F.interpolate(
                    image_mask_logits[None].float(), size=(height, width), mode="bilinear", align_corners=False
                )[0]
            masks = image_mask_logits.sigmoid() >= mask_threshold
            boxes = torch.tensor(boxes_per_image[image_index], dtype=torch.float32).reshape(-1, 4)
            boxes = boxes / 1000 * torch.tensor([width, height, width, height])
            results.append({"boxes": boxes, "masks": masks})
        return results


__all__ = [
    "XiaomiLens0Config",
    "XiaomiLens0TextConfig",
    "XiaomiLens0VisionConfig",
    "XiaomiLens0MaskDecoderConfig",
    "XiaomiLens0VisionModel",
    "XiaomiLens0TextModel",
    "XiaomiLens0Model",
    "XiaomiLens0ForConditionalGeneration",
    "XiaomiLens0GenerateOutput",
    "XiaomiLens0PreTrainedModel",
    "XiaomiLens0Processor",
]
