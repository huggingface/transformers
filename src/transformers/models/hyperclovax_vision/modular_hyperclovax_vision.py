# Copyright 2026 NAVER Corp. and The HuggingFace Team. All rights reserved.
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
"""PyTorch HyperCLOVAX Vision model."""

import math

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...image_processing_backends import PilBackend, TorchvisionBackend
from ...image_processing_utils import get_patch_output_size, select_best_resolution
from ...image_transforms import divide_to_patches
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MultimodalBaseModelOutputWithPast,
    MultimodalCausalLMOutputWithPast,
)
from ...modeling_utils import slice_logits_to_keep
from ...processing_utils import MultiModalData, ProcessorMixin, Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import select_hidden_states
from ..auto import AutoConfig
from ..convnext.modeling_convnext import ConvNextLayerNorm
from ..llava_next.image_processing_llava_next import LlavaNextImageProcessor
from ..llava_next.image_processing_pil_llava_next import LlavaNextImageProcessorPil
from ..llava_next.modeling_llava_next import (
    LlavaNextForConditionalGeneration,
    LlavaNextModel,
    LlavaNextPreTrainedModel,
    get_anyres_image_grid_shape,
    image_size_to_num_patches,
    unpad_image,
)
from ..llava_next.processing_llava_next import LlavaNextProcessor


def build_image_grid_pinpoints(max_num_grids: int, grid_size: int) -> list[list[int]]:
    """Builds every `(height, width)` resolution made of at most `max_num_grids` square grids of side `grid_size`."""
    return [
        [rows * grid_size, cols * grid_size]
        for rows in range(1, max_num_grids + 1)
        for cols in range(1, max_num_grids + 1)
        if rows * cols <= max_num_grids
    ]


def get_global_view_size(image_size: tuple[int, int], size: int) -> SizeDict:
    """Returns the size of an image of size `(height, width)` resized to `size` on its longest side, keeping its aspect ratio."""
    height, width = image_size
    if height >= width:
        return SizeDict(height=size, width=math.ceil(width / height * size))
    return SizeDict(height=math.ceil(height / width * size), width=size)


@auto_docstring(checkpoint="naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B")
@strict
class HyperCLOVAXVisionConfig(PreTrainedConfig):
    r"""
    projector_num_queries (`int`, *optional*, defaults to 81):
        Number of visual tokens the C-Abstractor projector outputs for each image grid. Must be a perfect square.
    projector_hidden_act (`str`, *optional*, defaults to `"silu"`):
        The non-linear activation function used in the C-Abstractor projector.
    image_grid_pinpoints (`list`, *optional*):
        A list of possible resolutions to use for processing high resolution images. Each item in the list should be a
        tuple or list of the form `(height, width)`. Defaults to every layout of at most 9 grids of size
        `vision_config.image_size`.

    Example:

    ```python
    >>> from transformers import HyperCLOVAXVisionConfig, HyperCLOVAXVisionForConditionalGeneration

    >>> # Initializing a HyperCLOVAX Vision configuration with defaults
    >>> configuration = HyperCLOVAXVisionConfig()

    >>> # Initializing a model from the configuration
    >>> model = HyperCLOVAXVisionForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "hyperclovax_vision"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="llama"),
        "vision_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="siglip_vision_model",
            init_kwargs={
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "image_size": 378,
                "patch_size": 14,
            },
        ),
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 100271
    vision_feature_layer: int = -2
    projector_num_queries: int = 81
    projector_hidden_act: str = "silu"
    image_grid_pinpoints: list | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        # The original checkpoints use a different `model_type` and attribute names
        if kwargs.get("model_type") == "hyperclovax_vlm":
            kwargs["model_type"] = "hyperclovax_vision"
            self.vision_feature_layer = kwargs.pop("use_nth_layer", self.vision_feature_layer)
            self.projector_num_queries = kwargs.pop("num_queries_vis_abstractor_image", self.projector_num_queries)
        max_num_grids = kwargs.pop("max_num_grids", 9)

        super().__post_init__(**kwargs)

        if self.image_grid_pinpoints is None:
            self.image_grid_pinpoints = build_image_grid_pinpoints(max_num_grids, self.vision_config.image_size)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if math.isqrt(self.projector_num_queries) ** 2 != self.projector_num_queries:
            raise ValueError(f"`projector_num_queries` must be a perfect square, got {self.projector_num_queries}.")


class HyperCLOVAXVisionImageProcessor(LlavaNextImageProcessor):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"shortest_edge": 378}
    crop_size = {"height": 378, "width": 378}
    image_grid_pinpoints = build_image_grid_pinpoints(max_num_grids=9, grid_size=378)

    def __init__(self, **kwargs):
        # The original checkpoints call the grid pinpoints `possible_resolutions`
        if "possible_resolutions" in kwargs:
            kwargs.setdefault("image_grid_pinpoints", kwargs.pop("possible_resolutions"))
        TorchvisionBackend.__init__(self, **kwargs)

    def _pad_for_patching(self, image: "torch.Tensor", target_resolution: tuple) -> "torch.Tensor":
        """Pad an image to a target resolution while maintaining aspect ratio, filling with the mean color."""
        new_resolution = get_patch_output_size(image, target_resolution, input_data_format=ChannelDimension.FIRST)
        padding = self._get_padding_size(new_resolution, target_resolution)
        background_color = [int(channel_mean * 255) for channel_mean in self.image_mean]
        return tvF.pad(image, padding=padding, fill=background_color)

    def _get_image_patches(
        self,
        image: "torch.Tensor",
        grid_pinpoints: list[list[int]],
        size: tuple,
        patch_size: int,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
    ) -> list["torch.Tensor"]:
        """
        Process an image with variable resolutions by dividing it into patches. Unlike LLaVA-NeXT, the global view
        keeps the aspect ratio of the image: its longest side is resized to `size` and it is then padded to a square.
        """
        if not isinstance(grid_pinpoints, list):
            raise TypeError("grid_pinpoints must be a list of possible resolutions.")

        image_size = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        best_resolution = select_best_resolution(image_size, grid_pinpoints)
        resized_image = self._resize_for_patching(
            image, best_resolution, resample=resample, input_data_format=ChannelDimension.FIRST
        )
        padded_image = self._pad_for_patching(resized_image, best_resolution)
        patches = divide_to_patches(padded_image, patch_size=patch_size)

        global_size = max(size)
        global_image = self.resize(image, size=get_global_view_size(image_size, global_size), resample=resample)
        global_image = self._pad_for_patching(global_image, (global_size, global_size))

        return [global_image] + patches


class HyperCLOVAXVisionImageProcessorPil(LlavaNextImageProcessorPil):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"shortest_edge": 378}
    crop_size = {"height": 378, "width": 378}
    image_grid_pinpoints = build_image_grid_pinpoints(max_num_grids=9, grid_size=378)

    def __init__(self, **kwargs):
        # The original checkpoints call the grid pinpoints `possible_resolutions`
        if "possible_resolutions" in kwargs:
            kwargs.setdefault("image_grid_pinpoints", kwargs.pop("possible_resolutions"))
        PilBackend.__init__(self, **kwargs)

    def _pad_for_patching(self, image: np.ndarray, target_resolution: tuple) -> np.ndarray:
        """Pad an image to a target resolution while maintaining aspect ratio, filling with the mean color."""
        new_resolution = get_patch_output_size(image, target_resolution, input_data_format=ChannelDimension.FIRST)
        (pad_top, _), (pad_left, _) = self._get_padding_size(new_resolution, target_resolution)
        background_color = np.array([int(channel_mean * 255) for channel_mean in self.image_mean], dtype=image.dtype)
        padded_image = np.empty((image.shape[0], *target_resolution), dtype=image.dtype)
        padded_image[:] = background_color[:, None, None]
        padded_image[:, pad_top : pad_top + image.shape[1], pad_left : pad_left + image.shape[2]] = image
        return padded_image

    def get_image_patches(
        self,
        image: np.ndarray,
        grid_pinpoints: list[list[int]],
        size: tuple,
        patch_size: int,
        resample: PILImageResampling,
    ) -> list[np.ndarray]:
        """
        Process an image with variable resolutions by dividing it into patches. Unlike LLaVA-NeXT, the global view
        keeps the aspect ratio of the image: its longest side is resized to `size` and it is then padded to a square.
        """
        if not isinstance(grid_pinpoints, list):
            raise TypeError("grid_pinpoints must be a list of possible resolutions.")

        image_size = image.shape[-2:]
        best_resolution = select_best_resolution(image_size, grid_pinpoints)
        resized_image = self._resize_for_patching(image, best_resolution, resample=resample)
        padded_image = self._pad_for_patching(resized_image, best_resolution)
        patches = divide_to_patches(padded_image, patch_size=patch_size)

        global_size = max(size)
        global_image = self.resize(image, size=get_global_view_size(image_size, global_size), resample=resample)
        global_image = self._pad_for_patching(global_image, (global_size, global_size))

        return [global_image] + patches


class HyperCLOVAXVisionLayerNorm(ConvNextLayerNorm):
    pass


class HyperCLOVAXVisionSqueezeExcitation(nn.Module):
    def __init__(self, hidden_size: int, reduced_size: int, hidden_act: str):
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_size, reduced_size, kernel_size=1)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Conv2d(reduced_size, hidden_size, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scale = hidden_states.mean((2, 3), keepdim=True)
        scale = self.fc2(self.act(self.fc1(scale)))
        return hidden_states * scale.sigmoid()


class HyperCLOVAXVisionProjectorBlock(nn.Module):
    """RegNet bottleneck block with a depthwise convolution and squeeze-excitation, as used by the C-Abstractor."""

    def __init__(self, config: HyperCLOVAXVisionConfig):
        super().__init__()
        hidden_size = config.vision_config.hidden_size
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.norm1 = HyperCLOVAXVisionLayerNorm(hidden_size, eps=1e-5, data_format="channels_first")
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size, bias=False)
        self.norm2 = HyperCLOVAXVisionLayerNorm(hidden_size, eps=1e-5, data_format="channels_first")
        self.se = HyperCLOVAXVisionSqueezeExcitation(hidden_size, hidden_size // 4, config.projector_hidden_act)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.norm3 = HyperCLOVAXVisionLayerNorm(hidden_size, eps=1e-5, data_format="channels_first")
        self.act = ACT2FN[config.projector_hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.act(self.norm1(self.conv1(hidden_states)))
        hidden_states = self.act(self.norm2(self.conv2(hidden_states)))
        hidden_states = self.se(hidden_states)
        hidden_states = self.norm3(self.conv3(hidden_states))
        return self.act(hidden_states + residual)


class HyperCLOVAXVisionMultiModalProjector(nn.Module):
    """
    C-Abstractor from [Honeybee](https://huggingface.co/papers/2312.06742): convolutional blocks around an adaptive
    pooling that reduces each grid to `projector_num_queries` visual tokens, followed by an MLP.
    """

    def __init__(self, config: HyperCLOVAXVisionConfig):
        super().__init__()
        vision_hidden_size = config.vision_config.hidden_size
        text_hidden_size = config.text_config.hidden_size
        num_patches = (config.vision_config.image_size // config.vision_config.patch_size) ** 2

        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, vision_hidden_size))
        self.stage1 = nn.ModuleList([HyperCLOVAXVisionProjectorBlock(config) for _ in range(3)])
        self.pooler = nn.AdaptiveAvgPool2d(math.isqrt(config.projector_num_queries))
        self.stage2 = nn.ModuleList([HyperCLOVAXVisionProjectorBlock(config) for _ in range(3)])
        self.linear_1 = nn.Linear(vision_hidden_size, text_hidden_size)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(text_hidden_size, text_hidden_size)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, hidden_size = image_features.shape
        side = math.isqrt(num_patches)

        hidden_states = image_features + self.position_embedding
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, hidden_size, side, side)
        for block in self.stage1:
            hidden_states = block(hidden_states)
        hidden_states = self.pooler(hidden_states)
        for block in self.stage2:
            hidden_states = block(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class HyperCLOVAXVisionPreTrainedModel(LlavaNextPreTrainedModel):
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, HyperCLOVAXVisionMultiModalProjector):
            init.normal_(module.position_embedding, mean=0.0, std=0.02)


class HyperCLOVAXVisionModel(LlavaNextModel):
    def pack_image_features(self, image_features, image_sizes, image_newline=None):
        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Args:
            image_features (`list[torch.Tensor]` of length num_images, each of shape `(num_patches, image_length, embed_dim)`)
                List of image feature tensor, each contains all the visual feature of all patches.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            image_newline (`torch.Tensor` of shape `(embed_dim)`)
                New line embedding vector.
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
            feature_lens (`list[int]`)
                token length of each image in image_features
        """
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = math.isqrt(self.config.projector_num_queries)

                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None]
                            .expand(*image_feature.shape[:-1], 1)
                            .to(image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features[0].device)
        return new_image_features, feature_lens

    @merge_with_config_defaults
    @can_return_tuple
    @select_hidden_states("vision_feature_layer")
    @auto_docstring(
        custom_intro="Obtains image last hidden states from the vision tower and apply multimodal projection."
    )
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.Tensor` of shape `(batch_size, num_patches, channels, height, width)`)
            The tensors corresponding to the input images.
        image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
            Actual image size of each images (H, W).
        vision_feature_layer (`int`, *optional*):
            The index of the layer to select the vision feature.
        """
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]
        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        image_outputs = self.vision_tower(pixel_values, return_dict=True, **kwargs)
        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = torch.split(image_features, image_num_patches, dim=0)
        image_features, feature_lens = self.pack_image_features(
            image_features, image_sizes, image_newline=self.image_newline
        )
        image_outputs.pooler_output = image_features

        return image_outputs

    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_sizes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        vision_feature_layer: int | None = None,
        use_cache: bool | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MultimodalBaseModelOutputWithPast:
        r"""
        image_sizes (`torch.LongTensor` of shape `(num_images, 2)`, *optional*):
            Actual image size of each images (H, W).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and mm_encoder_outputs is not None:
            raise ValueError("You cannot specify both pixel_values and mm_encoder_outputs at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None and pixel_values.size(0) > 0:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values, image_sizes, vision_feature_layer=vision_feature_layer, return_dict=True
            )

        if mm_encoder_outputs.get("image") is not None:
            image_features = torch.cat(mm_encoder_outputs["image"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return MultimodalBaseModelOutputWithPast(**outputs, image_outputs=mm_encoder_outputs.get("image"))


class HyperCLOVAXVisionForConditionalGeneration(LlavaNextForConditionalGeneration):
    def pack_image_features(self, image_features, image_sizes, image_newline=None):
        return self.model.pack_image_features(
            image_features=image_features, image_sizes=image_sizes, image_newline=image_newline
        )

    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.Tensor` of shape `(batch_size, num_patches, channels, height, width)`)
            The tensors corresponding to the input images.
        image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
            Actual image size of each images (H, W).
        vision_feature_layer (`int`, *optional*):
            The index of the layer to select the vision feature.
        """
        return self.model.get_image_features(
            pixel_values=pixel_values, image_sizes=image_sizes, vision_feature_layer=vision_feature_layer, **kwargs
        )

    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_sizes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        vision_feature_layer: int | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MultimodalCausalLMOutputWithPast:
        r"""
        image_sizes (`torch.LongTensor` of shape `(num_images, 2)`, *optional*):
            Actual image size of each images (H, W).
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ...,
            config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoProcessor, HyperCLOVAXVisionForConditionalGeneration

        >>> model = HyperCLOVAXVisionForConditionalGeneration.from_pretrained(
        ...     "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B", device_map="auto"
        ... )
        >>> processor = AutoProcessor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B")

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
        ...             {"type": "text", "text": "Describe this image."},
        ...         ],
        ...     }
        ... ]
        >>> inputs = processor.apply_chat_template(
        ...     messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ... ).to(model.device)
        >>> output = model.generate(**inputs, max_new_tokens=50)
        >>> processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        ```"""
        outputs = self.model(
            input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            vision_feature_layer=vision_feature_layer,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            mm_encoder_outputs=mm_encoder_outputs,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs[0]
        hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return MultimodalCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_outputs=outputs.image_outputs,
        )


class HyperCLOVAXVisionProcessor(LlavaNextProcessor):
    images_kwargs = AttributeError()

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token="<|dummy3|>",
        projector_num_queries=81,
        **kwargs,
    ):
        r"""
        image_token (`str`, *optional*, defaults to `"<|dummy3|>"`):
            Special token used to denote image location.
        projector_num_queries (`int`, *optional*, defaults to 81):
            Number of visual tokens the model's projector outputs for each image grid. Should be the same as in the
            model's config.
        """
        self.projector_num_queries = projector_num_queries
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        ProcessorMixin.__init__(self, image_processor, tokenizer, chat_template=chat_template)

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        image_size = image_inputs["image_sizes"][image_idx]
        if not isinstance(image_size, (list, tuple)):
            image_size = image_size.tolist()
        orig_height, orig_width = image_size
        height, width = image_inputs["pixel_values"][0][0].shape[-2:]
        num_image_tokens = self._get_number_of_features(orig_height, orig_width, height, width)
        return self.image_token * num_image_tokens

    def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints
        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height, scale_width = height_best_resolution // height, width_best_resolution // width

        patches_height = patches_width = math.isqrt(self.projector_num_queries)
        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height, orig_width, patches_height, patches_width, scale_height, scale_width
        )
        # The global view is reduced to `projector_num_queries` tokens, without newlines
        base_features = patches_height * patches_width
        return unpadded_features + newline_features + base_features

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Args:
            image_sizes (list[list[str]], *optional*):
                The input sizes formatted as (height, width) per each image.

        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = self._merge_kwargs(**kwargs)["images_kwargs"]
            crop_size = images_kwargs.get("crop_size", None) or self.image_processor.crop_size
            grid_height, grid_width = crop_size["height"], crop_size["width"]

            batch_num_image_tokens = [
                self._get_number_of_features(orig_height, orig_width, grid_height, grid_width)
                for orig_height, orig_width in image_sizes
            ]
            num_image_patches = [1] * len(image_sizes)
            vision_data.update({"num_image_tokens": batch_num_image_tokens, "num_image_patches": num_image_patches})
        return MultiModalData(**vision_data)


__all__ = [
    "HyperCLOVAXVisionConfig",
    "HyperCLOVAXVisionForConditionalGeneration",
    "HyperCLOVAXVisionImageProcessor",
    "HyperCLOVAXVisionImageProcessorPil",
    "HyperCLOVAXVisionModel",
    "HyperCLOVAXVisionPreTrainedModel",
    "HyperCLOVAXVisionProcessor",
]
