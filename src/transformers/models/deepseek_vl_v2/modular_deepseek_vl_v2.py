# Copyright 2026 Deepseek AI and The HuggingFace Team. All rights reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torchvision.transforms.v2 import functional as tvF

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature, select_best_resolution
from ...image_transforms import divide_to_patches, group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
    SizeDict,
    make_flat_list_of_images,
)
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MultimodalBaseModelOutputWithPast,
    MultimodalCausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel, slice_logits_to_keep
from ...processing_utils import ImagesKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import TensorType, TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..auto import AutoConfig, AutoModel
from ..deepseek_ocr2.modeling_deepseek_ocr2 import DeepseekOcr2ForConditionalGeneration, DeepseekOcr2Model


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="deepseek-community/deepseek-vl2-small")
@strict
class DeepseekVLV2Config(PreTrainedConfig):
    r"""
    downsample_ratio (`int`, *optional*, defaults to 2):
        Side of the square of neighbouring vision patches that the projector concatenates into one token.

    Example:

    ```python
    >>> from transformers import DeepseekVLV2Config, DeepseekVLV2ForConditionalGeneration

    >>> configuration = DeepseekVLV2Config()
    >>> model = DeepseekVLV2ForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_vl_v2"
    sub_configs_defaults = {
        "vision_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="siglip_vision_model",
            # SigLIP so400m/14 at 384 px; DeepSeek-VL2 reads every patch token and never uses the pooling head.
            init_kwargs={
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "image_size": 384,
                "patch_size": 14,
                "vision_use_head": False,
            },
        ),
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="deepseek_v2"),
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 128815
    downsample_ratio: int = 2
    projector_hidden_act: str = "gelu"
    tie_word_embeddings: bool = False


class DeepseekVLV2MultiModalProjector(nn.Module):
    """
    Merges each `downsample_ratio x downsample_ratio` block of vision patches into one token (zero-padding the grid
    when its side is odd), then maps it to the text hidden size with a two-layer MLP.
    """

    def __init__(self, config: DeepseekVLV2Config):
        super().__init__()
        self.downsample_ratio = config.downsample_ratio
        merged_size = config.vision_config.hidden_size * config.downsample_ratio**2
        self.linear_1 = nn.Linear(merged_size, config.text_config.hidden_size)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, channels = vision_features.shape
        side = math.isqrt(num_patches)
        grid = vision_features.reshape(batch_size, side, side, channels)
        pad = -side % self.downsample_ratio
        if pad:
            grid = F.pad(grid, (0, 0, 0, pad, 0, pad))
        grid = grid.permute(0, 3, 1, 2)
        merged = F.unfold(grid, kernel_size=self.downsample_ratio, stride=self.downsample_ratio)
        merged = merged.permute(0, 2, 1)
        return self.linear_2(self.act(self.linear_1(merged)))


@auto_docstring
class DeepseekVLV2PreTrainedModel(PreTrainedModel):
    config: DeepseekVLV2Config
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, DeepseekVLV2Model):
            embed_std = 1 / math.sqrt(self.config.text_config.hidden_size)
            init.normal_(module.image_newline, mean=0.0, std=embed_std)
            init.normal_(module.view_separator, mean=0.0, std=embed_std)


@auto_docstring
class DeepseekVLV2Model(DeepseekOcr2Model):
    def __init__(self, config: DeepseekVLV2Config):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = DeepseekVLV2MultiModalProjector(config)
        # Appended after every row of image tokens, global and local, so the language model sees the 2D layout.
        self.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size))
        # Sits between the global view and the stitched local tiles.
        self.view_separator = nn.Parameter(torch.empty(config.text_config.hidden_size))
        self.language_model = AutoModel.from_config(config.text_config)

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_values_local: torch.Tensor,
        image_spatial_crop: torch.Tensor | list[list[int]],
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.Tensor` of shape `(num_images, 3, image_size, image_size)`):
            The global (whole-image, padded) view of every image.
        pixel_values_local (`torch.Tensor` of shape `(total_tiles, 3, image_size, image_size)`):
            The local tiles of every image, row by row, concatenated across images.
        image_spatial_crop (`torch.Tensor` or `list[list[int]]` of shape `(num_images, 2)`):
            `[width_tiles, height_tiles]` of the local grid of each image.
        """
        # The tile grid sets how many tokens each image gets, so the output shapes need it on the host.
        if isinstance(image_spatial_crop, torch.Tensor):
            image_spatial_crop = image_spatial_crop.tolist()

        global_outputs = self.vision_tower(pixel_values, **kwargs)
        global_features = self.multi_modal_projector(global_outputs.last_hidden_state)
        local_outputs = self.vision_tower(pixel_values_local, **kwargs)
        local_features = self.multi_modal_projector(local_outputs.last_hidden_state)

        num_images, tokens_per_view, hidden_size = global_features.shape
        side = math.isqrt(tokens_per_view)
        # Every image token is one row of this table, so the whole layout is a single gather.
        table = torch.cat(
            [
                global_features.reshape(-1, hidden_size),
                local_features.reshape(-1, hidden_size),
                self.image_newline.to(global_features.dtype).unsqueeze(0),
                self.view_separator.to(global_features.dtype).unsqueeze(0),
            ]
        )
        newline_row, separator_row = table.shape[0] - 2, table.shape[0] - 1
        separator = torch.full((1,), separator_row, device=table.device)

        def canvas_rows(first_row: int, width_tiles: int, height_tiles: int) -> torch.Tensor:
            # Stitch row-major tiles of `side x side` tokens into one canvas and end each canvas row with a newline.
            y = torch.arange(height_tiles * side, device=table.device).unsqueeze(1)
            x = torch.arange(width_tiles * side, device=table.device).unsqueeze(0)
            tile = (y // side) * width_tiles + x // side
            rows = first_row + tile * tokens_per_view + (y % side) * side + x % side
            return torch.cat([rows, torch.full_like(rows[:, :1], newline_row)], dim=1).flatten()

        image_rows, first_tile = [], 0
        for image_idx, (width_tiles, height_tiles) in enumerate(image_spatial_crop):
            first_local_row = (num_images + first_tile) * tokens_per_view
            image_rows.append(
                torch.cat(
                    [
                        canvas_rows(image_idx * tokens_per_view, 1, 1),
                        separator,
                        canvas_rows(first_local_row, width_tiles, height_tiles),
                    ]
                )
            )
            first_tile += width_tiles * height_tiles
        image_features = table[torch.cat(image_rows)].split([len(rows) for rows in image_rows])

        return BaseModelOutputWithPooling(
            last_hidden_state=global_outputs.last_hidden_state,
            pooler_output=list(image_features),
            hidden_states=global_outputs.hidden_states,
            attentions=global_outputs.attentions,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_local: torch.Tensor | None = None,
        image_spatial_crop: torch.Tensor | list[list[int]] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MultimodalBaseModelOutputWithPast:
        r"""
        pixel_values_local (`torch.Tensor` of shape `(total_tiles, 3, image_size, image_size)`, *optional*):
            The local tiles of every image, row by row, concatenated across images.
        image_spatial_crop (`torch.Tensor` or `list[list[int]]` of shape `(num_images, 2)`, *optional*):
            `[width_tiles, height_tiles]` of the local grid of each image.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if pixel_values is not None and mm_encoder_outputs is not None:
            raise ValueError("You cannot specify both pixel_values and mm_encoder_outputs at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values, pixel_values_local, image_spatial_crop, return_dict=True
            )

        if mm_encoder_outputs.get("image") is not None:
            image_features = mm_encoder_outputs["image"].pooler_output
            image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return MultimodalBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_outputs=mm_encoder_outputs.get("image"),
        )


@auto_docstring
class DeepseekVLV2ForConditionalGeneration(DeepseekOcr2ForConditionalGeneration):
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_values_local: torch.Tensor,
        image_spatial_crop: torch.Tensor | list[list[int]],
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.Tensor` of shape `(num_images, 3, image_size, image_size)`):
            The global (whole-image, padded) view of every image.
        pixel_values_local (`torch.Tensor` of shape `(total_tiles, 3, image_size, image_size)`):
            The local tiles of every image, row by row, concatenated across images.
        image_spatial_crop (`torch.Tensor` or `list[list[int]]` of shape `(num_images, 2)`):
            `[width_tiles, height_tiles]` of the local grid of each image.
        """
        return self.model.get_image_features(pixel_values, pixel_values_local, image_spatial_crop, **kwargs)

    def pack_image_features(self):
        raise AttributeError("Not needed for DeepseekVLV2")

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_local: torch.Tensor | None = None,
        image_spatial_crop: torch.Tensor | list[list[int]] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MultimodalCausalLMOutputWithPast:
        r"""
        pixel_values_local (`torch.Tensor` of shape `(total_tiles, 3, image_size, image_size)`, *optional*):
            The local tiles of every image, row by row, concatenated across images.
        image_spatial_crop (`torch.Tensor` or `list[list[int]]` of shape `(num_images, 2)`, *optional*):
            `[width_tiles, height_tiles]` of the local grid of each image.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices in `[0, ..., config.text_config.vocab_size]`
            or -100 (masked).
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_local=pixel_values_local,
            image_spatial_crop=image_spatial_crop,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )

        hidden_states = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
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


# The 23 tilings DeepSeek-VL2 was trained with, as [width, height] in pixels (from the checkpoints' processor config).
DEFAULT_CANDIDATE_RESOLUTIONS = [
    [384, 384], [384, 768], [768, 384], [384, 1152], [1152, 384], [384, 1536], [1536, 384], [768, 768],
    [384, 1920], [1920, 384], [384, 2304], [2304, 384], [768, 1152], [1152, 768], [384, 2688], [2688, 384],
    [384, 3072], [3072, 384], [768, 1536], [1536, 768], [384, 3456], [3456, 384], [1152, 1152],
]  # fmt: skip


class DeepseekVLV2ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    candidate_resolutions (`list[list[int]]`, *optional*):
        Allowed `[width, height]` canvases for the local tiles; per image, the one that keeps the most of the image
        while wasting the least area is chosen.
    crop_to_patches (`bool`, *optional*, defaults to `True`):
        Whether to tile each image on its best canvas. When `False`, each image gets a single local tile, which is
        what DeepSeek-VL2 does for prompts with more than two images.
    """

    candidate_resolutions: list[list[int]]
    crop_to_patches: bool


@auto_docstring
class DeepseekVLV2ImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 384, "width": 384}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    crop_to_patches = True
    candidate_resolutions = DEFAULT_CANDIDATE_RESOLUTIONS
    valid_kwargs = DeepseekVLV2ImageProcessorKwargs
    model_input_names = ["pixel_values", "pixel_values_local", "image_spatial_crop"]

    def resize_and_pad(
        self, images: "torch.Tensor", height: int, width: int, resample, background_color: list[int]
    ) -> "torch.Tensor":
        """
        Fit `images` inside `height x width` keeping their aspect ratio, then pad them centred with `background_color`.
        Sizes and offsets are rounded as in `PIL.ImageOps.pad`, which the original model was trained with; the
        floor-split `pad_to_square` moves some images by one pixel.
        """
        image_height, image_width = images.shape[-2:]
        if image_width / image_height > width / height:
            new_height, new_width = round(image_height / image_width * width), width
        else:
            new_height, new_width = height, round(image_width / image_height * height)
        images = self.resize(images, SizeDict(height=new_height, width=new_width), resample=resample)
        top, left = round((height - new_height) * 0.5), round((width - new_width) * 0.5)
        padding = [left, top, width - new_width - left, height - new_height - top]
        return tvF.pad(images, padding=padding, fill=background_color)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        size: SizeDict,
        resample,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        crop_to_patches: bool = True,
        candidate_resolutions: list[list[int]] | None = None,
        **kwargs,
    ) -> BatchFeature:
        tile = size.height
        # select_best_resolution works on (height, width); the checkpoints store [width, height].
        canvases = [(height, width) for width, height in (candidate_resolutions or self.candidate_resolutions)]
        # Padding is the mean colour, so it normalises to zero.
        background_color = [int(x * 255) for x in image_mean]
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        global_views, local_tiles, spatial_crop = {}, {}, {}
        for shape, stacked_images in grouped_images.items():
            global_view = self.resize_and_pad(stacked_images, tile, tile, resample, background_color)
            global_views[shape] = self.rescale_and_normalize(
                global_view, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )

            best_height, best_width = (tile, tile)
            if crop_to_patches:
                best_height, best_width = select_best_resolution(tuple(stacked_images.shape[-2:]), canvases)
            canvas = self.resize_and_pad(stacked_images, best_height, best_width, resample, background_color)
            tiles = torch.stack(divide_to_patches(canvas, patch_size=tile), dim=1)
            local_tiles[shape] = self.rescale_and_normalize(
                tiles, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            spatial_crop[shape] = [[best_width // tile, best_height // tile]] * stacked_images.shape[0]

        local_tiles = reorder_images(local_tiles, grouped_images_index)
        return BatchFeature(
            data={
                "pixel_values": reorder_images(global_views, grouped_images_index),
                "pixel_values_local": [image_tile for image_tiles in local_tiles for image_tile in image_tiles],
                "image_spatial_crop": reorder_images(spatial_crop, grouped_images_index),
            },
            tensor_type=return_tensors,
        )


@auto_docstring
class DeepseekVLV2Processor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, patch_size=14, downsample_ratio=2):
        r"""
        patch_size (`int`, *optional*, defaults to 14):
            Patch size of the SigLIP vision encoder.
        downsample_ratio (`int`, *optional*, defaults to 2):
            Side of the square of vision patches the projector merges into one image token.
        """
        self.image_token = getattr(tokenizer, "image_token", "<image>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def prepare_inputs_layout(self, images=None, text=None, videos=None, audio=None, **kwargs):
        images, text, videos, audio = super().prepare_inputs_layout(
            images=images, text=text, videos=videos, audio=audio, **kwargs
        )
        if images is not None:
            images = make_flat_list_of_images(images)
        return images, text, videos, audio

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> TextInput:
        tile = (kwargs.get("size") or self.image_processor.size)["height"]
        side = math.ceil((tile // self.patch_size) / self.downsample_ratio)
        width_tiles, height_tiles = (int(n) for n in image_inputs["image_spatial_crop"][image_idx])
        # Global rows each ending in a newline, one view separator, then the stitched local rows with newlines.
        num_tokens = side * (side + 1) + 1 + (height_tiles * side) * (width_tiles * side + 1)
        return self.image_token * num_tokens


__all__ = [
    "DeepseekVLV2Config",
    "DeepseekVLV2PreTrainedModel",
    "DeepseekVLV2Model",
    "DeepseekVLV2ForConditionalGeneration",
    "DeepseekVLV2ImageProcessor",
    "DeepseekVLV2Processor",
]
