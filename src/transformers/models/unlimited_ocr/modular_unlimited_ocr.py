# Copyright 2026 the HuggingFace Team. All rights reserved.
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
from huggingface_hub.dataclasses import strict
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from ... import initialization as init
from ...cache_utils import (
    Cache,
    DynamicCache,
    DynamicLayer,
    DynamicSlidingWindowLayer,
    StaticSlidingWindowLayer,
)
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import PILImageResampling, SizeDict
from ...masking_utils import and_masks, causal_mask_function, create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...tokenization_utils_base import TextInput
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torchdynamo_compiling,
    torch_int,
)
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..clip.configuration_clip import CLIPVisionConfig
from ..clip.modeling_clip import CLIPAttention, CLIPEncoderLayer, CLIPVisionEmbeddings, CLIPVisionModel
from ..deepseek_ocr2.configuration_deepseek_ocr2 import (
    DeepseekOcr2Config,
    DeepseekOcr2TextConfig,
    DeepseekOcr2VisionConfig,
)
from ..deepseek_ocr2.image_processing_deepseek_ocr2 import (
    DeepseekOcr2ImageProcessor,
    DeepseekOcr2ImageProcessorKwargs,
    get_optimal_tiled_canvas,
)
from ..deepseek_ocr2.modeling_deepseek_ocr2 import (
    DeepseekOcr2CausalLMOutputWithPast,
    DeepseekOcr2ForConditionalGeneration,
    DeepseekOcr2Model,
    DeepseekOcr2ModelOutputWithPast,
    DeepseekOcr2ModelOutputWithPooling,
    DeepseekOcr2PreTrainedModel,
    DeepseekOcr2SamVisionEncoder,
    DeepseekOcr2TextModel,
    DeepseekOcr2TextPreTrainedModel,
    DeepseekOcr2VisionModel,
)
from ..deepseek_ocr2.processing_deepseek_ocr2 import DeepseekOcr2Processor, DeepseekOcr2ProcessorKwargs
from ..got_ocr2.configuration_got_ocr2 import GotOcr2VisionConfig


class UnlimitedOcrImageProcessorKwargs(DeepseekOcr2ImageProcessorKwargs):
    r"""
    crop_to_patches (`bool`, *optional*, defaults to `True`):
        Whether to crop the image to patches. Can be overridden by the `crop_to_patches` parameter in the
        `preprocess` method.
    min_patches (`int`, *optional*, defaults to `2`):
        The minimum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
        set to `True`. Can be overridden by the `min_patches` parameter in the `preprocess` method.
    max_patches (`int`, *optional*, defaults to `32`):
        The maximum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
        set to `True`. Can be overridden by the `max_patches` parameter in the `preprocess` method.
    tile_size (`int`, *optional*, defaults to `640`):
        The size of each local tile. Must match the model's query embedding size.
    pad_if_larger_than (`int`, *optional*, defaults to `640`):
        If `crop_to_patches` is `False` and `size.height/width` is larger than this value,
        the image will be resized directly to `size.height/width` without padding. Otherwise,
        images are resized and padded to `size.height/width` while preserving the aspect ratio.
    background_color (`list[int]`, *optional*, defaults to `[127, 127, 127]`):
        The background color for padding.
    """

    pad_if_larger_than: int


class UnlimitedOcrImageProcessor(DeepseekOcr2ImageProcessor):
    tile_size = 640
    pad_if_larger_than = 640
    max_patches = 32
    model_input_names = ["pixel_values", "num_local_patches", "patches_grid"]

    def pad_to_square(
        self,
        images: "torch.Tensor",
        background_color: int | tuple[int, int, int] = 0,
    ) -> "torch.Tensor":
        """
        Pads an image to a square based on the longest edge.

        Args:
            images (`torch.Tensor`):
                The images to pad. Shape: (batch_size, num_channels, height, width) or (num_channels, height, width).
            background_color (`int` or `tuple[int, int, int]`, *optional*, defaults to 0):
                The color to use for the padding. Can be an integer for single channel or a
                tuple of integers representing for multi-channel images. If passed as integer
                in multi-channel mode, it will default to `0` in subsequent channels.
        Returns:
            `torch.Tensor`: The padded images.
        """
        height, width = images.shape[-2:]

        if height == width:
            return images

        num_channels = images.shape[1] if len(images.shape) == 4 else images.shape[0]
        if isinstance(background_color, int):
            background_color = [background_color] + [0] * (num_channels - 1)
        elif len(background_color) != num_channels:
            raise ValueError(
                f"background_color must have no more than {num_channels} elements to match the number of channels"
            )

        max_dim = max(height, width)
        # Use round(x/2) instead of x//2 to match PIL.ImageOps.pad behavior
        paste_x_left = round((max_dim - width) / 2)
        paste_y_left = round((max_dim - height) / 2)
        paste_x_right = max_dim - width - paste_x_left
        paste_y_right = max_dim - height - paste_y_left
        padded_images = tvF.pad(
            images, padding=[paste_x_left, paste_y_left, paste_x_right, paste_y_right], fill=background_color
        )

        return padded_images

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        size: SizeDict,
        crop_to_patches: bool,
        min_patches: int,
        max_patches: int,
        tile_size: int,
        pad_if_larger_than: int,
        resample: "PILImageResampling | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: "float | list[float] | None",
        image_std: "float | list[float] | None",
        disable_grouping: bool | None,
        return_tensors: "str | TensorType | None",
        **kwargs,
    ) -> BatchFeature:
        # --- Local patches (batched by shape group) ---
        local_patches_grouped = {}

        if crop_to_patches:
            grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)

            for shape, stacked_images in grouped_images.items():
                height, width = shape[-2:]
                if max(height, width) > tile_size:
                    stacked_patches, _ = self.crop_image_to_patches(
                        stacked_images,
                        min_patches=min_patches,
                        max_patches=max_patches,
                        tile_size=tile_size,
                        resample=resample,
                    )
                    flat_patches = stacked_patches.reshape(-1, *stacked_patches.shape[2:])
                    flat_patches = self.rescale_and_normalize(
                        flat_patches, do_rescale, rescale_factor, do_normalize, image_mean, image_std
                    )
                    local_patches_grouped[shape] = flat_patches.reshape(stacked_patches.shape)
                else:
                    local_patches_grouped[shape] = [None] * stacked_images.shape[0]

            ordered_local = reorder_images(local_patches_grouped, grouped_images_index)
        else:
            ordered_local = []

        flat_local_list = [patch for item in ordered_local if item is not None for patch in item]

        # --- Global view (batched by shape group) ---
        global_target_size = max(size.height, size.width)

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_global_grouped = {}
        for shape, stacked in grouped_images.items():
            if not crop_to_patches and global_target_size <= pad_if_larger_than:
                stacked = self.resize(
                    stacked, SizeDict(height=global_target_size, width=global_target_size), resample=resample
                )
            else:
                height, width = shape[-2:]
                scale = global_target_size / max(height, width)
                new_height = round(height * scale)
                new_width = round(width * scale)
                stacked = self.resize(stacked, SizeDict(height=new_height, width=new_width), resample=resample)
                stacked = self.pad_to_square(stacked, background_color=self.background_color)
            stacked = self.rescale_and_normalize(
                stacked, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_global_grouped[shape] = stacked
        all_pixel_values_global = reorder_images(processed_global_grouped, grouped_images_index)

        data = {"pixel_values": all_pixel_values_global}
        if flat_local_list:
            data["pixel_values_local"] = flat_local_list

        # Compute per-image spatial crop grid and local-patch counts.
        patches_grid = []
        num_local_patches = []
        for image in images:
            height, width = image.shape[-2:]
            if crop_to_patches and max(height, width) > tile_size:
                num_columns, num_rows = get_optimal_tiled_canvas(
                    (height, width), (tile_size, tile_size), min_patches, max_patches
                )
                num_local_patches.append(num_columns * num_rows)
            else:
                num_columns, num_rows = 1, 1
                num_local_patches.append(0)
            patches_grid.append([num_columns, num_rows])

        data["num_local_patches"] = num_local_patches
        data["patches_grid"] = patches_grid

        return BatchFeature(
            data=data,
            tensor_type=return_tensors,
        )


class UnlimitedOcrProcessorKwargs(DeepseekOcr2ProcessorKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class UnlimitedOcrProcessor(DeepseekOcr2Processor):
    def replace_image_token(self, image_inputs: dict, image_idx: int) -> TextInput:
        size = self.image_processor.size["height"]
        tile_size = self.image_processor.tile_size

        num_queries_global = math.ceil(size / self.patch_size / self.downsample_ratio)
        num_queries_local = math.ceil(tile_size / self.patch_size / self.downsample_ratio)

        num_columns = int(image_inputs["patches_grid"][image_idx][0])
        num_rows = int(image_inputs["patches_grid"][image_idx][1])
        num_tokens = num_queries_global * (num_queries_global + 1) + 1
        if int(image_inputs["num_local_patches"][image_idx]) > 0:
            num_tokens += (num_rows * num_queries_local) * (num_columns * num_queries_local + 1)
        return self.image_token * num_tokens


@auto_docstring(checkpoint="baidu/Unlimited-OCR")
@strict
class UnlimitedOcrSamVisionConfig(GotOcr2VisionConfig):
    r"""
    output_channels (`int`, *optional*, defaults to 256):
        Dimensionality of the output channels in the Patch Encoder.
    use_abs_pos (`bool`, *optional*, defaults to `True`):
        Whether to use absolute position embedding.
    use_rel_pos (`bool`, *optional*, defaults to `True`):
        Whether to use relative position embedding.
    window_size (`int`, *optional*, defaults to 14):
        Window size for relative position.
    global_attn_indexes (`list[int]`, *optional*, defaults to `[2, 5, 8, 11]`):
        The indexes of the global attention layers.
    mlp_dim (`int`, *optional*, defaults to 3072):
        The dimensionality of the MLP layer in the Transformer encoder.
    downsample_channels (`list[int]`, *optional*, defaults to `(512, 1024)`):
        The channel dimensions for the multi-scale downsampling neck layers.
    """

    model_type = "unlimited_ocr_sam_vision_model"
    base_config_key = "sam_config"

    downsample_channels: list[int] | tuple[int, ...] | None = None

    def __post_init__(self, **kwargs):
        if self.downsample_channels is None:
            self.downsample_channels = [512, 1024]
        return PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring(checkpoint="baidu/Unlimited-OCR")
@strict
class UnlimitedOcrVisionEncoderConfig(CLIPVisionConfig):
    r"""
    Example:

    ```python
    >>> from transformers import UnlimitedOcrConfig

    >>> config = UnlimitedOcrConfig()
    >>> encoder_config = config.vision_config.encoder_config
    ```"""

    model_type = "unlimited_ocr_vision_encoder"
    base_config_key = "encoder_config"
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    patch_size: int | list[int] | tuple[int, int] | None = 14
    projection_dim = AttributeError()


@auto_docstring(checkpoint="baidu/Unlimited-OCR")
@strict
class UnlimitedOcrVisionConfig(DeepseekOcr2VisionConfig):
    model_type = "unlimited_ocr_vision"
    base_config_key = "vision_config"
    sub_configs = {
        "sam_config": UnlimitedOcrSamVisionConfig,
        "encoder_config": UnlimitedOcrVisionEncoderConfig,
    }

    def __post_init__(self, **kwargs):
        if self.sam_config is None:
            self.sam_config = self.sub_configs["sam_config"]()
        elif isinstance(self.sam_config, dict):
            self.sam_config = self.sub_configs["sam_config"](**self.sam_config)

        if self.encoder_config is None:
            self.encoder_config = self.sub_configs["encoder_config"]()
        elif isinstance(self.encoder_config, dict):
            self.encoder_config = self.sub_configs["encoder_config"](**self.encoder_config)

        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring(checkpoint="baidu/Unlimited-OCR")
@strict
class UnlimitedOcrTextConfig(DeepseekOcr2TextConfig):
    r"""
    n_group (`int`, *optional*):
        Number of groups for grouped top-k expert routing.
    topk_method (`str`, *optional*, defaults to `"greedy"`):
        Method for selecting top-k experts in MoE layers.
    mlp_layer_types (`list[str]`, *optional*):
        MLP type (`"dense"` or `"sparse"`) for each decoder layer, e.g. `["dense", "sparse", "sparse", ...]`.
    layer_types (`list[str]`, *optional*):
        Attention type for each decoder layer. Defaults to `"full_attention"` on every layer so the KV cache
        retains all tokens; the sliding window (`sliding_window`) is applied as a mask over generated tokens
        only, not by truncating the cache.
    sliding_window (`int`, *optional*, defaults to 128):
        If set, each token additionally attends only to the last `sliding_window` tokens. The image and prompt
        tokens processed during prefill stay fully visible (they are never evicted); the window only applies
        across generated tokens. Set to `None` for full causal attention.
    """

    model_type = "unlimited_ocr_text"
    base_config_key = "text_config"
    vocab_size: int = 129280
    hidden_size: int = 1280
    intermediate_size: int = 6848
    num_hidden_layers: int = 12
    num_attention_heads: int = 10
    num_key_value_heads: int | None = 10
    max_position_embeddings: int = 32768
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    moe_intermediate_size: int = 896
    n_group: int | None = 1
    topk_group: int | None = 1
    num_experts_per_tok: int | None = 6
    layer_types: list[str] | None = None
    use_sliding_window: bool = True
    sliding_window: int | None = 128

    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        if self.layer_types is None:
            self.layer_types = [
                "reference_sliding_attention" if self.use_sliding_window else "full_attention"
            ] * self.num_hidden_layers
        elif len(set(self.layer_types)) > 1:
            # This requires a custom create_causal_mask implementation for reference_sliding_attention
            # that fetches the first layer with is_sliding=True.
            raise ValueError(
                f"Combining multiple layer types ({set(self.layer_types)}) is not supported for this model."
            )
        if self.mlp_layer_types is None:
            # Some configs may use `first_k_dense_replace` instead of `layer_types`/`mlp_layer_types`
            first_k_dense_replace = kwargs.pop("first_k_dense_replace", 1)
            self.mlp_layer_types = [
                "sparse" if layer_idx >= first_k_dense_replace else "dense"
                for layer_idx in range(self.num_hidden_layers)
            ]
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="baidu/Unlimited-OCR")
@strict
class UnlimitedOcrConfig(DeepseekOcr2Config):
    model_type = "unlimited_ocr"
    sub_configs = {
        "vision_config": UnlimitedOcrVisionConfig,
        "text_config": UnlimitedOcrTextConfig,
    }

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        elif isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)

        if self.text_config is None:
            text_config_cls = self.sub_configs["text_config"]
            text_keys = text_config_cls().to_dict().keys()
            text_kwargs = {key: kwargs[key] for key in text_keys if key in kwargs}
            self.text_config = text_config_cls(**text_kwargs)
        elif isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        PreTrainedConfig.__post_init__(self, **kwargs)


class UnlimitedOcrModelOutputWithPooling(DeepseekOcr2ModelOutputWithPooling):
    pass


class UnlimitedOcrModelOutputWithPast(DeepseekOcr2ModelOutputWithPast):
    pass


class UnlimitedOcrCausalLMOutputWithPast(DeepseekOcr2CausalLMOutputWithPast):
    pass


class UnlimitedOcrPreTrainedModel(DeepseekOcr2PreTrainedModel):
    _no_split_modules = [
        "UnlimitedOcrEncoderLayer",
        "UnlimitedOcrSamVisionLayer",
        "UnlimitedOcrTextDecoderLayer",
    ]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, UnlimitedOcrModel):
            embed_std = 1 / math.sqrt(self.config.text_config.hidden_size)
            init.normal_(module.image_newline, mean=0.0, std=embed_std)
        elif isinstance(module, UnlimitedOcrVisionEmbeddings):
            factor = module.config.initializer_factor
            init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
            init.copy_(module.position_ids, torch.arange(module.num_positions).expand((1, -1)))


class UnlimitedOcrSamVisionEncoder(DeepseekOcr2SamVisionEncoder):
    pass


class UnlimitedOcrAttention(CLIPAttention):
    def __init__(self, config: UnlimitedOcrVisionEncoderConfig):
        super().__init__(config)
        # Required for repeat_kv(..., num_key_value_groups)
        self.num_key_value_groups = 1


class UnlimitedOcrEncoderLayer(CLIPEncoderLayer):
    pass


class UnlimitedOcrVisionEmbeddings(CLIPVisionEmbeddings):
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_patches = embeddings.shape[1] - 1
        position_embedding = self.position_embedding.weight.unsqueeze(0)
        num_positions = position_embedding.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        class_pos_embed = position_embedding[:, :1]
        patch_pos_embed = position_embedding[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        # TODO: check if we can drop dtype cast
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(torch.float32),
            size=(new_height, new_width),
            mode="bicubic",
            antialias=True,  # different from CLIP
            align_corners=False,
        ).to(target_dtype)

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, patch_embeds: torch.Tensor) -> torch.Tensor:
        r"""
        patch_embeds (`torch.Tensor` of shape `(batch_size, hidden_size, grid_height, grid_width)`):
            The SAM feature map, injected directly as the CLIP patch embeddings. The CLIP patch convolution
            (`self.patch_embedding`) is intentionally bypassed.
        """
        batch_size, _, grid_height, grid_width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.interpolate_pos_encoding(
            embeddings, grid_height * self.patch_size, grid_width * self.patch_size
        )
        return embeddings


class UnlimitedOcrVisionEncoder(CLIPVisionModel):
    main_input_name = "patch_embeds"
    _can_record_outputs = {
        "hidden_states": UnlimitedOcrEncoderLayer,
        "attentions": UnlimitedOcrAttention,
    }

    def __init__(self, config: UnlimitedOcrVisionEncoderConfig):
        super().__init__(config)
        del self.post_layernorm

    @can_return_tuple
    @capture_outputs
    @auto_docstring
    def forward(self, patch_embeds: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> BaseModelOutput:
        r"""
        patch_embeds (`torch.Tensor` of shape `(batch_size, hidden_size, grid_height, grid_width)`):
            Patch embeddings.
        """
        hidden_states = self.embeddings(patch_embeds)
        hidden_states = self.pre_layrnorm(hidden_states)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, **kwargs)
        return BaseModelOutput(last_hidden_state=encoder_outputs.last_hidden_state)


class UnlimitedOcrVisionModel(DeepseekOcr2VisionModel):
    def __init__(self, config: UnlimitedOcrVisionConfig):
        super().__init__(config)
        del self.query_768_resolution
        del self.query_1024_resolution

    @can_return_tuple
    @auto_docstring
    def forward(self, pixel_values: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> BaseModelOutput:
        sam_encoder_outputs = self.sam_encoder(pixel_values, **kwargs)
        sam_feature_map = sam_encoder_outputs.last_hidden_state

        vision_encoder_outputs = self.vision_encoder(sam_feature_map, **kwargs)
        vision_encoder_hidden_state = vision_encoder_outputs.last_hidden_state

        sam_hidden_state = sam_feature_map.flatten(2).transpose(1, 2)
        hidden_state = torch.cat([vision_encoder_hidden_state[:, 1:], sam_hidden_state], dim=-1)

        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=vision_encoder_outputs.hidden_states,
            attentions=vision_encoder_outputs.attentions,
        )


class UnlimitedOcrTextPreTrainedModel(DeepseekOcr2TextPreTrainedModel):
    pass


class UnlimitedOcrDynamicReferenceSlidingWindowLayer(DynamicSlidingWindowLayer):
    """Reference sliding-window attention (R-SWA) cache layer that keeps all prefill tokens before
    the first decode step and applies a sliding window to all decoded tokens.

    Once ``sliding_window`` decode tokens have accumulated, the oldest decode tokens are evicted and
    replaced by the most recent ones. The prefill tokens always remain in the cache.
    """

    layer_type = "reference_sliding_attention"

    def __init__(self, config: PreTrainedConfig | None = None, sliding_window: int | None = None):
        super().__init__(config=config, sliding_window=sliding_window)
        self.prefill_length: int | None = None

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        kv_length = key_states.shape[-2]
        self.cumulative_length += kv_length

        # Compute the full states
        full_key_states = torch.cat([self.keys, key_states], dim=-2)
        full_value_states = torch.cat([self.values, value_states], dim=-2)

        # Prefill
        if self.prefill_length is None and kv_length > 1:
            self.keys = full_key_states
            self.values = full_value_states
            return self.keys, self.values

        # First decode step (or an empty prompt that skipped prefill above): mark prefill as complete
        if self.prefill_length is None:
            self.prefill_length = self.keys.shape[-2] if self.keys.dim() > 1 else 0

        # Cache growing
        if self.cumulative_length <= self.prefill_length + self.sliding_window - 1:
            self.keys = full_key_states
            self.values = full_value_states
        # Cache full
        elif self.keys.shape[-2] == self.prefill_length + self.sliding_window - 1:
            self.keys[:, :, -self.sliding_window + 1 :].copy_(full_key_states[:, :, -self.sliding_window + 1 :, :])
            self.values[:, :, -self.sliding_window + 1 :].copy_(full_value_states[:, :, -self.sliding_window + 1 :, :])
        # Cache full after this update and full_key_states > cache size
        else:
            self.keys = torch.cat(
                [
                    full_key_states[:, :, : self.prefill_length, :],
                    full_key_states[:, :, -self.sliding_window + 1 :, :],
                ],
                dim=-2,
            )
            self.values = torch.cat(
                [
                    full_value_states[:, :, : self.prefill_length, :],
                    full_value_states[:, :, -self.sliding_window + 1 :, :],
                ],
                dim=-2,
            )

        # Return full states to avoid losing context in case we added multiple tokens at once
        return full_key_states, full_value_states

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        is_full = (
            self.prefill_length is not None and self.cumulative_length >= self.prefill_length + self.sliding_window
        )

        kv_offset = 0
        if is_full:
            kv_offset = max(self.cumulative_length - self.prefill_length - self.sliding_window + 1, 0)
            kv_length = self.prefill_length + self.sliding_window - 1 + query_length
        else:
            kv_length = self.cumulative_length + query_length

        # Returned kv_offset is with respect to sliding window keys.
        # Remove kv_offset from kv_idx to retrieve the prefill indices.
        return kv_length, kv_offset

    def get_max_length(self) -> int:
        """Return the maximum cache shape of the cache"""
        if self.prefill_length is None:
            return -1
        return self.prefill_length + self.sliding_window

    def crop(self, max_length: int) -> None:
        """
        Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens.
        """
        if self.prefill_length is not None and self.get_seq_length() >= self.prefill_length + self.sliding_window:
            raise ValueError(
                "Cannot `crop` a `UnlimitedOcrDynamicReferenceSlidingWindowLayer` after it has seen more tokens than its"
                "prefill + sliding window (otherwise some states are lost)"
            )
        DynamicLayer.crop(self, max_length)
        self.cumulative_length = self.keys.shape[-2]


class UnlimitedOcrStaticReferenceSlidingWindowLayer(StaticSlidingWindowLayer):
    """
    A static cache layer that stores the key and value states as static tensors of shape
    `[batch_size, num_heads, min(max_cache_len, prefill_size + sliding_window), head_dim]`.
    It lazily allocates its full backing tensors, and then mutates them in-place.
    Built for `torch.compile` support.

    The backing buffer is split in two regions: slots ``[0, prefill_length)`` hold the prefill (reference) slots
    that are never eviced from the cache. Slots ``[prefill_length, prefill_length + sliding_window)`` hold the
    sliding window decode slots where the oldest entries are always replaced by the newest ones.

    Args:
        max_cache_len (`int`):
            Maximum number of tokens that can be stored, used for tensor preallocation.
        sliding_window (`int`):
            The size of the sliding window.
    """

    layer_type = "reference_sliding_attention"

    def __init__(self, max_cache_len: int, sliding_window: int):
        super().__init__(max_cache_len=max_cache_len, sliding_window=sliding_window)
        # Keep `max_cache_len` as max value for length bookkeeping.
        # The physical buffer doesn't exceed `prefill_length + sliding_window`.
        self.max_cache_len = max_cache_len
        self.sliding_window = sliding_window
        self.prefill_length: int | None = None

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        kv_length = key_states.shape[-2]

        # Prefill
        if self.prefill_length is None and kv_length > 1:
            # Chunked prefill: the buffer was sized for the first chunk only, so grow it to hold all prefill seen so far plus this chunk plus the reserved window.
            required_length = min(self.max_cache_len, self.cumulative_length_int + kv_length + self.sliding_window)
            if self.keys.shape[-2] < required_length:
                self._allocate_key_value_buffers(required_length, copy_existing=True)
            # Note: very important to use the tensor version of the cumulative length here, as otherwise cudagraphs
            # (triggered by mode="reduced_overhead") will lead to random crashes, as the int would be overwritten
            cache_position = torch.arange(kv_length, device=self.device) + self.cumulative_length
            try:
                self.keys.index_copy_(2, cache_position, key_states)
                self.values.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                # Fallback for devices like MPS where index_copy_ might not be supported.
                self.keys[:, :, cache_position] = key_states
                self.values[:, :, cache_position] = value_states

            # Keep both the int (control flow) and the tensor (cudagraph-safe indexing) versions in sync.
            self.cumulative_length_int += kv_length
            self.cumulative_length.add_(kv_length)

            # Very important to return the `self` tensors here, as they have the static dynamo address
            return self.keys, self.values

        # First decode step (or an empty prompt that skipped prefill above): mark prefill as complete
        if self.prefill_length is None:
            self.prefill_length = self.cumulative_length_int

        # Everything below mirrors `StaticSlidingWindowLayer.update` with the difference that the sliding window
        # is in the `sliding_window` slots right after the prefill instead of spanning the whole buffer.
        window_start = self.prefill_length
        current_length = self.cumulative_length_int - self.prefill_length
        is_full = current_length >= self.sliding_window
        # Update it now that we saved the value above
        self.cumulative_length_int += kv_length

        if is_full:
            # In general, we should use a much simpler `cat` here as well, independently of the states size. However,
            # dynamo is currently bugged when doing it - see https://github.com/pytorch/pytorch/issues/159855 for more details
            if key_states.shape[-2] == 1:
                # Roll the window region to the left by 1 position (the pinned prefill in front stays put)
                new_keys = self.keys[:, :, window_start : window_start + self.sliding_window, :].roll(-1, dims=-2)
                new_values = self.values[:, :, window_start : window_start + self.sliding_window, :].roll(-1, dims=-2)
                # Overwrite the last position with new states
                # (note: very important to use a tensor to index here, see https://github.com/pytorch/pytorch/issues/159855)
                index = torch.tensor([-1], dtype=int, device=self.device)
                new_keys[:, :, index] = key_states
                new_values[:, :, index] = value_states

                # Copy back into `self` (do not just assign again) in order to keep the static dynamo address
                self.keys[:, :, window_start : window_start + self.sliding_window, :].copy_(new_keys)
                self.values[:, :, window_start : window_start + self.sliding_window, :].copy_(new_values)

                # Very important to return the `self` tensors here, as they have the static dynamo address
                return self.keys, self.values
            # Already full but using more than 1 new token (e.g. chat continuation, etc...)
            else:
                full_key_states = torch.cat(
                    (self.keys[:, :, window_start + 1 : window_start + self.sliding_window, :], key_states), dim=-2
                )
                full_value_states = torch.cat(
                    (self.values[:, :, window_start + 1 : window_start + self.sliding_window, :], value_states), dim=-2
                )
        # Not yet full, but becoming full on this update
        elif current_length + kv_length > self.sliding_window:
            # Fast path, no need to cat() in this case, as the window is currently empty
            if current_length == 0:
                full_key_states = key_states
                full_value_states = value_states
            else:
                window = slice(window_start, window_start + current_length)
                full_key_states = torch.cat((self.keys[:, :, window, :], key_states), dim=-2)
                full_value_states = torch.cat((self.values[:, :, window, :], value_states), dim=-2)
        else:
            # Note: very important to use the tensor version of the cumulative length here, as otherwise cudagraphs
            # (triggered by mode="reduced_overhead") will lead to random crashes, as the int would be overwritten.
            cache_position = torch.arange(kv_length, device=self.device) + self.cumulative_length
            try:
                self.keys.index_copy_(2, cache_position, key_states)
                self.values.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                # Fallback for devices like MPS where index_copy_ might not be supported.
                self.keys[:, :, cache_position] = key_states
                self.values[:, :, cache_position] = value_states

            # Update the tensor version of the length in-place (we don't need to update it if we are already outside
            # of this branch, as we don't need the tensor anymore)
            self.cumulative_length.add_(kv_length)

            # Very important to return the `self` tensors here, as they have the static dynamo address
            return self.keys, self.values

        # We only cache the last `sliding_window` tokens
        self.keys[:, :, window_start : window_start + self.sliding_window, :].copy_(
            full_key_states[:, :, -self.sliding_window :, :]
        )
        self.values[:, :, window_start : window_start + self.sliding_window, :].copy_(
            full_value_states[:, :, -self.sliding_window :, :]
        )
        return full_key_states, full_value_states

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.batch_size, self.num_heads = key_states.shape[:2]
        self.v_head_dim = value_states.shape[-1]
        self.k_head_dim = key_states.shape[-1]

        self.cumulative_length = self.cumulative_length.to(self.device)
        # Note: `mark_static_address` is used to tag the tensors as a fixed data pointer, preventing compiled graph
        # breaks or cudagraph skips due to inplace mutations when updating the cache. However, it is not supported when
        # tracing the graph, so we skip it in this case. As prefill should never be compiled, this is not an issue and it
        # will still be run (except when users compile prefill explicitly, but this should be avoided!)
        # Without this, we cannot use cudagraphs!!
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.cumulative_length)

        prefill_seen = key_states.shape[-2] if key_states.shape[-2] > 1 else 0
        self._allocate_key_value_buffers(min(self.max_cache_len, prefill_seen + self.sliding_window))

        self.is_initialized = True

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        is_full = (
            self.prefill_length is not None and self.cumulative_length_int >= self.prefill_length + self.sliding_window
        )

        kv_offset = 0
        # Prefill
        if self.prefill_length is None and query_length > 1:
            kv_length = min(self.max_cache_len, self.cumulative_length_int + query_length + self.sliding_window)
        # Decode: cache is already full
        elif is_full:
            kv_offset = max(self.cumulative_length_int - self.prefill_length - self.sliding_window + 1, 0)
            kv_length = self.prefill_length + self.sliding_window - 1 + query_length
        # Decode: cache not yet full, but becoming full on this update
        elif (
            self.prefill_length is not None
            and self.cumulative_length_int + query_length > self.prefill_length + self.sliding_window
        ):
            kv_length = self.cumulative_length_int + query_length
        # Decode: cache not yet full but we return the local size as it's static
        else:
            kv_length = self.keys.shape[-2]

        return kv_length, kv_offset

    def reset(self) -> None:
        super().reset()
        self.prefill_length = None

    def _allocate_key_value_buffers(self, physical_length: int, copy_existing: bool = False) -> None:
        """(Re)allocate the static key/value buffers to `physical_length` slots and (re)tag the static address.

        Only ever called from the eager prefill pass (initial allocation or chunked-prefill growth), never from a
        compiled decode step, so reallocating here is safe for cudagraphs.
        """
        new_keys = torch.zeros(
            (self.batch_size, self.num_heads, physical_length, self.k_head_dim), dtype=self.dtype, device=self.device
        )
        new_values = torch.zeros(
            (self.batch_size, self.num_heads, physical_length, self.v_head_dim), dtype=self.dtype, device=self.device
        )
        if copy_existing:
            old_length = self.keys.shape[-2]
            new_keys[:, :, :old_length, :].copy_(self.keys)
            new_values[:, :, :old_length, :].copy_(self.values)
        self.keys = new_keys
        self.values = new_values
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)


def create_reference_sliding_window_causal_mask(**kwargs):
    past_key_values = kwargs["past_key_values"]
    inputs_embeds = kwargs["inputs_embeds"]

    if past_key_values is None:
        prefill_length = float("inf")
        kv_offset = 0
    else:
        layer = next(layer for layer in past_key_values.layers if layer.layer_type == "reference_sliding_attention")
        prefill_length = float("inf") if layer.prefill_length is None else layer.prefill_length
        _, kv_offset = layer.get_mask_sizes(query_length=inputs_embeds.shape[1])

    def prefill_overlay(batch_idx, head_idx, q_idx, kv_idx):
        # Remove kv_offset to retrieve the kv_index with respect to prefill
        return kv_idx - kv_offset < prefill_length

    return create_sliding_window_causal_mask(
        or_mask_function=and_masks(causal_mask_function, prefill_overlay), **kwargs
    )


class UnlimitedOcrTextModel(DeepseekOcr2TextModel):
    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # It may already have been prepared by, e.g., `generate`
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "reference_sliding_attention": create_reference_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[layer_idx]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class UnlimitedOcrModel(DeepseekOcr2Model):
    _keys_to_ignore_on_load_unexpected = {"lm_head"}

    def __init__(self, config: UnlimitedOcrConfig):
        super().__init__(config)
        self.multi_modal_projector = nn.Linear(
            config.vision_config.sam_config.downsample_channels[-1] + config.vision_config.encoder_config.hidden_size,
            config.text_config.hidden_size,
        )
        self.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size))

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_values_local: torch.FloatTensor | None = None,
        num_local_patches: list[int] | torch.Tensor | None = None,
        patches_grid: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> "UnlimitedOcrModelOutputWithPooling":
        r"""
        pixel_values_local (`torch.FloatTensor` of shape `(total_patches, 3, height, width)`, *optional*):
            All local patches flattened across the batch, or `None` if no local views.
        num_local_patches (`list[int]` or `torch.Tensor`, *optional*):
            Number of local patches per image, e.g. `[6, 0, 4]`.
        patches_grid (`torch.Tensor` of shape `(num_images, 2)`, *optional*):
            The patches grid `(num_columns, num_rows)` per image.
        """
        if isinstance(num_local_patches, torch.Tensor):
            num_local_patches = num_local_patches.tolist()

        batch_size = pixel_values.shape[0]

        global_vision_outputs = self.vision_tower(pixel_values, **kwargs)
        global_features = self.multi_modal_projector(global_vision_outputs.last_hidden_state)

        local_outputs = {}
        if pixel_values_local is not None:
            local_vision_outputs = self.vision_tower(pixel_values_local, **kwargs)
            all_local_features = self.multi_modal_projector(local_vision_outputs.last_hidden_state)
            per_image_local = torch.split(all_local_features, num_local_patches, dim=0)
            local_outputs = {
                "local_last_hidden_state": local_vision_outputs.last_hidden_state,
                "local_hidden_states": local_vision_outputs.hidden_states,
                "local_attentions": local_vision_outputs.attentions,
            }
        else:
            per_image_local = [None] * batch_size

        hidden_size = global_features.shape[-1]
        newline = self.image_newline[None, None, :]
        view_separator = self.view_separator[None, :]

        all_features = []
        for idx in range(batch_size):
            num_queries_global = int(global_features.shape[1] ** 0.5)
            global_grid = global_features[idx].reshape(num_queries_global, num_queries_global, hidden_size)
            global_grid = torch.cat([global_grid, newline.expand(num_queries_global, 1, hidden_size)], dim=1)
            global_flat = global_grid.reshape(-1, hidden_size)

            local_features = per_image_local[idx]
            if local_features is not None and local_features.shape[0] > 0:
                num_columns, num_rows = int(patches_grid[idx][0]), int(patches_grid[idx][1])
                num_queries_local = int(local_features.shape[1] ** 0.5)
                local_grid = local_features.reshape(
                    num_rows, num_columns, num_queries_local, num_queries_local, hidden_size
                )
                local_grid = local_grid.permute(0, 2, 1, 3, 4).reshape(
                    num_rows * num_queries_local, num_columns * num_queries_local, hidden_size
                )
                local_grid = torch.cat(
                    [local_grid, newline.expand(num_rows * num_queries_local, 1, hidden_size)], dim=1
                )
                local_flat = local_grid.reshape(-1, hidden_size)
                all_features.append(torch.cat([local_flat, global_flat, view_separator], dim=0))
            else:
                all_features.append(torch.cat([global_flat, view_separator], dim=0))

        image_features = torch.cat(all_features, dim=0)
        return UnlimitedOcrModelOutputWithPooling(
            last_hidden_state=global_vision_outputs.last_hidden_state,
            pooler_output=image_features,
            hidden_states=global_vision_outputs.hidden_states,
            attentions=global_vision_outputs.attentions,
            **local_outputs,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_local: torch.FloatTensor | None = None,
        num_local_patches: list[int] | torch.Tensor | None = None,
        patches_grid: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | UnlimitedOcrModelOutputWithPast:
        r"""
        pixel_values_local (`torch.FloatTensor`, *optional*):
            Local patch pixel values of shape `(total_patches, 3, H, W)`.
        num_local_patches (`list[int]` or `torch.Tensor`, *optional*):
            Number of local patches per image in the batch.
        patches_grid (`torch.Tensor` of shape `(num_images, 2)`, *optional*):
            The patches grid `(num_columns, num_rows)` per image.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values, pixel_values_local, num_local_patches, patches_grid, return_dict=True
            ).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

            special_image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        return UnlimitedOcrModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )


class UnlimitedOcrForConditionalGeneration(DeepseekOcr2ForConditionalGeneration):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_local: torch.FloatTensor | None = None,
        num_local_patches: list[int] | torch.Tensor | None = None,
        patches_grid: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | UnlimitedOcrCausalLMOutputWithPast:
        r"""
        pixel_values_local (`torch.FloatTensor`, *optional*):
            Local patch pixel values of shape `(total_patches, 3, H, W)`.
        num_local_patches (`list[int]` or `torch.Tensor`, *optional*):
            Number of local patches per image in the batch.
        patches_grid (`torch.Tensor` of shape `(num_images, 2)`, *optional*):
            The patches grid `(num_columns, num_rows)` per image.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_local=pixel_values_local,
            num_local_patches=num_local_patches,
            patches_grid=patches_grid,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

        return UnlimitedOcrCausalLMOutputWithPast(
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
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_local=None,
        num_local_patches=None,
        patches_grid=None,
        attention_mask=None,
        logits_to_keep=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if is_first_iteration or not kwargs.get("use_cache", True):
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_values_local"] = pixel_values_local
            model_inputs["num_local_patches"] = num_local_patches
            model_inputs["patches_grid"] = patches_grid

        return model_inputs


__all__ = [
    "UnlimitedOcrConfig",
    "UnlimitedOcrTextConfig",
    "UnlimitedOcrVisionConfig",
    "UnlimitedOcrVisionEncoderConfig",
    "UnlimitedOcrSamVisionConfig",
    "UnlimitedOcrForConditionalGeneration",
    "UnlimitedOcrImageProcessor",
    "UnlimitedOcrModel",
    "UnlimitedOcrPreTrainedModel",
    "UnlimitedOcrProcessor",
    "UnlimitedOcrTextModel",
    "UnlimitedOcrTextPreTrainedModel",
    "UnlimitedOcrVisionModel",
]
