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
from ...cache_utils import Cache, DynamicCache, DynamicSlidingWindowLayer
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...masking_utils import (
    create_causal_mask,
    create_reference_sliding_window_causal_mask,
)
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
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
from ..deepseek_ocr2.image_processing_deepseek_ocr2 import DeepseekOcr2ImageProcessor, get_optimal_tiled_canvas
from ..deepseek_ocr2.modeling_deepseek_ocr2 import (
    DeepseekOcr2CausalLMOutputWithPast,
    DeepseekOcr2ForConditionalGeneration,
    DeepseekOcr2Model,
    DeepseekOcr2ModelOutputWithPast,
    DeepseekOcr2ModelOutputWithPooling,
    DeepseekOcr2PreTrainedModel,
    DeepseekOcr2TextModel,
    DeepseekOcr2TextPreTrainedModel,
    DeepseekOcr2VisionModel,
)
from ..deepseek_ocr2.processing_deepseek_ocr2 import DeepseekOcr2Processor, DeepseekOcr2ProcessorKwargs
from ..got_ocr2.configuration_got_ocr2 import GotOcr2VisionConfig


class UnlimitedOcrImageProcessor(DeepseekOcr2ImageProcessor):
    tile_size = 640
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
        num_local_patches_grouped = {}
        local_patches_grouped = {}

        if crop_to_patches:
            grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)

            for shape, stacked_images in grouped_images.items():
                h, w = shape[-2:]
                if max(h, w) > tile_size:
                    stacked_patches, n_patches = self.crop_image_to_patches(
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
                    num_local_patches_grouped[shape] = [n_patches] * stacked_images.shape[0]
                else:
                    local_patches_grouped[shape] = [None] * stacked_images.shape[0]
                    num_local_patches_grouped[shape] = [0] * stacked_images.shape[0]

            ordered_local = reorder_images(local_patches_grouped, grouped_images_index)
        else:
            ordered_local = []

        flat_local_list = [patch for item in ordered_local if item is not None for patch in item]

        # --- Global view (batched by shape group) ---
        global_target_size = size.height

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_global_grouped = {}
        for shape, stacked in grouped_images.items():
            h, w = shape[-2:]
            scale = global_target_size / max(h, w)
            new_h = round(h * scale)
            new_w = round(w * scale)
            stacked = self.resize(stacked, SizeDict(height=new_h, width=new_w), resample=resample)
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
    def _expand_image_tokens(
        self,
        text: list[TextInput],
        patches_grid: torch.Tensor,
        num_local_patches: list[int] | torch.Tensor,
    ) -> list[str]:
        num_images = len(patches_grid)
        total_image_tokens = sum(t.count(self.image_token) for t in text)
        if total_image_tokens != num_images:
            raise ValueError(
                f"Number of `{self.image_token}` tokens in the text ({total_image_tokens}) does not match the "
                f"number of images passed ({num_images}). Use one `{self.image_token}` placeholder per image, "
                f"e.g. `'<image>' * len(images) + 'Multi page parsing.'`"
            )

        size = self.image_processor.size["height"]
        tile_size = self.image_processor.tile_size

        num_queries_global = math.ceil(size / self.patch_size / self.downsample_ratio)
        num_queries_local = math.ceil(tile_size / self.patch_size / self.downsample_ratio)

        crop_index = 0
        for i in range(len(text)):
            while self.image_token in text[i]:
                num_columns = int(patches_grid[crop_index][0])
                num_rows = int(patches_grid[crop_index][1])
                num_tokens = num_queries_global * (num_queries_global + 1) + 1
                if int(num_local_patches[crop_index]) > 0:
                    num_tokens += (num_rows * num_queries_local) * (num_columns * num_queries_local + 1)
                text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_tokens, 1)
                crop_index += 1
            text[i] = text[i].replace("<|placeholder|>", self.image_token)
        return text

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[UnlimitedOcrProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Global view pixel values. Returned when `images` is not `None`.
            - **pixel_values_local** -- Local patch pixel values. Returned when `images` is not `None`.
            - **num_local_patches** -- Number of local patches per image. Returned when `images` is not `None`.
            - **patches_grid** -- Number of patch columns and rows per image. Returned when `images` is not `None`.
        """
        if images is None:
            raise ValueError("`images` are expected as arguments to a `UnlimitedOcrProcessor` instance.")
        if text is None:
            raise ValueError("`text` is required for `UnlimitedOcrProcessor`. Example: `'<image>\\nFree OCR.'`")

        output_kwargs = self._merge_kwargs(
            UnlimitedOcrProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        text = text.copy()  # below lines change text in-place

        image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        text = self._expand_image_tokens(text, image_inputs["patches_grid"], image_inputs["num_local_patches"])

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        return BatchFeature(
            data={**text_inputs, **image_inputs},
            tensor_type=return_tensors,
        )


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
        return PretrainedConfig.__post_init__(self, **kwargs)


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

        PretrainedConfig.__post_init__(self, **kwargs)


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
    sliding_window: int | None = 128

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if self.sliding_window is None else "reference_sliding_attention"
            ] * self.num_hidden_layers
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
        PretrainedConfig.__post_init__(self, **kwargs)


class UnlimitedOcrModelOutputWithPooling(DeepseekOcr2ModelOutputWithPooling):
    pass


class UnlimitedOcrModelOutputWithPast(DeepseekOcr2ModelOutputWithPast):
    pass


class UnlimitedOcrCausalLMOutputWithPast(DeepseekOcr2CausalLMOutputWithPast):
    pass


class UnlimitedOcrPreTrainedModel(DeepseekOcr2PreTrainedModel):
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


class DynamicReferenceSlidingWindowLayer(DynamicSlidingWindowLayer):
    """Reference sliding-window attention (R-SWA) cache layer that keeps all prefill tokens and windows
    only the generated ones.

    The stock `DynamicSlidingWindowLayer` evicts the oldest tokens once the window fills, which would
    discard the image/prompt prefill. Here the prefill (every token cached before the first single-token
    decode step) is kept intact and never evicted. Generated tokens are appended right after the prefill
    until ``sliding_window`` of them have accumulated; from then on the newest generated token overwrites
    the oldest one.

    While the window is filling, the cached length grows by one each step (exactly like the stock sliding
    layer), so within the first ``sliding_window`` generated tokens the layer is indistinguishable from
    full attention. Once the ring is full the cached tensors stay at a constant length
    (``prefill + sliding_window``), which lets the cuDNN SDPA backend reuse a single kernel plan across the
    (typically long) steady-state decode instead of re-planning on every distinct sequence length.
    """

    layer_type = "reference_sliding_attention"

    def __init__(self, config: PretrainedConfig | None = None, sliding_window: int | None = None):
        super().__init__(config=config, sliding_window=sliding_window)
        self.prefill_length: int | None = None
        self.ring_position = 0

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        sequence_length = key_states.shape[-2]
        self.cumulative_length += sequence_length

        # Prefill with prompt context
        if self.prefill_length is None and sequence_length > 1:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)
            return self.keys, self.values

        # First decode step
        # Handle generation with empty prompt
        if self.prefill_length is None:
            self.prefill_length = self.keys.shape[-2] if self.keys.dim() > 1 else 0

        # Append while window grows
        generated_length = self.keys.shape[-2] - self.prefill_length if self.keys.dim() > 1 else 0
        append_length = min(sequence_length, max(0, self.sliding_window - generated_length))
        if append_length > 0:
            self.keys = torch.cat([self.keys, key_states[..., :append_length, :]], dim=-2)
            self.values = torch.cat([self.values, value_states[..., :append_length, :]], dim=-2)

        # Overwrite if window size is reached
        overwrite_length = sequence_length - append_length
        if overwrite_length > 0:
            # Only the most recent `sliding_window` overwrites survive
            write_length = min(overwrite_length, self.sliding_window)
            start = self.ring_position + overwrite_length - write_length
            offsets = torch.arange(write_length, device=key_states.device)
            slots = self.prefill_length + (start + offsets) % self.sliding_window
            self.keys[..., slots, :] = key_states[..., sequence_length - write_length :, :]
            self.values[..., slots, :] = value_states[..., sequence_length - write_length :, :]
            self.ring_position = (self.ring_position + overwrite_length) % self.sliding_window

        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        # Full visibility, no sliding offset: every query attends to all cached keys (all prefill plus the
        # generated tokens currently held in the ring), so the prefill is never masked out.
        if self.prefill_length is None:
            return self.cumulative_length + query_length, 0
        return self.decode_kv_length(query_length), 0

    def decode_kv_length(self, query_length: int = 1) -> int | None:
        """Physical length of the cached buffer after the upcoming `update` of `query_length` tokens.

        Returns `None` before the layer is initialized. Used to size the all-visible decode mask, whose
        width must match the key/value tensors returned by `update`.
        """
        if not self.is_initialized:
            return None
        if self.prefill_length is not None:
            prefill_length = self.prefill_length
            generated_before = self.keys.shape[-2] - prefill_length if self.keys.dim() > 1 else 0
        else:
            # Before the first decode step the whole buffer is prefill.
            prefill_length = self.keys.shape[-2] if self.keys.dim() > 1 else 0
            generated_before = 0
        generated_after = min(generated_before + query_length, self.sliding_window)
        return prefill_length + generated_after


class UnlimitedOcrTextPreTrainedModel(DeepseekOcr2TextPreTrainedModel):
    pass


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
