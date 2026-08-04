# Copyright 2026 the MiniMax AI Team and HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...modeling_outputs import BaseModelOutputWithPooling
from ...processing_utils import ProcessingKwargs, Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ..auto import AutoConfig
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeRotaryEmbedding,
    apply_rotary_pos_emb,  # noqa: F401
)
from ..llava_next.configuration_llava_next import LlavaNextConfig
from ..llava_next.image_processing_llava_next import LlavaNextImageProcessor, LlavaNextImageProcessorKwargs
from ..llava_next.image_processing_pil_llava_next import LlavaNextImageProcessorPil
from ..llava_next.modeling_llava_next import (
    LlavaNextCausalLMOutputWithPast,
    LlavaNextForConditionalGeneration,
    LlavaNextModel,
    LlavaNextModelOutputWithPast,
    LlavaNextMultiModalProjector,
    LlavaNextPreTrainedModel,
    get_anyres_image_grid_shape,
    image_size_to_num_patches,
)
from ..llava_next.processing_llava_next import LlavaNextProcessor
from ..minimax.configuration_minimax import MiniMaxConfig
from ..minimax.modeling_minimax import MiniMaxCache, MiniMaxLightningAttention, MiniMaxModel


MINIMAX_VL_01_IMAGE_GRID_PINPOINTS = [
    [336, 336],
    [336, 672],
    [336, 1008],
    [336, 1344],
    [336, 1680],
    [336, 2016],
    [672, 336],
    [672, 672],
    [672, 1008],
    [672, 1344],
    [672, 1680],
    [672, 2016],
    [1008, 336],
    [1008, 672],
    [1008, 1008],
    [1008, 1344],
    [1008, 1680],
    [1008, 2016],
    [1344, 336],
    [1344, 672],
    [1344, 1008],
    [1344, 1344],
    [1680, 336],
    [1680, 672],
    [1680, 1008],
    [2016, 336],
    [2016, 672],
    [2016, 1008],
]


@auto_docstring(checkpoint="MiniMaxAI/MiniMax-VL-01")
@strict
class MiniMaxVL01TextConfig(MiniMaxConfig):
    model_type = "minimax_vl_01_text"
    base_config_key = "text_config"


class MiniMaxVL01TextCache(MiniMaxCache):
    def _get_attention_layer_idx(self, layer_idx: int) -> int:
        if layer_idx != 0 or DynamicCache.get_seq_length(self, layer_idx) > 0:
            return layer_idx

        # The released layout starts with recurrent layers, so layer 0 cannot provide the global sequence length.
        for attention_layer_idx in range(1, len(self.layers)):
            if DynamicCache.get_seq_length(self, attention_layer_idx) > 0:
                return attention_layer_idx
        return layer_idx

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return DynamicCache.get_seq_length(self, self._get_attention_layer_idx(layer_idx))

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        return DynamicCache.get_mask_sizes(self, query_length, self._get_attention_layer_idx(layer_idx))

    def batch_repeat_interleave(self, repeats: int):
        for layer_idx in range(len(self)):
            if layer_idx < len(self.linear_cache) and isinstance(self.linear_cache[layer_idx], torch.Tensor):
                self.linear_cache[layer_idx] = self.linear_cache[layer_idx].repeat_interleave(repeats, dim=0)
            elif layer_idx < len(self.layers) and self.layers[layer_idx].is_initialized:
                self.layers[layer_idx].batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        for layer_idx in range(len(self)):
            if layer_idx < len(self.linear_cache) and isinstance(self.linear_cache[layer_idx], torch.Tensor):
                self.linear_cache[layer_idx] = self.linear_cache[layer_idx][indices, ...]
            elif layer_idx < len(self.layers) and self.layers[layer_idx].is_initialized:
                self.layers[layer_idx].batch_select_indices(indices)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.batch_select_indices(beam_idx)


class MiniMaxVL01TextLightningAttention(MiniMaxLightningAttention):
    pass


class MiniMaxVL01TextRotaryEmbedding(Glm4MoeRotaryEmbedding):
    pass


class MiniMaxVL01TextModel(MiniMaxModel):
    pass


def _migrate_legacy_text_config(text_config: dict) -> dict:
    """Translate the released remote-code MiniMax-Text-01 schema to the native MiniMax schema."""
    text_config = dict(text_config)
    model_type = text_config.get("model_type")
    legacy_keys = {
        "attn_type_list",
        "layernorm_full_attention_alpha",
        "layernorm_full_attention_beta",
        "layernorm_linear_attention_alpha",
        "layernorm_linear_attention_beta",
        "layernorm_mlp_alpha",
        "layernorm_mlp_beta",
        "postnorm",
        "rotary_dim",
        "shared_intermediate_size",
        "shared_moe_mode",
    }
    is_legacy = model_type in {"minimax_text_01", "MiniMaxText01"} or bool(legacy_keys & text_config.keys())
    if not is_legacy:
        return text_config

    if text_config.pop("postnorm", True) is not True:
        raise ValueError("MiniMax-VL-01 only supports the released MiniMax-Text-01 `postnorm=True` layout.")

    shared_intermediate_size = text_config.pop("shared_intermediate_size", [0])
    if isinstance(shared_intermediate_size, (list, tuple)):
        if len(shared_intermediate_size) != 1:
            raise ValueError("`shared_intermediate_size` must contain exactly one value.")
        shared_intermediate_size = shared_intermediate_size[0]
    if shared_intermediate_size != 0:
        raise ValueError("MiniMax-VL-01 does not support the experimental shared-MoE branch.")
    text_config.pop("shared_moe_mode", None)

    attention_types = text_config.pop("attn_type_list", None)
    if attention_types is not None:
        if any(attention_type not in (0, 1) for attention_type in attention_types):
            raise ValueError("`attn_type_list` entries must be 0 (linear) or 1 (full attention).")
        num_hidden_layers = text_config.get("num_hidden_layers", len(attention_types))
        if len(attention_types) != num_hidden_layers:
            raise ValueError(
                f"`attn_type_list` has {len(attention_types)} entries but `num_hidden_layers` is {num_hidden_layers}."
            )
        text_config["layer_types"] = [
            "linear_attention" if attention_type == 0 else "full_attention" for attention_type in attention_types
        ]

    legacy_factor_mapping = {
        "layernorm_full_attention_alpha": "full_attn_alpha_factor",
        "layernorm_full_attention_beta": "full_attn_beta_factor",
        "layernorm_linear_attention_alpha": "linear_attn_alpha_factor",
        "layernorm_linear_attention_beta": "linear_attn_beta_factor",
        "layernorm_mlp_alpha": "mlp_alpha_factor",
        "layernorm_mlp_beta": "mlp_beta_factor",
    }
    for legacy_name, native_name in legacy_factor_mapping.items():
        if legacy_name in text_config:
            legacy_value = text_config.pop(legacy_name)
            if native_name in text_config and text_config[native_name] != legacy_value:
                raise ValueError(f"Conflicting `{legacy_name}` and `{native_name}` values.")
            text_config[native_name] = legacy_value

    rotary_dim = text_config.pop("rotary_dim", None)
    rope_theta = text_config.pop("rope_theta", 1_000_000.0)
    if rotary_dim is not None:
        head_dim = text_config.get("head_dim")
        if head_dim is None:
            head_dim = text_config["hidden_size"] // text_config["num_attention_heads"]
        if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
            raise ValueError(f"`rotary_dim` must be a positive even value no larger than `head_dim` ({head_dim}).")
        native_rope_parameters = {
            "rope_type": "default",
            "rope_theta": rope_theta,
            "partial_rotary_factor": rotary_dim / head_dim,
        }
        if "rope_parameters" in text_config and text_config["rope_parameters"] != native_rope_parameters:
            raise ValueError("Conflicting legacy RoPE fields and `rope_parameters` values.")
        text_config["rope_parameters"] = native_rope_parameters
    else:
        text_config.setdefault("rope_parameters", {"rope_type": "default", "rope_theta": rope_theta})

    text_config.setdefault("bos_token_id", None)
    text_config.setdefault("eos_token_id", None)
    text_config["model_type"] = "minimax"
    if text_config.get("architectures") == ["MiniMaxText01ForCausalLM"]:
        text_config["architectures"] = ["MiniMaxForCausalLM"]
    return text_config


@auto_docstring(checkpoint="MiniMaxAI/MiniMax-VL-01")
@strict
class MiniMaxVL01Config(LlavaNextConfig):
    r"""
    image_grid_pinpoints (`list[list[int]]`, *optional*):
        Candidate `(height, width)` resolutions used by the any-resolution image processor. By default this uses the
        28 resolutions released with MiniMax-VL-01.
    ignore_index (`int`, *optional*, defaults to `-100`):
        Label value ignored by the causal language modeling loss.
    """

    model_type = "minimax_vl_01"
    attribute_map = {"image_token_id": "image_token_index"}
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    image_token_index: int = 200025
    projector_hidden_act: str = "gelu"
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int | list[int] = -1
    multimodal_projector_bias: bool = True
    tie_word_embeddings: bool = False
    image_grid_pinpoints: list | None = None
    image_seq_length: int = 576
    ignore_index: int = -100

    def __post_init__(self, **kwargs):
        if self.vision_feature_select_strategy != "default":
            raise ValueError("MiniMax-VL-01 supports only `vision_feature_select_strategy='default'`.")

        self.image_grid_pinpoints = (
            self.image_grid_pinpoints
            if self.image_grid_pinpoints is not None
            else [pinpoint.copy() for pinpoint in MINIMAX_VL_01_IMAGE_GRID_PINPOINTS]
        )

        if isinstance(self.vision_config, dict):
            vision_config = dict(self.vision_config)
            vision_config["model_type"] = vision_config.get("model_type", "clip_vision_model")
            self.vision_config = AutoConfig.for_model(vision_config.pop("model_type"), **vision_config)
        elif self.vision_config is None:
            self.vision_config = AutoConfig.for_model(
                "clip_vision_model",
                hidden_act="gelu",
                hidden_size=1024,
                image_size=336,
                intermediate_size=4096,
                num_attention_heads=16,
                num_hidden_layers=24,
                patch_size=14,
                projection_dim=6144,
                vocab_size=32000,
            )
        if (
            not isinstance(self.vision_config, PreTrainedConfig)
            or self.vision_config.model_type != "clip_vision_model"
        ):
            raise TypeError("`vision_config` must be a dictionary or a native CLIP vision configuration instance.")

        if isinstance(self.text_config, dict):
            migrated_text_config = _migrate_legacy_text_config(self.text_config)
            self.text_config = AutoConfig.for_model(
                migrated_text_config.pop("model_type", "minimax"), **migrated_text_config
            )
        elif self.text_config is None:
            layernorm_factor = 3.5565588200778455
            self.text_config = AutoConfig.for_model(
                "minimax",
                vocab_size=200064,
                hidden_size=6144,
                intermediate_size=9216,
                num_hidden_layers=80,
                num_attention_heads=64,
                num_key_value_heads=8,
                head_dim=128,
                max_position_embeddings=8192,
                rms_norm_eps=1e-5,
                bos_token_id=None,
                eos_token_id=None,
                num_experts_per_tok=2,
                num_local_experts=32,
                layer_types=[
                    "full_attention" if (layer_idx + 1) % 8 == 0 else "linear_attention" for layer_idx in range(80)
                ],
                full_attn_alpha_factor=layernorm_factor,
                full_attn_beta_factor=1.0,
                linear_attn_alpha_factor=layernorm_factor,
                linear_attn_beta_factor=1.0,
                mlp_alpha_factor=layernorm_factor,
                mlp_beta_factor=1.0,
                rope_parameters={
                    "rope_type": "default",
                    "rope_theta": 10_000_000,
                    "partial_rotary_factor": 0.5,
                },
            )
        if not isinstance(self.text_config, PreTrainedConfig) or self.text_config.model_type != "minimax":
            raise TypeError("`text_config` must be a dictionary or a native MiniMax configuration instance.")

        PreTrainedConfig.__post_init__(self, **kwargs)


class MiniMaxVL01ImageProcessorKwargs(LlavaNextImageProcessorKwargs, total=False):
    r"""
    process_image_mode (`str`, *optional*, defaults to `"anyres"`):
        Image processing mode. The released checkpoint supports only `"anyres"`.
    patch_size (`int`, *optional*, defaults to `14`):
        Patch size of the CLIP vision encoder, used when expanding image placeholders.
    """

    process_image_mode: str
    patch_size: int


class MiniMaxVL01ImageProcessor(LlavaNextImageProcessor):
    model_input_names = ["pixel_values", "image_sizes"]
    valid_kwargs = MiniMaxVL01ImageProcessorKwargs
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 336, "width": 336}
    crop_size = None
    default_to_square = False
    do_resize = True
    do_center_crop = False
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_pad = False
    image_grid_pinpoints = MINIMAX_VL_01_IMAGE_GRID_PINPOINTS
    process_image_mode = "anyres"
    patch_size = 14

    def __init__(self, **kwargs: Unpack[MiniMaxVL01ImageProcessorKwargs]):
        if kwargs.get("process_image_mode", self.process_image_mode) != "anyres":
            raise ValueError("MiniMax-VL-01 supports only `process_image_mode='anyres'`.")
        super().__init__(**kwargs)

    def _resize_for_patching(
        self,
        image: "torch.Tensor",
        target_resolution: tuple,
        resample: "PILImageResampling | int | None",
        input_data_format: ChannelDimension,
    ) -> "torch.Tensor":
        original_height, original_width = get_image_size(image, channel_dim=input_data_format)
        target_height, target_width = target_resolution
        if original_width / original_height > target_width / target_height:
            new_width = target_width
            new_height = int(target_width * original_height / original_width)
        else:
            new_height = target_height
            new_width = int(target_height * original_width / original_height)
        return self.resize(
            image=image,
            size=SizeDict(height=new_height, width=new_width),
            resample=resample,
        )

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        image_grid_pinpoints: list[list[int]],
        resample: "PILImageResampling | int | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        if kwargs.get("process_image_mode", self.process_image_mode) != "anyres":
            raise ValueError("MiniMax-VL-01 supports only `process_image_mode='anyres'`.")

        processed_images = []
        image_sizes = []
        if size and size.height and size.width:
            size_tuple = (size.height, size.width)
        else:
            size_tuple = (size.shortest_edge, size.shortest_edge)
        patch_size = size.height if size and size.height else size.shortest_edge

        for image in images:
            image_patches = self._get_image_patches(
                image,
                image_grid_pinpoints,
                size=size_tuple,
                patch_size=patch_size,
                resample=resample,
            )
            processed_image_patches_grouped = {}
            grouped_image_patches, grouped_image_patches_index = group_images_by_shape(
                image_patches, disable_grouping=disable_grouping
            )
            for shape, stacked_image_patches in grouped_image_patches.items():
                if do_resize:
                    stacked_image_patches = self.resize(image=stacked_image_patches, size=size, resample=resample)
                if do_center_crop:
                    stacked_image_patches = self.center_crop(stacked_image_patches, crop_size)
                image_mean_tuple = tuple(image_mean) if isinstance(image_mean, list) else image_mean
                image_std_tuple = tuple(image_std) if isinstance(image_std, list) else image_std
                stacked_image_patches = self.rescale_and_normalize(
                    stacked_image_patches, do_rescale, rescale_factor, do_normalize, image_mean_tuple, image_std_tuple
                )
                processed_image_patches_grouped[shape] = stacked_image_patches
            processed_image_patches = reorder_images(processed_image_patches_grouped, grouped_image_patches_index)
            processed_images.append(torch.stack(processed_image_patches, dim=0))
            image_sizes.append(get_image_size(image, ChannelDimension.FIRST))

        pixel_values = torch.cat(processed_images, dim=0)
        return BatchFeature(
            data={"pixel_values": pixel_values, "image_sizes": image_sizes}, tensor_type=return_tensors
        )


class MiniMaxVL01ImageProcessorPil(LlavaNextImageProcessorPil):
    model_input_names = ["pixel_values", "image_sizes"]
    valid_kwargs = MiniMaxVL01ImageProcessorKwargs
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 336, "width": 336}
    crop_size = None
    default_to_square = False
    do_resize = True
    do_center_crop = False
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_pad = False
    image_grid_pinpoints = MINIMAX_VL_01_IMAGE_GRID_PINPOINTS
    process_image_mode = "anyres"
    patch_size = 14

    def __init__(self, **kwargs: Unpack[MiniMaxVL01ImageProcessorKwargs]):
        if kwargs.get("process_image_mode", self.process_image_mode) != "anyres":
            raise ValueError("MiniMax-VL-01 supports only `process_image_mode='anyres'`.")
        super().__init__(**kwargs)

    def _resize_for_patching(
        self,
        image: np.ndarray,
        target_resolution: tuple,
        resample: PILImageResampling,
    ) -> np.ndarray:
        original_height, original_width = image.shape[-2:]
        target_height, target_width = target_resolution
        if original_width / original_height > target_width / target_height:
            new_width = target_width
            new_height = int(target_width * original_height / original_width)
        else:
            new_height = target_height
            new_width = int(target_height * original_width / original_height)
        return self.resize(
            image=image,
            size=SizeDict(height=new_height, width=new_width),
            resample=resample,
        )

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        image_grid_pinpoints: list[list[int]],
        resample: "PILImageResampling | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        if kwargs.get("process_image_mode", self.process_image_mode) != "anyres":
            raise ValueError("MiniMax-VL-01 supports only `process_image_mode='anyres'`.")

        processed_images = []
        image_sizes = []
        if size and size.height and size.width:
            size_tuple = (size.height, size.width)
        else:
            size_tuple = (size.shortest_edge, size.shortest_edge)
        patch_size = size.height if size and size.height else size.shortest_edge

        for image in images:
            image_patches = self.get_image_patches(
                image,
                image_grid_pinpoints,
                size=size_tuple,
                patch_size=patch_size,
                resample=resample,
            )
            pixel_values = []
            for patch in image_patches:
                if do_resize:
                    patch = self.resize(image=patch, size=size, resample=resample)
                if do_center_crop:
                    patch = self.center_crop(image=patch, size=crop_size)
                if do_rescale:
                    patch = self.rescale(image=patch, scale=rescale_factor)
                if do_normalize:
                    patch = self.normalize(image=patch, mean=image_mean, std=image_std)
                pixel_values.append(patch)
            processed_images.append(np.asarray(pixel_values))
            image_sizes.append(image.shape[-2:])

        pixel_values = np.concatenate(processed_images, axis=0)
        return BatchFeature(
            data={"pixel_values": pixel_values, "image_sizes": image_sizes}, tensor_type=return_tensors
        )


class MiniMaxVL01ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


class MiniMaxVL01Processor(LlavaNextProcessor):
    valid_processor_kwargs = MiniMaxVL01ProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        vision_feature_select_strategy="default",
        chat_template=None,
        image_token="<image>",
        num_additional_image_tokens=1,
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*):
            Patch size from the vision tower. Defaults to the image processor's `patch_size`.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            Vision feature selection strategy used by the released MiniMax-VL-01 checkpoint.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote an image location.
        num_additional_image_tokens (`int`, *optional*, defaults to `1`):
            Number of non-spatial tokens emitted by CLIP before the `"default"` strategy removes its CLS token.
        """
        if vision_feature_select_strategy != "default":
            raise ValueError("MiniMax-VL-01 supports only `vision_feature_select_strategy='default'`.")
        if image_processor is not None:
            if getattr(image_processor, "process_image_mode", "anyres") != "anyres":
                raise ValueError("MiniMax-VL-01 supports only `process_image_mode='anyres'`.")
            if patch_size is None:
                patch_size = image_processor.patch_size
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=patch_size,
            vision_feature_select_strategy=vision_feature_select_strategy,
            chat_template=chat_template,
            image_token=image_token,
            num_additional_image_tokens=num_additional_image_tokens,
            **kwargs,
        )

    def _get_unpadded_features(self, height, width, patches_height, patches_width, scale_height, scale_width):
        current_height = patches_height * scale_height
        current_width = patches_width * scale_width

        original_aspect_ratio = width / height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            new_height = int(height * current_width) // width
            padding = (current_height - new_height) // 2
            current_height -= padding * 2
        else:
            new_width = int(width * current_height) // height
            padding = (current_width - new_width) // 2
            current_width -= padding * 2

        unpadded_features = current_height * current_width
        newline_features = current_height
        return unpadded_features, newline_features


def unpad_image(tensor, original_size):
    """Unpad spatial features using the floor rounding of the released MiniMax-VL-01 implementation."""
    if not isinstance(original_size, (list, tuple)):
        if not isinstance(original_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(original_size)} not valid, should be a list, tuple, array, or tensor"
            )
        original_size = original_size.tolist()
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height
    if original_aspect_ratio > current_aspect_ratio:
        new_height = int(original_height * current_width) // original_width
        padding = (current_height - new_height) // 2
        return tensor[:, padding : current_height - padding, :]

    new_width = int(original_width * current_height) // original_height
    padding = (current_width - new_width) // 2
    return tensor[:, :, padding : current_width - padding]


@auto_docstring(custom_intro="Base class for MiniMax-VL-01 model outputs.")
@dataclass
class MiniMaxVL01ModelOutputWithPast(LlavaNextModelOutputWithPast):
    r"""
    image_hidden_states (`torch.FloatTensor` of shape `(num_image_tokens, hidden_size)`, *optional*):
        Packed and projected image features inserted at image placeholder positions.
    """


@auto_docstring(custom_intro="Base class for MiniMax-VL-01 causal language model outputs.")
@dataclass
class MiniMaxVL01CausalLMOutputWithPast(LlavaNextCausalLMOutputWithPast):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
        Causal language modeling loss, returned when `labels` are provided.
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Float32 prediction scores before softmax.
    image_hidden_states (`torch.FloatTensor` of shape `(num_image_tokens, hidden_size)`, *optional*):
        Packed and projected image features inserted at image placeholder positions.
    """


class MiniMaxVL01MultiModalProjector(LlavaNextMultiModalProjector):
    pass


class MiniMaxVL01PreTrainedModel(LlavaNextPreTrainedModel):
    _no_split_modules = ["MiniMaxVL01TextDecoderLayer", "CLIPEncoderLayer"]
    _can_compile_fullgraph = False
    # The remote checkpoint does not store MiniMax's deterministic Lightning-Attention buffers. Its custom CLIP also
    # omits the final pooling LayerNorm, which is unused because MiniMax-VL-01 consumes encoder hidden states.
    _keys_to_ignore_on_load_missing = [
        r"(?:model\.)?language_model\.layers\.\d+\.self_attn\.(slope_rate|query_decay|key_decay|diagonal_decay)",
        r"(?:model\.)?vision_tower\.post_layernorm\.(weight|bias)",
    ]


@auto_docstring(
    custom_intro="The MiniMax-VL-01 model, consisting of a CLIP vision encoder and a MiniMax language backbone."
)
class MiniMaxVL01Model(LlavaNextModel):
    config_class = MiniMaxVL01Config

    @staticmethod
    def _flatten_pixel_values(pixel_values):
        if not isinstance(pixel_values, (list, tuple)):
            return pixel_values
        if not pixel_values:
            raise ValueError("`pixel_values` cannot be an empty list.")
        if not all(isinstance(value, torch.Tensor) for value in pixel_values):
            raise TypeError("List-form `pixel_values` must contain PyTorch tensors.")
        if all(value.ndim == 3 for value in pixel_values):
            return torch.stack(list(pixel_values), dim=0)
        if all(value.ndim == 4 for value in pixel_values):
            return torch.cat(list(pixel_values), dim=0)
        raise ValueError("List-form `pixel_values` must contain only 3D patches or only 4D patch batches.")

    def pack_image_features(self, image_features, image_sizes, vision_feature_select_strategy, image_newline=None):
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                if height * width != base_image_feature.shape[0]:
                    raise ValueError("The number of vision features is inconsistent with the configured image size.")

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
    @auto_docstring(
        custom_intro="Obtains image hidden states from the vision tower and applies multimodal projection."
    )
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor | list[torch.FloatTensor],
        image_sizes: torch.Tensor,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` or `list[torch.FloatTensor]`):
            A flat 4D patch tensor, a padded 5D patch tensor, or a list of 3D/4D patch tensors.
        image_sizes (`torch.Tensor` of shape `(num_images, 2)`):
            Original image sizes in `(height, width)` order.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            Feature selection strategy. MiniMax-VL-01 supports only `"default"`, which removes CLIP's CLS token.
        """
        if vision_feature_select_strategy != "default":
            raise ValueError("MiniMax-VL-01 supports only `vision_feature_select_strategy='default'`.")
        pixel_values = self._flatten_pixel_values(pixel_values)
        image_num_patches = [
            image_size_to_num_patches(
                image_size=image_size,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for image_size in image_sizes
        ]
        if pixel_values.dim() == 5:
            pixel_values = torch.cat(
                [pixel_value[:num_patch] for pixel_value, num_patch in zip(pixel_values, image_num_patches)], dim=0
            )
        elif pixel_values.dim() != 4:
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expected 4 or 5 dimensions")

        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, return_dict=True, **kwargs)
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        else:
            selected_image_feature = torch.cat(
                [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer], dim=-1
            )
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]

        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = torch.split(image_features, image_num_patches, dim=0)
        image_features, _ = self.pack_image_features(
            image_features,
            image_sizes,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_newline=self.image_newline,
        )
        image_outputs.pooler_output = image_features
        return image_outputs

    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | list[torch.FloatTensor] | None = None,
        image_sizes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MiniMaxVL01ModelOutputWithPast:
        r"""
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            Feature selection strategy. MiniMax-VL-01 supports only `"default"`, which removes CLIP's CLS token.
        """
        if vision_feature_select_strategy != "default":
            raise ValueError("MiniMax-VL-01 supports only `vision_feature_select_strategy='default'`.")
        pixel_values = self._flatten_pixel_values(pixel_values)
        if pixel_values is not None and pixel_values.numel() == 0:
            pixel_values = None
        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            use_cache=use_cache,
            **kwargs,
        )


@auto_docstring(
    custom_intro="MiniMax-VL-01 with a causal language modeling head for image-conditioned text generation."
)
class MiniMaxVL01ForConditionalGeneration(LlavaNextForConditionalGeneration):
    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor | list[torch.FloatTensor],
        image_sizes: torch.Tensor,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` or `list[torch.FloatTensor]`):
            A flat 4D patch tensor, a padded 5D patch tensor, or a list of 3D/4D patch tensors.
        image_sizes (`torch.Tensor` of shape `(num_images, 2)`):
            Original image sizes in `(height, width)` order.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            Feature selection strategy. MiniMax-VL-01 supports only `"default"`, which removes CLIP's CLS token.
        """
        return self.model.get_image_features(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            **kwargs,
        )

    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | list[torch.FloatTensor] | None = None,
        image_sizes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MiniMaxVL01CausalLMOutputWithPast:
        r"""
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            Feature selection strategy. MiniMax-VL-01 supports only `"default"`, which removes CLIP's CLS token.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for causal language modeling. Padding positions excluded by `attention_mask` and labels equal to
            `config.ignore_index` are ignored.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs[0]
        if labels is not None:
            slice_indices = slice(None)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]).float()

        loss = None
        if labels is not None:
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:].to(logits.device) != 0
                shift_logits = logits[..., :-1, :][shift_attention_mask].contiguous()
                shift_labels = labels[..., 1:].to(logits.device)[shift_attention_mask].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].to(logits.device).contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.ignore_index,
            )

        return MiniMaxVL01CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "MiniMaxVL01Config",
    "MiniMaxVL01ImageProcessor",
    "MiniMaxVL01ImageProcessorPil",
    "MiniMaxVL01Processor",
    "MiniMaxVL01PreTrainedModel",
    "MiniMaxVL01Model",
    "MiniMaxVL01ForConditionalGeneration",
    "MiniMaxVL01TextConfig",
    "MiniMaxVL01TextModel",
]
