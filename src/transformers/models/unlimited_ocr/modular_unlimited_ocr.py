# coding=utf-8
# Copyright 2023-2026 DeepSeek-AI, Baidu, and The HuggingFace Inc. team.
# Consolidated modular source for transformers.models.unlimited_ocr.
"""PyTorch UnlimitedOCR model (vision encoder + DeepSeek-V2 + UnlimitedOCR)."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict
from torchvision.transforms.v2 import functional as tvF

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import ImageInput, SizeDict, get_image_size
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TensorType, auto_docstring, can_return_tuple, logging
from ...utils.generic import TransformersKwargs
from ...utils.import_utils import requires
from ..clip.configuration_clip import CLIPVisionConfig
from ..clip.modeling_clip import (
    CLIPAttention,
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPMLP,
    CLIPVisionEmbeddings,
)
from ..deepseek_ocr2.configuration_deepseek_ocr2 import DeepseekOcr2Config, DeepseekOcr2VisionConfig
from ..deepseek_ocr2.image_processing_deepseek_ocr2 import (
    DeepseekOcr2ImageProcessor,
    DeepseekOcr2ImageProcessorKwargs,
    get_optimal_tiled_canvas,
)
from ..deepseek_ocr2.image_processing_pil_deepseek_ocr2 import DeepseekOcr2ImageProcessorPil
from ..deepseek_ocr2.modeling_deepseek_ocr2 import (
    DeepseekOcr2CausalLMOutputWithPast,
    DeepseekOcr2ForConditionalGeneration,
    DeepseekOcr2Model,
    DeepseekOcr2ModelOutputWithPast,
    DeepseekOcr2ModelOutputWithPooling,
    DeepseekOcr2SamPatchEmbeddings,
    DeepseekOcr2SamVisionEncoder,
    DeepseekOcr2SamVisionProj,
    DeepseekOcr2TextAttention,
    DeepseekOcr2TextDecoderLayer,
    DeepseekOcr2TextModel,
)
from ..llava_next.modeling_llava_next import LlavaNextPreTrainedModel
from ..dots1.configuration_dots1 import Dots1Config
from ..deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2MLP,
    DeepseekV2Moe,
    DeepseekV2PreTrainedModel,
    DeepseekV2RMSNorm,
)
from ..llama.modeling_llama import LlamaRotaryEmbedding
from ..sam.configuration_sam import SamVisionConfig
from ..sam.modeling_sam import (
    SamMLPBlock,
    SamVisionAttention,
    SamVisionLayer,
    SamVisionNeck,
)
from ..vitdet.modeling_vitdet import VitDetLayerNorm

logger = logging.get_logger(__name__)


# ============== configuration ==============

@auto_docstring
@strict
class UnlimitedOCRTextConfig(Dots1Config):
    r"""
    Text-backbone config for UnlimitedOCR (DeepSeek-V2 MoE + MHA).

    `layer_types` defaults to all-`full_attention` when unset so hub checkpoints that set
    `sliding_window` but omit `layer_types` keep bit-identical logits with the historical load path.
    """

    model_type = "unlimited_ocr_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_experts": "n_routed_experts",
        "sliding_window_size": "sliding_window",
    }

    vocab_size: int = 102400
    hidden_size: int = 4096
    intermediate_size: int = 11008
    moe_intermediate_size: int = 1407
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 32
    n_shared_experts: int | None = None
    n_routed_experts: int | None = None
    routed_scaling_factor: float = 1.0
    num_experts_per_tok: int | None = None
    first_k_dense_replace: int | None = 0
    bos_token_id: int | None = 100000
    eos_token_id: int | list[int] | None = 100001
    sliding_window: int | None = 0
    attention_bias: bool = False
    attention_dropout: float | int | None = 0.0
    mlp_bias: bool = False
    qkv_bias: bool = False
    aux_loss_alpha: float = 0.001
    seq_aux: bool = True
    pretraining_tp: int = 1
    topk_method: str | None = "greedy"
    topk_group: int | None = None
    norm_topk_prob: bool | None = False
    mlp_layer_types: list[str] | None = None
    # Disable Dots1's max_window_layers-based layer_types schedule.
    max_window_layers: int | None = None

    def __post_init__(self, **kwargs):
        self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.sliding_window is None:
            self.sliding_window = 0
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.mlp_layer_types is None:
            first_k = self.first_k_dense_replace or 0
            self.mlp_layer_types = [
                "dense" if i < first_k else "sparse" for i in range(self.num_hidden_layers)
            ]
        if self.rope_parameters is None:
            rope_theta = kwargs.pop("rope_theta", None) or 10000.0
            self.rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        self.qkv_bias = self.attention_bias
        # Skip Dots1Config.__post_init__ layer_types rewrite.
        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring
@strict
class UnlimitedOCRVisionEncoderConfig(CLIPVisionConfig):
    """CLIP-L tower config (`encoder_config` slot, DeepseekOcr2-style)."""

    model_type = "unlimited_ocr_vision_encoder"
    base_config_key = "encoder_config"
    attribute_map = {
        "num_layers": "num_hidden_layers",
        "ffn_hidden_size": "intermediate_size",
        "layernorm_epsilon": "layer_norm_eps",
    }

    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    image_size: int = 224
    patch_size: int = 14
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    # Hub-only fields kept for checkpoint / preprocessing compatibility.
    seq_length: int = 256
    pre_layernorm_epsilon: float = 1e-5


# Modular converter remaps CLIPAttention's `CLIPVisionConfig | CLIPTextConfig` union using the
# `UnlimitedOCRClipVision*` modeling prefix (not `VisionEncoder*`).
class UnlimitedOCRClipVisionVisionConfig(UnlimitedOCRVisionEncoderConfig):
    pass


class UnlimitedOCRClipVisionTextConfig(UnlimitedOCRVisionEncoderConfig):
    pass


@auto_docstring
@strict
class UnlimitedOCRSamVisionConfig(SamVisionConfig):
    r"""
    downsample_channels (`list[int]`, *optional*):
        Channel sizes for the two stride-2 convs after the SAM neck. Defaults to `[512, 1024]`.
    """

    # Explicit: converter CamelCase→snake of UnlimitedOCR incorrectly yields `unlimited_o_c_r_*`.
    model_type = "unlimited_ocr_sam_vision_model"
    base_config_key = "sam_config"
    num_pos_feats = AttributeError()
    downsample_channels: list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.downsample_channels is None:
            self.downsample_channels = [512, 1024]
        super().__post_init__(**kwargs)


@auto_docstring
@strict
class UnlimitedOCRVisionConfig(DeepseekOcr2VisionConfig):
    model_type = "unlimited_ocr_vision"


@auto_docstring
@strict
class UnlimitedOCRConfig(DeepseekOcr2Config):
    r"""
    Same layout as `DeepseekOcr2Config`. Projector input size is
    `vision_config.encoder_config.hidden_size + vision_config.sam_config.downsample_channels[-1]`
    (CLIP-L features concatenated with SAM downsample channels).
    """

    model_type = "unlimited_ocr"
    attribute_map = {"language_config": "text_config"}
    image_token_id: int = 128815

    def __post_init__(self, **kwargs):
        # Hub ships an opaque `vision_config` blob (`model_type: "vision"`); use defaults instead.
        if isinstance(self.vision_config, dict) and self.vision_config.get("model_type") == "vision":
            self.vision_config = None
        # Hub uses `language_config`; pop before PreTrainedConfig setattr (attribute_map) overwrites
        # a materialized `text_config` with the raw dict.
        if (language_config := kwargs.pop("language_config", None)) is not None and self.text_config is None:
            self.text_config = language_config
        # Hub-only nested projector blob; size is derived from vision configs in the model.
        kwargs.pop("projector_config", None)
        super().__post_init__(**kwargs)


# ============== image_processing ==============

class UnlimitedOCRImageProcessorKwargs(DeepseekOcr2ImageProcessorKwargs, total=False):
    r"""
    crop_to_patches (`bool`, *optional*, defaults to `self.crop_to_patches`):
        Whether to tile the image into local crops on top of the padded global view.
    min_patches (`int`, *optional*, defaults to `self.min_patches`):
        Minimum number of local crops. Only used when `crop_to_patches=True`.
    max_patches (`int`, *optional*, defaults to `self.max_patches`):
        Maximum number of local crops. Only used when `crop_to_patches=True`.
    tile_size (`int`, *optional*, defaults to 640):
        Side length of each local tile. Must match the model's query embedding size.
    background_color (`list[int]`, *optional*, defaults to `[127, 127, 127]`):
        Fill used when padding the global view to a square.
    """


@auto_docstring
class UnlimitedOCRImageProcessor(DeepseekOcr2ImageProcessor):
    valid_kwargs = UnlimitedOCRImageProcessorKwargs
    max_patches = 32
    tile_size = 640
    model_input_names = ["pixel_values", "pixel_values_local", "num_local_patches"]

    def pad_to_square(
        self, images: "torch.Tensor", background_color: int | tuple[int, int, int] = 0
    ) -> "torch.Tensor":
        """Pad to a square using `round(delta * 0.5)` offsets (matches `PIL.ImageOps.pad`)."""
        height, width = images.shape[-2:]
        if height == width:
            return images
        num_channels = images.shape[-3]
        if isinstance(background_color, int):
            background_color = [background_color] + [0] * (num_channels - 1)
        elif len(background_color) != num_channels:
            raise ValueError(
                f"background_color must have no more than {num_channels} elements to match the number of channels"
            )
        max_dim = max(height, width)
        left = round((max_dim - width) * 0.5)
        top = round((max_dim - height) * 0.5)
        return tvF.pad(images, padding=[left, top, max_dim - width - left, max_dim - height - top], fill=background_color)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        size: SizeDict,
        crop_to_patches: bool,
        min_patches: int,
        max_patches: int,
        tile_size: int,
        resample,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean,
        image_std,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        grouped_global, grouped_local, grouped_grid = {}, {}, {}
        global_target = size.height if crop_to_patches else tile_size
        for shape, stacked in grouped_images.items():
            height, width = stacked.shape[-2:]
            batch_size = stacked.shape[0]
            num_rows, num_columns = 1, 1
            if crop_to_patches and max(height, width) > tile_size:
                num_columns, num_rows = get_optimal_tiled_canvas(
                    (height, width), (tile_size, tile_size), min_patches, max_patches
                )
                patches, _ = self.crop_image_to_patches(stacked, min_patches, max_patches, tile_size, resample)
                grouped_local[shape] = self.rescale_and_normalize(
                    patches.flatten(0, 1), do_rescale, rescale_factor, do_normalize, image_mean, image_std
                ).reshape(patches.shape)
            else:
                grouped_local[shape] = [None] * batch_size
            grouped_grid[shape] = [[num_rows, num_columns]] * batch_size

            scale = global_target / max(height, width)
            global_images = self.resize(
                stacked, SizeDict(height=round(height * scale), width=round(width * scale)), resample=resample
            )
            global_images = self.pad_to_square(global_images, background_color=self.background_color)
            grouped_global[shape] = self.rescale_and_normalize(
                global_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )

        local_patches = reorder_images(grouped_local, grouped_index)
        data = {
            "pixel_values": reorder_images(grouped_global, grouped_index),
            "num_local_patches": reorder_images(grouped_grid, grouped_index),
        }
        if any(patches is not None for patches in local_patches):
            data["pixel_values_local"] = [
                patch for patches in local_patches if patches is not None for patch in patches
            ]
        return BatchFeature(data=data, tensor_type=return_tensors)


@requires(backends=("vision",))
@auto_docstring
class UnlimitedOCRImageProcessorPil(DeepseekOcr2ImageProcessorPil):
    valid_kwargs = UnlimitedOCRImageProcessorKwargs
    max_patches = 32
    tile_size = 640
    model_input_names = ["pixel_values", "pixel_values_local", "num_local_patches"]

    def pad_to_square(self, image: np.ndarray, background_color: int | tuple[int, int, int] = 0) -> np.ndarray:
        """Pad to a square using `round(delta * 0.5)` offsets (matches `PIL.ImageOps.pad`)."""
        num_channels, height, width = image.shape
        if height == width:
            return image
        if isinstance(background_color, int):
            background_color = [background_color]
        elif len(background_color) != num_channels:
            raise ValueError(
                f"background_color must have no more than {num_channels} elements to match the number of channels"
            )
        max_dim = max(height, width)
        padded = np.empty((num_channels, max_dim, max_dim), dtype=image.dtype)
        padded[:] = np.reshape(background_color, (-1, 1, 1))
        top = round((max_dim - height) * 0.5)
        left = round((max_dim - width) * 0.5)
        padded[:, top : top + height, left : left + width] = image
        return padded

    def _preprocess(
        self,
        images: list[np.ndarray],
        size: SizeDict,
        resample,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean,
        image_std,
        return_tensors: str | TensorType | None,
        crop_to_patches: bool = True,
        min_patches: int = 2,
        max_patches: int = 32,
        tile_size: int = 640,
        background_color: list[int] | None = None,
        **kwargs,
    ) -> BatchFeature:
        if background_color is None:
            background_color = self.background_color
        global_target = size.height if crop_to_patches else tile_size

        pixel_values, pixel_values_local, num_local_patches = [], [], []
        for image in images:
            height, width = get_image_size(image)
            num_rows, num_columns = 1, 1
            if crop_to_patches and max(height, width) > tile_size:
                num_columns, num_rows = get_optimal_tiled_canvas(
                    (height, width), (tile_size, tile_size), min_patches, max_patches
                )
                for patch in self.crop_image_to_patches(image, min_patches, max_patches, tile_size, resample):
                    if do_rescale:
                        patch = self.rescale(patch, rescale_factor)
                    if do_normalize:
                        patch = self.normalize(patch, image_mean, image_std)
                    pixel_values_local.append(patch)
            num_local_patches.append([num_rows, num_columns])

            scale = global_target / max(height, width)
            global_image = self.resize(
                image, SizeDict(height=round(height * scale), width=round(width * scale)), resample=resample
            )
            global_image = self.pad_to_square(global_image, background_color=background_color)
            if do_rescale:
                global_image = self.rescale(global_image, rescale_factor)
            if do_normalize:
                global_image = self.normalize(global_image, image_mean, image_std)
            pixel_values.append(global_image)

        data = {"pixel_values": pixel_values, "num_local_patches": num_local_patches}
        if pixel_values_local:
            data["pixel_values_local"] = pixel_values_local
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["UnlimitedOCRImageProcessor", "UnlimitedOCRImageProcessorPil"]

# ============== processing ==============

class UnlimitedOCRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "crop_to_patches": True,
            "min_patches": 2,
            "max_patches": 32,
        },
    }


@auto_docstring
class UnlimitedOCRProcessor(ProcessorMixin):
    valid_processor_kwargs = UnlimitedOCRProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        patch_size=16,
        downsample_ratio=4,
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*, defaults to `16`):
            The patch size used by the vision encoder.
        downsample_ratio (`int`, *optional*, defaults to `4`):
            The downsampling ratio applied after the vision encoder.
        """
        self.image_token = "<image>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        """
        Expand one `<image>` placeholder into the token count emitted for that image.

        Each view is a token grid with one `image_newline` per row and a closing `view_separator`.
        A `num_rows x num_columns` local crop grid therefore adds
        `(num_rows * queries) * (num_columns * queries + 1)` tokens on top of the global view.
        """
        num_rows, num_columns = image_inputs["num_local_patches"][image_idx]
        num_rows, num_columns = int(num_rows), int(num_columns)

        num_queries_global = math.ceil(self.image_processor.size["height"] / self.patch_size / self.downsample_ratio)
        num_queries_local = math.ceil(self.image_processor.tile_size / self.patch_size / self.downsample_ratio)

        num_tokens = num_queries_global * (num_queries_global + 1) + 1
        if num_rows * num_columns > 1:
            num_tokens += (num_rows * num_queries_local) * (num_columns * num_queries_local + 1)
        return self.image_token * num_tokens

    def validate_inputs(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos=None,
        audio=None,
        **kwargs: Unpack[UnlimitedOCRProcessorKwargs],
    ):
        super().validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)
        if images is None:
            raise ValueError("`images` are expected as arguments to a `UnlimitedOCRProcessor` instance.")
        if text is None:
            raise ValueError("`text` is required for `UnlimitedOCRProcessor`. Example: `'<image>document parsing.'`")
        if isinstance(text, str):
            return
        if not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")


__all__ = ["UnlimitedOCRProcessor"]

# ============== modeling ==============


class UnlimitedOCRModelOutputWithPooling(DeepseekOcr2ModelOutputWithPooling):
    pass


class UnlimitedOCRModelOutputWithPast(DeepseekOcr2ModelOutputWithPast):
    pass


class UnlimitedOCRCausalLMOutputWithPast(DeepseekOcr2CausalLMOutputWithPast):
    pass


class UnlimitedOCRClipVisionEmbeddings(CLIPVisionEmbeddings):
    """CLIP embeddings with SAM patch-grid injection + DeepSeek-OCR-2-style pos interpolate."""

    def interpolate_pos_encoding(self, height: int, width: int) -> torch.Tensor:
        """`height` / `width` are patch-grid sizes (not pixels)."""
        position_embedding = self.position_embedding.weight.unsqueeze(0)
        num_positions = position_embedding.shape[1] - 1
        src_size = int(num_positions**0.5)

        if not torch.jit.is_tracing() and src_size == height and height == width:
            return self.position_embedding(self.position_ids)

        class_pos_embed = position_embedding[:, :1]
        patch_pos_embed = position_embedding[:, 1:]
        dim = position_embedding.shape[-1]
        patch_pos_embed = patch_pos_embed.reshape(1, src_size, src_size, dim).permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(torch.float32),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).to(dtype=target_dtype)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat([class_pos_embed, patch_pos_embed], dim=1)

    def forward(self, pixel_values, patch_embeds=None):
        batch_size = pixel_values.shape[0]
        if patch_embeds is None:
            patch_embeds = self.patch_embedding(pixel_values)
        patch_height, patch_width = patch_embeds.shape[-2:]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.interpolate_pos_encoding(patch_height, patch_width)
        return embeddings


class UnlimitedOCRClipVisionAttention(CLIPAttention):
    pass


class UnlimitedOCRClipVisionMLP(CLIPMLP):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # QuickGELU in fp32 then cast back: matches hub TorchScript quick_gelu under bf16.
        hidden_states = self.fc1(hidden_states)
        dtype = hidden_states.dtype
        hidden_states = self.activation_fn(hidden_states.float()).to(dtype)
        return self.fc2(hidden_states)


class UnlimitedOCRClipVisionEncoderLayer(CLIPEncoderLayer):
    pass


class UnlimitedOCRClipVisionEncoder(CLIPEncoder):
    """Weight path stays `transformer.layers.*` (hub layout); returns the last hidden state tensor."""

    def __init__(self, config: UnlimitedOCRVisionEncoderConfig):
        # Hub logit parity: always SDPA for CLIP-L (independent of LM `_attn_implementation`).
        config._attn_implementation = "sdpa"
        super().__init__(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Explicit loop (avoid `super().forward` — modular conversion flattens parents).
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=None)
        return hidden_states


class UnlimitedOCRVisionEncoder(nn.Module):
    def __init__(self, config: UnlimitedOCRVisionEncoderConfig) -> None:
        super().__init__()
        self.embeddings = UnlimitedOCRClipVisionEmbeddings(config)
        self.transformer = UnlimitedOCRClipVisionEncoder(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.pre_layernorm_epsilon)

    def forward(self, x, patch_embeds):
        x = self.embeddings(x, patch_embeds)
        hidden_states = self.pre_layrnorm(x)
        return self.transformer(hidden_states)


class UnlimitedOCRTextRMSNorm(DeepseekV2RMSNorm):
    pass


ALL_LAYERNORM_LAYERS.append(UnlimitedOCRTextRMSNorm)


class UnlimitedOCRTextMLP(DeepseekV2MLP):
    pass


class UnlimitedOCRTextMoe(DeepseekV2Moe):
    pass


class UnlimitedOCRTextRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class UnlimitedOCRTextAttention(DeepseekOcr2TextAttention):
    pass


class UnlimitedOCRTextDecoderLayer(DeepseekOcr2TextDecoderLayer):
    pass


class UnlimitedOCRTextPreTrainedModel(DeepseekV2PreTrainedModel):
    # Text tower stays on eager MHA for bit-identical logits vs hub.
    _supports_sdpa = False
    _supports_flash_attn = False
    _supports_flex_attn = False
    _supports_attention_backend = False


class UnlimitedOCRPreTrainedModel(LlavaNextPreTrainedModel):
    config: UnlimitedOCRConfig
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["UnlimitedOCRTextDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    # SAM rel-pos + text MHA stay off flash/sdpa backends for hub logit parity.
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_attention_backend = False
    _supports_cache_class = True

    @torch.no_grad()
    def _init_weights(self, module):
        # Skip LlavaNext's `isinstance(..., LlavaNextModel)` path so we own both separator inits.
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, UnlimitedOCRSamVisionAttention):
            if module.use_rel_pos:
                init.zeros_(module.rel_pos_h)
                init.zeros_(module.rel_pos_w)
        elif isinstance(module, UnlimitedOCRSamVisionEncoder):
            if module.pos_embed is not None:
                init.zeros_(module.pos_embed)
        elif isinstance(module, UnlimitedOCRModel):
            embed_std = 1 / math.sqrt(self.config.text_config.hidden_size)
            init.normal_(module.view_separator, mean=0.0, std=embed_std)
            init.normal_(module.image_newline, mean=0.0, std=embed_std)


#=========================Sam-Vary=================================

class UnlimitedOCRSamLayerNorm(VitDetLayerNorm):
    """Channels-first LayerNorm (Detectron2 / VitDet); bit-identical to hub LayerNorm2d."""

    pass


class UnlimitedOCRSamVisionAttention(SamVisionAttention):
    pass


class UnlimitedOCRSamMLPBlock(SamMLPBlock):
    pass


class UnlimitedOCRSamVisionLayer(SamVisionLayer):
    pass


class UnlimitedOCRSamVisionNeck(SamVisionNeck):
    """SAM neck with Detectron2/VitDet channels-first LayerNorm (bit-identical to hub).

    ``SamVisionNeck`` uses ``SamLayerNorm`` (``nn.LayerNorm`` + permute), which diverges from the
    original manual LayerNorm2d under bf16. Bare ``nn.LayerNorm`` cannot be applied on BCHW.
    """

    def __init__(self, config: UnlimitedOCRSamVisionConfig):
        nn.Module.__init__(self)
        self.config = config
        self.conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, bias=False)
        self.layer_norm1 = UnlimitedOCRSamLayerNorm(config.output_channels)
        self.conv2 = nn.Conv2d(config.output_channels, config.output_channels, kernel_size=3, padding=1, bias=False)
        self.layer_norm2 = UnlimitedOCRSamLayerNorm(config.output_channels)


class UnlimitedOCRSamPatchEmbeddings(DeepseekOcr2SamPatchEmbeddings):
    pass


class UnlimitedOCRSamVisionProj(DeepseekOcr2SamVisionProj):
    pass


class UnlimitedOCRSamVisionEncoder(DeepseekOcr2SamVisionEncoder):
    """SAM ViT-B + neck + downsample proj; always-SDPA for hub logit parity."""

    def __init__(self, config: UnlimitedOCRSamVisionConfig):
        config._attn_implementation = "sdpa"
        super().__init__(config)


class UnlimitedOCRTextModel(DeepseekOcr2TextModel):
    """Language backbone (DeepSeek-V2 MoE + MHA). Forward/masks from DeepseekV2Model."""

    # Text tower stays on eager MHA for bit-identical logits vs hub (vision keeps SDPA separately).
    _supports_sdpa = False
    _supports_flash_attn = False
    _supports_flex_attn = False
    _supports_attention_backend = False


# UnlimitedOCRConfig lives in configuration_unlimited_ocr.py


class UnlimitedOCRVisionModel(UnlimitedOCRPreTrainedModel):
    """Vision pipeline: SAM ViT-B + CLIP-L, concatenated along the feature dimension."""

    def __init__(self, config: UnlimitedOCRVisionConfig):
        super().__init__(config)
        self.sam_encoder = UnlimitedOCRSamVisionEncoder(config.sam_config)
        self.vision_encoder = UnlimitedOCRVisionEncoder(config.encoder_config)
        self.post_init()

    @auto_docstring
    def forward(self, pixel_values: torch.Tensor, **kwargs) -> BaseModelOutput:
        sam_outputs = self.sam_encoder(pixel_values, **kwargs)
        sam_features = sam_outputs.last_hidden_state
        # The SAM patch grid seeds the CLIP tower, so the two towers stay spatially aligned.
        clip_features = self.vision_encoder(pixel_values, sam_features)
        fused = torch.cat((clip_features[:, 1:], sam_features.flatten(2).permute(0, 2, 1)), dim=-1)
        return BaseModelOutput(last_hidden_state=fused)


class UnlimitedOCRModel(DeepseekOcr2Model):
    def __init__(self, config: UnlimitedOCRConfig):
        super().__init__(config)
        # DeepseekOcr2 deletes LlavaNext's `image_newline`; UnlimitedOCR needs it for row packing.
        self.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size))
        # Fused CLIP-L + SAM downsample channels (Deepseek projector is encoder-only).
        self.multi_modal_projector = nn.Linear(
            config.vision_config.encoder_config.hidden_size
            + config.vision_config.sam_config.downsample_channels[-1],
            config.text_config.hidden_size,
        )

    def pack_image_features(
        self, features: torch.Tensor, num_rows: int, num_columns: int, image_newline: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Lay `num_rows * num_columns` square view features out as a single token grid and append one
        `image_newline` token at the end of every row, then flatten to `(seq_len, hidden_size)`.
        """
        if image_newline is None:
            image_newline = self.image_newline
        _, num_queries, hidden_size = features.shape
        queries_per_side = int(num_queries**0.5)
        features = (
            features.view(num_rows, num_columns, queries_per_side, queries_per_side, hidden_size)
            .permute(0, 2, 1, 3, 4)
            .reshape(num_rows * queries_per_side, num_columns * queries_per_side, hidden_size)
        )
        newlines = image_newline.expand(features.shape[0], 1, hidden_size)
        return torch.cat([features, newlines], dim=1).flatten(0, 1)

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        pixel_values_local: torch.FloatTensor | None = None,
        num_local_patches: list[list[int]] | torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_values_local (`torch.FloatTensor` of shape `(total_patches, 3, tile_size, tile_size)`, *optional*):
            All local crops flattened across the batch, or `None` if no image was tiled.
        num_local_patches (`list[list[int]]` or `torch.Tensor` of shape `(batch_size, 2)`, *optional*):
            Local crop grid `[num_rows, num_columns]` per image (same kwarg name as DeepseekOcr2 so
            inherited `forward` can thread it through). `[1, 1]` means the image was not tiled.
        """
        image_grid_hw = num_local_patches
        if isinstance(image_grid_hw, torch.Tensor):
            image_grid_hw = image_grid_hw.tolist()
        elif image_grid_hw is None:
            image_grid_hw = [[1, 1]] * pixel_values.shape[0]

        global_outputs = self.vision_tower(pixel_values, **kwargs)
        global_features = self.multi_modal_projector(global_outputs.last_hidden_state)

        local_outputs = {}
        patch_counts = [rows * columns if rows * columns > 1 else 0 for rows, columns in image_grid_hw]
        if pixel_values_local is not None:
            local_vision_outputs = self.vision_tower(pixel_values_local, **kwargs)
            all_local_features = self.multi_modal_projector(local_vision_outputs.last_hidden_state)
            per_image_local = torch.split(all_local_features, patch_counts, dim=0)
            local_outputs = {"local_last_hidden_state": local_vision_outputs.last_hidden_state}
        else:
            per_image_local = [None] * len(image_grid_hw)

        all_features = []
        view_separator = self.view_separator.to(global_features.device).unsqueeze(0)
        for index, (num_rows, num_columns) in enumerate(image_grid_hw):
            features = [self.pack_image_features(global_features[index : index + 1], 1, 1), view_separator]
            if patch_counts[index]:
                features.insert(0, self.pack_image_features(per_image_local[index], num_rows, num_columns))
            all_features.append(torch.cat(features, dim=0))

        return UnlimitedOCRModelOutputWithPooling(
            last_hidden_state=global_outputs.last_hidden_state,
            pooler_output=all_features,
            **local_outputs,
        )


class UnlimitedOCRForConditionalGeneration(DeepseekOcr2ForConditionalGeneration):
    def pack_image_features(self, features, num_rows, num_columns, image_newline=None):
        return self.model.pack_image_features(features, num_rows, num_columns, image_newline=image_newline)


__all__ = [
    "UnlimitedOCRTextConfig",
    "UnlimitedOCRConfig",
    "UnlimitedOCRVisionConfig",
    "UnlimitedOCRVisionEncoderConfig",
    "UnlimitedOCRClipVisionVisionConfig",
    "UnlimitedOCRClipVisionTextConfig",
    "UnlimitedOCRSamVisionConfig",
    "UnlimitedOCRImageProcessor",
    "UnlimitedOCRImageProcessorPil",
    "UnlimitedOCRProcessor",
    "UnlimitedOCRPreTrainedModel",
    "UnlimitedOCRTextModel",
    "UnlimitedOCRSamVisionEncoder",
    "UnlimitedOCRVisionEncoder",
    "UnlimitedOCRVisionModel",
    "UnlimitedOCRModel",
    "UnlimitedOCRForConditionalGeneration",
]
