# coding=utf-8
# Copyright 2023-2026 DeepSeek-AI, Baidu, and The HuggingFace Inc. team.
# Consolidated modular source for transformers.models.unlimitedocr.
"""PyTorch UnlimitedOCR model (vision encoder + DeepSeek-V2 + UnlimitedOCR)."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from PIL import Image, ImageOps
from torchvision import transforms

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_utils import ImageInput
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TensorType, auto_docstring, logging
from ..clip.configuration_clip import CLIPVisionConfig
from ..clip.modeling_clip import (
    CLIPAttention,
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPMLP,
    CLIPVisionEmbeddings,
)
from ..deepseek_ocr2.configuration_deepseek_ocr2 import DeepseekOcr2Config, DeepseekOcr2VisionConfig
from ..deepseek_ocr2.modeling_deepseek_ocr2 import (
    DeepseekOcr2SamPatchEmbeddings,
    DeepseekOcr2SamVisionProj,
    DeepseekOcr2TextAttention,
    DeepseekOcr2TextDecoderLayer,
    DeepseekOcr2TextModel,
)
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
    SamVisionLayer,
    SamVisionNeck,
    SamVisionSdpaAttention,
)
from ..vitdet.modeling_vitdet import VitDetLayerNorm

logger = logging.get_logger(__name__)


# ============== configuration ==============

@auto_docstring
@strict
class DeepseekV2Config(Dots1Config):
    r"""
    Text-backbone config for UnlimitedOCR (DeepSeek-V2 MoE + MHA).

    `layer_types` defaults to all-`full_attention` when unset so hub checkpoints that set
    `sliding_window` but omit `layer_types` keep bit-identical logits with the historical load path.
    """

    model_type = "deepseek_v2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_experts": "n_routed_experts"}

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


class UnlimitedOCRTextConfig(DeepseekV2Config):
    """Language-backbone config (DeepSeek-V2 MoE + MHA)."""

    base_config_key = "text_config"
    attribute_map = {
        "num_experts": "n_routed_experts",
        "sliding_window_size": "sliding_window",
    }


@auto_docstring
@strict
class UnlimitedOCRVisionEncoderConfig(CLIPVisionConfig):
    """CLIP-L tower config (`encoder_config` slot, DeepseekOcr2-style)."""

    model_type = "unlimitedocr_vision_encoder"
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
    model_type = "unlimitedocr_vision"


@auto_docstring
@strict
class UnlimitedOCRConfig(DeepseekOcr2Config):
    r"""
    projector_input_dim (`int`, *optional*, defaults to 2048):
        Input size of the vision→text linear projector (CLIP hidden + SAM downsample channels).
    """

    model_type = "unlimited-ocr"
    attribute_map = {"language_config": "text_config"}
    projector_input_dim: int = 2048
    image_token_id = AttributeError()

    def __post_init__(self, **kwargs):
        # Hub ships an opaque `vision_config` blob (`model_type: "vision"`); use defaults instead.
        if isinstance(self.vision_config, dict) and self.vision_config.get("model_type") == "vision":
            self.vision_config = None
        # Hub uses `language_config`; pop before PreTrainedConfig setattr (attribute_map) overwrites
        # a materialized `text_config` with the raw dict.
        if (language_config := kwargs.pop("language_config", None)) is not None and self.text_config is None:
            self.text_config = language_config
        super().__post_init__(**kwargs)
        # Parent attn dispatch can overwrite subconfigs; force hub CLIP tower back to SDPA.
        self.vision_config.encoder_config._attn_implementation = "sdpa"


# ============== image_processing ==============

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=32, image_size=640, use_thumbnail=False):
    """Tile `image` into local crops (Unlimited-OCR / DeepSeek-OCR gundam path)."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images, target_aspect_ratio


class UnlimitedOCRImageProcessor(BaseImageProcessor):
    r"""
    Image processor for Unlimited-OCR.

    Builds the global padded view (`base_size`) and optional local crops (`image_size`) used by
    the DeepEncoder, matching the original Unlimited-OCR / DeepSeek-OCR preprocessing.
    """

    model_input_names = ["images_crop", "images_ori", "images_spatial_crop"]

    def __init__(
        self,
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        min_crops: int = 2,
        max_crops: int = 32,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        do_normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.min_crops = min_crops
        self.max_crops = max_crops
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)
        self.do_normalize = do_normalize

        transform_pipelines = [transforms.ToTensor()]
        if do_normalize:
            transform_pipelines.append(transforms.Normalize(mean=self.image_mean, std=self.image_std))
        self._transform = transforms.Compose(transform_pipelines)

    def _load_image(self, image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _as_image_list(self, images: ImageInput) -> list[Image.Image]:
        if isinstance(images, (list, tuple)):
            return [self._load_image(img) for img in images]
        return [self._load_image(images)]

    def _process_one(self, image: Image.Image, crop_mode: bool):
        images_crop_list = []
        if crop_mode:
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = [1, 1]
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(
                    image,
                    min_num=self.min_crops,
                    max_num=self.max_crops,
                    image_size=self.image_size,
                )
            global_view = ImageOps.pad(
                image,
                (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.image_mean),
            )
            images_ori = self._transform(global_view).to(torch.bfloat16)
            width_crop_num, height_crop_num = crop_ratio
            if width_crop_num > 1 or height_crop_num > 1:
                for crop in images_crop_raw:
                    images_crop_list.append(self._transform(crop).to(torch.bfloat16))
        else:
            raise NotImplementedError("UnlimitedOCRImageProcessor currently supports crop_mode=True only")

        return images_ori, images_crop_list, [width_crop_num, height_crop_num]

    def preprocess(
        self,
        images: ImageInput,
        crop_mode: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        if crop_mode is None:
            crop_mode = self.crop_mode

        images = self._as_image_list(images)

        images_ori_list = []
        images_crop_list = []
        images_spatial_crop = []
        for image in images:
            images_ori, crops, spatial = self._process_one(image, crop_mode=crop_mode)
            images_ori_list.append(images_ori)
            images_crop_list.extend(crops)
            images_spatial_crop.append(spatial)

        images_ori = torch.stack(images_ori_list, dim=0)
        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            # Match original float32 placeholder when there are no local crops
            images_crop = torch.zeros((1, 3, self.base_size, self.base_size))
        images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)

        data = {
            "images_ori": images_ori,
            "images_crop": images_crop,
            "images_spatial_crop": images_spatial_crop,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["UnlimitedOCRImageProcessor", "dynamic_preprocess", "find_closest_aspect_ratio"]

# ============== processing ==============

def _text_encode(tokenizer, text: str, bos: bool = True, eos: bool = False):
    """Match Unlimited-OCR `text_encode` (hardcoded bos/eos ids 0/1)."""
    t = tokenizer.encode(text, add_special_tokens=False)
    if bos:
        t = [0] + t
    if eos:
        t = t + [1]
    return t


class UnlimitedOCRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {},
    }


@auto_docstring
class UnlimitedOCRProcessor(ProcessorMixin):
    r"""
    Constructs an Unlimited-OCR processor which wraps an image processor and a tokenizer.

    Builds `input_ids` / `images_seq_mask` by splitting on `<image>` and inserting the exact
    image-token grid used by the original Unlimited-OCR `infer` path (not by string-expanding
    `<image>` then calling `tokenizer` on the full prompt).
    """

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        patch_size: int = 16,
        downsample_ratio: int = 4,
        image_token: str = "<image>",
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*, defaults to `16`):
            Patch size of the SAM ViT vision encoder.
        downsample_ratio (`int`, *optional*, defaults to `4`):
            Downsample ratio after the vision encoder (token grid side is
            `ceil((size // patch_size) / downsample_ratio)`).
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Placeholder in the prompt string; expanded to vision placeholder ids in `input_ids`.
        """
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.image_token = (
            tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        )
        if tokenizer is not None:
            token_id = tokenizer.convert_tokens_to_ids(self.image_token)
            # convert_tokens_to_ids may return unk; fall back to the checkpoint id
            if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
                token_id = 128815
            self.image_token_id = int(token_id)
        else:
            self.image_token_id = 128815
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def _num_queries(self, size: int) -> int:
        return math.ceil((size // self.patch_size) / self.downsample_ratio)

    def _image_token_ids(self, width_crop_num: int, height_crop_num: int) -> list[int]:
        base_size = self.image_processor.base_size
        image_size = self.image_processor.image_size
        num_queries = self._num_queries(image_size)
        num_queries_base = self._num_queries(base_size)

        tokenized_image = (
            [self.image_token_id] * num_queries_base + [self.image_token_id]
        ) * num_queries_base
        tokenized_image += [self.image_token_id]
        if width_crop_num > 1 or height_crop_num > 1:
            tokenized_image += (
                [self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]
            ) * (num_queries * height_crop_num)
        return tokenized_image

    def _build_multimodal_ids(
        self,
        text: str,
        images_spatial_crop: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text = text.strip()
        text_splits = text.split(self.image_token)
        n_images = images_spatial_crop.shape[0]
        if len(text_splits) - 1 != n_images:
            raise ValueError(
                f"Prompt has {len(text_splits) - 1} `{self.image_token}` placeholders "
                f"but got {n_images} image(s)."
            )

        tokenized_str: list[int] = []
        images_seq_mask: list[bool] = []

        for text_sep, spatial in zip(text_splits[:-1], images_spatial_crop.tolist()):
            tokenized_sep = _text_encode(self.tokenizer, text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            width_crop_num, height_crop_num = int(spatial[0]), int(spatial[1])
            tokenized_image = self._image_token_ids(width_crop_num, height_crop_num)
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)

        tokenized_sep = _text_encode(self.tokenizer, text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        # Match original: prepend hardcoded bos_id=0 (not tokenizer.bos_token_id)
        tokenized_str = [0] + tokenized_str
        images_seq_mask = [False] + images_seq_mask

        return (
            torch.tensor(tokenized_str, dtype=torch.long),
            torch.tensor(images_seq_mask, dtype=torch.bool),
        )

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        **kwargs: Unpack[UnlimitedOCRProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`] with:

            - **input_ids** -- Token ids including expanded image placeholders.
            - **images_seq_mask** -- Bool mask True where image placeholders sit.
            - **images_crop** / **images_ori** -- Local crops and global padded view.
            - **images** -- `[(images_crop, images_ori)]` for `model.generate`.
            - **images_spatial_crop** -- Crop grid sizes `[n_images, 2]`.
        """
        if images is None:
            raise ValueError("`images` is required for `UnlimitedOCRProcessor`.")
        if text is None:
            raise ValueError(
                "`text` is required for `UnlimitedOCRProcessor`. Example: `'<image>document parsing.'`"
            )

        output_kwargs = self._merge_kwargs(
            UnlimitedOCRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        images_kwargs = output_kwargs["images_kwargs"]
        # return_tensors is applied after we assemble nested `images`
        images_kwargs.pop("return_tensors", None)

        if isinstance(text, str):
            texts = [text]
        elif isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text):
            texts = list(text)
        else:
            raise TypeError("Invalid input text. Provide a string or a list of strings.")

        if len(texts) != 1:
            raise NotImplementedError(
                "UnlimitedOCRProcessor currently supports batch size 1 (one prompt string)."
            )

        image_inputs = self.image_processor(images, return_tensors=None, **images_kwargs)
        images_ori = image_inputs["images_ori"]
        images_crop = image_inputs["images_crop"]
        images_spatial_crop = image_inputs["images_spatial_crop"]

        input_ids, images_seq_mask = self._build_multimodal_ids(texts[0], images_spatial_crop)

        data = {
            "input_ids": input_ids.unsqueeze(0),
            "images_seq_mask": images_seq_mask.unsqueeze(0),
            "images_spatial_crop": images_spatial_crop,
            "images_crop": images_crop,
            "images_ori": images_ori,
        }
        # Nested `(crop, ori)` is what `model.generate` expects; attach after tensor conversion
        # because BatchFeature cannot convert list-of-tuples to a single tensor.
        batch = BatchFeature(data=data, tensor_type=return_tensors)
        batch["images"] = [(batch["images_crop"], batch["images_ori"])]
        return batch


# Hub / processor_config.json historically used this name
UnlimitedOCRHFProcessor = UnlimitedOCRProcessor

__all__ = ["UnlimitedOCRProcessor", "UnlimitedOCRHFProcessor"]

# ============== modeling ==============


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
    """CLIP-L tower attention (SDPA via config._attn_implementation)."""

    pass


class UnlimitedOCRClipVisionMLP(CLIPMLP):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # QuickGELU in fp32 then cast back: matches hub TorchScript quick_gelu under bf16.
        hidden_states = self.fc1(hidden_states)
        dtype = hidden_states.dtype
        hidden_states = self.activation_fn(hidden_states.float()).to(dtype)
        return self.fc2(hidden_states)


class UnlimitedOCRClipVisionEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config: UnlimitedOCRVisionEncoderConfig):
        super().__init__(config)
        self.self_attn = UnlimitedOCRClipVisionAttention(config)
        self.mlp = UnlimitedOCRClipVisionMLP(config)


class UnlimitedOCRClipVisionEncoder(CLIPEncoder):
    """Weight path stays `transformer.layers.*` (hub layout)."""

    def __init__(self, config: UnlimitedOCRVisionEncoderConfig):
        nn.Module.__init__(self)
        self.config = config
        self.layers = nn.ModuleList(
            [UnlimitedOCRClipVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=None)
        return hidden_states


class UnlimitedOCRVisionEncoder(nn.Module):
    def __init__(self, cfg: UnlimitedOCRVisionEncoderConfig) -> None:
        super().__init__()
        self.embeddings = UnlimitedOCRClipVisionEmbeddings(cfg)
        self.transformer = UnlimitedOCRClipVisionEncoder(cfg)
        self.pre_layrnorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.pre_layernorm_epsilon)
        for p in self.parameters():
            p.micro_dp = True

    def forward(self, x, patch_embeds):
        x = self.embeddings(x, patch_embeds)
        hidden_states = self.pre_layrnorm(x)
        return self.transformer(hidden_states)


# ============== modeling_deepseekv2.py ==============

# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch DeepSeek model and compatible with both DeepSeekV2 and DeepSeekV3"""
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


class UnlimitedOCRPreTrainedModel(PreTrainedModel):
    config_class = UnlimitedOCRConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["UnlimitedOCRTextDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_attention_backend = False
    _supports_cache_class = True


#=========================Sam-Vary=================================

class UnlimitedOCRSamLayerNorm(VitDetLayerNorm):
    """Channels-first LayerNorm (Detectron2 / VitDet); bit-identical to hub LayerNorm2d."""

    pass


class UnlimitedOCRSamVisionSdpaAttention(SamVisionSdpaAttention):
    """Always-SDPA SAM attention (independent of LM `_attn_implementation`)."""

    pass


class UnlimitedOCRSamVisionLayer(SamVisionLayer):
    def __init__(self, config, window_size):
        super().__init__(config, window_size)
        self.attn = UnlimitedOCRSamVisionSdpaAttention(config, window_size)


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


class UnlimitedOCRSamVisionEncoder(UnlimitedOCRPreTrainedModel):
    """SAM ViT-B + neck + downsample proj (DeepseekOcr2SamVisionEncoder-style).

    Hand-built: subclassing ``SamVisionEncoder`` makes the converter emit broken ``Unlimitedocr*``
    renames (SamPreTrainedModel → UnlimitedocrConfig).
    """

    def __init__(self, config: UnlimitedOCRSamVisionConfig):
        super().__init__(config)
        self.config = config
        self.image_size = config.image_size
        self.patch_embed = UnlimitedOCRSamPatchEmbeddings(config)

        self.pos_embed = None
        if config.use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                )
            )

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = UnlimitedOCRSamVisionLayer(
                config,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
            )
            self.layers.append(layer)

        self.neck = UnlimitedOCRSamVisionNeck(config)
        self.proj = UnlimitedOCRSamVisionProj(config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.patch_embed

    def interpolate_pos_encoding(self, height: int, width: int) -> torch.Tensor:
        if not torch.jit.is_tracing() and self.pos_embed.shape[1] == height and self.pos_embed.shape[2] == width:
            return self.pos_embed
        target_dtype = self.pos_embed.dtype
        pos_embed = self.pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed.to(torch.float32),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).to(dtype=target_dtype)
        return pos_embed.permute(0, 2, 3, 1)

    def forward(self, pixel_values: torch.FloatTensor, **kwargs) -> torch.Tensor:
        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.interpolate_pos_encoding(
                hidden_states.shape[1], hidden_states.shape[2]
            )
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        hidden_states = self.neck(hidden_states)
        return self.proj(hidden_states)


class SlidingWindowNoRepeatNgramProcessor:
    """Block n-gram repetitions within a sliding window.
    Aligned with SGLang DeepseekOCRNoRepeatNGramLogitProcessor."""
    def __init__(self, ngram_size, window, whitelist_token_ids=None):
        self.ngram_size = ngram_size
        self.window = window
        self.whitelist = set(whitelist_token_ids) if whitelist_token_ids else set()

    def __call__(self, input_ids, scores):
        for batch_idx in range(input_ids.shape[0]):
            sequence = input_ids[batch_idx].tolist()
            if len(sequence) < self.ngram_size:
                continue
            search_start = max(0, len(sequence) - self.window)
            search_end = len(sequence) - self.ngram_size + 1
            if search_end <= search_start:
                continue
            if self.ngram_size > 1:
                current_prefix = tuple(sequence[-(self.ngram_size - 1):])
            else:
                current_prefix = tuple()
            banned = set()
            for idx in range(search_start, search_end):
                ngram = sequence[idx:idx + self.ngram_size]
                if self.ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
                    banned.add(ngram[-1])
            banned.difference_update(self.whitelist)
            for token_id in banned:
                scores[batch_idx, token_id] = float('-inf')
        return scores


class UnlimitedOCRTextModel(DeepseekOcr2TextModel):
    """Language backbone (DeepSeek-V2 MoE + MHA). Forward/masks from DeepseekV2Model."""

    # Text tower stays on eager MHA for bit-identical logits vs hub (vision keeps SDPA separately).
    _supports_sdpa = False
    _supports_flash_attn = False
    _supports_flex_attn = False
    _supports_attention_backend = False


# UnlimitedOCRConfig lives in configuration_unlimitedocr.py


class UnlimitedOCRVisionModel(UnlimitedOCRPreTrainedModel):
    """Vision pipeline: SAM ViT-B + CLIP-L (DeepseekOcr2VisionModel-style)."""

    def __init__(self, config: UnlimitedOCRVisionConfig):
        super().__init__(config)
        self.sam_encoder = UnlimitedOCRSamVisionEncoder(config.sam_config)
        self.vision_encoder = UnlimitedOCRVisionEncoder(config.encoder_config)
        self.post_init()


class UnlimitedOCRModel(UnlimitedOCRPreTrainedModel):
    """Multimodal wrapper: vision tower + projector + language model (Step3p7-style)."""

    config_class = UnlimitedOCRConfig

    def __init__(self, config: UnlimitedOCRConfig):
        super().__init__(config)
        self.vision_model = UnlimitedOCRVisionModel(config.vision_config)
        self.language_model = UnlimitedOCRTextModel(config.text_config)
        n_embed = config.text_config.hidden_size
        self.multi_modal_projector = nn.Linear(config.projector_input_dim, n_embed)
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def _encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """SAM + CLIP concat → projector (shared by local crops and global view)."""
        sam_feats = self.vision_model.sam_encoder(pixel_values)
        clip_feats = self.vision_model.vision_encoder(pixel_values, sam_feats)
        fused = torch.cat((clip_feats[:, 1:], sam_feats.flatten(2).permute(0, 2, 1)), dim=-1)
        return self.multi_modal_projector(fused)

    def _add_image_newlines(self, features: torch.Tensor) -> torch.Tensor:
        _, hw, n_dim = features.shape
        h = w = int(hw**0.5)
        features = features.view(h, w, n_dim)
        features = torch.cat([features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1)
        return features.view(-1, n_dim)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.FloatTensor] = None,
        images_spatial_crop: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if (
            images is not None
            and (input_ids.shape[1] != 1 or self.training)
            and torch.sum(images[0][1]).item() != 0
        ):
            idx = 0
            for image, crop_shape in zip(images, images_spatial_crop):
                images_in_this_batch = []
                patches = image[0]
                image_ori = image[1]

                with torch.no_grad():
                    if torch.sum(patches).item() != 0:
                        local_features = self._encode_vision(patches)
                        global_features = self._add_image_newlines(self._encode_vision(image_ori))

                        _, hw2, n_dim2 = local_features.shape
                        h2 = w2 = int(hw2**0.5)
                        width_crop_num, height_crop_num = crop_shape[0], crop_shape[1]
                        local_features = (
                            local_features.view(height_crop_num, width_crop_num, h2, w2, n_dim2)
                            .permute(0, 2, 1, 3, 4)
                            .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
                        )
                        local_features = torch.cat(
                            [
                                local_features,
                                self.image_newline[None, None, :].expand(height_crop_num * h2, 1, n_dim2),
                            ],
                            dim=1,
                        ).view(-1, n_dim2)

                        images_in_this_batch.append(
                            torch.cat([local_features, global_features, self.view_seperator[None, :]], dim=0)
                        )
                    else:
                        for img_idx in range(image_ori.shape[0]):
                            global_features = self._add_image_newlines(
                                self._encode_vision(image_ori[img_idx : img_idx + 1])
                            )
                            images_in_this_batch.append(
                                torch.cat([global_features, self.view_seperator[None, :]], dim=0)
                            )

                if images_in_this_batch:
                    images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                    inputs_embeds[idx].masked_scatter_(
                        images_seq_mask[idx].unsqueeze(-1).cuda(), images_in_this_batch
                    )
                idx += 1

        return self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class UnlimitedOCRForConditionalGeneration(UnlimitedOCRPreTrainedModel, GenerationMixin):
    config_class = UnlimitedOCRConfig
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = UnlimitedOCRModel(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.FloatTensor] = None,
        images_spatial_crop: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).float()
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.get_seq_length()
                max_cache_length = getattr(past_key_values, "get_max_length", lambda: None)()
                # Newer DynamicCache uses -1 for "unbounded"; slicing with -(-1) drops the first token.
                if max_cache_length is not None and max_cache_length < 0:
                    max_cache_length = None
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        elif position_ids is not None and position_ids.shape[-1] != input_ids.shape[-1]:
            position_ids = position_ids[:, -input_ids.shape[1] :]

        cache_position = torch.arange(past_length, past_length + position_ids.shape[-1], device=position_ids.device)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        _is_prefill = past_key_values is None or (
            isinstance(past_key_values, Cache) and past_key_values.get_seq_length() == 0
        )
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None) if _is_prefill else None,
                "images_seq_mask": kwargs.get("images_seq_mask", None) if _is_prefill else None,
                "images_spatial_crop": kwargs.get("images_spatial_crop", None) if _is_prefill else None,
            }
        )
        return model_inputs


# Checkpoint / AutoModel compat (architectures still list ForCausalLM on some hubs)
UnlimitedOCRForCausalLM = UnlimitedOCRForConditionalGeneration


__all__ = [
    "DeepseekV2Config",
    "UnlimitedOCRTextConfig",
    "UnlimitedOCRConfig",
    "UnlimitedOCRVisionConfig",
    "UnlimitedOCRVisionEncoderConfig",
    "UnlimitedOCRClipVisionVisionConfig",
    "UnlimitedOCRClipVisionTextConfig",
    "UnlimitedOCRSamVisionConfig",
    "UnlimitedOCRImageProcessor",
    "UnlimitedOCRProcessor",
    "UnlimitedOCRHFProcessor",
    "UnlimitedOCRPreTrainedModel",
    "UnlimitedOCRTextModel",
    "UnlimitedOCRSamVisionEncoder",
    "UnlimitedOCRVisionEncoder",
    "UnlimitedOCRVisionModel",
    "UnlimitedOCRModel",
    "UnlimitedOCRForConditionalGeneration",
    "UnlimitedOCRForCausalLM",
]
