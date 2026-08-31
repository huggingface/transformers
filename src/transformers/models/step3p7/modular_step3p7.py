# Copyright 2026 The StepFun and HuggingFace Inc. team. All rights reserved.
#
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
from collections.abc import Callable

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import divide_to_patches, group_images_by_shape, reorder_images
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, ImageInput, PILImageResampling, SizeDict
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ImagesKwargs, ProcessorMixin, Unpack
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
    no_inherit_decorator,
    torch_int,
)
from ...utils.generic import maybe_autocast
from ...utils.output_capturing import capture_outputs
from ...vision_utils import get_vision_position_ids
from ..deepseek_ocr2.modeling_deepseek_ocr2 import DeepseekOcr2ForConditionalGeneration, DeepseekOcr2Model
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts, DeepseekV4MLP
from ..gemma3.modeling_gemma3 import Gemma3TextModel
from ..gemma4.modeling_gemma4 import Gemma4VisionRotaryEmbedding
from ..laguna.modeling_laguna import (
    LagunaAttention,
    LagunaDecoderLayer,
    LagunaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
from ..minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3VLRMSNorm,
    MiniMaxM3VLSparseMoeBlock,
    MiniMaxM3VLTopKRouter,
    MiniMaxM3VLVisionAttention,
    MiniMaxM3VLVisionMLP,
)

# Unused here, but load-bearing: `Step3p7VisionAttention` inherits `MiniMaxM3VLVisionAttention`, so
# without this import the converter would pull MiniMax's 3-axis `apply_rotary_pos_emb_vision` into the
# generated file. This tower is 2-D (t=1, spatial_merge_size=1), so it needs Qwen2-VL's 2-axis one.
from ..qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision  # noqa: F401
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..siglip.modeling_siglip import SiglipVisionEmbeddings


logger = logging.get_logger(__name__)

__all__ = [
    "Step3p7ForConditionalGeneration",
    "Step3p7Model",
    "Step3p7PreTrainedModel",
    "Step3p7TextModel",
    "Step3p7VisionModel",
    "Step3p7VisionConfig",
    "Step3p7TextConfig",
    "Step3p7Config",
    "Step3p7ImageProcessor",
    "Step3p7Processor",
]


@auto_docstring(checkpoint="stepfun-ai/Step-3.7-Flash")
@strict
class Step3p7VisionConfig(SiglipVisionConfig):
    model_type = "step3p5_vision"
    base_config_key = "vision_config"

    # SiGLIP field overrides
    hidden_size: int = 1536
    num_hidden_layers: int = 47
    num_attention_heads: int = 16
    image_size: int = 728
    patch_size: int = 14
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    # New fields
    mlp_ratio: float = 8960 / 1536
    layer_scale_init_value: float = 0.1
    # RoPE config (compatible with Gemma4VisionRotaryEmbedding)
    rope_parameters: dict | None = None
    max_position_embeddings: int = 2704  # (image_size // patch_size)^2 = (728//14)^2

    def __post_init__(self, **kwargs):
        self.hidden_size = kwargs.pop("width", self.hidden_size)
        self.num_hidden_layers = kwargs.pop("layers", self.num_hidden_layers)
        self.num_attention_heads = kwargs.pop("heads", self.num_attention_heads)
        self.layer_scale_init_value = kwargs.pop("ls_init_value", self.layer_scale_init_value)
        PreTrainedConfig.__post_init__(self, **kwargs)
        self.intermediate_size = int(self.hidden_size * self.mlp_ratio)


@auto_docstring(checkpoint="stepfun-ai/Step-3.7-Flash")
@strict
class Step3p7TextConfig(MiniMaxM3VLTextConfig):
    r"""
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer MLP type: `"sparse"` (MoE) or `"dense"`. If not provided, derived from the legacy
        `moe_layers_enum` hub-config kwarg (comma-separated string or list of MoE layer indices),
        defaulting to all layers from index 3 onward being MoE.
    n_routed_experts (`int`, *optional*, defaults to 288):
        Total number of routed experts. Accessible as `num_local_experts` via `attribute_map`.
    share_expert_dim (`int`, *optional*, defaults to 1280):
        Intermediate size of the always-active shared expert.
    num_sliding_attention_heads (`int`, *optional*):
        Attention head count for `"sliding_attention"` layers, if different from `num_attention_heads`.
        Defaults to the legacy `attention_other_setting` hub-config kwarg's `num_attention_heads` entry.
        Applied via a `per_layer_config` override (see `PreTrainedConfig`), not a per-layer list field.
    query_pre_attn_scalar (`int` or `float`, *optional*):
        `Step3p7Attention.__init__` hook point: defaults to `head_dim`, giving standard
        `head_dim ** -0.5` scaling; overridable per released checkpoint variant.
    moe_router_scaling_factor (`float`, *optional*, defaults to 1.0):
        Scaling factor applied to the MoE block's routed-expert output (`routed_scaling_factor` in
        `Step3p7SparseMoeBlock`).
    swiglu_limits (`list[float | None]`, *optional*):
        Per-layer gate/up clamping bound; `None` means no clamping.
    swiglu_limits_shared (`list[float | int | None]`, *optional*):
        Per-layer gate/up clamping bound for the always-active shared expert; `None` means no clamping.
    mtp_layer_types (`list[str]`, *optional*):
        Per-MTP-layer attention type; split off `layer_types`'s legacy trailing pad instead of being
        discarded, so `Step3p7Config.get_mtp_config()` can build the MTP layers for `generate(use_mtp=True)`.
    mtp_mlp_layer_types (`list[str]`, *optional*):
        Per-MTP-layer MLP type, analogous to `mtp_layer_types` for `mlp_layer_types`.
    """

    model_type = "step3p5"
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "num_attention_groups": "num_key_value_heads",
        "moe_num_experts": "n_routed_experts",
        "moe_top_k": "num_experts_per_tok",
        "share_expert_dims": "share_expert_dim",
        "num_mtp_layers": "num_nextn_predict_layers",
    }
    default_theta = 10000.0
    gating = True
    use_bidirectional_attention = False
    # Same as `MiniMaxM3VLTextConfig.base_model_tp_plan` plus `g_proj` (sharded like q/k/v, since it
    # gates their gathered output). Spelled out in full, not `{**MiniMaxM3VLTextConfig.base_model_tp_plan, ...}`:
    # generated files have no cross-model imports, so that name wouldn't resolve there.
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_gather_output",
        "layers.*.self_attn.k_proj": "colwise_gather_output",
        "layers.*.self_attn.v_proj": "colwise_gather_output",
        "layers.*.self_attn.g_proj": "colwise_gather_output",
        "layers.*.self_attn.o_proj": "rowwise_split_input",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    # Unused fields inherited from `MiniMaxM3VLTextConfig`; removed rather than documented.
    dense_intermediate_size = AttributeError()
    shared_intermediate_size = AttributeError()
    rotary_dim = AttributeError()
    swiglu_alpha = AttributeError()
    swiglu_limit = AttributeError()
    index_n_heads = AttributeError()
    index_head_dim = AttributeError()
    index_block_size = AttributeError()
    index_topk_blocks = AttributeError()
    index_local_blocks = AttributeError()
    output_router_logits = AttributeError()
    routed_scaling_factor = AttributeError()
    router_aux_loss_coef = AttributeError()
    router_jitter_noise = AttributeError()

    hidden_size: int = 4096
    intermediate_size: int = 11264
    num_key_value_heads: int = 8
    num_hidden_layers: int = 45
    max_position_embeddings: int = 128000
    vocab_size: int = 128815
    rms_norm_eps: float = 1e-5
    moe_intermediate_size: int = 1280
    n_routed_experts: int = 288
    num_experts_per_tok: int = 8
    share_expert_dim: int = 1280
    sliding_window: int | None = None
    num_sliding_attention_heads: int | None = None
    pad_token_id: int = 1
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    attention_bias: bool = False
    query_pre_attn_scalar: int | float | None = None
    moe_router_scaling_factor: float = 1.0
    mlp_bias: bool = False
    swiglu_limits: list[float | int | None] | None = None
    swiglu_limits_shared: list[float | int | None] | None = None
    mtp_layer_types: list[str] | None = None
    mtp_mlp_layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        # Legacy hub configs pad these per-layer lists with `num_nextn_predict_layers` trailing MTP
        # entries. Split each into `mtp_*` (used by `Step3p7Config.get_mtp_config()` for
        # `generate(use_mtp=True)`) instead of discarding it. Trimming itself is required either way:
        # `validate_layer_type`'s `@strict` check rejects `num_hidden_layers != len(layer_types)`.
        num_nextn_predict_layers = kwargs.pop("num_nextn_predict_layers", 0)
        self.num_nextn_predict_layers = num_nextn_predict_layers
        n, padded = self.num_hidden_layers, self.num_hidden_layers + num_nextn_predict_layers
        if num_nextn_predict_layers:
            for field, mtp_field in (
                ("layer_types", "mtp_layer_types"),
                ("mlp_layer_types", "mtp_mlp_layer_types"),
            ):
                value = getattr(self, field)
                if isinstance(value, list) and len(value) == padded:
                    setattr(self, field, value[:n])
                    setattr(self, mtp_field, value[n:padded])
            for field in ("swiglu_limits", "swiglu_limits_shared"):
                value = getattr(self, field)
                if isinstance(value, list) and len(value) == padded:
                    setattr(self, field, value[:n])
            for key in ("rope_theta", "partial_rotary_factors"):
                value = kwargs.get(key)
                if isinstance(value, list) and len(value) == padded:
                    kwargs[key] = value[:n]

        if self.layer_types is None:
            self.layer_types = ["full_attention"] * n
            if num_nextn_predict_layers:
                self.mtp_layer_types = ["full_attention"] * num_nextn_predict_layers

        if self.mlp_layer_types is None:
            # `moe_layers_enum` is the legacy hub-config alias for `mlp_layer_types`, read here only
            # Derived over the padded range so the trailing MTP layers get an `mtp_mlp_layer_types` entry.
            moe_layers_enum = kwargs.pop("moe_layers_enum", None)
            if moe_layers_enum is not None:
                items = moe_layers_enum.split(",") if isinstance(moe_layers_enum, str) else moe_layers_enum
                moe_set = {int(i) for i in items if str(i).strip()}
            else:
                moe_set = set(range(3, n))
            mlp_layer_types = ["sparse" if i in moe_set else "dense" for i in range(padded)]
            self.mlp_layer_types = mlp_layer_types[:n]
            if num_nextn_predict_layers:
                self.mtp_mlp_layer_types = mlp_layer_types[n:padded]

        if self.num_sliding_attention_heads is None:
            # `attention_other_setting` is a legacy hub-config dict overriding num_attention_heads/
            # num_key_value_heads/head_dim for "sliding_attention" layers. Keep `num_attention_heads`
            attention_other_setting = kwargs.pop("attention_other_setting", None)
            if attention_other_setting:
                self.num_sliding_attention_heads = attention_other_setting.get(
                    "num_attention_heads", self.num_attention_heads
                )
            else:
                self.num_sliding_attention_heads = self.num_attention_heads

        # On reload: `per_layer_config` is already in kwargs from the saved config.
        kwargs.setdefault(
            "per_layer_config",
            {
                layer_idx: {"num_attention_heads": self.num_sliding_attention_heads}
                for layer_idx, layer_type in enumerate(self.layer_types)
                if layer_type == "sliding_attention"
            },
        )

        if self.query_pre_attn_scalar is None:
            self.query_pre_attn_scalar = self.head_dim

        # `rope_theta`/`partial_rotary_factors` are per-layer (or `rope_theta` a single scalar shared
        # by all layers). Pop before `super().__post_init__()`: its RoPE handling only supports one
        # global scalar and would corrupt `self.rope_parameters` with a list.
        rope_theta = kwargs.pop("rope_theta", self.default_theta)
        partial_rotary_factors = kwargs.pop("partial_rotary_factors", None)
        rope_scaling = kwargs.pop("rope_scaling", None)

        super().__post_init__(**kwargs)

        # `DSV4Config` pattern: if `rope_parameters` is already resolved per type (reload), keep
        # only those sub-dicts. Otherwise build fresh, taking each type's value from any one of its layers (they all agree).
        layer_type_set = set(self.layer_types)
        rp = self.rope_parameters or {}
        if all(isinstance(rp.get(layer_type), dict) for layer_type in layer_type_set):
            self.rope_parameters = {layer_type: rp[layer_type] for layer_type in layer_type_set}
        else:
            if not isinstance(rope_theta, list):
                rope_theta = [rope_theta] * len(self.layer_types)
            self.rope_parameters = {}
            for layer_type in layer_type_set:
                i = self.layer_types.index(layer_type)
                params = {"rope_type": "default", "rope_theta": rope_theta[i]}
                if partial_rotary_factors:
                    params["partial_rotary_factor"] = partial_rotary_factors[i]
                self.rope_parameters[layer_type] = params
            if rope_scaling and "full_attention" in self.rope_parameters:
                self.rope_parameters["full_attention"].update(rope_scaling)


@auto_docstring(checkpoint="stepfun-ai/Step-3.7-Flash")
@strict
class Step3p7Config(PreTrainedConfig):
    model_type = "step3p7"
    sub_configs = {"vision_config": Step3p7VisionConfig, "text_config": Step3p7TextConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    projector_bias: bool = False
    image_token_id: int = 151679

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = Step3p7VisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = Step3p7VisionConfig(
                **{k: v for k, v in self.vision_config.items() if k != "model_type"}
            )

        if self.text_config is None:
            self.text_config = Step3p7TextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = Step3p7TextConfig(**{k: v for k, v in self.text_config.items() if k != "model_type"})

        super().__post_init__(**kwargs)


class Step3p7ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to 504):
        Target size (height = width) for each local patch crop.
    max_image_size (`int`, *optional*, defaults to 3024):
        Images larger than this (on their longest side) are scaled down uniformly before patch
        planning.
    """

    patch_size: int
    max_image_size: int


@auto_docstring
class Step3p7ImageProcessor(TorchvisionBackend):
    """
    Image processor for Step-3.7-Flash.

    Each input image is split into a global down-scaled view plus zero or more
    local patch crops via a sliding-window strategy, then every sub-image is
    resized and normalised independently.
    """

    resample = PILImageResampling.BILINEAR
    size = {"height": 728, "width": 728}
    patch_size: int = 504
    do_rescale = True
    do_normalize = True
    image_mean: list[float] = OPENAI_CLIP_MEAN
    image_std: list[float] = OPENAI_CLIP_STD
    do_convert_rgb = True
    valid_kwargs = Step3p7ImageProcessorKwargs
    model_input_names = ["pixel_values", "pixel_values_local", "num_local_patches"]

    max_image_size: int = 3024
    # ViT patch size (`Step3p7VisionEmbeddings.patch_size`) and the vision tower's total downsampling
    # stride (two stride-2 convolutions, `downsampler1`/`downsampler2`); `Step3p7Processor.__init__`
    # derives `num_image_features`/`num_patch_features` from these plus `size`/`patch_size`.
    vision_patch_size: int = 14
    downsampler_stride: int = 4

    @staticmethod
    def _is_extreme_aspect(width: int, height: int) -> bool:
        """`True` for near-degenerate images (min side < 32px, aspect ratio > 4:1)."""
        return min(width, height) < 32 and max(width / height, height / width) > 4

    def _plan_patches(
        self, width: int, height: int, image_size: int, patch_size: int
    ) -> tuple[tuple[int, int], tuple[int, int], int, int, int, bool]:
        """Compute the sliding-window patch layout for one image.

        Unlike models with a fixed tile size (Idefics3, LLaVA-NeXT), Step3p7
        adapts the window size to the image's aspect ratio, so this cannot be
        reduced to a simple ``ceil(h / tile) × ceil(w / tile)`` formula.

        Step 1 = normalise extreme inputs:
          - extreme-aspect images (min_side < 32, ratio > 4) are squared
          - images larger than ``max_image_size`` are scaled down uniformly

        Step 2 — choose window size from the normalised aspect ratio:
          - fits in global view (long_side ≤ image_size): tile only if elongated
            (long_side / short_side > 1.5), using short_side as the window
          - very elongated (ratio > 4): ``min(short_side, patch_size)``
          - standard case: ``patch_size``

        Step 3 — snap each dimension to the nearest window multiple
          (snap up when the remainder exceeds 20 % of the window size).

        Returns:
            global_width_height: ``(width, height)`` to resize the global view to before squaring
            crop_width_height: ``(crop_width, crop_height)`` snapped dimensions for patch extraction
            window_size: tile side length (``0`` → no local patches)
            num_patches_x: number of patches along the width
            num_patches_y: number of patches along the height
            needs_square_pad: whether the raw image must be zero-padded to a square before resizing
        """
        # Step 1 — normalise
        needs_square_pad = self._is_extreme_aspect(width, height)
        if needs_square_pad:
            width = height = max(width, height)
        if max(height, width) > self.max_image_size:
            scale = self.max_image_size / max(height, width)
            width, height = int(width * scale), int(height * scale)

        short_side, long_side = min(height, width), max(height, width)

        # Step 2 — choose window size
        if long_side <= image_size:
            window_size = short_side if long_side / short_side > 1.5 else 0
        elif long_side / short_side > 4:
            window_size = min(short_side, patch_size)
        else:
            window_size = patch_size

        if window_size == 0:
            return (width, height), (width, height), 0, 0, 0, needs_square_pad

        # Step 3 — snap each dimension to the nearest window-size multiple
        crop_width, crop_height = (
            (
                window_size * (dim // window_size + (dim % window_size > 0.2 * window_size))
                if dim >= window_size
                else dim
            )
            for dim in (width, height)
        )
        num_patches_x = max(1, crop_width // window_size)
        num_patches_y = max(1, crop_height // window_size)
        return (width, height), (crop_width, crop_height), window_size, num_patches_x, num_patches_y, needs_square_pad

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> int:
        """Return the number of local patches for an image of the given size."""
        images_kwargs = images_kwargs or {}
        size = images_kwargs.get("size", self.size)
        image_size = size["height"]
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        num_patches_x, num_patches_y = self._plan_patches(width, height, image_size, patch_size)[3:5]
        return num_patches_x * num_patches_y

    def _get_image_patches(
        self,
        img: torch.Tensor,
        image_size: int,
        patch_size: int,
        resample: "PILImageResampling",
    ) -> tuple[torch.Tensor, list[torch.Tensor], int, int]:
        """Step3p7-specific cropping: square-pad extreme aspect ratios, resize the global view,
        and slice out raw (pre-final-resize) local-patch tiles per `_plan_patches`'s layout.

        Returns the resized global view, the list of raw local-patch tensors (still at
        `window_size`, not yet resized to `patch_size`), and the patch grid dimensions.
        """
        _, height, width = img.shape
        (
            (global_width, global_height),
            (crop_width, crop_height),
            window_size,
            num_patches_x,
            num_patches_y,
            needs_square_pad,
        ) = self._plan_patches(width, height, image_size, patch_size)

        # Pad extreme-aspect-ratio images to square (original at top-left, zeros elsewhere)
        if needs_square_pad:
            side = max(width, height)
            img = self.pad([img], pad_size=SizeDict(height=side, width=side))[0]

        img_batch = img.unsqueeze(0)

        # Global view: resize to (global_width, global_height) then square to image_size × image_size
        global_img = self.resize(img_batch, SizeDict(height=global_height, width=global_width), resample=resample)
        global_img = self.resize(global_img, SizeDict(height=image_size, width=image_size), resample=resample).squeeze(
            0
        )

        if window_size == 0:
            return global_img, [], num_patches_x, num_patches_y

        img_for_crop = self.resize(
            img_batch, SizeDict(height=crop_height, width=crop_width), resample=resample
        ).squeeze(0)
        patches = divide_to_patches(img_for_crop, patch_size=window_size)
        return global_img, patches, num_patches_x, num_patches_y

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        resample: "PILImageResampling",
        size: SizeDict,
        patch_size: int,
        disable_grouping: bool | None,
        return_tensors: "str | TensorType | None",
        **kwargs,
    ) -> BatchFeature:
        image_size = size.height
        global_images, nested_patches, num_local_patches, patch_newline_masks = [], [], [], []

        for img in images:
            global_img, patches, num_patches_x, num_patches_y = self._get_image_patches(
                img, image_size, patch_size, resample
            )
            global_images.append(global_img)
            num_local_patches.append(len(patches))
            nested_patches.append(patches)
            # Newline after the last patch in each row except the final row
            patch_newline_masks.append(
                [
                    col == num_patches_x - 1 and row < num_patches_y - 1
                    for row in range(num_patches_y)
                    for col in range(num_patches_x)
                ]
            )

        # Global views already share a uniform (image_size × image_size) shape, so batch directly.
        global_stack = self.rescale_and_normalize(
            torch.stack(global_images), do_rescale, rescale_factor, do_normalize, image_mean, image_std
        )

        data = {
            "pixel_values": global_stack,
            "num_local_patches": num_local_patches,
        }
        # Built before the local-patch fields below are assigned: `result[key] = ...` (unlike `data[key] = ...`
        # pre-construction) bypasses `BatchFeature`'s tensor conversion, which matters for
        # `patch_newline_masks` — it must stay a plain list of `bool`s, not a tensor.
        result = BatchFeature(data=data, tensor_type=return_tensors)

        max_patches = max(num_local_patches, default=0)
        if max_patches:
            # Group by shape while keeping each image's patches nested (`is_nested=True`, the same
            # convention Idefics3/Maskformer use for a variable number of sub-images per sample)
            # instead of flattening every image's patches into one list and tracking counts by hand.
            grouped_patches, grouped_index = group_images_by_shape(
                nested_patches, is_nested=True, disable_grouping=disable_grouping
            )
            for shape, stacked_patches in grouped_patches.items():
                resized = self.resize(
                    stacked_patches, SizeDict(height=patch_size, width=patch_size), resample=resample
                )
                grouped_patches[shape] = self.rescale_and_normalize(
                    resized, do_rescale, rescale_factor, do_normalize, image_mean, image_std
                )
            nested_pixel_values_local = reorder_images(grouped_patches, grouped_index, is_nested=True)
            # Flatten back to (total_patches, C, H, W): `Step3p7Model.get_image_features` slices this
            # flat tensor per image using `num_local_patches`.
            result["pixel_values_local"] = torch.stack(
                [patch for per_image_patches in nested_pixel_values_local for patch in per_image_patches]
            )
            # Pad every image's mask to `max_patches` so the output is a uniform (batch, max_patches)
            result["patch_newline_masks"] = [
                mask + [False] * (max_patches - len(mask)) for mask in patch_newline_masks
            ]
        return result

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Step3p7ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)


#  Vision encoder


class Step3p7VisionRotaryEmbedding(Gemma4VisionRotaryEmbedding):
    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (position_ids[..., None].float() * self.inv_freq.to(x.device)).flatten(-2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * self.attention_scaling).to(dtype=x.dtype)
        sin = (emb.sin() * self.attention_scaling).to(dtype=x.dtype)
        return cos, sin


class Step3p7VisionMLP(MiniMaxM3VLVisionMLP):
    pass


class Step3p7VisionAttention(MiniMaxM3VLVisionAttention):
    pass


class Step3p7VisionEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Step3p7VisionConfig):
        super().__init__()
        self.config = config
        self.self_attn = Step3p7VisionAttention(config)
        self.mlp = Step3p7VisionMLP(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lambda_1 = nn.Parameter(config.layer_scale_init_value * torch.ones(config.hidden_size))
        self.lambda_2 = nn.Parameter(config.layer_scale_init_value * torch.ones(config.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.lambda_1 * hidden_states
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.lambda_2 * hidden_states
        hidden_states = hidden_states + residual
        return hidden_states


class Step3p7VisionEmbeddings(SiglipVisionEmbeddings):
    def __init__(self, config: Step3p7VisionConfig):
        super().__init__(config)
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_positions = self.position_embedding.weight.shape[0]
        new_height = height // self.patch_size
        new_width = width // self.patch_size
        sqrt_num_positions = torch_int(num_positions**0.5)
        if not torch.jit.is_tracing() and new_height == sqrt_num_positions and new_width == sqrt_num_positions:
            return self.position_embedding.weight.unsqueeze(0)
        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)
        dim = embeddings.shape[-1]
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bilinear",  # intentionally bilinear; SiGLIP uses bicubic
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed


@auto_docstring
class Step3p7PreTrainedModel(PreTrainedModel):
    config: Step3p7Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Step3p7VisionEncoderLayer", "Step3p7DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_static_cache = True
    _supports_attention_backend = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Step3p7VisionEmbeddings):
            module.register_buffer(
                "position_ids", torch.arange(module.num_positions).expand((1, -1)), persistent=False
            )
        elif isinstance(module, Step3p7VisionEncoderLayer):
            nn.init.constant_(module.lambda_1, module.config.layer_scale_init_value)
            nn.init.constant_(module.lambda_2, module.config.layer_scale_init_value)
        elif isinstance(module, Step3p7RotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)
        elif isinstance(module, Step3p7Experts):
            # `gate_up_proj`/`down_proj` are raw `nn.Parameter(torch.empty(...))` (DeepseekV4Experts);
            # unlike `DeepseekV4PreTrainedModel`/`MiniMaxM3VLPreTrainedModel`, Step3p7 doesn't inherit
            # either parent's `_init_weights`, so without this they stay uninitialized memory forever.
            std = getattr(self.config, "initializer_range", 0.02)
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, Step3p7TopKRouter):
            std = getattr(self.config, "initializer_range", 0.02)
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, Step3p7RMSNorm):
            init.zeros_(module.weight)


@auto_docstring
class Step3p7VisionModel(Step3p7PreTrainedModel):
    """Vision encoder: patch embeddings → 2-D RoPE transformer layers → conv downsampler.

    The rotary embedding (``self.rotary_emb``) and layer stack (``self.layers``) are
    held directly on this module, following the Gemma4 convention of not wrapping
    them in a separate ``Encoder`` submodule.
    """

    config: Step3p7VisionConfig
    _can_record_outputs = {
        "hidden_states": Step3p7VisionEncoderLayer,
        "attentions": Step3p7VisionAttention,
    }

    def __init__(self, config: Step3p7VisionConfig):
        super().__init__(config)
        self.embeddings = Step3p7VisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = Step3p7VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList([Step3p7VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.downsampler1 = nn.Conv2d(config.hidden_size, config.hidden_size * 2, kernel_size=3, stride=2, padding=1)
        self.downsampler2 = nn.Conv2d(
            config.hidden_size * 2, config.hidden_size * 4, kernel_size=3, stride=2, padding=1
        )
        self.post_init()

    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(self, pixel_values: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> BaseModelOutput:
        batch_size, _, height, width = pixel_values.shape
        grid_h = height // self.embeddings.patch_size
        grid_w = width // self.embeddings.patch_size
        hidden_state = self.embeddings(pixel_values, interpolate_pos_encoding=True)
        hidden_state = self.pre_layernorm(hidden_state)
        # (1, grid_h * grid_w, 2) (row, col) position ids for 2-D RoPE. A single grid entry (no
        # temporal/merge dims: t=1, spatial_merge_size=1) broadcasts across the whole batch, since
        # every image in `pixel_values` shares the same (grid_h, grid_w).
        grid_thw = torch.tensor([[1, grid_h, grid_w]], device=hidden_state.device)
        position_ids = get_vision_position_ids(grid_thw, spatial_merge_size=1).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_state, position_ids)
        for layer in self.layers:
            hidden_state = layer(hidden_state, position_embeddings=position_embeddings, **kwargs)
        # `hidden_state.shape[1] == grid_h * grid_w`
        channels = hidden_state.shape[-1]
        hidden_state = hidden_state.permute(0, 2, 1).view(batch_size, channels, grid_h, grid_w)
        hidden_state = self.downsampler1(hidden_state)
        hidden_state = self.downsampler2(hidden_state)
        return BaseModelOutput(last_hidden_state=hidden_state.flatten(2).permute(0, 2, 1))


class Step3p7RotaryEmbedding(LagunaRotaryEmbedding):
    # `LagunaRotaryEmbedding` is `Gemma3RotaryEmbedding` plus `partial_rotary_factor` support
    # (Gemma3 itself never needs partial rotation; Step3p7's full-attention layers do).
    pass


class Step3p7RMSNorm(MiniMaxM3VLRMSNorm):
    pass


class Step3p7MLP(DeepseekV4MLP):
    def __init__(self, config, layer_idx, is_shared_expert=False):
        super().__init__(config)
        self.intermediate_size = config.share_expert_dim if is_shared_expert else config.intermediate_size
        # CODEPATH: stepfun-ai/Step-3.7-Flash clamps layers 43-44 (bound 16) via `swiglu_limits_shared`;
        # a `0.0` entry or no list at all means "no clamp", hence the `or float("inf")`.
        self.limit = (config.swiglu_limits_shared[layer_idx] if config.swiglu_limits_shared else 0) or float("inf")

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x)).clamp(max=self.limit)
        up = self.up_proj(x).clamp(min=-self.limit, max=self.limit)
        return self.down_proj(gate * up)


@no_inherit_decorator
class Step3p7Experts(DeepseekV4Experts):
    _fp8_experts_clamp_after_activation = True

    def __init__(self, config, swiglu_limit=None):
        super().__init__(config)
        self.intermediate_dim = config.moe_intermediate_size
        self.limit = float("inf") if swiglu_limit is None else swiglu_limit

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = self.act_fn(gate).clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return gate * up


class Step3p7TopKRouter(MiniMaxM3VLTopKRouter):
    pass


class Step3p7SparseMoeBlock(MiniMaxM3VLSparseMoeBlock):
    def __init__(self, config, layer_idx):
        nn.Module.__init__(self)
        # CODEPATH: stepfun-ai/Step-3.7-Flash clamps the routed experts on layers 43-44 (bound 7) via
        # `swiglu_limits`; a `0.0` entry or no list at all means "no clamp".
        swiglu_limit = (config.swiglu_limits[layer_idx] or None) if config.swiglu_limits else None
        self.gate = Step3p7TopKRouter(config)
        self.experts = Step3p7Experts(config, swiglu_limit=swiglu_limit)
        self.shared_experts = Step3p7MLP(config, layer_idx, is_shared_expert=True)
        self.routed_scaling_factor = config.moe_router_scaling_factor


class Step3p7Attention(LagunaAttention):
    def __init__(self, config: Step3p7TextConfig, layer_idx: int):
        num_heads = config.num_attention_heads
        super().__init__(config, layer_idx, num_heads)
        self.scaling = config.query_pre_attn_scalar**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        gate_states = self.g_proj(hidden_states)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1)
        output_shape = attn_output.shape
        attn_output = attn_output.view(*output_shape[:-1], self.num_heads, self.head_dim)
        attn_output = attn_output * gate_states.unsqueeze(-1).sigmoid()
        attn_output = attn_output.view(*output_shape)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Step3p7DecoderLayer(LagunaDecoderLayer):
    def __init__(self, config, layer_idx):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        # Inherited `LlamaAttention.__init__` reads `config.num_attention_heads` unconditionally --
        # ambiguous here (per-layer). Resolve it first, then restore `config` identity so
        # `set_attn_implementation` still reaches this module.
        layer_config = config.per_layer_config[layer_idx]
        self.self_attn = Step3p7Attention(layer_config, layer_idx)
        self.self_attn.config = config
        self.attention_type = config.layer_types[layer_idx]

        # CODEPATH: on stepfun-ai/Step-3.7-Flash `moe_layers_enum` marks layers 3-44 `"sparse"` and 0-2
        # `"dense"`; a config without it is all-sparse and never builds the dense branch.
        mlp_class = Step3p7SparseMoeBlock if config.mlp_layer_types[layer_idx] == "sparse" else Step3p7MLP
        self.mlp = mlp_class(config, layer_idx)

        self.input_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Step3p7TextModel(Gemma3TextModel):
    config: Step3p7TextConfig
    _can_record_outputs = {
        "hidden_states": Step3p7DecoderLayer,
        "attentions": Step3p7Attention,
    }

    def __init__(self, config: Step3p7TextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Step3p7DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Step3p7RotaryEmbedding(config=config)
        # CODEPATH: stepfun-ai/Step-3.7-Flash sets `num_nextn_predict_layers=3`, so its weights carry
        # three trailing MTP layers that this branch filters out of the plain (non-MTP) load. A
        # checkpoint without MTP layers leaves the field at 0 and skips it.
        if config.num_nextn_predict_layers:
            # Checkpoints append `num_nextn_predict_layers` MTP layers; ignore them as unexpected keys
            # on regular load. Matches loosely on `layers.<N>.` (not anchored to this model's module
            # path) so it also works as `MtpModel.from_pretrained`'s discovery pattern over raw keys.
            mtp_layers = range(config.num_hidden_layers, config.num_hidden_layers + config.num_nextn_predict_layers)
            patterns = {rf"(^|\.)layers\.{i}\." for i in mtp_layers}
            self._keys_to_ignore_on_load_unexpected = set(self._keys_to_ignore_on_load_unexpected or []) | patterns
        self.post_init()


class Step3p7Model(DeepseekOcr2Model):
    config: Step3p7Config

    def __init__(self, config: Step3p7Config):
        Step3p7PreTrainedModel.__init__(self, config)
        self.vision_model = Step3p7VisionModel(config.vision_config)
        self.language_model = Step3p7TextModel(config.text_config)
        self.vocab_size = config.text_config.vocab_size
        # `* 4`: two stride-2 downsampler convolutions (`downsampler1`, `downsampler2`), each doubling
        # the channel count, so the vision tower's output width is `hidden_size * 2 * 2`.
        self.multi_modal_projector = nn.Linear(
            config.vision_config.hidden_size * 4, config.text_config.hidden_size, bias=config.projector_bias
        )
        self.image_placeholder_token_id = config.image_token_id
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        pixel_values_local: torch.Tensor | None = None,
        num_local_patches: list[int] | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPooling:
        vision_output = self.vision_model(pixel_values, **kwargs)
        image_features = self.multi_modal_projector(vision_output.last_hidden_state)
        patch_image_features = (
            self.multi_modal_projector(
                self.vision_model(pixel_values_local, **{**kwargs, "output_hidden_states": False}).last_hidden_state
            )
            if pixel_values_local is not None
            else None
        )

        if num_local_patches is None:
            num_local_patches = [0] * image_features.shape[0]

        merged = []
        cur_patch_idx = 0
        for i, num_patch in enumerate(num_local_patches):
            cur_feature = []
            if num_patch > 0:
                patch_slice = patch_image_features[cur_patch_idx : cur_patch_idx + num_patch]
                cur_feature.append(patch_slice.view(-1, patch_slice.shape[-1]))
            cur_feature.append(image_features[i].view(-1, image_features.shape[-1]))
            cur_patch_idx += num_patch
            merged.append(torch.cat(cur_feature) if len(cur_feature) > 1 else cur_feature[0])
        return BaseModelOutputWithPooling(
            last_hidden_state=vision_output.last_hidden_state,
            pooler_output=merged,
            hidden_states=vision_output.hidden_states,
            attentions=vision_output.attentions,
        )


class Step3p7ForConditionalGeneration(DeepseekOcr2ForConditionalGeneration):
    config: Step3p7Config


@auto_docstring
class Step3p7Processor(ProcessorMixin):
    """Processor for Step-3.7-Flash.

    Uses :class:`ProcessorMixin.__call__` for the standard image-token expansion
    flow: the image processor splits each image into global + local patch crops,
    then :meth:`replace_image_token` builds the per-image replacement string
    that ``get_text_with_replacements`` substitutes into the text.
    """

    def __init__(self, image_processor, tokenizer=None, chat_template=None, **kwargs) -> None:
        self.image_token = "<im_patch>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token) if tokenizer is not None else None
        stride = image_processor.vision_patch_size * image_processor.downsampler_stride
        self.num_image_feature_size = (image_processor.size["height"] // stride) ** 2
        self.num_patch_feature_size = (image_processor.patch_size // stride) ** 2
        self.image_feature_placeholder = self.image_token * self.num_image_feature_size
        self.patch_feature_placeholder = self.image_token * self.num_patch_feature_size
        super().__init__(image_processor=image_processor, tokenizer=tokenizer, chat_template=chat_template, **kwargs)

    @property
    def unused_input_names(self) -> list[str]:
        return ["patch_newline_masks"]

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        """Return the expanded token string for image *image_idx* (patches + global view)."""
        num_patches = image_inputs["num_local_patches"][image_idx]
        patch_newline_masks = image_inputs.get("patch_newline_masks")
        patch_newline_mask = patch_newline_masks[image_idx] if patch_newline_masks is not None else None
        repl = ""
        for i in range(num_patches):
            repl += f"<patch_start>{self.patch_feature_placeholder}<patch_end>"
            if patch_newline_mask and patch_newline_mask[i]:
                repl += "<patch_newline>"
        repl += f"<im_start>{self.image_feature_placeholder}<im_end>"
        return repl
