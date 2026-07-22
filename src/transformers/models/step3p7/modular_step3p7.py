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
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, RopeParameters
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
from ...utils.generic import maybe_autocast, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ...vision_utils import get_vision_position_ids
from ..deepseek_ocr2.modeling_deepseek_ocr2 import DeepseekOcr2ForConditionalGeneration, DeepseekOcr2Model
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts, DeepseekV4MLP
from ..gemma3.modeling_gemma3 import Gemma3Attention, Gemma3TextModel
from ..gemma4.modeling_gemma4 import Gemma4VisionRotaryEmbedding
from ..laguna.modeling_laguna import LagunaRotaryEmbedding, apply_rotary_pos_emb, eager_attention_forward
from ..minimax_m3_vl.modeling_minimax_m3_vl import (
    MiniMaxM3VLDecoderLayer,
    MiniMaxM3VLRMSNorm,
    MiniMaxM3VLSparseMoeBlock,
    MiniMaxM3VLTopKRouter,
    MiniMaxM3VLVisionAttention,
    MiniMaxM3VLVisionMLP,
)
from ..qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision  # noqa: F401
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..siglip.modeling_siglip import SiglipVisionEmbeddings


logger = logging.get_logger(__name__)

__all__ = [
    "Step3p7ForConditionalGeneration",
    "Step3p7Model",
    "Step3p7VisionConfig",
    "Step3p7TextConfig",
    "Step3p7Config",
    "Step3p7ImageProcessor",
    "Step3p7Processor",
]


# ── Config ────────────────────────────────────────────────────────────────────


@auto_docstring
@strict
class Step3p7VisionConfig(SiglipVisionConfig):
    r"""
    mlp_ratio (`float`, *optional*, defaults to `8960/1536`):
        Ratio of MLP hidden size to `hidden_size`; `intermediate_size` is set to
        `int(hidden_size * mlp_ratio)`.
    layer_scale_init_value (`float`, *optional*, defaults to 0.1):
        Initial value for per-channel residual-scale parameters.
    """

    model_type = "step3p5_vision"
    base_config_key = "vision_config"
    # Backward-compat key aliases from legacy config.json (e.g. width → hidden_size)
    attribute_map = {
        "width": "hidden_size",
        "layers": "num_hidden_layers",
        "heads": "num_attention_heads",
        "ls_init_value": "layer_scale_init_value",
    }

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
        # Call `PreTrainedConfig.__post_init__` directly (not `super()`): `SiglipVisionConfig` has no
        # `__post_init__` for the modular converter to splice a `super().func()` into. This also applies
        # attribute_map aliases (e.g. width -> hidden_size) before `intermediate_size` is computed below.
        # `rope_parameters` defaults come from `RotaryEmbeddingConfigMixin` automatically; no need to seed them.
        PreTrainedConfig.__post_init__(self, **kwargs)
        self.intermediate_size = int(self.hidden_size * self.mlp_ratio)


@auto_docstring
@strict
class Step3p7TextConfig(PreTrainedConfig):
    r"""
    moe_intermediate_size (`int`, *optional*, defaults to 1280):
        Intermediate size of each routed expert.
    n_routed_experts (`int`, *optional*, defaults to 288):
        Total number of routed experts. Accessible as `num_local_experts` via `attribute_map`.
    share_expert_dim (`int`, *optional*, defaults to 1280):
        Intermediate size of the always-active shared expert.
    layer_types (`list[str]`, *optional*):
        Per-layer attention type; `"full_attention"` or `"sliding_attention"`. Defaults to all
        `"full_attention"` if not provided.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer MLP type; `"sparse"` for MoE layers, `"dense"` otherwise. If not provided, derived
        from the legacy `moe_layers_enum` hub-config kwarg (a comma-separated string or list of MoE
        layer indices), defaulting to all layers from index 3 onward being MoE.
    swiglu_limits (`list[float | None]`, *optional*):
        Per-layer gate/up clamping bound; `None` means no clamping.
    num_nextn_predict_layers (`int`, *optional*):
        Legacy hub-config kwarg: number of trailing speculative-decoding ("MTP") layers this
        implementation doesn't model. When set, every per-layer list above (plus `rope_theta` and
        `partial_rotary_factors` below) is trimmed from `num_hidden_layers + num_nextn_predict_layers`
        back down to `num_hidden_layers` entries. Example: with `num_hidden_layers=4` and
        `num_nextn_predict_layers=1`, an incoming
        `layer_types=["full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]`
        (5 entries: 4 real layers + 1 MTP layer) is trimmed to
        `["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"]` (the
        trailing MTP entry is dropped). Persisted as `self.num_nextn_predict_layers` (rather than
        only consumed here and discarded) so `Step3p7TextModel` can compute which trailing layer
        indices are MTP layers at load time, instead of a hardcoded checkpoint-specific range. Also
        exposed generically as `num_mtp_layers` (via `attribute_map`, the same convention used by
        `DeepseekV3Config`/`Glm4MoeConfig`), which lets `PreTrainedConfig.get_mtp_config()` build the
        real MTP submodel for `generate(..., use_mtp=True)`.
    rope_theta (`float | list[float]`, *optional*, defaults to 10000.0):
        Legacy hub-config kwarg. Real checkpoints give one value *per decoder layer* rather than one
        value for the whole model; since it only ever varies by `layer_types[layer_idx]`, this is
        collapsed into one theta per layer type. Superseded by `rope_parameters` once set.
    partial_rotary_factors (`list[float]`, *optional*):
        Legacy hub-config kwarg, one value per decoder layer; collapsed into one value per layer type
        the same way as `rope_theta`.
    max_seq_len (`int`, *optional*, defaults to 128000):
        Legacy hub-config kwarg mirroring `max_position_embeddings` (real checkpoints ship both); not
        currently read by the modeling code.
    norm_expert_weight (`bool`, *optional*, defaults to `True`):
        Legacy hub-config kwarg from the original checkpoint; not currently read by the modeling code
        (`Step3p7TopKRouter` always normalizes top-k expert weights to sum to 1).
    num_sliding_attention_heads (`int`, *optional*):
        Number of attention heads for `"sliding_attention"` layers, when different from
        `num_attention_heads`. Derived from `attention_other_setting` if not provided.
    attention_other_setting (`dict`, *optional*):
        Legacy hub-config dict overriding `num_attention_heads`/`num_key_value_heads`/`head_dim` for
        `"sliding_attention"` layers (real checkpoints use a different head count per attention type).
    num_attention_heads_per_layer (`list[int]`, *optional*):
        Resolved (not a hub-config kwarg) per-layer head count, one entry per decoder layer: each
        entry is `num_sliding_attention_heads` or `num_attention_heads` depending on
        `layer_types[layer_idx]`. Computed once here (the same pattern `LagunaConfig` uses) so
        `Step3p7Attention` takes its head count as a plain constructor argument instead of picking
        between the two scalar fields itself.
    query_pre_attn_scalar (`int` or `float`, *optional*):
        `Gemma3Attention.__init__` hook point (`Step3p7Attention` inherits from it): defaults to
        `head_dim`, giving the standard `head_dim ** -0.5` attention scaling rather than Gemma3's own
        hyperparameter of the same name.
    attn_logit_softcapping (`float`, *optional*):
        `Gemma3Attention.__init__` hook point; unused by Step3p7's own `eager_attention_forward`, so
        always `None` (no softcapping).
    use_head_wise_attn_gate (`bool`, *optional*, defaults to `False`):
        Legacy hub-config kwarg from the original checkpoint; not currently read by the modeling code.
    use_moe_router_bias (`bool`, *optional*, defaults to `False`):
        Legacy hub-config kwarg from the original checkpoint; not currently read by the modeling code.
    moe_router_activation (`str`, *optional*, defaults to `"softmax"`):
        Legacy hub-config kwarg from the original checkpoint; not currently read by the modeling code
        (`Step3p7TopKRouter` always applies a sigmoid).
    moe_router_scaling_factor (`float`, *optional*, defaults to 1.0):
        Scaling factor applied to the MoE block's routed-expert output (`routed_scaling_factor` in
        `Step3p7SparseMoeBlock`).
    need_fp32_gate (`bool`, *optional*, defaults to `False`):
        Legacy hub-config kwarg from the original checkpoint; not currently read by the modeling code.
    swiglu_limits_shared (`list[float | int | None]`, *optional*):
        Per-layer gate/up clamping bound for the always-active shared expert; `None` means no clamping.
    use_rope_layers (`list[bool]`, *optional*):
        Legacy hub-config kwarg from the original checkpoint; not currently read by the modeling code.
    yarn_only_types (`list[str]`, *optional*):
        Legacy hub-config kwarg from the original checkpoint; not currently read by the modeling code.
    use_bidirectional_attention (`bool`, *optional*, defaults to `False`):
        Legacy hub-config kwarg from the original checkpoint; not currently read by the modeling code.
    """

    model_type = "step3p5"
    base_config_key = "text_config"
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "num_attention_groups": "num_key_value_heads",
        "moe_num_experts": "n_routed_experts",
        "moe_top_k": "num_experts_per_tok",
        "share_expert_dims": "share_expert_dim",
        "num_mtp_layers": "num_nextn_predict_layers",
    }
    # Copied (not inherited) from `MiniMaxM3VLTextConfig`: `Step3p7Attention`/`Step3p7SparseMoeBlock`
    # reuse its `q_proj`/`k_proj`/`v_proj`/`o_proj` and `experts.gate_up_proj`/`down_proj`/`gate`
    # submodule naming verbatim, so the same TP/PP/EP plans apply unchanged.
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_gather_output",
        "layers.*.self_attn.k_proj": "colwise_gather_output",
        "layers.*.self_attn.v_proj": "colwise_gather_output",
        "layers.*.self_attn.o_proj": "rowwise_split_input",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    hidden_size: int = 4096
    intermediate_size: int = 11264
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    num_hidden_layers: int = 45
    max_position_embeddings: int = 128000
    max_seq_len: int = 128000
    vocab_size: int = 128815
    rms_norm_eps: float = 1e-5
    moe_intermediate_size: int = 1280
    n_routed_experts: int = 288
    num_experts_per_tok: int = 8
    rope_parameters: RopeParameters | dict | None = None
    share_expert_dim: int = 1280
    head_dim: int = 128
    norm_expert_weight: bool = True
    layer_types: list[str] | None = None
    mlp_layer_types: list[str] | None = None
    sliding_window: int | None = None
    num_sliding_attention_heads: int | None = None
    num_attention_heads_per_layer: list[int] | None = None
    attention_other_setting: dict | None = None
    pad_token_id: int = 1
    attention_dropout: float = 0.0
    attention_bias: bool = False
    # `Gemma3Attention.__init__` hook points (`Step3p7Attention` inherits from it): `None` resolves
    # to the standard `head_dim ** -0.5` scaling below instead of Gemma3's own hyperparameter, and
    # Step3p7 never applies logit softcapping (its own `eager_attention_forward` doesn't read it).
    query_pre_attn_scalar: int | float | None = None
    attn_logit_softcapping: float | None = None
    use_head_wise_attn_gate: bool = False
    use_moe_router_bias: bool = False
    moe_router_activation: str = "softmax"
    moe_router_scaling_factor: float = 1.0
    need_fp32_gate: bool = False
    hidden_act: str = "silu"
    mlp_bias: bool = False
    swiglu_limits: list[float | int | None] | None = None
    swiglu_limits_shared: list[float | int | None] | None = None
    use_rope_layers: list[bool] | None = None
    yarn_only_types: list[str] | None = None
    use_bidirectional_attention: bool | None = False

    def __post_init__(self, **kwargs):
        # Legacy hub configs pad every per-layer list below with `num_nextn_predict_layers` trailing
        # speculative-decoding ("MTP") entries this implementation doesn't model; trim back down to
        # `num_hidden_layers`. E.g. `num_hidden_layers=4, num_nextn_predict_layers=1`: an incoming
        # 5-entry `layer_types` (4 real layers + 1 MTP) is trimmed to just the 4 real layers.
        num_nextn_predict_layers = kwargs.pop("num_nextn_predict_layers", 0)
        self.num_nextn_predict_layers = num_nextn_predict_layers
        if num_nextn_predict_layers:
            n, padded = self.num_hidden_layers, self.num_hidden_layers + num_nextn_predict_layers
            trimmed_fields = []
            for field in (
                "layer_types",
                "mlp_layer_types",
                "swiglu_limits",
                "swiglu_limits_shared",
                "use_rope_layers",
            ):
                value = getattr(self, field)
                if isinstance(value, list) and len(value) == padded:
                    setattr(self, field, value[:n])
                    trimmed_fields.append(field)
            for key in ("rope_theta", "partial_rotary_factors"):
                value = kwargs.get(key)
                if isinstance(value, list) and len(value) == padded:
                    kwargs[key] = value[:n]
                    trimmed_fields.append(key)
            if trimmed_fields:
                logger.info(
                    f"Trimmed {trimmed_fields} from {padded} to {n} entries "
                    f"(num_nextn_predict_layers={num_nextn_predict_layers} speculative-decoding "
                    "layers this implementation doesn't model)."
                )

        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        if self.mlp_layer_types is None:
            # `moe_layers_enum` is a legacy hub-config alias for `mlp_layer_types` (a comma-separated
            # or list of layer indices that are MoE); it's only ever read here, never persisted.
            moe_layers_enum = kwargs.pop("moe_layers_enum", None)
            if moe_layers_enum is not None:
                items = moe_layers_enum.split(",") if isinstance(moe_layers_enum, str) else moe_layers_enum
                moe_set = {int(i) for i in items if str(i).strip()}
            else:
                moe_set = set(range(3, self.num_hidden_layers))
            self.mlp_layer_types = ["sparse" if i in moe_set else "dense" for i in range(self.num_hidden_layers)]

        if self.num_sliding_attention_heads is None:
            if self.attention_other_setting:
                self.num_sliding_attention_heads = self.attention_other_setting.get(
                    "num_attention_heads", self.num_attention_heads
                )
            else:
                self.num_sliding_attention_heads = self.num_attention_heads

        if self.num_attention_heads_per_layer is None:
            self.num_attention_heads_per_layer = [
                self.num_sliding_attention_heads if layer_type == "sliding_attention" else self.num_attention_heads
                for layer_type in self.layer_types
            ]

        if self.query_pre_attn_scalar is None:
            self.query_pre_attn_scalar = self.head_dim

        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        # Overridden: the generic version's `setdefault("rope_theta", ...)` on the outer dict would
        # corrupt our layer-type-keyed `rope_parameters`. `rope_theta`/`partial_rotary_factors` are
        # popped from kwargs (not fields, so not redundantly persisted); checkpoints give one value per
        # decoder layer, so zipping against `layer_types` collapses that to one value per type.
        rope_scaling = kwargs.pop("rope_scaling", None) or self.rope_scaling
        rope_theta = kwargs.pop("rope_theta", self.default_theta)
        partial_rotary_factors = kwargs.pop("partial_rotary_factors", None)
        if self.rope_parameters is None:
            if isinstance(rope_theta, list):
                theta_by_type = dict(zip(self.layer_types, rope_theta))
            else:
                theta_by_type = dict.fromkeys(set(self.layer_types), rope_theta)
            factor_by_type = dict(zip(self.layer_types, partial_rotary_factors)) if partial_rotary_factors else {}
            self.rope_parameters = {
                layer_type: {"rope_theta": theta_by_type[layer_type]}
                | ({"partial_rotary_factor": factor_by_type[layer_type]} if layer_type in factor_by_type else {})
                for layer_type in set(self.layer_types)
            }
            if rope_scaling and "full_attention" in self.rope_parameters:
                self.rope_parameters["full_attention"].update(rope_scaling)
        self.standardize_rope_params()
        return kwargs


@auto_docstring
@strict
class Step3p7Config(PreTrainedConfig):
    r"""
    vision_config (`dict` or [`Step3p7VisionConfig`], *optional*):
        Vision encoder configuration. Defaults to `Step3p7VisionConfig()`.
    text_config (`dict` or [`Step3p7TextConfig`], *optional*):
        Text decoder configuration. Defaults to `Step3p7TextConfig()`.
    image_token_id (`int`, *optional*, defaults to 151679):
        Token ID used as the image placeholder in the text sequence.
    projector_bias (`bool`, *optional*, defaults to `False`):
        Whether the vision-to-text projection uses a bias term.
    """

    model_type = "step3p7"
    sub_configs = {"vision_config": Step3p7VisionConfig, "text_config": Step3p7TextConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    projector_bias: bool = False
    image_token_id: int = 151679
    max_position_embeddings: int | None = None

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

        self.max_position_embeddings = self.text_config.max_position_embeddings
        super().__post_init__(**kwargs)


# ── Image Processor ───────────────────────────────────────────────────────────


class Step3p7ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to 504):
        Target size (height = width) for each local patch crop.
    num_image_features (`int`, *optional*, defaults to 169):
        Number of placeholder tokens the processor expands the global-view image into. Must match
        `(size["height"] // vision_patch_size // downsampler_stride) ** 2` for the paired vision
        encoder, so only override this alongside a matching `size` change.
    num_patch_features (`int`, *optional*, defaults to 81):
        Number of placeholder tokens the processor expands each local patch crop into; the
        `patch_size` analogue of `num_image_features`.
    """

    patch_size: int
    num_image_features: int
    num_patch_features: int


@auto_docstring
class Step3p7ImageProcessor(TorchvisionBackend):
    """
    Image processor for Step-3.7-Flash.

    Each input image is split into a global down-scaled view plus zero or more
    local patch crops via a sliding-window strategy, then every sub-image is
    resized and normalised independently.

    Args:
        size (`dict[str, int]`, *optional*, defaults to `{"height": 728, "width": 728}`):
            Target size for the global image view.
        patch_size (`int`, *optional*, defaults to 504):
            Target size (height = width) for each local patch crop.
        image_mean (`list[float]`, *optional*):
            Per-channel mean for normalisation; defaults to ``OPENAI_CLIP_MEAN``.
        image_std (`list[float]`, *optional*):
            Per-channel std for normalisation; defaults to ``OPENAI_CLIP_STD``.
    """

    resample = PILImageResampling.BILINEAR
    size = {"height": 728, "width": 728}
    default_to_square = True
    patch_size: int = 504
    do_rescale = True
    rescale_factor: float = 1 / 255
    do_normalize = True
    image_mean: list[float] = OPENAI_CLIP_MEAN
    image_std: list[float] = OPENAI_CLIP_STD
    do_convert_rgb = True
    valid_kwargs = Step3p7ImageProcessorKwargs
    model_input_names = ["pixel_values", "pixel_values_local", "num_local_patches"]

    MAX_IMAGE_SIZE: int = 3024
    # (image_size / patch_size / downsampler_stride)^2: 728→169 tokens, 504→81 tokens
    num_image_features: int = 169
    num_patch_features: int = 81

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
          - images larger than ``MAX_IMAGE_SIZE`` are scaled down uniformly

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
        if max(height, width) > self.MAX_IMAGE_SIZE:
            scale = self.MAX_IMAGE_SIZE / max(height, width)
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

        # Step 3 — snap to window-size multiples
        def _snap(dim: int) -> int:
            return (
                window_size * (dim // window_size + (dim % window_size > 0.2 * window_size))
                if dim >= window_size
                else dim
            )

        crop_width, crop_height = _snap(width), _snap(height)
        num_patches_x = max(1, crop_width // window_size)
        num_patches_y = max(1, crop_height // window_size)
        return (width, height), (crop_width, crop_height), window_size, num_patches_x, num_patches_y, needs_square_pad

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> tuple[int, int]:
        """Return ``(num_patches, num_newline_tokens)`` for an image of the given size."""
        images_kwargs = images_kwargs or {}
        size = images_kwargs.get("size", self.size)
        image_size = size["height"]
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        *_, num_patches_x, num_patches_y, _ = self._plan_patches(width, height, image_size, patch_size)
        num_patches = num_patches_x * num_patches_y
        num_newlines = num_patches_y - 1 if num_patches > 0 else 0
        return num_patches, num_newlines

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
        plan = self._plan_patches(width, height, image_size, patch_size)
        (global_width, global_height), (crop_width, crop_height), window_size, num_patches_x, num_patches_y = plan[:5]
        needs_square_pad = plan[5]

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
            data["pixel_values_local"] = torch.stack(
                [patch for per_image_patches in nested_pixel_values_local for patch in per_image_patches]
            )

        result = BatchFeature(data=data, tensor_type=return_tensors)
        if max_patches:
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
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        dim_freqs = []
        for i in range(2):
            dim_pos = position_ids[:, :, i][:, None, :].float()
            with maybe_autocast(device_type=device_type, enabled=False):
                dim_freqs.append((inv_freq_expanded.float() @ dim_pos.float()).transpose(1, 2))
        freqs = torch.cat(dim_freqs, dim=-1)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * self.attention_scaling).to(dtype=x.dtype)
        sin = (emb.sin() * self.attention_scaling).to(dtype=x.dtype)
        return cos, sin


class Step3p7VisionMLP(MiniMaxM3VLVisionMLP):
    pass


class Step3p7VisionAttention(MiniMaxM3VLVisionAttention):
    pass


class Step3p7VisionEncoderLayer(nn.Module):
    """Vision encoder layer with layer scale (``lambda_1``/``lambda_2``, naming convention shared with
    :class:`~transformers.models.internvl.modeling_internvl.InternVLVisionLayer`) and RoPE-aware 2-D
    attention via ``position_embeddings`` forwarding.
    """

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
        attn_out, _ = self.self_attn(
            self.layernorm_before(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.lambda_1 * attn_out + hidden_states
        return self.lambda_2 * self.mlp(self.layernorm_after(hidden_states)) + hidden_states


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


class Step3p7PreTrainedModel(PreTrainedModel):
    config: Step3p7Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
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
            # `Step3p7RMSNorm` computes `normed * (weight + 1)`, so zero (not the default all-ones
            # from `torch.ones(hidden_size)`) is the identity-scale initialization.
            init.zeros_(module.weight)


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

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(self, pixel_values: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> BaseModelOutput:
        _, _, height, width = pixel_values.shape
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
        batch_size, num_patches, channels = hidden_state.shape
        # `num_patches == grid_h * grid_w` (no CLS token is ever prepended), so reuse the grid
        # dimensions already computed above instead of assuming a square grid via `num_patches**0.5`.
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


def _swiglu_clamp(gate: torch.Tensor, up: torch.Tensor, act_fn: Callable, limit: float) -> torch.Tensor:
    """Step3p7 order: SiLU then clamp (DeepseekV4 clamps before SiLU)."""
    return act_fn(gate).clamp(max=limit) * up.clamp(min=-limit, max=limit)


class Step3p7MLP(DeepseekV4MLP):
    def __init__(self, config, intermediate_size=None, swiglu_limit=None):
        super().__init__(config)
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.limit = float("inf") if swiglu_limit is None else swiglu_limit

    def forward(self, x):
        return self.down_proj(_swiglu_clamp(self.gate_proj(x), self.up_proj(x), self.act_fn, self.limit))


@no_inherit_decorator
class Step3p7Experts(DeepseekV4Experts):
    _fp8_experts_clamp_after_activation = True

    def __init__(self, config, swiglu_limit=None):
        super().__init__(config)
        self.intermediate_dim = config.moe_intermediate_size
        self.limit = float("inf") if swiglu_limit is None else swiglu_limit

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return _swiglu_clamp(gate, up, self.act_fn, self.limit)


class Step3p7TopKRouter(MiniMaxM3VLTopKRouter):
    pass


class Step3p7SparseMoeBlock(MiniMaxM3VLSparseMoeBlock):
    def __init__(self, config, layer_idx):
        nn.Module.__init__(self)
        swiglu_limit = (config.swiglu_limits[layer_idx] or None) if config.swiglu_limits else None
        swiglu_limit_shared = (config.swiglu_limits_shared[layer_idx] or None) if config.swiglu_limits_shared else None
        self.gate = Step3p7TopKRouter(config)
        self.experts = Step3p7Experts(config, swiglu_limit=swiglu_limit)
        self.shared_experts = Step3p7MLP(
            config, intermediate_size=config.share_expert_dim, swiglu_limit=swiglu_limit_shared
        )
        self.routed_scaling_factor = getattr(config, "moe_router_scaling_factor", 1.0)


class Step3p7Attention(Gemma3Attention):
    def __init__(self, config: Step3p7TextConfig, layer_idx: int, num_attention_heads: int):
        self.num_attention_heads = num_attention_heads
        super().__init__(config, layer_idx)
        self.num_key_value_groups = self.num_attention_heads // config.num_key_value_heads
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.g_proj = nn.Linear(config.hidden_size, self.num_attention_heads, bias=False)

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
        attn_output = (
            attn_output.view(*attn_output.shape[:-1], self.num_attention_heads, self.head_dim)
            * gate_states.unsqueeze(-1).sigmoid()
        ).view(*attn_output.shape)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Step3p7DecoderLayer(MiniMaxM3VLDecoderLayer):
    def __init__(self, config, layer_idx):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Step3p7Attention(config, layer_idx, config.num_attention_heads_per_layer[layer_idx])
        self.attention_type = config.layer_types[layer_idx]

        swiglu_limit_shared = (config.swiglu_limits_shared[layer_idx] or None) if config.swiglu_limits_shared else None
        self.mlp = (
            Step3p7SparseMoeBlock(config, layer_idx)
            if config.mlp_layer_types[layer_idx] == "sparse"
            else Step3p7MLP(config, swiglu_limit=swiglu_limit_shared)
        )

        self.input_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Step3p7TextModel(Gemma3TextModel):
    _no_split_modules = ["Step3p7DecoderLayer"]
    config: Step3p7TextConfig
    _can_record_outputs = {"hidden_states": Step3p7DecoderLayer}

    def __init__(self, config: Step3p7TextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Step3p7DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Step3p7RotaryEmbedding(config=config)
        if config.num_nextn_predict_layers:
            # Checkpoints append `num_nextn_predict_layers` unmodeled speculative-decoding layers right
            # after the real ones. Ignore them on load instead of reporting them as unexpected keys.
            mtp_layers = range(config.num_hidden_layers, config.num_hidden_layers + config.num_nextn_predict_layers)
            pattern = rf"language_model\.layers\.({'|'.join(map(str, mtp_layers))})\."
            self._keys_to_ignore_on_load_unexpected = set(self._keys_to_ignore_on_load_unexpected or []) | {pattern}
        self.post_init()


class Step3p7Model(DeepseekOcr2Model):
    config: Step3p7Config

    def __init__(self, config: Step3p7Config):
        Step3p7PreTrainedModel.__init__(self, config)
        self.vision_model = Step3p7VisionModel(config.vision_config)
        self.language_model = Step3p7TextModel(config.text_config)
        self.vocab_size = config.text_config.vocab_size
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
        )


class Step3p7ForConditionalGeneration(DeepseekOcr2ForConditionalGeneration):
    config: Step3p7Config


# ── Processor ─────────────────────────────────────────────────────────────────


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
        self.num_image_feature_size = image_processor.num_image_features
        self.num_patch_feature_size = image_processor.num_patch_features
        self.image_feature_placeholder = self.image_token * self.num_image_feature_size
        self.patch_feature_placeholder = self.image_token * self.num_patch_feature_size
        super().__init__(image_processor=image_processor, tokenizer=tokenizer, chat_template=chat_template, **kwargs)

    @property
    def unused_input_names(self) -> list[str]:
        return ["patch_newline_masks"]

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
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
