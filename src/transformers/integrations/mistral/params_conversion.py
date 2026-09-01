# Copyright 2026 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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
"""Config conversion for Mistral models.

params.json to native lives in `native_config.py`; this module converts native to and from HF.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...models.ministral3.configuration_ministral3 import Ministral3Config
from ...models.mistral.configuration_mistral import MistralConfig
from ...models.mistral3.configuration_mistral3 import Mistral3Config
from ...models.mistral4.configuration_mistral4 import Mistral4Config
from ...models.pixtral.configuration_pixtral import PixtralVisionConfig
from ...quantizers.auto import AutoQuantizationConfig
from ...utils.quantization_config import QuantizationConfigMixin
from .native_config import (
    _QUANTIZATION_SCHEME_MAP,
    Llama4Scaling,
    MistralNativeConfig,
    MOEModelArgs,
    QFormat,
    QuantizationArgs,
    VisionEncoderArgs,
    YarnArgs,
)


_REVERSE_QUANTIZATION_SCHEME_MAP = {v: k for k, v in _QUANTIZATION_SCHEME_MAP.items()}


MistralHFConfigType = MistralConfig | Mistral3Config | Ministral3Config | Mistral4Config


_MISTRAL_EXTRAS_KEY = "mistral_extras"
_MISTRAL_EXTRAS_DOC = (
    "params.json fields with no HF equivalent, kept for reverse conversion. Not read by modeling code."
)


# Native `params.json` fields with no home in the HF config schema, preserved in `mistral_extras`
# on the outermost config so the native config can be rebuilt losslessly.
_RESIDUAL_FIELD_NAMES: dict[str, tuple[str, ...]] = {
    "moe": ("expert_parallel", "expert_model_parallel", "route_every_n"),
    "vision_encoder": ("max_image_size", "image_break_token_id", "image_end_token_id"),
}


def _build_mistral_extras(moe: MOEModelArgs | None, vision_encoder: VisionEncoderArgs | None) -> dict[str, Any]:
    """Build the `mistral_extras` config entry from residual native fields.

    Args:
        moe (`MOEModelArgs`):
            The native MoE args to read residual fields from, or `None`.
        vision_encoder (`VisionEncoderArgs`):
            The native vision encoder args to read residual fields from, or `None`.

    Returns:
        A dict with a `moe` and/or `vision_encoder` sub-dict (only present when the
        corresponding native section exists) plus a `_comment` entry, or `{}` when there
        is nothing to preserve.
    """
    extras: dict[str, Any] = {}
    if moe is not None:
        extras["moe"] = {field: getattr(moe, field) for field in _RESIDUAL_FIELD_NAMES["moe"]}
    if vision_encoder is not None:
        extras["vision_encoder"] = {
            field: getattr(vision_encoder, field) for field in _RESIDUAL_FIELD_NAMES["vision_encoder"]
        }
    if not extras:
        return {}
    extras["_comment"] = _MISTRAL_EXTRAS_DOC
    return extras


def _extract_residual_fields(extras: dict[str, Any] | None, section_name: str) -> dict[str, Any]:
    """Read a `mistral_extras` residual section.

    Args:
        extras (`dict`):
            The parsed `mistral_extras` payload, or `None`.
        section_name (`str`):
            Which residual section to read, a key of `_RESIDUAL_FIELD_NAMES`.

    Returns:
        The residual fields `mistral_extras` actually carries for `section_name`. A field the
        payload omits is left out of the result entirely, so the section dataclass's own
        constructor default applies.
    """
    field_names = _RESIDUAL_FIELD_NAMES[section_name]
    section = (extras or {}).get(section_name) or {}
    return {name: section[name] for name in field_names if name in section}


def mistral_native_config_to_hf_config(
    native_config: MistralNativeConfig,
) -> MistralHFConfigType:
    """Convert a native Mistral config to the corresponding HF config, attaching residue.

    Args:
        native_config (`MistralNativeConfig`):
            The native config to convert.

    Returns:
        The HF config, with any residual native fields with no HF equivalent attached as
        a `mistral_extras` attribute so the native config can be rebuilt losslessly.
    """
    hf_config = _dispatch_mistral_native_config_to_hf_config(native_config)
    extras = _build_mistral_extras(native_config.moe, native_config.vision_encoder)
    if extras:
        setattr(hf_config, _MISTRAL_EXTRAS_KEY, extras)
    return hf_config


def _dispatch_mistral_native_config_to_hf_config(
    native_config: MistralNativeConfig,
) -> MistralHFConfigType:
    """Dispatch a native Mistral config to the correct HF config class.

    The mapping is the following:
    - If it has vision, it is mapped to Mistral 3 that will resolve sub text config using the same function.
    - If it is a MOE, it should also be MLA and it is mapped to Mistral 4.
    - If it has yarn, it is mapped to Ministral 3 else to Mistral.

    Args:
        native_config (`MistralNativeConfig`):
            The native config to dispatch.

    Returns:
        The HF config class matching `native_config`'s architecture.

    Raises:
        ValueError: If `native_config` combines MoE and MLA inconsistently, or if
            `native_config.vision_encoder` is set to a type other than `VisionEncoderArgs` or `None`.
    """
    is_moe = native_config.moe is not None
    is_mla = native_config.q_lora_rank is not None
    has_yarn = native_config.yarn is not None

    if not (is_moe == is_mla):
        raise ValueError("MOE and MLA config are only supported together. Please ensure to have a valid model config.")

    match native_config.vision_encoder, is_moe, has_yarn:
        case VisionEncoderArgs() as vision_encoder, _, _:
            return _native_config_to_mistral3(native_config=native_config, vision_encoder=vision_encoder)
        case None, True, _:
            return _native_config_to_mistral4(native_config=native_config)
        case None, False, True:
            return _native_config_to_text_config(native_config=native_config, config_cls=Ministral3Config)
        case None, False, False:
            return _native_config_to_text_config(native_config=native_config, config_cls=MistralConfig)
        case _:
            raise ValueError(
                f"Unsupported vision_encoder type {type(native_config.vision_encoder).__name__}; "
                "expected a VisionEncoderArgs instance or None."
            )


def _get_maybe_quant_config(
    is_vision_model: bool, quantization_args: QuantizationArgs | None
) -> QuantizationConfigMixin | None:
    """Build an HF quantization config from Mistral native quantization args.

    Args:
        is_vision_model (`bool`):
            If `True`, adds vision/projector modules to `modules_to_not_convert`.
        quantization_args (`QuantizationArgs`):
            Native quantization parameters, or `None`. Returns `None` when this is `None`.

    Returns:
        The HF quantization config, or `None` if no quantization is configured.

    Raises:
        ValueError: If the quantization format is not `FP8_E4M3`.
    """
    if quantization_args is None:
        return None

    modules_to_not_convert = ["lm_head"]
    if is_vision_model:
        modules_to_not_convert += [
            "model.vision_tower",
            "model.multi_modal_projector",
        ]

    match quantization_args.qformat_weight:
        case QFormat.FP8_E4M3:
            # `QuantizationArgs.__post_init__` already validated `qscheme_act` against
            # `_QUANTIZATION_SCHEME_MAP`'s keys, so this lookup cannot miss.
            activation_scheme = _QUANTIZATION_SCHEME_MAP[quantization_args.qscheme_act]
            quantization_config = {
                "activation_scheme": activation_scheme,
                "modules_to_not_convert": modules_to_not_convert,
                "quant_method": "fp8",
                "weight_block_size": None,
            }
            return AutoQuantizationConfig.from_dict(quantization_config)
        case _:
            raise ValueError(f"invalid quantization config {quantization_args.qformat_weight=}.")


def _get_rope_parameters(
    rope_theta: float,
    yarn_args: YarnArgs | None,
    llama4_scaling: Llama4Scaling | None,
    qk_rope: int | None,
    qk_nope: int | None,
) -> RopeParameters:
    """Build a `RopeParameters` dict from native rope/yarn/MLA fields.

    Args:
        rope_theta (`float`):
            The base RoPE theta value.
        yarn_args (`YarnArgs`):
            The native YaRN parameters, or `None` for default (non-extended) RoPE.
        llama4_scaling (`Llama4Scaling`):
            The native Llama 4 scaling parameters, or `None`. Only meaningful when `yarn_args`
            is also set.
        qk_rope (`int`):
            The MLA `qk_rope_head_dim`, or `None` for a non-MLA architecture.
        qk_nope (`int`):
            The MLA `qk_nope_head_dim`, or `None` for a non-MLA architecture.

    Returns:
        The `RopeParameters` dict for the HF config.

    Raises:
        ValueError: If `qk_rope` and `qk_nope` are not both set or both `None`.
    """
    if (qk_rope is None) != (qk_nope is None):
        raise ValueError(f"qk_rope and qk_nope must both be None or both set, got {qk_rope=}, {qk_nope=}")
    rope_kwargs = {}

    if qk_rope is not None and qk_nope is not None:
        rope_kwargs["partial_rotary_factor"] = qk_rope / (qk_nope + qk_rope)

    if yarn_args is None:
        if llama4_scaling is not None:
            raise ValueError(
                "`llama_4_scaling` is only meaningful as a modifier on top of yarn's rope "
                "extension; it is not supported without `yarn` also being set."
            )
        return RopeParameters(rope_type="default", rope_theta=rope_theta, **rope_kwargs)
    elif llama4_scaling is not None:
        if yarn_args.original_max_position_embeddings != llama4_scaling.original_max_position_embeddings:
            raise ValueError(
                f"yarn.original_max_position_embeddings ({yarn_args.original_max_position_embeddings}) "
                f"must match llama_4_scaling.original_max_position_embeddings "
                f"({llama4_scaling.original_max_position_embeddings})"
            )
        rope_kwargs["llama_4_scaling_beta"] = llama4_scaling.beta
    else:
        # Sentinel for "no llama4 scaling": beta=0 is a no-op in the attention scale formula
        # (see `get_llama_4_attn_scale` in modeling_mistral4.py), so it round-trips back to
        # absent in `_extract_llama4_scaling_from_rope_params` below.
        rope_kwargs["llama_4_scaling_beta"] = 0

    mscale = 1.0
    mscale_all_dim = 0.0 if yarn_args.apply_scale else 1.0

    return RopeParameters(
        rope_type="yarn",
        rope_theta=rope_theta,
        factor=float(yarn_args.factor),
        original_max_position_embeddings=yarn_args.original_max_position_embeddings,
        beta_fast=float(yarn_args.beta),
        beta_slow=float(yarn_args.alpha),
        mscale=mscale,
        mscale_all_dim=mscale_all_dim,
        **rope_kwargs,
    )


def _hf_config_base_kwargs(
    native_config: MistralNativeConfig,
    rope_parameters: RopeParameters,
    include_head_dim: bool = True,
) -> dict[str, Any]:
    """Build the base HF config kwargs shared by all forward converters.

    Args:
        native_config (`MistralNativeConfig`):
            The native config to read common fields from.
        rope_parameters (`RopeParameters`):
            The already-resolved `RopeParameters` for this config.
        include_head_dim (`bool`, *optional*, defaults to `True`):
            If `False`, omits `head_dim` (e.g. for MLA architectures that derive it from
            `qk_rope_head_dim` / `qk_nope_head_dim` instead).

    Returns:
        A kwargs dict ready to be passed to an HF config constructor.
    """
    kwargs: dict[str, Any] = {
        "hidden_size": native_config.dim,
        "num_hidden_layers": native_config.n_layers,
        "intermediate_size": native_config.hidden_dim,
        "num_attention_heads": native_config.n_heads,
        "rms_norm_eps": native_config.norm_eps,
        "vocab_size": native_config.vocab_size,
        "num_key_value_heads": native_config.n_kv_heads,
        "sliding_window": native_config.sliding_window,
        "max_position_embeddings": native_config.max_position_embeddings,
        "tie_word_embeddings": native_config.tied_embeddings,
        "rope_parameters": rope_parameters,
    }
    if include_head_dim:
        kwargs["head_dim"] = native_config.head_dim
    return kwargs


def _resolve_native_quant_config_kwargs(
    native_config: MistralNativeConfig, is_vision_model: bool = False
) -> dict[str, Any]:
    """Resolve the `quantization_config` kwarg (if any) for a forward converter.

    Args:
        native_config (`MistralNativeConfig`):
            The native config to read quantization fields from.
        is_vision_model (`bool`, *optional*, defaults to `False`):
            If `True`, adds vision/projector modules to `modules_to_not_convert` when a
            quantization config is built.

    Returns:
        An empty dict, or a dict with a single `quantization_config` key.
    """
    quant_config = native_config.quantization_config or _get_maybe_quant_config(
        is_vision_model=is_vision_model, quantization_args=native_config.quantization
    )
    optional_kwargs: dict[str, Any] = {}
    if quant_config is not None:
        optional_kwargs["quantization_config"] = quant_config
    return optional_kwargs


def _native_config_to_text_config(
    native_config: MistralNativeConfig, config_cls: type[MistralConfig] | type[Ministral3Config]
) -> MistralConfig | Ministral3Config:
    """Convert a native config to a `MistralConfig` or `Ministral3Config`.

    Args:
        native_config (`MistralNativeConfig`):
            The native config to convert.
        config_cls (`type[MistralConfig] | type[Ministral3Config]`):
            `MistralConfig` for a config without yarn, `Ministral3Config` for one with yarn.
            The dispatcher (`_dispatch_mistral_native_config_to_hf_config`) has already
            determined which applies.

    Returns:
        The converted HF config.
    """
    rope_parameters = _get_rope_parameters(
        rope_theta=native_config.rope_theta,
        yarn_args=native_config.yarn,
        llama4_scaling=native_config.llama_4_scaling,
        qk_rope=None,
        qk_nope=None,
    )
    base_kwargs = _hf_config_base_kwargs(native_config=native_config, rope_parameters=rope_parameters)
    optional_kwargs = _resolve_native_quant_config_kwargs(native_config)

    return config_cls(**base_kwargs, **optional_kwargs)


def _native_config_to_mistral4(native_config: MistralNativeConfig) -> Mistral4Config:
    """Convert a native config to a `Mistral4Config` (MLA + MoE architecture).

    Args:
        native_config (`MistralNativeConfig`):
            The native config to convert. Must have both `moe` and `q_lora_rank` set.

    Returns:
        The converted `Mistral4Config`.

    Raises:
        ValueError: If `head_dim` is set to a value other than
            `qk_nope_head_dim + qk_rope_head_dim`. The HF Mistral4
            architecture always derives `head_dim` from those two fields
            (see the reverse converter), so any other value cannot be
            represented and would otherwise be silently discarded.
    """
    if native_config.qk_nope_head_dim is not None and native_config.qk_rope_head_dim is not None:
        expected_head_dim = native_config.qk_nope_head_dim + native_config.qk_rope_head_dim
        if native_config.head_dim != expected_head_dim:
            raise ValueError(
                f"Unsupported head_dim={native_config.head_dim!r} for a Mistral4 (MLA) config; "
                f"the only supported value is qk_nope_head_dim + qk_rope_head_dim = {expected_head_dim}."
            )
    rope_parameters = _get_rope_parameters(
        rope_theta=native_config.rope_theta,
        yarn_args=native_config.yarn,
        llama4_scaling=native_config.llama_4_scaling,
        qk_rope=native_config.qk_rope_head_dim,
        qk_nope=native_config.qk_nope_head_dim,
    )
    base_kwargs = _hf_config_base_kwargs(
        native_config=native_config,
        rope_parameters=rope_parameters,
        include_head_dim=False,
    )
    optional_kwargs = _resolve_native_quant_config_kwargs(native_config)

    return Mistral4Config(
        **base_kwargs,
        q_lora_rank=native_config.q_lora_rank,
        qk_rope_head_dim=native_config.qk_rope_head_dim,
        qk_nope_head_dim=native_config.qk_nope_head_dim,
        kv_lora_rank=native_config.kv_lora_rank,
        v_head_dim=native_config.v_head_dim,
        n_routed_experts=native_config.moe.num_experts,
        num_experts_per_tok=native_config.moe.num_experts_per_tok,
        first_k_dense_replace=native_config.moe.first_k_dense_replace,
        n_shared_experts=native_config.moe.num_shared_experts,
        moe_intermediate_size=native_config.moe.expert_hidden_dim,
        routed_scaling_factor=native_config.moe.routed_scale,
        n_group=native_config.moe.num_expert_groups,
        topk_group=native_config.moe.num_expert_groups_per_tok,
        norm_topk_prob=True,
        **optional_kwargs,
    )


def _native_config_to_mistral3(
    native_config: MistralNativeConfig, vision_encoder: VisionEncoderArgs
) -> Mistral3Config:
    """Convert a native config to a `Mistral3Config` (vision-language model).

    Args:
        native_config (`MistralNativeConfig`):
            The native config to convert.
        vision_encoder (`VisionEncoderArgs`):
            `native_config.vision_encoder`, non-`None`. The dispatcher
            (`_dispatch_mistral_native_config_to_hf_config`) has already established this and
            passes it separately so its non-optional type is visible here.

    Returns:
        The converted `Mistral3Config`.

    Raises:
        ValueError: If `mm_projector_id` or `add_pre_mm_projector_layer_norm` are set
            to a value other than what `Mistral3MultiModalProjector`
            unconditionally implements.
    """
    if vision_encoder.mm_projector_id != "patch_merge":
        raise ValueError(
            f"Unsupported vision_encoder.mm_projector_id={vision_encoder.mm_projector_id!r}; "
            "Mistral3MultiModalProjector unconditionally builds a patch-merge projector, so "
            "'patch_merge' is the only supported value."
        )
    if not vision_encoder.add_pre_mm_projector_layer_norm:
        raise ValueError(
            f"Unsupported vision_encoder.add_pre_mm_projector_layer_norm={vision_encoder.add_pre_mm_projector_layer_norm!r}; "
            "Mistral3MultiModalProjector unconditionally builds a pre-projector norm layer, so "
            "`True` is the only supported value."
        )
    vision_hf = PixtralVisionConfig(
        hidden_size=vision_encoder.hidden_size,
        num_hidden_layers=vision_encoder.num_hidden_layers,
        num_attention_heads=vision_encoder.num_attention_heads,
        patch_size=vision_encoder.patch_size,
        image_size=vision_encoder.image_size,
        intermediate_size=vision_encoder.intermediate_size,
        num_channels=vision_encoder.num_channels,
        hidden_act="silu",
        rope_theta=vision_encoder.rope_theta,
    )
    optional_kwargs = _resolve_native_quant_config_kwargs(native_config, is_vision_model=True)

    native_text_config = replace(native_config, vision_encoder=None, quantization=None, quantization_config=None)

    # Use the raw dispatcher (not the `mistral_native_config_to_hf_config` wrapper) so the nested text
    # config does not get its own `mistral_extras`: the outer wrapper call attaches a
    # single combined entry (moe + vision_encoder) to this function's returned `Mistral3Config`.
    text_hf = _dispatch_mistral_native_config_to_hf_config(native_config=native_text_config)

    return Mistral3Config(
        vision_config=vision_hf,
        text_config=text_hf,
        multimodal_projector_bias=vision_encoder.adapter_bias,
        image_token_id=vision_encoder.image_token_id,
        spatial_merge_size=vision_encoder.spatial_merge_size,
        vision_feature_layer=-1,
        tie_word_embeddings=native_config.tied_embeddings,
        **optional_kwargs,
    )


def _extract_hf_quantization_config(
    hf_config: PreTrainedConfig,
) -> QuantizationConfigMixin | None:
    """Extract the `quantization_config` attribute from an HF config, if present.

    Args:
        hf_config (`PreTrainedConfig`):
            The HF config to read `quantization_config` from.

    Returns:
        The quantization config, or `None` if `hf_config` has none.

    Raises:
        TypeError: If `quantization_config` is set to a type other than
            `QuantizationConfigMixin`, `dict`, or `None`.
    """
    quant_cfg = getattr(hf_config, "quantization_config", None)
    if quant_cfg is None:
        return None
    if isinstance(quant_cfg, QuantizationConfigMixin):
        return quant_cfg
    if isinstance(quant_cfg, dict):
        return AutoQuantizationConfig.from_dict(quant_cfg)
    raise TypeError(
        f"Unsupported `quantization_config` type {type(quant_cfg).__name__}; expected "
        "`QuantizationConfigMixin`, `dict`, or `None`."
    )


def _hf_quant_config_to_native(quant_cfg: QuantizationConfigMixin | None) -> QuantizationArgs | None:
    """Convert an already-extracted HF quantization config back to native `QuantizationArgs`.

    Args:
        quant_cfg (`QuantizationConfigMixin`):
            The HF quantization config to convert, or `None`.

    Returns:
        The native `QuantizationArgs`, or `None` if `quant_cfg` is `None` or the mapping is
        not reversible (e.g. a non-fp8 quant method or an unsupported activation scheme).
    """
    # TODO: The issue here is that ignored_layers are probably not in Mistral format which would ultimately fail.
    # We need to map the patterns from one format to another.
    if quant_cfg is None:
        return None
    qc = quant_cfg.to_dict()
    if qc.get("quant_method") != "fp8":
        return None
    scheme = _REVERSE_QUANTIZATION_SCHEME_MAP.get(qc.get("activation_scheme", "static"))
    if scheme is None:
        return None
    return QuantizationArgs(qformat_weight=QFormat.FP8_E4M3, qscheme_act=scheme)


def _resolve_hf_quant_for_native(
    hf_config: PreTrainedConfig,
) -> tuple[QuantizationArgs | None, QuantizationConfigMixin | None]:
    """Resolve an HF quantization config to native fields.

    Tries the direct reverse mapping first. If that is not feasible (e.g.
    non-fp8 quant method), the original HF config is preserved as-is in
    `quantization_config` so it is not silently dropped.

    Args:
        hf_config (`PreTrainedConfig`):
            The HF config to read `quantization_config` from.

    Returns:
        A `(quantization, quantization_config)` pair where exactly one
        (or neither) is non-`None`.

    Raises:
        TypeError: If `hf_config.quantization_config` is set to a type other than
            `QuantizationConfigMixin`, `dict`, or `None`, propagated from
            `_extract_hf_quantization_config`.
    """
    hf_quant = _extract_hf_quantization_config(hf_config)
    native_quant = _hf_quant_config_to_native(hf_quant)
    if native_quant is not None:
        return native_quant, None
    return None, hf_quant


def _extract_llama4_scaling_from_rope_params(
    rope_params: dict | RopeParameters | None,
) -> Llama4Scaling | None:
    """Extract `Llama4Scaling` from an HF `rope_parameters` dict, if present.

    Args:
        rope_params (`dict | RopeParameters`):
            The HF `rope_parameters` dict to read `llama_4_scaling_beta` from, or `None`.

    Returns:
        The `Llama4Scaling`, or `None` if `rope_params` carries no (or a no-op) llama4 scaling.

    Raises:
        ValueError: If `llama_4_scaling_beta` is set but `original_max_position_embeddings` is missing.
    """
    if (
        not rope_params
        or not isinstance(rope_params, dict)
        or (beta := rope_params.get("llama_4_scaling_beta")) is None
        # 0 is the sentinel `_get_rope_parameters` writes for "no llama4 scaling"; treat it the
        # same as a missing key rather than a real (no-op) scaling factor.
        or beta == 0
    ):
        return None

    if (original_max_position_embeddings := rope_params.get("original_max_position_embeddings")) is None:
        raise ValueError("original_max_position_embeddings should not be None if llama4 scaling is set.")

    return Llama4Scaling(
        original_max_position_embeddings=int(original_max_position_embeddings),
        beta=float(beta),
    )


def mistral_native_config_from_hf_config(hf_config: MistralHFConfigType) -> MistralNativeConfig:
    """Convert an HF config to a `MistralNativeConfig`.

    The target architecture is determined by the runtime type of `hf_config`.

    Args:
        hf_config (`MistralHFConfigType`):
            The HuggingFace config to convert.

    Returns:
        The converted native config.

    Raises:
        ValueError: If the config type is unsupported.
        TypeError: If `hf_config.quantization_config` is set to a type other than
            `QuantizationConfigMixin`, `dict`, or `None`, propagated from
            `_extract_hf_quantization_config`.
    """
    match hf_config:
        case Mistral3Config():
            return _hf_mistral3_to_native(hf_config)
        case Mistral4Config():
            return _hf_mistral4_to_native(hf_config)
        case Ministral3Config():
            return _hf_ministral3_to_native(hf_config)
        case MistralConfig():
            return _hf_mistral_to_native(hf_config)
        case _:
            raise ValueError(f"Unsupported HF config type: {type(hf_config).__name__}")


def _extract_rope_theta(config: PreTrainedConfig) -> float:
    """Extract `rope_theta` from an HF config, checking `rope_parameters` first."""
    rope_params = getattr(config, "rope_parameters", None)
    if rope_params and isinstance(rope_params, dict) and "rope_theta" in rope_params:
        return float(rope_params["rope_theta"])
    elif hasattr(config, "rope_theta"):
        return float(config.rope_theta)
    raise ValueError("`rope_theta` not found.")


def _extract_yarn(config: PreTrainedConfig) -> YarnArgs | None:
    """Extract YaRN parameters from an HF config's `rope_parameters`."""
    rope_params = getattr(config, "rope_parameters", None)
    if not rope_params or not isinstance(rope_params, dict):
        return None
    rope_type = rope_params.get("rope_type", rope_params.get("type"))
    if rope_type != "yarn":
        return None
    rope_mscale_all_dim = rope_params.get("mscale_all_dim")
    apply_scale = rope_mscale_all_dim is None or rope_mscale_all_dim != 1.0
    return YarnArgs(
        factor=rope_params["factor"],
        original_max_position_embeddings=rope_params["original_max_position_embeddings"],
        beta=float(rope_params["beta_fast"]),
        alpha=float(rope_params["beta_slow"]),
        apply_scale=apply_scale,
    )


def _native_config_base_kwargs(hf_config: PreTrainedConfig, head_dim: int) -> dict[str, Any]:
    """Build the base native config kwargs shared by all reverse converters.

    Args:
        hf_config (`PreTrainedConfig`):
            The HF config to read common fields from.
        head_dim (`int`):
            The resolved `head_dim` value (callers derive this differently depending on
            whether the architecture uses MLA).

    Returns:
        A kwargs dict ready to be passed to `MistralNativeConfig`.
    """
    quantization, quantization_config = _resolve_hf_quant_for_native(hf_config)
    return {
        "dim": hf_config.hidden_size,
        "n_layers": hf_config.num_hidden_layers,
        "head_dim": head_dim,
        "hidden_dim": hf_config.intermediate_size,
        "n_heads": hf_config.num_attention_heads,
        "n_kv_heads": hf_config.num_key_value_heads,
        "rope_theta": _extract_rope_theta(hf_config),
        "norm_eps": hf_config.rms_norm_eps,
        "vocab_size": hf_config.vocab_size,
        "max_position_embeddings": hf_config.max_position_embeddings,
        "sliding_window": getattr(hf_config, "sliding_window", None),
        "tied_embeddings": hf_config.tie_word_embeddings,
        "yarn": _extract_yarn(hf_config),
        "quantization": quantization,
        "quantization_config": quantization_config,
    }


def _hf_mistral_to_native(hf_config: MistralConfig) -> MistralNativeConfig:
    """Convert a `MistralConfig` back to a `MistralNativeConfig`.

    Args:
        hf_config (`MistralConfig`):
            The HF config to convert.

    Returns:
        The converted native config.

    Raises:
        ValueError: If `head_dim` is not set on the HF config.
    """
    if hf_config.head_dim is None:
        raise ValueError("head_dim must be set on the HF config")
    base_kwargs = _native_config_base_kwargs(hf_config=hf_config, head_dim=hf_config.head_dim)

    return MistralNativeConfig(**base_kwargs)


def _hf_ministral3_to_native(hf_config: Ministral3Config) -> MistralNativeConfig:
    """Convert a `Ministral3Config` back to a `MistralNativeConfig`."""
    base_kwargs = _native_config_base_kwargs(hf_config=hf_config, head_dim=hf_config.head_dim)

    return MistralNativeConfig(
        **base_kwargs,
        llama_4_scaling=_extract_llama4_scaling_from_rope_params(
            getattr(hf_config, "rope_parameters", None),
        ),
    )


def _hf_mistral4_to_native(hf_config: Mistral4Config) -> MistralNativeConfig:
    """Convert a `Mistral4Config` (MLA + MoE) back to a `MistralNativeConfig`.

    The MoE parallelism-topology fields with no HF representation
    (`expert_parallel`, `expert_model_parallel`, `route_every_n`) are read
    back from `mistral_extras` when present, falling back to their
    native defaults otherwise (e.g. for a hand-written HF config).

    Args:
        hf_config (`Mistral4Config`):
            The HF config to convert.

    Returns:
        The converted native config.

    Raises:
        ValueError: If any of `num_key_value_heads`, `num_experts_per_tok`,
            `first_k_dense_replace`, `n_group`, or `topk_group` is `None` on `hf_config`.
    """
    rope_params = getattr(hf_config, "rope_parameters", None)
    required_fields = {
        "num_key_value_heads": hf_config.num_key_value_heads,
        "num_experts_per_tok": hf_config.num_experts_per_tok,
        "first_k_dense_replace": hf_config.first_k_dense_replace,
        "n_group": hf_config.n_group,
        "topk_group": hf_config.topk_group,
    }
    missing = [name for name, value in required_fields.items() if value is None]
    if missing:
        raise ValueError(f"Mistral4 config requires non-None fields: {missing}")
    base_kwargs = _native_config_base_kwargs(
        hf_config=hf_config,
        head_dim=hf_config.qk_nope_head_dim + hf_config.qk_rope_head_dim,
    )
    moe_residual = _extract_residual_fields(extras=getattr(hf_config, _MISTRAL_EXTRAS_KEY, None), section_name="moe")

    return MistralNativeConfig(
        **base_kwargs,
        q_lora_rank=hf_config.q_lora_rank,
        qk_rope_head_dim=hf_config.qk_rope_head_dim,
        qk_nope_head_dim=hf_config.qk_nope_head_dim,
        kv_lora_rank=hf_config.kv_lora_rank,
        v_head_dim=hf_config.v_head_dim,
        llama_4_scaling=_extract_llama4_scaling_from_rope_params(rope_params),
        moe=MOEModelArgs(
            num_experts=hf_config.n_routed_experts,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            first_k_dense_replace=hf_config.first_k_dense_replace,
            num_shared_experts=hf_config.n_shared_experts,
            expert_hidden_dim=hf_config.moe_intermediate_size,
            routed_scale=hf_config.routed_scaling_factor,
            num_expert_groups=hf_config.n_group,
            num_expert_groups_per_tok=hf_config.topk_group,
            **moe_residual,
        ),
    )


def _hf_mistral3_to_native(hf_config: Mistral3Config) -> MistralNativeConfig:
    """Convert a `Mistral3Config` (VLM) back to a `MistralNativeConfig`."""
    text_native = mistral_native_config_from_hf_config(hf_config.text_config)
    vision_hf: PixtralVisionConfig = hf_config.vision_config
    extras = getattr(hf_config, _MISTRAL_EXTRAS_KEY, None)

    vision_residual = {
        "max_image_size": vision_hf.image_size,
        **_extract_residual_fields(extras=extras, section_name="vision_encoder"),
    }
    vision_encoder = VisionEncoderArgs(
        hidden_size=vision_hf.hidden_size,
        num_hidden_layers=vision_hf.num_hidden_layers,
        num_attention_heads=vision_hf.num_attention_heads,
        patch_size=vision_hf.patch_size,
        image_size=vision_hf.image_size,
        intermediate_size=vision_hf.intermediate_size,
        num_channels=vision_hf.num_channels,
        rope_theta=_extract_rope_theta(vision_hf),
        adapter_bias=hf_config.multimodal_projector_bias,
        spatial_merge_size=hf_config.spatial_merge_size,
        image_token_id=hf_config.image_token_id,
        mm_projector_id="patch_merge",
        add_pre_mm_projector_layer_norm=True,
        **vision_residual,
    )

    moe = text_native.moe
    moe_residual = _extract_residual_fields(extras=extras, section_name="moe")
    if moe is not None and moe_residual:
        moe = replace(moe, **moe_residual)

    quantization, quantization_config = _resolve_hf_quant_for_native(hf_config)
    return replace(
        text_native,
        vision_encoder=vision_encoder,
        quantization=quantization,
        quantization_config=quantization_config,
        moe=moe,
    )
