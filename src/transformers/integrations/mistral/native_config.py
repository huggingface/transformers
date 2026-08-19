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
"""Mistral native params.json schema and its parsing into MistralNativeConfig."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from ...quantizers.auto import AutoQuantizationConfig
from ...utils.quantization_config import QuantizationConfigMixin


@dataclass
class Llama4Scaling:
    """Llama 4 scaling parameters for extended context RoPE."""

    original_max_position_embeddings: int
    beta: float


@dataclass
class YarnArgs:
    """YaRN (Yet another RoPE extensioN) parameters.

    Note:
        The Mistral native format uses `beta` / `alpha`, which map to HF
        `rope_parameters["beta_fast"]` / `rope_parameters["beta_slow"]` respectively.
    """

    factor: float
    original_max_position_embeddings: int
    beta: float  # corresponds to HF rope_parameters["beta_fast"]
    alpha: float  # corresponds to HF rope_parameters["beta_slow"]
    apply_scale: bool


# Maps a native `qscheme_act` value to the corresponding HF quantization `activation_scheme`.
# The single source of truth for which quantization schemes this module supports.
_QUANTIZATION_SCHEME_MAP: dict[str, str] = {"TENSOR": "static"}


class QFormat(str, Enum):
    """Supported quantization weight formats."""

    FP8_E4M3 = "fp8_e4m3"


@dataclass
class QuantizationArgs:
    """Native Mistral quantization configuration.

    `qformat_weight` accepts a `str` (e.g. `"fp8_e4m3"`) at construction time; `__post_init__`
    coerces it to `QFormat`, so it is a `QFormat` on every instance afterward.
    """

    qformat_weight: QFormat
    qscheme_act: str

    def __post_init__(self) -> None:
        try:
            self.qformat_weight = QFormat(self.qformat_weight)
        except ValueError as e:
            raise ValueError(
                f"Unsupported quantization format {self.qformat_weight!r}; only {[q.value for q in QFormat]} are supported."
            ) from e
        if self.qscheme_act not in _QUANTIZATION_SCHEME_MAP:
            raise ValueError(
                f"Unsupported quantization scheme {self.qscheme_act!r}; "
                f"supported schemes: {sorted(_QUANTIZATION_SCHEME_MAP)}."
            )


@dataclass
class MOEModelArgs:
    """Mixture-of-Experts architecture parameters."""

    first_k_dense_replace: int
    num_experts: int
    num_experts_per_tok: int
    num_expert_groups: int
    num_expert_groups_per_tok: int
    routed_scale: float
    expert_hidden_dim: int
    num_shared_experts: int
    # Runtime topology, unrepresentable in HF configs. These constructor defaults are what
    # applies when `mistral_extras` is absent or omits the field.
    expert_parallel: int = 1
    expert_model_parallel: int = 1
    route_every_n: int = 1


@dataclass
class VisionEncoderArgs:
    """Vision encoder configuration for multimodal Mistral models."""

    image_token_id: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    mm_projector_id: str
    spatial_merge_size: int
    hidden_size: int
    num_channels: int
    image_size: int
    max_image_size: int
    patch_size: int
    rope_theta: float
    add_pre_mm_projector_layer_norm: bool
    adapter_bias: bool
    # Token id defaults match Mistral's tekken.json: [IMG_BREAK] = 12, [IMG_END] = 13.
    # `max_image_size` above has no fixed default; callers derive one from the vision config
    # instead.
    image_break_token_id: int = 12
    image_end_token_id: int = 13


@dataclass
class MistralNativeConfig:
    """Complete native Mistral configuration from params.json."""

    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    rope_theta: float
    norm_eps: float
    vocab_size: int
    max_position_embeddings: int | None = None
    sliding_window: int | None = None
    tied_embeddings: bool = False
    llama_4_scaling: Llama4Scaling | None = None
    q_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    qk_nope_head_dim: int | None = None
    kv_lora_rank: int | None = None
    v_head_dim: int | None = None
    quantization: QuantizationArgs | None = None
    quantization_config: QuantizationConfigMixin | None = None
    yarn: YarnArgs | None = None
    moe: MOEModelArgs | None = None
    vision_encoder: VisionEncoderArgs | None = None

    def __post_init__(self) -> None:
        if self.quantization is not None and self.quantization_config is not None:
            raise ValueError(
                "Cannot set both `quantization` (Mistral format) and `quantization_config` (HF format) "
                "at the same time. Use one or the other."
            )


# Top-level `params.json` keys this parser knows how to interpret when their value is a
# nested dict (a "section" describing a sub-architecture). A dict-valued key outside this set
# describes a sub-architecture this module has no converter for (e.g. `multimodal` for Voxtral's
# audio encoder); see `_check_supported_architecture`.
_KNOWN_SECTION_KEYS: frozenset[str] = frozenset(
    {"yarn", "llama_4_scaling", "quantization", "quantization_config", "moe", "vision_encoder"}
)

_DEFAULT_MAX_POSITION_EMBEDDINGS = 32768


def _check_supported_architecture(params: dict) -> None:
    """Raise if `params` describes a sub-architecture this module has no converter for.

    Args:
        params (`dict`):
            Raw key/value pairs from a Mistral `params.json` file.

    Raises:
        ValueError: If `params` contains a dict-valued key not in `_KNOWN_SECTION_KEYS`.
    """
    unsupported_sections = sorted(
        key for key, value in params.items() if isinstance(value, dict) and key not in _KNOWN_SECTION_KEYS
    )
    if unsupported_sections:
        raise ValueError(
            f"params.json contains section(s) {unsupported_sections} describing an architecture that "
            "`transformers.integrations.mistral` does not support converting."
        )


def _resolve_sliding_window(sliding_window: Any) -> int | None:
    """Coerce a raw `params.json` `sliding_window` value to `int | None`.

    Args:
        sliding_window (`Any`):
            The raw `params.json` `sliding_window` value.

    Returns:
        The coerced `int`, or `None` if `sliding_window` is `None`.

    Raises:
        TypeError: If `sliding_window` is a list (interleaved per-layer pattern).
    """
    if isinstance(sliding_window, list):
        raise TypeError(
            f"params.json `sliding_window={sliding_window!r}` is a per-layer interleaved "
            "pattern, which is not supported."
        )
    if sliding_window is None:
        return None
    return int(sliding_window)


_SectionT = TypeVar("_SectionT")


def _section_from_dict(section_name: str, section_dict: dict[str, Any], section_type: type[_SectionT]) -> _SectionT:
    """Build a section dataclass from its raw `params.json` sub-dict.

    Field names match `params.json` keys one-for-one, so the dict is splatted directly. An
    undeclared key is an error rather than dropped: it means `params.json` grew a field this
    converter does not model yet.

    Args:
        section_name (`str`):
            The `params.json` key this section came from, used only for the error message.
        section_dict (`dict`):
            The raw `params.json` sub-dict for this section.
        section_type (`type`):
            The dataclass to build, e.g. `YarnArgs`.

    Returns:
        The built section dataclass instance.

    Raises:
        ValueError: If `section_dict` omits a field `section_type` cannot default, or carries
            a key `section_type` does not declare.
    """
    try:
        return section_type(**section_dict)
    except TypeError as e:
        raise ValueError(f"params.json '{section_name}' section is invalid: {e}") from e


def mistral_native_config_from_params(params: dict[str, Any]) -> MistralNativeConfig:
    """Build a `MistralNativeConfig` from a raw `params.json` dict.

    Args:
        params (`dict`):
            Raw key/value pairs from a Mistral `params.json` file.

    Raises:
        KeyError: If a required top-level field is missing from `params`.
        ValueError: If a section is missing a required field or carries an unknown one.

    Returns:
        The parsed native config.
    """
    _check_supported_architecture(params)

    sections: dict[str, Any] = {
        name: _section_from_dict(name, section_dict, section_type)
        for name, section_type in (
            ("yarn", YarnArgs),
            ("llama_4_scaling", Llama4Scaling),
            ("quantization", QuantizationArgs),
            ("moe", MOEModelArgs),
            ("vision_encoder", VisionEncoderArgs),
        )
        if (section_dict := params.get(name)) is not None
    }

    quantization_config = params.get("quantization_config")
    if quantization_config is not None:
        quantization_config = AutoQuantizationConfig.from_dict(quantization_config)

    return MistralNativeConfig(
        dim=params["dim"],
        n_layers=params["n_layers"],
        head_dim=params["head_dim"],
        hidden_dim=params["hidden_dim"],
        n_heads=params["n_heads"],
        n_kv_heads=params["n_kv_heads"],
        rope_theta=params["rope_theta"],
        norm_eps=params["norm_eps"],
        vocab_size=params["vocab_size"],
        max_position_embeddings=params.get("max_position_embeddings", _DEFAULT_MAX_POSITION_EMBEDDINGS),
        sliding_window=_resolve_sliding_window(params.get("sliding_window")),
        tied_embeddings=params.get("tied_embeddings", False),
        q_lora_rank=params.get("q_lora_rank"),
        qk_rope_head_dim=params.get("qk_rope_head_dim"),
        qk_nope_head_dim=params.get("qk_nope_head_dim"),
        kv_lora_rank=params.get("kv_lora_rank"),
        v_head_dim=params.get("v_head_dim"),
        quantization_config=quantization_config,
        **sections,
    )
