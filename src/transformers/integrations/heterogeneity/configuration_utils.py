# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from transformers.utils import logging


if TYPE_CHECKING:
    from transformers import PreTrainedConfig


logger = logging.get_logger(__name__)

_SENTINEL = object()


class AmbiguousGlobalPerLayerAttributeError(RuntimeError):
    """Raised when a per-layer attribute is read from a heterogeneous global config."""


@dataclass
class _HeterogeneitySpec:
    per_layer_overrides: dict[int, dict[str, Any]]
    per_layer_attributes: set[str]
    explicit_per_layer_attributes: set[str]


def _normalize_layer_overrides(layer_overrides: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(layer_overrides)

    if "skip" in normalized:
        skip = normalized.pop("skip")
        if isinstance(skip, str) or not isinstance(skip, Iterable):
            raise TypeError("`skip` must be an iterable of strings.")

        skip = set(skip)
        if not all(isinstance(item, str) for item in skip):
            raise TypeError("`skip` must contain only strings.")

        if skip:
            normalized["skip"] = sorted(skip)

    return normalized


def _validate_layer_indices(config: PreTrainedConfig, per_layer_overrides: dict[int, dict[str, Any]]) -> None:
    if not per_layer_overrides:
        return

    num_hidden_layers = config.num_hidden_layers
    invalid_layer_indices = [
        layer_idx for layer_idx in per_layer_overrides if layer_idx < 0 or layer_idx >= num_hidden_layers
    ]
    if invalid_layer_indices:
        raise ValueError(
            f"`per_layer_config` keys must be integer layer indices in the range [0, {num_hidden_layers}); "
            f"got {invalid_layer_indices}."
        )


def _validate_sliding_window_and_attention_chunk_size(
    config: PreTrainedConfig, per_layer_overrides: dict[int, dict[str, Any]]
) -> None:
    problematic_indices = []
    for layer_idx in range(config.num_hidden_layers):
        layer_overrides = per_layer_overrides.get(layer_idx, {})

        sliding_window = layer_overrides.get(
            "sliding_window", config._getattr_without_heterogeneous_validation("sliding_window", None)
        )
        attention_chunk_size = layer_overrides.get(
            "attention_chunk_size",
            config._getattr_without_heterogeneous_validation("attention_chunk_size", None),
        )

        if sliding_window is not None and attention_chunk_size is not None:
            problematic_indices.append(layer_idx)

    if problematic_indices:
        raise ValueError(
            f"The following layers have the mutually exclusive `sliding_window` and `attention_chunk_size` both defined: "
            f"{problematic_indices}. To fix this, either remove a conflicting attribute from the global config,"
            f"or set it to `None` in `per_layer_config` for the problematic layers."
        )


def _get_per_layer_attributes(per_layer_overrides: dict[int, dict[str, Any]]) -> set[str]:
    per_layer_attributes: set[str] = set()
    for layer_overrides in per_layer_overrides.values():
        per_layer_attributes.update(layer_overrides)

    per_layer_attributes.discard("skip")
    return per_layer_attributes


def _modify_config_and_create_heterogeneity_spec(
    config: PreTrainedConfig, per_layer_overrides: dict[int, dict[str, Any]]
) -> _HeterogeneitySpec:
    explicit_per_layer_attributes = _get_per_layer_attributes(per_layer_overrides)

    # Ensure all required global attributes are defined
    missing_required_global_attributes = set()
    for attr in explicit_per_layer_attributes:
        if len(per_layer_overrides) != config.num_hidden_layers:
            if not config._hasattr_without_heterogeneous_validation(attr):
                missing_required_global_attributes.add(attr)
        else:
            for layer_overrides in per_layer_overrides.values():
                if attr not in layer_overrides:
                    if not config._hasattr_without_heterogeneous_validation(attr):
                        missing_required_global_attributes.add(attr)
                    break

    if missing_required_global_attributes:
        raise ValueError(
            f"The following attributes are missing: {sorted(missing_required_global_attributes)}\nPlease define them globally, or provide them for every layer in `per_layer_config`"
        )

    # Remove per-layer overrides that match the global value
    for attr in explicit_per_layer_attributes:
        if not config._hasattr_without_heterogeneous_validation(attr):
            continue

        global_value = config._getattr_without_heterogeneous_validation(attr)
        for layer_overrides in per_layer_overrides.values():
            if attr in layer_overrides and layer_overrides[attr] == global_value:
                del layer_overrides[attr]

    # Delete all empty layer configs
    for layer_idx, layer_overrides in list(per_layer_overrides.items()):
        if not layer_overrides:
            del per_layer_overrides[layer_idx]

    per_layer_attributes = _get_per_layer_attributes(per_layer_overrides)
    heterogeneity_spec = _HeterogeneitySpec(
        per_layer_overrides=per_layer_overrides,
        per_layer_attributes=per_layer_attributes,
        explicit_per_layer_attributes=explicit_per_layer_attributes,
    )

    return heterogeneity_spec


def _apply_heterogeneous_config(
    config: PreTrainedConfig,
    per_layer_config: dict[int | str, dict[str, Any]],
) -> None:
    """Register per-layer configuration overrides on a model config.

    In a heterogeneous model, individual layers can differ from the global config
    (e.g., different ``intermediate_size``, ``num_key_value_heads``, or entire
    sub-layers skipped via the ``skip`` attribute).

    This function validates the overrides and stores a ``_HeterogeneitySpec`` on ``config._heterogeneity_spec``.
    At model-init time, ``apply_heterogeneous_modeling`` reads this spec to patch
    each layer with its resolved config.

    Args:
        config: The global model config to modify in-place.
        per_layer_config: Mapping from layer index to a dictionary
            of attribute overrides. Only layers that differ from the global
            config need to be included.
    """

    normalized_per_layer_overrides = {
        int(layer_idx): _normalize_layer_overrides(layer_overrides)
        for layer_idx, layer_overrides in per_layer_config.items()
    }

    _validate_layer_indices(config, normalized_per_layer_overrides)
    _validate_sliding_window_and_attention_chunk_size(config, normalized_per_layer_overrides)

    config._heterogeneity_spec = _modify_config_and_create_heterogeneity_spec(config, normalized_per_layer_overrides)


def _get_layer_config(
    config: PreTrainedConfig,
    layer_overrides: dict[str, Any],
) -> PreTrainedConfig:
    output_config = copy.copy(config)
    output_config.__dict__.pop("_heterogeneity_spec", None)

    output_config.skip = layer_overrides.get("skip", [])

    for attr, value in layer_overrides.items():
        if attr == "skip":
            continue
        setattr(output_config, attr, value)

    return output_config


class _PerLayerConfigView(Sequence["PreTrainedConfig"]):
    def __init__(self, config: PreTrainedConfig) -> None:
        self._config = config

    def __len__(self) -> int:
        return self._config.num_hidden_layers

    def __getitem__(self, layer_idx: int | slice) -> PreTrainedConfig | list[PreTrainedConfig]:
        if isinstance(layer_idx, slice):
            return [self[i] for i in range(*layer_idx.indices(len(self)))]

        if layer_idx < 0:
            layer_idx += len(self)
        if layer_idx < 0 or layer_idx >= len(self):
            raise IndexError("list index out of range")

        heterogeneity_spec = self._config._heterogeneity_spec
        return _get_layer_config(
            self._config,
            heterogeneity_spec.per_layer_overrides.get(layer_idx, {}),
        )


def _get_explicit_per_layer_overrides(config: PreTrainedConfig) -> dict[int, dict[str, Any]]:
    heterogeneity_spec = config._heterogeneity_spec
    explicit_per_layer_overrides = {}

    for layer_idx in range(config.num_hidden_layers):
        layer_overrides = copy.deepcopy(heterogeneity_spec.per_layer_overrides.get(layer_idx, {}))

        for attr in heterogeneity_spec.explicit_per_layer_attributes:
            if attr not in layer_overrides:
                layer_overrides[attr] = config._getattr_without_heterogeneous_validation(attr)

        if layer_overrides:
            explicit_per_layer_overrides[layer_idx] = layer_overrides

    return explicit_per_layer_overrides


class HeterogeneousConfigMixin:
    """Mixin for heterogeneous per-layer config behavior.

    This mixin owns heterogeneity-specific state and rules. ``PreTrainedConfig`` assigns the ``per_layer_config``
    property in the post-init phase and calls hook methods where heterogeneity needs to participate in the config lifecycle: attribute
    access, key iteration, and serialization.
    """

    def __getattribute__(self, key: str) -> Any:
        # In heterogeneous configs, per-layer attributes are ambiguous on the global config.
        # Callers must read them from a concrete layer unless they explicitly opt into the global value.
        heterogeneity_spec = super().__getattribute__("__dict__").get("_heterogeneity_spec")
        if heterogeneity_spec is not None:
            if key in heterogeneity_spec.per_layer_attributes:
                if not super().__getattribute__("allow_global_per_layer_attribute_access"):
                    raise AmbiguousGlobalPerLayerAttributeError(
                        f"'{key}' is a per-layer attribute and may vary across layers. Access it via the individual layer "
                        f"configs instead (e.g. config.per_layer_config[i].{key}). To read the global config value from "
                        f"config.{key} anyway, set `allow_global_per_layer_attribute_access` to `True` on the config. "
                        f"Warning: only do this if the caller can safely handle heterogeneous configs; code that assumes "
                        f"a homogeneous model may use the global value incorrectly."
                    )

                logger.warning_once(
                    f"Reading global config value for per-layer attribute `{key}` on a heterogeneous config. "
                    "Only do this if the caller can safely handle heterogeneous configs; code that assumes a homogeneous "
                    "model may use the global value incorrectly."
                )

        return super().__getattribute__(key)

    @property
    def is_heterogeneous(self) -> bool:
        return hasattr(self, "_heterogeneity_spec")

    @property
    def per_layer_config(self) -> Sequence[PreTrainedConfig] | None:
        if not self.is_heterogeneous:
            return None
        return _PerLayerConfigView(self)

    @per_layer_config.setter
    def per_layer_config(self, per_layer_config: dict[int | str, dict[str, Any]] | None) -> None:
        if per_layer_config is None:
            self.__dict__.pop("_heterogeneity_spec", None)
            return

        _apply_heterogeneous_config(self, per_layer_config)

    @property
    def per_layer_attributes(self) -> set[str] | None:
        if not self.is_heterogeneous:
            return None
        return self._heterogeneity_spec.per_layer_attributes

    @property
    def allow_global_per_layer_attribute_access(self) -> bool:
        return self.__dict__.get("allow_global_per_layer_attribute_access", False)

    @allow_global_per_layer_attribute_access.setter
    def allow_global_per_layer_attribute_access(self, value: bool) -> None:
        self.__dict__["allow_global_per_layer_attribute_access"] = value

    @property
    def serialize_explicit_per_layer_config(self) -> bool:
        return self.__dict__.get("serialize_explicit_per_layer_config", False)

    @serialize_explicit_per_layer_config.setter
    def serialize_explicit_per_layer_config(self, value: bool) -> None:
        self.__dict__["serialize_explicit_per_layer_config"] = value

    def _iter_config_keys_with_heterogeneous_adjustment(self, keys: Iterable[str]) -> Iterable[str]:
        # Per-layer attributes intentionally raise on direct access and should not be exposed by iteration,
        # unless `allow_global_per_layer_attribute_access` is True.
        if self.is_heterogeneous and not self.allow_global_per_layer_attribute_access:
            for key in keys:
                if key not in self.per_layer_attributes:
                    yield key
        else:
            yield from keys

    def _update_heterogeneous_to_dict_output(self, d: dict[str, Any]) -> None:
        if not self.is_heterogeneous:
            return

        if self.serialize_explicit_per_layer_config:
            per_layer_overrides = _get_explicit_per_layer_overrides(self)
        else:
            per_layer_overrides = self._heterogeneity_spec.per_layer_overrides

        if per_layer_overrides:
            # Zero-pad so keys sort numerically in JSON (0,1,...,10 not 0,1,10,2,...)
            max_digits = len(str(max(per_layer_overrides.keys())))
            d["per_layer_config"] = {
                str(layer_idx).zfill(max_digits): copy.deepcopy(layer_overrides)
                for layer_idx, layer_overrides in per_layer_overrides.items()
            }
        else:
            d["per_layer_config"] = {}

        d.pop("_heterogeneity_spec", None)

    def _getattr_without_heterogeneous_validation(self, key: str, default: Any = _SENTINEL) -> Any:
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]

        try:
            return super().__getattribute__(key)
        except AttributeError:
            if default is _SENTINEL:
                raise
            return default

    def _hasattr_without_heterogeneous_validation(self, key: str) -> bool:
        try:
            self._getattr_without_heterogeneous_validation(key)
        except AttributeError:
            return False
        return True
