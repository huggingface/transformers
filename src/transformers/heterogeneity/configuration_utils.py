from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from transformers import PreTrainedConfig


@dataclass
class HeterogeneitySpec:
    per_layer_overrides: dict[int, dict[str, Any]]
    per_layer_attributes: set[str]
    fallback_values: dict[str, Any]


def apply_heterogeneous_config(
    config: PreTrainedConfig,
    per_layer_config: dict[int | str, dict[str, Any]],
    explicit: bool = False,
) -> None:
    """Register per-layer configuration overrides on a model config.

    In a heterogeneous model, individual layers can differ from the global config
    (e.g., different ``intermediate_size``, ``num_key_value_heads``, or entire
    sub-layers skipped via the ``skip`` attribute).

    This function validates the overrides, computes fallback values from the global
    config, and stores a ``HeterogeneitySpec`` on ``config._heterogeneity_spec``.
    At model-init time, ``apply_heterogeneous_modeling`` reads this spec to patch
    each layer with its resolved config.

    Args:
        config: The global model config to modify in-place.
        per_layer_config: Mapping from layer index to a dictionary
            of attribute overrides. Only layers that differ from the global
            config need to be included.
        explicit: Whether to enforce that `per_layer_config` exists for each layer
            and that each layer has all per-layer attributes defined.
    """

    normalized_per_layer_overrides = _normalize_per_layer_overrides(per_layer_config)

    _validate_layer_indices(config, normalized_per_layer_overrides)
    _validate_sliding_window_and_attention_chunk_size(config, normalized_per_layer_overrides)

    config._heterogeneity_spec = _modify_config_and_create_heterogeneity_spec(
        config, normalized_per_layer_overrides, explicit=explicit
    )


def heterogeneous_to_dict_helper(config: PreTrainedConfig, d: dict[str, Any]) -> None:
    per_layer_overrides = config._heterogeneity_spec.per_layer_overrides
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


def get_per_layer_config(config: PreTrainedConfig) -> Sequence[PreTrainedConfig]:
    return _PerLayerConfigView(config)


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
            heterogeneity_spec.per_layer_overrides.get(layer_idx),
            heterogeneity_spec.per_layer_attributes,
            heterogeneity_spec.fallback_values,
        )


def _get_layer_config(
    config: PreTrainedConfig,
    layer_overrides: dict[str, Any] | None,
    per_layer_attributes: set[str],
    fallback_values: dict[str, Any],
) -> PreTrainedConfig:
    output_config = copy.copy(config)

    if hasattr(output_config, "_heterogeneity_spec"):
        del output_config._heterogeneity_spec

    output_config.skip = layer_overrides.get("skip", []) if layer_overrides is not None else []

    for attr in per_layer_attributes:
        value = fallback_values[attr]
        if layer_overrides is not None and attr in layer_overrides:
            value = layer_overrides[attr]
        setattr(output_config, attr, value)

    return output_config


def _normalize_per_layer_overrides(
    per_layer_overrides: dict[int | str, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    return {
        int(layer_idx): _normalize_layer_overrides(layer_overrides)
        for layer_idx, layer_overrides in per_layer_overrides.items()
    }


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
        layer_idx
        for layer_idx in per_layer_overrides
        if layer_idx < 0 or layer_idx >= num_hidden_layers
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

        sliding_window = layer_overrides.get("sliding_window", getattr(config, "sliding_window", None))
        attention_chunk_size = layer_overrides.get(
            "attention_chunk_size", getattr(config, "attention_chunk_size", None)
        )

        if sliding_window is not None and attention_chunk_size is not None:
            problematic_indices.append(layer_idx)

    if problematic_indices:
        raise ValueError(
            f"The following layers have the mutually exclusive `sliding_window` and `attention_chunk_size` both defined: "
            f"{problematic_indices}. To fix this, either remove a conflicting attribute from the global config,"
            f"or set it to `None` in `per_layer_config` for the problematic layers."
        )


def _modify_config_and_create_heterogeneity_spec(
    config: PreTrainedConfig, per_layer_overrides: dict[int, dict[str, Any]], explicit: bool
) -> HeterogeneitySpec:
    per_layer_attributes = _get_per_layer_attributes(per_layer_overrides)

    # Ensure all required global attributes are defined
    missing_required_global_attributes = set()
    for attr in per_layer_attributes:
        if len(per_layer_overrides) != config.num_hidden_layers:
            if not hasattr(config, attr):
                missing_required_global_attributes.add(attr)
        else:
            for layer_overrides in per_layer_overrides.values():
                if attr not in layer_overrides:
                    if not hasattr(config, attr):
                        missing_required_global_attributes.add(attr)
                    break

    if missing_required_global_attributes:
        raise ValueError(
            f"The following attributes are missing: {sorted(missing_required_global_attributes)}\nPlease define them globally, or provide them for every layer in `per_layer_config`"
        )

    for attr in per_layer_attributes:
        # Gather all values for this attribute across all layers,
        # and if `explicit` is True, enforce that `per_layer_overrides` has overrides for each layer
        # and that each layer has all per-layer attributes defined.
        values_list = []
        for layer_idx in range(config.num_hidden_layers):
            layer_overrides = per_layer_overrides.get(layer_idx)

            if explicit:
                if layer_overrides is None:
                    layer_overrides = {}
                    per_layer_overrides[layer_idx] = layer_overrides

                if attr not in layer_overrides:
                    layer_overrides[attr] = getattr(config, attr)

            value = (
                layer_overrides[attr]
                if layer_overrides is not None and attr in layer_overrides
                else getattr(config, attr)
            )
            if value not in values_list:
                values_list.append(value)

        if not explicit and len(values_list) == 1:
            # All layer configs have the same value for this attribute, so it can be a global attribute
            setattr(config, attr, values_list[0])
            for layer_idx, layer_overrides in per_layer_overrides.items():
                layer_overrides.pop(attr, None)

    # Delete all empty layer configs
    for layer_idx, layer_overrides in list(per_layer_overrides.items()):
        if not layer_overrides:
            del per_layer_overrides[layer_idx]

    per_layer_attributes = _get_per_layer_attributes(per_layer_overrides)
    fallback_values = {attr: getattr(config, attr, None) for attr in per_layer_attributes}

    heterogeneity_spec = HeterogeneitySpec(
        per_layer_overrides=per_layer_overrides,
        per_layer_attributes=per_layer_attributes,
        fallback_values=fallback_values,
    )

    return heterogeneity_spec


def _get_per_layer_attributes(per_layer_overrides: dict[int, dict[str, Any]]) -> set[str]:
    return {attr for layer_overrides in per_layer_overrides.values() for attr in layer_overrides if attr != "skip"}
