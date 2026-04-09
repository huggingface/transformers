from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from transformers import PreTrainedConfig


class LayerConfig(SimpleNamespace):
    @property
    def attributes(self) -> set[str]:
        return set(vars(self).keys())

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


@dataclass
class HeterogeneitySpec:
    per_layer_config: dict[int, dict[str, Any] | LayerConfig]
    per_layer_attributes: set[str]
    fallback_values: dict[str, Any]


def apply_heterogeneous_config(
    config: PreTrainedConfig, per_layer_config: dict[int, dict[str, Any] | LayerConfig], explicit: bool = False
) -> None:
    """Register per-layer configuration overrides on a model config.

    In a heterogeneous model, individual layers can differ from the global config
    (e.g., different ``intermediate_size``, ``num_key_value_heads``, or entire
    sub-layers skipped via ``skip_*`` attributes).

    This function validates the overrides, computes fallback values from the global
    config, and stores a ``HeterogeneitySpec`` on ``config._heterogeneity_spec``.
    At model-init time, ``apply_heterogeneous_modeling`` reads this spec to patch
    each layer with its resolved config.

    Args:
        config: The global model config to modify in-place.
        per_layer_config: Mapping from layer index to a dict or ``LayerConfig``
            of attribute overrides. Only layers that differ from the global
            config need to be included.
        explicit: Whether to enforce that `per_layer_config` has a LayerConfig for each layer
            and that each layer has all per-layer attributes defined.
    """

    per_layer_config = {
        layer_idx: LayerConfig(**layer_config) if isinstance(layer_config, dict) else layer_config
        for layer_idx, layer_config in per_layer_config.items()
    }

    _validate_num_hetero_layers(config, per_layer_config)
    _validate_sliding_window_and_attention_chunk_size(config, per_layer_config)

    config._heterogeneity_spec = _modify_config_and_create_heterogeneity_spec(
        config, per_layer_config, explicit=explicit
    )


def heterogeneous_to_dict_helper(config: PreTrainedConfig, d: dict[str, Any]) -> None:
    if config.per_layer_config:
        # Zero-pad so keys sort numerically in JSON (0,1,...,10 not 0,1,10,2,...)
        max_digits = len(str(max(config.per_layer_config.keys())))
        d["per_layer_config"] = {
            str(layer_idx).zfill(max_digits): layer_config.to_dict()
            for layer_idx, layer_config in config.per_layer_config.items()
        }
    else:
        d["per_layer_config"] = {}

    d.pop("_heterogeneity_spec", None)


def get_full_layer_config(config: PreTrainedConfig, layer_idx: int) -> PreTrainedConfig:
    output_config = copy.copy(config)
    del output_config._heterogeneity_spec

    layer_config = config.per_layer_config.get(layer_idx, None)

    if layer_config is not None:
        for attr in layer_config.attributes:
            if attr.startswith("skip_"):
                setattr(output_config, attr, getattr(layer_config, attr))

    for attr in config.per_layer_attributes:
        value = config._heterogeneity_spec.fallback_values[attr]
        if layer_config is not None:
            value = getattr(layer_config, attr, value)
        setattr(output_config, attr, value)

    return output_config


def _validate_num_hetero_layers(config: PreTrainedConfig, per_layer_config: dict[int, LayerConfig]) -> None:
    if not per_layer_config:
        return

    num_hidden_layers = config.num_hidden_layers
    max_layer_idx = max(per_layer_config.keys())
    if max_layer_idx >= num_hidden_layers:
        raise ValueError(
            f"The number of hidden layers ({num_hidden_layers}) does not match the indices of `per_layer_config` (the maximal index is {max_layer_idx})"
        )


def _validate_sliding_window_and_attention_chunk_size(
    config: PreTrainedConfig, per_layer_config: dict[int, LayerConfig]
) -> None:
    problematic_indices = []
    for layer_idx in range(config.num_hidden_layers):
        layer_config = per_layer_config.get(layer_idx)
        if layer_config is None:
            layer_config = LayerConfig()

        sliding_window = getattr(layer_config, "sliding_window", getattr(config, "sliding_window", None))
        attention_chunk_size = getattr(
            layer_config, "attention_chunk_size", getattr(config, "attention_chunk_size", None)
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
    config: PreTrainedConfig, per_layer_config: dict[int, LayerConfig], explicit: bool
) -> HeterogeneitySpec:
    per_layer_attributes = _get_per_layer_attributes(per_layer_config)

    # Ensure all required global attributes are defined
    missing_required_global_attributes = set()
    for attr in per_layer_attributes:
        if len(per_layer_config) != config.num_hidden_layers:
            if not hasattr(config, attr):
                missing_required_global_attributes.add(attr)
        else:
            for layer_config in per_layer_config.values():
                if not hasattr(layer_config, attr):
                    if not hasattr(config, attr):
                        missing_required_global_attributes.add(attr)
                    break

    if missing_required_global_attributes:
        raise ValueError(
            f"The following attributes are missing: {sorted(missing_required_global_attributes)}\nPlease add them globally, or make sure they are defined in all of the per-layer configs"
        )

    for attr in per_layer_attributes:
        # Gather all values for this attribute across all layers,
        # and if `explicit` is True, enforce that `per_layer_config` has a LayerConfig for each layer
        # and that each layer has all per-layer attributes defined.
        values_list = []
        for layer_idx in range(config.num_hidden_layers):
            layer_config = per_layer_config.get(layer_idx)

            if explicit:
                if layer_config is None:
                    layer_config = LayerConfig()
                    per_layer_config[layer_idx] = layer_config

                if not hasattr(layer_config, attr):
                    setattr(layer_config, attr, getattr(config, attr))

            value = (
                getattr(layer_config, attr)
                if layer_config is not None and hasattr(layer_config, attr)
                else getattr(config, attr)
            )
            if value not in values_list:
                values_list.append(value)

        if not explicit and len(values_list) == 1:
            # All layer configs have the same value for this attribute, so it can be a global attribute
            setattr(config, attr, values_list[0])
            for layer_idx, layer_config in per_layer_config.items():
                if hasattr(layer_config, attr):
                    delattr(layer_config, attr)

    # Delete all empty layer configs
    for layer_idx, layer_config in list(per_layer_config.items()):
        if not layer_config.attributes:
            del per_layer_config[layer_idx]

    per_layer_attributes = _get_per_layer_attributes(per_layer_config)
    fallback_values = {attr: getattr(config, attr, None) for attr in per_layer_attributes}

    heterogeneity_spec = HeterogeneitySpec(
        per_layer_config=per_layer_config,
        per_layer_attributes=per_layer_attributes,
        fallback_values=fallback_values,
    )
    return heterogeneity_spec


def _get_per_layer_attributes(per_layer_config: dict[int, LayerConfig]) -> set[str]:
    return {
        attr
        for layer_config in per_layer_config.values()
        for attr in layer_config.attributes
        if not attr.startswith("skip_")
    }
