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

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias


if TYPE_CHECKING:
    from torch import nn

    from transformers.modeling_utils import PreTrainedModel

SkipReplacements: TypeAlias = "dict[str | tuple[str, type], Callable[[], nn.Module]]"


@dataclass(frozen=True)
class SkipDescriptor:
    """Describes the module replacements and cache effect of a heterogeneous skip type.

    Args:
        replacements: Factories for the modules that replace layer members, keyed by one of two forms:
            - `"member_name"`: always replaces that member (e.g. `"self_attn"`).
            - `("member_name", member_class)`: replaces the member only when it is an instance of `member_class`,
            taking precedence over a plain member-name key (e.g. `("mixer", NemotronHAttention)`).
        replaces_kv_cache_updater: Whether this skip replaces the member that updates the layer's KV cache,
        leaving the layer without KV-cache state.
    """

    replacements: SkipReplacements
    replaces_kv_cache_updater: bool


@dataclass(frozen=True)
class HeterogeneousModelingSpec:
    layer_cls: type[nn.Module]
    # Layer-index argument or local variable name used by the model's layer construction path.
    # Common names models use include `layer_idx`, `idx`, `layer_id`, `layer_number`, `i`, and `_`.
    layer_idx_variable_name: str
    skip_descriptors: dict[str, SkipDescriptor] | None = None


def nest_skip_descriptor_paths(
    skip_descriptors: dict[str, SkipDescriptor] | None, parent_path: str
) -> dict[str, SkipDescriptor] | None:
    """Return new skip descriptors whose replacement paths are nested under a parent attribute path.

    Args:
        skip_descriptors: Skip descriptors to adapt, or `None`.
        parent_path: Attribute path under which to nest every replacement path.

    Returns:
        New skip descriptors with nested replacement paths, or `None` when `skip_descriptors` is `None`.
    """
    if skip_descriptors is None:
        return None

    nested_descriptors = {}
    for skip_type, descriptor in skip_descriptors.items():
        replacements = {}
        for key, replacement in descriptor.replacements.items():
            if isinstance(key, tuple):
                member_path, member_cls = key
                nested_key = (f"{parent_path}.{member_path}", member_cls)
            else:
                nested_key = f"{parent_path}.{key}"
            replacements[nested_key] = replacement

        nested_descriptors[skip_type] = SkipDescriptor(
            replacements=replacements,
            replaces_kv_cache_updater=descriptor.replaces_kv_cache_updater,
        )

    return nested_descriptors


def get_heterogeneous_modeling_spec(model: PreTrainedModel) -> HeterogeneousModelingSpec | None:
    heterogeneous_modeling_spec = getattr(model, "_heterogeneous_modeling_spec", None)

    if heterogeneous_modeling_spec is not None:
        return heterogeneous_modeling_spec

    if getattr(model, "_disable_heterogeneous_modeling_patching", False):
        return None

    model_type = model.config.model_type

    from transformers.integrations.heterogeneity.supported_models import (
        MODEL_TYPE_TO_SPEC_FACTORY,
        MODEL_TYPES_WITH_HETEROGENEOUS_MODELING_PATCHING_DISABLED,
    )

    if model_type in MODEL_TYPES_WITH_HETEROGENEOUS_MODELING_PATCHING_DISABLED:
        return None

    spec_factory = MODEL_TYPE_TO_SPEC_FACTORY.get(model_type)

    if spec_factory is None:
        raise ValueError(
            f"No heterogeneous modeling behavior is defined for model type `{model_type}`.\n\n"
            "Choose one of the following:\n"
            "1. Generic patching:\n"
            "   - Custom model: set `_heterogeneous_modeling_spec` on the model class.\n"
            "   - Built-in model: add a spec factory in "
            "`transformers.integrations.heterogeneity.supported_models.MODEL_TYPE_TO_SPEC_FACTORY`.\n"
            "2. Patching disabled:\n"
            "   - Custom model: set `_disable_heterogeneous_modeling_patching = True` on the model "
            "class.\n"
            "   - Built-in model: add its model type to "
            "`transformers.integrations.heterogeneity.supported_models."
            "MODEL_TYPES_WITH_HETEROGENEOUS_MODELING_PATCHING_DISABLED`.\n\n"
            "See the heterogeneous modeling guide at "
            "https://huggingface.co/docs/transformers/main/en/heterogeneous_modeling."
        )

    return spec_factory()
