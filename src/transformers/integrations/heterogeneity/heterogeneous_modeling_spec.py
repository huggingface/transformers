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

_MODEL_MODULE_PREFIX = "transformers.models."

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


@dataclass
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


def get_heterogeneous_modeling_spec(model: PreTrainedModel) -> HeterogeneousModelingSpec:
    heterogeneous_modeling_spec = getattr(model, "_heterogeneous_modeling_spec", None)

    if heterogeneous_modeling_spec is not None:
        return heterogeneous_modeling_spec

    model_module_name = type(model).__module__
    if not model_module_name.startswith(_MODEL_MODULE_PREFIX):
        raise ValueError(
            f"No heterogeneous modeling spec is defined for `{model.__class__.__name__}` in `{model_module_name}`. Make sure `_heterogeneous_modeling_spec` is set on the model class."
        ) from None

    model_package_name = model_module_name.removeprefix(_MODEL_MODULE_PREFIX).split(".", 1)[0]
    from transformers.integrations.heterogeneity.supported_models import MODEL_TO_SPEC_FACTORY

    spec_factory = MODEL_TO_SPEC_FACTORY.get(model_package_name)

    if spec_factory is None:
        raise ValueError(
            f"No heterogeneous modeling spec is defined for `{model_package_name}`. Built-in heterogeneous modeling "
            "support is only available for models listed in "
            "`transformers.integrations.heterogeneity.supported_models.MODEL_TO_SPEC_FACTORY`. Alternatively, set `_heterogeneous_modeling_spec` on the model class."
        )

    return spec_factory()
