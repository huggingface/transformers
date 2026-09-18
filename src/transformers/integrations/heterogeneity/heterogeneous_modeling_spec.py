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

from transformers.integrations.heterogeneity.layer_idx_resolvers import LayerIdxResolver


if TYPE_CHECKING:
    from torch import nn

    from transformers.modeling_utils import PreTrainedModel


# Class-specific (member name, member class) keys take precedence over plain member names.
SkipDescriptors: TypeAlias = dict[str | tuple[str, type], Callable[[], "nn.Module"]]


@dataclass(frozen=True)
class HeterogeneousModelingSpec:
    layer_cls: type[nn.Module]
    layer_idx_resolver: LayerIdxResolver
    skip_descriptors: dict[str, SkipDescriptors] | None = None


def nest_skip_descriptor_paths(
    skip_descriptors: dict[str, SkipDescriptors] | None, parent_path: str
) -> dict[str, SkipDescriptors] | None:
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
    for skip_type, targets in skip_descriptors.items():
        nested_targets = {}
        for key, replacement_factory in targets.items():
            if isinstance(key, tuple):
                member_path, member_cls = key
                nested_key = (f"{parent_path}.{member_path}", member_cls)
            else:
                nested_key = f"{parent_path}.{key}"
            nested_targets[nested_key] = replacement_factory

        nested_descriptors[skip_type] = nested_targets

    return nested_descriptors


def get_heterogeneous_modeling_spec(model: PreTrainedModel) -> HeterogeneousModelingSpec | None:
    """Return the generic heterogeneous modeling spec explicitly enabled for ``model``, if any.

    A model class may provide its own spec. Otherwise, built-in support is resolved from the model-type registry.
    Models with neither declaration may still consume ``per_layer_config`` natively and are left unpatched.
    """
    heterogeneous_modeling_spec = getattr(model, "_heterogeneous_modeling_spec", None)

    if heterogeneous_modeling_spec is not None:
        return heterogeneous_modeling_spec

    model_type = model.config.model_type

    from transformers.integrations.heterogeneity.supported_models import MODEL_TYPE_TO_SPEC_FACTORY

    spec_factory = MODEL_TYPE_TO_SPEC_FACTORY.get(model_type)
    return spec_factory() if spec_factory is not None else None
