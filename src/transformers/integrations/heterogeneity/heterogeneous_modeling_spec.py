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

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias


if TYPE_CHECKING:
    from torch import nn

    from transformers.modeling_utils import PreTrainedModel

_MODEL_MODULE_PREFIX = "transformers.models."

SkipDescriptor: TypeAlias = "dict[str | tuple[str, type], type[nn.Module]]"


@dataclass
class HeterogeneousModelingSpec:
    layer_cls: type[nn.Module]
    # Layer-index argument or local variable name used by the model's layer construction path.
    # Common names include `layer_idx`, `idx`, `layer_id`, `layer_number`, `i`, and `_`.
    layer_idx_variable_name: str
    skip_descriptors: dict[str, SkipDescriptor] | None = None


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
