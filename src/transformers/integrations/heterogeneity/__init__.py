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

from .configuration_utils import AmbiguousGlobalPerLayerAttributeError, HeterogeneousConfigMixin
from .heterogeneous_modeling_spec import HeterogeneousModelingSpec, SkipDescriptor, get_heterogeneous_modeling_spec
from .modeling_utils import (
    apply_heterogeneous_modeling,
    wrap_model_init_with_heterogeneous_cleanup,
)
from .skip_utils import ReturnEntry, get_skip_replacement


__all__ = [
    "AmbiguousGlobalPerLayerAttributeError",
    "HeterogeneousConfigMixin",
    "HeterogeneousModelingSpec",
    "ReturnEntry",
    "SkipDescriptor",
    "apply_heterogeneous_modeling",
    "get_heterogeneous_modeling_spec",
    "get_skip_replacement",
    "wrap_model_init_with_heterogeneous_cleanup",
]
