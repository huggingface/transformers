# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Shared-parameter execution plans for constant-width decoder-only layer stacks."""

from .adapters import DecoderLayerExecutionAdapter, register_layer_execution_adapter
from .cache import LayerExecutionCache
from .executor import get_layer_execution_plan, set_layer_execution_plan
from .plan import LayerExecutionPlan, LayerExecutionStep, RepeatRange
from .state import LayerExecutionState


__all__ = [
    "DecoderLayerExecutionAdapter",
    "LayerExecutionCache",
    "LayerExecutionPlan",
    "LayerExecutionState",
    "LayerExecutionStep",
    "RepeatRange",
    "get_layer_execution_plan",
    "register_layer_execution_adapter",
    "set_layer_execution_plan",
]
