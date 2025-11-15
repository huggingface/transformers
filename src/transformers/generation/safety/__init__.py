# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from ...utils import is_torch_available
from .base import SafetyChecker, SafetyMetrics, SafetyResult, SafetyState, SafetyViolation
from .configuration import LENIENT_PRESET, MODERATE_PRESET, STRICT_PRESET, SafetyConfig


if is_torch_available():
    from .processors import SafetyLogitsProcessor, SafetyStoppingCriteria
else:
    SafetyLogitsProcessor = None
    SafetyStoppingCriteria = None


__all__ = [
    "SafetyChecker",
    "SafetyResult",
    "SafetyViolation",
    "SafetyMetrics",
    "SafetyState",
    "SafetyConfig",
    "STRICT_PRESET",
    "MODERATE_PRESET",
    "LENIENT_PRESET",
]

if is_torch_available():
    __all__ += ["SafetyLogitsProcessor", "SafetyStoppingCriteria"]
