# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="wfzyx/von")
@strict
class VonConfig(PreTrainedConfig):
    r"""
    noul_prior (`dict[str, float]`, *optional*):
        Debiasing of zero-shot `noul` questions by the block without the state: the score of `true` is lowered by
        `a * (null_true - null_false) + b`.

    Example:
    ```python
    >>> from transformers import VonConfig, VonDecisionHead

    >>> head = VonDecisionHead(VonConfig(hidden_size=1024))
    ```"""

    model_type = "von"

    hidden_size: int = 1024
    noul_prior: dict[str, float] | None = None
    initializer_range: float = 0.02


__all__ = ["VonConfig"]
