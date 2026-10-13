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
    sliding_window (`int`, *optional*, defaults to 64):
        Window of the sliding attention layers of the backbone, on the position ids of the options.
    calibration (`dict[str, float]`, *optional*):
        Weights of the input-conditioned temperature, `clamp(bias + entropy * H + log_tokens * log10(state tokens) / 4
        + n_options * K / 8, lo, hi)`, with `H` the normalized entropy of the scores and `K` the number of options.
    temperature (`float`, *optional*, defaults to 1.0):
        Temperature of the scores when there is no `calibration`.
    noul_prior (`dict[str, float]`, *optional*):
        Debiasing of zero-shot `noul` questions by the block without the state: the score of `true` is lowered by
        `a * (null_true - null_false) + b`.
    noul_band (`list[float]`, *optional*):
        `[edge, slope]` of the band of `noul` answers, `edge + slope * (p - 0.5)` above 0.5 and mirrored below.

    Example:
    ```python
    >>> from transformers import VonConfig, VonDecisionHead

    >>> head = VonDecisionHead(VonConfig(hidden_size=1024))
    ```"""

    model_type = "von"

    hidden_size: int = 1024
    sliding_window: int = 64
    calibration: dict[str, float] | None = None
    temperature: float = 1.0
    noul_prior: dict[str, float] | None = None
    noul_band: list[float] | None = None
    initializer_range: float = 0.02


__all__ = ["VonConfig"]
