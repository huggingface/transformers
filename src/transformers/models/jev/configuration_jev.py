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

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="autotrust/JEV-9B")
@strict
class JevConfig(PreTrainedConfig):
    r"""
    num_verbalizers (`int`, *optional*, defaults to 24):
        Number of single-token labels scored by the head, for all question types.
    offsets (`dict[str, int]`, *optional*, defaults to `{"noul": 0, "score": 2, "choice": 8}`):
        Row of the first label of each question type: the options of a question are scored by the rows from
        `offsets[question.type]` on, in order.
    softcap (`float`, *optional*):
        Soft cap of the scores, `softcap * tanh(scores / softcap)`, like the logits of a language model head.
    keep_in_fp32_modules (`list[str]`, *optional*):
        Patterns of the parameters kept and run in float32 whatever the dtype of the head, like `["^proj"]` for the
        float32 readout of vllm-sr/d3.

    Example:
    ```python
    >>> from transformers import JevConfig, JevDecisionHead

    >>> head = JevDecisionHead(JevConfig(hidden_size=4096))
    ```"""

    model_type = "jev"
    sub_configs_defaults = {"backbone_config": SubConfigSpec(config_class=AutoConfig, model_type="qwen3_5_text")}

    hidden_size: int = 4096
    num_verbalizers: int = 24
    offsets: dict[str, int] | None = None
    softcap: float | None = None
    keep_in_fp32_modules: list[str] | None = None
    question_types: list[str] | None = None
    backbone_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.offsets is None:
            self.offsets = {"noul": 0, "score": 2, "choice": 8}
        if self.question_types is None:
            self.question_types = ["noul", "choice", "score"]
        super().__post_init__(**kwargs)


__all__ = ["JevConfig"]
