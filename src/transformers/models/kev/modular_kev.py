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
import math
from dataclasses import dataclass

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...decision import DecisionMixin
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ..auto import AutoConfig, AutoModel


@auto_docstring(checkpoint="jaredpalmer/kev-4b")
@strict
class KevConfig(PreTrainedConfig):
    r"""
    head_dim (`int`, *optional*, defaults to 256):
        Dimension of the projections of the hidden states of the options and of the question.
    text_config (`dict` or `PreTrainedConfig`, *optional*):
        Config of the model the head reads, the `config.json` of its repo.

    Example:
    ```python
    >>> from transformers import KevConfig, KevForDecision

    >>> model = KevForDecision(KevConfig(hidden_size=2560))
    ```"""

    model_type = "kev"
    sub_configs_defaults = {"text_config": SubConfigSpec(config_class=AutoConfig, model_type="qwen3_5_text")}

    hidden_size: int = 2560
    head_dim: int = 256
    text_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02


@auto_docstring
class KevPreTrainedModel(PreTrainedModel):
    config: KevConfig
    base_model_prefix = "model"
    _supports_sdpa = _supports_flash_attn = _supports_flex_attn = True


@auto_docstring
@dataclass
class KevDecisionOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `()`, *optional*, returned when `labels` is provided):
        Cross-entropy averaged over the questions of the prompt.
    logits (`torch.FloatTensor` of shape `(total_num_options,)`):
        Unnormalized option scores, concatenated in question order.
    """

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None


@auto_docstring(
    custom_intro="""
    kev (jaredpalmer/kev-4b): the model of its repo and a decision head, each option is scored by the dot product of the
    projected hidden states at the end of the option and at the end of the question. Each question has one `<|mark|>`
    after each of its options and one after itself.
    """
)
class KevForDecision(DecisionMixin, KevPreTrainedModel):
    accepts_loss_kwargs = False

    def __init__(self, config: KevConfig):
        super().__init__(config)
        self.model = AutoModel.from_config(config.text_config)
        self.key = nn.Linear(config.hidden_size, config.head_dim)
        self.query = nn.Linear(config.hidden_size, config.head_dim)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        marks: torch.LongTensor,
        num_options: torch.LongTensor,
        question_type_ids: torch.LongTensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> KevDecisionOutput:
        r"""
        labels (`torch.Tensor`, *optional*):
            Integer option indices of shape `(1, num_questions)`, with `-100` for ignored questions, or floating-point
            target distributions of shape `(1, total_num_options)`, concatenated in question order. Each distribution
            must sum to one over that question's options.
        """
        marks, num_options = marks[0], num_options[0]  # a batch of one prompt
        outputs = self.model(input_ids=input_ids, use_cache=False, return_dict=True, **kwargs)
        hidden_states = outputs.last_hidden_state[0].to(self.key.weight.dtype)
        blocks = torch.split(hidden_states[marks - 1], (num_options + 1).tolist())
        scores = torch.cat([self.key(block[:-1]) @ self.query(block[-1]) for block in blocks])
        scores = scores / math.sqrt(self.config.head_dim)
        loss = (
            self.loss_function(logits=scores, labels=labels, num_options=num_options) if labels is not None else None
        )
        return KevDecisionOutput(loss=loss, logits=scores)


__all__ = ["KevConfig", "KevForDecision", "KevPreTrainedModel"]
