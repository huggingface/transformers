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

import torch
from torch import nn

from ...decision import DecisionPrompt
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from .configuration_kev import KevConfig


@auto_docstring
class KevPreTrainedModel(PreTrainedModel):
    config: KevConfig
    main_input_name = "hidden_states"


@auto_docstring(
    custom_intro="""
    Decision head of kev (jaredpalmer/kev-4b): each option is scored by the dot product of the projected hidden states at
    the end of the option and at the end of the question. Each question has one `<|mark|>` after each of its options and
    one after itself.
    """
)
class KevDecisionHead(KevPreTrainedModel):
    def __init__(self, config: KevConfig):
        super().__init__(config)
        self.key = nn.Linear(config.hidden_size, config.head_dim)
        self.query = nn.Linear(config.hidden_size, config.head_dim)
        self.post_init()

    @auto_docstring
    def forward(
        self, hidden_states: torch.Tensor, prompt: DecisionPrompt, lm_head: nn.Module | None = None, **kwargs
    ) -> list[torch.Tensor]:
        r"""
        hidden_states (`torch.Tensor` of shape `(sequence_length, hidden_size)`):
            Final hidden states of the backbone for one prompt.
        """
        states = hidden_states[[mark - 1 for mark in prompt.marks]]
        blocks = states.split([len(question.options) + 1 for question in prompt.questions])
        return [self.key(block[:-1]) @ self.query(block[-1]) / math.sqrt(self.config.head_dim) for block in blocks]


__all__ = ["KevDecisionHead", "KevPreTrainedModel"]
