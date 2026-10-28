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

from ...decision.modeling_utils import ForDecision
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from .configuration_kev import KevConfig


@auto_docstring
class KevPreTrainedModel(PreTrainedModel):
    config: KevConfig


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
        self,
        input_ids: torch.LongTensor,
        marks: torch.LongTensor,
        num_options: torch.LongTensor,
        backbone: PreTrainedModel,
        backbone_kwargs: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        outputs = backbone.base_model(input_ids=input_ids[None], use_cache=False, **(backbone_kwargs or {}))
        hidden_states = outputs.last_hidden_state[0].to(self.dtype)
        blocks = torch.split(hidden_states[marks - 1], (num_options + 1).tolist())
        scores = torch.cat([self.key(block[:-1]) @ self.query(block[-1]) for block in blocks])
        return scores / math.sqrt(self.config.head_dim)


class KevForDecision(ForDecision):
    """Kev: the model of a repo and its [`KevDecisionHead`]."""


__all__ = ["KevDecisionHead", "KevForDecision", "KevPreTrainedModel"]
