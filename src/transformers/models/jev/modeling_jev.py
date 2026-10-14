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
import torch
from torch import nn

from ...decision import QUESTION_TYPES
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from .configuration_jev import JevConfig


@auto_docstring
class JevPreTrainedModel(PreTrainedModel):
    config: JevConfig
    main_input_name = "hidden_states"


@auto_docstring(
    custom_intro="""
    Decision head of JEV (autotrust/JEV-9B): linear scores of single-token labels at the token before the `<|mark|>` of
    each question, soft-capped like the logits of a language model head.
    """
)
class JevDecisionHead(JevPreTrainedModel):
    def __init__(self, config: JevConfig):
        super().__init__(config)
        self.proj = nn.Linear(config.hidden_size, config.num_verbalizers)
        self.post_init()

    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        marks: torch.LongTensor,
        num_options: torch.LongTensor,
        question_type_ids: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        hidden_states (`torch.Tensor` of shape `(sequence_length, hidden_size)`):
            Final hidden states of the backbone for one prompt.
        """
        logits = self.proj(hidden_states[marks - 1])
        if self.config.softcap is not None:
            logits = self.config.softcap * torch.tanh(logits / self.config.softcap)
        # The options of a question are scored by the rows from the offset of its type on, in order
        starts = torch.zeros_like(question_type_ids)
        for type_id, question_type in enumerate(QUESTION_TYPES):
            starts[question_type_ids == type_id] = self.config.offsets[question_type]
        rows = torch.arange(logits.shape[-1], device=logits.device)
        return logits[(rows >= starts[:, None]) & (rows < (starts + num_options)[:, None])]


__all__ = ["JevDecisionHead", "JevPreTrainedModel"]
