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

from ...decision import Question
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ..auto import AutoModel
from .configuration_jev import JevConfig


@auto_docstring
class JevPreTrainedModel(PreTrainedModel):
    config: JevConfig
    base_model_prefix = "model"
    _supports_sdpa = _supports_flash_attn = _supports_flex_attn = True


@auto_docstring(
    custom_intro="""
    JEV (autotrust/JEV-9B): the model of its repo and a decision head, linear scores of single-token labels at the token
    before the `<|mark|>` of each question, soft-capped like the logits of a language model head.
    """
)
class JevForDecision(JevPreTrainedModel):
    def __init__(self, config: JevConfig):
        super().__init__(config)
        self.model = AutoModel.from_config(config.text_config)
        self.proj = nn.Linear(config.hidden_size, config.num_verbalizers)
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        marks: torch.LongTensor,
        num_options: torch.LongTensor,
        question_type_ids: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids=input_ids[None], use_cache=False, **kwargs).last_hidden_state[0]
        logits = self.proj(hidden_states[marks - 1].to(self.proj.weight.dtype))
        if self.config.softcap is not None:
            logits = self.config.softcap * torch.tanh(logits / self.config.softcap)
        # The options of a question are scored by the rows from the offset of its type on, in order
        offsets = [self.config.offsets[question_type] for question_type in Question.questions_types]
        starts = torch.tensor(offsets, device=logits.device)[question_type_ids]
        rows = torch.arange(logits.shape[-1], device=logits.device)
        return logits[(rows >= starts[:, None]) & (rows < (starts + num_options)[:, None])]


__all__ = ["JevForDecision", "JevPreTrainedModel"]
