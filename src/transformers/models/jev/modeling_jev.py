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
from dataclasses import dataclass

import torch
from torch import nn

from ...decision import Question
from ...decision.modeling_utils import DecisionMixin
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple
from ..auto import AutoModel
from .configuration_jev import JevConfig


@auto_docstring
class JevPreTrainedModel(PreTrainedModel):
    config: JevConfig
    base_model_prefix = "model"
    _supports_sdpa = _supports_flash_attn = _supports_flex_attn = True


@auto_docstring
@dataclass
class JevDecisionOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `()`, *optional*):
        Mean KL loss plus the weighted ranked probability score on `score` questions, when labels are supplied.
    logits (`torch.FloatTensor` of shape `(total_num_options,)`):
        Uncalibrated active-option scores concatenated in question order.
    """

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None


@auto_docstring(
    custom_intro="""
    JEV (autotrust/JEV-9B): the model of its repo and a decision head, linear scores of single-token labels at the token
    before the `<|mark|>` of each question, soft-capped like the logits of a language model head.
    """
)
class JevForDecision(DecisionMixin, JevPreTrainedModel):
    # Packed soft labels count options, so Trainer must not infer a question count from their shape.
    accepts_loss_kwargs = False

    def __init__(self, config: JevConfig):
        super().__init__(config)
        self.model = AutoModel.from_config(config.text_config)
        self.proj = nn.Linear(config.hidden_size, config.num_verbalizers)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        marks: torch.LongTensor,
        num_options: torch.LongTensor,
        question_type_ids: torch.LongTensor,
        labels: torch.Tensor | None = None,
        loss_weights: torch.Tensor | None = None,
        **kwargs,
    ) -> JevDecisionOutput:
        r"""
        labels (`torch.Tensor`, *optional*):
            Integer option indices of shape `(1, num_questions)`, with `-100` for ignored questions, or floating-point
            target distributions of shape `(1, total_num_options)`, concatenated in question order. Each distribution
            must sum to one over that question's active options.
        loss_weights (`torch.Tensor` of shape `(1, num_questions)`, *optional*):
            Per-question weights. The weighted loss is divided by the number of valid questions, not the sum of weights.
        """
        # A batch of one prompt
        marks, num_options, question_type_ids = marks[0], num_options[0], question_type_ids[0]
        hidden_states = self.model(input_ids=input_ids, use_cache=False, **kwargs).last_hidden_state[0]
        logits = self.proj(hidden_states[marks - 1].to(self.proj.weight.dtype))
        if self.config.softcap is not None:
            logits = self.config.softcap * torch.tanh(logits / self.config.softcap)
        # The options of a question are scored by the rows from the offset of its type on, in order
        offsets = [self.config.offsets[question_type] for question_type in Question.questions_types]
        starts = torch.tensor(offsets, device=logits.device)[question_type_ids]
        rows = torch.arange(logits.shape[-1], device=logits.device)
        scores = logits[(rows >= starts[:, None]) & (rows < (starts + num_options)[:, None])]
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=scores,
                labels=labels,
                num_options=num_options,
                is_score=question_type_ids == Question.questions_types.index("score"),
                rps_weight=self.config.rps_weight,
                loss_weights=loss_weights,
                **kwargs,
            )
        return JevDecisionOutput(loss=loss, logits=scores)


__all__ = ["JevForDecision", "JevPreTrainedModel"]
