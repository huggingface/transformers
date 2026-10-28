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
"""`decide`, the `generate` of decision models: like the `GenerationMixin` of the models that generate."""

from typing import Any

import torch

from .configuration_utils import DecisionConfig
from .template_utils import Question


def read_marks(inputs: dict[str, Any], sentinel_ids: tuple[int, int]) -> tuple[dict[str, Any], torch.Tensor]:
    """
    Drops the sentinels from the inputs of a prompt (a batch of one): the marks are the number of tokens before each
    `<|mark|>`, or all of them for a prompt without marks (read after the generation prompt).
    """
    mark, split = sentinel_ids
    input_ids = inputs["input_ids"][0]
    keep = (input_ids != mark) & (input_ids != split)
    marks = keep.cumsum(0)[input_ids == mark]
    inputs = {
        key: value[:, keep] if getattr(value, "shape", ())[-1:] == input_ids.shape else value
        for key, value in inputs.items()
        if key != "attention_mask"
    }
    return inputs, marks if len(marks) else keep.sum(0, keepdim=True)


class DecisionMixin:
    """
    The `decide` of the `XxxForDecision` models, with their `decision_config`: the decision config of their repo, or the
    one of the checkpoint they read the head of, like `generation_config` for `generate`.
    """

    @property
    def decision_config(self) -> DecisionConfig:
        if getattr(self, "_decision_config", None) is None:
            try:
                self._decision_config = DecisionConfig.from_pretrained(self.name_or_path)
            except OSError:  # a checkpoint shipping the head next to its weights
                from_checkpoint = getattr(type(self.config), "decision_config", None)
                self._decision_config = (from_checkpoint and from_checkpoint(self.name_or_path)) or DecisionConfig()
        return self._decision_config

    @decision_config.setter
    def decision_config(self, decision_config: DecisionConfig):
        self._decision_config = decision_config

    @torch.no_grad()
    def decide(
        self,
        input_ids: torch.LongTensor,
        decision_questions: list[Question],
        decision_config: DecisionConfig | None = None,
        **kwargs,
    ) -> list[torch.Tensor]:
        """
        Scores the options of each question of a prompt rendered with `decision_config.chat_template_kwargs`: one
        forward pass, then the logits processors of the decision config. Returns the scores of each question, which
        `decode_decision` turns into answers.
        """
        decision_config = decision_config or self.decision_config
        inputs, marks = read_marks({"input_ids": input_ids, **kwargs}, decision_config._sentinel_ids)
        num_options = torch.tensor([[len(question.options) for question in decision_questions]])
        question_types = [Question.questions_types.index(question.type) for question in decision_questions]
        scores = self(
            marks=marks[None].to(self.device),
            num_options=num_options.to(self.device),
            question_type_ids=torch.tensor([question_types], device=self.device),
            **{key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()},
        )
        # The tokens after the last mark, where the templates write the state
        tail = inputs["input_ids"][0, int(marks[-1]) :]
        return [
            decision_config.get_logits_processor(question.type)(tail, question_scores.float())
            for question, question_scores in zip(decision_questions, scores.split(num_options[0].tolist()))
        ]


__all__ = ["DecisionMixin"]
