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
"""Decision heads: the mixin of the head models, and the default head of models without one."""

import torch

from .template_utils import DecisionPrompt, label_token_ids


DECISION_HEAD_FOLDER = "decision_head"


class DecisionHeadMixin:
    """
    Decision heads are models named after the first model that used them, saved in the `decision_head` folder next to
    the original checkpoint of their backbone. `forward(hidden_states, prompt, lm_head)` scores the options of each
    question of the prompt.
    """

    def backbone_inputs(self, prompt: DecisionPrompt, model) -> dict:
        return {"input_ids": torch.tensor([prompt.input_ids], device=model.device), "use_cache": False}

    def decide(self, model, prompt: DecisionPrompt) -> list[torch.Tensor]:
        hidden_states = model.base_model(**self.backbone_inputs(prompt, model)).last_hidden_state[0]
        return self(hidden_states.to(self.dtype), prompt, model.get_output_embeddings())


class LabelHead:
    """Default head: the logits of the single-token labels of the options, at the token before each `<|mark|>`."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def decide(self, model, prompt: DecisionPrompt) -> list[torch.Tensor]:
        input_ids = torch.tensor([prompt.input_ids], device=model.device)
        positions = torch.tensor(prompt.marks, device=model.device) - 1
        logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=positions).logits[0]
        return [
            logits[index, label_token_ids(self.tokenizer, question)] for index, question in enumerate(prompt.questions)
        ]


__all__ = ["DECISION_HEAD_FOLDER", "DecisionHeadMixin", "LabelHead"]
