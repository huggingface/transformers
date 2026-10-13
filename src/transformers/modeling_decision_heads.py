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
"""Inference helpers of decision heads, and the default head of models without one."""

import inspect

import torch

from .decision_utils import DecisionPrompt, label_token_ids


DECISION_HEAD_FOLDER = "decision_head"


class DecisionHeadMixin:
    """
    Decision heads are models (`models/<name>/modeling_<name>.py`, named after the first model that used them) whose
    `forward(hidden_states, prompt, lm_head)` scores the options of each question of a [`DecisionPrompt`]. They are saved
    in the `decision_head` folder of a repo, next to the original checkpoint of their backbone.
    """

    def backbone_inputs(self, prompt: DecisionPrompt) -> dict:
        """The inputs of the backbone for `prompt`."""
        return {"input_ids": torch.tensor([prompt.input_ids])}

    def decide(self, model, prompt: DecisionPrompt) -> list[torch.Tensor]:
        """Run the backbone of `model` and then the head on `prompt`: one 1D tensor of option scores per question."""
        backbone = model.base_model
        inputs = {
            name: value.to(model.device)
            if torch.is_tensor(value)
            else {k: v.to(model.device) for k, v in value.items()}
            for name, value in self.backbone_inputs(prompt).items()
        }
        if "use_cache" in inspect.signature(backbone.forward).parameters:
            inputs["use_cache"] = False
        hidden_states = backbone(**inputs).last_hidden_state[0]
        return self(hidden_states.to(self.dtype), prompt, model.get_output_embeddings())


class LabelHead:
    """Default head: the logits of the single-token labels of the options, at the token before each `<|mark|>`."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def decide(self, model, prompt: DecisionPrompt) -> list[torch.Tensor]:
        input_ids = torch.tensor([prompt.input_ids], device=model.device)
        positions = torch.tensor(prompt.marks, device=model.device) - 1
        logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=positions).logits[0]
        return [logits[i, label_token_ids(self.tokenizer, q)] for i, q in enumerate(prompt.questions)]
