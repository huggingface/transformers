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
"""Heads of decision models: they score the options of the questions of a [`DecisionPrompt`]."""

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from .decision_utils import DECISION_HEAD_NAME, DecisionConfig, DecisionPrompt, label_token_ids
from .utils import cached_file


class DecisionHead(nn.Module):
    """
    Base class of the heads of decision models. A head is built from `decision_config.head` and its weights live in
    `decision_head.safetensors`, so it can be pushed alone next to the original checkpoint of the backbone.

    Subclasses implement `score`, which gets the final hidden states of one prompt, `(seq_len, hidden_size)`.
    """

    def forward(self, model, prompt: DecisionPrompt) -> list[torch.Tensor]:
        """Run the backbone of `model` on `prompt`, and return one 1D tensor of option scores per question."""
        input_ids = torch.tensor([prompt.input_ids], device=model.device)
        hidden_states = model.base_model(input_ids=input_ids, use_cache=False).last_hidden_state[0]
        dtype = next(self.parameters(), hidden_states).dtype
        return self.score(hidden_states.to(dtype), prompt, model.get_output_embeddings())

    def score(
        self, hidden_states: torch.Tensor, prompt: DecisionPrompt, lm_head: nn.Module | None
    ) -> list[torch.Tensor]:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, config: DecisionConfig, **kwargs):
        """Build the head of `config.head` and load `decision_head.safetensors` (`kwargs` go to `cached_file`)."""
        head_kwargs = {key: value for key, value in config.head.items() if key != "type"}
        head = DECISION_HEADS[config.head["type"]](**head_kwargs)
        head.load_state_dict(load_file(cached_file(pretrained_model_name_or_path, DECISION_HEAD_NAME, **kwargs)))
        return head.eval()


class LabelHead(DecisionHead):
    """Scores the options with the logits of their single-token labels, at the last token before each `<|mark|>`."""

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, model, prompt: DecisionPrompt) -> list[torch.Tensor]:
        input_ids = torch.tensor([prompt.input_ids], device=model.device)
        positions = torch.tensor(prompt.marks, device=model.device) - 1
        logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=positions).logits[0]
        return [logits[i, label_token_ids(self.tokenizer, q)] for i, q in enumerate(prompt.questions)]


DECISION_HEADS = {}
