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
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ...decision_utils import DecisionPrompt
from ...modeling_decision_heads import DecisionHeadMixin
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from .configuration_von import VonConfig


@auto_docstring
class VonPreTrainedModel(PreTrainedModel):
    config: VonConfig
    main_input_name = "hidden_states"


@auto_docstring(
    custom_intro="""
    Decision head of von (wfzyx/von): each option is scored by an MLP on the hidden state of its `[MASK]` (the token before
    each `<|mark|>`, a last mark closing the block), with options that only attend to the prompt and to themselves. A
    zero-shot `noul` question has a second block without the state, whose scores debias the first, and the state is
    written once more after the last block for the token count of the input-conditioned temperature.
    """
)
class VonDecisionHead(VonPreTrainedModel, DecisionHeadMixin):
    def __init__(self, config: VonConfig):
        super().__init__(config)
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.norm = nn.LayerNorm(config.hidden_size // 2)
        self.out_proj = nn.Linear(config.hidden_size // 2, 1)
        self.post_init()

    def blocks(self, prompt: DecisionPrompt) -> list[tuple[int, list[int], int]]:
        """`(start, [MASK] positions, end)` of each block."""
        per_block, marks = len(prompt.questions[0].options) + 1, prompt.marks
        if len(prompt.questions) != 1 or len(marks) not in (per_block, 2 * per_block):
            raise ValueError(f"Expected one question and {per_block} or {2 * per_block} marks, got {len(marks)}.")
        groups = [marks[i : i + per_block] for i in range(0, len(marks), per_block)]
        starts = [0, *marks[per_block - 1 :: per_block]]
        return [(start, [mark - 1 for mark in group[:-1]], group[-1]) for start, group in zip(starts, groups)]

    def backbone_inputs(self, prompt: DecisionPrompt) -> dict[str, Any]:
        """Options attend to the prompt and to themselves, and their positions restart after the prompt."""
        blocks = self.blocks(prompt)
        length = blocks[-1][2]
        position_ids, option_ids = torch.arange(length), torch.full((length,), -1)
        block_ids = torch.zeros(length, dtype=torch.long)
        for block, (start, reads, end) in enumerate(blocks):
            block_ids[start:end] = block
            position_ids[start:end] = torch.arange(end - start)
            for k, (begin, stop) in enumerate(zip(reads, [*reads[1:], end - 1])):
                option_ids[begin:stop] = k
                position_ids[begin:stop] = torch.arange(reads[0] - start, reads[0] - start + stop - begin)
        query, key = option_ids[:, None], option_ids[None, :]
        allowed = ((query == -1) & (key == -1)) | ((query != -1) & ((key == -1) | (query == key)))
        allowed = (allowed & (block_ids[:, None] == block_ids[None, :])) | torch.eye(length, dtype=torch.bool)
        local = (position_ids[:, None] - position_ids[None, :]).abs() <= self.config.sliding_window
        return {
            "input_ids": torch.tensor([prompt.input_ids[:length]]),
            "position_ids": position_ids[None],
            "attention_mask": {
                "full_attention": allowed[None, None],
                "sliding_attention": (allowed & local)[None, None],
            },
        }

    def option_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.norm(F.gelu(self.dense(self.input_norm(hidden_states)))))[:, 0]

    @auto_docstring
    def forward(
        self, hidden_states: torch.Tensor, prompt: DecisionPrompt, lm_head: nn.Module | None = None
    ) -> list[torch.Tensor]:
        r"""
        hidden_states (`torch.Tensor` of shape `(sequence_length, hidden_size)`):
            Final hidden states of the backbone for one prompt.
        prompt (`DecisionPrompt`):
            The prompt: its input ids, the positions recorded by `<|mark|>` and its questions.
        lm_head (`nn.Module`, *optional*):
            The output embeddings of the backbone.
        """
        (question,), blocks, config = prompt.questions, self.blocks(prompt), self.config
        logits = self.option_logits(hidden_states[blocks[0][1]])
        if len(blocks) == 2:  # zero-shot noul
            true, false = question.options.index("true"), question.options.index("false")
            null = self.option_logits(hidden_states[blocks[1][1]])
            correction = config.noul_prior["a"] * (null[true] - null[false]) + config.noul_prior["b"]
            logits = torch.stack([logits[i] - correction if i == true else logits[i] for i in range(2)])
        temperature = config.temperature
        if config.calibration:
            probabilities, n = torch.softmax(logits.float(), dim=-1), logits.numel()
            entropy = (
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum().item() / math.log(n) if n > 1 else 0.0
            )
            state_tokens = len(prompt.input_ids) - prompt.marks[-1]
            features = {
                "bias": 1.0,
                "entropy": entropy,
                "log_tokens": math.log10(max(state_tokens, 1)) / 4,
                "n_options": n / 8,
            }
            weighted = sum(config.calibration.get(name, 0.0) * value for name, value in features.items())
            temperature = min(config.calibration["hi"], max(config.calibration["lo"], weighted))
        logits = logits / max(temperature, 1e-4)
        if question.type == "noul" and config.noul_band:
            edge, slope = config.noul_band
            true = question.options.index("true")
            p = min(1.0, max(0.0, torch.softmax(logits.float(), dim=-1)[true].item()))
            p = min(1.0, max(0.0, edge + slope * (p - 0.5) if p >= 0.5 else (1.0 - edge) - slope * (0.5 - p)))
            logits = torch.tensor([p if i == true else 1.0 - p for i in range(2)], device=logits.device).log()
        return [logits]


__all__ = ["VonDecisionHead", "VonPreTrainedModel"]
