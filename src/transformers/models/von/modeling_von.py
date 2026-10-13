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
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ...decision import DecisionHeadMixin, DecisionPrompt
from ...masking_utils import (
    and_masks,
    create_bidirectional_mask,
    or_masks,
    packed_sequence_mask_function,
    padding_mask_function,
    sliding_window_bidirectional_overlay,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from .configuration_von import VonConfig


def option_mask_function(block_ids: torch.Tensor, option_ids: torch.Tensor) -> Callable:
    """
    Within a block, tokens attend to the prompt (option `-1`) and to their own option: the prompt only sees itself, and
    every option the prompt and itself.
    """
    return and_masks(
        packed_sequence_mask_function(block_ids),
        or_masks(packed_sequence_mask_function(option_ids), padding_mask_function(option_ids == -1)),
    )


def on_position_ids(mask_function: Callable, position_ids: torch.Tensor) -> Callable:
    """`mask_function` applied to the position ids of the query and key tokens instead of their indices."""

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return mask_function(batch_idx, head_idx, position_ids[batch_idx, q_idx], position_ids[batch_idx, kv_idx])

    return inner_mask


@auto_docstring
class VonPreTrainedModel(PreTrainedModel):
    config: VonConfig
    main_input_name = "hidden_states"


@auto_docstring(
    custom_intro="""
    Decision head of von (wfzyx/von): each option is scored by an MLP on the hidden state of its `[MASK]` (the token before
    each `<|mark|>`, a last mark closing the block), with options that only attend to the prompt and to themselves. A
    zero-shot `noul` question has a second block without the state, whose scores debias the first. The state written
    once more after the last block is not given to the backbone, the pipeline counts its tokens.
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
        groups = [marks[start : start + per_block] for start in range(0, len(marks), per_block)]
        starts = [0, *marks[per_block - 1 :: per_block]]
        return [(start, [mark - 1 for mark in group[:-1]], group[-1]) for start, group in zip(starts, groups)]

    def backbone_inputs(self, prompt: DecisionPrompt, model) -> dict[str, Any]:
        """
        Blocks are mutually invisible, options attend to the prompt of their block and to themselves, and the positions of
        every option restart after the prompt, also for the sliding window. The masking API makes the masks for the
        attention implementation of the backbone.
        """
        blocks = self.blocks(prompt)
        length = blocks[-1][2]  # without the state written after the last block
        block_ids, option_ids, position_ids = [0] * length, [-1] * length, list(range(length))
        for block, (start, reads, end) in enumerate(blocks):
            block_ids[start:end] = [block] * (end - start)
            position_ids[start:end] = range(end - start)
            for option_index, (begin, stop) in enumerate(zip(reads, [*reads[1:], end - 1])):
                option_ids[begin:stop] = [option_index] * (stop - begin)
                position_ids[begin:stop] = range(reads[0] - start, reads[0] - start + stop - begin)
        rows = [prompt.input_ids[:length], block_ids, option_ids, position_ids]
        input_ids, block_ids, option_ids, position_ids = torch.tensor(rows, device=model.device)[:, None]
        config, inputs_embeds = model.base_model.config, model.get_input_embeddings()(input_ids)
        options = option_mask_function(block_ids, option_ids)
        window = on_position_ids(sliding_window_bidirectional_overlay(config.sliding_window), position_ids)
        mask_kwargs = {"config": config, "inputs_embeds": inputs_embeds, "attention_mask": None}
        mask_functions = {"full_attention": options, "sliding_attention": and_masks(options, window)}
        attention_mask = {
            layer_type: create_bidirectional_mask(**mask_kwargs, and_mask_function=mask_function)
            for layer_type, mask_function in mask_functions.items()
        }
        if any(mask is None for mask in attention_mask.values()):
            raise ValueError(
                f"The masks of von cannot be expressed with `{config._attn_implementation}` attention, use `sdpa`, "
                "`eager` or `flex_attention`."
            )
        return {"inputs_embeds": inputs_embeds, "position_ids": position_ids, "attention_mask": attention_mask}

    def option_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.norm(F.gelu(self.dense(self.input_norm(hidden_states)))))[:, 0]

    @auto_docstring
    def forward(
        self, hidden_states: torch.Tensor, prompt: DecisionPrompt, lm_head: nn.Module | None = None, **kwargs
    ) -> list[torch.Tensor]:
        r"""
        hidden_states (`torch.Tensor` of shape `(sequence_length, hidden_size)`):
            Final hidden states of the backbone for one prompt.
        prompt (`DecisionPrompt`):
            The prompt: its input ids, the positions recorded by `<|mark|>` and its questions.
        lm_head (`nn.Module`, *optional*):
            The output embeddings of the backbone.
        """
        (question,), blocks = prompt.questions, self.blocks(prompt)
        logits = self.option_logits(hidden_states[blocks[0][1]])
        if len(blocks) == 2:  # zero-shot noul
            true, false = question.options.index("true"), question.options.index("false")
            null = self.option_logits(hidden_states[blocks[1][1]])
            correction = self.config.noul_prior["a"] * (null[true] - null[false]) + self.config.noul_prior["b"]
            is_true = torch.arange(len(logits), device=logits.device) == true
            logits = torch.where(is_true, logits - correction, logits)
        return [logits]


__all__ = ["VonDecisionHead", "VonPreTrainedModel"]
