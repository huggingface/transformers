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

from ...decision import DecisionPrompt
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
class VonDecisionHead(VonPreTrainedModel):
    def __init__(self, config: VonConfig):
        super().__init__(config)
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.norm = nn.LayerNorm(config.hidden_size // 2)
        self.out_proj = nn.Linear(config.hidden_size // 2, 1)
        self.post_init()

    def split_marks(self, prompt: DecisionPrompt) -> tuple[torch.Tensor, torch.Tensor]:
        """The `[MASK]` positions `(blocks, options)` and the end of each block."""
        marks = torch.tensor(prompt.marks).view(-1, len(prompt.questions[0].options) + 1)
        return marks[:, :-1] - 1, marks[:, -1].contiguous()

    def backbone_inputs(self, prompt: DecisionPrompt, model) -> dict[str, Any]:
        """
        Blocks are mutually invisible, options attend to the prompt of their block and to themselves, and the positions of
        every option restart after the prompt, also for the sliding window. The masking API makes the masks for the
        attention implementation of the backbone.
        """
        reads, ends = self.split_marks(prompt)
        positions = torch.arange(ends[-1])  # the state written after the last block is not part of the input
        block_ids = torch.searchsorted(ends, positions, right=True)
        starts, block_reads = torch.cat([ends.new_zeros(1), ends[:-1]])[block_ids], reads[block_ids]
        # The option of each token is the last `[MASK]` before it, the prompt (-1) before the first one and at the end
        option_ids = (block_reads <= positions[:, None]).sum(-1) - 1
        option_ids = option_ids.masked_fill(positions == ends[block_ids] - 1, -1)
        option_starts = block_reads.gather(1, option_ids.clamp(min=0)[:, None])[:, 0]
        prompt_lengths = block_reads[:, 0] - starts
        position_ids = torch.where(option_ids >= 0, prompt_lengths + positions - option_starts, positions - starts)

        rows = torch.stack([torch.tensor(prompt.input_ids[: len(positions)]), block_ids, option_ids, position_ids])
        input_ids, block_ids, option_ids, position_ids = rows.to(model.device)[:, None]
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
        reads, _ = self.split_marks(prompt)
        logits = self.out_proj(
            self.norm(F.gelu(self.dense(self.input_norm(hidden_states[reads.to(hidden_states.device)]))))
        )
        logits, scores = logits[..., 0], logits[0, :, 0]
        if len(logits) == 2:  # zero-shot noul: the scores of the block without the state debias the first block
            options = prompt.questions[0].options
            true, false = options.index("true"), options.index("false")
            correction = (
                self.config.noul_prior["a"] * (logits[1, true] - logits[1, false]) + self.config.noul_prior["b"]
            )
            scores = scores - correction * (torch.arange(len(scores), device=scores.device) == true)
        return [scores]


__all__ = ["VonDecisionHead", "VonPreTrainedModel"]
