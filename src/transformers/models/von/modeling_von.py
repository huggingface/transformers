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
from typing import Any

import torch
from torch import nn

from ...masking_utils import (
    and_masks,
    create_bidirectional_mask,
    or_masks,
    packed_sequence_mask_function,
    padding_mask_function,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring
from ..auto import AutoModel
from .configuration_von import VonConfig


@auto_docstring
class VonPreTrainedModel(PreTrainedModel):
    config: VonConfig
    base_model_prefix = "model"
    _supports_sdpa = _supports_flash_attn = _supports_flex_attn = True


@auto_docstring(
    custom_intro="""
    von (wfzyx/von): the model of its repo and a decision head, each option is scored by an MLP on the hidden state of its `[MASK]` (the token before
    each `<|mark|>`, a last mark closing the block), with options that only attend to the prompt and to themselves. A
    zero-shot `noul` question has a second block without the state, whose scores debias the first, with `true` first as
    in its decision config. The state written once more after the last block is not given to the model, the pipeline
    counts its tokens.
    """
)
class VonForDecision(VonPreTrainedModel):
    def __init__(self, config: VonConfig):
        super().__init__(config)
        self.model = AutoModel.from_config(config.text_config)
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.norm = nn.LayerNorm(config.hidden_size // 2)
        self.out_proj = nn.Linear(config.hidden_size // 2, 1)
        self.post_init()

    def split_marks(self, marks: torch.LongTensor, num_options: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The `[MASK]` positions `(blocks, options)` and the end of each block, for the one question of the prompt."""
        marks = marks.view(-1, int(num_options[0]) + 1)
        return marks[:, :-1] - 1, marks[:, -1].contiguous()

    def backbone_inputs(
        self,
        input_ids: torch.LongTensor,
        marks: torch.LongTensor,
        num_options: torch.LongTensor,
    ) -> dict[str, Any]:
        """
        Blocks are mutually invisible, options attend to the prompt of their block and to themselves, and the positions of
        every option restart after the prompt, also for the sliding window. The masking API makes the masks for the
        attention implementation of the model.
        """
        reads, ends = self.split_marks(marks, num_options)
        # The state written after the last block is not part of the input
        positions = torch.arange(int(ends[-1]), device=marks.device)
        block_ids = torch.searchsorted(ends, positions, right=True)
        block_reads = reads[block_ids]
        # The option of each token is the last `[MASK]` before it, the prompt (-1) before the first one and at the end
        option_ids = (block_reads <= positions[:, None]).sum(-1) - 1
        option_ids = option_ids.masked_fill(positions == ends[block_ids] - 1, -1)
        # Positions restart at each block, and every option starts where the first one does
        option_starts = block_reads.gather(1, option_ids.clamp(min=0)[:, None])[:, 0]
        block_starts = torch.cat([ends.new_zeros(1), ends[:-1]])[block_ids]
        position_ids = positions - block_starts - (option_starts - block_reads[:, 0])

        rows = torch.stack([input_ids[: len(positions)], block_ids, option_ids, position_ids])
        input_ids, block_ids, option_ids, position_ids = rows[:, None]
        config, inputs_embeds = self.config.text_config, self.model.get_input_embeddings()(input_ids)
        options = and_masks(
            packed_sequence_mask_function(block_ids),
            or_masks(packed_sequence_mask_function(option_ids), padding_mask_function(option_ids == -1)),
        )

        def sliding_window(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
            return abs(position_ids[batch_idx, q_idx] - position_ids[batch_idx, kv_idx]) <= config.sliding_window

        mask_kwargs = {"config": config, "inputs_embeds": inputs_embeds, "attention_mask": None}
        attention_mask = {
            "full_attention": create_bidirectional_mask(**mask_kwargs, and_mask_function=options),
            "sliding_attention": create_bidirectional_mask(
                **mask_kwargs, and_mask_function=and_masks(options, sliding_window)
            ),
        }
        if any(mask is None for mask in attention_mask.values()):
            raise ValueError(
                f"The masks of von cannot be expressed with `{config._attn_implementation}` attention, use `sdpa`, "
                "`eager` or `flex_attention`."
            )
        return {"inputs_embeds": inputs_embeds, "position_ids": position_ids, "attention_mask": attention_mask}

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        marks: torch.LongTensor,
        num_options: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        backbone_inputs = self.backbone_inputs(input_ids, marks, num_options)
        hidden_states = self.model(**backbone_inputs, **kwargs).last_hidden_state[0]
        hidden_states = hidden_states.to(self.out_proj.weight.dtype)
        reads, _ = self.split_marks(marks, num_options)
        hidden_states = self.input_norm(hidden_states[reads])
        logits = self.out_proj(self.norm(nn.functional.gelu(self.dense(hidden_states))))[..., 0]
        scores = logits[0]
        if len(logits) == 2:  # zero-shot noul: the block without the state lowers the score of `true`
            prior = self.config.noul_prior
            correction = prior["a"] * (logits[1, 0] - logits[1, 1]) + prior["b"]
            scores = torch.stack([scores[0] - correction, scores[1]])
        return scores


__all__ = ["VonForDecision", "VonPreTrainedModel"]
