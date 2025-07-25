# Copyright 2025 HuggingFace Inc.
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

import unittest

from transformers.testing_utils import is_torch_available, require_torch


if is_torch_available():
    import torch
    from torch.nn.attention.flex_attention import create_block_mask

    from transformers import LlamaConfig
    from transformers.masking_utils import create_causal_mask, find_packed_sequence_indices


# fmt: off
EXPECTED_PACKED_MASK = torch.tensor([[[
    [ True, False, False, False, False, False, False, False, False, False],
    [ True,  True, False, False, False, False, False, False, False, False],
    [ True,  True,  True, False, False, False, False, False, False, False],
    [ True,  True,  True,  True, False, False, False, False, False, False],
    [False, False, False, False,  True, False, False, False, False, False],
    [False, False, False, False,  True,  True, False, False, False, False],
    [False, False, False, False, False, False,  True, False, False, False],
    [False, False, False, False, False, False,  True,  True, False, False],
    [False, False, False, False, False, False,  True,  True,  True, False],
    [False, False, False, False, False, False,  True,  True,  True,  True]]],


  [[[ True, False, False, False, False, False, False, False, False, False],
    [ True,  True, False, False, False, False, False, False, False, False],
    [ True,  True,  True, False, False, False, False, False, False, False],
    [ True,  True,  True,  True, False, False, False, False, False, False],
    [ True,  True,  True,  True,  True, False, False, False, False, False],
    [ True,  True,  True,  True,  True,  True, False, False, False, False],
    [False, False, False, False, False, False,  True, False, False, False],
    [False, False, False, False, False, False,  True,  True, False, False],
    [False, False, False, False, False, False,  True,  True,  True, False],
    [False, False, False, False, False, False,  True,  True,  True,  True]
]]], dtype=torch.bool)
# fmt: on


@require_torch
class MaskTest(unittest.TestCase):
    def test_packed_sequence_mask_sdpa(self):
        config = LlamaConfig()
        config._attn_implementation = "sdpa"

        batch_size = 2
        sequence_length = 10
        cache_position = torch.arange(sequence_length)

        # First batch has 3 packed sequences of 4, 2 and 4 tokens respectively, second has 2 of 6 and 4 tokens
        position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]])

        causal_mask = create_causal_mask(
            config=config,
            # we only need batch size, seq_length and dtype here - we don't care about the values of the embeddings
            input_embeds=torch.empty((batch_size, sequence_length), dtype=torch.float16),
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        self.assertTrue((causal_mask == EXPECTED_PACKED_MASK).all())

    def test_packed_sequence_mask_eager(self):
        config = LlamaConfig()
        config._attn_implementation = "eager"

        batch_size = 2
        sequence_length = 10
        cache_position = torch.arange(sequence_length)

        # First batch has 3 packed sequences of 4, 2 and 4 tokens respectively, second has 2 of 6 and 4 tokens
        position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]])

        causal_mask = create_causal_mask(
            config=config,
            # we only need batch size, seq_length and dtype here - we don't care about the values of the embeddings
            input_embeds=torch.empty((batch_size, sequence_length), dtype=torch.float16),
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        min_dtype = torch.finfo(torch.float16).min
        self.assertTrue((causal_mask == torch.where(EXPECTED_PACKED_MASK, 0.0, min_dtype)).all())

    def test_packed_sequence_mask_flex_attention(self):
        config = LlamaConfig()
        config._attn_implementation = "flex_attention"

        batch_size = 2
        sequence_length = 10
        cache_position = torch.arange(sequence_length)

        # First batch has 3 packed sequences of 4, 2 and 4 tokens respectively, second has 2 of 6 and 4 tokens
        position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]])

        causal_mask = create_causal_mask(
            config=config,
            # we only need batch size, seq_length and dtype here - we don't care about the values of the embeddings
            input_embeds=torch.empty((batch_size, sequence_length), dtype=torch.float16),
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        def dummy_mask_mod(b, h, q, kv):
            return EXPECTED_PACKED_MASK[b, h, q, kv]

        EXPECTED_BLOCK_MASK = create_block_mask(dummy_mask_mod, 2, None, 10, 10, device="cpu")

        # We compatre the str representations, as the BlockMask objects themselves cannot easily be compared
        self.assertEqual(causal_mask.to_string(), EXPECTED_BLOCK_MASK.to_string())

    def test_find_packed_sequence_indices(self):
        position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]])
        EXPECTED_SEQUENCE_INDICES = torch.tensor([[0, 0, 0, 0, 1, 1, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])
        self.assertTrue((find_packed_sequence_indices(position_ids) == EXPECTED_SEQUENCE_INDICES).all())
