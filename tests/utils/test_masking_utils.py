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

from transformers.testing_utils import (
    cleanup,
    is_torch_available,
    require_torch,
    torch_device,
)


if is_torch_available():
    import torch
    from torch.nn.attention.flex_attention import create_block_mask

    from transformers import DynamicCache, LlamaConfig
    from transformers.cache_utils import DynamicSlidingWindowLayer
    from transformers.masking_utils import (
        create_bidirectional_mask,
        create_causal_mask,
        create_chunked_causal_mask,
        find_packed_sequence_indices,
    )


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
    def setup(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

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

    def test_chunked_mask_with_left_padding_and_large_prefill(self):
        # Make sure we have an attention_chunk_size in the config
        config = LlamaConfig(attention_chunk_size=3, attn_implementation="sdpa")

        batch_size = 2
        sequence_length = 8
        pad_tokens = 4

        input_ids = torch.randint(100, 200, (batch_size, sequence_length))
        attention_mask = torch.tensor(
            [[0 if i < pad_tokens else 1 for i in range(sequence_length)], [1] * sequence_length]
        )
        inputs_embeds = torch.empty_like(input_ids, dtype=torch.float16)
        cache_position = torch.arange(sequence_length)
        position_ids = torch.empty(batch_size, sequence_length, dtype=cache_position.dtype)
        position_ids[0, :pad_tokens] = 1
        position_ids[0, pad_tokens:] = torch.arange(sequence_length - pad_tokens)
        position_ids[1, :] = cache_position

        chunked_attention_mask = create_chunked_causal_mask(
            config=config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        # fmt: off
        EXPECTED_CHUNKED_MASK = torch.tensor(
            # Here, for the padded sequence, the chunk size should start correctly at index 4 (otherwise, with 4 padding
            # tokens are chunk_size=3, the first chunk is from indices 0-2, then 3-6 if we don't account for the padding correctly)
            [[[[False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False,  True, False, False, False],
                [False, False, False, False,  True,  True, False, False],
                [False, False, False, False,  True,  True,  True, False],
                [False, False, False, False, False, False, False,  True]]],


            [[[ True, False, False, False, False, False, False, False],
                [ True,  True, False, False, False, False, False, False],
                [ True,  True,  True, False, False, False, False, False],
                [False, False, False,  True, False, False, False, False],
                [False, False, False,  True,  True, False, False, False],
                [False, False, False,  True,  True,  True, False, False],
                [False, False, False, False, False, False,  True, False],
                [False, False, False, False, False, False,  True,  True]]]],
            dtype=torch.bool)
        # fmt: on

        self.assertTrue((chunked_attention_mask == EXPECTED_CHUNKED_MASK).all())

    def test_chunked_mask_with_left_padding_decoding(self):
        # Make sure we have an attention_chunk_size in the config
        config = LlamaConfig(attention_chunk_size=4, attn_implementation="sdpa", num_hidden_layers=1)

        cache = DynamicCache(config=config)
        # Sanity check
        self.assertEqual(len(cache), 1)
        self.assertTrue(isinstance(cache.layers[0], DynamicSlidingWindowLayer))

        # Fill-in the Cache (sequence length is bigger than chunk size here)
        batch_size = 2
        prefill_size = 8
        pad_tokens = 7
        fake_kv = torch.rand(batch_size, 32, prefill_size, 32)
        cache.update(fake_kv, fake_kv, 0, torch.arange(prefill_size))

        # Create a new input after the prefill
        input_ids = torch.randint(100, 200, (batch_size, 1))
        attention_mask = torch.tensor(
            [[0 if i < pad_tokens else 1 for i in range(prefill_size + 1)], [1] * (prefill_size + 1)]
        )
        inputs_embeds = torch.empty_like(input_ids, dtype=torch.float16)
        cache_position = torch.tensor([prefill_size], dtype=int)
        position_ids = torch.tensor([[prefill_size - pad_tokens], [prefill_size]])

        chunked_attention_mask = create_chunked_causal_mask(
            config=config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=cache,
            position_ids=position_ids,
        )

        # To understand a bit more the following expected mask, here is the full 2d mask, where the "|" characters are the chunk
        # separators (where the tokens should stop seeing each other)
        # [0, 0, 0, 0, 0, 0, 0, | 1, 1],    -> due to left padding, the first chunk only starts after the padding tokens
        # [| 1, 1, 1, 1, | 1, 1, 1, 1, | 1]])  -> easy case, each 4 tokens is a new chunk

        # fmt: off
        EXPECTED_CHUNKED_MASK = torch.tensor(
            # Here, for the padded sequence, the chunk size should start correctly at index 7 (the first unpadded
            # index), and so only indices 7 and 8 should be True
            [[[[False, False,  True,  True]]],

            # Here, for the unpadded sequence, the chunks start at index 0. Since we have 9 tokens in total, the last
            # token (index 8) will only see itself (we have 2 full chunks before)
            [[[False, False, False,  True]]]],
            dtype=torch.bool)
        # fmt: on

        self.assertTrue((chunked_attention_mask == EXPECTED_CHUNKED_MASK).all())

    @staticmethod
    def _run_bidirectional_mask(mask_fn, attn_implementation):
        def run_mask_creation(mask_fn, config, input_embeds, encoder_mask, cross_mask, encoder_hidden_states):
            encoder_attn_mask = mask_fn(
                config=config,
                input_embeds=input_embeds,
                attention_mask=encoder_mask,
            )
            cross_attn_mask = mask_fn(
                config=config,
                input_embeds=input_embeds,
                attention_mask=cross_mask,
                encoder_hidden_states=encoder_hidden_states,
            )
            return encoder_attn_mask, cross_attn_mask

        # We use llama but could be also bert/bart --> we only need the `_attn_implementation` here
        config = LlamaConfig()
        config._attn_implementation = attn_implementation

        # Meta data
        batch_size = 2
        q_length = 10
        kv_length = 5

        input_embeds = torch.ones((batch_size, q_length, 1), device=torch_device, dtype=torch.float16)
        encoder_hidden_states = torch.ones((batch_size, kv_length, 1), device=torch_device, dtype=torch.float16)

        encoder_mask = torch.ones_like(input_embeds)[..., 0]
        cross_mask = torch.ones_like(encoder_hidden_states)[..., 0]

        # Case 1: Full mask
        full_mask_encoder_1, full_mask_cross_1 = run_mask_creation(
            mask_fn=mask_fn,
            config=config,
            input_embeds=input_embeds,
            encoder_mask=encoder_mask,
            cross_mask=cross_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        full_mask_encoder_2, full_mask_cross_2 = run_mask_creation(
            mask_fn=mask_fn,
            config=config,
            input_embeds=input_embeds,
            encoder_mask=None,
            cross_mask=None,
            encoder_hidden_states=encoder_hidden_states,
        )

        # Case 2: Padding involved
        cross_mask[:, -1] = 0
        encoder_mask[:, -1] = 0

        padded_mask_encoder, padded_mask_cross = run_mask_creation(
            mask_fn=mask_fn,
            config=config,
            input_embeds=input_embeds,
            encoder_mask=encoder_mask,
            cross_mask=cross_mask,
            encoder_hidden_states=encoder_hidden_states,
        )

        full_masks = (full_mask_encoder_1, full_mask_encoder_2), (full_mask_cross_1, full_mask_cross_2)
        padded_masks = (padded_mask_encoder, padded_mask_cross)
        return full_masks, padded_masks

    def test_bidirectional_mask_cudagraphs(self):
        """
        Checks whether the bidirectional mask creation is compatible with cuda graphs, i.e. we do not into any error
        during this test.
        """
        mask_creation_function = torch.compile(create_bidirectional_mask, mode="reduce-overhead")
        self._run_bidirectional_mask(mask_fn=mask_creation_function, attn_implementation="sdpa")

    def test_bidirectional_mask_skip_eager(self):
        """
        Checks whether the bidirectional mask creation can skip the mask creation if we have a full mask.
        """
        full_masks, padded_mask = self._run_bidirectional_mask(
            mask_fn=create_bidirectional_mask, attn_implementation="eager"
        )

        for alternative_masks in full_masks:
            self.assertTrue(alternative_masks[0] is None)
            self.assertTrue(alternative_masks[1] is None)

        self.assertTrue(padded_mask[0] is not None)
        self.assertTrue(padded_mask[1] is not None)
