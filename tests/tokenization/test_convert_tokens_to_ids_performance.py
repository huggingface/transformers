"""Tests for convert_tokens_to_ids performance regression.

Regression test for: https://github.com/huggingface/transformers/issues/46315
"""

import time

from transformers import BertTokenizerLegacy


class TestConvertTokensToIdsPerformance:
    """Test that convert_tokens_to_ids scales linearly with sequence length,
    not with the number of added tokens."""

    def test_convert_tokens_to_ids_performance_with_many_added_tokens(self):
        """Test that convert_tokens_to_ids is O(T) not O(T·N·logN).

        Before fix: runtime scales with number of added tokens (N)
        After fix: runtime scales only with sequence length (T)
        """
        tok = BertTokenizerLegacy.from_pretrained("google-bert/bert-base-chinese")

        # Add many tokens to simulate a model with large added vocabulary
        tok.add_tokens([chr(c) for c in range(0x4E00, 0x4E00 + 5000)])

        words = tok.tokenize("这是一个用于测试分词速度的中文句子。" * 20)

        # Warm up
        tok.convert_tokens_to_ids(words)

        # Time the conversion
        start = time.perf_counter()
        for _ in range(100):
            tok.convert_tokens_to_ids(words)
        elapsed = time.perf_counter() - start

        # With 5000 added tokens and ~400 tokens in the sequence,
        # the old O(T·N·logN) implementation would take >10s.
        # The fixed O(T) implementation should take <1s.
        assert elapsed < 2.0, (
            f"convert_tokens_to_ids took {elapsed:.2f}s, expected <2s. "
            f"This indicates the O(T·N·logN) performance regression."
        )

    def test_added_tokens_encoder_consistency(self):
        """Test that _added_tokens_encoder and added_tokens_encoder return same mapping."""
        tok = BertTokenizerLegacy.from_pretrained("google-bert/bert-base-chinese")
        tok.add_tokens(["custom_token_1", "custom_token_2"])

        # The cached dict should match the property
        assert tok._added_tokens_encoder == tok.added_tokens_encoder

        # Token lookup should work correctly
        assert tok._convert_token_to_id_with_added_voc("custom_token_1") == tok.added_tokens_encoder["custom_token_1"]
        assert tok._convert_token_to_id_with_added_voc("custom_token_2") == tok.added_tokens_encoder["custom_token_2"]
