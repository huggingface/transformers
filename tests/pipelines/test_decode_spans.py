"""Regression tests for decode_spans in document_question_answering pipeline.

See https://github.com/huggingface/transformers/issues/44327
"""

import unittest

import numpy as np

from transformers.pipelines.document_question_answering import decode_spans


class DecodeSpansTests(unittest.TestCase):
    def test_topk_equals_scores_length_no_crash(self):
        """When len(scores_flat) == topk, argpartition must not be called with kth == len(arr).

        Before fix: ValueError: kth(=100) out of bounds (100)
        """
        seq_len = 10  # 10^2 = 100 elements
        topk = 100
        max_answer_len = 15
        np.random.seed(42)
        start = np.random.rand(1, seq_len)
        end = np.random.rand(1, seq_len)
        undesired_tokens = np.ones(seq_len, dtype=bool)

        # Should not raise ValueError
        starts, ends, scores = decode_spans(start, end, topk, max_answer_len, undesired_tokens)
        self.assertGreater(len(starts), 0)

    def test_topk_greater_than_scores_length(self):
        """topk > len(scores_flat) should use argsort path."""
        seq_len = 3  # 9 elements, topk=100
        topk = 100
        max_answer_len = 15
        np.random.seed(42)
        start = np.random.rand(1, seq_len)
        end = np.random.rand(1, seq_len)
        undesired_tokens = np.ones(seq_len, dtype=bool)

        starts, ends, scores = decode_spans(start, end, topk, max_answer_len, undesired_tokens)
        self.assertGreater(len(starts), 0)

    def test_topk_one(self):
        """topk=1 uses argmax path."""
        seq_len = 5
        topk = 1
        max_answer_len = 15
        np.random.seed(42)
        start = np.random.rand(1, seq_len)
        end = np.random.rand(1, seq_len)
        undesired_tokens = np.ones(seq_len, dtype=bool)

        starts, ends, scores = decode_spans(start, end, topk, max_answer_len, undesired_tokens)
        self.assertGreater(len(starts), 0)

    def test_topk_less_than_scores_length(self):
        """Normal case: topk < len(scores_flat), uses argpartition."""
        seq_len = 20  # 400 elements
        topk = 10
        max_answer_len = 15
        np.random.seed(42)
        start = np.random.rand(1, seq_len)
        end = np.random.rand(1, seq_len)
        undesired_tokens = np.ones(seq_len, dtype=bool)

        starts, ends, scores = decode_spans(start, end, topk, max_answer_len, undesired_tokens)
        self.assertGreater(len(starts), 0)


if __name__ == "__main__":
    unittest.main()
