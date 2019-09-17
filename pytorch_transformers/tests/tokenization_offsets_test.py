from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

from pytorch_transformers.tokenization_offsets import whitespace_reduce, match_back_by_length, match_back_by_text


class TokenizeWithOffsetsTester(unittest.TestCase):
    def test_whitespace_reduce(self):
        offsets = np.array([[0, 3], [3, 7], [7, 9]])
        offsets_orig = np.array(offsets, copy=True)
        text = 'He saw it\n'
        whitespace_reduce(offsets, text)
        for i in range(offsets.shape[0]):
            self.assertEqual(text[offsets[i, 0]:offsets[i, 1]], text[offsets_orig[i, 0]:offsets_orig[i, 1]].strip())

    def test_match_back_by_length(self):
        chunk = 'capitalization'
        tokens = ['cap', 'it', 'aliza', 'tion']
        offsets = np.array([[0, len(chunk)]]*len(tokens))
        tlens = [len(t) for t in tokens]
        match_back_by_length(tlens, offsets, 0, len(tokens))
        for i in range(offsets.shape[0]):
            self.assertEqual(chunk[offsets[i, 0]:offsets[i, 1]], tokens[i])

    def test_match_back_by_text(self):
        chunk = 'capitalization'
        tokens = ['cap', 'it', 'aliza', 'tion']
        offsets = np.array([[0, len(chunk)]] * len(tokens))
        tlens = [len(t) for t in tokens]
        match_back_by_text(tokens, chunk, tlens, offsets, 0, len(tokens))
        for i in range(offsets.shape[0]):
            self.assertEqual(chunk[offsets[i, 0]:offsets[i, 1]], tokens[i])
