import unittest

import torch

from transformers.models.unlimited_ocr.generation_unlimited_ocr import (
    UnlimitedOcrSlidingWindowNoRepeatNgramLogitsProcessor,
)
from transformers.testing_utils import require_torch, torch_device


@require_torch
class UnlimitedOcrSlidingWindowNoRepeatNgramLogitsProcessorTest(unittest.TestCase):
    def test_window_limits_ngram_lookup(self):
        vocab_size = 3
        # The (0, 1) bigram appears at the start, so a full-sequence processor would forbid token 1
        # after the trailing 0. A small window should not see that early bigram.
        input_ids = torch.tensor([[0, 1, 2, 0]], device=torch_device, dtype=torch.long)
        scores = torch.zeros((1, vocab_size), device=torch_device, dtype=torch.float)

        small_window = UnlimitedOcrSlidingWindowNoRepeatNgramLogitsProcessor(ngram_size=2, window_size=2)
        full_window = UnlimitedOcrSlidingWindowNoRepeatNgramLogitsProcessor(ngram_size=2, window_size=4)

        self.assertListEqual(torch.isinf(small_window(input_ids, scores.clone())).tolist(), [[False, False, False]])
        self.assertListEqual(torch.isinf(full_window(input_ids, scores.clone())).tolist(), [[False, True, False]])
