# Copyright 2026 the HuggingFace Team. All rights reserved.
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
