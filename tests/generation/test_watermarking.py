# Copyright 2026 The HuggingFace Team Inc.
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

import math
import unittest

from parameterized import parameterized

from transformers import GPT2Config, WatermarkingConfig, is_torch_available
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch

    from transformers import WatermarkDetector
    from transformers.generation import WatermarkLogitsProcessor


@require_torch
class WatermarkDetectorTest(unittest.TestCase):
    @parameterized.expand(
        [
            (
                f"width_{context_width}_ignore_repeated_{ignore_repeated_ngrams}",
                context_width,
                ignore_repeated_ngrams,
                1 if ignore_repeated_ngrams else num_windows,
            )
            for context_width, num_windows in [(1, 9), (2, 8)]
            for ignore_repeated_ngrams in (True, False)
        ]
    )
    def test_identical_tokens_lefthash(self, name, context_width, ignore_repeated_ngrams, expected_count):
        self._check_identical_tokens("lefthash", context_width, ignore_repeated_ngrams, expected_count)

    @parameterized.expand(
        [
            (
                f"width_{context_width}_ignore_repeated_{ignore_repeated_ngrams}",
                context_width,
                ignore_repeated_ngrams,
                1 if ignore_repeated_ngrams else num_windows,
            )
            for context_width, num_windows in [(1, 10), (2, 9)]
            for ignore_repeated_ngrams in (True, False)
        ]
    )
    def test_identical_tokens_selfhash(self, name, context_width, ignore_repeated_ngrams, expected_count):
        self._check_identical_tokens("selfhash", context_width, ignore_repeated_ngrams, expected_count)

    @parameterized.expand(
        [
            (
                f"width_{context_width}_{input_name}_ignore_repeated_{ignore_repeated_ngrams}",
                context_width,
                tokens,
                ignore_repeated_ngrams,
                num_unique if ignore_repeated_ngrams else num_windows,
            )
            for context_width, num_windows in [(1, 9), (2, 8)]
            for input_name, tokens, num_unique in [
                ("alternating", [1, 2] * 5, 2),
                ("distinct", list(range(1, 11)), num_windows),
            ]
            for ignore_repeated_ngrams in (True, False)
        ]
    )
    def test_num_tokens_scored_lefthash(self, name, context_width, tokens, ignore_repeated_ngrams, table_count):
        self._check_num_tokens_scored("lefthash", context_width, tokens, ignore_repeated_ngrams, table_count)

    @parameterized.expand(
        [
            (
                f"width_{context_width}_{input_name}_ignore_repeated_{ignore_repeated_ngrams}",
                context_width,
                tokens,
                ignore_repeated_ngrams,
                num_unique if ignore_repeated_ngrams else num_windows,
            )
            for context_width, num_windows in [(1, 10), (2, 9)]
            for input_name, tokens, num_unique in [
                ("alternating", [1, 2] * 5, 2),
                ("distinct", list(range(1, 11)), num_windows),
            ]
            for ignore_repeated_ngrams in (True, False)
        ]
    )
    def test_num_tokens_scored_selfhash(self, name, context_width, tokens, ignore_repeated_ngrams, table_count):
        self._check_num_tokens_scored("selfhash", context_width, tokens, ignore_repeated_ngrams, table_count)

    def _check_identical_tokens(self, seeding_scheme, context_width, ignore_repeated_ngrams, expected_count):
        result = self._check_num_tokens_scored(
            seeding_scheme, context_width, [3] * 10, ignore_repeated_ngrams, expected_count
        )

        # Use a separate processor for green membership; the greenlist depends on the device.
        watermark_config = WatermarkingConfig(
            greenlist_ratio=0.25, seeding_scheme=seeding_scheme, context_width=context_width
        )
        reference_processor = WatermarkLogitsProcessor(
            vocab_size=128, device=torch_device, **watermark_config.to_dict()
        )
        prefix = torch.tensor([3] * context_width, dtype=torch.long, device=torch_device)
        greenlist_ids = reference_processor._get_greenlist_ids(prefix)
        expected_green_count = expected_count if 3 in greenlist_ids else 0
        self.assertEqual(result.num_green_tokens[0], expected_green_count)

        ratio = watermark_config.greenlist_ratio
        expected_z_score = (expected_green_count - ratio * expected_count) / math.sqrt(
            expected_count * ratio * (1 - ratio)
        )
        self.assertAlmostEqual(result.z_score[0], expected_z_score, places=7)
        self.assertEqual(bool(result.prediction[0]), expected_z_score > 3.0)

    def _check_num_tokens_scored(self, seeding_scheme, context_width, tokens, ignore_repeated_ngrams, table_count):
        # Count Python integer windows independently of the detector.
        ngram_length = context_width + 1 if seeding_scheme == "lefthash" else context_width
        ngrams = [tuple(tokens[start : start + ngram_length]) for start in range(len(tokens) - ngram_length + 1)]
        expected_count = len(set(ngrams)) if ignore_repeated_ngrams else len(ngrams)
        self.assertEqual(expected_count, table_count)

        model_config = GPT2Config(vocab_size=128, bos_token_id=127, eos_token_id=127)
        watermark_config = WatermarkingConfig(
            greenlist_ratio=0.25, seeding_scheme=seeding_scheme, context_width=context_width
        )
        detector = WatermarkDetector(
            model_config=model_config,
            device=torch_device,
            watermarking_config=watermark_config,
            ignore_repeated_ngrams=ignore_repeated_ngrams,
        )
        input_ids = torch.tensor([tokens], dtype=torch.long, device=torch_device)
        result = detector(input_ids, z_threshold=3.0, return_dict=True)

        self.assertEqual(result.num_tokens_scored[0], expected_count)

        return result
