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
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import WatermarkDetector
    from transformers.generation import WatermarkLogitsProcessor


@require_torch
class WatermarkDetectorTest(unittest.TestCase):
    @parameterized.expand(
        [
            (
                f"{seeding_scheme}_width_{context_width}_{input_name}_ignore_repeated_{ignore_repeated_ngrams}",
                seeding_scheme,
                context_width,
                tokens,
                ignore_repeated_ngrams,
                num_unique if ignore_repeated_ngrams else num_windows,
            )
            for seeding_scheme, context_width, num_windows in [
                ("lefthash", 1, 9),
                ("lefthash", 2, 8),
                ("selfhash", 1, 10),
                ("selfhash", 2, 9),
            ]
            for input_name, tokens, num_unique in [
                ("identical", [3] * 10, 1),
                ("alternating", [1, 2] * 5, 2),
                ("distinct", list(range(1, 11)), num_windows),
            ]
            for ignore_repeated_ngrams in (True, False)
        ]
    )
    def test_num_tokens_scored(self, name, seeding_scheme, context_width, tokens, ignore_repeated_ngrams, table_count):
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
            device="cpu",
            watermarking_config=watermark_config,
            ignore_repeated_ngrams=ignore_repeated_ngrams,
        )
        input_ids = torch.tensor([tokens], dtype=torch.long, device="cpu")
        z_threshold = 3.0
        result = detector(input_ids, z_threshold=z_threshold, return_dict=True)

        self.assertEqual(result.num_tokens_scored[0], expected_count)

        if len(set(tokens)) == 1:
            # Use a separate processor for green membership; do not assume token 3 is always green.
            reference_processor = WatermarkLogitsProcessor(
                vocab_size=model_config.vocab_size, device="cpu", **watermark_config.to_dict()
            )
            prefix_values = ngrams[0][:-1] if seeding_scheme == "lefthash" else ngrams[0]
            prefix = torch.tensor(prefix_values, dtype=torch.long, device="cpu")
            greenlist_ids = reference_processor._get_greenlist_ids(prefix)
            expected_green_count = expected_count if ngrams[0][-1] in greenlist_ids else 0
            if seeding_scheme == "lefthash" and context_width == 1:
                self.assertEqual(expected_green_count, expected_count)
            self.assertEqual(result.num_green_tokens[0], expected_green_count)

            ratio = watermark_config.greenlist_ratio
            expected_z_score = (expected_green_count - ratio * expected_count) / math.sqrt(
                expected_count * ratio * (1 - ratio)
            )
            self.assertAlmostEqual(result.z_score[0], expected_z_score, places=7)
            self.assertEqual(bool(result.prediction[0]), expected_z_score > z_threshold)
