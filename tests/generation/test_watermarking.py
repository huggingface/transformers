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

import numpy as np
from parameterized import parameterized

from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, WatermarkingConfig, is_torch_available
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

    def test_compute_pval(self):
        model_config = GPT2Config(vocab_size=128, bos_token_id=127, eos_token_id=127)
        detector = WatermarkDetector(model_config, torch_device, WatermarkingConfig())
        z_scores = np.array([-2.0, 0.0, 1.0, 2.0, 4.0, 6.0, 10.0])
        expected = [0.5 * math.erfc(z / math.sqrt(2)) for z in z_scores]
        np.testing.assert_allclose(detector._compute_pval(z_scores), expected, rtol=1e-12)

    def test_watermark_detector_null_calibration(self):
        """Unwatermarked random sequences should yield calibrated p-values and few false positives."""
        model_config = GPT2Config(vocab_size=128, bos_token_id=127, eos_token_id=127)
        watermark_config = WatermarkingConfig()
        detector = WatermarkDetector(model_config, "cpu", watermark_config)

        num_sequences = 500
        seq_length = 64
        rng = np.random.default_rng(50232)

        p_values = []
        predictions = []
        for _ in range(num_sequences):
            tokens = rng.integers(0, model_config.vocab_size - 1, size=seq_length, dtype=np.int64)
            input_ids = torch.tensor([tokens.tolist()], dtype=torch.long)
            result = detector(input_ids, z_threshold=3.0, return_dict=True)
            p_values.append(result.p_value[0])
            predictions.append(bool(result.prediction[0]))

        p_values = np.array(p_values)
        self.assertTrue(np.all((p_values >= 0.0) & (p_values <= 1.0)))
        self.assertAlmostEqual((p_values < 0.01).mean(), 0.01, delta=0.02)
        self.assertAlmostEqual((p_values < 0.05).mean(), 0.05, delta=0.04)
        self.assertLessEqual(np.mean(predictions), 0.02)

    def test_watermark_detector_detects_watermarked_generation(self):
        """Text generated with WatermarkingConfig on a tiny model should be flagged by the detector."""
        model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        prompt = "Hello"
        model_inputs = tokenizer(prompt, return_tensors="pt")
        input_len = model_inputs["input_ids"].shape[-1]

        watermark_config = WatermarkingConfig(bias=2.5, seeding_scheme="selfhash")
        torch.manual_seed(50232)
        with torch.no_grad():
            output = model.generate(
                **model_inputs,
                watermarking_config=watermark_config,
                do_sample=False,
                max_new_tokens=20,
            )

        detector = WatermarkDetector(model.config, "cpu", watermark_config)
        result = detector(output[:, input_len:], return_dict=True)

        self.assertTrue(result.prediction[0])
        self.assertLess(result.p_value[0], 0.01)
        self.assertGreater(result.z_score[0], 3.0)

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
