# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import numpy as np
from parameterized import parameterized

from transformers import AutoProcessor, DacFeatureExtractor, VoxCPM2Processor, VoxCPM2Tokenizer
from transformers.testing_utils import require_torch

from ...test_processing_common import ProcessorTesterMixin


def get_tiny_voxcpm2_processor() -> VoxCPM2Processor:
    return VoxCPM2ProcessorTest._setup_from_components()


@require_torch
class VoxCPM2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = VoxCPM2Processor
    audio_input_name = "prompt_input_values"

    @classmethod
    def _setup_tokenizer(cls):
        vocab = {
            "<unk>": 0,
            "<s>": 1,
            "</s>": 2,
            "<pad>": 3,
            "▁": 4,
            "你": 5,
            "好": 6,
            "你好": 7,
            "▁你好": 8,
            "A": 9,
            "B": 10,
            "AB": 11,
            "▁AB": 12,
        }
        tokenizer = VoxCPM2Tokenizer(
            vocab=vocab,
            merges=[("你", "好"), ("▁", "你好"), ("A", "B"), ("▁", "AB")],
            pad_token="<pad>",
        )
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<|audio_start|>",
                    "<|audio_prompt_start|>",
                    "<|audio_prompt_end|>",
                ]
            }
        )
        return tokenizer

    @classmethod
    def _setup_feature_extractor(cls):
        return DacFeatureExtractor(sampling_rate=16000, hop_length=1, return_attention_mask=True)

    @staticmethod
    def prepare_processor_dict():
        return {"audio_patch_size": 4}

    def maybe_skip_typed_test_for_modality(self, modality, attributes):
        super().maybe_skip_typed_test_for_modality(modality, attributes)
        self.skipTest("VoxCPM2 takes explicit generation arguments rather than typed processor kwargs.")

    @unittest.skip("VoxCPM2 takes explicit generation arguments and does not accept text_kwargs.")
    def test_flat_kwarg_applied_when_modality_dict_lacks_it(self):
        pass

    @parameterized.expand([("text",), ("images",), ("videos",), ("audio",)])
    def test_subprocessor_defaults(self, modality):
        processor = self.get_processor()
        if modality == "text":
            text_ids = processor.tokenizer("A", add_special_tokens=False).input_ids
            inputs = processor(text="A", return_tensors="np")
            self.assertListEqual(inputs.input_ids.tolist(), [text_ids + [processor.audio_start_token_id]])
            self.assertTrue(inputs.text_mask.all())
            self.assertFalse(inputs.audio_mask.any())
        elif modality == "audio":
            processor.feature_extractor.hop_length = 4
            prompt_audio = np.arange(5, dtype=np.float32)
            reference_audio = np.arange(3, dtype=np.float32)
            inputs = processor(
                text="A",
                audio=prompt_audio,
                prompt_text="B",
                reference_audio=reference_audio,
                sampling_rate=16000,
                return_tensors="np",
            )
            for prefix, audio in (("prompt", prompt_audio), ("reference", reference_audio)):
                with self.subTest(prefix=prefix):
                    expected = processor.feature_extractor(
                        audio, padding=True, sampling_rate=16000, return_tensors="np"
                    )
                    np.testing.assert_array_equal(inputs[f"{prefix}_input_values"], expected.input_values)
                    np.testing.assert_array_equal(inputs[f"{prefix}_attention_mask"], expected.padding_mask)
            self.assertEqual(inputs.prompt_input_values.shape, (1, 1, 8))
            self.assertEqual(inputs.reference_input_values.shape, (1, 1, 4))
            self.assertEqual(inputs.prompt_attention_mask.sum(), 5)
            self.assertEqual(inputs.reference_attention_mask.sum(), 3)
            self.assertEqual(inputs.audio_mask.sum(), 3)
        else:
            self.skipTest("VoxCPM2 processes text and audio.")

    def test_processor_with_multiple_inputs(self):
        processor = self.get_processor()
        prompt_audio = np.arange(5, dtype=np.float32)
        reference_audio = np.arange(3, dtype=np.float32)
        target_ids = processor.tokenizer("A", add_special_tokens=False).input_ids
        continuation_ids = processor.tokenizer("BA", add_special_tokens=False).input_ids
        audio_start = processor.audio_start_token_id
        reference_prefix = [processor.reference_audio_start_token_id, 0, processor.reference_audio_end_token_id]

        cases = (
            (
                "continuation",
                {"audio": prompt_audio, "prompt_text": "B"},
                continuation_ids + [audio_start, 0, 0],
                [0] * (len(continuation_ids) + 1) + [1, 1],
            ),
            (
                "reference",
                {"reference_audio": reference_audio},
                reference_prefix + target_ids + [audio_start],
                [0, 1, 0] + [0] * (len(target_ids) + 1),
            ),
            (
                "combined",
                {"audio": prompt_audio, "prompt_text": "B", "reference_audio": reference_audio},
                reference_prefix + continuation_ids + [audio_start, 0, 0],
                [0, 1, 0] + [0] * (len(continuation_ids) + 1) + [1, 1],
            ),
        )
        for name, kwargs, expected_ids, expected_audio_mask in cases:
            with self.subTest(mode=name):
                inputs = processor(text="A", sampling_rate=16000, return_tensors="pt", **kwargs)
                self.assertListEqual(inputs.input_ids.tolist(), [expected_ids])
                self.assertListEqual(inputs.audio_mask.tolist(), [expected_audio_mask])
                self.assertListEqual(inputs.text_mask.tolist(), [[1 - value for value in expected_audio_mask]])
                self.assertTrue(inputs.attention_mask.bool().all())

        with self.assertRaises((TypeError, ValueError)):
            processor()

    def test_model_input_names(self):
        processor = self.get_processor()
        inputs = processor(
            text="A",
            audio=np.arange(5, dtype=np.float32),
            prompt_text="B",
            reference_audio=np.arange(3, dtype=np.float32),
            sampling_rate=16000,
            return_tensors="pt",
        )
        self.assertSetEqual(set(inputs), set(processor.model_input_names))

    def test_singleton_text_batch(self):
        processor = self.get_processor()
        string_inputs = processor(text="A", return_tensors="pt")
        list_inputs = processor(text=["A"], return_tensors="pt")
        for input_name in ("input_ids", "attention_mask", "text_mask", "audio_mask"):
            np.testing.assert_array_equal(string_inputs[input_name], list_inputs[input_name])

        with self.assertRaisesRegex(TypeError, "batch size 1"):
            processor(text=["A", "B"], return_tensors="pt")

    def test_invalid_audio_inputs(self):
        processor = self.get_processor()
        audio = np.arange(5, dtype=np.float32)
        cases = (
            ({"audio": audio}, "provided together"),
            ({"prompt_text": "B"}, "provided together"),
            ({"audio": [audio, audio], "prompt_text": "B"}, "one audio sample"),
            ({"audio": np.zeros((2, 5), dtype=np.float32), "prompt_text": "B"}, "Expected mono audio"),
        )
        for kwargs, message in cases:
            with self.subTest(inputs=kwargs), self.assertRaisesRegex(ValueError, message):
                processor(text="A", sampling_rate=16000, **kwargs)

    def test_auto_processor(self):
        self.assertIsInstance(AutoProcessor.from_pretrained(self.tmpdirname), self.processor_class)
