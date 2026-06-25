# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from transformers import AutoProcessor
from transformers.testing_utils import require_mistral_common, require_torch
from transformers.utils import is_soundfile_available, is_torch_available


if is_torch_available():
    import torch


@require_mistral_common
@require_torch
@unittest.skipUnless(is_soundfile_available(), "test requires soundfile")
class VoxtralRealtimeProcessorTest(unittest.TestCase):
    checkpoint_name = "mistralai/Voxtral-Mini-4B-Realtime-2602"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint_name)

    def _dummy_audio(self, seed: int = 0, duration_s: float = 1.0):
        sampling_rate = self.processor.feature_extractor.sampling_rate
        rng = np.random.default_rng(seed)
        num_samples = int(duration_s * sampling_rate)
        return (rng.standard_normal(num_samples) * 0.1).clip(-1.0, 1.0).astype(np.float32)

    def _assert_streaming_prefill_input_ids(self, input_ids):
        # Prefill prompt = BOS + [STREAMING_PAD] placeholders (left padding + delay tokens).
        processor = self.processor
        bos_token_id = processor.tokenizer.bos_token_id
        streaming_pad_id = processor.tokenizer.convert_tokens_to_ids("[STREAMING_PAD]")

        self.assertEqual(input_ids.shape[0], 1)
        ids = input_ids[0].tolist()

        self.assertEqual(ids[0], bos_token_id)
        self.assertEqual(set(ids[1:]), {streaming_pad_id})

        expected_num_placeholders = (
            processor.mistral_common_audio_config.streaming_n_left_pad_tokens + processor.num_delay_tokens
        )
        self.assertEqual(len(ids) - 1, expected_num_placeholders)

    def test_offline_call(self):
        audio = self._dummy_audio()
        encoding = self.processor(audio, return_tensors="pt")

        self.assertEqual(set(encoding.keys()), {"input_ids", "attention_mask", "input_features", "num_delay_tokens"})
        self.assertEqual(encoding["attention_mask"].shape, encoding["input_ids"].shape)
        self.assertEqual(encoding["input_features"].shape[0], 1)
        self.assertEqual(int(encoding["num_delay_tokens"]), self.processor.num_delay_tokens)
        self._assert_streaming_prefill_input_ids(encoding["input_ids"])

    def test_online_streaming_call(self):
        processor = self.processor

        # First chunk: produces the streaming prefill prompt together with the audio features.
        first_encoding = processor(
            self._dummy_audio(), is_streaming=True, is_first_audio_chunk=True, return_tensors="pt"
        )
        self.assertEqual(
            set(first_encoding.keys()), {"input_ids", "attention_mask", "input_features", "num_delay_tokens"}
        )
        self.assertEqual(first_encoding["input_features"].shape[0], 1)
        self._assert_streaming_prefill_input_ids(first_encoding["input_ids"])

        # Subsequent chunks: only the audio is encoded, no new prompt/text tokens are produced.
        next_chunk = self._dummy_audio(seed=1)[: processor.num_samples_per_audio_chunk]
        next_encoding = processor(next_chunk, is_streaming=True, is_first_audio_chunk=False, return_tensors="pt")
        self.assertIn("input_features", next_encoding)
        self.assertNotIn("input_ids", next_encoding)
        self.assertNotIn("attention_mask", next_encoding)

    def test_non_streaming_with_non_first_chunk_raises(self):
        audio = self._dummy_audio()
        with self.assertRaises(ValueError):
            self.processor(audio, is_streaming=False, is_first_audio_chunk=False)

    def test_batched_audio(self):
        audios = [self._dummy_audio(seed=1), self._dummy_audio(seed=2)]
        encoding = self.processor(audios, return_tensors="pt")

        self.assertEqual(encoding["input_ids"].shape[0], 2)
        self.assertEqual(encoding["input_features"].shape[0], 2)

    def test_audio_config_properties(self):
        # Audio-config-derived properties resolve to positive ints (num_delay_tokens / num_right_pad_tokens
        # wrap mistral-common methods that must be called).
        processor = self.processor
        for name in (
            "num_delay_tokens",
            "num_right_pad_tokens",
            "audio_length_per_tok",
            "raw_audio_length_per_tok",
            "num_mel_frames_first_audio_chunk",
            "num_samples_first_audio_chunk",
            "num_samples_per_audio_chunk",
        ):
            value = getattr(processor, name)
            self.assertIsInstance(value, int, f"{name} should be an int, got {type(value)}")
            self.assertGreater(value, 0, f"{name} should be positive, got {value}")

    def test_online_streaming_matches_low_level_audio_encoder(self):
        # Online streaming builds the prefill from the audio encoder primitives; check it matches.
        processor = self.processor
        sampling_rate = processor.feature_extractor.sampling_rate
        audio = self._dummy_audio(seed=42)

        encoding = processor(audio, is_streaming=True, is_first_audio_chunk=True, return_tensors="pt")

        instruct_tokenizer = processor.tokenizer.tokenizer.instruct_tokenizer
        audio_encoder = instruct_tokenizer.audio_encoder
        expected_tokens = instruct_tokenizer.start() + audio_encoder.encode_streaming_tokens()
        self.assertEqual(encoding["input_ids"].tolist(), [expected_tokens])

        left_pad, _ = audio_encoder.get_padding_audio()
        expected_features = processor.feature_extractor(
            np.concatenate((left_pad.audio_array, audio)),
            center=True,
            sampling_rate=sampling_rate,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )["input_features"]
        torch.testing.assert_close(encoding["input_features"], expected_features)
