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

import tempfile
import unittest

import numpy as np

from transformers import AutoProcessor, Nemotron3DiarizationProcessor, NemotronAsrStreamingFeatureExtractor
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@require_torch
class Nemotron3DiarizationProcessorTest(unittest.TestCase):
    # Model-card latency profiles: (chunk_length, chunk_right_context) -> input buffer latency in ms.
    PROFILES = {(340, 40): 30400, (9, 4): 1040, (6, 2): 640, (3, 1): 320}

    def get_processor(self, **kwargs) -> Nemotron3DiarizationProcessor:
        feature_extractor = NemotronAsrStreamingFeatureExtractor(feature_size=128)
        return Nemotron3DiarizationProcessor(feature_extractor=feature_extractor, **kwargs)

    def test_save_load_roundtrip(self):
        processor = self.get_processor(chunk_length=9, chunk_right_context=4)
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor.save_pretrained(tmp_dir)
            reloaded = AutoProcessor.from_pretrained(tmp_dir)
        self.assertIsInstance(reloaded, Nemotron3DiarizationProcessor)
        self.assertEqual(reloaded.chunk_length, 9)
        self.assertEqual(reloaded.chunk_right_context, 4)
        self.assertEqual(reloaded.subsampling_factor, processor.subsampling_factor)
        self.assertEqual(reloaded.feature_extractor.feature_size, 128)

    def test_streaming_latencies(self):
        processor = self.get_processor()
        for (chunk_length, chunk_right_context), latency_ms in self.PROFILES.items():
            processor.set_streaming_profile(chunk_length=chunk_length, chunk_right_context=chunk_right_context)
            self.assertEqual(processor.streaming_latency_ms, latency_ms)
            self.assertEqual(
                processor.num_mel_frames_per_audio_chunk,
                (chunk_length + chunk_right_context) * processor.subsampling_factor,
            )
            self.assertEqual(processor.num_mel_frames_per_step, chunk_length * processor.subsampling_factor)

    def test_call_requires_first_chunk_when_not_streaming(self):
        processor = self.get_processor()
        audio = np.zeros(16000, dtype=np.float32)
        with self.assertRaises(ValueError):
            processor(audio, sampling_rate=16000, is_first_audio_chunk=False)

    def test_chunk_sizes_yield_expected_number_of_frames(self):
        processor = self.get_processor(chunk_length=9, chunk_right_context=4)
        audio = np.random.RandomState(0).randn(2 * 16000).astype(np.float32)
        expected_frames = processor.num_mel_frames_per_audio_chunk

        first = processor(audio[: processor.num_samples_first_audio_chunk], sampling_rate=16000, is_streaming=True)
        self.assertEqual(int(first.attention_mask.sum()), expected_frames)
        # streaming chunks carry no padded frame, so they reach the model as they are
        self.assertEqual(first.input_features.shape[1], expected_frames)

        start = processor.audio_chunk_start(processor.num_mel_frames_per_step)
        later = processor(
            audio[start : start + processor.num_samples_per_audio_chunk],
            sampling_rate=16000,
            is_streaming=True,
            is_first_audio_chunk=False,
        )
        self.assertEqual(int(later.attention_mask.sum()), expected_frames)
        self.assertEqual(later.input_features.shape[1], expected_frames)

    def test_streaming_chunks_match_full_utterance(self):
        """Per-chunk extraction must reproduce, frame for frame, a single pass over the whole audio."""
        processor = self.get_processor(chunk_length=9, chunk_right_context=4)
        audio = np.random.RandomState(0).randn(4 * 16000).astype(np.float32)
        full = processor(audio, sampling_rate=16000)
        num_frames = int(full.attention_mask.sum())

        frame_idx = 0
        while True:
            is_first = frame_idx == 0
            start = 0 if is_first else processor.audio_chunk_start(frame_idx)
            num_samples = (
                processor.num_samples_first_audio_chunk if is_first else processor.num_samples_per_audio_chunk
            )
            if start + num_samples > audio.shape[0]:
                break
            chunk = processor(
                audio[start : start + num_samples],
                sampling_rate=16000,
                is_streaming=True,
                is_first_audio_chunk=is_first,
            )
            num_chunk_frames = processor.num_mel_frames_per_audio_chunk
            torch.testing.assert_close(
                chunk.input_features[0, :num_chunk_frames],
                full.input_features[0, frame_idx : frame_idx + num_chunk_frames],
                atol=1e-4,
                rtol=1e-4,
            )
            frame_idx += processor.num_mel_frames_per_step
        self.assertGreater(frame_idx, processor.num_mel_frames_per_step)
        self.assertLess(frame_idx, num_frames)
