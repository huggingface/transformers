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
from transformers.models.nemotron3_diarization.processing_nemotron3_diarization import DEFAULT_STREAMING_MODES
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@require_torch
class Nemotron3DiarizationProcessorTest(unittest.TestCase):
    # Model-card latency profiles: streaming mode -> input buffer latency in ms.
    LATENCIES = {"low_latency": 1040, "very_low_latency": 640, "ultra_low_latency": 320}

    def get_processor(self, **kwargs) -> Nemotron3DiarizationProcessor:
        feature_extractor = NemotronAsrStreamingFeatureExtractor(feature_size=128)
        return Nemotron3DiarizationProcessor(feature_extractor=feature_extractor, **kwargs)

    def test_save_load_roundtrip(self):
        processor = self.get_processor()
        self.assertEqual(processor.streaming_mode, "low_latency")
        self.assertEqual(processor.streaming_modes, DEFAULT_STREAMING_MODES)
        processor.set_streaming_mode("ultra_low_latency")
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor.save_pretrained(tmp_dir)
            reloaded = AutoProcessor.from_pretrained(tmp_dir)
        self.assertIsInstance(reloaded, Nemotron3DiarizationProcessor)
        self.assertEqual(reloaded.subsampling_factor, processor.subsampling_factor)
        self.assertEqual(reloaded.streaming_mode, "ultra_low_latency")
        # the modes travel with the checkpoint (tuples become lists in JSON)
        self.assertEqual(
            {mode: tuple(sizes) for mode, sizes in reloaded.streaming_modes.items()}, DEFAULT_STREAMING_MODES
        )
        self.assertEqual(reloaded.num_mel_frames_per_audio_chunk, processor.num_mel_frames_per_audio_chunk)

        # a checkpoint can ship its own modes
        custom = self.get_processor(streaming_modes={"fast": (2, 1)}, streaming_mode="fast")
        self.assertEqual(custom.num_mel_frames_per_audio_chunk, 3 * custom.subsampling_factor)
        with self.assertRaises(ValueError):
            custom.set_streaming_mode("low_latency")
        self.assertEqual(reloaded.feature_extractor.feature_size, 128)

    def test_streaming_modes(self):
        processor = self.get_processor()
        self.assertEqual(set(processor.streaming_modes), set(self.LATENCIES))
        with self.assertRaises(ValueError):
            processor.set_streaming_mode("offline")
        with self.assertRaises(ValueError):
            self.get_processor(streaming_mode="offline")
        for mode, latency_ms in self.LATENCIES.items():
            processor.set_streaming_mode(mode)
            chunk_length, chunk_right_context = processor.streaming_modes[mode]
            self.assertEqual(processor.streaming_latency_ms, latency_ms)
            self.assertEqual(
                processor.num_mel_frames_per_audio_chunk,
                (chunk_length + chunk_right_context) * processor.subsampling_factor,
            )
            self.assertEqual(processor.num_mel_frames_per_step, chunk_length * processor.subsampling_factor)

    def test_call_modes(self):
        """
        Offline outputs are the features alone. Streaming checks the chunk size of the mode, and every chunk but the
        last carries `num_lookahead_frames`, which puts the model in streaming mode.
        """
        processor = self.get_processor(streaming_mode="very_low_latency")
        self.assertIn("num_lookahead_frames", processor.model_input_names)
        audio = np.zeros(4 * 16000, dtype=np.float32)

        offline = processor(audio, sampling_rate=16000)
        self.assertEqual(set(offline.keys()), {"input_features", "attention_mask"})
        with self.assertRaises(ValueError):
            processor(audio, sampling_rate=16000, is_first_audio_chunk=False)
        with self.assertRaises(ValueError):
            processor(audio, sampling_rate=16000, is_last_audio_chunk=True)

        first = processor(audio[: processor.num_samples_first_audio_chunk], sampling_rate=16000, is_streaming=True)
        self.assertEqual(first["num_lookahead_frames"], processor.streaming_modes["very_low_latency"][1])
        # ints survive the device placement of the batch
        self.assertEqual(first.to("cpu", dtype=torch.float32)["num_lookahead_frames"], 2)
        # the mode can change between sessions
        processor.set_streaming_mode("low_latency")
        first = processor(audio[: processor.num_samples_first_audio_chunk], sampling_rate=16000, is_streaming=True)
        self.assertEqual(first["num_lookahead_frames"], 4)

        # a chunk of the wrong size is rejected unless it is the last one
        with self.assertRaises(ValueError):
            processor(audio[:16000], sampling_rate=16000, is_streaming=True)
        last = processor(
            audio[:16000], sampling_rate=16000, is_streaming=True, is_first_audio_chunk=False, is_last_audio_chunk=True
        )
        self.assertNotIn("num_lookahead_frames", last)

    def test_extract_speaker_dict(self):
        """Same segment format as `VibeVoiceAsrProcessor.extract_speaker_dict`, without `Content`."""
        processor = self.get_processor()
        logits = torch.full((2, 10, 3), -5.0)
        logits[0, 2:5, 0] = 5.0  # speaker 0: frames 2-4
        logits[0, 4:6, 1] = 5.0  # speaker 1: frames 4-5, overlapping speaker 0
        logits[0, 8:, 0] = 5.0  # speaker 0 again, until the end
        logits[1, :, 2] = 5.0  # speaker 2 all along, but the sample is 6 frames long
        attention_mask = torch.ones(2, 10, dtype=torch.long)
        attention_mask[1, 6:] = 0

        speaker_dicts = processor.extract_speaker_dict(logits, attention_mask)
        self.assertEqual(len(speaker_dicts), 2)
        self.assertEqual(
            speaker_dicts[0],
            [
                {"Start": 0.02, "End": 0.05, "Speaker": 0},
                {"Start": 0.04, "End": 0.06, "Speaker": 1},
                {"Start": 0.08, "End": 0.1, "Speaker": 0},
            ],
        )
        self.assertEqual(speaker_dicts[1], [{"Start": 0.0, "End": 0.06, "Speaker": 2}])
        # without the mask, the padded frames of the second sample count
        self.assertEqual(processor.extract_speaker_dict(logits)[1], [{"Start": 0.0, "End": 0.1, "Speaker": 2}])
        # a stricter threshold silences everyone
        self.assertEqual(processor.extract_speaker_dict(logits, threshold=1.0), [[], []])

    def test_chunk_sizes_yield_expected_number_of_frames(self):
        processor = self.get_processor(streaming_mode="low_latency")
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
        processor = self.get_processor(streaming_mode="low_latency")
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
