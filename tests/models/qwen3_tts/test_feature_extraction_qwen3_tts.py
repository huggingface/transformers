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

import io
import itertools
import tempfile
import unittest

import numpy as np

from transformers import Qwen3TTSFeatureExtractor
from transformers.testing_utils import require_soundfile, require_torch, require_torchaudio, slow


global_rng = np.random.default_rng(0)


def floats_list(shape, scale=1.0):
    """Create a random float list of a given shape."""
    return [[scale * global_rng.random() for _ in range(shape[1])] for _ in range(shape[0])]


class Qwen3TTSFeatureExtractionTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=4000,
        max_seq_length=8000,
        feature_size=128,
        padding_value=0.0,
        sampling_rate=24000,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        fmin=0.0,
        fmax=12000.0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "fmin": self.fmin,
            "fmax": self.fmax,
        }

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        def _flatten(list_of_lists):
            return list(itertools.chain(*list_of_lists))

        if equal_length:
            audio_inputs = floats_list((self.batch_size, self.max_seq_length))
        else:
            audio_inputs = [
                _flatten(floats_list((x, 1)))
                for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)
            ]

        if numpify:
            audio_inputs = [np.asarray(x) for x in audio_inputs]

        return audio_inputs


@require_torch
class Qwen3TTSFeatureExtractionTest(unittest.TestCase):
    """
    The common sequence tester is not used here: this extractor produces one mel spectrogram per waveform for the
    speaker encoder and implements no padding or truncation, so that mixin's padding, truncation and
    attention-mask tests describe behaviour it does not have.

    The expected values below come from the original Qwen3-TTS implementation, so they pin the extractor to the
    recipe the speaker encoder was trained on rather than to whatever this code currently produces.

    reproducer: https://gist.github.com/ShahVandit/cab13f3b7232c52b4ff93cce592950c4#file-reproducer_qwen3_tts-py
    """

    feature_extraction_class = Qwen3TTSFeatureExtractor

    def setUp(self):
        self.feat_extract_tester = Qwen3TTSFeatureExtractionTester(self)

    def _load_datasamples(self, num_samples):
        import soundfile as sf
        import torch
        import torchaudio
        from datasets import load_dataset

        dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        samples = []
        for raw in dataset.data.column("audio").to_pylist()[:num_samples]:
            audio_bytes = raw.get("bytes")
            array, original_sampling_rate = (
                sf.read(io.BytesIO(audio_bytes), dtype="float32")
                if audio_bytes
                else sf.read(raw["path"], dtype="float32")
            )
            if array.ndim > 1:
                array = array.mean(axis=1)
            if original_sampling_rate != self.feat_extract_tester.sampling_rate:
                array = torchaudio.functional.resample(
                    torch.from_numpy(array), original_sampling_rate, self.feat_extract_tester.sampling_rate
                ).numpy()
            samples.append(np.asarray(array, dtype=np.float32))
        return samples

    def test_call(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        audio_inputs = self.feat_extract_tester.prepare_inputs_for_common(numpify=True)
        tester = self.feat_extract_tester

        # a single waveform yields one `(num_frames, num_mel_bins)` array
        features = feature_extractor(audio_inputs[0], sampling_rate=tester.sampling_rate).input_features
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].shape[-1], tester.feature_size)

        # a batch keeps one entry per waveform, each with the number of frames its own length implies
        batched = feature_extractor(audio_inputs, sampling_rate=tester.sampling_rate).input_features
        self.assertEqual(len(batched), len(audio_inputs))
        reflect_padding = (tester.n_fft - tester.hop_length) // 2
        for audio, feature in zip(audio_inputs, batched):
            expected_frames = (len(audio) + 2 * reflect_padding - tester.n_fft) // tester.hop_length + 1
            self.assertEqual(feature.shape, (expected_frames, tester.feature_size))

        # batching must not change a waveform's own features
        alone = feature_extractor(audio_inputs[2], sampling_rate=tester.sampling_rate).input_features[0]
        np.testing.assert_allclose(batched[2], alone, atol=1e-6)

        # lists and arrays describing the same audio agree
        as_list = feature_extractor(audio_inputs[0].tolist(), sampling_rate=tester.sampling_rate).input_features
        np.testing.assert_allclose(features[0], as_list[0], atol=1e-6)

    def test_sampling_rate_mismatch_raises(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        with self.assertRaises(ValueError):
            feature_extractor(np.zeros(4000, dtype=np.float32), sampling_rate=16000)

    def test_silence_hits_the_compression_floor(self):
        """Digital silence must land on `log(1e-5)`; a different clamp or a power spectrum would not."""
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        features = feature_extractor(np.zeros(8000, dtype=np.float32), sampling_rate=24000).input_features[0]
        np.testing.assert_allclose(features, np.full_like(features, np.log(1e-5)), atol=1e-6)

    def test_save_load_round_trip(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_extractor.save_pretrained(tmpdir)
            reloaded = self.feature_extraction_class.from_pretrained(tmpdir)

        self.assertEqual(feature_extractor.to_dict(), reloaded.to_dict())
        np.testing.assert_allclose(feature_extractor.mel_filters, reloaded.mel_filters)

    @slow
    @require_soundfile
    @require_torchaudio
    def test_integration(self):
        import torch

        # fmt: off
        EXPECTED_FRAME_0 = torch.tensor(
            [-4.966196, -5.039193, -5.931322, -6.520872, -6.368503, -7.140932,
             -7.316255, -7.836317, -7.641678, -6.957040, -7.106559, -7.789589]
        )
        EXPECTED_FRAME_50 = torch.tensor(
            [-4.403539, -4.045483, -4.245713, -4.783569, -5.301741, -5.376821,
             -5.069336, -5.781121, -5.842892, -6.089342, -6.680262, -6.379679]
        )
        # fmt: on

        audio = self._load_datasamples(1)[0]
        feature_extractor = self.feature_extraction_class()
        features = torch.from_numpy(feature_extractor(audio, sampling_rate=24000).input_features[0])

        self.assertEqual(features.shape, (548, 128))
        torch.testing.assert_close(features[0, :12], EXPECTED_FRAME_0, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(features[50, :12], EXPECTED_FRAME_50, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(features.mean(), torch.tensor(-5.953048), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(features.min(), torch.tensor(-11.512925), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(features.max(), torch.tensor(0.302033), rtol=1e-4, atol=1e-4)
