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

from transformers import (
    CohereAsrFeatureExtractor,
    LasrFeatureExtractor,
    NemotronAsrStreamingFeatureExtractor,
    ParakeetFeatureExtractor,
    Phi4MultimodalFeatureExtractor,
    VoxtralRealtimeFeatureExtractor,
)
from transformers.testing_utils import require_librosa, require_torch


@require_torch
@require_librosa
class BatchedMultichannelASRFeatureExtractionTest(unittest.TestCase):
    def _assert_batched_multichannel_matches_mono(self, feature_extractor):
        rng = np.random.default_rng(0)
        mono = rng.normal(size=(800,)).astype(np.float32)
        stereo = np.stack([mono, mono], axis=-1)

        mono_inputs = feature_extractor([mono, mono], sampling_rate=16000, return_tensors="pt")
        stereo_inputs = feature_extractor([stereo, stereo], sampling_rate=16000, return_tensors="pt")

        self.assertEqual(stereo_inputs.keys(), mono_inputs.keys())
        for key in mono_inputs:
            np.testing.assert_allclose(
                stereo_inputs[key].cpu().numpy(),
                mono_inputs[key].cpu().numpy(),
                rtol=1e-5,
                atol=1e-6,
            )

    def test_cohere_asr(self):
        self._assert_batched_multichannel_matches_mono(CohereAsrFeatureExtractor(dither=0.0))

    def test_lasr(self):
        self._assert_batched_multichannel_matches_mono(LasrFeatureExtractor(feature_size=8))

    def test_nemotron_asr_streaming(self):
        self._assert_batched_multichannel_matches_mono(NemotronAsrStreamingFeatureExtractor())

    def test_parakeet(self):
        self._assert_batched_multichannel_matches_mono(ParakeetFeatureExtractor())

    def test_phi4_multimodal(self):
        self._assert_batched_multichannel_matches_mono(Phi4MultimodalFeatureExtractor())

    def test_voxtral_realtime(self):
        self._assert_batched_multichannel_matches_mono(VoxtralRealtimeFeatureExtractor())
