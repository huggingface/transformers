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
import json
import tempfile
import unittest

import numpy as np

from transformers import Qwen3TTSTokenizerSingleCodebookFeatureExtractor
from transformers.testing_utils import require_torch


@require_torch
class Qwen3TTSTokenizerSingleCodebookFeatureExtractionTest(unittest.TestCase):
    def _feat_extract(self):
        return Qwen3TTSTokenizerSingleCodebookFeatureExtractor(audio_vq_ds_rate=2)

    def test_call_shapes_and_masks(self):
        feat_extract = self._feat_extract()
        raw_audio = [np.zeros(321, dtype=np.float32), np.zeros(640, dtype=np.float32)]
        inputs = feat_extract(raw_audio, sampling_rate=16000, return_tensors="pt")

        self.assertEqual(
            set(inputs.keys()),
            {
                "input_features",
                "input_features_mask",
                "input_values",
                "padding_mask",
                "ref_mel_features",
                "ref_mel_attention_mask",
            },
        )
        self.assertEqual(inputs["input_values"].shape, (2, 640))
        self.assertEqual(inputs["padding_mask"].sum(dim=-1).tolist(), [321, 640])
        self.assertEqual(inputs["input_features"].shape[1], feat_extract.feature_size)
        self.assertEqual(inputs["ref_mel_features"].shape[-1], feat_extract.ref_num_mel_bins)

    def test_to_dict_is_json_native(self):
        feat_extract = self._feat_extract()
        serialized = feat_extract.to_dict()
        json.dumps(serialized)
        self.assertNotIn("waveform_padder", serialized)
        self.assertNotIn("mel_filters", serialized)
        self.assertNotIn("ref_mel_filters", serialized)

    def test_save_load_roundtrip(self):
        feat_extract = self._feat_extract()
        with tempfile.TemporaryDirectory() as tmpdirname:
            feat_extract.save_pretrained(tmpdirname)
            loaded = Qwen3TTSTokenizerSingleCodebookFeatureExtractor.from_pretrained(tmpdirname)
        self.assertEqual(feat_extract.feature_size, loaded.feature_size)
        self.assertEqual(feat_extract.audio_vq_ds_rate, loaded.audio_vq_ds_rate)
        self.assertEqual(feat_extract.ref_num_mel_bins, loaded.ref_num_mel_bins)
        self.assertTrue(hasattr(loaded, "waveform_padder"))
