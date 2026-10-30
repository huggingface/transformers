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

from transformers import MiMoV2FeatureExtractor


class MiMoV2FeatureExtractorTest(unittest.TestCase):
    def test_unpadded_codec_batch_preserves_group_boundaries(self):
        extractor = MiMoV2FeatureExtractor(audio_channels=2, audio_group_size=2)
        codes = np.arange(10, dtype=np.int64).reshape(5, 2)
        result = extractor([codes[:3], codes], is_tokenized=True, padding=False)
        self.assertEqual(result.audio_feature_lens, [2, 3])
        self.assertEqual([sample.shape for sample in result.input_audio_codes], [(2, 2, 2), (3, 2, 2)])
        np.testing.assert_array_equal(result.input_audio_codes[0][-1], [[4, 5], [4, 5]])
        np.testing.assert_array_equal(result.input_audio_codes[1][-1], [[8, 9], [8, 9]])
        self.assertEqual(result.attention_mask, [[1, 1], [1, 1, 1]])

    def test_codec_mask_can_be_omitted(self):
        extractor = MiMoV2FeatureExtractor(audio_channels=2)
        result = extractor(np.ones((5, 2), dtype=np.int64), is_tokenized=True, return_attention_mask=False)
        self.assertNotIn("attention_mask", result)
        self.assertEqual(result.audio_feature_lens, [2])
