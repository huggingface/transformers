# Copyright 2026 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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

from transformers import FunAsrNanoFeatureExtractor
from transformers.testing_utils import require_torch, require_torchaudio


@require_torch
@require_torchaudio
class FunAsrNanoFeatureExtractionTest(unittest.TestCase):
    def test_returns_attention_mask_and_feature_lengths(self):
        import torch

        feature_extractor = FunAsrNanoFeatureExtractor()
        audio = [np.ones(16000, dtype=np.float32), np.ones(8000, dtype=np.float32)]

        outputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")

        self.assertIn("input_features", outputs)
        self.assertIn("attention_mask", outputs)
        self.assertIn("feature_lengths", outputs)
        self.assertTrue(torch.equal(outputs["attention_mask"].sum(-1), outputs["feature_lengths"]))


if __name__ == "__main__":
    unittest.main()
