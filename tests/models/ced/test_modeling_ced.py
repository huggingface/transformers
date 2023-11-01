# coding=utf-8
# Copyright 2023 Xiaomi Corporation and The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch CED (Ced) model. """

import unittest

from transformers.testing_utils import require_torch, require_torchaudio, slow, torch_device
from transformers.utils import is_torch_available, is_torchaudio_available


if is_torch_available():
    import torch

    from transformers import CedForAudioClassification


if is_torchaudio_available():
    from transformers import CedFeatureExtractor


@require_torch
@require_torchaudio
class CedModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_audio_classification(self):
        feature_extractor = CedFeatureExtractor()
        model = CedForAudioClassification.from_pretrained("mispeech/ced-tiny").eval()
        audio = torch.arange(1, 16000).unsqueeze(0) / 1e4
        feature = feature_extractor(audio)["input_values"]
        outputs = model(feature)

        # verify the logits
        expected_shape = torch.Size((1, 527))
        self.assertEqual(outputs.shape, expected_shape)

        expected_slice = torch.tensor([3.0647e-03, 8.3724e-05, 7.5101e-05]).to(torch_device)

        self.assertTrue(torch.allclose(outputs[0, :3], expected_slice, atol=1e-4))
