# Copyright 2021 HuggingFace Inc.
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
import os
import random
import tempfile

import numpy as np

from transformers.testing_utils import check_json_file_has_correct_format, require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


class FeatureExtractionSavingTestMixin:
    test_cast_dtype = None

    def prepare_audio_inputs(batch_size, seq_len, scale=1.0, rng=None, torchify=False, numpify=False):
        rng = random.Random()
        values = []
        for batch_idx in range(batch_size):
            values.append([])
            for _ in range(seq_len):
                values[-1].append(rng.random() * scale)

        if numpify:
            values = np.array(values)
        elif torchify:
            values = torch.tensor(values)

        return values

    def test_feat_extract_to_json_string(self):
        feat_extract = self.feature_extraction_class(**self.feat_extract_dict)
        obj = json.loads(feat_extract.to_json_string())
        for key, value in self.feat_extract_dict.items():
            self.assertEqual(obj[key], value)

    def test_feat_extract_to_json_file(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "feat_extract.json")
            feat_extract_first.to_json_file(json_file_path)
            feat_extract_second = self.feature_extraction_class.from_json_file(json_file_path)

        self.assertEqual(feat_extract_second.to_dict(), feat_extract_first.to_dict())

    def test_feat_extract_from_and_save_pretrained(self):
        feat_extract_first = self.feature_extraction_class(**self.feat_extract_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = feat_extract_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname)

        self.assertEqual(feat_extract_second.to_dict(), feat_extract_first.to_dict())

    def test_init_without_params(self):
        feat_extract = self.feature_extraction_class()
        self.assertIsNotNone(feat_extract)

    @require_torch
    def test_call_with_device(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        list_audio_inputs = self.prepare_audio_inputs(numpify=True)
        numpy_audio_inputs = self.prepare_audio_inputs(numpify=True)
        torch_audio_inputs = self.prepare_audio_inputs(torchify=True)

        for inputs in [list_audio_inputs, numpy_audio_inputs, torch_audio_inputs]:
            encoded_audio = feature_extractor(inputs, return_tensors="pt")
            for key in encoded_audio:
                self.assertIsInstance(encoded_audio[key], torch.Tensor)
                self.assertTrue(encoded_audio[key].device.type == "cpu")

            encoded_audio = feature_extractor(inputs, return_tensors="pt", device="cuda")
            for key in encoded_audio:
                self.assertIsInstance(encoded_audio[key], torch.Tensor)
                self.assertTrue(encoded_audio[key].device.type == "cuda")
