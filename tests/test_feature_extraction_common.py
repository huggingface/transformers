# coding=utf-8
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
import tempfile

from transformers.file_utils import is_torch_available, is_vision_available


if is_torch_available():
    import numpy as np
    import torch

if is_vision_available():
    from PIL import Image


def prepare_image_inputs(feature_extract_tester, equal_resolution=False, numpify=False, torchify=False):
    """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
    or a list of PyTorch tensors if one specifies torchify=True.
    """

    assert not (numpify and torchify), "You cannot specify both numpy and PyTorch tensors at the same time"

    if equal_resolution:
        image_inputs = []
        for i in range(feature_extract_tester.batch_size):
            image_inputs.append(
                np.random.randint(
                    255,
                    size=(
                        feature_extract_tester.num_channels,
                        feature_extract_tester.max_resolution,
                        feature_extract_tester.max_resolution,
                    ),
                    dtype=np.uint8,
                )
            )
    else:
        image_inputs = []
        for i in range(feature_extract_tester.batch_size):
            width, height = np.random.choice(
                np.arange(feature_extract_tester.min_resolution, feature_extract_tester.max_resolution), 2
            )
            image_inputs.append(
                np.random.randint(255, size=(feature_extract_tester.num_channels, width, height), dtype=np.uint8)
            )

    if not numpify and not torchify:
        # PIL expects the channel dimension as last dimension
        image_inputs = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in image_inputs]

    if torchify:
        image_inputs = [torch.from_numpy(x) for x in image_inputs]

    return image_inputs


class FeatureExtractionSavingTestMixin:
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
            feat_extract_first.save_pretrained(tmpdirname)
            feat_extract_second = self.feature_extraction_class.from_pretrained(tmpdirname)

        self.assertEqual(feat_extract_second.to_dict(), feat_extract_first.to_dict())

    def test_init_without_params(self):
        feat_extract = self.feature_extraction_class()
        self.assertIsNotNone(feat_extract)
