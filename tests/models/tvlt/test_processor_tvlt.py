# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import shutil
import tempfile
import unittest

import numpy as np

from transformers import is_speech_available, is_vision_available
from transformers.testing_utils import require_torch


if is_vision_available():
    from transformers import TvltImageProcessor

if is_speech_available():
    from transformers import TvltFeatureExtractor

from transformers import TvltProcessor


@require_torch
class TvltProcessorTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = "TVLT/tvlt-base"
        self.tmpdirname = tempfile.mkdtemp()

    def get_image_processor(self, **kwargs):
        return TvltImageProcessor.from_pretrained(self.checkpoint, **kwargs)

    def get_feature_extractor(self, **kwargs):
        return TvltFeatureExtractor.from_pretrained(self.checkpoint, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_feature_extractor(self):
        image_processor = self.get_image_processor()
        feature_extractor = self.get_feature_extractor()

        processor = TvltProcessor(image_processor=image_processor, feature_extractor=feature_extractor)

        audio_inputs = np.ones([12000])

        audio_inputs_dict = feature_extractor(audio_inputs, return_tensors="np")
        input_processor = processor(audio_inputs=audio_inputs, return_tensors="np")

        for key in audio_inputs_dict.keys():
            self.assertAlmostEqual(audio_inputs_dict[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        feature_extractor = self.get_feature_extractor()

        processor = TvltProcessor(image_processor=image_processor, feature_extractor=feature_extractor)

        visual_inputs = np.ones([3, 224, 224])

        visual_inputs_dict = image_processor(visual_inputs, return_tensors="np")
        input_processor = processor(visual_inputs=visual_inputs, return_tensors="np")

        for key in visual_inputs_dict.keys():
            self.assertAlmostEqual(visual_inputs_dict[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        feature_extractor = self.get_feature_extractor()

        processor = TvltProcessor(image_processor=image_processor, feature_extractor=feature_extractor)

        self.assertListEqual(
            processor.model_input_names,
            image_processor.model_input_names + feature_extractor.model_input_names,
            msg="`processor` and `image_processor`+`feature_extractor` model input names do not match",
        )
