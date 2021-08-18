# coding=utf-8
# Copyright 2021 the HuggingFace Inc. team.
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

import os
import tempfile
import unittest

from transformers import AutoFeatureExtractor, Wav2Vec2Config, Wav2Vec2FeatureExtractor


SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
SAMPLE_FEATURE_EXTRACTION_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures/dummy_feature_extractor_config.json"
)
SAMPLE_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/dummy-config.json")


class AutoFeatureExtractorTest(unittest.TestCase):
    def test_feature_extractor_from_model_shortcut(self):
        config = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        self.assertIsInstance(config, Wav2Vec2FeatureExtractor)

    def test_feature_extractor_from_local_directory_from_key(self):
        config = AutoFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
        self.assertIsInstance(config, Wav2Vec2FeatureExtractor)

    def test_feature_extractor_from_local_directory_from_config(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_config = Wav2Vec2Config()

            # remove feature_extractor_type to make sure config.json alone is enough to load feature processor locally
            config_dict = AutoFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR).to_dict()
            config_dict.pop("feature_extractor_type")
            config = Wav2Vec2FeatureExtractor(config_dict)

            # save in new folder
            model_config.save_pretrained(tmpdirname)
            config.save_pretrained(tmpdirname)

            config = AutoFeatureExtractor.from_pretrained(tmpdirname)

        self.assertIsInstance(config, Wav2Vec2FeatureExtractor)

    def test_feature_extractor_from_local_file(self):
        config = AutoFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG)
        self.assertIsInstance(config, Wav2Vec2FeatureExtractor)
