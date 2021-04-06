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
import unittest

from transformers.models.auto.configuration_auto import CONFIG_MAPPING, AutoConfig
from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.testing_utils import DUMMY_UNKWOWN_IDENTIFIER


SAMPLE_ROBERTA_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/dummy-config.json")


class AutoFeatureExtractorTest(unittest.TestCase):
    def test_feature_extractor_from_model_shortcut(self):
        config = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        self.assertIsInstance(config, Wav2Vec2FeatureExtractor)

    def test_config_model_type_from_local_file(self):
        config = AutoFeatureExtractor.from_pretrained(SAMPLE_ROBERTA_CONFIG)
        self.assertIsInstance(config, RobertaConfig)

    def test_config_model_type_from_model_identifier(self):
        config = AutoConfig.from_pretrained(DUMMY_UNKWOWN_IDENTIFIER)
        self.assertIsInstance(config, RobertaConfig)

    def test_config_for_model_str(self):
        config = AutoConfig.for_model("roberta")
        self.assertIsInstance(config, RobertaConfig)

    def test_pattern_matching_fallback(self):
        """
        In cases where config.json doesn't include a model_type,
        perform a few safety checks on the config mapping's order.
        """
        # no key string should be included in a later key string (typical failure case)
        keys = list(CONFIG_MAPPING.keys())
        for i, key in enumerate(keys):
            self.assertFalse(any(key in later_key for later_key in keys[i + 1 :]))
