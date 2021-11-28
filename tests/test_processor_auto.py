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

from transformers import AutoProcessor, BeitFeatureExtractor, BertTokenizerFast, Wav2Vec2Config, Wav2Vec2Processor
from transformers.testing_utils import require_torch


SAMPLE_PROCESSOR_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
SAMPLE_PROCESSOR_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures/dummy_feature_extractor_config.json"
)
SAMPLE_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/dummy-config.json")


class AutoFeatureExtractorTest(unittest.TestCase):
    def test_processor_from_model_shortcut(self):
        processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_local_directory_from_config(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_config = Wav2Vec2Config()
            processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

            # save in new folder
            model_config.save_pretrained(tmpdirname)
            processor.save_pretrained(tmpdirname)

            processor = AutoProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_auto_processor_reverts_to_tokenizer(self):
        processor = AutoProcessor.from_pretrained("bert-base-cased")
        self.assertIsInstance(processor, BertTokenizerFast)

    @require_torch
    def test_auto_processor_reverts_to_feature_extractor(self):
        processor = AutoProcessor.from_pretrained("microsoft/beit-base-patch16-224")
        self.assertIsInstance(processor, BeitFeatureExtractor)
