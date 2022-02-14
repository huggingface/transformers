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

import json
import os
import tempfile
import unittest
from shutil import copyfile

from transformers import AutoProcessor, AutoTokenizer, Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers.file_utils import FEATURE_EXTRACTOR_NAME, is_tokenizers_available
from transformers.tokenization_utils import TOKENIZER_CONFIG_FILE


SAMPLE_PROCESSOR_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures/dummy_feature_extractor_config.json"
)
SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/vocab.json")


class AutoFeatureExtractorTest(unittest.TestCase):
    def test_processor_from_model_shortcut(self):
        processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_local_directory_from_repo(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_config = Wav2Vec2Config()
            processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

            # save in new folder
            model_config.save_pretrained(tmpdirname)
            processor.save_pretrained(tmpdirname)

            processor = AutoProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_local_directory_from_extractor_config(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # copy relevant files
            copyfile(SAMPLE_PROCESSOR_CONFIG, os.path.join(tmpdirname, FEATURE_EXTRACTOR_NAME))
            copyfile(SAMPLE_VOCAB, os.path.join(tmpdirname, "vocab.json"))

            processor = AutoProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_feat_extr_processor_class(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            feature_extractor = Wav2Vec2FeatureExtractor()
            tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

            processor = Wav2Vec2Processor(feature_extractor, tokenizer)

            # save in new folder
            processor.save_pretrained(tmpdirname)

            # drop `processor_class` in tokenizer
            with open(os.path.join(tmpdirname, TOKENIZER_CONFIG_FILE), "r") as f:
                config_dict = json.load(f)
                config_dict.pop("processor_class")

            with open(os.path.join(tmpdirname, TOKENIZER_CONFIG_FILE), "w") as f:
                f.write(json.dumps(config_dict))

            processor = AutoProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_tokenizer_processor_class(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            feature_extractor = Wav2Vec2FeatureExtractor()
            tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

            processor = Wav2Vec2Processor(feature_extractor, tokenizer)

            # save in new folder
            processor.save_pretrained(tmpdirname)

            # drop `processor_class` in feature extractor
            with open(os.path.join(tmpdirname, FEATURE_EXTRACTOR_NAME), "r") as f:
                config_dict = json.load(f)
                config_dict.pop("processor_class")

            with open(os.path.join(tmpdirname, FEATURE_EXTRACTOR_NAME), "w") as f:
                f.write(json.dumps(config_dict))

            processor = AutoProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_processor_from_local_directory_from_model_config(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_config = Wav2Vec2Config(processor_class="Wav2Vec2Processor")
            model_config.save_pretrained(tmpdirname)
            # copy relevant files
            copyfile(SAMPLE_VOCAB, os.path.join(tmpdirname, "vocab.json"))
            # create emtpy sample processor
            with open(os.path.join(tmpdirname, FEATURE_EXTRACTOR_NAME), "w") as f:
                f.write("{}")

            processor = AutoProcessor.from_pretrained(tmpdirname)

        self.assertIsInstance(processor, Wav2Vec2Processor)

    def test_from_pretrained_dynamic_processor(self):
        processor = AutoProcessor.from_pretrained("hf-internal-testing/test_dynamic_processor", trust_remote_code=True)
        self.assertTrue(processor.special_attribute_present)
        self.assertEqual(processor.__class__.__name__, "NewProcessor")

        feature_extractor = processor.feature_extractor
        self.assertTrue(feature_extractor.special_attribute_present)
        self.assertEqual(feature_extractor.__class__.__name__, "NewFeatureExtractor")

        tokenizer = processor.tokenizer
        self.assertTrue(tokenizer.special_attribute_present)
        if is_tokenizers_available():
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizerFast")

            # Test we can also load the slow version
            processor = AutoProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_processor", trust_remote_code=True, use_fast=False
            )
            tokenizer = processor.tokenizer
            self.assertTrue(tokenizer.special_attribute_present)
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizer")
        else:
            self.assertEqual(tokenizer.__class__.__name__, "NewTokenizer")
