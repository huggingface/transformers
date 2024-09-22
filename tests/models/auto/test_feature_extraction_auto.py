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
import sys
import tempfile
import unittest
from pathlib import Path

import transformers
from transformers import (
    CONFIG_MAPPING,
    FEATURE_EXTRACTOR_MAPPING,
    AutoConfig,
    AutoFeatureExtractor,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
)
from transformers.testing_utils import DUMMY_UNKNOWN_IDENTIFIER, get_tests_dir


sys.path.append(str(Path(__file__).parent.parent.parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig  # noqa E402
from test_module.custom_feature_extraction import CustomFeatureExtractor  # noqa E402


SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR = get_tests_dir("fixtures")
SAMPLE_FEATURE_EXTRACTION_CONFIG = get_tests_dir("fixtures/dummy_feature_extractor_config.json")
SAMPLE_CONFIG = get_tests_dir("fixtures/dummy-config.json")


class AutoFeatureExtractorTest(unittest.TestCase):
    def setUp(self):
        transformers.dynamic_module_utils.TIME_OUT_REMOTE_CODE = 0

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
            config = Wav2Vec2FeatureExtractor(**config_dict)

            # save in new folder
            model_config.save_pretrained(tmpdirname)
            config.save_pretrained(tmpdirname)

            config = AutoFeatureExtractor.from_pretrained(tmpdirname)

            # make sure private variable is not incorrectly saved
            dict_as_saved = json.loads(config.to_json_string())
            self.assertTrue("_processor_class" not in dict_as_saved)

        self.assertIsInstance(config, Wav2Vec2FeatureExtractor)

    def test_feature_extractor_from_local_file(self):
        config = AutoFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG)
        self.assertIsInstance(config, Wav2Vec2FeatureExtractor)

    def test_repo_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, "bert-base is not a local folder and is not a valid model identifier"
        ):
            _ = AutoFeatureExtractor.from_pretrained("bert-base")

    def test_revision_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, r"aaaaaa is not a valid git identifier \(branch name, tag name or commit id\)"
        ):
            _ = AutoFeatureExtractor.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, revision="aaaaaa")

    def test_feature_extractor_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError,
            "hf-internal-testing/config-no-model does not appear to have a file named preprocessor_config.json.",
        ):
            _ = AutoFeatureExtractor.from_pretrained("hf-internal-testing/config-no-model")

    def test_from_pretrained_dynamic_feature_extractor(self):
        # If remote code is not set, we will time out when asking whether to load the model.
        with self.assertRaises(ValueError):
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                "hf-internal-testing/test_dynamic_feature_extractor"
            )
        # If remote code is disabled, we can't load this config.
        with self.assertRaises(ValueError):
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                "hf-internal-testing/test_dynamic_feature_extractor", trust_remote_code=False
            )

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "hf-internal-testing/test_dynamic_feature_extractor", trust_remote_code=True
        )
        self.assertEqual(feature_extractor.__class__.__name__, "NewFeatureExtractor")

        # Test the dynamic module is loaded only once.
        reloaded_feature_extractor = AutoFeatureExtractor.from_pretrained(
            "hf-internal-testing/test_dynamic_feature_extractor", trust_remote_code=True
        )
        self.assertIs(feature_extractor.__class__, reloaded_feature_extractor.__class__)

        # Test feature extractor can be reloaded.
        with tempfile.TemporaryDirectory() as tmp_dir:
            feature_extractor.save_pretrained(tmp_dir)
            reloaded_feature_extractor = AutoFeatureExtractor.from_pretrained(tmp_dir, trust_remote_code=True)
        self.assertEqual(reloaded_feature_extractor.__class__.__name__, "NewFeatureExtractor")

        # The feature extractor file is cached in the snapshot directory. So the module file is not changed after dumping
        # to a temp dir. Because the revision of the module file is not changed.
        # Test the dynamic module is loaded only once if the module file is not changed.
        self.assertIs(feature_extractor.__class__, reloaded_feature_extractor.__class__)

        # Test the dynamic module is reloaded if we force it.
        reloaded_feature_extractor = AutoFeatureExtractor.from_pretrained(
            "hf-internal-testing/test_dynamic_feature_extractor", trust_remote_code=True, force_download=True
        )
        self.assertIsNot(feature_extractor.__class__, reloaded_feature_extractor.__class__)

    def test_new_feature_extractor_registration(self):
        try:
            AutoConfig.register("custom", CustomConfig)
            AutoFeatureExtractor.register(CustomConfig, CustomFeatureExtractor)
            # Trying to register something existing in the Transformers library will raise an error
            with self.assertRaises(ValueError):
                AutoFeatureExtractor.register(Wav2Vec2Config, Wav2Vec2FeatureExtractor)

            # Now that the config is registered, it can be used as any other config with the auto-API
            feature_extractor = CustomFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
            with tempfile.TemporaryDirectory() as tmp_dir:
                feature_extractor.save_pretrained(tmp_dir)
                new_feature_extractor = AutoFeatureExtractor.from_pretrained(tmp_dir)
                self.assertIsInstance(new_feature_extractor, CustomFeatureExtractor)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in FEATURE_EXTRACTOR_MAPPING._extra_content:
                del FEATURE_EXTRACTOR_MAPPING._extra_content[CustomConfig]

    def test_from_pretrained_dynamic_feature_extractor_conflict(self):
        class NewFeatureExtractor(Wav2Vec2FeatureExtractor):
            is_local = True

        try:
            AutoConfig.register("custom", CustomConfig)
            AutoFeatureExtractor.register(CustomConfig, NewFeatureExtractor)
            # If remote code is not set, the default is to use local
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                "hf-internal-testing/test_dynamic_feature_extractor"
            )
            self.assertEqual(feature_extractor.__class__.__name__, "NewFeatureExtractor")
            self.assertTrue(feature_extractor.is_local)

            # If remote code is disabled, we load the local one.
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                "hf-internal-testing/test_dynamic_feature_extractor", trust_remote_code=False
            )
            self.assertEqual(feature_extractor.__class__.__name__, "NewFeatureExtractor")
            self.assertTrue(feature_extractor.is_local)

            # If remote is enabled, we load from the Hub
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                "hf-internal-testing/test_dynamic_feature_extractor", trust_remote_code=True
            )
            self.assertEqual(feature_extractor.__class__.__name__, "NewFeatureExtractor")
            self.assertTrue(not hasattr(feature_extractor, "is_local"))

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in FEATURE_EXTRACTOR_MAPPING._extra_content:
                del FEATURE_EXTRACTOR_MAPPING._extra_content[CustomConfig]
