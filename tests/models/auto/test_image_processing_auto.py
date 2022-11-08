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

from transformers import (
    CONFIG_MAPPING,
    IMAGE_PROCESSOR_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    CLIPConfig,
    CLIPImageProcessor,
)
from transformers.testing_utils import DUMMY_UNKNOWN_IDENTIFIER, get_tests_dir


sys.path.append(str(Path(__file__).parent.parent.parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig  # noqa E402
from test_module.custom_image_processing import CustomImageProcessor  # noqa E402


SAMPLE_IMAGE_PROCESSING_CONFIG_DIR = get_tests_dir("fixtures")
SAMPLE_IMAGE_PROCESSING_CONFIG = get_tests_dir("fixtures/dummy_image_processor_config.json")


class AutoImageProcessorTest(unittest.TestCase):
    def test_image_processor_from_model_shortcut(self):
        config = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.assertIsInstance(config, CLIPImageProcessor)

    def test_image_processor_from_local_directory_from_key(self):
        config = AutoImageProcessor.from_pretrained(
            SAMPLE_IMAGE_PROCESSING_CONFIG_DIR, _configuration_file="image_processor_config.json"
        )
        self.assertIsInstance(config, CLIPImageProcessor)

    def test_image_processor_from_local_directory_from_config(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_config = CLIPConfig()

            # remove image_processor_type to make sure config.json alone is enough to load image processor locally
            config_dict = AutoImageProcessor.from_pretrained(
                SAMPLE_IMAGE_PROCESSING_CONFIG_DIR, _configuration_file="image_processor_config.json"
            ).to_dict()

            config_dict.pop("image_processor_type")
            config = CLIPImageProcessor(**config_dict)

            # save in new folder
            model_config.save_pretrained(tmpdirname)
            config.save_pretrained(tmpdirname)

            config = AutoImageProcessor.from_pretrained(tmpdirname)

            # make sure private variable is not incorrectly saved
            dict_as_saved = json.loads(config.to_json_string())
            self.assertTrue("_processor_class" not in dict_as_saved)

        self.assertIsInstance(config, CLIPImageProcessor)

    def test_image_processor_from_local_file(self):
        config = AutoImageProcessor.from_pretrained(SAMPLE_IMAGE_PROCESSING_CONFIG)
        self.assertIsInstance(config, CLIPImageProcessor)

    def test_repo_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, "clip-base is not a local folder and is not a valid model identifier"
        ):
            _ = AutoImageProcessor.from_pretrained("clip-base")

    def test_revision_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, r"aaaaaa is not a valid git identifier \(branch name, tag name or commit id\)"
        ):
            _ = AutoImageProcessor.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, revision="aaaaaa")

    def test_image_processor_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError,
            "hf-internal-testing/config-no-model does not appear to have a file named preprocessor_config.json.",
        ):
            _ = AutoImageProcessor.from_pretrained("hf-internal-testing/config-no-model")

    def test_from_pretrained_dynamic_image_processor(self):
        model = AutoImageProcessor.from_pretrained(
            "hf-internal-testing/test_dynamic_image_processor", trust_remote_code=True
        )
        self.assertEqual(model.__class__.__name__, "NewImageProcessor")

    def test_new_image_processor_registration(self):
        try:
            AutoConfig.register("custom", CustomConfig)
            AutoImageProcessor.register(CustomConfig, CustomImageProcessor)
            # Trying to register something existing in the Transformers library will raise an error
            with self.assertRaises(ValueError):
                AutoImageProcessor.register(CLIPConfig, CLIPImageProcessor)

            # Now that the config is registered, it can be used as any other config with the auto-API
            image_processor = CustomImageProcessor.from_pretrained(SAMPLE_IMAGE_PROCESSING_CONFIG_DIR)
            with tempfile.TemporaryDirectory() as tmp_dir:
                image_processor.save_pretrained(tmp_dir)
                new_image_processor = AutoImageProcessor.from_pretrained(tmp_dir)
                self.assertIsInstance(new_image_processor, CustomImageProcessor)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in IMAGE_PROCESSOR_MAPPING._extra_content:
                del IMAGE_PROCESSOR_MAPPING._extra_content[CustomConfig]
