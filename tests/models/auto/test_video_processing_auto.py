# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team.
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
    VIDEO_PROCESSOR_MAPPING,
    AutoConfig,
    AutoVideoProcessor,
    LlavaOnevisionConfig,
    LlavaOnevisionVideoProcessor,
)
from transformers.testing_utils import DUMMY_UNKNOWN_IDENTIFIER


sys.path.append(str(Path(__file__).parent.parent.parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig  # noqa E402
from test_module.custom_video_processing import CustomVideoProcessor  # noqa E402


class AutoVideoProcessorTest(unittest.TestCase):
    def setUp(self):
        transformers.dynamic_module_utils.TIME_OUT_REMOTE_CODE = 0

    def test_video_processor_from_model_shortcut(self):
        config = AutoVideoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
        self.assertIsInstance(config, LlavaOnevisionVideoProcessor)

    def test_video_processor_from_local_directory_from_key(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor_tmpfile = Path(tmpdirname) / "video_preprocessor_config.json"
            config_tmpfile = Path(tmpdirname) / "config.json"
            json.dump(
                {"video_processor_type": "LlavaOnevisionVideoProcessor", "processor_class": "LlavaOnevisionProcessor"},
                open(processor_tmpfile, "w"),
            )
            json.dump({"model_type": "llava_onevision"}, open(config_tmpfile, "w"))

            config = AutoVideoProcessor.from_pretrained(tmpdirname)
            self.assertIsInstance(config, LlavaOnevisionVideoProcessor)

    def test_video_processor_from_local_directory_from_preprocessor_key(self):
        # Ensure we can load the image processor from the feature extractor config
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor_tmpfile = Path(tmpdirname) / "preprocessor_config.json"
            config_tmpfile = Path(tmpdirname) / "config.json"
            json.dump(
                {"video_processor_type": "LlavaOnevisionVideoProcessor", "processor_class": "LlavaOnevisionProcessor"},
                open(processor_tmpfile, "w"),
            )
            json.dump({"model_type": "llava_onevision"}, open(config_tmpfile, "w"))

            config = AutoVideoProcessor.from_pretrained(tmpdirname)
            self.assertIsInstance(config, LlavaOnevisionVideoProcessor)

    def test_video_processor_from_local_directory_from_config(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_config = LlavaOnevisionConfig()

            # Create a dummy config file with image_proceesor_type
            processor_tmpfile = Path(tmpdirname) / "video_preprocessor_config.json"
            config_tmpfile = Path(tmpdirname) / "config.json"
            json.dump(
                {"video_processor_type": "LlavaOnevisionVideoProcessor", "processor_class": "LlavaOnevisionProcessor"},
                open(processor_tmpfile, "w"),
            )
            json.dump({"model_type": "llava_onevision"}, open(config_tmpfile, "w"))

            # remove video_processor_type to make sure config.json alone is enough to load image processor locally
            config_dict = AutoVideoProcessor.from_pretrained(tmpdirname).to_dict()

            config_dict.pop("video_processor_type")
            config = LlavaOnevisionVideoProcessor(**config_dict)

            # save in new folder
            model_config.save_pretrained(tmpdirname)
            config.save_pretrained(tmpdirname)

            config = AutoVideoProcessor.from_pretrained(tmpdirname)

            # make sure private variable is not incorrectly saved
            dict_as_saved = json.loads(config.to_json_string())
            self.assertTrue("_processor_class" not in dict_as_saved)

        self.assertIsInstance(config, LlavaOnevisionVideoProcessor)

    def test_video_processor_from_local_file(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor_tmpfile = Path(tmpdirname) / "video_preprocessor_config.json"
            json.dump(
                {"video_processor_type": "LlavaOnevisionVideoProcessor", "processor_class": "LlavaOnevisionProcessor"},
                open(processor_tmpfile, "w"),
            )

            config = AutoVideoProcessor.from_pretrained(processor_tmpfile)
            self.assertIsInstance(config, LlavaOnevisionVideoProcessor)

    def test_repo_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError,
            "llava-hf/llava-doesnt-exist is not a local folder and is not a valid model identifier",
        ):
            _ = AutoVideoProcessor.from_pretrained("llava-hf/llava-doesnt-exist")

    def test_revision_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, r"aaaaaa is not a valid git identifier \(branch name, tag name or commit id\)"
        ):
            _ = AutoVideoProcessor.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, revision="aaaaaa")

    def test_video_processor_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError,
            "hf-internal-testing/config-no-model does not appear to have a file named video_preprocessor_config.json.",
        ):
            _ = AutoVideoProcessor.from_pretrained("hf-internal-testing/config-no-model")

    def test_from_pretrained_dynamic_video_processor(self):
        # If remote code is not set, we will time out when asking whether to load the model.
        with self.assertRaises(ValueError):
            video_processor = AutoVideoProcessor.from_pretrained("hf-internal-testing/test_dynamic_video_processor")
        # If remote code is disabled, we can't load this config.
        with self.assertRaises(ValueError):
            video_processor = AutoVideoProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_video_processor", trust_remote_code=False
            )

        video_processor = AutoVideoProcessor.from_pretrained(
            "hf-internal-testing/test_dynamic_video_processor", trust_remote_code=True
        )
        self.assertEqual(video_processor.__class__.__name__, "NewVideoProcessor")

        # Test the dynamic module is loaded only once.
        reloaded_video_processor = AutoVideoProcessor.from_pretrained(
            "hf-internal-testing/test_dynamic_video_processor", trust_remote_code=True
        )
        self.assertIs(video_processor.__class__, reloaded_video_processor.__class__)

        # Test image processor can be reloaded.
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_processor.save_pretrained(tmp_dir)
            reloaded_video_processor = AutoVideoProcessor.from_pretrained(tmp_dir, trust_remote_code=True)
        self.assertEqual(reloaded_video_processor.__class__.__name__, "NewVideoProcessor")

        # The image processor file is cached in the snapshot directory. So the module file is not changed after dumping
        # to a temp dir. Because the revision of the module file is not changed.
        # Test the dynamic module is loaded only once if the module file is not changed.
        self.assertIs(video_processor.__class__, reloaded_video_processor.__class__)

        # Test the dynamic module is reloaded if we force it.
        reloaded_video_processor = AutoVideoProcessor.from_pretrained(
            "hf-internal-testing/test_dynamic_video_processor", trust_remote_code=True, force_download=True
        )
        self.assertIsNot(video_processor.__class__, reloaded_video_processor.__class__)

    def test_new_video_processor_registration(self):
        try:
            AutoConfig.register("custom", CustomConfig)
            AutoVideoProcessor.register(CustomConfig, CustomVideoProcessor)
            # Trying to register something existing in the Transformers library will raise an error
            with self.assertRaises(ValueError):
                AutoVideoProcessor.register(LlavaOnevisionConfig, LlavaOnevisionVideoProcessor)

            with tempfile.TemporaryDirectory() as tmpdirname:
                processor_tmpfile = Path(tmpdirname) / "video_preprocessor_config.json"
                config_tmpfile = Path(tmpdirname) / "config.json"
                json.dump(
                    {
                        "video_processor_type": "LlavaOnevisionVideoProcessor",
                        "processor_class": "LlavaOnevisionProcessor",
                    },
                    open(processor_tmpfile, "w"),
                )
                json.dump({"model_type": "llava_onevision"}, open(config_tmpfile, "w"))

                video_processor = CustomVideoProcessor.from_pretrained(tmpdirname)

            # Now that the config is registered, it can be used as any other config with the auto-API
            with tempfile.TemporaryDirectory() as tmp_dir:
                video_processor.save_pretrained(tmp_dir)
                new_video_processor = AutoVideoProcessor.from_pretrained(tmp_dir)
                self.assertIsInstance(new_video_processor, CustomVideoProcessor)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in VIDEO_PROCESSOR_MAPPING._extra_content:
                del VIDEO_PROCESSOR_MAPPING._extra_content[CustomConfig]

    def test_from_pretrained_dynamic_video_processor_conflict(self):
        class NewVideoProcessor(LlavaOnevisionVideoProcessor):
            is_local = True

        try:
            AutoConfig.register("custom", CustomConfig)
            AutoVideoProcessor.register(CustomConfig, NewVideoProcessor)
            # If remote code is not set, the default is to use local
            video_processor = AutoVideoProcessor.from_pretrained("hf-internal-testing/test_dynamic_video_processor")
            self.assertEqual(video_processor.__class__.__name__, "NewVideoProcessor")
            self.assertTrue(video_processor.is_local)

            # If remote code is disabled, we load the local one.
            video_processor = AutoVideoProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_video_processor", trust_remote_code=False
            )
            self.assertEqual(video_processor.__class__.__name__, "NewVideoProcessor")
            self.assertTrue(video_processor.is_local)

            # If remote is enabled, we load from the Hub
            video_processor = AutoVideoProcessor.from_pretrained(
                "hf-internal-testing/test_dynamic_video_processor", trust_remote_code=True
            )
            self.assertEqual(video_processor.__class__.__name__, "NewVideoProcessor")
            self.assertTrue(not hasattr(video_processor, "is_local"))

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            if CustomConfig in VIDEO_PROCESSOR_MAPPING._extra_content:
                del VIDEO_PROCESSOR_MAPPING._extra_content[CustomConfig]
