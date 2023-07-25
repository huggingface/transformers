# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

from huggingface_hub import HfFolder, delete_repo
from requests.exceptions import HTTPError

from transformers import AutoImageProcessor, ViTImageProcessor
from transformers.testing_utils import TOKEN, USER, get_tests_dir, is_staging_test


sys.path.append(str(Path(__file__).parent.parent / "utils"))

from test_module.custom_image_processing import CustomImageProcessor  # noqa E402


SAMPLE_IMAGE_PROCESSING_CONFIG_DIR = get_tests_dir("fixtures")


class ImageProcessorUtilTester(unittest.TestCase):
    def test_cached_files_are_used_when_internet_is_down(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}

        # Download this model to make sure it's in the cache.
        _ = ViTImageProcessor.from_pretrained("hf-internal-testing/tiny-random-vit")
        # Under the mock environment we get a 500 error when trying to reach the model.
        with mock.patch("requests.Session.request", return_value=response_mock) as mock_head:
            _ = ViTImageProcessor.from_pretrained("hf-internal-testing/tiny-random-vit")
            # This check we did call the fake head request
            mock_head.assert_called()

    def test_legacy_load_from_url(self):
        # This test is for deprecated behavior and can be removed in v5
        _ = ViTImageProcessor.from_pretrained(
            "https://huggingface.co/hf-internal-testing/tiny-random-vit/resolve/main/preprocessor_config.json"
        )

    def test_image_processor_from_pretrained_subfolder(self):
        with self.assertRaises(OSError):
            # config is in subfolder, the following should not work without specifying the subfolder
            _ = AutoImageProcessor.from_pretrained("hf-internal-testing/stable-diffusion-all-variants")

        config = AutoImageProcessor.from_pretrained(
            "hf-internal-testing/stable-diffusion-all-variants", subfolder="feature_extractor"
        )

        self.assertIsNotNone(config)


@is_staging_test
class ImageProcessorPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    @classmethod
    def tearDownClass(cls):
        try:
            delete_repo(token=cls._token, repo_id="test-image-processor")
        except HTTPError:
            pass

        try:
            delete_repo(token=cls._token, repo_id="valid_org/test-image-processor-org")
        except HTTPError:
            pass

        try:
            delete_repo(token=cls._token, repo_id="test-dynamic-image-processor")
        except HTTPError:
            pass

    def test_push_to_hub(self):
        image_processor = ViTImageProcessor.from_pretrained(SAMPLE_IMAGE_PROCESSING_CONFIG_DIR)
        image_processor.push_to_hub("test-image-processor", use_auth_token=self._token)

        new_image_processor = ViTImageProcessor.from_pretrained(f"{USER}/test-image-processor")
        for k, v in image_processor.__dict__.items():
            self.assertEqual(v, getattr(new_image_processor, k))

        # Reset repo
        delete_repo(token=self._token, repo_id="test-image-processor")

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_processor.save_pretrained(
                tmp_dir, repo_id="test-image-processor", push_to_hub=True, use_auth_token=self._token
            )

        new_image_processor = ViTImageProcessor.from_pretrained(f"{USER}/test-image-processor")
        for k, v in image_processor.__dict__.items():
            self.assertEqual(v, getattr(new_image_processor, k))

    def test_push_to_hub_in_organization(self):
        image_processor = ViTImageProcessor.from_pretrained(SAMPLE_IMAGE_PROCESSING_CONFIG_DIR)
        image_processor.push_to_hub("valid_org/test-image-processor", use_auth_token=self._token)

        new_image_processor = ViTImageProcessor.from_pretrained("valid_org/test-image-processor")
        for k, v in image_processor.__dict__.items():
            self.assertEqual(v, getattr(new_image_processor, k))

        # Reset repo
        delete_repo(token=self._token, repo_id="valid_org/test-image-processor")

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_processor.save_pretrained(
                tmp_dir, repo_id="valid_org/test-image-processor-org", push_to_hub=True, use_auth_token=self._token
            )

        new_image_processor = ViTImageProcessor.from_pretrained("valid_org/test-image-processor-org")
        for k, v in image_processor.__dict__.items():
            self.assertEqual(v, getattr(new_image_processor, k))

    def test_push_to_hub_dynamic_image_processor(self):
        CustomImageProcessor.register_for_auto_class()
        image_processor = CustomImageProcessor.from_pretrained(SAMPLE_IMAGE_PROCESSING_CONFIG_DIR)

        image_processor.push_to_hub("test-dynamic-image-processor", use_auth_token=self._token)

        # This has added the proper auto_map field to the config
        self.assertDictEqual(
            image_processor.auto_map,
            {"AutoImageProcessor": "custom_image_processing.CustomImageProcessor"},
        )

        new_image_processor = AutoImageProcessor.from_pretrained(
            f"{USER}/test-dynamic-image-processor", trust_remote_code=True
        )
        # Can't make an isinstance check because the new_image_processor is from the CustomImageProcessor class of a dynamic module
        self.assertEqual(new_image_processor.__class__.__name__, "CustomImageProcessor")
