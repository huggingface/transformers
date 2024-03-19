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


import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

from huggingface_hub import HfFolder, delete_repo
from requests.exceptions import HTTPError

from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor
from transformers.testing_utils import TOKEN, USER, get_tests_dir, is_staging_test


sys.path.append(str(Path(__file__).parent.parent / "utils"))

from test_module.custom_feature_extraction import CustomFeatureExtractor  # noqa E402


SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR = get_tests_dir("fixtures")


class FeatureExtractorUtilTester(unittest.TestCase):
    def test_cached_files_are_used_when_internet_is_down(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}

        # Download this model to make sure it's in the cache.
        _ = Wav2Vec2FeatureExtractor.from_pretrained("hf-internal-testing/tiny-random-wav2vec2")
        # Under the mock environment we get a 500 error when trying to reach the model.
        with mock.patch("requests.Session.request", return_value=response_mock) as mock_head:
            _ = Wav2Vec2FeatureExtractor.from_pretrained("hf-internal-testing/tiny-random-wav2vec2")
            # This check we did call the fake head request
            mock_head.assert_called()


@is_staging_test
class FeatureExtractorPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    @classmethod
    def tearDownClass(cls):
        try:
            delete_repo(token=cls._token, repo_id="test-feature-extractor")
        except HTTPError:
            pass

        try:
            delete_repo(token=cls._token, repo_id="valid_org/test-feature-extractor-org")
        except HTTPError:
            pass

        try:
            delete_repo(token=cls._token, repo_id="test-dynamic-feature-extractor")
        except HTTPError:
            pass

    def test_push_to_hub(self):
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
        feature_extractor.push_to_hub("test-feature-extractor", token=self._token)

        new_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(f"{USER}/test-feature-extractor")
        for k, v in feature_extractor.__dict__.items():
            self.assertEqual(v, getattr(new_feature_extractor, k))

        # Reset repo
        delete_repo(token=self._token, repo_id="test-feature-extractor")

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            feature_extractor.save_pretrained(
                tmp_dir, repo_id="test-feature-extractor", push_to_hub=True, token=self._token
            )

        new_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(f"{USER}/test-feature-extractor")
        for k, v in feature_extractor.__dict__.items():
            self.assertEqual(v, getattr(new_feature_extractor, k))

    def test_push_to_hub_in_organization(self):
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
        feature_extractor.push_to_hub("valid_org/test-feature-extractor", token=self._token)

        new_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("valid_org/test-feature-extractor")
        for k, v in feature_extractor.__dict__.items():
            self.assertEqual(v, getattr(new_feature_extractor, k))

        # Reset repo
        delete_repo(token=self._token, repo_id="valid_org/test-feature-extractor")

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            feature_extractor.save_pretrained(
                tmp_dir, repo_id="valid_org/test-feature-extractor-org", push_to_hub=True, token=self._token
            )

        new_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("valid_org/test-feature-extractor-org")
        for k, v in feature_extractor.__dict__.items():
            self.assertEqual(v, getattr(new_feature_extractor, k))

    def test_push_to_hub_dynamic_feature_extractor(self):
        CustomFeatureExtractor.register_for_auto_class()
        feature_extractor = CustomFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)

        feature_extractor.push_to_hub("test-dynamic-feature-extractor", token=self._token)

        # This has added the proper auto_map field to the config
        self.assertDictEqual(
            feature_extractor.auto_map,
            {"AutoFeatureExtractor": "custom_feature_extraction.CustomFeatureExtractor"},
        )

        new_feature_extractor = AutoFeatureExtractor.from_pretrained(
            f"{USER}/test-dynamic-feature-extractor", trust_remote_code=True
        )
        # Can't make an isinstance check because the new_feature_extractor is from the CustomFeatureExtractor class of a dynamic module
        self.assertEqual(new_feature_extractor.__class__.__name__, "CustomFeatureExtractor")
