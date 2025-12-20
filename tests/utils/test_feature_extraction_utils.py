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

import httpx
import numpy as np

from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.testing_utils import TOKEN, TemporaryHubRepo, get_tests_dir, is_staging_test, require_torch
from transformers.utils import is_torch_available


sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))

from test_module.custom_feature_extraction import CustomFeatureExtractor  # noqa E402


if is_torch_available():
    import torch


SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR = get_tests_dir("fixtures")


class BatchFeatureTester(unittest.TestCase):
    """Tests for the BatchFeature class and tensor conversion."""

    def test_batch_feature_basic_access_and_no_conversion(self):
        """Test basic dict/attribute access and no conversion when tensor_type=None."""
        data = {"input_values": [[1, 2, 3], [4, 5, 6]], "labels": [0, 1]}
        batch = BatchFeature(data)

        # Dict-style and attribute-style access
        self.assertEqual(batch["input_values"], [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(batch.labels, [0, 1])

        # No conversion without tensor_type
        self.assertIsInstance(batch["input_values"], list)

    @require_torch
    def test_batch_feature_numpy_conversion(self):
        """Test conversion to numpy arrays from lists and existing numpy arrays."""
        # From lists
        batch = BatchFeature({"input_values": [[1, 2, 3], [4, 5, 6]]}, tensor_type="np")
        self.assertIsInstance(batch["input_values"], np.ndarray)
        self.assertEqual(batch["input_values"].shape, (2, 3))

        # From numpy arrays (should remain numpy)
        numpy_data = np.array([[1, 2, 3], [4, 5, 6]])
        batch_arrays = BatchFeature({"input_values": numpy_data}, tensor_type="np")
        np.testing.assert_array_equal(batch_arrays["input_values"], numpy_data)

        # From list of numpy arrays with same shape should stack
        numpy_data = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]])]
        batch_stacked = BatchFeature({"input_values": numpy_data}, tensor_type="np")
        np.testing.assert_array_equal(
            batch_stacked["input_values"], np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        )

        # from tensor
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        batch_tensor = BatchFeature({"input_values": tensor}, tensor_type="np")
        np.testing.assert_array_equal(batch_tensor["input_values"], tensor.numpy())

        # from list of tensors with same shape should stack
        tensors = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[7, 8, 9], [10, 11, 12]])]
        batch_stacked = BatchFeature({"input_values": tensors}, tensor_type="np")
        self.assertIsInstance(batch_stacked["input_values"], np.ndarray)
        np.testing.assert_array_equal(
            batch_stacked["input_values"], np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        )

    @require_torch
    def test_batch_feature_pytorch_conversion(self):
        """Test conversion to PyTorch tensors from various input types."""
        # From lists
        batch = BatchFeature({"input_values": [[1, 2, 3], [4, 5, 6]]}, tensor_type="pt")
        self.assertIsInstance(batch["input_values"], torch.Tensor)
        self.assertEqual(batch["input_values"].shape, (2, 3))

        # from tensor (should be returned as-is)
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        batch_tensor = BatchFeature({"input_values": tensor}, tensor_type="pt")
        torch.testing.assert_close(batch_tensor["input_values"], tensor)

        # From numpy arrays
        batch_numpy = BatchFeature({"input_values": np.array([[1, 2]])}, tensor_type="pt")
        self.assertIsInstance(batch_numpy["input_values"], torch.Tensor)

        # List of same-shape tensors should stack
        tensors = [torch.randn(3, 10, 10) for _ in range(3)]
        batch_stacked = BatchFeature({"pixel_values": tensors}, tensor_type="pt")
        self.assertEqual(batch_stacked["pixel_values"].shape, (3, 3, 10, 10))

        # List of same-shape numpy arrays should stack
        numpy_arrays = [np.random.randn(3, 10, 10) for _ in range(3)]
        batch_stacked = BatchFeature({"pixel_values": numpy_arrays}, tensor_type="pt")
        self.assertIsInstance(batch_stacked["pixel_values"], torch.Tensor)
        self.assertEqual(batch_stacked["pixel_values"].shape, (3, 3, 10, 10))

    @require_torch
    def test_batch_feature_error_handling(self):
        """Test clear error messages for common conversion failures."""
        # Ragged tensors (different shapes)
        data_ragged = {"values": [torch.randn(3, 224, 224), torch.randn(3, 448, 448)]}
        with self.assertRaises(ValueError) as context:
            BatchFeature(data_ragged, tensor_type="pt")
        error_msg = str(context.exception)
        self.assertIn("stack expects each tensor to be equal size", error_msg.lower())
        self.assertIn("return_tensors=None", error_msg)

        # Ragged numpy arrays (different shapes)
        data_ragged = {"values": [np.random.randn(3, 224, 224), np.random.randn(3, 448, 448)]}
        with self.assertRaises(ValueError) as context:
            BatchFeature(data_ragged, tensor_type="np")
        error_msg = str(context.exception)
        self.assertIn("inhomogeneous", error_msg.lower())
        self.assertIn("return_tensors=None", error_msg)

    @require_torch
    def test_batch_feature_auto_skip_non_array_like(self):
        """Test that non-array-like values are automatically skipped during tensor conversion."""
        data = {
            "values": [[1, 2]],
            "metadata": {"key": "val"},
            "image_path": "/path/to/image.jpg",
            "tags": ["tag1", "tag2"],
            "extra": None,
        }
        batch = BatchFeature(data, tensor_type="pt")

        # values should be converted
        self.assertIsInstance(batch["values"], torch.Tensor)

        # Non-array-like values should remain unchanged
        self.assertIsInstance(batch["metadata"], dict)
        self.assertEqual(batch["metadata"], {"key": "val"})
        self.assertIsInstance(batch["image_path"], str)
        self.assertIsInstance(batch["tags"], list)
        self.assertEqual(batch["tags"], ["tag1", "tag2"])
        self.assertIsNone(batch["extra"])

    @require_torch
    def test_batch_feature_skip_tensor_conversion(self):
        """Test skip_tensor_conversion parameter for metadata fields."""
        import torch

        data = {"pixel_values": [[1, 2, 3]], "num_crops": [1, 2], "sizes": [(224, 224)]}
        batch = BatchFeature(data, tensor_type="pt", skip_tensor_conversion=["num_crops", "sizes"])

        # pixel_values should be converted
        self.assertIsInstance(batch["pixel_values"], torch.Tensor)
        # num_crops and sizes should remain as lists
        self.assertIsInstance(batch["num_crops"], list)
        self.assertIsInstance(batch["sizes"], list)

    @require_torch
    def test_batch_feature_convert_to_tensors_method(self):
        """Test convert_to_tensors method can be called after initialization."""
        import torch

        data = {"input_values": [[1, 2, 3]], "metadata": [1, 2]}
        batch = BatchFeature(data)  # No conversion initially
        self.assertIsInstance(batch["input_values"], list)

        # Convert with skip parameter
        batch.convert_to_tensors(tensor_type="pt", skip_tensor_conversion=["metadata"])
        self.assertIsInstance(batch["input_values"], torch.Tensor)
        self.assertIsInstance(batch["metadata"], list)

    @require_torch
    def test_batch_feature_to_with_nested_tensors(self):
        """Test .to() method works recursively with nested lists and tuples of tensors."""
        batch = BatchFeature(
            {
                "list_tensors": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
                "nested_list": [[torch.tensor([1.0]), torch.tensor([2.0])]],
                "tuple_tensors": (torch.tensor([5.0]), torch.tensor([6.0])),
            }
        )

        batch_fp16 = batch.to(torch.float16)

        # Check lists of tensors are converted
        self.assertIsInstance(batch_fp16["list_tensors"], list)
        self.assertEqual(batch_fp16["list_tensors"][0].dtype, torch.float16)
        self.assertEqual(batch_fp16["list_tensors"][1].dtype, torch.float16)

        # Check nested lists are converted
        self.assertIsInstance(batch_fp16["nested_list"][0], list)
        self.assertEqual(batch_fp16["nested_list"][0][0].dtype, torch.float16)

        # Check tuples are preserved and converted
        self.assertIsInstance(batch_fp16["tuple_tensors"], tuple)
        self.assertEqual(batch_fp16["tuple_tensors"][0].dtype, torch.float16)


class FeatureExtractorUtilTester(unittest.TestCase):
    def test_cached_files_are_used_when_internet_is_down(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            "failed", request=mock.Mock(), response=mock.Mock()
        )
        response_mock.json.return_value = {}

        # Download this model to make sure it's in the cache.
        _ = Wav2Vec2FeatureExtractor.from_pretrained("hf-internal-testing/tiny-random-wav2vec2")
        # Under the mock environment we get a 500 error when trying to reach the model.
        with mock.patch("httpx.Client.request", return_value=response_mock) as mock_head:
            _ = Wav2Vec2FeatureExtractor.from_pretrained("hf-internal-testing/tiny-random-wav2vec2")
            # This check we did call the fake head request
            mock_head.assert_called()


@is_staging_test
class FeatureExtractorPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN

    def test_push_to_hub(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
            feature_extractor.push_to_hub(tmp_repo.repo_id, token=self._token)

            new_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(tmp_repo.repo_id)
            for k, v in feature_extractor.__dict__.items():
                self.assertEqual(v, getattr(new_feature_extractor, k))

    def test_push_to_hub_via_save_pretrained(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                feature_extractor.save_pretrained(
                    tmp_dir, repo_id=tmp_repo.repo_id, push_to_hub=True, token=self._token
                )

            new_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(tmp_repo.repo_id)
            for k, v in feature_extractor.__dict__.items():
                self.assertEqual(v, getattr(new_feature_extractor, k))

    def test_push_to_hub_in_organization(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
            feature_extractor.push_to_hub(tmp_repo.repo_id, token=self._token)

            new_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(tmp_repo.repo_id)
            for k, v in feature_extractor.__dict__.items():
                self.assertEqual(v, getattr(new_feature_extractor, k))

    def test_push_to_hub_in_organization_via_save_pretrained(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                feature_extractor.save_pretrained(
                    tmp_dir, repo_id=tmp_repo.repo_id, push_to_hub=True, token=self._token
                )

            new_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(tmp_repo.repo_id)
            for k, v in feature_extractor.__dict__.items():
                self.assertEqual(v, getattr(new_feature_extractor, k))

    def test_push_to_hub_dynamic_feature_extractor(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            CustomFeatureExtractor.register_for_auto_class()
            feature_extractor = CustomFeatureExtractor.from_pretrained(SAMPLE_FEATURE_EXTRACTION_CONFIG_DIR)

            feature_extractor.push_to_hub(tmp_repo.repo_id, token=self._token)

            # This has added the proper auto_map field to the config
            self.assertDictEqual(
                feature_extractor.auto_map,
                {"AutoFeatureExtractor": "custom_feature_extraction.CustomFeatureExtractor"},
            )

            new_feature_extractor = AutoFeatureExtractor.from_pretrained(tmp_repo.repo_id, trust_remote_code=True)
            # Can't make an isinstance check because the new_feature_extractor is from the CustomFeatureExtractor class of a dynamic module
            self.assertEqual(new_feature_extractor.__class__.__name__, "CustomFeatureExtractor")
