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


import json
import os
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

from huggingface_hub import HfFolder, delete_repo, set_access_token
from requests.exceptions import HTTPError

from transformers import AutoImageProcessor, ViTImageProcessor
from transformers.testing_utils import (
    TOKEN,
    USER,
    check_json_file_has_correct_format,
    get_tests_dir,
    is_staging_test,
    require_torch,
    require_vision,
)
from transformers.utils import is_torch_available, is_vision_available


sys.path.append(str(Path(__file__).parent.parent / "utils"))

from test_module.custom_image_processing import CustomImageProcessor  # noqa E402


if is_torch_available():
    import numpy as np
    import torch

if is_vision_available():
    from PIL import Image


SAMPLE_IMAGE_PROCESSING_CONFIG_DIR = get_tests_dir("fixtures")


def prepare_image_inputs(image_processor_tester, equal_resolution=False, numpify=False, torchify=False):
    """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
    or a list of PyTorch tensors if one specifies torchify=True.

    One can specify whether the images are of the same resolution or not.
    """

    assert not (numpify and torchify), "You cannot specify both numpy and PyTorch tensors at the same time"

    image_inputs = []
    for i in range(image_processor_tester.batch_size):
        if equal_resolution:
            width = height = image_processor_tester.max_resolution
        else:
            # To avoid getting image width/height 0
            min_resolution = image_processor_tester.min_resolution
            if getattr(image_processor_tester, "size_divisor", None):
                # If `size_divisor` is defined, the image needs to have width/size >= `size_divisor`
                min_resolution = max(image_processor_tester.size_divisor, min_resolution)
            width, height = np.random.choice(np.arange(min_resolution, image_processor_tester.max_resolution), 2)
        image_inputs.append(
            np.random.randint(255, size=(image_processor_tester.num_channels, width, height), dtype=np.uint8)
        )

    if not numpify and not torchify:
        # PIL expects the channel dimension as last dimension
        image_inputs = [Image.fromarray(np.moveaxis(image, 0, -1)) for image in image_inputs]

    if torchify:
        image_inputs = [torch.from_numpy(image) for image in image_inputs]

    return image_inputs


def prepare_video(image_processor_tester, width=10, height=10, numpify=False, torchify=False):
    """This function prepares a video as a list of PIL images/NumPy arrays/PyTorch tensors."""

    video = []
    for i in range(image_processor_tester.num_frames):
        video.append(np.random.randint(255, size=(image_processor_tester.num_channels, width, height), dtype=np.uint8))

    if not numpify and not torchify:
        # PIL expects the channel dimension as last dimension
        video = [Image.fromarray(np.moveaxis(frame, 0, -1)) for frame in video]

    if torchify:
        video = [torch.from_numpy(frame) for frame in video]

    return video


def prepare_video_inputs(image_processor_tester, equal_resolution=False, numpify=False, torchify=False):
    """This function prepares a batch of videos: a list of list of PIL images, or a list of list of numpy arrays if
    one specifies numpify=True, or a list of list of PyTorch tensors if one specifies torchify=True.

    One can specify whether the videos are of the same resolution or not.
    """

    assert not (numpify and torchify), "You cannot specify both numpy and PyTorch tensors at the same time"

    video_inputs = []
    for i in range(image_processor_tester.batch_size):
        if equal_resolution:
            width = height = image_processor_tester.max_resolution
        else:
            width, height = np.random.choice(
                np.arange(image_processor_tester.min_resolution, image_processor_tester.max_resolution), 2
            )
            video = prepare_video(
                image_processor_tester=image_processor_tester,
                width=width,
                height=height,
                numpify=numpify,
                torchify=torchify,
            )
        video_inputs.append(video)

    return video_inputs


class ImageProcessingSavingTestMixin:
    test_cast_dtype = None

    def test_image_processor_to_json_string(self):
        image_processor = self.image_processing_class(**self.image_processor_dict)
        obj = json.loads(image_processor.to_json_string())
        for key, value in self.image_processor_dict.items():
            self.assertEqual(obj[key], value)

    def test_image_processor_to_json_file(self):
        image_processor_first = self.image_processing_class(**self.image_processor_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "image_processor.json")
            image_processor_first.to_json_file(json_file_path)
            image_processor_second = self.image_processing_class.from_json_file(json_file_path)

        self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

    def test_image_processor_from_and_save_pretrained(self):
        image_processor_first = self.image_processing_class(**self.image_processor_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = image_processor_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            image_processor_second = self.image_processing_class.from_pretrained(tmpdirname)

        self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

    def test_init_without_params(self):
        image_processor = self.image_processing_class()
        self.assertIsNotNone(image_processor)

    @require_torch
    @require_vision
    def test_cast_dtype_device(self):
        if self.test_cast_dtype is not None:
            # Initialize image_processor
            image_processor = self.image_processing_class(**self.image_processor_dict)

            # create random PyTorch tensors
            image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False, torchify=True)

            encoding = image_processor(image_inputs, return_tensors="pt")
            # for layoutLM compatiblity
            self.assertEqual(encoding.pixel_values.device, torch.device("cpu"))
            self.assertEqual(encoding.pixel_values.dtype, torch.float32)

            encoding = image_processor(image_inputs, return_tensors="pt").to(torch.float16)
            self.assertEqual(encoding.pixel_values.device, torch.device("cpu"))
            self.assertEqual(encoding.pixel_values.dtype, torch.float16)

            encoding = image_processor(image_inputs, return_tensors="pt").to("cpu", torch.bfloat16)
            self.assertEqual(encoding.pixel_values.device, torch.device("cpu"))
            self.assertEqual(encoding.pixel_values.dtype, torch.bfloat16)

            with self.assertRaises(TypeError):
                _ = image_processor(image_inputs, return_tensors="pt").to(torch.bfloat16, "cpu")

            # Try with text + image feature
            encoding = image_processor(image_inputs, return_tensors="pt")
            encoding.update({"input_ids": torch.LongTensor([[1, 2, 3], [4, 5, 6]])})
            encoding = encoding.to(torch.float16)

            self.assertEqual(encoding.pixel_values.device, torch.device("cpu"))
            self.assertEqual(encoding.pixel_values.dtype, torch.float16)
            self.assertEqual(encoding.input_ids.dtype, torch.long)


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
        with mock.patch("requests.request", return_value=response_mock) as mock_head:
            _ = ViTImageProcessor.from_pretrained("hf-internal-testing/tiny-random-vit")
            # This check we did call the fake head request
            mock_head.assert_called()

    def test_legacy_load_from_url(self):
        # This test is for deprecated behavior and can be removed in v5
        _ = ViTImageProcessor.from_pretrained(
            "https://huggingface.co/hf-internal-testing/tiny-random-vit/resolve/main/preprocessor_config.json"
        )


@is_staging_test
class ImageProcessorPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        set_access_token(TOKEN)
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
            {"ImageProcessor": "custom_image_processing.CustomImageProcessor"},
        )

        new_image_processor = AutoImageProcessor.from_pretrained(
            f"{USER}/test-dynamic-image-processor", trust_remote_code=True
        )
        # Can't make an isinstance check because the new_image_processor is from the CustomImageProcessor class of a dynamic module
        self.assertEqual(new_image_processor.__class__.__name__, "CustomImageProcessor")
