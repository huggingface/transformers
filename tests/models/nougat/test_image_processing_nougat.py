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


import unittest

import numpy as np
from huggingface_hub import hf_hub_download

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import cached_property, is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import NougatImageProcessor

    if is_torchvision_available():
        from transformers import NougatImageProcessorFast


class NougatImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_crop_margin=True,
        do_resize=True,
        size=None,
        do_thumbnail=True,
        do_align_long_axis: bool = False,
        do_pad=True,
        do_normalize: bool = True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ):
        size = size if size is not None else {"height": 20, "width": 20}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_crop_margin = do_crop_margin
        self.do_resize = do_resize
        self.size = size
        self.do_thumbnail = do_thumbnail
        self.do_align_long_axis = do_align_long_axis
        self.do_pad = do_pad
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.data_format = "channels_first"
        self.input_data_format = "channels_first"

    def prepare_image_processor_dict(self):
        return {
            "do_crop_margin": self.do_crop_margin,
            "do_resize": self.do_resize,
            "size": self.size,
            "do_thumbnail": self.do_thumbnail,
            "do_align_long_axis": self.do_align_long_axis,
            "do_pad": self.do_pad,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]

    def prepare_dummy_image(self):
        filepath = hf_hub_download(
            repo_id="hf-internal-testing/fixtures_docvqa", filename="nougat_pdf.png", repo_type="dataset"
        )
        image = Image.open(filepath).convert("RGB")
        return image

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


def to_channels_first(image):
    """
    Converts a NumPy image from channels-last (H, W, C) to channels-first (C, H, W)
    if needed. Leaves PyTorch tensors and non-NumPy types unchanged.
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[-1] in [1, 3]:
            return torch.tensor(np.transpose(image, (2, 0, 1)))
        return torch.tensor(image)
    return image


@require_torch
@require_vision
class NougatImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = NougatImageProcessor if is_vision_available() else None
    fast_image_processing_class = NougatImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = NougatImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    @cached_property
    def image_processor(self):
        return self.image_processing_class(**self.image_processor_dict)

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 20, "width": 20})

    def test_expected_output(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            dummy_image = self.image_processor_tester.prepare_dummy_image()
            inputs = image_processor(dummy_image, return_tensors="pt")
            torch.testing.assert_close(inputs["pixel_values"].mean(), torch.tensor(0.4906), rtol=1e-3, atol=1e-3)

    def test_crop_margin_all_white(self):
        for image_processing_class in self.image_processor_list:
            image = torch.ones((3, 100, 100), dtype=torch.uint8) * 255
            image_processor = image_processing_class(**self.image_processor_dict)
            cropped_image = image_processor.crop_margin(image)
            cropped_image = to_channels_first(cropped_image)
            self.assertTrue(torch.equal(image, cropped_image))

    def test_crop_margin_centered_black_square(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            image = torch.ones((3, 100, 100)) * 255
            image[:, 45:55, 45:55] = 0
            cropped_image = image_processor.crop_margin(image)
            expected_cropped = image[:, 45:55, 45:55]
            cropped_image = to_channels_first(cropped_image)
            self.assertTrue(torch.equal(expected_cropped, cropped_image))

    def test_align_long_axis_no_rotation(self):
        for image_processing_class in self.image_processor_list:
            if image_processing_class.__name__ == "image_processing_class":
                size = {"height": 200, "width": 300}
            elif image_processing_class.__name__ == "fast_image_processing_class":
                size = SizeDict(height=200, width=300)
            else:
                continue

            image = np.uint8(np.ones((3, 100, 200)) * 255)
            image_processor = image_processing_class(**self.image_processor_dict)
            aligned_image = image_processor.align_long_axis(image, size)

            aligned_image = to_channels_first(aligned_image)
            self.assertEqual(torch.tensor(image).shape, aligned_image.shape)

    def test_align_long_axis_with_rotation(self):
        for image_processing_class in self.image_processor_list:
            if image_processing_class.__name__ == "image_processing_class":
                size = {"height": 300, "width": 200}
            elif image_processing_class.__name__ == "fast_image_processing_class":
                size = SizeDict(height=300, width=200)
            else:
                continue

            image = np.uint8(np.ones((3, 200, 100)) * 255)
            image_processor = image_processing_class(**self.image_processor_dict)
            aligned_image = image_processor.align_long_axis(image, size)

            aligned_image = to_channels_first(aligned_image)
            self.assertEqual(torch.size([3, 200, 100]), aligned_image.shape)

    def test_align_long_axis_data_format(self):
        for image_processing_class in self.image_processor_list:
            if image_processing_class.__name__ == "image_processing_class":
                size = {"height": 300, "width": 200}
            elif image_processing_class.__name__ == "fast_image_processing_class":
                size = SizeDict(height=300, width=200)
            else:
                continue

            image = np.uint8(np.ones((3, 100, 200)) * 255)
            image_processor = image_processing_class(**self.image_processor_dict)
            aligned_image = image_processor.align_long_axis(image, size)

            aligned_image = to_channels_first(aligned_image)
            self.assertEqual(torch.size([3, 100, 200]), aligned_image.shape)

    def prepare_dummy_np_image(self):
        filepath = hf_hub_download(
            repo_id="hf-internal-testing/fixtures_docvqa", filename="nougat_pdf.png", repo_type="dataset"
        )
        image = Image.open(filepath).convert("RGB")
        return np.array(image)

    def test_crop_margin_equality_cv2_python(self):
        for image_processing_class in self.image_processor_list:
            image = self.prepare_dummy_np_image()
            image_processor = self.image_processor
            image_cropped_python = image_processor.crop_margin(image)
            image_cropped_python = to_channels_first(image_cropped_python)
            self.assertEqual(image_cropped_python.shape, torch.Size([3, 850, 685]))
            self.assertAlmostEqual(image_cropped_python.float().mean().item(), 237.43881150708458, delta=0.001)

    def test_call_numpy_4_channels(self):
        # Test that can process images which have an arbitrary number of channels
        # Initialize image_processing
        image_processor = self.image_processing_class(**self.image_processor_dict)

        # create random numpy tensors
        self.image_processor_tester.num_channels = 4
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)

        # Test not batched input
        encoded_images = image_processor(
            image_inputs[0],
            return_tensors="pt",
            input_data_format="channels_last",
            image_mean=0,
            image_std=1,
        ).pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
        self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

        # Test batched
        encoded_images = image_processor(
            image_inputs,
            return_tensors="pt",
            input_data_format="channels_last",
            image_mean=0,
            image_std=1,
        ).pixel_values
        expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
        self.assertEqual(
            tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
        )
