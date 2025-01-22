# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
from typing import Tuple, Union

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_vision_available():
    from PIL import Image

    from transformers import LlavaImageProcessor

    if is_torchvision_available():
        from torchvision.transforms import functional as F

        from transformers import LlavaImageProcessorFast


class LlavaImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_pad=True,
        do_resize=True,
        size=None,
        do_center_crop=True,
        crop_size=None,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        do_convert_rgb=True,
    ):
        super().__init__()
        size = size if size is not None else {"shortest_edge": 20}
        crop_size = crop_size if crop_size is not None else {"height": 18, "width": 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_pad = do_pad
        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_pad": self.do_pad,
            "do_resize": self.do_resize,
            "size": self.size,
            "do_center_crop": self.do_center_crop,
            "crop_size": self.crop_size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
        }

    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTester.expected_output_image_shape
    def expected_output_image_shape(self, images):
        return self.num_channels, self.crop_size["height"], self.crop_size["width"]

    # Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTester.prepare_image_inputs
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


@require_torch
@require_vision
# Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest with CLIP->Llava
class LlavaImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = LlavaImageProcessor if is_vision_available() else None
    fast_image_processing_class = LlavaImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = LlavaImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    # Ignore copy
    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_pad"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "do_center_crop"))
            self.assertTrue(hasattr(image_processing, "center_crop"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"shortest_edge": 20})
            self.assertEqual(image_processor.crop_size, {"height": 18, "width": 18})

            image_processor = self.image_processing_class.from_dict(self.image_processor_dict, size=42, crop_size=84)
            self.assertEqual(image_processor.size, {"shortest_edge": 42})
            self.assertEqual(image_processor.crop_size, {"height": 84, "width": 84})

    # Ignore copy
    def test_padding(self):
        """
        LLaVA needs to pad images to square size before processing as per orig implementation.
        Checks that image processor pads images correctly given different background colors.
        """

        # taken from original implementation: https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/mm_utils.py#L152
        def pad_to_square_original(
            image: Image.Image, background_color: Union[int, Tuple[int, int, int]] = 0
        ) -> Image.Image:
            width, height = image.size
            if width == height:
                return image
            elif width > height:
                result = Image.new(image.mode, (width, width), background_color)
                result.paste(image, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(image.mode, (height, height), background_color)
                result.paste(image, ((height - width) // 2, 0))
                return result

        for i, image_processing_class in enumerate(self.image_processor_list):
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            numpify = i == 0
            torchify = i == 1
            image_inputs = self.image_processor_tester.prepare_image_inputs(
                equal_resolution=False, numpify=numpify, torchify=torchify
            )

            # test with images in channel-last and channel-first format (only channel-first for torch)
            for image in image_inputs:
                padded_image = image_processor.pad_to_square(image)
                if i == 0:
                    padded_image_original = pad_to_square_original(Image.fromarray(image))
                    padded_image_original = np.array(padded_image_original)

                    np.testing.assert_allclose(padded_image, padded_image_original)

                    padded_image = image_processor.pad_to_square(
                        image.transpose(2, 0, 1), input_data_format="channels_first"
                    )
                    padded_image = padded_image.transpose(1, 2, 0)

                    np.testing.assert_allclose(padded_image, padded_image_original)
                else:
                    padded_image_original = pad_to_square_original(F.to_pil_image(image))
                    padded_image = padded_image.permute(1, 2, 0)
                    np.testing.assert_allclose(padded_image, padded_image_original)

            # test background color
            background_color = (122, 116, 104)
            for image in image_inputs:
                padded_image = image_processor.pad_to_square(image, background_color=background_color)
                if i == 0:
                    padded_image_original = pad_to_square_original(
                        Image.fromarray(image), background_color=background_color
                    )
                else:
                    padded_image_original = pad_to_square_original(
                        F.to_pil_image(image), background_color=background_color
                    )
                    padded_image = padded_image.permute(1, 2, 0)
                padded_image_original = np.array(padded_image_original)

                np.testing.assert_allclose(padded_image, padded_image_original)

            background_color = 122
            for image in image_inputs:
                padded_image = image_processor.pad_to_square(image, background_color=background_color)
                if i == 0:
                    padded_image_original = pad_to_square_original(
                        Image.fromarray(image), background_color=background_color
                    )
                else:
                    padded_image_original = pad_to_square_original(
                        F.to_pil_image(image), background_color=background_color
                    )
                    padded_image = padded_image.permute(1, 2, 0)
                padded_image_original = np.array(padded_image_original)
                np.testing.assert_allclose(padded_image, padded_image_original)

            # background color length should match channel length
            with self.assertRaises(ValueError):
                padded_image = image_processor.pad_to_square(image_inputs[0], background_color=(122, 104))

            with self.assertRaises(ValueError):
                padded_image = image_processor.pad_to_square(image_inputs[0], background_color=(122, 104, 0, 0))

    @unittest.skip(reason="LLaVa does not support 4 channel images yet")
    # Ignore copy
    def test_call_numpy_4_channels(self):
        pass
