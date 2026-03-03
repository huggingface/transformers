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


import unittest

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_torchvision_available():
    from torchvision import transforms

if is_vision_available():
    from PIL import Image


class IdeficsImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "image_size": self.image_size,
        }

    def expected_output_image_shape(self, images):
        return (self.num_channels, self.image_size, self.image_size)

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
class IdeficsImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = IdeficsImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "image_size"))

    def test_image_processor_from_dict_with_kwargs(self):
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertNotEqual(image_processor.image_size, 30)

            image_processor = image_processing_class.from_dict(self.image_processor_dict, image_size=42)
            self.assertEqual(image_processor.image_size, 42)

    @require_torchvision
    def test_torchvision_numpy_transforms_equivalency(self):
        # Verify that the default inference transforms match an equivalent torchvision.Compose pipeline.
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        image_processing_class = self.image_processing_classes.get(
            "torchvision", next(iter(self.image_processing_classes.values()))
        )
        image_processor = image_processing_class(**self.image_processor_dict)

        def convert_to_rgb(image):
            if image.mode == "RGB":
                return image
            image_rgba = image.convert("RGBA")
            background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
            alpha_composite = Image.alpha_composite(background, image_rgba)
            alpha_composite = alpha_composite.convert("RGB")
            return alpha_composite

        image_size = image_processor.image_size
        image_mean = image_processor.image_mean
        image_std = image_processor.image_std

        transform = transforms.Compose(
            [
                convert_to_rgb,
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std),
            ]
        )

        pixel_values_transform_implied = image_processor(image_inputs, transform=None, return_tensors="pt")
        pixel_values_transform_supplied = image_processor(image_inputs, transform=transform, return_tensors="pt")

        torch.testing.assert_close(
            pixel_values_transform_implied["pixel_values"],
            pixel_values_transform_supplied,
            rtol=1e-2,
            atol=2e-2,
        )

    @unittest.skip(reason="not supported")
    def test_call_numpy(self):
        pass

    @unittest.skip(reason="not supported")
    def test_call_numpy_4_channels(self):
        pass

    @unittest.skip(reason="not supported")
    def test_call_pil(self):
        pass

    @unittest.skip(reason="not supported")
    def test_call_pytorch(self):
        pass
