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

import torch


if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = torch._dynamo.is_compiling

from transformers.image_transforms import (
    convert_to_rgb,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    infer_channel_dimension_format,
    load_image,
    make_flat_list_of_images,
    to_numpy_array,
)
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available, is_vision_available

from ...test_image_processing_common import (
    ImageProcessingTestMixin,
    prepare_image_inputs,
)
from ...test_processing_common import url_to_local_path


if is_vision_available():
    from PIL import Image

    from transformers import Fgclip2ImageProcessor


if is_torchvision_available():
    from transformers import Fgclip2ImageProcessorFast


def _determine_max_value(image, patch_size: int = 16) -> int:
    image_height = image.shape[0]
    image_width = image.shape[1]

    num_patches = (image_width // patch_size) * (image_height // patch_size)

    if num_patches > 784:
        return 1024
    elif num_patches > 576:
        return 784
    elif num_patches > 256:
        return 576
    elif num_patches > 128:
        return 256
    else:
        return 128


class Fgclip2ImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=512,
        do_resize=True,
        size=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        resample=None,
        patch_size=16,
        max_num_patches=256,
        dynamic_max_patches=True,
    ):
        size = size if size is not None else {"height": 18, "width": 18}
        resample = resample if resample is not None else Image.Resampling.BILINEAR
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.resample = resample
        self.patch_size = patch_size
        self.max_num_patches = max_num_patches
        self.dynamic_max_patches = dynamic_max_patches

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "resample": self.resample,
            "patch_size": self.patch_size,
            "max_num_patches": self.max_num_patches,
            "dynamic_max_patches": self.dynamic_max_patches,
        }

    def expected_output_image_shape(self, images):
        images = make_flat_list_of_images(images)

        images = [convert_to_rgb(image) for image in images]
        images = [to_numpy_array(image) for image in images]

        data_format = ChannelDimension.LAST
        input_data_format = infer_channel_dimension_format(images[0])
        images = [
            to_channel_dimension_format(
                image,
                data_format,
                input_channel_dim=input_data_format,
            )
            for image in images
        ]

        if self.dynamic_max_patches:
            candidate_values = [_determine_max_value(img, patch_size=self.patch_size) for img in images]
            max_num_patches = max(candidate_values)
        else:
            max_num_patches = self.max_num_patches

        return max_num_patches, self.patch_size * self.patch_size * self.num_channels

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
# Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest with CLIP->Fgclip2
class Fgclip2ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = Fgclip2ImageProcessor if is_vision_available() else None
    fast_image_processing_class = Fgclip2ImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = Fgclip2ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    # Ignore copy
    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "resample"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "patch_size"))
            self.assertTrue(hasattr(image_processing, "max_num_patches"))
            self.assertTrue(hasattr(image_processing, "dynamic_max_patches"))

    # Ignore copy
    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.max_num_patches, 256)
            self.assertEqual(image_processor.patch_size, 16)
            self.assertEqual(image_processor.dynamic_max_patches, True)

            image_processor = self.image_processing_class.from_dict(
                self.image_processor_dict,
                patch_size=32,
                max_num_patches=512,
                dynamic_max_patches=False,
            )
            self.assertEqual(image_processor.patch_size, 32)
            self.assertEqual(image_processor.max_num_patches, 512)
            self.assertEqual(image_processor.dynamic_max_patches, False)

    @unittest.skip(reason="not supported")
    # Ignore copy
    def test_call_numpy_4_channels(self):
        pass

    # Ignore copy
    def test_slow_fast_equivalence(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        dummy_image = load_image(url_to_local_path("http://images.cocodataset.org/val2017/000000039769.jpg"))
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_image, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_image, return_tensors="pt")
        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)

    # increase mean tolerance to 1e-3 -> 2e-3
    # Ignore copy
    def test_slow_fast_equivalence_batched(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        if hasattr(self.image_processor_tester, "do_center_crop") and self.image_processor_tester.do_center_crop:
            self.skipTest(
                reason="Skipping as do_center_crop is True and center_crop functions are not equivalent for fast and slow processors"
            )

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_images, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_images, return_tensors="pt")

        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)
