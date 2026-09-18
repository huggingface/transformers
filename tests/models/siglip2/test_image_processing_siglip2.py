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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_vision_available():
    from PIL import Image


class Siglip2ImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, size=None, **kwargs):
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("do_rescale", True)
        kwargs.setdefault("rescale_factor", 1 / 255)
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("image_mean", [0.5, 0.5, 0.5])
        kwargs.setdefault("image_std", [0.5, 0.5, 0.5])
        kwargs.setdefault("resample", Image.Resampling.BILINEAR)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("max_num_patches", 256)
        super().__init__(parent, **kwargs)
        self.size = size if size is not None else {"height": 18, "width": 18}

    def expected_output_image_shape(self, images):
        return self.max_num_patches, self.patch_size * self.patch_size * self.num_channels


@require_torch
@require_vision
# Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest with CLIP->Siglip2
class Siglip2ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = Siglip2ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    # Ignore copy
    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
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

    # Ignore copy
    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.max_num_patches, 256)
            self.assertEqual(image_processor.patch_size, 16)

            image_processor = image_processing_class.from_dict(
                self.image_processor_dict, patch_size=32, max_num_patches=512
            )
            self.assertEqual(image_processor.patch_size, 32)
            self.assertEqual(image_processor.max_num_patches, 512)

    @unittest.skip(reason="not supported")
    # Ignore copy
    def test_call_numpy_4_channels(self):
        pass
