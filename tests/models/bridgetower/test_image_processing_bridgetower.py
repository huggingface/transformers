# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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

from transformers.image_utils import load_image
from transformers.testing_utils import require_torch, require_vision

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin
from ...test_processing_common import url_to_local_path


class BridgeTowerImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, do_rescale=True, rescale_factor=1 / 255, do_center_crop=True, do_pad=True, **kwargs):
        kwargs.setdefault("image_mean", [0.48145466, 0.4578275, 0.40821073])
        kwargs.setdefault("image_std", [0.26862954, 0.26130258, 0.27577711])
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"shortest_edge": 288})
        kwargs.setdefault("size_divisor", 32)
        super().__init__(parent, **kwargs)
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_center_crop = do_center_crop
        self.do_pad = do_pad

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["shortest_edge"], self.size["shortest_edge"]


@require_torch
@require_vision
class BridgeTowerImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = BridgeTowerImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "size_divisor"))

    @require_vision
    @require_torch
    def test_backends_equivalence(self):
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_image = load_image(
            url_to_local_path(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"
            )
        )

        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_image, return_tensors="pt")

        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_pixel_values = encodings[reference_backend].pixel_values
        reference_pixel_mask = encodings[reference_backend].pixel_mask.float()
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_pixel_values, encodings[backend_name].pixel_values)
            self._assert_tensors_equivalence(reference_pixel_mask, encodings[backend_name].pixel_mask.float())

    @require_vision
    @require_torch
    def test_slow_fast_equivalence_batched(self):
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        if hasattr(self.image_processor_tester, "do_center_crop") and self.image_processor_tester.do_center_crop:
            self.skipTest(
                reason="Skipping as do_center_crop is True and center_crop functions are not equivalent for fast and slow processors"
            )

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_images, return_tensors="pt")

        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_pixel_values = encodings[reference_backend].pixel_values
        reference_pixel_mask = encodings[reference_backend].pixel_mask.float()
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_pixel_values, encodings[backend_name].pixel_values)
            self._assert_tensors_equivalence(reference_pixel_mask, encodings[backend_name].pixel_mask.float())
