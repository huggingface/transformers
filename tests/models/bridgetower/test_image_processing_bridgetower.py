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
    def __init__(self, **kwargs):
        kwargs.setdefault("do_rescale", True)
        kwargs.setdefault("rescale_factor", 1 / 255)
        kwargs.setdefault("do_center_crop", True)
        kwargs.setdefault("do_pad", True)
        kwargs.setdefault("image_mean", [0.48145466, 0.4578275, 0.40821073])
        kwargs.setdefault("image_std", [0.26862954, 0.26130258, 0.27577711])
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"shortest_edge": 288})
        kwargs.setdefault("size_divisor", 32)
        super().__init__(**kwargs)

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["shortest_edge"], self.size["shortest_edge"]


@require_torch
@require_vision
class BridgeTowerImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_tester_class = BridgeTowerImageProcessingTester

    @property
    def image_processor_dict(self):
        return self.image_processing_tester.prepare_image_processor_dict()

    @require_vision
    @require_torch
    def test_backends_equivalence(self):
        if len(self.image_processor_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_image = load_image(
            url_to_local_path(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"
            )
        )

        encodings = {}
        for backend_name, image_processing_class in self.image_processor_classes.items():
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
        if len(self.image_processor_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        if hasattr(self.image_processing_tester, "do_center_crop") and self.image_processing_tester.do_center_crop:
            self.skipTest(
                reason="Skipping as do_center_crop is True and center_crop functions are not equivalent for fast and slow processors"
            )

        dummy_images = self.image_processing_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        encodings = {}
        for backend_name, image_processing_class in self.image_processor_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_images, return_tensors="pt")

        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_pixel_values = encodings[reference_backend].pixel_values
        reference_pixel_mask = encodings[reference_backend].pixel_mask.float()
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_pixel_values, encodings[backend_name].pixel_values)
            self._assert_tensors_equivalence(reference_pixel_mask, encodings[backend_name].pixel_mask.float())
