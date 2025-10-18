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

from transformers.testing_utils import require_torch, require_vision, slow
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_vision_available():
    from PIL import Image

    from transformers import AutoProcessor, Owlv2ForObjectDetection, Owlv2ImageProcessor

if is_torch_available():
    import torch

    from transformers import Owlv2ImageProcessorFast


class Owlv2ImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        do_convert_rgb=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 18, "width": 18}
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]

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
class Owlv2ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = Owlv2ImageProcessor if is_vision_available() else None
    fast_image_processing_class = Owlv2ImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = Owlv2ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

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
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 18, "width": 18})

            image_processor = image_processing_class.from_dict(
                self.image_processor_dict, size={"height": 42, "width": 42}
            )
            self.assertEqual(image_processor.size, {"height": 42, "width": 42})

    @slow
    def test_image_processor_integration_test(self):
        for image_processing_class in self.image_processor_list:
            processor = image_processing_class()

            image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
            pixel_values = processor(image, return_tensors="pt").pixel_values

            mean_value = round(pixel_values.mean().item(), 4)
            self.assertEqual(mean_value, 0.2353)

    @slow
    def test_image_processor_integration_test_resize(self):
        for use_fast in [False, True]:
            checkpoint = "google/owlv2-base-patch16-ensemble"
            processor = AutoProcessor.from_pretrained(checkpoint, use_fast=use_fast)
            model = Owlv2ForObjectDetection.from_pretrained(checkpoint)

            image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
            text = ["cat"]
            target_size = image.size[::-1]
            expected_boxes = torch.tensor(
                [
                    [341.66656494140625, 23.38756561279297, 642.321044921875, 371.3482971191406],
                    [6.753320693969727, 51.96149826049805, 326.61810302734375, 473.12982177734375],
                ]
            )

            # single image
            inputs = processor(text=[text], images=[image], return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_object_detection(outputs, threshold=0.2, target_sizes=[target_size])[0]

            boxes = results["boxes"]
            torch.testing.assert_close(boxes, expected_boxes, atol=1e-1, rtol=1e-1)

            # batch of images
            inputs = processor(text=[text, text], images=[image, image], return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            results = processor.post_process_object_detection(
                outputs, threshold=0.2, target_sizes=[target_size, target_size]
            )

            for result in results:
                boxes = result["boxes"]
                torch.testing.assert_close(boxes, expected_boxes, atol=1e-1, rtol=1e-1)

    @unittest.skip(reason="OWLv2 doesn't treat 4 channel PIL and numpy consistently yet")  # FIXME Amy
    def test_call_numpy_4_channels(self):
        pass
