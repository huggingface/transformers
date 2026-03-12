# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from transformers import AutoImageProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from transformers import RfDetrImageProcessor

    if is_torchvision_available():
        from transformers import RfDetrImageProcessorFast


class RfDetrImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=4,
        num_channels=3,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        num_top_queries=300,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 512, "width": 512}
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.485, 0.456, 0.406]
        self.image_std = image_std if image_std is not None else [0.229, 0.224, 0.225]
        self.num_top_queries = num_top_queries

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "num_top_queries": self.num_top_queries,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        image_inputs = prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=30,
            max_resolution=400,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )
        if torchify:
            image_inputs = [image.to(dtype=torch.float32) / 255.0 for image in image_inputs]
        return image_inputs


@require_torch
@require_vision
class RfDetrImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = RfDetrImageProcessor if is_vision_available() else None
    fast_image_processing_class = RfDetrImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = RfDetrImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processor, "do_resize"))
            self.assertTrue(hasattr(image_processor, "size"))
            self.assertTrue(hasattr(image_processor, "resample"))
            self.assertTrue(hasattr(image_processor, "do_normalize"))
            self.assertTrue(hasattr(image_processor, "image_mean"))
            self.assertTrue(hasattr(image_processor, "image_std"))
            self.assertTrue(hasattr(image_processor, "num_top_queries"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 512, "width": 512})
            self.assertEqual(image_processor.num_top_queries, 300)

            image_processor = image_processing_class.from_dict(
                self.image_processor_dict, size={"height": 384, "width": 384}, num_top_queries=100
            )
            self.assertEqual(image_processor.size, {"height": 384, "width": 384})
            self.assertEqual(image_processor.num_top_queries, 100)

    def test_rejects_torch_inputs_above_one(self):
        image = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            with self.assertRaises(ValueError):
                _ = image_processor(image, return_tensors="pt")

    def test_call_numpy_4_channels(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            image_4_channels = (
                torch.randint(0, 255, (image_inputs[0].shape[0], image_inputs[0].shape[1], 4), dtype=torch.uint8)
                .cpu()
                .numpy()
            )
            with self.assertRaises(ValueError):
                _ = image_processor(
                    image_4_channels,
                    return_tensors="pt",
                    input_data_format="channels_last",
                    image_mean=[0.0, 0.0, 0.0, 0.0],
                    image_std=[1.0, 1.0, 1.0, 1.0],
                )

    def test_save_load_fast_slow(self):
        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest("Skipping slow/fast save/load test as one of the image processors is not defined")

        image_processor_dict = self.image_processor_tester.prepare_image_processor_dict()
        image_processor_slow = self.image_processing_class(**image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**image_processor_dict)

        with TemporaryDirectory() as tmpdir:
            image_processor_slow.save_pretrained(tmpdir)
            loaded_fast = self.fast_image_processing_class.from_pretrained(tmpdir)
        self.assertIsInstance(loaded_fast, self.fast_image_processing_class)

        with TemporaryDirectory() as tmpdir:
            image_processor_fast.save_pretrained(tmpdir)
            loaded_slow = self.image_processing_class.from_pretrained(tmpdir)
        self.assertIsInstance(loaded_slow, self.image_processing_class)

    def test_save_load_fast_slow_auto(self):
        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest("Skipping slow/fast save/load test as one of the image processors is not defined")

        image_processor_dict = self.image_processor_tester.prepare_image_processor_dict()
        image_processor_slow = self.image_processing_class(**image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**image_processor_dict)

        with TemporaryDirectory() as tmpdir:
            image_processor_slow.save_pretrained(tmpdir)
            loaded_fast = AutoImageProcessor.from_pretrained(tmpdir, use_fast=True)
        self.assertIsInstance(loaded_fast, self.fast_image_processing_class)

        with TemporaryDirectory() as tmpdir:
            image_processor_fast.save_pretrained(tmpdir)
            loaded_slow = AutoImageProcessor.from_pretrained(tmpdir, use_fast=False)
        self.assertIsInstance(loaded_slow, self.image_processing_class)

    def test_post_process_object_detection_slow_fast_equivalence(self):
        if self.fast_image_processing_class is None:
            self.skipTest("Fast image processor is not available.")

        slow = self.image_processing_class(**self.image_processor_dict)
        fast = self.fast_image_processing_class(**self.image_processor_dict)

        outputs = SimpleNamespace(
            logits=torch.tensor(
                [
                    [
                        [-1.0, 0.1, 2.0],
                        [0.5, -0.2, 0.3],
                        [1.2, -1.0, 0.0],
                        [-0.5, 0.7, 0.4],
                    ]
                ],
                dtype=torch.float32,
            ),
            pred_boxes=torch.tensor(
                [
                    [
                        [0.5, 0.5, 0.2, 0.4],
                        [0.25, 0.25, 0.1, 0.1],
                        [0.75, 0.75, 0.3, 0.2],
                        [0.4, 0.6, 0.5, 0.3],
                    ]
                ],
                dtype=torch.float32,
            ),
        )
        target_sizes = torch.tensor([[128, 256]], dtype=torch.int64)

        slow_outputs = slow.post_process_object_detection(
            outputs=outputs, threshold=0.0, target_sizes=target_sizes, num_top_queries=6
        )
        fast_outputs = fast.post_process_object_detection(
            outputs=outputs, threshold=0.0, target_sizes=target_sizes, num_top_queries=6
        )

        self.assertEqual(len(slow_outputs), len(fast_outputs))
        torch.testing.assert_close(slow_outputs[0]["scores"], fast_outputs[0]["scores"])
        torch.testing.assert_close(slow_outputs[0]["boxes"], fast_outputs[0]["boxes"])
        self.assertTrue(torch.equal(slow_outputs[0]["labels"], fast_outputs[0]["labels"]))

    def test_post_process_instance_segmentation_slow_fast_equivalence(self):
        if self.fast_image_processing_class is None:
            self.skipTest("Fast image processor is not available.")

        slow = self.image_processing_class(**self.image_processor_dict)
        fast = self.fast_image_processing_class(**self.image_processor_dict)

        outputs = SimpleNamespace(
            logits=torch.tensor(
                [
                    [
                        [-1.0, 0.1, 2.0],
                        [0.5, -0.2, 0.3],
                        [1.2, -1.0, 0.0],
                        [-0.5, 0.7, 0.4],
                    ]
                ],
                dtype=torch.float32,
            ),
            pred_boxes=torch.tensor(
                [
                    [
                        [0.5, 0.5, 0.2, 0.4],
                        [0.25, 0.25, 0.1, 0.1],
                        [0.75, 0.75, 0.3, 0.2],
                        [0.4, 0.6, 0.5, 0.3],
                    ]
                ],
                dtype=torch.float32,
            ),
            pred_masks=torch.randn(1, 4, 24, 24),
        )
        target_sizes = torch.tensor([[96, 160]], dtype=torch.int64)

        slow_outputs = slow.post_process_instance_segmentation(
            outputs=outputs, threshold=0.0, mask_threshold=0.0, target_sizes=target_sizes, num_top_queries=6
        )
        fast_outputs = fast.post_process_instance_segmentation(
            outputs=outputs, threshold=0.0, mask_threshold=0.0, target_sizes=target_sizes, num_top_queries=6
        )

        self.assertEqual(len(slow_outputs), len(fast_outputs))
        torch.testing.assert_close(slow_outputs[0]["scores"], fast_outputs[0]["scores"])
        torch.testing.assert_close(slow_outputs[0]["boxes"], fast_outputs[0]["boxes"])
        self.assertTrue(torch.equal(slow_outputs[0]["labels"], fast_outputs[0]["labels"]))
        self.assertTrue(torch.equal(slow_outputs[0]["masks"], fast_outputs[0]["masks"]))
