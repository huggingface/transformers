# Copyright 2022 HuggingFace Inc.
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
from transformers.utils import is_torch_available

from ...test_image_processing_common import (
    ImageProcessingTester,
    ImageProcessingTestMixin,
    PostProcessSemanticSegmentationTestMixin,
)
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch

    from transformers.modeling_outputs import SemanticSegmenterOutput


class MobileNetV2ImageProcessingTester(ImageProcessingTester):
    num_labels = 5

    # Image processor init kwargs
    do_resize = True
    size = {"shortest_edge": 20}
    do_center_crop = True
    crop_size = {"height": 18, "width": 18}
    do_reduce_labels = False

    def prepare_post_process_semantic_segmentation_inputs(self):
        inputs = {
            "outputs": SemanticSegmenterOutput(
                logits=torch.randn(
                    self.batch_size,
                    self.num_labels,
                    self.crop_size["height"],
                    self.crop_size["width"],
                )
            )
        }
        expected_shape = {
            "num_labels": self.num_labels,
            "height": self.crop_size["height"],
            "width": self.crop_size["width"],
        }
        return inputs, expected_shape


@require_torch
@require_vision
class MobileNetV2ImageProcessingTest(
    ImageProcessingTestMixin, PostProcessSemanticSegmentationTestMixin, unittest.TestCase
):
    image_processing_tester_class = MobileNetV2ImageProcessingTester

    def test_call_segmentation_maps(self):
        # Initialize image_processing
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
            maps = []
            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)
                maps.append(torch.zeros(image.shape[-2:]).long())

            # Test not batched input
            encoding = image_processing(image_inputs[0], maps[0], return_tensors="pt")
            self.assertEqual(
                encoding["pixel_values"].shape,
                (
                    1,
                    self.image_processor_tester.num_channels,
                    self.image_processor_tester.crop_size["height"],
                    self.image_processor_tester.crop_size["width"],
                ),
            )
            self.assertEqual(
                encoding["labels"].shape,
                (
                    1,
                    self.image_processor_tester.crop_size["height"],
                    self.image_processor_tester.crop_size["width"],
                ),
            )
            self.assertEqual(encoding["labels"].dtype, torch.long)
            self.assertTrue(encoding["labels"].min().item() >= 0)
            self.assertTrue(encoding["labels"].max().item() <= 255)

            # Test batched
            encoding = image_processing(image_inputs, maps, return_tensors="pt")
            self.assertEqual(
                encoding["pixel_values"].shape,
                (
                    self.image_processor_tester.batch_size,
                    self.image_processor_tester.num_channels,
                    self.image_processor_tester.crop_size["height"],
                    self.image_processor_tester.crop_size["width"],
                ),
            )
            self.assertEqual(
                encoding["labels"].shape,
                (
                    self.image_processor_tester.batch_size,
                    self.image_processor_tester.crop_size["height"],
                    self.image_processor_tester.crop_size["width"],
                ),
            )
            self.assertEqual(encoding["labels"].dtype, torch.long)
            self.assertTrue(encoding["labels"].min().item() >= 0)
            self.assertTrue(encoding["labels"].max().item() <= 255)

            # Test not batched input (PIL images)
            image, segmentation_map = self.image_processor_tester.prepare_semantic_segmentation_inputs_ade20k()

            encoding = image_processing(image, segmentation_map, return_tensors="pt")
            self.assertEqual(
                encoding["pixel_values"].shape,
                (
                    1,
                    self.image_processor_tester.num_channels,
                    self.image_processor_tester.crop_size["height"],
                    self.image_processor_tester.crop_size["width"],
                ),
            )
            self.assertEqual(
                encoding["labels"].shape,
                (
                    1,
                    self.image_processor_tester.crop_size["height"],
                    self.image_processor_tester.crop_size["width"],
                ),
            )
            self.assertEqual(encoding["labels"].dtype, torch.long)
            self.assertTrue(encoding["labels"].min().item() >= 0)
            self.assertTrue(encoding["labels"].max().item() <= 255)

            # Test batched input (PIL images)
            images, segmentation_maps = self.image_processor_tester.prepare_semantic_segmentation_inputs_ade20k(
                batched=True
            )

            encoding = image_processing(images, segmentation_maps, return_tensors="pt")
            self.assertEqual(
                encoding["pixel_values"].shape,
                (
                    2,
                    self.image_processor_tester.num_channels,
                    self.image_processor_tester.crop_size["height"],
                    self.image_processor_tester.crop_size["width"],
                ),
            )
            self.assertEqual(
                encoding["labels"].shape,
                (
                    2,
                    self.image_processor_tester.crop_size["height"],
                    self.image_processor_tester.crop_size["width"],
                ),
            )
            self.assertEqual(encoding["labels"].dtype, torch.long)
            self.assertTrue(encoding["labels"].min().item() >= 0)
            self.assertTrue(encoding["labels"].max().item() <= 255)

    def test_reduce_labels(self):
        # Initialize image_processing
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)

            # ADE20k has 150 classes, and the background is included, so labels should be between 0 and 150
            image, map = self.image_processor_tester.prepare_semantic_segmentation_inputs_ade20k()
            encoding = image_processing(image, map, return_tensors="pt")
            self.assertTrue(encoding["labels"].min().item() >= 0)
            self.assertTrue(encoding["labels"].max().item() <= 150)

            image_processing.do_reduce_labels = True
            encoding = image_processing(image, map, return_tensors="pt")
            self.assertTrue(encoding["labels"].min().item() >= 0)
            self.assertTrue(encoding["labels"].max().item() <= 255)
            # Ensure reduce label returns the same number of masks
            image, map = self.image_processor_tester.prepare_semantic_segmentation_inputs_ade20k(batched=True)
            encoding = image_processing(image, map, return_tensors="pt")
            self.assertTrue(len(encoding["labels"]) == len(map))

    def test_backends_equivalence(self):
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        # Test with single image
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
        reference_encoding = encodings[reference_backend].pixel_values
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_encoding, encodings[backend_name].pixel_values)

        # Test with single image and segmentation map
        image, segmentation_map = self.image_processor_tester.prepare_semantic_segmentation_inputs_ade20k()
        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(image, segmentation_map, return_tensors="pt")

        reference_encoding = encodings[reference_backend]
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_encoding.pixel_values, encodings[backend_name].pixel_values)
            torch.testing.assert_close(reference_encoding.labels, encodings[backend_name].labels, atol=1e-1, rtol=1e-3)
