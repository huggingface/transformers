# Copyright 2025 HuggingFace Inc.
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

from transformers.file_utils import is_torch_available
from transformers.testing_utils import require_torch, require_vision

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_torch_available():
    import torch


class SamImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_mean", [0.5, 0.5, 0.5])
        kwargs.setdefault("image_std", [0.5, 0.5, 0.5])
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"longest_edge": 20})
        kwargs.setdefault("do_pad", True)
        kwargs.setdefault("pad_size", {"height": 20, "width": 20})
        kwargs.setdefault("mask_size", {"longest_edge": 12})
        kwargs.setdefault("mask_pad_size", {"height": 12, "width": 12})
        super().__init__(parent, **kwargs)

    def expected_output_image_shape(self, images):
        return self.num_channels, self.pad_size["height"], self.pad_size["width"]


@require_torch
@require_vision
class SamImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = SamImageProcessingTester(self)

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
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "do_pad"))
            self.assertTrue(hasattr(image_processing, "pad_size"))
            self.assertTrue(hasattr(image_processing, "mask_size"))
            self.assertTrue(hasattr(image_processing, "mask_pad_size"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing_class = image_processing_class(**self.image_processor_dict)
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"longest_edge": 20})

            image_processor = image_processing_class.from_dict(self.image_processor_dict, size={"longest_edge": 42})
            self.assertEqual(image_processor.size, {"longest_edge": 42})

    def test_call_segmentation_maps(self):
        for image_processing_class in self.image_processing_classes.values():
            # Initialize image_processor
            image_processor = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
            maps = []
            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)
                maps.append(torch.zeros(image.shape[-2:]).long())

            # Test not batched input
            encoding = image_processor(image_inputs[0], maps[0], return_tensors="pt")
            self.assertEqual(
                encoding["pixel_values"].shape,
                (
                    1,
                    self.image_processor_tester.num_channels,
                    self.image_processor_tester.pad_size["height"],
                    self.image_processor_tester.pad_size["width"],
                ),
            )
            self.assertEqual(
                encoding["labels"].shape,
                (
                    1,
                    self.image_processor_tester.mask_pad_size["height"],
                    self.image_processor_tester.mask_pad_size["width"],
                ),
            )
            self.assertEqual(encoding["labels"].dtype, torch.long)
            self.assertTrue(encoding["labels"].min().item() >= 0)
            self.assertTrue(encoding["labels"].max().item() <= 255)

            # Test batched
            encoding = image_processor(image_inputs, maps, return_tensors="pt")
            self.assertEqual(
                encoding["pixel_values"].shape,
                (
                    self.image_processor_tester.batch_size,
                    self.image_processor_tester.num_channels,
                    self.image_processor_tester.pad_size["height"],
                    self.image_processor_tester.pad_size["width"],
                ),
            )
            self.assertEqual(
                encoding["labels"].shape,
                (
                    self.image_processor_tester.batch_size,
                    self.image_processor_tester.mask_pad_size["height"],
                    self.image_processor_tester.mask_pad_size["width"],
                ),
            )
            self.assertEqual(encoding["labels"].dtype, torch.long)
            self.assertTrue(encoding["labels"].min().item() >= 0)
            self.assertTrue(encoding["labels"].max().item() <= 255)

            # Test not batched input (PIL images)
            image, segmentation_map = self.image_processor_tester.prepare_semantic_segmentation_inputs_ade20k()

            encoding = image_processor(image, segmentation_map, return_tensors="pt")
            self.assertEqual(
                encoding["pixel_values"].shape,
                (
                    1,
                    self.image_processor_tester.num_channels,
                    self.image_processor_tester.pad_size["height"],
                    self.image_processor_tester.pad_size["width"],
                ),
            )
            self.assertEqual(
                encoding["labels"].shape,
                (
                    1,
                    self.image_processor_tester.mask_pad_size["height"],
                    self.image_processor_tester.mask_pad_size["width"],
                ),
            )
            self.assertEqual(encoding["labels"].dtype, torch.long)
            self.assertTrue(encoding["labels"].min().item() >= 0)
            self.assertTrue(encoding["labels"].max().item() <= 255)

            # Test batched input (PIL images)
            images, segmentation_maps = self.image_processor_tester.prepare_semantic_segmentation_inputs_ade20k(
                batched=True
            )

            encoding = image_processor(images, segmentation_maps, return_tensors="pt")
            self.assertEqual(
                encoding["pixel_values"].shape,
                (
                    2,
                    self.image_processor_tester.num_channels,
                    self.image_processor_tester.pad_size["height"],
                    self.image_processor_tester.pad_size["width"],
                ),
            )
            self.assertEqual(
                encoding["labels"].shape,
                (
                    2,
                    self.image_processor_tester.mask_pad_size["height"],
                    self.image_processor_tester.mask_pad_size["width"],
                ),
            )
            self.assertEqual(encoding["labels"].dtype, torch.long)
            self.assertTrue(encoding["labels"].min().item() >= 0)
            self.assertTrue(encoding["labels"].max().item() <= 255)

    def test_backends_equivalence(self):
        """Override base class test to also compare segmentation labels."""
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_image, dummy_map = self.image_processor_tester.prepare_semantic_segmentation_inputs_ade20k()

        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_image, segmentation_maps=dummy_map, return_tensors="pt")

        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(
                encodings[reference_backend].pixel_values, encodings[backend_name].pixel_values, atol=1e-1
            )
            self.assertLessEqual(
                torch.mean(
                    torch.abs(encodings[reference_backend].pixel_values - encodings[backend_name].pixel_values)
                ).item(),
                1e-3,
            )
            self._assert_tensors_equivalence(
                encodings[reference_backend].labels.float(), encodings[backend_name].labels.float(), atol=1e-1
            )

    def test_backends_equivalence_batched(self):
        """Override base class test to also compare segmentation labels."""
        if len(self.image_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_images, dummy_maps = self.image_processor_tester.prepare_semantic_segmentation_inputs_ade20k(
            batched=True
        )

        encodings = {}
        for backend_name, image_processing_class in self.image_processing_classes.items():
            image_processor = image_processing_class(**self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_images, segmentation_maps=dummy_maps, return_tensors="pt")

        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(
                encodings[reference_backend].pixel_values, encodings[backend_name].pixel_values, atol=1e-1
            )
            self.assertLessEqual(
                torch.mean(
                    torch.abs(encodings[reference_backend].pixel_values - encodings[backend_name].pixel_values)
                ).item(),
                1e-3,
            )
            self._assert_tensors_equivalence(
                encodings[reference_backend].labels.float(), encodings[backend_name].labels.float(), atol=1e-1
            )
