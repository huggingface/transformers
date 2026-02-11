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

import numpy as np
from datasets import load_dataset

from transformers.file_utils import is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from transformers import DPTImageProcessor


class DPTImageProcessingTester:
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
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_reduce_labels=False,
    ):
        size = size if size is not None else {"height": 18, "width": 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_reduce_labels = do_reduce_labels

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
            "do_reduce_labels": self.do_reduce_labels,
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


# Copied from transformers.tests.models.beit.test_image_processing_beit.prepare_semantic_single_inputs
def prepare_semantic_single_inputs():
    ds = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")
    example = ds[0]
    return example["image"], example["map"]


# Copied from transformers.tests.models.beit.test_image_processing_beit.prepare_semantic_batch_inputs
def prepare_semantic_batch_inputs():
    ds = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")
    return list(ds["image"][:2]), list(ds["map"][:2])


@require_torch
@require_vision
class DPTImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = DPTImageProcessor if is_vision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = DPTImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for backend_name in self.image_processors_backends_list:
            image_processing = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "do_pad"))
            self.assertTrue(hasattr(image_processing, "size_divisor"))
            self.assertTrue(hasattr(image_processing, "do_reduce_labels"))

    def test_image_processor_from_dict_with_kwargs(self):
        for backend_name in self.image_processors_backends_list:
            image_processor = self.image_processing_class.from_dict(self.image_processor_dict, backend=backend_name)
            self.assertEqual(image_processor.size, {"height": 18, "width": 18})

            image_processor = self.image_processing_class.from_dict(
                self.image_processor_dict, size=42, backend=backend_name
            )
            self.assertEqual(image_processor.size, {"height": 42, "width": 42})

    def test_padding(self):
        for backend_name in self.image_processors_backends_list:
            image_processor = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            if backend_name == "torchvision":
                image = torch.arange(0, 366777, 1, dtype=torch.uint8).reshape(3, 249, 491)
                padded_image = image_processor._backend_instance.pad_image(image, size_divisor=4)
                self.assertTrue(padded_image.shape[1] % 4 == 0)
                self.assertTrue(padded_image.shape[2] % 4 == 0)
                pixel_values = image_processor.preprocess(
                    image, do_rescale=False, do_resize=False, do_pad=True, size_divisor=4, return_tensors="pt"
                ).pixel_values
                self.assertTrue(pixel_values.shape[2] % 4 == 0)
                self.assertTrue(pixel_values.shape[3] % 4 == 0)
            else:
                image_processor = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
                image = np.random.randn(3, 249, 491)
                image = image_processor._backend_instance.pad_image(image, size_divisor=4)
                self.assertTrue(image.shape[1] % 4 == 0)
                self.assertTrue(image.shape[2] % 4 == 0)
                pixel_values = image_processor.preprocess(
                    image, do_rescale=False, do_resize=False, do_pad=True, size_divisor=4, return_tensors="pt"
                ).pixel_values
                self.assertTrue(pixel_values.shape[2] % 4 == 0)
                self.assertTrue(pixel_values.shape[3] % 4 == 0)

    def test_keep_aspect_ratio(self):
        size = {"height": 512, "width": 512}
        for backend_name in self.image_processors_backends_list:
            image_processor = self.image_processing_class(
                backend=backend_name, size=size, keep_aspect_ratio=True, ensure_multiple_of=32
            )

            image = np.zeros((489, 640, 3))

            pixel_values = image_processor(image, return_tensors="pt").pixel_values

            self.assertEqual(list(pixel_values.shape), [1, 3, 512, 672])

    # Copied from transformers.tests.models.beit.test_image_processing_beit.BeitImageProcessingTest.test_call_segmentation_maps
    def test_call_segmentation_maps(self):
        for backend_name in self.image_processors_backends_list:
            # Initialize image_processor
            image_processor = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
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
                    self.image_processor_tester.size["height"],
                    self.image_processor_tester.size["width"],
                ),
            )
            self.assertEqual(
                encoding["labels"].shape,
                (
                    1,
                    self.image_processor_tester.size["height"],
                    self.image_processor_tester.size["width"],
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
                    self.image_processor_tester.size["height"],
                    self.image_processor_tester.size["width"],
                ),
            )
            self.assertEqual(
                encoding["labels"].shape,
                (
                    self.image_processor_tester.batch_size,
                    self.image_processor_tester.size["height"],
                    self.image_processor_tester.size["width"],
                ),
            )
            self.assertEqual(encoding["labels"].dtype, torch.long)
            self.assertTrue(encoding["labels"].min().item() >= 0)
            self.assertTrue(encoding["labels"].max().item() <= 255)

            # Test not batched input (PIL images)
            image, segmentation_map = prepare_semantic_single_inputs()

            encoding = image_processor(image, segmentation_map, return_tensors="pt")
            self.assertEqual(
                encoding["pixel_values"].shape,
                (
                    1,
                    self.image_processor_tester.num_channels,
                    self.image_processor_tester.size["height"],
                    self.image_processor_tester.size["width"],
                ),
            )
            self.assertEqual(
                encoding["labels"].shape,
                (
                    1,
                    self.image_processor_tester.size["height"],
                    self.image_processor_tester.size["width"],
                ),
            )
            self.assertEqual(encoding["labels"].dtype, torch.long)
            self.assertTrue(encoding["labels"].min().item() >= 0)
            self.assertTrue(encoding["labels"].max().item() <= 255)

            # Test batched input (PIL images)
            images, segmentation_maps = prepare_semantic_batch_inputs()

            encoding = image_processor(images, segmentation_maps, return_tensors="pt")
            self.assertEqual(
                encoding["pixel_values"].shape,
                (
                    2,
                    self.image_processor_tester.num_channels,
                    self.image_processor_tester.size["height"],
                    self.image_processor_tester.size["width"],
                ),
            )
            self.assertEqual(
                encoding["labels"].shape,
                (
                    2,
                    self.image_processor_tester.size["height"],
                    self.image_processor_tester.size["width"],
                ),
            )
            self.assertEqual(encoding["labels"].dtype, torch.long)
            self.assertTrue(encoding["labels"].min().item() >= 0)
            self.assertTrue(encoding["labels"].max().item() <= 255)

    def test_reduce_labels(self):
        for backend_name in self.image_processors_backends_list:
            image_processor = self.image_processing_class(backend=backend_name, **self.image_processor_dict)

            # ADE20k has 150 classes, and the background is included, so labels should be between 0 and 150
            image, map = prepare_semantic_single_inputs()
            encoding = image_processor(image, map, return_tensors="pt")
            labels_no_reduce = encoding["labels"].clone()
            self.assertTrue(labels_no_reduce.min().item() >= 0)
            self.assertTrue(labels_no_reduce.max().item() <= 150)
            # Get the first non-zero label coords and value, for comparison when do_reduce_labels is True
            non_zero_positions = (labels_no_reduce > 0).nonzero()
            first_non_zero_coords = tuple(non_zero_positions[0].tolist())
            first_non_zero_value = labels_no_reduce[first_non_zero_coords].item()

            image_processor.do_reduce_labels = True
            encoding = image_processor(image, map, return_tensors="pt")
            self.assertTrue(encoding["labels"].min().item() >= 0)
            self.assertTrue(encoding["labels"].max().item() <= 255)
            # Compare with non-reduced label to see if it's reduced by 1
            self.assertEqual(encoding["labels"][first_non_zero_coords].item(), first_non_zero_value - 1)

            # Ensure reduce label returns the same number of masks
            image, map = prepare_semantic_batch_inputs()
            encoding = image_processor(image, map, return_tensors="pt")
            self.assertTrue(len(encoding["labels"]) == len(map))

    @require_vision
    @require_torch
    def test_backends_equivalence(self):
        if len(self.image_processors_backends_list) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        if self.image_processing_class is None:
            self.skipTest(reason="Skipping backends equivalence test as image processor is not defined")

        dummy_image, dummy_map = prepare_semantic_single_inputs()

        # Create processors for each backend
        encodings = {}
        for backend_name in self.image_processors_backends_list:
            image_processor = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_image, segmentation_maps=dummy_map, return_tensors="pt")

        # Compare all backends to the first one (reference backend)
        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_encoding = encodings[reference_backend]

        for backend_name in backend_names[1:]:
            # Check pixel_values
            self.assertTrue(
                torch.allclose(reference_encoding.pixel_values, encodings[backend_name].pixel_values, atol=1e-1)
            )
            self.assertLessEqual(
                torch.mean(torch.abs(reference_encoding.pixel_values - encodings[backend_name].pixel_values)).item(),
                1e-3,
            )
            # Check labels (custom check for DPT)
            self.assertTrue(torch.allclose(reference_encoding.labels, encodings[backend_name].labels, atol=1e-1))

    @require_vision
    @require_torch
    def test_slow_fast_equivalence_batched(self):
        if len(self.image_processors_backends_list) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        if self.image_processing_class is None:
            self.skipTest(reason="Skipping backends equivalence test as image processor is not defined")

        if hasattr(self.image_processor_tester, "do_center_crop") and self.image_processor_tester.do_center_crop:
            self.skipTest(
                reason="Skipping as do_center_crop is True and center_crop functions are not equivalent for fast and slow processors"
            )

        dummy_images, dummy_maps = prepare_semantic_batch_inputs()

        # Create processors for each backend
        encodings = {}
        for backend_name in self.image_processors_backends_list:
            image_processor = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_images, segmentation_maps=dummy_maps, return_tensors="pt")

        # Compare all backends to the first one (reference backend)
        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_encoding = encodings[reference_backend]

        for backend_name in backend_names[1:]:
            # Check pixel_values
            self.assertTrue(
                torch.allclose(reference_encoding.pixel_values, encodings[backend_name].pixel_values, atol=1e-1)
            )
            self.assertLessEqual(
                torch.mean(torch.abs(reference_encoding.pixel_values - encodings[backend_name].pixel_values)).item(),
                1e-3,
            )
