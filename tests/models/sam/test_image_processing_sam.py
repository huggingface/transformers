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

from datasets import load_dataset

from transformers.file_utils import is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from transformers import SamImageProcessor

    if is_torchvision_available():
        from transformers import SamImageProcessorFast


class SamImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_pad=True,
        pad_size=None,
        mask_size=None,
        mask_pad_size=None,
        do_resize=True,
        size=None,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ):
        size = size if size is not None else {"longest_edge": 20}
        pad_size = pad_size if pad_size is not None else {"height": 20, "width": 20}
        mask_size = mask_size if mask_size is not None else {"longest_edge": 12}
        mask_pad_size = mask_pad_size if mask_pad_size is not None else {"height": 12, "width": 12}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_pad = do_pad
        self.pad_size = pad_size
        self.mask_size = mask_size
        self.mask_pad_size = mask_pad_size
        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_image_processor_dict(self):
        return {
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "size": self.size,
            "do_pad": self.do_pad,
            "pad_size": self.pad_size,
            "mask_size": self.mask_size,
            "mask_pad_size": self.mask_pad_size,
        }

    def expected_output_image_shape(self, images):
        return self.num_channels, self.pad_size["height"], self.pad_size["width"]

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
class SamImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = SamImageProcessor if is_vision_available() else None
    fast_image_processing_class = SamImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = SamImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
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
        for image_processing_class in self.image_processor_list:
            image_processing_class = image_processing_class(**self.image_processor_dict)
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"longest_edge": 20})

            image_processor = image_processing_class.from_dict(self.image_processor_dict, size={"longest_edge": 42})
            self.assertEqual(image_processor.size, {"longest_edge": 42})

    def test_call_segmentation_maps(self):
        for image_processing_class in self.image_processor_list:
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
            image, segmentation_map = prepare_semantic_single_inputs()

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
            images, segmentation_maps = prepare_semantic_batch_inputs()

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

    def test_slow_fast_equivalence(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        dummy_image, dummy_map = prepare_semantic_single_inputs()

        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        image_encoding_slow = image_processor_slow(dummy_image, segmentation_maps=dummy_map, return_tensors="pt")
        image_encoding_fast = image_processor_fast(dummy_image, segmentation_maps=dummy_map, return_tensors="pt")

        self.assertTrue(torch.allclose(image_encoding_slow.pixel_values, image_encoding_fast.pixel_values, atol=1e-1))
        self.assertLessEqual(
            torch.mean(torch.abs(image_encoding_slow.pixel_values - image_encoding_fast.pixel_values)).item(), 1e-3
        )
        self.assertTrue(torch.allclose(image_encoding_slow.labels, image_encoding_fast.labels, atol=1e-1))

    def test_slow_fast_equivalence_batched(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        dummy_images, dummy_maps = prepare_semantic_batch_inputs()

        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_images, segmentation_maps=dummy_maps, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_images, segmentation_maps=dummy_maps, return_tensors="pt")

        self.assertTrue(torch.allclose(encoding_slow.pixel_values, encoding_fast.pixel_values, atol=1e-1))
        self.assertLessEqual(
            torch.mean(torch.abs(encoding_slow.pixel_values - encoding_fast.pixel_values)).item(), 1e-3
        )
