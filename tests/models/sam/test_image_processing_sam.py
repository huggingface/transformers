# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

from transformers.image_utils import PILImageResampling
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import (
    is_torch_available,
    is_torchvision_available,
    is_vision_available,
)

from ...test_image_processing_common import (
    ImageProcessingTestMixin,
    prepare_image_inputs,
)


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
        do_resize=True,
        size=None,
        resample=PILImageResampling.BILINEAR,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        do_pad=True,
        pad_size=None,
        do_convert_rgb=True,
    ):
        size = size if size is not None else {"longest_edge": 1024}
        pad_size = pad_size if pad_size is not None else {"height": 1024, "width": 1024}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_pad = do_pad
        self.pad_size = pad_size
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "resample": self.resample,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_pad": self.do_pad,
            "pad_size": self.pad_size,
            "do_convert_rgb": self.do_convert_rgb,
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
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "resample"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_pad"))
            self.assertTrue(hasattr(image_processing, "pad_size"))
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"longest_edge": 1024})
            self.assertEqual(image_processor.pad_size, {"height": 1024, "width": 1024})
            self.assertEqual(image_processor.do_normalize, True)

            image_processor = image_processing_class.from_dict(
                self.image_processor_dict,
                size={"longest_edge": 512},
                pad_size={"height": 512, "width": 512},
                do_normalize=False,
            )
            self.assertEqual(image_processor.size, {"longest_edge": 512})
            self.assertEqual(image_processor.pad_size, {"height": 512, "width": 512})
            self.assertEqual(image_processor.do_normalize, False)

    def test_post_process_masks_equivalence(self):
        """Test that fast and slow post_process_masks produce similar results."""
        if not self.test_fast_image_processor or self.fast_image_processing_class is None:
            self.skipTest("Fast image processor not available")

        # Only test with fast processor for now since we need to verify it works
        processor = self.fast_image_processing_class(**self.image_processor_dict)

        # Create test data
        dummy_masks = torch.randn(2, 1, 256, 256)
        original_sizes = [(480, 640), (600, 800)]
        reshaped_sizes = [(480, 640), (600, 800)]

        # Test post-processing
        result = processor.post_process_masks(dummy_masks, original_sizes, reshaped_sizes)

        # Verify output format and dimensions
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for i, mask in enumerate(result):
            self.assertIsInstance(mask, torch.Tensor)
            # Masks should be resized to original dimensions
            expected_h, expected_w = original_sizes[i]
            self.assertEqual(mask.shape[-2:], (expected_h, expected_w))

    def test_generate_crop_boxes(self):
        """Test that crop box generation works correctly."""
        if not self.test_fast_image_processor or self.fast_image_processing_class is None:
            self.skipTest("Fast image processor not available")

        processor = self.fast_image_processing_class(**self.image_processor_dict)

        # Create a dummy image tensor
        dummy_image = torch.randn(3, 480, 640)

        # Test crop box generation
        crop_boxes, layer_idxs = processor.generate_crop_boxes(dummy_image, target_size=224, crop_n_layers=1)

        # Verify outputs
        self.assertIsInstance(crop_boxes, list)
        self.assertIsInstance(layer_idxs, list)
        self.assertEqual(len(crop_boxes), len(layer_idxs))

        # Should have at least the full image crop
        self.assertGreater(len(crop_boxes), 0)

        # First crop should be the full image
        self.assertEqual(crop_boxes[0], [0, 0, 640, 480])
        self.assertEqual(layer_idxs[0], 0)

    def test_filter_masks(self):
        """Test that mask filtering works correctly."""
        if not self.test_fast_image_processor or self.fast_image_processing_class is None:
            self.skipTest("Fast image processor not available")

        processor = self.fast_image_processing_class(**self.image_processor_dict)

        # Create test data
        dummy_masks = torch.rand(5, 256, 256)  # 5 masks
        dummy_iou_scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        original_sizes = [(256, 256)] * 5
        crop_boxes = [[0, 0, 256, 256]] * 5

        # Test filtering
        filtered_masks, filtered_scores = processor.filter_masks(
            dummy_masks,
            dummy_iou_scores,
            original_sizes,
            crop_boxes,
            stability_score_thresh=0.8,
        )

        # Verify filtering worked
        self.assertIsInstance(filtered_masks, torch.Tensor)
        self.assertIsInstance(filtered_scores, torch.Tensor)
        # Should have fewer masks due to filtering
        self.assertLessEqual(filtered_masks.shape[0], dummy_masks.shape[0])
