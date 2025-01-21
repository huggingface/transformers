# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import shutil
import tempfile
import unittest

import numpy as np

from transformers.testing_utils import (
    is_pt_tf_cross_test,
    require_tf,
    require_torch,
    require_torchvision,
    require_vision,
)
from transformers.utils import is_tf_available, is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin, prepare_image_inputs


if is_vision_available():
    from PIL import Image

    from transformers import AutoProcessor, SamImageProcessor, SamProcessor

if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf


@require_vision
@require_torchvision
class SamProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = SamProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = SamImageProcessor()
        processor = SamProcessor(image_processor)
        processor.save_pretrained(self.tmpdirname)

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_mask_inputs(self):
        """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
        or a list of PyTorch tensors if one specifies torchify=True.
        """
        mask_inputs = [np.random.randint(255, size=(30, 400), dtype=np.uint8)]
        mask_inputs = [Image.fromarray(x) for x in mask_inputs]
        return mask_inputs

    def test_chat_template_save_loading(self):
        self.skipTest("SamProcessor does not have a tokenizer")

    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        self.skipTest("SamProcessor does not have a tokenizer")

    def test_kwargs_overrides_default_image_processor_kwargs(self):
        self.skipTest("SamProcessor does not have a tokenizer")

    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        self.skipTest("SamProcessor does not have a tokenizer")

    def test_tokenizer_defaults_preserved_by_kwargs(self):
        self.skipTest("SamProcessor does not have a tokenizer")

    def test_save_load_pretrained_additional_features(self):
        processor = SamProcessor(image_processor=self.get_image_processor())
        processor.save_pretrained(self.tmpdirname)

        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

        processor = SamProcessor.from_pretrained(self.tmpdirname, do_normalize=False, padding_value=1.0)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, SamImageProcessor)

    def test_image_processor_no_masks(self):
        image_processor = self.get_image_processor()

        processor = SamProcessor(image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_feat_extract = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

        for image in input_feat_extract.pixel_values:
            self.assertEqual(image.shape, (3, 1024, 1024))

        for original_size in input_feat_extract.original_sizes:
            np.testing.assert_array_equal(original_size, np.array([30, 400]))

        for reshaped_input_size in input_feat_extract.reshaped_input_sizes:
            np.testing.assert_array_equal(
                reshaped_input_size, np.array([77, 1024])
            )  # reshaped_input_size value is before padding

    def test_image_processor_with_masks(self):
        image_processor = self.get_image_processor()

        processor = SamProcessor(image_processor=image_processor)

        image_input = self.prepare_image_inputs()
        mask_input = self.prepare_mask_inputs()

        input_feat_extract = image_processor(images=image_input, segmentation_maps=mask_input, return_tensors="np")
        input_processor = processor(images=image_input, segmentation_maps=mask_input, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

        for label in input_feat_extract.labels:
            self.assertEqual(label.shape, (256, 256))

    @require_torch
    def test_post_process_masks(self):
        image_processor = self.get_image_processor()

        processor = SamProcessor(image_processor=image_processor)
        dummy_masks = [torch.ones((1, 3, 5, 5))]

        original_sizes = [[1764, 2646]]

        reshaped_input_size = [[683, 1024]]
        masks = processor.post_process_masks(dummy_masks, original_sizes, reshaped_input_size)
        self.assertEqual(masks[0].shape, (1, 3, 1764, 2646))

        masks = processor.post_process_masks(
            dummy_masks, torch.tensor(original_sizes), torch.tensor(reshaped_input_size)
        )
        self.assertEqual(masks[0].shape, (1, 3, 1764, 2646))

        # should also work with np
        dummy_masks = [np.ones((1, 3, 5, 5))]
        masks = processor.post_process_masks(dummy_masks, np.array(original_sizes), np.array(reshaped_input_size))

        self.assertEqual(masks[0].shape, (1, 3, 1764, 2646))

        dummy_masks = [[1, 0], [0, 1]]
        with self.assertRaises(ValueError):
            masks = processor.post_process_masks(dummy_masks, np.array(original_sizes), np.array(reshaped_input_size))

    def test_generate_crop_boxes_batched(self):
        """Test that generate_crop_boxes handles batched inputs correctly"""
        image_processor = self.get_image_processor()
        
        # Create a test image
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        
        points_per_side = 32
        # Test with crops_n_layers=1
        crop_boxes1, points_per_crop1, cropped_images1, input_labels1 = image_processor.generate_crop_boxes(
            image1,
            target_size=64,
            crop_n_layers=1,
            points_per_crop=points_per_side
        )
        
        # Basic shape checks
        self.assertEqual(points_per_crop1.ndim, 4)  # Should be [num_crops, batch, num_points, 2]
        self.assertEqual(input_labels1.shape[0], points_per_crop1.shape[0])  # Same number of crops
        
        # Convert to numpy if tensor
        if hasattr(points_per_crop1, 'numpy'):
            points_per_crop1 = points_per_crop1.numpy()
        if hasattr(crop_boxes1, 'numpy'):
            crop_boxes1 = crop_boxes1.numpy()
        
        # Verify each crop's points
        for crop_idx in range(len(crop_boxes1)):
            crop_box = crop_boxes1[crop_idx]
            points = points_per_crop1[crop_idx]
            
            # Check points shape
            self.assertEqual(points.shape[0], 1)  # batch dimension
            self.assertEqual(points.shape[-1], 2)  # x,y coordinates
            
            # Verify grid dimensions
            x_coords = points[0, :, 0]  # Take first batch
            y_coords = points[0, :, 1]
            unique_x = np.unique(x_coords)
            unique_y = np.unique(y_coords)
            
            # Should have points_per_side unique coordinates in each dimension
            self.assertEqual(len(unique_x), points_per_side)
            self.assertEqual(len(unique_y), points_per_side)
            
            # Total points should be points_per_side squared
            total_points = points.size // 2
            self.assertEqual(total_points, points_per_side * points_per_side)
            
            # Verify points are monotonically increasing
            sorted_x = np.sort(unique_x)
            sorted_y = np.sort(unique_y)
            self.assertTrue(np.all(np.diff(sorted_x) > 0))
            self.assertTrue(np.all(np.diff(sorted_y) > 0))
            
            # Get the actual bounds of the points
            x_min_points = np.min(x_coords)
            x_max_points = np.max(x_coords)
            y_min_points = np.min(y_coords)
            y_max_points = np.max(y_coords)
            
            # Verify points form a regular grid
            x_spacing = (x_max_points - x_min_points) / (points_per_side - 1)
            y_spacing = (y_max_points - y_min_points) / (points_per_side - 1)
            
            # Check that spacings are roughly equal (allowing for floating point imprecision)
            x_spacings = np.diff(sorted_x)
            y_spacings = np.diff(sorted_y)
            self.assertTrue(np.allclose(x_spacings, x_spacing, rtol=1e-5))
            self.assertTrue(np.allclose(y_spacings, y_spacing, rtol=1e-5))
            
            # Verify the grid covers a reasonable portion of the crop box
            x_min, y_min, x_max, y_max = crop_box
            box_width = x_max - x_min
            box_height = y_max - y_min
            
            # Points should cover at least 60% of the crop box dimensions
            coverage_threshold = 0.6
            x_coverage = (x_max_points - x_min_points) / box_width
            y_coverage = (y_max_points - y_min_points) / box_height
            self.assertGreater(x_coverage, coverage_threshold)
            self.assertGreater(y_coverage, coverage_threshold)


@require_vision
@require_tf
class TFSamProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = SamImageProcessor()
        processor = SamProcessor(image_processor)
        processor.save_pretrained(self.tmpdirname)

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    # This is to avoid repeating the skipping of the common tests
    def prepare_image_inputs(self):
        """This function prepares a list of PIL images."""
        return prepare_image_inputs()

    def test_save_load_pretrained_additional_features(self):
        processor = SamProcessor(image_processor=self.get_image_processor())
        processor.save_pretrained(self.tmpdirname)

        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

        processor = SamProcessor.from_pretrained(self.tmpdirname, do_normalize=False, padding_value=1.0)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, SamImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()

        processor = SamProcessor(image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_feat_extract = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        input_feat_extract.pop("original_sizes")  # pop original_sizes as it is popped in the processor
        input_feat_extract.pop("reshaped_input_sizes")  # pop reshaped_input_sizes as it is popped in the processor

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    @require_tf
    def test_post_process_masks(self):
        image_processor = self.get_image_processor()

        processor = SamProcessor(image_processor=image_processor)
        dummy_masks = [tf.ones((1, 3, 5, 5))]

        original_sizes = [[1764, 2646]]

        reshaped_input_size = [[683, 1024]]
        masks = processor.post_process_masks(dummy_masks, original_sizes, reshaped_input_size, return_tensors="tf")
        self.assertEqual(masks[0].shape, (1, 3, 1764, 2646))

        masks = processor.post_process_masks(
            dummy_masks,
            tf.convert_to_tensor(original_sizes),
            tf.convert_to_tensor(reshaped_input_size),
            return_tensors="tf",
        )
        self.assertEqual(masks[0].shape, (1, 3, 1764, 2646))

        # should also work with np
        dummy_masks = [np.ones((1, 3, 5, 5))]
        masks = processor.post_process_masks(
            dummy_masks, np.array(original_sizes), np.array(reshaped_input_size), return_tensors="tf"
        )

        self.assertEqual(masks[0].shape, (1, 3, 1764, 2646))

        dummy_masks = [[1, 0], [0, 1]]
        with self.assertRaises(tf.errors.InvalidArgumentError):
            masks = processor.post_process_masks(
                dummy_masks, np.array(original_sizes), np.array(reshaped_input_size), return_tensors="tf"
            )


@require_vision
@require_torchvision
class SamProcessorEquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = SamImageProcessor()
        processor = SamProcessor(image_processor)
        processor.save_pretrained(self.tmpdirname)

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    # This is to avoid repeating the skipping of the common tests
    def prepare_image_inputs(self):
        """This function prepares a list of PIL images."""
        return prepare_image_inputs()

    @is_pt_tf_cross_test
    def test_post_process_masks_equivalence(self):
        image_processor = self.get_image_processor()

        processor = SamProcessor(image_processor=image_processor)
        dummy_masks = np.random.randint(0, 2, size=(1, 3, 5, 5)).astype(np.float32)
        tf_dummy_masks = [tf.convert_to_tensor(dummy_masks)]
        pt_dummy_masks = [torch.tensor(dummy_masks)]

        original_sizes = [[1764, 2646]]

        reshaped_input_size = [[683, 1024]]
        tf_masks = processor.post_process_masks(
            tf_dummy_masks, original_sizes, reshaped_input_size, return_tensors="tf"
        )
        pt_masks = processor.post_process_masks(
            pt_dummy_masks, original_sizes, reshaped_input_size, return_tensors="pt"
        )

        self.assertTrue(np.all(tf_masks[0].numpy() == pt_masks[0].numpy()))

    @is_pt_tf_cross_test
    def test_image_processor_equivalence(self):
        image_processor = self.get_image_processor()

        processor = SamProcessor(image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        pt_input_feat_extract = image_processor(image_input, return_tensors="pt")["pixel_values"].numpy()
        pt_input_processor = processor(images=image_input, return_tensors="pt")["pixel_values"].numpy()

        tf_input_feat_extract = image_processor(image_input, return_tensors="tf")["pixel_values"].numpy()
        tf_input_processor = processor(images=image_input, return_tensors="tf")["pixel_values"].numpy()

        self.assertTrue(np.allclose(pt_input_feat_extract, pt_input_processor))
        self.assertTrue(np.allclose(pt_input_feat_extract, tf_input_feat_extract))
        self.assertTrue(np.allclose(pt_input_feat_extract, tf_input_processor))
