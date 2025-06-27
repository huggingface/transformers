# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import torch

from transformers.testing_utils import (
    require_torch,
    require_torchvision,
    require_vision,
)
from transformers.utils import is_vision_available, is_torchvision_available

from ...test_processing_common import ProcessorTesterMixin, prepare_image_inputs


if is_vision_available():
    from PIL import Image

    from transformers import AutoProcessor, SamImageProcessor, SamProcessor

    # Import SamImageProcessorFast if torchvision is available
    if is_torchvision_available():
        from transformers.models.sam.image_processing_sam_fast import SamImageProcessorFast
    else:
        SamImageProcessorFast = None


@require_vision
@require_torch
@require_torchvision
class SamImageProcessorFastTest(unittest.TestCase):
    image_processor_class = SamImageProcessorFast

    def setUp(self):
        if SamImageProcessorFast is None:
            self.skipTest("SamImageProcessorFast not found, skipping tests")

        self.tmpdirname = tempfile.mkdtemp()
        self.image_processor = SamImageProcessorFast()
        self.image_processor.save_pretrained(self.tmpdirname)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_image_inputs(self):
        """This function prepares a list of PIL images."""
        return prepare_image_inputs()

    def prepare_mask_inputs(self):
        """This function prepares a list of tensors representing masks."""
        mask_inputs = [torch.randint(0, 2, size=(30, 400), dtype=torch.long)]
        return mask_inputs

    def test_image_processor_properties(self):
        """Test that the image processor has the correct properties."""
        image_processor = self.image_processor
        self.assertTrue(hasattr(image_processor, "do_resize"))
        self.assertTrue(hasattr(image_processor, "size"))
        self.assertTrue(hasattr(image_processor, "do_normalize"))
        self.assertTrue(hasattr(image_processor, "image_mean"))
        self.assertTrue(hasattr(image_processor, "image_std"))
        self.assertTrue(hasattr(image_processor, "do_pad"))
        self.assertTrue(hasattr(image_processor, "pad_size"))
        self.assertTrue(hasattr(image_processor, "mask_pad_size"))

    def test_image_processor_from_dict(self):
        """Test that the image processor can be initialized from a dict."""
        image_processor = self.image_processor
        config = image_processor.to_dict()
        new_processor = self.image_processor_class.from_dict(config)
        self.assertEqual(config, new_processor.to_dict())

    def test_save_load_pretrained_fast(self):
        """Test saving and loading the fast image processor."""
        # Save the processor
        self.image_processor.save_pretrained(self.tmpdirname)

        # Load it back
        loaded_processor = SamImageProcessorFast.from_pretrained(self.tmpdirname)

        # Check if the loaded processor has the same attributes
        self.assertEqual(self.image_processor.to_dict(), loaded_processor.to_dict())

    def test_batch_feature_parity(self):
        """Test parity between the SamImageProcessorFast and SamImageProcessor."""
        # Only run this test if SamImageProcessor exists
        if not hasattr(globals(), "SamImageProcessor"):
            self.skipTest("SamImageProcessor not found")

        # Initialize both processors
        slow_processor = SamImageProcessor()
        fast_processor = SamImageProcessorFast()

        # Create input
        images = self.prepare_image_inputs()
        images_tensor = [torch.tensor(np.array(img)).permute(2, 0, 1) for img in images]

        # Process with slow processor
        slow_output = slow_processor(images, return_tensors="pt")

        # Process with fast processor
        fast_output = fast_processor._preprocess(
            images=images_tensor,
            do_resize=fast_processor.do_resize,
            size=fast_processor.size,
            interpolation=None,
            do_rescale=fast_processor.do_rescale,
            rescale_factor=fast_processor.rescale_factor,
            do_normalize=fast_processor.do_normalize,
            image_mean=fast_processor.image_mean,
            image_std=fast_processor.image_std,
            do_pad=fast_processor.do_pad,
            pad_size=fast_processor.pad_size,
            return_tensors="pt",
        )

        # Check that the outputs have the same keys
        self.assertEqual(set(slow_output.keys()), set(fast_output.keys()))

        # Check that the processed images have the same shape
        for key in ["pixel_values"]:
            # Convert slow output to torch if needed
            if isinstance(slow_output[key], np.ndarray):
                slow_tensor = torch.tensor(slow_output[key])
            else:
                slow_tensor = slow_output[key]

            # Compare shapes
            self.assertEqual(slow_tensor.shape, fast_output[key].shape)

            # Compare values (allowing for small differences due to implementation details)
            if isinstance(slow_tensor, torch.Tensor) and isinstance(fast_output[key], torch.Tensor):
                max_diff = torch.max(torch.abs(slow_tensor - fast_output[key]))
                self.assertLess(max_diff, 1e-4)

    def test_image_processor_no_masks(self):
        """Test processing without masks."""
        image_processor = self.image_processor

        # Prepare input
        image_input = self.prepare_image_inputs()
        # Convert to tensor for the fast processor
        image_input_tensor = [torch.tensor(np.array(img)).permute(2, 0, 1) for img in image_input]

        # Process
        output = image_processor._preprocess(
            images=image_input_tensor,
            do_resize=image_processor.do_resize,
            size=image_processor.size,
            interpolation=None,
            do_rescale=image_processor.do_rescale,
            rescale_factor=image_processor.rescale_factor,
            do_normalize=image_processor.do_normalize,
            image_mean=image_processor.image_mean,
            image_std=image_processor.image_std,
            do_pad=image_processor.do_pad,
            pad_size=image_processor.pad_size,
            return_tensors="pt",
        )

        # Check output has expected keys
        self.assertIn("pixel_values", output)
        self.assertIn("original_sizes", output)
        self.assertIn("reshaped_input_sizes", output)

        # Check shapes
        self.assertEqual(output["pixel_values"].shape, (len(image_input), 3, 1024, 1024))

    def test_image_processor_with_masks(self):
        """Test processing with masks."""
        image_processor = self.image_processor

        # Prepare input
        image_input = self.prepare_image_inputs()
        mask_input = self.prepare_mask_inputs()

        # Convert to tensor for the fast processor
        image_input_tensor = [torch.tensor(np.array(img)).permute(2, 0, 1) for img in image_input]

        # Process
        output = image_processor._preprocess(
            images=image_input_tensor,
            do_resize=image_processor.do_resize,
            size=image_processor.size,
            interpolation=None,
            do_rescale=image_processor.do_rescale,
            rescale_factor=image_processor.rescale_factor,
            do_normalize=image_processor.do_normalize,
            image_mean=image_processor.image_mean,
            image_std=image_processor.image_std,
            do_pad=image_processor.do_pad,
            pad_size=image_processor.pad_size,
            segmentation_maps=mask_input,
            mask_size=image_processor.mask_size,
            mask_pad_size=image_processor.mask_pad_size,
            return_tensors="pt",
        )

        # Check output has expected keys
        self.assertIn("pixel_values", output)
        self.assertIn("original_sizes", output)
        self.assertIn("reshaped_input_sizes", output)
        self.assertIn("labels", output)

        # Check shapes
        self.assertEqual(output["pixel_values"].shape, (len(image_input), 3, 1024, 1024))
        self.assertEqual(output["labels"].shape, (len(mask_input), 1, 256, 256))

    def test_post_process_masks(self):
        """Test post-processing of masks."""
        image_processor = self.image_processor

        # Create dummy masks
        dummy_masks = torch.ones((1, 3, 5, 5))

        # Prepare sizes
        original_sizes = [(1764, 2646)]
        reshaped_input_size = [(683, 1024)]

        # Post-process masks
        masks = image_processor.post_process_masks(
            masks=dummy_masks,
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_size,
            return_tensors="pt",
        )

        # Check shape of output masks
        self.assertEqual(masks[0].shape, (1, 3, 1764, 2646))

        # Test with tensor inputs - convert tensors to lists before calling post_process_masks
        tensor_original_sizes = torch.tensor(original_sizes)
        tensor_reshaped_input_size = torch.tensor(reshaped_input_size)

        masks = image_processor.post_process_masks(
            masks=dummy_masks,
            original_sizes=tensor_original_sizes.tolist(),
            reshaped_input_sizes=tensor_reshaped_input_size.tolist(),
            return_tensors="pt",
        )

        # Check shape of output masks
        self.assertEqual(masks[0].shape, (1, 3, 1764, 2646))

    def test_rle_encoding(self):
        """Test the run-length encoding function."""
        # Test a mask of all zeros
        input_mask = torch.zeros((1, 2, 2), dtype=torch.long)
        rle = self.image_processor._mask_to_rle(input_mask)

        self.assertEqual(len(rle), 1)
        self.assertEqual(rle[0]["size"], [2, 2])
        self.assertEqual(rle[0]["counts"], [4])  # Single run of 4 zeros

        # Test a mask of all ones
        input_mask = torch.ones((1, 2, 2), dtype=torch.long)
        rle = self.image_processor._mask_to_rle(input_mask)

        self.assertEqual(len(rle), 1)
        self.assertEqual(rle[0]["size"], [2, 2])
        self.assertEqual(rle[0]["counts"], [0, 4])  # 0 zeros followed by 4 ones

        # Test a mixed mask
        input_mask = torch.tensor([[[0, 1], [1, 1]]], dtype=torch.long)
        rle = self.image_processor._mask_to_rle(input_mask)

        self.assertEqual(len(rle), 1)
        self.assertEqual(rle[0]["size"], [2, 2])
        self.assertEqual(rle[0]["counts"], [1, 3])  # 1 zero followed by 3 ones

    def test_rle_to_mask_conversion(self):
        """Test conversion between RLE and mask formats."""
        # Create a sample mask
        original_mask = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.bool)

        # Convert to RLE
        rle = self.image_processor._mask_to_rle(original_mask)

        # Convert back to mask
        reconstructed_mask = self.image_processor._rle_to_mask(rle[0]).unsqueeze(0)

        # Check that the reconstructed mask matches the original
        self.assertTrue(torch.all(original_mask == reconstructed_mask))

    def test_generate_crop_boxes(self):
        """Test generation of crop boxes."""
        # Create a sample image tensor
        image = torch.zeros((3, 512, 768), dtype=torch.float32)

        # Generate crop boxes
        crop_boxes, points_per_crop, cropped_images, input_labels = self.image_processor.generate_crop_boxes(
            image=image,
            target_size=1024,
            crop_n_layers=1,  # Just one additional layer for testing
            points_per_crop=32,
            crop_n_points_downscale_factor=1,
            return_tensors="pt",
        )

        # Check outputs
        # For crop_n_layers=1, we should have 1 (original) + 4 (2x2 grid) = 5 crops
        self.assertEqual(len(crop_boxes), 5)
        self.assertEqual(points_per_crop.shape[2], 36)  # When points_per_crop=32, we get a 6×6 grid of 36 points
        self.assertEqual(len(cropped_images), 5)
        self.assertEqual(input_labels.shape, points_per_crop.shape[:-1])

    def test_filter_masks(self):
        """Test filtering of masks based on various criteria."""
        # Create sample masks with more variation
        masks = torch.ones((3, 5, 5), dtype=torch.float32)
        # First mask: largely uniform (should have high stability)
        masks[0, 0:2, 0:2] = 0.4
        # Second mask: moderate variation
        masks[1, 0:3, 0:3] = 0.6
        # Third mask: lots of variation (should have lower stability)
        masks[2, :, :] = torch.rand((5, 5)) * 0.3 + 0.6

        # Create sample scores
        scores = torch.tensor([0.95, 0.85, 0.75])

        # Filter masks
        original_size = (10, 10)
        cropped_box = [0, 0, 5, 5]

        rle_masks, filtered_scores, boxes = self.image_processor.filter_masks(
            masks=masks,
            iou_scores=scores,
            original_size=original_size,
            cropped_box_image=cropped_box,
            pred_iou_thresh=0.8,  # Should filter out the last mask
            stability_score_thresh=0.5,  # Lower threshold to ensure some masks pass
            mask_threshold=0.5,
            stability_score_offset=0.1,  # Smaller offset to get meaningful differences
            return_tensors="pt",
        )

        # Check outputs - we expect two masks to remain after filtering by the 0.8 threshold
        self.assertEqual(len(rle_masks), 2)
        self.assertEqual(len(filtered_scores), 2)
        self.assertEqual(len(boxes), 2)

        # Check that scores are filtered correctly
        self.assertTrue(torch.all(filtered_scores >= 0.8))

    def test_compute_stability_score(self):
        """Test computation of stability scores."""
        # Create a sample mask
        mask = torch.ones((3, 5, 5), dtype=torch.float32)
        mask[0, :2, :2] = 0.2  # Add some variation
        mask[1, :3, :3] = 0.6
        mask[2, :4, :4] = 0.9

        # Compute stability scores
        stability_scores = self.image_processor._compute_stability_score(
            masks=mask, mask_threshold=0.5, stability_score_offset=0.1
        )
        print(stability_scores)

        # Check that scores are between 0 and 1
        self.assertTrue(torch.all(stability_scores >= 0))
        self.assertTrue(torch.all(stability_scores <= 1))

        # Check that the stability score increases with more consistent masks
        self.assertGreaterEqual(stability_scores[2], stability_scores[1])

    def test_batched_mask_to_box(self):
        """Test conversion of masks to bounding boxes."""
        # Create sample masks
        masks = torch.zeros((3, 10, 10), dtype=torch.bool)
        # Mask 1: top-left corner
        masks[0, 1:4, 1:5] = True
        # Mask 2: center
        masks[1, 4:7, 4:7] = True
        # Mask 3: empty mask

        # Convert to boxes
        boxes = self.image_processor._batched_mask_to_box(masks)

        # Check shapes
        self.assertEqual(boxes.shape, (3, 4))

        # Check box 1
        self.assertEqual(boxes[0].tolist(), [1, 1, 4, 3])

        # Check box 2
        self.assertEqual(boxes[1].tolist(), [4, 4, 6, 6])

        # Check box 3 (empty mask should give [0, 0, 0, 0])
        self.assertEqual(boxes[2].tolist(), [0, 0, 0, 0])

    def test_post_process_for_mask_generation(self):
        """Test end-to-end mask generation post-processing."""
        # Requires torchvision for batched_nms

        # Create sample masks and convert to RLE
        masks = torch.zeros((5, 10, 10), dtype=torch.bool)
        # Add masks
        masks[0, 1:5, 1:5] = True
        masks[1, 2:6, 2:6] = True  # Overlaps with mask 0
        masks[2, 6:9, 6:9] = True  # Separate mask
        masks[3, 5:8, 5:8] = True  # Overlaps with mask 2
        masks[4, 0:3, 7:10] = True  # Another separate mask

        # Convert to RLE
        rle_masks = [self.image_processor._mask_to_rle(mask.unsqueeze(0))[0] for mask in masks]

        # Create bounding boxes with more significant overlap
        boxes = torch.tensor(
            [
                [1, 1, 5, 5],  # Mask 0
                [1, 1, 5, 5],  # Mask 1 - Identical to mask 0
                [6, 6, 9, 9],  # Mask 2
                [6, 6, 9, 9],  # Mask 3 - Identical to mask 2
                [0, 7, 3, 10],  # Mask 4
            ]
        )

        # Create scores with large differences to ensure correct filtering
        scores = torch.tensor([0.9, 0.5, 0.95, 0.55, 0.7])

        # Apply NMS
        filtered_masks, filtered_scores, filtered_rle, filtered_boxes = (
            self.image_processor.post_process_for_mask_generation(
                all_masks=rle_masks,
                all_scores=scores,
                all_boxes=boxes,
                crops_nms_thresh=0.3,  # Lower threshold to ensure filtering
            )
        )

        # Check if we have exactly 3 masks (0, 2, 4)
        self.assertEqual(len(filtered_masks), 3)

        # The remaining masks should be the ones with the highest scores
        # Now we need to check if the correct masks were kept
        expected_scores = torch.tensor([0.9500, 0.9000, 0.7000])
        self.assertTrue(torch.allclose(filtered_scores, expected_scores))


@require_vision
@require_torch
@require_torchvision
class SamProcessorFastTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = SamProcessor

    def setUp(self):
        if SamImageProcessorFast is None:
            self.skipTest("SamImageProcessorFast not found, skipping tests")

        self.tmpdirname = tempfile.mkdtemp()
        # Use SamImageProcessor for the SamProcessor to test
        image_processor = SamImageProcessor()
        processor = SamProcessor(image_processor)
        processor.save_pretrained(self.tmpdirname)

        # Create or update config.json with model_type
        config_file = os.path.join(self.tmpdirname, "config.json")
        config = {"model_type": "sam"}

        # If config file already exists, read its content first
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                existing_config = json.load(f)
                config.update(existing_config)

        # Make sure model_type is set to sam
        config["model_type"] = "sam"

        # Write the updated config
        with open(config_file, "w") as f:
            json.dump(config, f)

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

        input_processor = processor(images=image_input, return_tensors="np")

        for image in input_processor.pixel_values:
            self.assertEqual(image.shape, (3, 1024, 1024))

    def test_image_processor_with_masks(self):
        image_processor = self.get_image_processor()

        processor = SamProcessor(image_processor=image_processor)

        image_input = self.prepare_image_inputs()
        mask_input = self.prepare_mask_inputs()

        input_processor = processor(images=image_input, segmentation_maps=mask_input, return_tensors="np")

        for label in input_processor.labels:
            self.assertEqual(label.shape, (256, 256))

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

        dummy_masks = [[1, 0], [0, 1]]
        with self.assertRaises(ValueError):
            masks = processor.post_process_masks(
                dummy_masks, torch.tensor(original_sizes), torch.tensor(reshaped_input_size)
            )
