import unittest

import numpy as np

from transformers import is_torch_available, is_vision_available
from transformers.testing_utils import (
    require_torch,
    require_torchvision,
    require_vision,
)


if is_torch_available() and is_vision_available():
    import torch

    from transformers import FuyuImageProcessor, FuyuImageProcessorFast

if is_vision_available():
    from PIL import Image

@require_torch
@require_vision
@require_torchvision
class TestFuyuImageProcessorFast(unittest.TestCase):
    def setUp(self):
        self.size = {"height": 160, "width": 320}
        self.processor_fast = FuyuImageProcessorFast(size=self.size, padding_value=1.0)
        self.processor_slow = FuyuImageProcessor(size=self.size, padding_value=1.0)
        self.batch_size = 3
        self.channels = 3
        self.height = 300
        self.width = 300

        self.image_input = torch.rand(self.batch_size, self.channels, self.height, self.width)

        self.image_patch_dim_h = 30
        self.image_patch_dim_w = 30
        self.sample_image = np.zeros((450, 210, 3), dtype=np.uint8)
        self.sample_image_pil = Image.fromarray(self.sample_image)

    def test_is_fast(self):
        """Test that the processor correctly reports being a fast processor."""
        self.assertTrue(self.processor_fast.is_fast)
        self.assertFalse(hasattr(self.processor_slow, "is_fast") and self.processor_slow.is_fast)

    def test_patches(self):
        """Test that patchify_image produces the expected number of patches."""
        expected_num_patches = self.processor_fast.get_num_patches(image_height=self.height, image_width=self.width)

        patches_final = self.processor_fast.patchify_image(image=self.image_input)
        assert patches_final.shape[1] == expected_num_patches, (
            f"Expected {expected_num_patches} patches, got {patches_final.shape[1]}."
        )

    def test_patches_match_slow(self):
        """Test that fast processor produces same patches as slow processor."""
        patches_fast = self.processor_fast.patchify_image(image=self.image_input)
        patches_slow = self.processor_slow.patchify_image(image=self.image_input)
        self.assertEqual(patches_fast.shape, patches_slow.shape)
        torch.testing.assert_close(patches_fast, patches_slow, rtol=1e-4, atol=1e-4)

    def test_scale_to_target_aspect_ratio(self):
        """Test that resize maintains aspect ratio correctly."""
        # Convert to torch tensor for fast processor
        sample_tensor = torch.from_numpy(self.sample_image).permute(2, 0, 1).float()
        # (h:450, w:210) fitting (160, 320) -> (160, 210*160/450) = (160, 74.67) -> (160, 74)
        from transformers.image_utils import SizeDict
        size_dict = SizeDict(height=self.size["height"], width=self.size["width"])
        scaled_image = self.processor_fast.resize(sample_tensor, size=size_dict)
        self.assertEqual(scaled_image.shape[1], 160)
        self.assertEqual(scaled_image.shape[2], 74)

    def test_apply_transformation_numpy(self):
        """Test preprocessing with numpy input."""
        transformed_image = self.processor_fast.preprocess(self.sample_image).images[0][0]
        self.assertEqual(transformed_image.shape[1], 160)
        self.assertEqual(transformed_image.shape[2], 320)

    def test_apply_transformation_pil(self):
        """Test preprocessing with PIL input."""
        transformed_image = self.processor_fast.preprocess(self.sample_image_pil).images[0][0]
        self.assertEqual(transformed_image.shape[1], 160)
        self.assertEqual(transformed_image.shape[2], 320)

    def test_preprocess_output_structure(self):
        """Test that preprocess returns correct output structure."""
        result = self.processor_fast.preprocess(self.sample_image)
        # Check that result has expected keys
        self.assertIn("images", result)
        self.assertIn("image_unpadded_heights", result)
        self.assertIn("image_unpadded_widths", result)
        self.assertIn("image_scale_factors", result)
        # Check structure
        self.assertEqual(len(result.images), 1)  # One image
        self.assertEqual(len(result.images[0]), 1)  # Wrapped in list
        self.assertEqual(len(result.image_unpadded_heights), 1)
        self.assertEqual(len(result.image_unpadded_widths), 1)
        self.assertEqual(len(result.image_scale_factors), 1)

    def test_batch_processing(self):
        """Test processing multiple images."""
        images = [self.sample_image, self.sample_image_pil]
        result = self.processor_fast.preprocess(images)
        self.assertEqual(len(result.images), 2)
        for img in result.images:
            self.assertEqual(len(img), 1)
            self.assertEqual(img[0].shape[1], 160)
            self.assertEqual(img[0].shape[2], 320)

    def test_pad_image(self):
        """Test that padding works correctly."""
        from transformers.image_utils import SizeDict
        # Create small image
        small_image = torch.rand(3, 100, 100)
        size_dict = SizeDict(height=160, width=320)
        padded = self.processor_fast.pad_image(small_image, size=size_dict, constant_values=1.0)
        self.assertEqual(padded.shape[1], 160)
        self.assertEqual(padded.shape[2], 320)  
        # Check that padding values are correct (bottom right should be 1.0)
        self.assertTrue(torch.allclose(padded[:, 100:, :], torch.ones_like(padded[:, 100:, :])))
        self.assertTrue(torch.allclose(padded[:, :, 100:], torch.ones_like(padded[:, :, 100:])))

    def test_consistency_with_slow_processor(self):
        """Test that fast processor produces similar results to slow processor."""
        # Process with both processors
        result_fast = self.processor_fast.preprocess(self.sample_image_pil)
        result_slow = self.processor_slow.preprocess(self.sample_image_pil)
        # Compare images (allowing for small numerical differences)
        image_fast = result_fast.images[0][0]
        image_slow = result_slow.images[0][0]
        # Convert slow processor output to torch if needed
        if isinstance(image_slow, np.ndarray):
            image_slow = torch.from_numpy(image_slow)
        self.assertEqual(image_fast.shape, image_slow.shape)
        torch.testing.assert_close(image_fast, image_slow, rtol=1e-3, atol=1e-3)
        # Compare metadata
        self.assertEqual(result_fast.image_unpadded_heights, result_slow.image_unpadded_heights)
        self.assertEqual(result_fast.image_unpadded_widths, result_slow.image_unpadded_widths)
        # Scale factors should be very close
        scale_fast = result_fast.image_scale_factors[0][0]
        scale_slow = result_slow.image_scale_factors[0][0]
        self.assertAlmostEqual(scale_fast, scale_slow, places=5)

    def test_preprocess_with_tokenizer_info(self):
        """Test preprocess_with_tokenizer_info functionality."""
        batch_size = 2
        subseq_size = 1
        image_input = torch.rand(batch_size, subseq_size, self.channels, 180, 300)
        image_present = torch.ones(batch_size, subseq_size, dtype=torch.bool)
        image_unpadded_h = torch.tensor([[160], [160]])
        image_unpadded_w = torch.tensor([[320], [320]])
        result = self.processor_fast.preprocess_with_tokenizer_info(
            image_input=image_input,
            image_present=image_present,
            image_unpadded_h=image_unpadded_h,
            image_unpadded_w=image_unpadded_w,
            image_placeholder_id=100,
            image_newline_id=101,
            variable_sized=True,
        ) 
        # Check output structure
        self.assertIn("images", result)
        self.assertIn("image_input_ids", result)
        self.assertIn("image_patches", result)
        self.assertIn("image_patch_indices_per_batch", result)
        self.assertIn("image_patch_indices_per_subsequence", result)
        # Check batch structure
        self.assertEqual(len(result.images), batch_size)
        self.assertEqual(len(result.image_input_ids), batch_size)
        self.assertEqual(len(result.image_patches), batch_size)

    def test_device_handling(self):
        """Test that processor can handle device placement."""
        if torch.cuda.is_available():
            # Test with CUDA device
            result_cuda = self.processor_fast.preprocess(self.sample_image, device="cuda")
            self.assertEqual(result_cuda.images[0][0].device.type, "cuda")
        # Test with CPU device (should always work)
        result_cpu = self.processor_fast.preprocess(self.sample_image, device="cpu")
        self.assertEqual(result_cpu.images[0][0].device.type, "cpu")

    def test_return_tensors(self):
        """Test return_tensors parameter."""
        # Test with return_tensors="pt"
        result_pt = self.processor_fast.preprocess(self.sample_image, return_tensors="pt")
        self.assertIsInstance(result_pt.images[0][0], torch.Tensor)
        # Test without return_tensors (should still return tensors for fast processor)
        result_none = self.processor_fast.preprocess(self.sample_image)
        self.assertIsInstance(result_none.images[0][0], torch.Tensor)

    def test_do_not_resize_if_smaller(self):
        """Test that images smaller than target size are not resized."""
        from transformers.image_utils import SizeDict
        # Create small image
        small_image = torch.rand(3, 100, 150)
        size_dict = SizeDict(height=160, width=320)
        resized = self.processor_fast.resize(small_image, size=size_dict)
        # Should not be resized since it's smaller
        self.assertEqual(resized.shape[1], 100)
        self.assertEqual(resized.shape[2], 150)
        