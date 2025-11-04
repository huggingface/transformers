import io
import unittest

import httpx
import numpy as np
import pytest
from packaging import version

from transformers.image_utils import SizeDict
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_torchvision,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin


if is_torch_available() and is_vision_available():
    import torch

    from transformers import FuyuImageProcessor, FuyuImageProcessorFast

if is_vision_available():
    from PIL import Image


class FuyuImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_pad=True,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_rescale=True,
        rescale_factor=1 / 255,
        patch_size=None,
    ):
        size = size if size is not None else {"height": 180, "width": 360}
        patch_size = patch_size if patch_size is not None else {"height": 30, "width": 30}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = 30
        self.max_resolution = 360
        self.do_resize = do_resize
        self.size = size
        self.do_pad = do_pad
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.patch_size = patch_size

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_pad": self.do_pad,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "patch_size": self.patch_size,
        }

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        """Prepares a batch of images for testing"""
        if equal_resolution:
            image_inputs = [
                np.random.randint(
                    0, 256, (self.num_channels, self.max_resolution, self.max_resolution), dtype=np.uint8
                )
                for _ in range(self.batch_size)
            ]
        else:
            heights = [
                h - (h % 30) for h in np.random.randint(self.min_resolution, self.max_resolution, self.batch_size)
            ]
            widths = [
                w - (w % 30) for w in np.random.randint(self.min_resolution, self.max_resolution, self.batch_size)
            ]

            image_inputs = [
                np.random.randint(0, 256, (self.num_channels, height, width), dtype=np.uint8)
                for height, width in zip(heights, widths)
            ]

        if not numpify and not torchify:
            image_inputs = [Image.fromarray(np.moveaxis(img, 0, -1)) for img in image_inputs]

        if torchify:
            image_inputs = [torch.from_numpy(img) for img in image_inputs]

        return image_inputs

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]


@require_torch
@require_vision
@require_torchvision
class FuyuImageProcessorTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = FuyuImageProcessor
    fast_image_processing_class = FuyuImageProcessorFast

    # Skip tests that expect pixel_values output
    test_cast_dtype = None

    def setUp(self):
        self.image_processor_tester = FuyuImageProcessingTester(self)
        self.image_processor_dict = self.image_processor_tester.prepare_image_processor_dict()

        # Initialize image_processor_list (from ImageProcessingTestMixin)
        image_processor_list = []
        if self.test_slow_image_processor and self.image_processing_class:
            image_processor_list.append(self.image_processing_class)
        if self.test_fast_image_processor and self.fast_image_processing_class:
            image_processor_list.append(self.fast_image_processing_class)
        self.image_processor_list = image_processor_list

    def test_call_pil(self):
        """Override to handle Fuyu's custom output structure"""
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            encoded_images = image_processing(image_inputs[0], return_tensors="pt")
            self.assertIn("images", encoded_images)
            self.assertEqual(len(encoded_images.images), 1)

            encoded_images = image_processing(image_inputs, return_tensors="pt")
            self.assertIn("images", encoded_images)
            self.assertEqual(len(encoded_images.images), self.image_processor_tester.batch_size)

    def test_call_numpy(self):
        """Override to handle Fuyu's custom output structure"""
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            encoded_images = image_processing(image_inputs[0], return_tensors="pt")
            self.assertIn("images", encoded_images)
            self.assertEqual(len(encoded_images.images), 1)

            encoded_images = image_processing(image_inputs, return_tensors="pt")
            self.assertIn("images", encoded_images)
            self.assertEqual(len(encoded_images.images), self.image_processor_tester.batch_size)

    def test_call_pytorch(self):
        """Override to handle Fuyu's custom output structure"""
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            encoded_images = image_processing(image_inputs[0], return_tensors="pt")
            self.assertIn("images", encoded_images)
            self.assertEqual(len(encoded_images.images), 1)

            encoded_images = image_processing(image_inputs, return_tensors="pt")
            self.assertIn("images", encoded_images)
            self.assertEqual(len(encoded_images.images), self.image_processor_tester.batch_size)

    def test_call_numpy_4_channels(self):
        """Skip this test as Fuyu doesn't support arbitrary channels"""
        self.skipTest("Fuyu processor is designed for 3-channel RGB images")

    def test_slow_fast_equivalence(self):
        """Override to handle Fuyu's custom output structure"""
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")
        dummy_image = Image.open(
            io.BytesIO(
                httpx.get("http://images.cocodataset.org/val2017/000000039769.jpg", follow_redirects=True).content
            )
        )
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_image, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_image, return_tensors="pt")

        self._assert_slow_fast_tensors_equivalence(encoding_slow.images[0][0], encoding_fast.images[0][0])

    def test_slow_fast_equivalence_batched(self):
        """Override to handle Fuyu's custom output structure"""
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_images, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_images, return_tensors="pt")

        # Compare each image tensor
        for slow_img, fast_img in zip(encoding_slow.images, encoding_fast.images):
            self._assert_slow_fast_tensors_equivalence(slow_img[0], fast_img[0])

    @slow
    @require_torch_accelerator
    @require_vision
    @pytest.mark.torch_compile_test
    def test_can_compile_fast_image_processor(self):
        if self.fast_image_processing_class is None:
            self.skipTest("Skipping compilation test as fast image processor is not defined")
        if version.parse(torch.__version__) < version.parse("2.3"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        torch.compiler.reset()
        input_image = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
        image_processor = self.fast_image_processing_class(**self.image_processor_dict)
        output_eager = image_processor(input_image, device=torch_device, return_tensors="pt")

        image_processor = torch.compile(image_processor, mode="reduce-overhead")
        output_compiled = image_processor(input_image, device=torch_device, return_tensors="pt")
        self._assert_slow_fast_tensors_equivalence(
            output_eager.images[0][0], output_compiled.images[0][0], atol=1e-4, rtol=1e-4, mean_atol=1e-5
        )

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processor, "do_resize"))
            self.assertTrue(hasattr(image_processor, "size"))
            self.assertTrue(hasattr(image_processor, "do_pad"))
            self.assertTrue(hasattr(image_processor, "do_normalize"))
            self.assertTrue(hasattr(image_processor, "image_mean"))
            self.assertTrue(hasattr(image_processor, "image_std"))
            self.assertTrue(hasattr(image_processor, "do_rescale"))
            self.assertTrue(hasattr(image_processor, "rescale_factor"))
            self.assertTrue(hasattr(image_processor, "patch_size"))

    def test_patches(self):
        """Test that patchify_image produces the expected number of patches."""
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            batch_size = 3
            channels = 3
            height = 300
            width = 300
            image_input = torch.rand(batch_size, channels, height, width)

            expected_num_patches = image_processor.get_num_patches(image_height=height, image_width=width)
            patches_final = image_processor.patchify_image(image=image_input)

            self.assertEqual(patches_final.shape[1], expected_num_patches)

    def test_patches_match_slow_fast(self):
        """Test that fast processor produces same patches as slow processor."""
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast patch equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(
                reason="Skipping slow/fast patch equivalence test as one of the image processors is not defined"
            )

        batch_size = 3
        channels = 3
        height = 300
        width = 300
        image_input = torch.rand(batch_size, channels, height, width)

        processor_slow = self.image_processing_class(**self.image_processor_dict)
        processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        patches_fast = processor_fast.patchify_image(image=image_input)
        patches_slow = processor_slow.patchify_image(image=image_input)

        self.assertEqual(patches_fast.shape, patches_slow.shape)
        torch.testing.assert_close(patches_fast, patches_slow, rtol=1e-4, atol=1e-4)

    def test_scale_to_target_aspect_ratio(self):
        """Test that resize maintains aspect ratio correctly."""
        sample_image = np.zeros((450, 210, 3), dtype=np.uint8)

        if self.test_slow_image_processor and self.image_processing_class:
            image_processor = self.image_processing_class(**self.image_processor_dict)
            scaled_image = image_processor.resize(sample_image, size=self.image_processor_dict["size"])
            self.assertEqual(scaled_image.shape[0], 180)
            self.assertEqual(scaled_image.shape[1], 84)

        if self.test_fast_image_processor and self.fast_image_processing_class:
            image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)
            sample_tensor = torch.from_numpy(sample_image).permute(2, 0, 1).float()

            size_dict = SizeDict(
                height=self.image_processor_dict["size"]["height"], width=self.image_processor_dict["size"]["width"]
            )
            scaled_image = image_processor_fast.resize(sample_tensor, size=size_dict)

            self.assertEqual(scaled_image.shape[1], 180)
            self.assertEqual(scaled_image.shape[2], 84)

    def test_apply_transformation_numpy(self):
        """Test preprocessing with numpy input."""
        sample_image = np.zeros((450, 210, 3), dtype=np.uint8)

        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            transformed_image = image_processor.preprocess(sample_image).images[0][0]
            self.assertEqual(transformed_image.shape[1], 180)
            self.assertEqual(transformed_image.shape[2], 360)

    def test_apply_transformation_pil(self):
        """Test preprocessing with PIL input."""
        sample_image = np.zeros((450, 210, 3), dtype=np.uint8)
        sample_image_pil = Image.fromarray(sample_image)

        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            transformed_image = image_processor.preprocess(sample_image_pil).images[0][0]
            self.assertEqual(transformed_image.shape[1], 180)
            self.assertEqual(transformed_image.shape[2], 360)

    def test_preprocess_output_structure(self):
        """Test that preprocess returns correct output structure."""
        sample_image = np.zeros((450, 210, 3), dtype=np.uint8)

        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            result = image_processor.preprocess(sample_image)

            self.assertIn("images", result)
            self.assertIn("image_unpadded_heights", result)
            self.assertIn("image_unpadded_widths", result)
            self.assertIn("image_scale_factors", result)

            self.assertEqual(len(result.images), 1)
            self.assertEqual(len(result.images[0]), 1)
            self.assertEqual(len(result.image_unpadded_heights), 1)
            self.assertEqual(len(result.image_unpadded_widths), 1)
            self.assertEqual(len(result.image_scale_factors), 1)

    def test_batch_processing(self):
        """Test processing multiple images."""
        sample_image = np.zeros((450, 210, 3), dtype=np.uint8)
        sample_image_pil = Image.fromarray(sample_image)
        images = [sample_image, sample_image_pil]

        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            result = image_processor.preprocess(images)

            self.assertEqual(len(result.images), 2)
            for img in result.images:
                self.assertEqual(len(img), 1)
                if hasattr(img[0], "shape"):
                    if len(img[0].shape) == 3:
                        self.assertEqual(img[0].shape[1], 180)
                        self.assertEqual(img[0].shape[2], 360)

    def test_pad_image_fast(self):
        """Test that padding works correctly for fast processor."""
        if not self.test_fast_image_processor or self.fast_image_processing_class is None:
            self.skipTest(reason="Fast processor not available")

        from transformers.image_utils import SizeDict

        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        small_image = torch.rand(3, 100, 100)
        size_dict = SizeDict(height=180, width=360)

        padded = image_processor_fast.pad([small_image], pad_size=size_dict, fill_value=1.0)[0]
        self.assertEqual(padded.shape[1], 180)
        self.assertEqual(padded.shape[2], 360)

        self.assertTrue(torch.allclose(padded[:, 100:, :], torch.ones_like(padded[:, 100:, :])))
        self.assertTrue(torch.allclose(padded[:, :, 100:], torch.ones_like(padded[:, :, 100:])))

    def test_preprocess_with_tokenizer_info(self):
        """Test preprocess_with_tokenizer_info functionality."""
        batch_size = 2
        subseq_size = 1
        channels = 3
        image_input = torch.rand(batch_size, subseq_size, channels, 180, 360)
        image_present = torch.ones(batch_size, subseq_size, dtype=torch.bool)
        image_unpadded_h = torch.tensor([[180], [180]])
        image_unpadded_w = torch.tensor([[360], [360]])

        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)

            result = image_processor.preprocess_with_tokenizer_info(
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

    def test_device_handling_fast(self):
        """Test that fast processor can handle device placement."""
        if not self.test_fast_image_processor or self.fast_image_processing_class is None:
            self.skipTest(reason="Fast processor not available")

        sample_image = np.zeros((450, 210, 3), dtype=np.uint8)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        if torch.cuda.is_available():
            result_cuda = image_processor_fast.preprocess(sample_image, device="cuda")
            self.assertEqual(result_cuda.images[0][0].device.type, "cuda")

        result_cpu = image_processor_fast.preprocess(sample_image, device="cpu")
        self.assertEqual(result_cpu.images[0][0].device.type, "cpu")

    def test_do_not_resize_if_smaller(self):
        """Test that images smaller than target size are not resized."""
        if not self.test_fast_image_processor or self.fast_image_processing_class is None:
            self.skipTest(reason="Fast processor not available")

        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        small_image = torch.rand(3, 100, 150)
        size_dict = SizeDict(height=180, width=360)

        resized = image_processor_fast.resize(small_image, size=size_dict)

        self.assertEqual(resized.shape[1], 100)
        self.assertEqual(resized.shape[2], 150)
