import itertools
import tempfile
import unittest

import numpy as np
import requests

from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from transformers.models.videollama3.image_processing_videollama3 import smart_resize
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs, prepare_video_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import Videollama3ImageProcessor

    if is_torchvision_available():
        from transformers import Videollama3ImageProcessorFast


class Videollama3ImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        num_frames=10,
        min_resolution=56,
        max_resolution=1024,
        min_tokens=16,
        max_tokens=16384,
        do_normalize=True,
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        do_resize=True,
        patch_size=14,
        image_merge_size=1,
        video_merge_size=2,
        do_convert_rgb=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.image_mean = IMAGENET_STANDARD_MEAN
        self.image_std = IMAGENET_STANDARD_STD
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.patch_size = patch_size
        self.image_merge_size = image_merge_size
        self.video_merge_size = video_merge_size
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "patch_size": self.patch_size,
            "image_merge_size": self.image_merge_size,
            "video_merge_size": self.video_merge_size,
        }

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        images = prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )
        return [[image] for image in images]

    def prepare_video_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_video_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            num_frames=self.num_frames,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


@require_torch
@require_vision
class Videollama3ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = Videollama3ImageProcessor if is_vision_available() else None
    fast_image_processing_class = Videollama3ImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = Videollama3ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_normalize"))
            self.assertTrue(hasattr(image_processing, "image_mean"))
            self.assertTrue(hasattr(image_processing, "image_std"))
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "min_tokens"))
            self.assertTrue(hasattr(image_processing, "max_tokens"))
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))
            self.assertTrue(hasattr(image_processing, "patch_size"))
            self.assertTrue(hasattr(image_processing, "image_merge_size"))
            self.assertTrue(hasattr(image_processing, "video_merge_size"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.min_tokens, 16)
            self.assertEqual(image_processor.max_tokens, 16384)

            image_processor = image_processing_class.from_dict(
                self.image_processor_dict, min_tokens=16, max_tokens=16384
            )
            self.assertEqual(image_processor.min_tokens, 16)
            self.assertEqual(image_processor.max_tokens, 16384)

    def test_select_best_resolution(self):
        # Test with a final resize resolution
        best_resolution = smart_resize(561, 278, factor=14)
        self.assertEqual(best_resolution, (560, 280))

    def test_call_pil(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            for image in image_inputs:
                self.assertIsInstance(image[0], Image.Image)

            # Test not batched input
            process_out = image_processing(image_inputs[0], return_tensors="pt")
            encoded_images = process_out.pixel_values
            grid_sizes = process_out.grid_sizes
            expected_output_image_shape = (5329, 588)
            expected_grid_sizes = torch.Tensor([[1, 73, 73]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((grid_sizes == expected_grid_sizes).all())

            # Test batched
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            grid_sizes = process_out.grid_sizes
            expected_output_image_shape = (37303, 588)
            expected_grid_sizes = torch.Tensor([[1, 73, 73]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((grid_sizes == expected_grid_sizes).all())

    def test_call_numpy(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image[0], np.ndarray)

            # Test not batched input
            process_out = image_processing(image_inputs[0], return_tensors="pt")
            encoded_images = process_out.pixel_values
            grid_sizes = process_out.grid_sizes
            expected_output_image_shape = (5329, 588)
            expected_grid_sizes = torch.Tensor([[1, 73, 73]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((grid_sizes == expected_grid_sizes).all())

            # Test batched
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            grid_sizes = process_out.grid_sizes
            expected_output_image_shape = (37303, 588)
            expected_grid_sizes = torch.Tensor([[1, 73, 73]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((grid_sizes == expected_grid_sizes).all())

    def test_call_pytorch(self):
        for image_processing_class in self.image_processor_list:
            # Initialize image_processing
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)

            for image in image_inputs:
                self.assertIsInstance(image[0], torch.Tensor)

            # Test not batched input
            process_out = image_processing(image_inputs[0], return_tensors="pt")
            encoded_images = process_out.pixel_values
            grid_sizes = process_out.grid_sizes
            expected_output_image_shape = (5329, 588)
            expected_grid_sizes = torch.Tensor([[1, 73, 73]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((grid_sizes == expected_grid_sizes).all())

            # Test batched
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            grid_sizes = process_out.grid_sizes
            expected_output_image_shape = (37303, 588)
            expected_grid_sizes = torch.Tensor([[1, 73, 73]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((grid_sizes == grid_sizes).all())

    @unittest.skip(reason="Videollama3ImageProcessor doesn't treat 4 channel PIL and numpy consistently yet")
    def test_call_numpy_4_channels(self):
        pass

    def test_nested_input(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)

            # Test batched as a list of images
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            grid_sizes = process_out.grid_sizes
            expected_output_image_shape = (37303, 588)
            expected_grid_sizes = torch.Tensor([[1, 73, 73]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((grid_sizes == expected_grid_sizes).all())

            # Test batched as a nested list of images, where each sublist is one batch
            image_inputs_nested = image_inputs[:3] + image_inputs[3:]
            process_out = image_processing(image_inputs_nested, return_tensors="pt")
            encoded_images_nested = process_out.pixel_values
            grid_sizes_nested = process_out.grid_sizes
            expected_output_image_shape = (37303, 588)
            expected_grid_sizes = torch.Tensor([[1, 73, 73]] * 7)
            self.assertEqual(tuple(encoded_images_nested.shape), expected_output_image_shape)
            self.assertTrue((grid_sizes_nested == expected_grid_sizes).all())

            # Image processor should return same pixel values, independently of ipnut format
            self.assertTrue((encoded_images_nested == encoded_images).all())
            self.assertTrue((grid_sizes_nested == grid_sizes).all())

    def test_video_inputs(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            expected_dims_by_frames = {i: 38332 * i for i in range(1, 7)}

            for num_frames, expected_dims in expected_dims_by_frames.items():
                image_processor_tester = Videollama3ImageProcessingTester(self, num_frames=num_frames)
                video_inputs = image_processor_tester.prepare_video_inputs(equal_resolution=True)
                process_out = image_processing(None, videos=video_inputs, return_tensors="pt")
                encoded_video = process_out.pixel_values_videos
                expected_output_video_shape = (expected_dims, 588)
                self.assertEqual(tuple(encoded_video.shape), expected_output_video_shape)

    def test_custom_image_size(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            with tempfile.TemporaryDirectory() as tmpdirname:
                image_processing.save_pretrained(tmpdirname)
                image_processor_loaded = image_processing_class.from_pretrained(
                    tmpdirname, min_tokens=16, max_tokens=1024
                )

            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            process_out = image_processor_loaded(image_inputs, return_tensors="pt")
            expected_output_video_shape = [7168, 588]
            self.assertListEqual(list(process_out.pixel_values.shape), expected_output_video_shape)

    @require_vision
    @require_torch
    def test_slow_fast_equivalence(self):
        dummy_image = Image.open(
            requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw
        )

        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_image, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_image, return_tensors="pt")

        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)
        self.assertEqual(encoding_slow.grid_sizes.dtype, encoding_fast.grid_sizes.dtype)
        self._assert_slow_fast_tensors_equivalence(
            encoding_slow.grid_sizes.float(), encoding_fast.grid_sizes.float()
        )
        self.assertEqual(encoding_slow.merge_sizes.dtype, encoding_fast.merge_sizes.dtype)
        self._assert_slow_fast_tensors_equivalence(
            encoding_slow.merge_sizes.float(), encoding_fast.merge_sizes.float()
        )

    @require_vision
    @require_torch
    def test_slow_fast_equivalence_batched(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        if hasattr(self.image_processor_tester, "do_center_crop") and self.image_processor_tester.do_center_crop:
            self.skipTest(
                reason="Skipping as do_center_crop is True and center_crop functions are not equivalent for fast and slow processors"
            )

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_images, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_images, return_tensors="pt")

        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)
        self.assertEqual(encoding_slow.grid_sizes.dtype, encoding_fast.grid_sizes.dtype)
        self._assert_slow_fast_tensors_equivalence(
            encoding_slow.grid_sizes.float(), encoding_fast.grid_sizes.float()
        )
        self.assertEqual(encoding_slow.merge_sizes.dtype, encoding_fast.merge_sizes.dtype)
        self._assert_slow_fast_tensors_equivalence(
            encoding_slow.merge_sizes.float(), encoding_fast.merge_sizes.float()
        )
