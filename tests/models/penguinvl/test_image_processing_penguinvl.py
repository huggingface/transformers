# Copyright 2025 Tencent and The HuggingFace Team. All rights reserved.
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

import itertools
import json
import tempfile
import unittest

import numpy as np
import requests

from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from transformers.models.penguinvl.image_processing_penguinvl import smart_resize
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import PenguinVLImageProcessor

    if is_torchvision_available():
        from transformers import PenguinVLImageProcessorFast


class PenguinVLImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        num_frames=4,
        min_resolution=56,
        max_resolution=1024,
        min_pixels=14 * 14 * 16,
        max_pixels=14 * 14 * 16384,
        do_normalize=True,
        image_mean=IMAGENET_STANDARD_MEAN,
        image_std=IMAGENET_STANDARD_STD,
        do_resize=True,
        patch_size=14,
        merge_size=1,
        do_convert_rgb=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.image_mean = image_mean
        self.image_std = image_std
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
        }

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

    def prepare_video_clip(self, num_frames=None, equal_resolution=True, numpify=False, torchify=False):
        """Prepare a single video clip as a list of frames."""
        n = num_frames if num_frames is not None else self.num_frames
        frames = prepare_image_inputs(
            batch_size=n,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )
        return frames


@require_torch
@require_vision
class PenguinVLImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = PenguinVLImageProcessor if is_vision_available() else None
    fast_image_processing_class = PenguinVLImageProcessorFast if is_torchvision_available() else None

    def setUp(self):
        super().setUp()
        self.image_processor_tester = PenguinVLImageProcessingTester(self)

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
            self.assertTrue(hasattr(image_processing, "do_convert_rgb"))
            self.assertTrue(hasattr(image_processing, "patch_size"))
            self.assertTrue(hasattr(image_processing, "merge_size"))
            self.assertTrue(hasattr(image_processing, "min_pixels"))
            self.assertTrue(hasattr(image_processing, "max_pixels"))

    def test_image_processor_to_json_string(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class(**self.image_processor_dict)
            obj = json.loads(image_processor.to_json_string())
            for key, value in self.image_processor_dict.items():
                if key not in ["min_pixels", "max_pixels"]:
                    self.assertEqual(obj[key], value)

    def test_smart_resize(self):
        best_resolution = smart_resize(561, 278, factor=28)
        self.assertEqual(best_resolution, (560, 280))

        h, w = smart_resize(300, 400, factor=14)
        self.assertEqual(h % 14, 0)
        self.assertEqual(w % 14, 0)

        min_pixels = 56 * 56
        max_pixels = 28 * 28 * 1280
        h, w = smart_resize(100, 100, factor=14, min_pixels=min_pixels, max_pixels=max_pixels)
        self.assertGreaterEqual(h * w, min_pixels)
        self.assertLessEqual(h * w, max_pixels)

    def test_call_pil(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Test single image (not batched)
            process_out = image_processing(image_inputs[0], return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (5329, 588)
            expected_image_grid_thws = torch.Tensor([[1, 73, 73]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Test batched
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (15463, 588)
            expected_image_grid_thws = torch.Tensor([[1, 47, 47]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

    def test_call_numpy(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            # Test single image
            process_out = image_processing(image_inputs[0], return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (5329, 588)
            expected_image_grid_thws = torch.Tensor([[1, 73, 73]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Test batched
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (15463, 588)
            expected_image_grid_thws = torch.Tensor([[1, 47, 47]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

    def test_call_pytorch(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=True)
            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            # Test single image
            process_out = image_processing(image_inputs[0], return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (5329, 588)
            expected_image_grid_thws = torch.Tensor([[1, 73, 73]])
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

            # Test batched
            process_out = image_processing(image_inputs, return_tensors="pt")
            encoded_images = process_out.pixel_values
            image_grid_thws = process_out.image_grid_thw
            expected_output_image_shape = (15463, 588)
            expected_image_grid_thws = torch.Tensor([[1, 47, 47]] * 7)
            self.assertEqual(tuple(encoded_images.shape), expected_output_image_shape)
            self.assertTrue((image_grid_thws == expected_image_grid_thws).all())

    @unittest.skip(reason="PenguinVLImageProcessor doesn't treat 4-channel PIL and numpy consistently")
    def test_call_numpy_4_channels(self):
        pass

    def test_video_inputs(self):
        """Test processing a single video clip (nested list [[frame1, frame2, ...]])."""
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            frames = self.image_processor_tester.prepare_video_clip(num_frames=4, equal_resolution=True)
            # Wrap in outer list to form a single clip
            video_clip = [frames]

            process_out = image_processing(video_clip, merge_size=2, return_tensors="pt")
            image_grid_thws = process_out.image_grid_thw
            image_merge_sizes = process_out.image_merge_sizes
            num_frames_per_clip = process_out.num_frames_per_clip

            # 4 frames → 4 entries in image_grid_thw
            self.assertEqual(image_grid_thws.shape[0], 4)
            # All frames in the clip should have merge_size=2
            self.assertTrue((image_merge_sizes == 2).all())
            # 1 clip with 4 frames
            self.assertEqual(len(num_frames_per_clip), 1)
            self.assertEqual(num_frames_per_clip[0], 4)

    def test_multi_image_inputs(self):
        """Test processing multiple independent images (list [img1, img2, img3])."""
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            images = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)[:3]

            process_out = image_processing(images, merge_size=1, return_tensors="pt")
            image_grid_thws = process_out.image_grid_thw
            image_merge_sizes = process_out.image_merge_sizes
            num_frames_per_clip = process_out.num_frames_per_clip

            # 3 independent images → 3 clips of 1 frame each
            self.assertEqual(image_grid_thws.shape[0], 3)
            self.assertTrue((image_merge_sizes == 1).all())
            self.assertEqual(len(num_frames_per_clip), 3)
            for n in num_frames_per_clip:
                self.assertEqual(n, 1)

    def test_nested_clip_inputs(self):
        """Test mixed nested input: [[image], [frame1, frame2, frame3]] for one image + one video."""
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            images = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)[:4]
            # First clip is a single image; second clip is a 3-frame video
            nested_clips = [[images[0]], [images[1], images[2], images[3]]]

            process_out = image_processing(nested_clips, merge_size=[1, 2], return_tensors="pt")
            num_frames_per_clip = process_out.num_frames_per_clip
            image_merge_sizes = process_out.image_merge_sizes

            self.assertEqual(len(num_frames_per_clip), 2)
            self.assertEqual(num_frames_per_clip[0], 1)  # single image clip
            self.assertEqual(num_frames_per_clip[1], 3)  # video clip

            # First frame should have merge_size=1, last 3 frames merge_size=2
            self.assertEqual(int(image_merge_sizes[0]), 1)
            self.assertTrue((image_merge_sizes[1:] == 2).all())

    def test_frame_types(self):
        """Test TRA (Temporal Redundancy-Aware) processing with frame type annotations."""
        if self.image_processing_class is None:
            self.skipTest("image_processing_class is None")

        image_processing = self.image_processing_class(**self.image_processor_dict)
        frames = self.image_processor_tester.prepare_video_clip(num_frames=4, equal_resolution=True)
        video_clip = [frames]

        # 4-frame video: frame_types 0=keyframe, 1=intermediate
        frame_types = [[0, 1, 0, 1]]

        # Without frame types
        out_no_ft = image_processing(video_clip, merge_size=2, return_tensors="pt")
        # With frame types
        out_with_ft = image_processing(video_clip, merge_size=2, frame_types=frame_types, return_tensors="pt")

        # Both should produce the same number of grid entries (one per frame)
        self.assertEqual(out_no_ft.image_grid_thw.shape[0], out_with_ft.image_grid_thw.shape[0])

        # Keyframes (type 0) should have higher or equal resolution than intermediate frames (type 1)
        grids = out_with_ft.image_grid_thw
        for i, ft in enumerate(frame_types[0]):
            grid_area = int(grids[i][1]) * int(grids[i][2])
            if ft == 0:
                # Keyframe area >= intermediate frame area in same clip
                for j, ft_j in enumerate(frame_types[0]):
                    if ft_j == 1:
                        inter_area = int(grids[j][1]) * int(grids[j][2])
                        self.assertGreaterEqual(grid_area, inter_area)

    def test_custom_image_size(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            with tempfile.TemporaryDirectory() as tmpdirname:
                image_processing.save_pretrained(tmpdirname)
                image_processor_loaded = image_processing_class.from_pretrained(
                    tmpdirname, max_pixels=56 * 56, min_pixels=28 * 28
                )

            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=True)
            process_out = image_processor_loaded(image_inputs, return_tensors="pt")
            expected_output_image_shape = [63, 588]
            self.assertListEqual(list(process_out.pixel_values.shape), expected_output_image_shape)

    def test_custom_pixels(self):
        pixel_choices = frozenset(itertools.product((100, 150, 200, 20000), (100, 150, 200, 20000)))
        for image_processing_class in self.image_processor_list:
            image_processor_dict = self.image_processor_dict.copy()
            for a_pixels, b_pixels in pixel_choices:
                image_processor_dict["min_pixels"] = min(a_pixels, b_pixels)
                image_processor_dict["max_pixels"] = max(a_pixels, b_pixels)
                image_processor = image_processing_class(**image_processor_dict)
                image_inputs = self.image_processor_tester.prepare_image_inputs()
                image_processor(image_inputs, return_tensors="pt")

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
        self.assertEqual(encoding_slow.image_grid_thw.dtype, encoding_fast.image_grid_thw.dtype)
        self._assert_slow_fast_tensors_equivalence(
            encoding_slow.image_grid_thw.float(), encoding_fast.image_grid_thw.float()
        )

    @require_vision
    @require_torch
    def test_slow_fast_equivalence_batched(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_images, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_images, return_tensors="pt")

        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)
        self.assertEqual(encoding_slow.image_grid_thw.dtype, encoding_fast.image_grid_thw.dtype)
        self._assert_slow_fast_tensors_equivalence(
            encoding_slow.image_grid_thw.float(), encoding_fast.image_grid_thw.float()
        )

    def test_get_num_patches_without_images(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)

            num_patches = image_processing.get_number_of_image_patches(height=100, width=100, images_kwargs={})
            self.assertEqual(num_patches, 49)

            num_patches = image_processing.get_number_of_image_patches(height=200, width=50, images_kwargs={})
            self.assertEqual(num_patches, 56)

            num_patches = image_processing.get_number_of_image_patches(
                height=100, width=100, images_kwargs={"patch_size": 28}
            )
            self.assertEqual(num_patches, 16)
