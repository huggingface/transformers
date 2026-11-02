# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, ChannelDimension, PILImageResampling
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class MiniMaxVL01ImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        num_channels=3,
        image_size=16,
        min_resolution=16,
        max_resolution=64,
        do_resize=True,
        do_normalize=True,
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
        do_convert_rgb=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": {"height": self.image_size, "width": self.image_size},
            "image_grid_pinpoints": [[self.image_size, self.image_size]],
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
        }

    def expected_output_image_shape(self, images):
        # The single-cell grid produces a thumbnail and one crop for each image.
        return 2 * len(images), self.num_channels, self.image_size, self.image_size

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
class MiniMaxVL01ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    test_cast_dtype = True

    def setUp(self):
        super().setUp()
        self.image_processor_tester = MiniMaxVL01ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def _test_call(self, numpify=False, torchify=False, **kwargs):
        image_inputs = self.image_processor_tester.prepare_image_inputs(numpify=numpify, torchify=torchify)
        expected_type = torch.Tensor if torchify else np.ndarray if numpify else Image.Image
        for image in image_inputs:
            self.assertIsInstance(image, expected_type)

        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            for inputs in (image_inputs[:1], image_inputs):
                with self.subTest(backend=image_processing_class.__name__, batch_size=len(inputs)):
                    encoded_images = image_processor(
                        inputs[0] if len(inputs) == 1 else inputs, return_tensors="pt", **kwargs
                    ).pixel_values
                    expected_shape = self.image_processor_tester.expected_output_image_shape(inputs)
                    self.assertEqual(tuple(encoded_images.shape), expected_shape)

    # VL-01 concatenates crops across images instead of returning an image batch dimension.
    def test_call_pil(self):
        self._test_call()

    def test_call_numpy(self):
        self._test_call(numpify=True)

    def test_call_pytorch(self):
        self._test_call(torchify=True)

    def test_call_numpy_4_channels(self):
        self.image_processor_tester.num_channels = 4
        self._test_call(
            numpify=True,
            input_data_format=ChannelDimension.LAST,
            do_convert_rgb=False,
            image_mean=[0.0] * 4,
            image_std=[1.0] * 4,
        )

    @staticmethod
    def _processor_kwargs(grid_pinpoints):
        return {
            "size": {"height": 2, "width": 2},
            "patch_size": 1,
            "image_grid_pinpoints": grid_pinpoints,
            "resample": PILImageResampling.NEAREST,
            "do_resize": True,
            "do_center_crop": False,
            "do_rescale": False,
            "do_normalize": False,
            "do_convert_rgb": True,
            "process_image_mode": "anyres",
        }

    @staticmethod
    def _fixture_images():
        square = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )

        portrait = np.empty((4, 2, 3), dtype=np.uint8)
        portrait[:2] = [20, 21, 22]
        portrait[2:] = [40, 41, 42]

        landscape = np.empty((2, 4, 3), dtype=np.uint8)
        landscape[:, :2] = [60, 61, 62]
        landscape[:, 2:] = [80, 81, 82]
        return square, portrait, landscape

    @staticmethod
    def _channels_first(array):
        return np.moveaxis(array, -1, 0)

    def _expected_flat_patch_stream(self):
        square, portrait, landscape = self._fixture_images()
        portrait_base = np.empty((2, 2, 3), dtype=np.uint8)
        portrait_base[:1] = [20, 21, 22]
        portrait_base[1:] = [40, 41, 42]
        landscape_base = np.empty((2, 2, 3), dtype=np.uint8)
        landscape_base[:, :1] = [60, 61, 62]
        landscape_base[:, 1:] = [80, 81, 82]

        expected_patches = [
            square,
            square,
            portrait_base,
            portrait[:2],
            portrait[2:],
            landscape_base,
            landscape[:, :2],
            landscape[:, 2:],
        ]
        return np.stack([self._channels_first(patch) for patch in expected_patches])

    def test_square_portrait_landscape_patch_count_and_source_order(self):
        images = self._fixture_images()
        expected = self._expected_flat_patch_stream()
        grid_pinpoints = [[2, 2], [4, 2], [2, 4]]

        for backend_name, image_processing_class in self.image_processing_classes.items():
            with self.subTest(backend=backend_name):
                image_processor = image_processing_class(**self._processor_kwargs(grid_pinpoints))
                output = image_processor(list(images), return_tensors="pt")

                self.assertEqual(tuple(output.pixel_values.shape), (8, 3, 2, 2))
                self.assertEqual(output.image_sizes.tolist(), [[2, 2], [4, 2], [2, 4]])
                np.testing.assert_array_equal(output.pixel_values.cpu().numpy(), expected)

    def test_floor_rounded_keep_ratio_resize_and_asymmetric_padding(self):
        wide = np.ones((3, 7, 3), dtype=np.uint8)
        tall = np.ones((7, 3, 3), dtype=np.uint8)

        for backend_name, image_processing_class in self.image_processing_classes.items():
            with self.subTest(backend=backend_name):
                processor_kwargs = self._processor_kwargs([[4, 4]])
                processor_kwargs["size"] = {"height": 4, "width": 4}
                image_processor = image_processing_class(**processor_kwargs)
                if backend_name == "torchvision":
                    wide_input = torch.from_numpy(self._channels_first(wide))
                    tall_input = torch.from_numpy(self._channels_first(tall))
                    wide_resized = image_processor._resize_for_patching(
                        wide_input,
                        (4, 4),
                        image_processor.resample,
                        input_data_format=ChannelDimension.FIRST,
                    )
                    tall_resized = image_processor._resize_for_patching(
                        tall_input,
                        (4, 4),
                        image_processor.resample,
                        input_data_format=ChannelDimension.FIRST,
                    )
                else:
                    wide_input = self._channels_first(wide)
                    tall_input = self._channels_first(tall)
                    wide_resized = image_processor._resize_for_patching(wide_input, (4, 4), image_processor.resample)
                    tall_resized = image_processor._resize_for_patching(tall_input, (4, 4), image_processor.resample)

                self.assertEqual(tuple(wide_resized.shape[-2:]), (1, 4))
                self.assertEqual(tuple(tall_resized.shape[-2:]), (4, 1))

                wide_padded = image_processor._pad_for_patching(wide_resized, (4, 4))
                tall_padded = image_processor._pad_for_patching(tall_resized, (4, 4))
                wide_padded = np.asarray(wide_padded)
                tall_padded = np.asarray(tall_padded)

                expected_wide = np.zeros((3, 4, 4), dtype=np.uint8)
                expected_wide[:, 1] = 1
                expected_tall = np.zeros((3, 4, 4), dtype=np.uint8)
                expected_tall[:, :, 1] = 1
                np.testing.assert_array_equal(wide_padded, expected_wide)
                np.testing.assert_array_equal(tall_padded, expected_tall)

    def test_rejects_unreleased_image_modes(self):
        for backend_name, image_processing_class in self.image_processing_classes.items():
            with self.subTest(backend=backend_name, stage="construction"):
                with self.assertRaisesRegex(ValueError, "only.*anyres"):
                    image_processing_class(process_image_mode="dynamic_res")

            with self.subTest(backend=backend_name, stage="preprocess"):
                image_processor = image_processing_class(**self._processor_kwargs([[2, 2]]))
                with self.assertRaisesRegex(ValueError, "only.*anyres"):
                    image_processor(self._fixture_images()[0], process_image_mode="dynamic_res", return_tensors="pt")


if __name__ == "__main__":
    unittest.main()
