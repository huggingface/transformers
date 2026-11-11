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

"""Testing suite for the Emu3 image processor."""

import unittest

import numpy as np
from parameterized import parameterized

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_vision_available():
    from PIL import Image

    from transformers.models.emu3.image_processing_emu3 import Emu3ImageProcessor, smart_resize


# The smallest pixel budget the processor accepts, so the dynamic resize stays fast. The
# processor does not declare `min_pixels` and `max_pixels` in its kwargs class, so they are
# passed to it explicitly rather than through the tester's processor dict.
MIN_PIXELS = 28 * 28
MAX_PIXELS = 56 * 56


class Emu3ImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        kwargs.setdefault("min_pixels", MIN_PIXELS)
        kwargs.setdefault("max_pixels", MAX_PIXELS)
        super().__init__(**kwargs)

    def expected_output_image_shape(self, images):
        """The processor resizes each image with `smart_resize`, then pads the batch to the largest shape."""
        expected_height = 0
        expected_width = 0
        for image in images:
            if isinstance(image, Image.Image):
                width, height = image.size
            elif isinstance(image, np.ndarray):
                height, width = image.shape[0], image.shape[1]
            else:
                height, width = image.shape[1], image.shape[2]

            resized_height, resized_width = smart_resize(
                height, width, factor=8, min_pixels=self.min_pixels, max_pixels=self.max_pixels
            )
            expected_height = max(expected_height, resized_height)
            expected_width = max(expected_width, resized_width)

        return self.num_channels, expected_height, expected_width


@require_vision
class Emu3ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processor_tester_class = Emu3ImageProcessingTester

    @property
    def image_processor_dict(self):
        # `image_processor_dict` is rebuilt from the tester on every access, so the values the
        # processor takes as constructor arguments are merged here rather than mutated in setUp.
        return {**super().image_processor_dict, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}

    def _assert_pixel_bounds(self, image_processor):
        self.assertIsInstance(image_processor, Emu3ImageProcessor)
        self.assertEqual(image_processor.min_pixels, MIN_PIXELS)
        self.assertEqual(image_processor.max_pixels, MAX_PIXELS)
        self.assertEqual(image_processor.size, {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS})
        self.assertEqual(image_processor.spatial_factor, 8)

    def test_image_processor_has_attributes(self):
        """The pixel bounds are constructor arguments and are stored in `size`, not in a `shortest_edge` entry."""
        for image_processing_class in self.image_processing_classes.values():
            self._assert_pixel_bounds(image_processing_class(**self.image_processor_dict))

    def test_image_processor_from_dict_has_attributes(self):
        """The constructor arguments are preserved when the processor is built from a dict."""
        for image_processing_class in self.image_processing_classes.values():
            self._assert_pixel_bounds(image_processing_class.from_dict(self.image_processor_dict))

    @unittest.skip(
        reason="Emu3ImageProcessor shares the class level `image_mean` list between instances, so the shared test for "
        "independent instance attributes fails"
    )
    def test_override_instance_attributes_does_not_affect_other_instances(self):
        pass

    def test_call_numpy_float(self):
        """Float numpy images with values in [0, 1] are a supported input type, see the `do_rescale` docstring."""
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict, do_rescale=False)
            image_inputs = [
                np.random.rand(30, 30, 3).astype(np.float32),
                np.random.rand(40, 30, 3).astype(np.float32),
            ]

            encoded_images = image_processor(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            encoded_images = image_processor(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(tuple(encoded_images.shape), (len(image_inputs), *expected_output_image_shape))

    def test_call_numpy_float_4_channels(self):
        """Float RGBA numpy images are composited onto a white background."""
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict, do_rescale=False)
            image = np.random.rand(30, 30, 4).astype(np.float32)

            encoded_images = image_processor(image, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

    @parameterized.expand([(1, "np"), (2, "np"), (1, "pt"), (2, "pt")])
    def test_image_sizes_after_channel_conversion(self, batch_size, return_tensors):
        image_processor = Emu3ImageProcessor(**self.image_processor_dict)
        images = [np.full((32, 48, 3), 100, dtype=np.uint8), np.full((48, 32, 3), 120, dtype=np.uint8)]
        images = images[:batch_size]
        reference = image_processor(images, return_tensors=return_tensors)
        self.assertEqual(reference.image_sizes.tolist(), [[32, 48], [48, 32]][:batch_size])

        for input_format in ("channels_first", "channels_last"):
            inputs = [image.transpose(2, 0, 1) for image in images] if input_format == "channels_first" else images
            for output_format in ("channels_first", "channels_last"):
                with self.subTest(input_format=input_format, output_format=output_format):
                    output = image_processor(
                        inputs,
                        input_data_format=input_format,
                        data_format=output_format,
                        return_tensors=return_tensors,
                    )
                    np.testing.assert_array_equal(output.image_sizes, reference.image_sizes)
                    pixels = np.asarray(output.pixel_values)
                    if output_format == "channels_last":
                        pixels = pixels.transpose(0, 3, 1, 2)
                    np.testing.assert_array_equal(pixels, reference.pixel_values)

    def test_image_sizes_without_resize_or_padding(self):
        image_processor = Emu3ImageProcessor(**self.image_processor_dict)
        images = [np.full((31, 47, 3), 100, dtype=np.uint8), np.full((47, 31, 3), 120, dtype=np.uint8)]
        for output_format in ("channels_first", "channels_last"):
            with self.subTest(output_format=output_format):
                output = image_processor(
                    images,
                    input_data_format="channels_last",
                    data_format=output_format,
                    do_resize=False,
                    do_pad=False,
                )
                self.assertEqual(output.image_sizes, [(31, 47), (47, 31)])
                for image, size in zip(output.pixel_values, output.image_sizes):
                    self.assertEqual(image.shape[1:] if output_format == "channels_first" else image.shape[:2], size)

    def test_padding_uses_output_channel_format(self):
        image_processor = Emu3ImageProcessor(**self.image_processor_dict)
        images = [np.full((3, width, 3), 100, dtype=np.uint8) for width in (16, 24)]
        output = image_processor(
            images,
            input_data_format="channels_last",
            data_format="channels_last",
            do_convert_rgb=False,
            do_resize=False,
            return_tensors="np",
        )
        self.assertEqual(output.pixel_values.shape, (2, 3, 24, 3))
        np.testing.assert_array_equal(output.image_sizes, [[3, 16], [3, 24]])
        np.testing.assert_array_equal(output.pixel_values[0, :, 16:, :], 0)

    @unittest.skip(reason="Emu3ImageProcessor converts every input to RGB, so it does not accept 4 channel images")
    def test_call_numpy_4_channels(self):
        pass

    @unittest.skip(
        reason="Emu3ImageProcessor converts every input to RGB and the shared conversion accepts PIL and numpy inputs, "
        "not torch tensors"
    )
    def test_call_pytorch(self):
        pass
