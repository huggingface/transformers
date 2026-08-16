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

import tempfile
import unittest

import numpy as np

from transformers import AutoImageProcessor, MiniMaxVL01ImageProcessor, MiniMaxVL01ImageProcessorPil
from transformers.image_utils import ChannelDimension, PILImageResampling
from transformers.testing_utils import require_torch, require_torchvision, require_vision


@require_torch
@require_torchvision
@require_vision
class MiniMaxVL01ImageProcessingTest(unittest.TestCase):
    image_processing_classes = {
        "pil": MiniMaxVL01ImageProcessorPil,
        "torchvision": MiniMaxVL01ImageProcessor,
    }

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

                # Source order is base thumbnail first, followed by row-major high-resolution tiles.
                self.assertEqual(output.pixel_values[:2].shape[0], 2)
                self.assertEqual(output.pixel_values[2:5].shape[0], 3)
                self.assertEqual(output.pixel_values[5:].shape[0], 3)

    def test_floor_rounded_keep_ratio_resize_and_asymmetric_padding(self):
        wide = np.ones((3, 7, 3), dtype=np.uint8)
        tall = np.ones((7, 3, 3), dtype=np.uint8)

        for backend_name, image_processing_class in self.image_processing_classes.items():
            with self.subTest(backend=backend_name):
                processor_kwargs = self._processor_kwargs([[4, 4]])
                processor_kwargs["size"] = {"height": 4, "width": 4}
                image_processor = image_processing_class(**processor_kwargs)
                if backend_name == "torchvision":
                    import torch

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

    def test_backends_are_exact_for_nearest_neighbor_fixture(self):
        kwargs = self._processor_kwargs([[2, 2], [4, 2], [2, 4]])
        outputs = {
            name: image_processing_class(**kwargs)(list(self._fixture_images()), return_tensors="pt")
            for name, image_processing_class in self.image_processing_classes.items()
        }

        np.testing.assert_array_equal(
            outputs["pil"].pixel_values.cpu().numpy(), outputs["torchvision"].pixel_values.cpu().numpy()
        )
        self.assertEqual(outputs["pil"].image_sizes.tolist(), outputs["torchvision"].image_sizes.tolist())

    def test_rejects_unreleased_image_modes(self):
        for backend_name, image_processing_class in self.image_processing_classes.items():
            with self.subTest(backend=backend_name, stage="construction"):
                with self.assertRaisesRegex(ValueError, "only.*anyres"):
                    image_processing_class(process_image_mode="dynamic_res")

            with self.subTest(backend=backend_name, stage="preprocess"):
                image_processor = image_processing_class(**self._processor_kwargs([[2, 2]]))
                with self.assertRaisesRegex(ValueError, "only.*anyres"):
                    image_processor(self._fixture_images()[0], process_image_mode="dynamic_res", return_tensors="pt")

    def test_save_reload_through_auto_image_processor_for_both_backends(self):
        for backend_name, image_processing_class in self.image_processing_classes.items():
            with self.subTest(backend=backend_name), tempfile.TemporaryDirectory() as tmpdirname:
                image_processor = image_processing_class(**self._processor_kwargs([[2, 2]]))
                image_processor.save_pretrained(tmpdirname)
                reloaded = AutoImageProcessor.from_pretrained(tmpdirname, backend=backend_name)

                self.assertIsInstance(reloaded, image_processing_class)
                self.assertEqual(reloaded.to_dict(), image_processor.to_dict())


if __name__ == "__main__":
    unittest.main()
