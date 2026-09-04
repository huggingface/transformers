# Copyright 2026 H Company and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the NeoMME image processor."""

import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_vision_available():
    from PIL import Image


class NeoMMEImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=5,
        num_channels=3,
        min_resolution=30,
        max_resolution=80,
        do_resize=True,
        do_rescale=True,
        rescale_factor=1 / 127.5,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        patch_size=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        # These values implement `pixel / 127.5 - 1`; they are not dataset statistics.
        self.image_mean = image_mean if image_mean is not None else [1.0, 1.0, 1.0]
        self.image_std = image_std if image_std is not None else [1.0, 1.0, 1.0]
        self.patch_size = patch_size

    def prepare_image_processor_dict(self):
        """Return mixin kwargs without resolution budgets."""
        return {
            "do_resize": self.do_resize,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "patch_size": self.patch_size,
        }

    def expected_num_patches(self, image) -> int:
        """Return the native-resolution patch count."""
        if isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, np.ndarray):
            height, width = image.shape[:2] if image.shape[-1] in (1, 3, 4) else image.shape[-2:]
        else:
            height, width = image.shape[-2:]
        return -(-height // self.patch_size) * (-(-width // self.patch_size))

    def expected_output_image_shape(self, images) -> tuple[int, int]:
        """Return the shape of the concatenated, unpadded patch table."""
        return sum(self.expected_num_patches(image) for image in images), 3 * self.patch_size**2

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
class NeoMMEImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = NeoMMEImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)
            for attribute in ("do_resize", "do_rescale", "rescale_factor", "do_normalize", "patch_size"):
                self.assertTrue(hasattr(image_processing, attribute))
            for attribute in ("max_side", "size"):
                self.assertTrue(hasattr(image_processing, attribute))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.patch_size, self.image_processor_tester.patch_size)
            self.assertIsNone(image_processor.max_side)
            self.assertIsNone(image_processor.size)

            image_processor = image_processing_class.from_dict(
                self.image_processor_dict,
                patch_size=8,
                max_side=64,
                size={"min_pixels": 256, "max_pixels": 1024},
            )
            self.assertEqual(image_processor.patch_size, 8)
            self.assertEqual(image_processor.max_side, 64)
            self.assertEqual(dict(image_processor.size), {"min_pixels": 256, "max_pixels": 1024})

    def _check_call(self, image_inputs) -> None:
        for image_processing_class in self.image_processing_classes.values():
            image_processing = image_processing_class(**self.image_processor_dict)

            single = image_processing(image_inputs[0], return_tensors="pt")
            self.assertEqual(
                tuple(single.pixel_values.shape),
                self.image_processor_tester.expected_output_image_shape([image_inputs[0]]),
            )
            self.assertEqual(tuple(single.image_grid_hw.shape), (1, 2))

            batched = image_processing(image_inputs, return_tensors="pt")
            self.assertEqual(
                tuple(batched.pixel_values.shape),
                self.image_processor_tester.expected_output_image_shape(image_inputs),
            )
            self.assertEqual(tuple(batched.image_grid_hw.shape), (len(image_inputs), 2))
            self.assertEqual(int(batched.image_grid_hw.prod(dim=-1).sum()), batched.pixel_values.shape[0])

    def test_call_pil(self):
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)
        self._check_call(image_inputs)

    def test_call_numpy(self):
        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)
        self._check_call(image_inputs)

    def test_call_pytorch(self):
        import torch

        image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)
        self._check_call(image_inputs)

    @unittest.skip(reason="NeoMME is RGB-only: a 4-channel input is converted, so the patch width is always 3 * p^2")
    def test_call_numpy_4_channels(self):
        pass

    def make_image(self, height: int, width: int) -> "Image.Image":
        rng = np.random.default_rng(0)
        return Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))

    def test_rescale_and_padding(self):
        """Padding is added before rescaling, so padded pixels become exactly -1."""
        patch_size = self.image_processor_tester.patch_size
        image = Image.fromarray(np.full((patch_size, patch_size + 1, 3), 255, dtype=np.uint8))

        for backend_name, image_processing_class in self.image_processing_classes.items():
            with self.subTest(backend=backend_name):
                outputs = image_processing_class(patch_size=patch_size)(images=[image], return_tensors="np")
                self.assertEqual(outputs["image_grid_hw"].tolist(), [[1, 2]])
                np.testing.assert_allclose(outputs["pixel_values"][0], np.full(3 * patch_size**2, 1.0), atol=1e-6)
                self.assertAlmostEqual(float(outputs["pixel_values"][1].min()), -1.0, places=6)

    def test_patch_layout(self):
        patch_size = self.image_processor_tester.patch_size
        height, width = 2 * patch_size, 2 * patch_size
        array = np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(array)

        for backend_name, image_processing_class in self.image_processing_classes.items():
            with self.subTest(backend=backend_name):
                patches = image_processing_class(patch_size=patch_size)(images=[image], return_tensors="np")[
                    "pixel_values"
                ]
                self.assertEqual(patches.shape, (4, 3 * patch_size**2))

                for patch_index, (row, column) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                    block = array[
                        row * patch_size : (row + 1) * patch_size, column * patch_size : (column + 1) * patch_size
                    ]
                    np.testing.assert_allclose(patches[patch_index], block.reshape(-1) / 127.5 - 1.0, atol=1e-6)

    def test_grouped_preprocessing_matches_ungrouped(self):
        cases = {
            "repeated_shapes": ([self.make_image(8, 12), self.make_image(8, 12)], {}),
            "mixed_shapes": ([self.make_image(8, 12), self.make_image(12, 8), self.make_image(8, 12)], {}),
            "resized_to_same_shape": ([self.make_image(32, 16), self.make_image(64, 32)], {"max_side": 16}),
        }

        for backend_name, image_processing_class in self.image_processing_classes.items():
            processor = image_processing_class(patch_size=self.image_processor_tester.patch_size)
            for case, (images, kwargs) in cases.items():
                with self.subTest(backend=backend_name, case=case):
                    grouped = processor(images=images, disable_grouping=False, return_tensors="pt", **kwargs)
                    ungrouped = processor(images=images, disable_grouping=True, return_tensors="pt", **kwargs)
                    self.assertTrue(grouped.pixel_values.equal(ungrouped.pixel_values))
                    self.assertTrue(grouped.image_grid_hw.equal(ungrouped.image_grid_hw))

    def test_resolution_budgets(self):
        patch_size = self.image_processor_tester.patch_size
        image = self.make_image(64, 32)
        small = self.make_image(patch_size, patch_size)

        for backend_name, image_processing_class in self.image_processing_classes.items():
            with self.subTest(backend=backend_name):
                processor = image_processing_class(patch_size=patch_size)
                self.assertEqual(processor(images=[image], return_tensors="np")["image_grid_hw"].tolist(), [[16, 8]])
                capped = processor(images=[image], max_side=16, return_tensors="np")
                self.assertEqual(capped["image_grid_hw"].tolist(), [[4, 2]])

                # `max_side` only shrinks images; `min_pixels` can enlarge them.
                self.assertEqual(
                    processor(images=[small], max_side=1024, return_tensors="np")["image_grid_hw"].tolist(), [[1, 1]]
                )
                self.assertEqual(
                    processor(
                        images=[small],
                        size={"min_pixels": 16 * 16, "max_pixels": 10**9},
                        return_tensors="np",
                    )["image_grid_hw"].tolist(),
                    [[4, 4]],
                )

                strict_processor = image_processing_class(patch_size=1)
                side_capped = strict_processor(images=[self.make_image(101, 200)], max_side=65, return_tensors="np")[
                    "image_grid_hw"
                ][0]
                self.assertEqual(side_capped.tolist(), [33, 65])
                capped_size = strict_processor(
                    images=[self.make_image(16, 20)],
                    size={"min_pixels": 1, "max_pixels": 106},
                    return_tensors="np",
                )["image_grid_hw"][0]
                self.assertEqual(capped_size.tolist(), [9, 11])
                self.assertLessEqual(int(capped_size.prod()), 106)

                floored_size = strict_processor(
                    images=[self.make_image(16, 16)],
                    size={"min_pixels": 341, "max_pixels": 10**9},
                    return_tensors="np",
                )["image_grid_hw"][0]
                self.assertEqual(floored_size.tolist(), [19, 19])
                self.assertGreaterEqual(int(floored_size.prod()), 341)

                narrow_capped = strict_processor(
                    images=[self.make_image(1000, 1)],
                    size={"min_pixels": 1, "max_pixels": 10},
                    return_tensors="np",
                )["image_grid_hw"][0]
                self.assertEqual(narrow_capped.tolist(), [10, 1])

                rounded_cap = strict_processor(
                    images=[self.make_image(16, 16)],
                    size={"min_pixels": 300, "max_pixels": 300},
                    return_tensors="np",
                )["image_grid_hw"][0]
                self.assertEqual(rounded_cap.tolist(), [17, 17])

    def test_caps_clamp_min_pixels(self):
        """A cap takes precedence over the minimum pixel floor."""
        patch_size = self.image_processor_tester.patch_size
        image = self.make_image(64, 32)

        for backend_name, image_processing_class in self.image_processing_classes.items():
            with self.subTest(backend=backend_name):
                processor = image_processing_class(patch_size=patch_size)
                for cap, floor in (
                    ({"max_side": 16}, {"max_side": 16, "size": {"min_pixels": 10**6, "max_pixels": 10**9}}),
                    (
                        {"size": {"min_pixels": 1, "max_pixels": 64 * 32 // 4}},
                        {"size": {"min_pixels": 10**6, "max_pixels": 64 * 32 // 4}},
                    ),
                ):
                    with self.subTest(cap=cap):
                        capped = processor(images=[image], return_tensors="np", **cap)["image_grid_hw"].tolist()
                        floored = processor(images=[image], return_tensors="np", **floor)["image_grid_hw"].tolist()
                        self.assertEqual(floored, capped)

                grid = processor(
                    images=[self.make_image(4, 4)],
                    max_side=8,
                    size={"min_pixels": 1024, "max_pixels": 10**9},
                    return_tensors="np",
                )
                self.assertEqual(grid["image_grid_hw"].tolist(), [[2, 2]])

    def test_unsupported_image_kwargs_raise(self):
        processor = self.image_processing_classes["torchvision"](patch_size=self.image_processor_tester.patch_size)
        image = self.make_image(16, 16)
        for kwargs in ({"size": 8}, {"do_center_crop": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                processor(images=[image], **kwargs)

    def test_get_number_of_image_patches(self):
        patch_size = self.image_processor_tester.patch_size
        cases = [
            (9, 13, {}),
            (64, 32, {"max_side": 16}),
            (64, 32, {"do_resize": False, "max_side": 16}),
            (4, 4, {"size": {"min_pixels": 256, "max_pixels": 10**9}}),
            (64, 32, {"size": {"min_pixels": 1, "max_pixels": 24 * 24}}),
            (16, 20, {"size": {"min_pixels": 1, "max_pixels": 106}}),
            (16, 16, {"size": {"min_pixels": 341, "max_pixels": 10**9}}),
        ]

        for backend_name, image_processing_class in self.image_processing_classes.items():
            with self.subTest(backend=backend_name):
                processor = image_processing_class(patch_size=patch_size)
                for height, width, kwargs in cases:
                    outputs = processor(images=[self.make_image(height, width)], return_tensors="np", **kwargs)
                    expected = int(np.prod(outputs["image_grid_hw"][0]))
                    self.assertEqual(processor.get_number_of_image_patches(height, width, kwargs), expected)
