# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import time
import unittest

import numpy as np
import pytest
from packaging import version
from parameterized import parameterized

from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_torchvision_available, is_vision_available

from ...test_image_processing_common import (
    ImageProcessingTestMixin,
    prepare_image_inputs,
)


if is_torch_available():
    import numpy as np
    import torch

    from transformers.models.superglue.modeling_superglue import KeypointMatchingOutput

if is_vision_available():
    from transformers import SuperGlueImageProcessor

    if is_torchvision_available():
        from transformers import SuperGlueImageProcessorFast


def random_array(size):
    return np.random.randint(255, size=size)


def random_tensor(size):
    return torch.rand(size)


class SuperGlueImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=6,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        do_grayscale=True,
    ):
        size = size if size is not None else {"height": 480, "width": 640}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_grayscale = do_grayscale

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_grayscale": self.do_grayscale,
        }

    def expected_output_image_shape(self, images):
        return 2, self.num_channels, self.size["height"], self.size["width"]

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False, pairs=True, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        image_inputs = prepare_image_inputs(
            batch_size=batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )
        if pairs:
            image_inputs = [image_inputs[i : i + 2] for i in range(0, len(image_inputs), 2)]
        return image_inputs

    def prepare_keypoint_matching_output(self, pixel_values):
        max_number_keypoints = 50
        batch_size = len(pixel_values)
        mask = torch.zeros((batch_size, 2, max_number_keypoints), dtype=torch.int)
        keypoints = torch.zeros((batch_size, 2, max_number_keypoints, 2))
        matches = torch.full((batch_size, 2, max_number_keypoints), -1, dtype=torch.int)
        scores = torch.zeros((batch_size, 2, max_number_keypoints))
        for i in range(batch_size):
            random_number_keypoints0 = np.random.randint(10, max_number_keypoints)
            random_number_keypoints1 = np.random.randint(10, max_number_keypoints)
            random_number_matches = np.random.randint(5, min(random_number_keypoints0, random_number_keypoints1))
            mask[i, 0, :random_number_keypoints0] = 1
            mask[i, 1, :random_number_keypoints1] = 1
            keypoints[i, 0, :random_number_keypoints0] = torch.rand((random_number_keypoints0, 2))
            keypoints[i, 1, :random_number_keypoints1] = torch.rand((random_number_keypoints1, 2))
            random_matches_indices0 = torch.randperm(random_number_keypoints1, dtype=torch.int)[:random_number_matches]
            random_matches_indices1 = torch.randperm(random_number_keypoints0, dtype=torch.int)[:random_number_matches]
            matches[i, 0, random_matches_indices1] = random_matches_indices0
            matches[i, 1, random_matches_indices0] = random_matches_indices1
            scores[i, 0, random_matches_indices1] = torch.rand((random_number_matches,))
            scores[i, 1, random_matches_indices0] = torch.rand((random_number_matches,))
        return KeypointMatchingOutput(mask=mask, keypoints=keypoints, matches=matches, matching_scores=scores)


@require_torch
@require_vision
class SuperGlueImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = SuperGlueImageProcessor if is_vision_available() else None
    fast_image_processing_class = SuperGlueImageProcessorFast if is_torchvision_available() else None

    def setUp(self) -> None:
        super().setUp()
        self.image_processor_tester = SuperGlueImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processing(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "do_rescale"))
            self.assertTrue(hasattr(image_processing, "rescale_factor"))
            self.assertTrue(hasattr(image_processing, "do_grayscale"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 480, "width": 640})

            image_processor = image_processing_class.from_dict(
                self.image_processor_dict, size={"height": 42, "width": 42}
            )
            self.assertEqual(image_processor.size, {"height": 42, "width": 42})

    @unittest.skip(reason="SuperPointImageProcessor is always supposed to return a grayscaled image")
    def test_call_numpy_4_channels(self):
        pass

    def test_number_and_format_of_images_in_input(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)

            # Cases where the number of images and the format of lists in the input is correct
            image_input = self.image_processor_tester.prepare_image_inputs(pairs=False, batch_size=2)
            image_processed = image_processor.preprocess(image_input, return_tensors="pt")
            self.assertEqual((1, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

            image_input = self.image_processor_tester.prepare_image_inputs(pairs=True, batch_size=2)
            image_processed = image_processor.preprocess(image_input, return_tensors="pt")
            self.assertEqual((1, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

            image_input = self.image_processor_tester.prepare_image_inputs(pairs=True, batch_size=4)
            image_processed = image_processor.preprocess(image_input, return_tensors="pt")
            self.assertEqual((2, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

            image_input = self.image_processor_tester.prepare_image_inputs(pairs=True, batch_size=6)
            image_processed = image_processor.preprocess(image_input, return_tensors="pt")
            self.assertEqual((3, 2, 3, 480, 640), tuple(image_processed["pixel_values"].shape))

            # Cases where the number of images or the format of lists in the input is incorrect
            ## List of 4 images
            image_input = self.image_processor_tester.prepare_image_inputs(pairs=False, batch_size=4)
            with self.assertRaises(ValueError) as cm:
                image_processor.preprocess(image_input, return_tensors="pt")
            self.assertEqual(ValueError, cm.exception.__class__)

            ## List of 3 images
            image_input = self.image_processor_tester.prepare_image_inputs(pairs=False, batch_size=3)
            with self.assertRaises(ValueError) as cm:
                image_processor.preprocess(image_input, return_tensors="pt")
            self.assertEqual(ValueError, cm.exception.__class__)

            ## List of 2 pairs and 1 image
            image_input = self.image_processor_tester.prepare_image_inputs(pairs=True, batch_size=3)
            with self.assertRaises(ValueError) as cm:
                image_processor.preprocess(image_input, return_tensors="pt")
            self.assertEqual(ValueError, cm.exception.__class__)

    @parameterized.expand(
        [
            ([random_array((3, 100, 200)), random_array((3, 100, 200))], (1, 2, 3, 480, 640)),
            ([[random_array((3, 100, 200)), random_array((3, 100, 200))]], (1, 2, 3, 480, 640)),
            ([random_tensor((3, 100, 200)), random_tensor((3, 100, 200))], (1, 2, 3, 480, 640)),
            ([random_tensor((3, 100, 200)), random_tensor((3, 100, 200))], (1, 2, 3, 480, 640)),
        ],
    )
    def test_valid_image_shape_in_input(self, image_input, output):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            image_processed = image_processor.preprocess(image_input, return_tensors="pt")
            self.assertEqual(output, tuple(image_processed["pixel_values"].shape))

    @parameterized.expand(
        [
            (random_array((3, 100, 200)),),
            ([random_array((3, 100, 200))],),
            (random_array((1, 3, 100, 200)),),
            ([[random_array((3, 100, 200))]],),
            ([[random_array((3, 100, 200))], [random_array((3, 100, 200))]],),
            ([random_array((1, 3, 100, 200)), random_array((1, 3, 100, 200))],),
            (random_array((1, 1, 3, 100, 200)),),
        ],
    )
    def test_invalid_image_shape_in_input(self, image_input):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            with self.assertRaises(ValueError) as cm:
                image_processor(image_input, return_tensors="pt")
            self.assertEqual(ValueError, cm.exception.__class__)

    def test_input_images_properly_paired(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs()
            pre_processed_images = image_processor(image_inputs, return_tensors="pt")
            self.assertEqual(len(pre_processed_images["pixel_values"].shape), 5)
            self.assertEqual(pre_processed_images["pixel_values"].shape[1], 2)

    def test_input_not_paired_images_raises_error(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(pairs=False)
            with self.assertRaises(ValueError):
                image_processor(image_inputs[0])

    def test_input_image_properly_converted_to_grayscale(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs()
            pre_processed_images = image_processor(image_inputs, return_tensors="pt")
            for image_pair in pre_processed_images["pixel_values"]:
                for image in image_pair:
                    self.assertTrue(
                        torch.all(image[0, ...] == image[1, ...]) and torch.all(image[1, ...] == image[2, ...])
                    )

    def test_call_numpy(self):
        # Test overwritten because SuperGlueImageProcessor combines images by pair to feed it into SuperGlue

        # Initialize image_processing
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random numpy tensors
            image_pairs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            for image_pair in image_pairs:
                self.assertEqual(len(image_pair), 2)

            expected_batch_size = int(self.image_processor_tester.batch_size / 2)

            # Test with 2 images
            encoded_images = image_processing(image_pairs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_pairs[0])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test with list of pairs
            encoded_images = image_processing(image_pairs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_pairs)
            self.assertEqual(tuple(encoded_images.shape), (expected_batch_size, *expected_output_image_shape))

            # Test without paired images
            image_pairs = self.image_processor_tester.prepare_image_inputs(
                equal_resolution=False, numpify=True, pairs=False
            )
            with self.assertRaises(ValueError):
                image_processing(image_pairs, return_tensors="pt").pixel_values

    def test_call_pil(self):
        # Test overwritten because SuperGlueImageProcessor combines images by pair to feed it into SuperGlue

        # Initialize image_processing
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PIL images
            image_pairs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for image_pair in image_pairs:
                self.assertEqual(len(image_pair), 2)

            expected_batch_size = int(self.image_processor_tester.batch_size / 2)

            # Test with 2 images
            encoded_images = image_processing(image_pairs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_pairs[0])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test with list of pairs
            encoded_images = image_processing(image_pairs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_pairs)
            self.assertEqual(tuple(encoded_images.shape), (expected_batch_size, *expected_output_image_shape))

            # Test without paired images
            image_pairs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, pairs=False)
            with self.assertRaises(ValueError):
                image_processing(image_pairs, return_tensors="pt").pixel_values

    def test_call_pytorch(self):
        # Test overwritten because SuperGlueImageProcessor combines images by pair to feed it into SuperGlue

        # Initialize image_processing
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            # create random PyTorch tensors
            image_pairs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)
            for image_pair in image_pairs:
                self.assertEqual(len(image_pair), 2)

            expected_batch_size = int(self.image_processor_tester.batch_size / 2)

            # Test with 2 images
            encoded_images = image_processing(image_pairs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_pairs[0])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test with list of pairs
            encoded_images = image_processing(image_pairs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_pairs)
            self.assertEqual(tuple(encoded_images.shape), (expected_batch_size, *expected_output_image_shape))

            # Test without paired images
            image_pairs = self.image_processor_tester.prepare_image_inputs(
                equal_resolution=False, torchify=True, pairs=False
            )
            with self.assertRaises(ValueError):
                image_processing(image_pairs, return_tensors="pt").pixel_values

    def test_image_processor_with_list_of_two_images(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)

            image_pairs = self.image_processor_tester.prepare_image_inputs(
                equal_resolution=False, numpify=True, batch_size=2, pairs=False
            )
            self.assertEqual(len(image_pairs), 2)
            self.assertTrue(isinstance(image_pairs[0], np.ndarray))
            self.assertTrue(isinstance(image_pairs[1], np.ndarray))

            expected_batch_size = 1
            encoded_images = image_processing(image_pairs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_pairs[0])
            self.assertEqual(tuple(encoded_images.shape), (expected_batch_size, *expected_output_image_shape))

    @require_torch
    def test_post_processing_keypoint_matching(self):
        def check_post_processed_output(post_processed_output, image_pair_size):
            for post_processed_output, (image_size0, image_size1) in zip(post_processed_output, image_pair_size):
                self.assertTrue("keypoints0" in post_processed_output)
                self.assertTrue("keypoints1" in post_processed_output)
                self.assertTrue("matching_scores" in post_processed_output)
                keypoints0 = post_processed_output["keypoints0"]
                keypoints1 = post_processed_output["keypoints1"]
                all_below_image_size0 = torch.all(keypoints0[:, 0] <= image_size0[1]) and torch.all(
                    keypoints0[:, 1] <= image_size0[0]
                )
                all_below_image_size1 = torch.all(keypoints1[:, 0] <= image_size1[1]) and torch.all(
                    keypoints1[:, 1] <= image_size1[0]
                )
                all_above_zero0 = torch.all(keypoints0[:, 0] >= 0) and torch.all(keypoints0[:, 1] >= 0)
                all_above_zero1 = torch.all(keypoints0[:, 0] >= 0) and torch.all(keypoints0[:, 1] >= 0)
                self.assertTrue(all_below_image_size0)
                self.assertTrue(all_below_image_size1)
                self.assertTrue(all_above_zero0)
                self.assertTrue(all_above_zero1)
                all_scores_different_from_minus_one = torch.all(post_processed_output["matching_scores"] != -1)
                self.assertTrue(all_scores_different_from_minus_one)

        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs()
            pre_processed_images = image_processor.preprocess(image_inputs, return_tensors="pt")
            outputs = self.image_processor_tester.prepare_keypoint_matching_output(**pre_processed_images)

            tuple_image_sizes = [
                ((image_pair[0].size[0], image_pair[0].size[1]), (image_pair[1].size[0], image_pair[1].size[1]))
                for image_pair in image_inputs
            ]
            tuple_post_processed_outputs = image_processor.post_process_keypoint_matching(outputs, tuple_image_sizes)

            check_post_processed_output(tuple_post_processed_outputs, tuple_image_sizes)

            tensor_image_sizes = torch.tensor(
                [(image_pair[0].size, image_pair[1].size) for image_pair in image_inputs]
            ).flip(2)
            tensor_post_processed_outputs = image_processor.post_process_keypoint_matching(outputs, tensor_image_sizes)

            check_post_processed_output(tensor_post_processed_outputs, tensor_image_sizes)

    @require_torch
    def test_post_processing_keypoint_matching_with_padded_match_indices(self):
        """
        Test that post_process_keypoint_matching correctly handles matches pointing to padded keypoints.
        This tests the edge case where a match index points beyond the actual number of real keypoints,
        which would cause an out-of-bounds error without proper filtering.
        """
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)

            # Create a specific scenario with intentional padding issues
            batch_size = 1
            max_number_keypoints = 50

            # Image 0 has 10 real keypoints, image 1 has only 5 real keypoints
            num_keypoints0 = 10
            num_keypoints1 = 5

            mask = torch.zeros((batch_size, 2, max_number_keypoints), dtype=torch.int)
            keypoints = torch.zeros((batch_size, 2, max_number_keypoints, 2))
            matches = torch.full((batch_size, 2, max_number_keypoints), -1, dtype=torch.int)
            scores = torch.zeros((batch_size, 2, max_number_keypoints))

            # Set up real keypoints
            mask[0, 0, :num_keypoints0] = 1
            mask[0, 1, :num_keypoints1] = 1
            keypoints[0, 0, :num_keypoints0] = torch.rand((num_keypoints0, 2))
            keypoints[0, 1, :num_keypoints1] = torch.rand((num_keypoints1, 2))

            # Create a match that points to a padded keypoint in image 1
            # This would cause IndexError before the fix
            matches[0, 0, 0] = 8  # Points to index 8, but image 1 only has 5 real keypoints (indices 0-4)
            scores[0, 0, 0] = 0.9  # High confidence score

            # Create a valid match for comparison
            matches[0, 0, 1] = 2  # Points to index 2, which is valid
            scores[0, 0, 1] = 0.8

            outputs = KeypointMatchingOutput(mask=mask, keypoints=keypoints, matches=matches, matching_scores=scores)

            image_sizes = [((480, 640), (480, 640))]

            # This should not raise an IndexError and should filter out the invalid match
            post_processed = image_processor.post_process_keypoint_matching(outputs, image_sizes)

            # Check that we got results
            self.assertEqual(len(post_processed), 1)
            result = post_processed[0]

            # Should only have 1 valid match (index 1), the out-of-bounds match (index 0) should be filtered out
            self.assertEqual(result["keypoints0"].shape[0], 1)
            self.assertEqual(result["keypoints1"].shape[0], 1)
            self.assertEqual(result["matching_scores"].shape[0], 1)

            # Verify the match score corresponds to the valid match
            self.assertAlmostEqual(result["matching_scores"][0].item(), 0.8, places=5)

    @unittest.skip(reason="Many failing cases. This test needs a more deep investigation.")
    def test_fast_is_faster_than_slow(self):
        """Override the generic test since EfficientLoFTR requires image pairs."""
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast speed test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast speed test as one of the image processors is not defined")

        # Create image pairs for speed test
        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=False)
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        # Time slow processor
        start_time = time.time()
        for _ in range(10):
            _ = image_processor_slow(dummy_images, return_tensors="pt")
        slow_time = time.time() - start_time

        # Time fast processor
        start_time = time.time()
        for _ in range(10):
            _ = image_processor_fast(dummy_images, return_tensors="pt")
        fast_time = time.time() - start_time

        # Fast should be faster (or at least not significantly slower)
        self.assertLessEqual(
            fast_time, slow_time * 1.2, "Fast processor should not be significantly slower than slow processor"
        )

    @require_vision
    @require_torch
    def test_slow_fast_equivalence(self):
        if not self.test_slow_image_processor or not self.test_fast_image_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.image_processing_class is None or self.fast_image_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the image processors is not defined")

        dummy_image = self.image_processor_tester.prepare_image_inputs(
            equal_resolution=False, numpify=True, batch_size=2, pairs=False
        )
        image_processor_slow = self.image_processing_class(**self.image_processor_dict)
        image_processor_fast = self.fast_image_processing_class(**self.image_processor_dict)

        encoding_slow = image_processor_slow(dummy_image, return_tensors="pt")
        encoding_fast = image_processor_fast(dummy_image, return_tensors="pt")

        self._assert_slow_fast_tensors_equivalence(encoding_slow.pixel_values, encoding_fast.pixel_values)

    @slow
    @require_torch_accelerator
    @require_vision
    @pytest.mark.torch_compile_test
    def test_can_compile_fast_image_processor(self):
        """Override the generic test since EfficientLoFTR requires image pairs."""
        if self.fast_image_processing_class is None:
            self.skipTest("Skipping compilation test as fast image processor is not defined")
        if version.parse(torch.__version__) < version.parse("2.3"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        torch.compiler.reset()
        input_image = self.image_processor_tester.prepare_image_inputs(equal_resolution=True, torchify=False)
        image_processor = self.fast_image_processing_class(**self.image_processor_dict)
        output_eager = image_processor(input_image, device=torch_device, return_tensors="pt")

        image_processor = torch.compile(image_processor, mode="reduce-overhead")
        output_compiled = image_processor(input_image, device=torch_device, return_tensors="pt")
        self._assert_slow_fast_tensors_equivalence(
            output_eager.pixel_values, output_compiled.pixel_values, atol=1e-4, rtol=1e-4, mean_atol=1e-5
        )
