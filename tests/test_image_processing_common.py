# Copyright 2023 HuggingFace Inc.
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
import inspect
import io
import json
import os
import pathlib
import subprocess
import tempfile
import warnings
from copy import deepcopy
from datetime import datetime

import httpx
import numpy as np
import pytest

from transformers import AutoImageProcessor, BatchFeature
from transformers.image_utils import AnnotationFormat, AnnotionFormat
from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING_NAMES
from transformers.testing_utils import (
    check_json_file_has_correct_format,
    require_torch,
    require_torch_accelerator,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


def prepare_image_inputs(
    batch_size,
    min_resolution,
    max_resolution,
    num_channels,
    size_divisor=None,
    equal_resolution=False,
    numpify=False,
    torchify=False,
):
    """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
    or a list of PyTorch tensors if one specifies torchify=True.

    One can specify whether the images are of the same resolution or not.
    """

    assert not (numpify and torchify), "You cannot specify both numpy and PyTorch tensors at the same time"

    image_inputs = []
    for i in range(batch_size):
        if equal_resolution:
            width = height = max_resolution
        else:
            # To avoid getting image width/height 0
            if size_divisor is not None:
                # If `size_divisor` is defined, the image needs to have width/size >= `size_divisor`
                min_resolution = max(size_divisor, min_resolution)
            width, height = np.random.choice(np.arange(min_resolution, max_resolution), 2)
        image_inputs.append(np.random.randint(255, size=(num_channels, width, height), dtype=np.uint8))

    if not numpify and not torchify:
        # PIL expects the channel dimension as last dimension
        image_inputs = [Image.fromarray(np.moveaxis(image, 0, -1)) for image in image_inputs]

    if torchify:
        image_inputs = [torch.from_numpy(image) for image in image_inputs]

    if numpify:
        # Numpy images are typically in channels last format
        image_inputs = [image.transpose(1, 2, 0) for image in image_inputs]

    return image_inputs


def prepare_video(num_frames, num_channels, width=10, height=10, numpify=False, torchify=False):
    """This function prepares a video as a list of PIL images/NumPy arrays/PyTorch tensors."""

    video = []
    for i in range(num_frames):
        video.append(np.random.randint(255, size=(num_channels, width, height), dtype=np.uint8))

    if not numpify and not torchify:
        # PIL expects the channel dimension as last dimension
        video = [Image.fromarray(np.moveaxis(frame, 0, -1)) for frame in video]

    if torchify:
        video = [torch.from_numpy(frame) for frame in video]

    return video


def prepare_video_inputs(
    batch_size,
    num_frames,
    num_channels,
    min_resolution,
    max_resolution,
    equal_resolution=False,
    numpify=False,
    torchify=False,
):
    """This function prepares a batch of videos: a list of list of PIL images, or a list of list of numpy arrays if
    one specifies numpify=True, or a list of list of PyTorch tensors if one specifies torchify=True.

    One can specify whether the videos are of the same resolution or not.
    """

    assert not (numpify and torchify), "You cannot specify both numpy and PyTorch tensors at the same time"

    video_inputs = []
    for _ in range(batch_size):
        if equal_resolution:
            width = height = max_resolution
        else:
            width, height = np.random.choice(np.arange(min_resolution, max_resolution), 2)
        video = prepare_video(
            num_frames=num_frames,
            num_channels=num_channels,
            width=width,
            height=height,
            numpify=numpify,
            torchify=torchify,
        )
        video_inputs.append(video)

    return video_inputs


class ImageProcessingTestMixin:
    test_cast_dtype = None
    image_processing_class = None
    image_processors_backends_list = None

    def setUp(self):
        self.image_processors_backends_list = [
            backend_name
            for backend_name in self.image_processing_class._backend_classes.keys()
            if self.image_processing_class._backend_availability_checks[backend_name]()
        ]

    def _assert_tensors_equivalence(self, tensor1, tensor2, atol=1e-1, rtol=1e-3, mean_atol=5e-3):
        """Assert that two tensors are equivalent within specified tolerances."""
        torch.testing.assert_close(tensor1, tensor2, atol=atol, rtol=rtol)
        self.assertLessEqual(torch.mean(torch.abs(tensor1 - tensor2)).item(), mean_atol)

    @require_vision
    @require_torch
    def test_backends_equivalence(self):
        if len(self.image_processors_backends_list) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_image = Image.open(
            io.BytesIO(
                httpx.get("http://images.cocodataset.org/val2017/000000039769.jpg", follow_redirects=True).content
            )
        )

        # Create processors for each backend
        encodings = {}
        for backend_name in self.image_processors_backends_list:
            image_processor = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_image, return_tensors="pt")

        # Compare all backends to the first one (reference backend)
        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_encoding = encodings[reference_backend].pixel_values
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_encoding, encodings[backend_name].pixel_values)

    @require_vision
    @require_torch
    def test_slow_fast_equivalence_batched(self):
        if len(self.image_processors_backends_list) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        dummy_images = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

        # Create processors for each backend
        encodings = {}
        for backend_name in self.image_processors_backends_list:
            image_processor = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            encodings[backend_name] = image_processor(dummy_images, return_tensors="pt")

        # Compare all backends to the first one (reference backend)
        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_encoding = encodings[reference_backend].pixel_values
        for backend_name in backend_names[1:]:
            self._assert_tensors_equivalence(reference_encoding, encodings[backend_name].pixel_values)

    def test_image_processor_to_json_string(self):
        for backend_name in self.image_processors_backends_list:
            image_processor = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            obj = json.loads(image_processor.to_json_string())
            for key, value in self.image_processor_dict.items():
                self.assertEqual(obj[key], value)

    def test_image_processor_to_json_file(self):
        for backend_name in self.image_processors_backends_list:
            image_processor_first = self.image_processing_class(backend=backend_name, **self.image_processor_dict)

            with tempfile.TemporaryDirectory() as tmpdirname:
                json_file_path = os.path.join(tmpdirname, "image_processor.json")
                image_processor_first.to_json_file(json_file_path)
                image_processor_second = self.image_processing_class.from_json_file(json_file_path)

            self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

    def test_image_processor_from_and_save_pretrained(self):
        for backend_name in self.image_processors_backends_list:
            image_processor_first = self.image_processing_class(backend=backend_name, **self.image_processor_dict)

            with tempfile.TemporaryDirectory() as tmpdirname:
                saved_file = image_processor_first.save_pretrained(tmpdirname)[0]
                check_json_file_has_correct_format(saved_file)
                image_processor_second = self.image_processing_class.from_pretrained(tmpdirname)

            self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

    def test_image_processor_save_load_with_autoimageprocessor(self):
        for backend_name in self.image_processors_backends_list:
            image_processor_first = self.image_processing_class(backend=backend_name, **self.image_processor_dict)

            with tempfile.TemporaryDirectory() as tmpdirname:
                saved_file = image_processor_first.save_pretrained(tmpdirname)[0]
                check_json_file_has_correct_format(saved_file)

                image_processor_second = AutoImageProcessor.from_pretrained(tmpdirname, backend=backend_name)

            self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

    def test_save_load_backends(self):
        "Test that we can load image processors with different backends from each other."
        if len(self.image_processors_backends_list) < 2:
            self.skipTest("Skipping backend save/load test as there are less than 2 backends")

        image_processor_dict = self.image_processor_tester.prepare_image_processor_dict()
        backend_names = self.image_processors_backends_list

        # Test cross-loading between all backend pairs
        for backend1 in backend_names:
            processor1 = self.image_processing_class(backend=backend1, **image_processor_dict)

            for backend2 in backend_names:
                if backend1 == backend2:
                    continue

                # Load backend2 processor from backend1 saved one
                with tempfile.TemporaryDirectory() as tmpdirname:
                    processor1.save_pretrained(tmpdirname)
                    processor2 = self.image_processing_class.from_pretrained(tmpdirname, backend=backend2)

                # Compare dictionaries (allowing for backend-specific differences)
                dict1 = processor1.to_dict()
                dict2 = processor2.to_dict()
                difference = {
                    key: dict1.get(key) if key in dict1 else dict2.get(key) for key in set(dict1) ^ set(dict2)
                }
                dict1_common = {key: dict1[key] for key in set(dict1) & set(dict2)}
                dict2_common = {key: dict2[key] for key in set(dict1) & set(dict2)}
                # check that all additional keys are None, except for `default_to_square` and `data_format` which are backend-specific
                self.assertTrue(
                    all(
                        value is None
                        for key, value in difference.items()
                        if key not in ["default_to_square", "data_format"]
                    ),
                    f"Backends {backend1} and {backend2} differ in unexpected keys: {difference}",
                )
                # check that the remaining keys are the same
                self.assertEqual(
                    dict1_common, dict2_common, f"Backends {backend1} and {backend2} differ in common keys"
                )

    def test_save_load_backends_auto(self):
        "Test that we can load image processors with different backends from each other using AutoImageProcessor."
        if len(self.image_processors_backends_list) < 2:
            self.skipTest("Skipping backend save/load test as there are less than 2 backends")

        image_processor_dict = self.image_processor_tester.prepare_image_processor_dict()
        backend_names = self.image_processors_backends_list

        # Test cross-loading between all backend pairs using AutoImageProcessor
        for backend1 in backend_names:
            processor1 = self.image_processing_class(backend=backend1, **image_processor_dict)

            for backend2 in backend_names:
                if backend1 == backend2:
                    continue

                # Load backend2 processor from backend1 saved one using AutoImageProcessor
                with tempfile.TemporaryDirectory() as tmpdirname:
                    processor1.save_pretrained(tmpdirname)
                    processor2 = AutoImageProcessor.from_pretrained(tmpdirname, backend=backend2)

                # Compare dictionaries (allowing for backend-specific differences)
                dict1 = processor1.to_dict()
                dict2 = processor2.to_dict()
                difference = {
                    key: dict1.get(key) if key in dict1 else dict2.get(key) for key in set(dict1) ^ set(dict2)
                }
                dict1_common = {key: dict1[key] for key in set(dict1) & set(dict2)}
                dict2_common = {key: dict2[key] for key in set(dict1) & set(dict2)}
                # check that all additional keys are None, except for `default_to_square` and `data_format` which are backend-specific
                self.assertTrue(
                    all(
                        value is None
                        for key, value in difference.items()
                        if key not in ["default_to_square", "data_format"]
                    ),
                    f"Backends {backend1} and {backend2} differ in unexpected keys: {difference}",
                )
                # check that the remaining keys are the same
                self.assertEqual(
                    dict1_common, dict2_common, f"Backends {backend1} and {backend2} differ in common keys"
                )

    def test_init_without_params(self):
        for backend_name in self.image_processors_backends_list:
            image_processor = self.image_processing_class(backend=backend_name)
            self.assertIsNotNone(image_processor)

    @require_torch
    @require_vision
    def test_cast_dtype_device(self):
        for backend_name in self.image_processors_backends_list:
            if self.test_cast_dtype is not None:
                # Initialize image_processor
                image_processor = self.image_processing_class(backend=backend_name, **self.image_processor_dict)

                # create random PyTorch tensors
                image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

                encoding = image_processor(image_inputs, return_tensors="pt")
                # for layoutLM compatibility
                self.assertEqual(encoding.pixel_values.device, torch.device("cpu"))
                self.assertEqual(encoding.pixel_values.dtype, torch.float32)

                encoding = image_processor(image_inputs, return_tensors="pt").to(torch.float16)
                self.assertEqual(encoding.pixel_values.device, torch.device("cpu"))
                self.assertEqual(encoding.pixel_values.dtype, torch.float16)

                encoding = image_processor(image_inputs, return_tensors="pt").to("cpu", torch.bfloat16)
                self.assertEqual(encoding.pixel_values.device, torch.device("cpu"))
                self.assertEqual(encoding.pixel_values.dtype, torch.bfloat16)

                with self.assertRaises(TypeError):
                    _ = image_processor(image_inputs, return_tensors="pt").to(torch.bfloat16, "cpu")

                # Try with text + image feature
                encoding = image_processor(image_inputs, return_tensors="pt")
                encoding.update({"input_ids": torch.LongTensor([[1, 2, 3], [4, 5, 6]])})
                encoding = encoding.to(torch.float16)

                self.assertEqual(encoding.pixel_values.device, torch.device("cpu"))
                self.assertEqual(encoding.pixel_values.dtype, torch.float16)
                self.assertEqual(encoding.input_ids.dtype, torch.long)

    def test_call_pil(self):
        for backend_name in self.image_processors_backends_list:
            # Initialize image_processing
            image_processing = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            # create random PIL images
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)
            for image in image_inputs:
                self.assertIsInstance(image, Image.Image)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_call_numpy(self):
        for backend_name in self.image_processors_backends_list:
            # Initialize image_processing
            image_processing = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            # create random numpy tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)
            for image in image_inputs:
                self.assertIsInstance(image, np.ndarray)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_call_pytorch(self):
        for backend_name in self.image_processors_backends_list:
            # Initialize image_processing
            image_processing = self.image_processing_class(backend=backend_name, **self.image_processor_dict)
            # create random PyTorch tensors
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

            for image in image_inputs:
                self.assertIsInstance(image, torch.Tensor)

            # Test not batched input
            encoded_images = image_processing(image_inputs[0], return_tensors="pt").pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            encoded_images = image_processing(image_inputs, return_tensors="pt").pixel_values
            self.assertEqual(
                tuple(encoded_images.shape),
                (self.image_processor_tester.batch_size, *expected_output_image_shape),
            )

    def test_call_numpy_4_channels(self):
        for backend_name in self.image_processors_backends_list:
            # Test that can process images which have an arbitrary number of channels
            # Initialize image_processing
            image_processor = self.image_processing_class(backend=backend_name, **self.image_processor_dict)

            # create random numpy tensors
            self.image_processor_tester.num_channels = 4
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)

            # Test not batched input
            encoded_images = image_processor(
                image_inputs[0],
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=[0.0, 0.0, 0.0, 0.0],
                image_std=[1.0, 1.0, 1.0, 1.0],
            ).pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape([image_inputs[0]])
            self.assertEqual(tuple(encoded_images.shape), (1, *expected_output_image_shape))

            # Test batched
            encoded_images = image_processor(
                image_inputs,
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=[0.0, 0.0, 0.0, 0.0],
                image_std=[1.0, 1.0, 1.0, 1.0],
            ).pixel_values
            expected_output_image_shape = self.image_processor_tester.expected_output_image_shape(image_inputs)
            self.assertEqual(
                tuple(encoded_images.shape), (self.image_processor_tester.batch_size, *expected_output_image_shape)
            )

    def test_image_processor_preprocess_arguments(self):
        is_tested = False

        for backend_name in self.image_processors_backends_list:
            image_processor = self.image_processing_class(backend=backend_name, **self.image_processor_dict)

            # validation done by _valid_processor_keys attribute
            if hasattr(image_processor, "_valid_processor_keys") and hasattr(image_processor, "preprocess"):
                preprocess_parameter_names = inspect.getfullargspec(image_processor.preprocess).args
                preprocess_parameter_names.remove("self")
                preprocess_parameter_names.sort()
                valid_processor_keys = image_processor._valid_processor_keys
                valid_processor_keys.sort()
                self.assertEqual(preprocess_parameter_names, valid_processor_keys)
                is_tested = True

            # validation done by @filter_out_non_signature_kwargs decorator
            if hasattr(image_processor.preprocess, "_filter_out_non_signature_kwargs"):
                if hasattr(self.image_processor_tester, "prepare_image_inputs"):
                    inputs = self.image_processor_tester.prepare_image_inputs()
                elif hasattr(self.image_processor_tester, "prepare_video_inputs"):
                    inputs = self.image_processor_tester.prepare_video_inputs()
                else:
                    self.skipTest(reason="No valid input preparation method found")

                with warnings.catch_warnings(record=True) as raised_warnings:
                    warnings.simplefilter("always")
                    image_processor(inputs, extra_argument=True)

                messages = " ".join([str(w.message) for w in raised_warnings])
                self.assertGreaterEqual(len(raised_warnings), 1)
                self.assertIn("extra_argument", messages)
                is_tested = True

        if not is_tested:
            self.skipTest(reason="No validation found for `preprocess` method")

    def test_override_instance_attributes_does_not_affect_other_instances(self):
        # Test with all available backends
        for backend_name in self.image_processors_backends_list:
            with self.subTest(backend=backend_name):
                image_processor_1 = self.image_processing_class(backend=backend_name)
                image_processor_2 = self.image_processing_class(backend=backend_name)
                if not (hasattr(image_processor_1, "size") and isinstance(image_processor_1.size, dict)) or not (
                    hasattr(image_processor_1, "image_mean") and isinstance(image_processor_1.image_mean, list)
                ):
                    self.skipTest(
                        reason="Skipping test as the image processor does not have dict size or list image_mean attributes"
                    )

                original_size_2 = deepcopy(image_processor_2.size)
                for key in image_processor_1.size:
                    image_processor_1.size[key] = -1
                modified_copied_size_1 = deepcopy(image_processor_1.size)

                original_image_mean_2 = deepcopy(image_processor_2.image_mean)
                image_processor_1.image_mean[0] = -1
                modified_copied_image_mean_1 = deepcopy(image_processor_1.image_mean)

                # check that the original attributes of the second instance are not affected
                self.assertEqual(image_processor_2.size, original_size_2)
                self.assertEqual(image_processor_2.image_mean, original_image_mean_2)

                for key in image_processor_2.size:
                    image_processor_2.size[key] = -2
                image_processor_2.image_mean[0] = -2

                # check that the modified attributes of the first instance are not affected by the second instance
                self.assertEqual(image_processor_1.size, modified_copied_size_1)
                self.assertEqual(image_processor_1.image_mean, modified_copied_image_mean_1)

    @slow
    @require_torch_accelerator
    @require_vision
    @pytest.mark.torch_compile_test
    def test_can_compile_torchvision_backend(self):
        # Test compilation with torchvision backend (equivalent to fast processor)
        if "torchvision" not in self.image_processors_backends_list:
            self.skipTest("Skipping compilation test as torchvision backend is not available")

        torch.compiler.reset()
        input_image = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
        image_processor = self.image_processing_class(backend="torchvision", **self.image_processor_dict)
        output_eager = image_processor(input_image, device=torch_device, return_tensors="pt")

        image_processor = torch.compile(image_processor, mode="reduce-overhead")
        output_compiled = image_processor(input_image, device=torch_device, return_tensors="pt")
        self._assert_tensors_equivalence(
            output_eager.pixel_values, output_compiled.pixel_values, atol=1e-4, rtol=1e-4, mean_atol=1e-5
        )

    def test_new_models_require_torchvision_backend(self):
        """
        Test that new models support the torchvision backend.
        For more information on how to implement backend support, see this issue: https://github.com/huggingface/transformers/issues/36978,
        and ping @yonigozlan for help.
        """
        # Check if torchvision backend is available
        if "torchvision" in self.image_processors_backends_list:
            return
        if self.image_processing_class is None:
            self.skipTest("No image processing class defined")

        def _is_old_model_by_commit_date(model_type, date_cutoff=(2025, 9, 1)):
            try:
                # Convert model_type to directory name and construct file path
                model_dir = model_type.replace("-", "_")
                slow_processor_file = f"src/transformers/models/{model_dir}/image_processing_{model_dir}.py"
                # Check if the file exists otherwise skip the test
                if not os.path.exists(slow_processor_file):
                    return None
                # Get the first commit date of the slow processor file
                result = subprocess.run(
                    ["git", "log", "--reverse", "--pretty=format:%ad", "--date=iso", slow_processor_file],
                    capture_output=True,
                    text=True,
                    cwd=os.getcwd(),
                )
                if result.returncode != 0 or not result.stdout.strip():
                    return None
                # Parse the first line (earliest commit)
                first_line = result.stdout.strip().split("\n")[0]
                date_part = first_line.split(" ")[0]  # Extract just the date part
                commit_date = datetime.strptime(date_part, "%Y-%m-%d")
                # Check if committed before the cutoff date
                cutoff_date = datetime(*date_cutoff)
                return commit_date <= cutoff_date

            except Exception:
                # If any error occurs, skip the test
                return None

        image_processor_name = self.image_processing_class.__name__
        model_type = None
        for mapping_model_type, (slow_class, _) in IMAGE_PROCESSOR_MAPPING_NAMES.items():
            if slow_class == image_processor_name:
                model_type = mapping_model_type
                break

        if model_type is None:
            self.skipTest(f"Could not find model type for {image_processor_name} in IMAGE_PROCESSOR_MAPPING_NAMES")
        # Check if this is a new model (added after 2024-01-01) based on git history
        is_old_model = _is_old_model_by_commit_date(model_type)
        if is_old_model is None:
            self.skipTest(f"Could not determine if {model_type} is new based on git history")
        # New models must support torchvision backend
        self.assertTrue(
            is_old_model,
            f"Model '{model_type}' (processor: {image_processor_name}) was added after the cutoff date and must support "
            f"the torchvision backend. Please ensure torchvision backend is available.",
        )

    def test_fast_image_processor_explicit_none_preserved(self):
        """Test that explicitly setting an attribute to None is preserved through save/load."""
        # Test with torchvision backend (equivalent to fast processor)
        if "torchvision" not in self.image_processors_backends_list:
            self.skipTest("Skipping test as torchvision backend is not available")

        # Find an attribute with a non-None class default to test explicit None override
        test_attr = None
        for attr in ["do_resize", "do_rescale", "do_normalize"]:
            if getattr(self.image_processing_class, attr, None) is not None:
                test_attr = attr
                break

        if test_attr is None:
            self.skipTest("Could not find a suitable attribute to test")

        # Create processor with explicit None (override the attribute)
        kwargs = self.image_processor_dict.copy()
        kwargs[test_attr] = None
        image_processor = self.image_processing_class(backend="torchvision", **kwargs)

        # Verify it's in to_dict() as None (not filtered out)
        self.assertIn(test_attr, image_processor.to_dict())
        self.assertIsNone(image_processor.to_dict()[test_attr])

        # Verify explicit None survives save/load cycle
        with tempfile.TemporaryDirectory() as tmpdirname:
            image_processor.save_pretrained(tmpdirname)
            reloaded = self.image_processing_class.from_pretrained(tmpdirname, backend="torchvision")

        self.assertIsNone(getattr(reloaded, test_attr), f"Explicit None for {test_attr} was lost after reload")


class AnnotationFormatTestMixin:
    # this mixin adds a test to assert that usages of the
    # to-be-deprecated `AnnotionFormat` continue to be
    # supported for the time being

    def test_processor_can_use_legacy_annotation_format(self):
        image_processor_dict = self.image_processor_tester.prepare_image_processor_dict()
        fixtures_path = pathlib.Path(__file__).parent / "fixtures" / "tests_samples" / "COCO"

        with open(fixtures_path / "coco_annotations.txt") as f:
            detection_target = json.loads(f.read())

        detection_annotations = {"image_id": 39769, "annotations": detection_target}

        detection_params = {
            "images": Image.open(fixtures_path / "000000039769.png"),
            "annotations": detection_annotations,
            "return_tensors": "pt",
        }

        with open(fixtures_path / "coco_panoptic_annotations.txt") as f:
            panoptic_target = json.loads(f.read())

        panoptic_annotations = {"file_name": "000000039769.png", "image_id": 39769, "segments_info": panoptic_target}

        masks_path = pathlib.Path(fixtures_path / "coco_panoptic")

        panoptic_params = {
            "images": Image.open(fixtures_path / "000000039769.png"),
            "annotations": panoptic_annotations,
            "return_tensors": "pt",
            "masks_path": masks_path,
        }

        test_cases = [
            ("coco_detection", detection_params),
            ("coco_panoptic", panoptic_params),
            (AnnotionFormat.COCO_DETECTION, detection_params),
            (AnnotionFormat.COCO_PANOPTIC, panoptic_params),
            (AnnotationFormat.COCO_DETECTION, detection_params),
            (AnnotationFormat.COCO_PANOPTIC, panoptic_params),
        ]

        def _compare(a, b) -> None:
            if isinstance(a, (dict, BatchFeature)):
                self.assertEqual(a.keys(), b.keys())
                for k, v in a.items():
                    _compare(v, b[k])
            elif isinstance(a, list):
                self.assertEqual(len(a), len(b))
                for idx in range(len(a)):
                    _compare(a[idx], b[idx])
            elif isinstance(a, torch.Tensor):
                torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)
            elif isinstance(a, str):
                self.assertEqual(a, b)

        for annotation_format, params in test_cases:
            with self.subTest(annotation_format):
                image_processor_params = {**image_processor_dict, **{"format": annotation_format}}
                image_processor_first = self.image_processing_class(**image_processor_params)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    image_processor_first.save_pretrained(tmpdirname)
                    image_processor_second = self.image_processing_class.from_pretrained(tmpdirname)

                # check the 'format' key exists and that the dicts of the
                # first and second processors are equal
                self.assertIn("format", image_processor_first.to_dict().keys())
                self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

                # perform encoding using both processors and compare
                # the resulting BatchFeatures
                first_encoding = image_processor_first(**params)
                second_encoding = image_processor_second(**params)
                _compare(first_encoding, second_encoding)
