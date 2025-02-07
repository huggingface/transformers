# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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
import json
import os
import tempfile
import time
import warnings

import numpy as np
from packaging import version

from transformers import AutoVideoProcessor
from transformers.testing_utils import (
    check_json_file_has_correct_format,
    require_torch,
    require_torch_gpu,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


def prepare_video(num_frames, num_channels, width=10, height=10, return_tensors="pil"):
    """This function prepares a video as a list of PIL images/NumPy arrays/PyTorch tensors."""

    video = []
    for i in range(num_frames):
        video.append(np.random.randint(255, size=(width, height, num_channels), dtype=np.uint8))

    if return_tensors == "pil":
        # PIL expects the channel dimension as last dimension
        video = [Image.fromarray(frame) for frame in video]
    elif return_tensors == "torch":
        # Torch images are typically in channels first format
        video = torch.tensor(video).permute(0, 3, 1, 2)
    elif return_tensors == "np":
        # Numpy images are typically in channels last format
        video = np.array(video)

    return video


def prepare_video_inputs(
    batch_size,
    num_frames,
    num_channels,
    min_resolution,
    max_resolution,
    equal_resolution=False,
    return_tensors="pil",
):
    """This function prepares a batch of videos: a list of list of PIL images, or a list of list of numpy arrays if
    one specifies return_tensors="np", or a list of list of PyTorch tensors if one specifies return_tensors="torch".

    One can specify whether the videos are of the same resolution or not.
    """

    video_inputs = []
    for i in range(batch_size):
        if equal_resolution:
            width = height = max_resolution
        else:
            width, height = np.random.choice(np.arange(min_resolution, max_resolution), 2)
        video = prepare_video(
            num_frames=num_frames,
            num_channels=num_channels,
            width=width,
            height=height,
            return_tensors=return_tensors,
        )
        video_inputs.append(video)

    return video_inputs


class VideoProcessingTestMixin:
    test_cast_dtype = None
    video_processing_class = None
    fast_video_processing_class = None
    video_processor_list = None
    test_slow_video_processor = True
    test_fast_video_processor = True
    input_name = "pixel_values_videos"

    def setUp(self):
        video_processor_list = []

        if self.test_slow_video_processor and self.video_processing_class:
            video_processor_list.append(self.video_processing_class)

        if self.test_fast_video_processor and self.fast_video_processing_class:
            video_processor_list.append(self.fast_video_processing_class)

        self.video_processor_list = video_processor_list

    def test_video_processor_to_json_string(self):
        for video_processing_class in self.video_processor_list:
            video_processor = video_processing_class(**self.video_processor_dict)
            obj = json.loads(video_processor.to_json_string())
            for key, value in self.video_processor_dict.items():
                self.assertEqual(obj[key], value)

    def test_video_processor_to_json_file(self):
        for video_processing_class in self.video_processor_list:
            video_processor_first = video_processing_class(**self.video_processor_dict)

            with tempfile.TemporaryDirectory() as tmpdirname:
                json_file_path = os.path.join(tmpdirname, "video_processor.json")
                video_processor_first.to_json_file(json_file_path)
                video_processor_second = video_processing_class.from_json_file(json_file_path)

            self.assertEqual(video_processor_second.to_dict(), video_processor_first.to_dict())

    def test_video_processor_from_dict_with_kwargs(self):
        video_processor = self.video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.size, {"shortest_edge": 20})
        self.assertEqual(video_processor.crop_size, {"height": 18, "width": 18})

        video_processor = self.video_processing_class.from_dict(self.video_processor_dict, size=42, crop_size=84)
        self.assertEqual(video_processor.size, {"shortest_edge": 42})
        self.assertEqual(video_processor.crop_size, {"height": 84, "width": 84})

    def test_video_processor_from_and_save_pretrained(self):
        for video_processing_class in self.video_processor_list:
            video_processor_first = video_processing_class(**self.video_processor_dict)

            with tempfile.TemporaryDirectory() as tmpdirname:
                saved_file = video_processor_first.save_pretrained(tmpdirname)[0]
                check_json_file_has_correct_format(saved_file)
                video_processor_second = video_processing_class.from_pretrained(tmpdirname)

            self.assertEqual(video_processor_second.to_dict(), video_processor_first.to_dict())

    def test_video_processor_save_load_with_autovideoprocessor(self):
        for video_processing_class in self.video_processor_list:
            video_processor_first = video_processing_class(**self.video_processor_dict)

            with tempfile.TemporaryDirectory() as tmpdirname:
                saved_file = video_processor_first.save_pretrained(tmpdirname)[0]
                check_json_file_has_correct_format(saved_file)

                use_fast = video_processing_class.__name__.endswith("Fast")
                video_processor_second = AutoVideoProcessor.from_pretrained(tmpdirname, use_fast=use_fast)

            self.assertEqual(video_processor_second.to_dict(), video_processor_first.to_dict())

    def test_init_without_params(self):
        for video_processing_class in self.video_processor_list:
            video_processor = video_processing_class()
            self.assertIsNotNone(video_processor)

    @require_vision
    @require_torch
    def test_slow_fast_equivalence(self):
        if not self.test_slow_video_processor or not self.test_fast_video_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.video_processing_class is None or self.fast_video_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the video processors is not defined")

        video_inputs = self.video_processor_tester.prepare_video_inputs(equal_resolution=False, return_tensors="torch")
        video_processor_slow = self.video_processing_class(**self.video_processor_dict)
        video_processor_fast = self.fast_video_processing_class(**self.video_processor_dict)

        encoding_slow = video_processor_slow(video_inputs, return_tensors="pt")
        encoding_fast = video_processor_fast(video_inputs, return_tensors="pt")
        self.assertTrue(torch.allclose(encoding_slow[self.input_name], encoding_fast[self.input_name], atol=1e-1))
        self.assertLessEqual(
            torch.mean(torch.abs(encoding_slow[self.input_name] - encoding_fast[self.input_name])).item(), 1e-3
        )

    @require_vision
    @require_torch
    def test_slow_fast_equivalence_batched(self):
        if not self.test_slow_video_processor or not self.test_fast_video_processor:
            self.skipTest(reason="Skipping slow/fast equivalence test")

        if self.video_processing_class is None or self.fast_video_processing_class is None:
            self.skipTest(reason="Skipping slow/fast equivalence test as one of the video processors is not defined")

        if hasattr(self.video_processor_tester, "do_center_crop") and self.video_processor_tester.do_center_crop:
            self.skipTest(
                reason="Skipping as do_center_crop is True and center_crop functions are not equivalent for fast and slow processors"
            )

        video_inputs = self.video_processor_tester.prepare_video_inputs(equal_resolution=False, return_tensors="torch")
        video_processor_slow = self.video_processing_class(**self.video_processor_dict)
        video_processor_fast = self.fast_video_processing_class(**self.video_processor_dict)

        encoding_slow = video_processor_slow(video_inputs, return_tensors="pt")
        encoding_fast = video_processor_fast(video_inputs, return_tensors="pt")

        self.assertTrue(torch.allclose(encoding_slow[self.input_name], encoding_fast[self.input_name], atol=1e-1))
        self.assertLessEqual(
            torch.mean(torch.abs(encoding_slow[self.input_name] - encoding_fast[self.input_name])).item(), 1e-3
        )

    @require_vision
    @require_torch
    def test_fast_is_faster_than_slow(self):
        if not self.test_slow_video_processor or not self.test_fast_video_processor:
            self.skipTest(reason="Skipping speed test")

        if self.video_processing_class is None or self.fast_video_processing_class is None:
            self.skipTest(reason="Skipping speed test as one of the video processors is not defined")

        def measure_time(video_processor, video):
            # Warmup
            for _ in range(5):
                _ = video_processor(video, return_tensors="pt")
            start = time.perf_counter()
            _ = video_processor(video, return_tensors="pt")
            return time.perf_counter() - start

        video_processor_slow = self.video_processing_class(**self.video_processor_dict)
        video_processor_fast = self.fast_video_processing_class(**self.video_processor_dict)

        # Inputs in torch.Tensor
        video_inputs_torch = self.video_processor_tester.prepare_video_inputs(
            equal_resolution=False, return_tensors="torch"
        )
        fast_time_torch = measure_time(video_processor_fast, video_inputs_torch)
        slow_time_torch = measure_time(video_processor_slow, video_inputs_torch)

        self.assertLessEqual(fast_time_torch, slow_time_torch)

        # Inputs in np.ndarray
        self.video_processor_tester.batch_size = 7
        video_inputs_np = self.video_processor_tester.prepare_video_inputs(equal_resolution=True, return_tensors="np")
        fast_time_np = measure_time(video_processor_fast, video_inputs_np)
        slow_time_np = measure_time(video_processor_slow, video_inputs_np)

        self.assertLessEqual(fast_time_np, slow_time_np)

        # Inputs in PIL.Image
        video_inputs_pil = self.video_processor_tester.prepare_video_inputs(
            equal_resolution=False, return_tensors="pil"
        )
        fast_time_pil = measure_time(video_processor_fast, video_inputs_pil)
        slow_time_pil = measure_time(video_processor_slow, video_inputs_pil)

        self.assertLessEqual(fast_time_pil, slow_time_pil)

    def test_save_load_fast_slow(self):
        "Test that we can load a fast video processor from a slow one and vice-versa."
        if self.video_processing_class is None or self.fast_video_processing_class is None:
            self.skipTest("Skipping slow/fast save/load test as one of the video processors is not defined")

        video_processor_dict = self.video_processor_tester.prepare_video_processor_dict()
        video_processor_slow_0 = self.video_processing_class(**video_processor_dict)

        # Load fast video processor from slow one
        with tempfile.TemporaryDirectory() as tmpdirname:
            video_processor_slow_0.save_pretrained(tmpdirname)
            video_processor_fast_0 = self.fast_video_processing_class.from_pretrained(tmpdirname)

        video_processor_fast_1 = self.fast_video_processing_class(**video_processor_dict)

        # Load slow video processor from fast one
        with tempfile.TemporaryDirectory() as tmpdirname:
            video_processor_fast_1.save_pretrained(tmpdirname)
            video_processor_slow_1 = self.video_processing_class.from_pretrained(tmpdirname)

        dict_slow_0 = video_processor_slow_0.to_dict()
        dict_slow_1 = video_processor_slow_1.to_dict()
        difference = {
            key: dict_slow_0.get(key) if key in dict_slow_0 else dict_slow_1.get(key)
            for key in set(dict_slow_0) ^ set(dict_slow_1)
        }
        dict_slow_0 = {key: dict_slow_0[key] for key in set(dict_slow_0) & set(dict_slow_1)}
        dict_slow_1 = {key: dict_slow_1[key] for key in set(dict_slow_0) & set(dict_slow_1)}
        # check that all additional keys are None, except for `default_to_square` which is only set in fast processors
        self.assertTrue(all(value is None for key, value in difference.items() if key not in ["default_to_square"]))
        # check that the remaining keys are the same
        self.assertEqual(dict_slow_1, dict_slow_0)

        dict_fast_0 = video_processor_fast_0.to_dict()
        dict_fast_1 = video_processor_fast_1.to_dict()
        difference = {
            key: dict_fast_0.get(key) if key in dict_fast_0 else dict_fast_1.get(key)
            for key in set(dict_fast_0) ^ set(dict_fast_1)
        }
        dict_fast_0 = {key: dict_fast_0[key] for key in set(dict_fast_0) & set(dict_fast_1)}
        dict_fast_1 = {key: dict_fast_1[key] for key in set(dict_fast_0) & set(dict_fast_1)}
        # check that all additional keys are None, except for `default_to_square` which is only set in fast processors
        self.assertTrue(all(value is None for key, value in difference.items() if key not in ["default_to_square"]))
        # check that the remaining keys are the same
        self.assertEqual(dict_fast_0, dict_fast_1)

    def test_save_load_fast_slow_auto(self):
        "Test that we can load a fast video processor from a slow one and vice-versa using AutoVideoProcessor."
        if self.video_processing_class is None or self.fast_video_processing_class is None:
            self.skipTest("Skipping slow/fast save/load test as one of the video processors is not defined")

        video_processor_dict = self.video_processor_tester.prepare_video_processor_dict()
        video_processor_slow_0 = self.video_processing_class(**video_processor_dict)

        # Load fast video processor from slow one
        with tempfile.TemporaryDirectory() as tmpdirname:
            video_processor_slow_0.save_pretrained(tmpdirname)
            video_processor_fast_0 = AutoVideoProcessor.from_pretrained(tmpdirname, use_fast=True)

        video_processor_fast_1 = self.fast_video_processing_class(**video_processor_dict)

        # Load slow video processor from fast one
        with tempfile.TemporaryDirectory() as tmpdirname:
            video_processor_fast_1.save_pretrained(tmpdirname)
            video_processor_slow_1 = AutoVideoProcessor.from_pretrained(tmpdirname, use_fast=False)

        dict_slow_0 = video_processor_slow_0.to_dict()
        dict_slow_1 = video_processor_slow_1.to_dict()
        difference = {
            key: dict_slow_0.get(key) if key in dict_slow_0 else dict_slow_1.get(key)
            for key in set(dict_slow_0) ^ set(dict_slow_1)
        }
        dict_slow_0 = {key: dict_slow_0[key] for key in set(dict_slow_0) & set(dict_slow_1)}
        dict_slow_1 = {key: dict_slow_1[key] for key in set(dict_slow_0) & set(dict_slow_1)}
        # check that all additional keys are None, except for `default_to_square` which is only set in fast processors
        self.assertTrue(all(value is None for key, value in difference.items() if key not in ["default_to_square"]))
        # check that the remaining keys are the same
        self.assertEqual(dict_slow_0, dict_slow_1)

        dict_fast_0 = video_processor_fast_0.to_dict()
        dict_fast_1 = video_processor_fast_1.to_dict()
        difference = {
            key: dict_fast_0.get(key) if key in dict_fast_0 else dict_fast_1.get(key)
            for key in set(dict_fast_0) ^ set(dict_fast_1)
        }
        dict_fast_0 = {key: dict_fast_0[key] for key in set(dict_fast_0) & set(dict_fast_1)}
        dict_fast_1 = {key: dict_fast_1[key] for key in set(dict_fast_0) & set(dict_fast_1)}
        # check that all additional keys are None, except for `default_to_square` which is only set in fast processors
        self.assertTrue(all(value is None for key, value in difference.items() if key not in ["default_to_square"]))
        # check that the remaining keys are the same
        self.assertEqual(dict_fast_0, dict_fast_1)

    @slow
    @require_torch_gpu
    @require_vision
    def test_can_compile_fast_video_processor(self):
        if self.fast_video_processing_class is None:
            self.skipTest("Skipping compilation test as fast video processor is not defined")
        if version.parse(torch.__version__) < version.parse("2.3"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        torch.compiler.reset()
        video_inputs = self.video_processor_tester.prepare_video_inputs(equal_resolution=False, return_tensors="torch")
        video_processor = self.fast_video_processing_class(**self.video_processor_dict)
        output_eager = video_processor(video_inputs, device=torch_device, return_tensors="pt")

        video_processor = torch.compile(video_processor, mode="reduce-overhead")
        output_compiled = video_processor(video_inputs, device=torch_device, return_tensors="pt")

        torch.testing.assert_close(
            output_eager[self.input_name], output_compiled[self.input_name], rtol=1e-4, atol=1e-4
        )

    @require_torch
    @require_vision
    def test_cast_dtype_device(self):
        for video_processing_class in self.video_processor_list:
            if self.test_cast_dtype is not None:
                # Initialize video_processor
                video_processor = video_processing_class(**self.video_processor_dict)

                # create random PyTorch tensors
                video_inputs = self.video_processor_tester.prepare_video_inputs(
                    equal_resolution=False, return_tensors="torch"
                )

                encoding = video_processor(video_inputs, return_tensors="pt")

                self.assertEqual(encoding[self.input_name].device, torch.device("cpu"))
                self.assertEqual(encoding[self.input_name].dtype, torch.float32)

                encoding = video_processor(video_inputs, return_tensors="pt").to(torch.float16)
                self.assertEqual(encoding[self.input_name].device, torch.device("cpu"))
                self.assertEqual(encoding[self.input_name].dtype, torch.float16)

                encoding = video_processor(video_inputs, return_tensors="pt").to("cpu", torch.bfloat16)
                self.assertEqual(encoding[self.input_name].device, torch.device("cpu"))
                self.assertEqual(encoding[self.input_name].dtype, torch.bfloat16)

                with self.assertRaises(TypeError):
                    _ = video_processor(video_inputs, return_tensors="pt").to(torch.bfloat16, "cpu")

                # Try with text + video feature
                encoding = video_processor(video_inputs, return_tensors="pt")
                encoding.update({"input_ids": torch.LongTensor([[1, 2, 3], [4, 5, 6]])})
                encoding = encoding.to(torch.float16)

                self.assertEqual(encoding[self.input_name].device, torch.device("cpu"))
                self.assertEqual(encoding[self.input_name].dtype, torch.float16)
                self.assertEqual(encoding.input_ids.dtype, torch.long)

    def test_call_pil(self):
        for video_processing_class in self.video_processor_list:
            # Initialize video_processing
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(equal_resolution=False)

            # Each video is a list of PIL Images
            for video in video_inputs:
                self.assertIsInstance(video[0], Image.Image)

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(tuple(encoded_videos.shape), (1, *expected_output_video_shape))

            # Test batched
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(
                tuple(encoded_videos.shape), (self.video_processor_tester.batch_size, *expected_output_video_shape)
            )

    def test_call_numpy(self):
        for video_processing_class in self.video_processor_list:
            # Initialize video_processing
            video_processing = video_processing_class(**self.video_processor_dict)
            # create random numpy tensors
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="np"
            )
            for video in video_inputs:
                self.assertIsInstance(video, np.ndarray)

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(tuple(encoded_videos.shape), (1, *expected_output_video_shape))

            # Test batched
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            self.assertEqual(
                tuple(encoded_videos.shape), (self.video_processor_tester.batch_size, *expected_output_video_shape)
            )

    def test_call_pytorch(self):
        for video_processing_class in self.video_processor_list:
            # Initialize video_processing
            video_processing = video_processing_class(**self.video_processor_dict)
            # create random PyTorch tensors
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="torch"
            )

            for video in video_inputs:
                self.assertIsInstance(video, torch.Tensor)

            # Test not batched input
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(tuple(encoded_videos.shape), (1, *expected_output_video_shape))

            # Test batched
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            self.assertEqual(
                tuple(encoded_videos.shape),
                (self.video_processor_tester.batch_size, *expected_output_video_shape),
            )

    def test_nested_input(self):
        """Tests that the processor can work with nested list where each video is a list of arrays"""
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="np"
            )

            # Test not batched input
            video_inputs = [list(video) for video in video_inputs]
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt")[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            self.assertEqual(tuple(encoded_videos.shape), (1, *expected_output_video_shape))

            # Test batched
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            encoded_videos = video_processing(video_inputs, return_tensors="pt")[self.input_name]
            self.assertEqual(
                tuple(encoded_videos.shape),
                (self.video_processor_tester.batch_size, *expected_output_video_shape),
            )

    def test_call_numpy_4_channels(self):
        for video_processing_class in self.video_processor_list:
            # Test that can process videos which have an arbitrary number of channels
            # Initialize video_processing
            video_processor = video_processing_class(**self.video_processor_dict)

            # create random numpy tensors
            self.video_processor_tester.num_channels = 4
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False, return_tensors="pil"
            )

            # Test not batched input
            encoded_videos = video_processor(
                video_inputs[0],
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=0,
                image_std=1,
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape([video_inputs[0]])
            if video_processor.do_convert_rgb:
                expected_output_video_shape = list(expected_output_video_shape)
                expected_output_video_shape[1] = 3
            self.assertEqual(tuple(encoded_videos.shape), (1, *expected_output_video_shape))

            # Test batched
            encoded_videos = video_processor(
                video_inputs,
                return_tensors="pt",
                input_data_format="channels_last",
                image_mean=0,
                image_std=1,
            )[self.input_name]
            expected_output_video_shape = self.video_processor_tester.expected_output_video_shape(video_inputs)
            if video_processor.do_convert_rgb:
                expected_output_video_shape = list(expected_output_video_shape)
                expected_output_video_shape[1] = 3
            self.assertEqual(
                tuple(encoded_videos.shape), (self.video_processor_tester.batch_size, *expected_output_video_shape)
            )

    def test_video_processor_preprocess_arguments(self):
        is_tested = False

        for video_processing_class in self.video_processor_list:
            video_processor = video_processing_class(**self.video_processor_dict)

            # validation done by _valid_processor_keys attribute
            if hasattr(video_processor, "_valid_processor_keys") and hasattr(video_processor, "preprocess"):
                preprocess_parameter_names = inspect.getfullargspec(video_processor.preprocess).args
                preprocess_parameter_names.remove("self")
                preprocess_parameter_names.sort()
                valid_processor_keys = video_processor._valid_processor_keys
                valid_processor_keys.sort()
                self.assertEqual(preprocess_parameter_names, valid_processor_keys)
                is_tested = True

            # validation done by @filter_out_non_signature_kwargs decorator
            if hasattr(video_processor.preprocess, "_filter_out_non_signature_kwargs"):
                if hasattr(self.video_processor_tester, "prepare_video_inputs"):
                    inputs = self.video_processor_tester.prepare_video_inputs()
                elif hasattr(self.video_processor_tester, "prepare_video_inputs"):
                    inputs = self.video_processor_tester.prepare_video_inputs()
                else:
                    self.skipTest(reason="No valid input preparation method found")

                with warnings.catch_warnings(record=True) as raised_warnings:
                    warnings.simplefilter("always")
                    video_processor(inputs, extra_argument=True)

                messages = " ".join([str(w.message) for w in raised_warnings])
                self.assertGreaterEqual(len(raised_warnings), 1)
                self.assertIn("extra_argument", messages)
                is_tested = True

        if not is_tested:
            self.skipTest(reason="No validation found for `preprocess` method")
