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
import warnings
from copy import deepcopy

import numpy as np
import pytest
from packaging import version

from transformers import AutoVideoProcessor
from transformers.testing_utils import (
    check_json_file_has_correct_format,
    require_torch,
    require_torch_accelerator,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available
from transformers.video_utils import VideoMetadata


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
    fast_video_processing_class = None
    video_processor_list = None
    input_name = "pixel_values_videos"

    def setUp(self):
        video_processor_list = []

        if self.fast_video_processing_class:
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
        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict)
        self.assertEqual(video_processor.size, {"shortest_edge": 20})
        self.assertEqual(video_processor.crop_size, {"height": 18, "width": 18})

        video_processor = self.fast_video_processing_class.from_dict(self.video_processor_dict, size=42, crop_size=84)
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

    @slow
    @require_torch_accelerator
    @require_vision
    @pytest.mark.torch_compile_test
    def test_can_compile_fast_video_processor(self):
        if self.fast_video_processing_class is None:
            self.skipTest("Skipping compilation test as fast video processor is not defined")
        if version.parse(torch.__version__) < version.parse("2.3"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        torch.compiler.reset()
        video_inputs = self.video_processor_tester.prepare_video_inputs(equal_resolution=False, return_tensors="torch")
        video_processor = self.fast_video_processing_class(**self.video_processor_dict)
        output_eager = video_processor(video_inputs, device=torch_device, do_sample_frames=False, return_tensors="pt")

        video_processor = torch.compile(video_processor, mode="reduce-overhead")
        output_compiled = video_processor(
            video_inputs, device=torch_device, do_sample_frames=False, return_tensors="pt"
        )

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

    def test_call_sample_frames(self):
        for video_processing_class in self.video_processor_list:
            video_processing = video_processing_class(**self.video_processor_dict)

            prev_num_frames = self.video_processor_tester.num_frames
            self.video_processor_tester.num_frames = 8
            video_inputs = self.video_processor_tester.prepare_video_inputs(
                equal_resolution=False,
                return_tensors="torch",
            )

            # Force set sampling to False. No sampling is expected even when `num_frames` exists
            video_processing.do_sample_frames = False

            encoded_videos = video_processing(video_inputs[0], return_tensors="pt", num_frames=3)[self.input_name]
            encoded_videos_batched = video_processing(video_inputs, return_tensors="pt", num_frames=3)[self.input_name]
            self.assertEqual(encoded_videos.shape[1], 8)
            self.assertEqual(encoded_videos_batched.shape[1], 8)

            # Set sampling to True. Video frames should be sampled with `num_frames` in the output
            video_processing.do_sample_frames = True

            encoded_videos = video_processing(video_inputs[0], return_tensors="pt", num_frames=3)[self.input_name]
            encoded_videos_batched = video_processing(video_inputs, return_tensors="pt", num_frames=3)[self.input_name]
            self.assertEqual(encoded_videos.shape[1], 3)
            self.assertEqual(encoded_videos_batched.shape[1], 3)

            # Sample with `fps` requires metadata to infer number of frames from total duration
            with self.assertRaises(ValueError):
                metadata = VideoMetadata(**{"total_num_frames": 8})
                video_processing.sample_frames(metadata=metadata, fps=3)

            metadata = [[{"duration": 2.0, "total_num_frames": 8, "fps": 4}]]
            batched_metadata = metadata * len(video_inputs)
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt", fps=3, video_metadata=metadata)[
                self.input_name
            ]
            encoded_videos_batched = video_processing(
                video_inputs, return_tensors="pt", fps=3, video_metadata=batched_metadata
            )[self.input_name]
            self.assertEqual(encoded_videos.shape[1], 6)
            self.assertEqual(encoded_videos_batched.shape[1], 6)

            # The same as above but uses a `VideoMetadata` object in the input
            metadata = [[VideoMetadata(duration=2.0, total_num_frames=8, fps=4)]]
            batched_metadata = metadata * len(video_inputs)
            encoded_videos = video_processing(video_inputs[0], return_tensors="pt", fps=3, video_metadata=metadata)[
                self.input_name
            ]

            # We should raise error when asked to sample more frames than there are in input video
            with self.assertRaises(ValueError):
                encoded_videos = video_processing(video_inputs[0], return_tensors="pt", num_frames=10)[self.input_name]
                encoded_videos_batched = video_processing(video_inputs, return_tensors="pt", num_frames=10)[
                    self.input_name
                ]

            # Assign back the actual num frames in tester
            self.video_processor_tester.num_frames = prev_num_frames

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

    def test_override_instance_attributes_does_not_affect_other_instances(self):
        if self.fast_video_processing_class is None:
            self.skipTest(
                "Only testing fast video processor, as most slow processors break this test and are to be deprecated"
            )

        video_processing_class = self.fast_video_processing_class
        video_processor_1 = video_processing_class()
        video_processor_2 = video_processing_class()
        if not (hasattr(video_processor_1, "size") and isinstance(video_processor_1.size, dict)) or not (
            hasattr(video_processor_1, "image_mean") and isinstance(video_processor_1.image_mean, list)
        ):
            self.skipTest(
                reason="Skipping test as the image processor does not have dict size or list image_mean attributes"
            )

        original_size_2 = deepcopy(video_processor_2.size)
        for key in video_processor_1.size:
            video_processor_1.size[key] = -1
        modified_copied_size_1 = deepcopy(video_processor_1.size)

        original_image_mean_2 = deepcopy(video_processor_2.image_mean)
        video_processor_1.image_mean[0] = -1
        modified_copied_image_mean_1 = deepcopy(video_processor_1.image_mean)

        # check that the original attributes of the second instance are not affected
        self.assertEqual(video_processor_2.size, original_size_2)
        self.assertEqual(video_processor_2.image_mean, original_image_mean_2)

        for key in video_processor_2.size:
            video_processor_2.size[key] = -2
        video_processor_2.image_mean[0] = -2

        # check that the modified attributes of the first instance are not affected by the second instance
        self.assertEqual(video_processor_1.size, modified_copied_size_1)
        self.assertEqual(video_processor_1.image_mean, modified_copied_image_mean_1)
