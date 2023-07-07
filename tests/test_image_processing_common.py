# coding=utf-8
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

import json
import os
import tempfile

from transformers.testing_utils import check_json_file_has_correct_format, require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available


if is_torch_available():
    import numpy as np
    import torch

if is_vision_available():
    from PIL import Image


def prepare_image_inputs(image_processor_tester, equal_resolution=False, numpify=False, torchify=False):
    """This function prepares a list of PIL images, or a list of numpy arrays if one specifies numpify=True,
    or a list of PyTorch tensors if one specifies torchify=True.

    One can specify whether the images are of the same resolution or not.
    """

    assert not (numpify and torchify), "You cannot specify both numpy and PyTorch tensors at the same time"

    image_inputs = []
    for i in range(image_processor_tester.batch_size):
        if equal_resolution:
            width = height = image_processor_tester.max_resolution
        else:
            # To avoid getting image width/height 0
            min_resolution = image_processor_tester.min_resolution
            if getattr(image_processor_tester, "size_divisor", None):
                # If `size_divisor` is defined, the image needs to have width/size >= `size_divisor`
                min_resolution = max(image_processor_tester.size_divisor, min_resolution)
            width, height = np.random.choice(np.arange(min_resolution, image_processor_tester.max_resolution), 2)
        image_inputs.append(
            np.random.randint(255, size=(image_processor_tester.num_channels, width, height), dtype=np.uint8)
        )

    if not numpify and not torchify:
        # PIL expects the channel dimension as last dimension
        image_inputs = [Image.fromarray(np.moveaxis(image, 0, -1)) for image in image_inputs]

    if torchify:
        image_inputs = [torch.from_numpy(image) for image in image_inputs]

    return image_inputs


def prepare_video(image_processor_tester, width=10, height=10, numpify=False, torchify=False):
    """This function prepares a video as a list of PIL images/NumPy arrays/PyTorch tensors."""

    video = []
    for i in range(image_processor_tester.num_frames):
        video.append(np.random.randint(255, size=(image_processor_tester.num_channels, width, height), dtype=np.uint8))

    if not numpify and not torchify:
        # PIL expects the channel dimension as last dimension
        video = [Image.fromarray(np.moveaxis(frame, 0, -1)) for frame in video]

    if torchify:
        video = [torch.from_numpy(frame) for frame in video]

    return video


def prepare_video_inputs(image_processor_tester, equal_resolution=False, numpify=False, torchify=False):
    """This function prepares a batch of videos: a list of list of PIL images, or a list of list of numpy arrays if
    one specifies numpify=True, or a list of list of PyTorch tensors if one specifies torchify=True.

    One can specify whether the videos are of the same resolution or not.
    """

    assert not (numpify and torchify), "You cannot specify both numpy and PyTorch tensors at the same time"

    video_inputs = []
    for i in range(image_processor_tester.batch_size):
        if equal_resolution:
            width = height = image_processor_tester.max_resolution
        else:
            width, height = np.random.choice(
                np.arange(image_processor_tester.min_resolution, image_processor_tester.max_resolution), 2
            )
            video = prepare_video(
                image_processor_tester=image_processor_tester,
                width=width,
                height=height,
                numpify=numpify,
                torchify=torchify,
            )
        video_inputs.append(video)

    return video_inputs


class ImageProcessingSavingTestMixin:
    test_cast_dtype = None

    def test_image_processor_to_json_string(self):
        image_processor = self.image_processing_class(**self.image_processor_dict)
        obj = json.loads(image_processor.to_json_string())
        for key, value in self.image_processor_dict.items():
            self.assertEqual(obj[key], value)

    def test_image_processor_to_json_file(self):
        image_processor_first = self.image_processing_class(**self.image_processor_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "image_processor.json")
            image_processor_first.to_json_file(json_file_path)
            image_processor_second = self.image_processing_class.from_json_file(json_file_path)

        self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

    def test_image_processor_from_and_save_pretrained(self):
        image_processor_first = self.image_processing_class(**self.image_processor_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            saved_file = image_processor_first.save_pretrained(tmpdirname)[0]
            check_json_file_has_correct_format(saved_file)
            image_processor_second = self.image_processing_class.from_pretrained(tmpdirname)

        self.assertEqual(image_processor_second.to_dict(), image_processor_first.to_dict())

    def test_init_without_params(self):
        image_processor = self.image_processing_class()
        self.assertIsNotNone(image_processor)

    @require_torch
    @require_vision
    def test_cast_dtype_device(self):
        if self.test_cast_dtype is not None:
            # Initialize image_processor
            image_processor = self.image_processing_class(**self.image_processor_dict)

            # create random PyTorch tensors
            image_inputs = prepare_image_inputs(self.image_processor_tester, equal_resolution=False, torchify=True)

            encoding = image_processor(image_inputs, return_tensors="pt")
            # for layoutLM compatiblity
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
