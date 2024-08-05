# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from transformers import (
    MODEL_FOR_IMAGE_TO_IMAGE_MAPPING,
    AutoImageProcessor,
    AutoModelForImageToImage,
    ImageToImagePipeline,
    is_vision_available,
    pipeline,
)
from transformers.testing_utils import (
    is_pipeline_test,
    require_torch,
    require_vision,
    slow,
)


if is_vision_available():
    from PIL import Image

else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


@is_pipeline_test
@require_torch
@require_vision
class ImageToImagePipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_IMAGE_TO_IMAGE_MAPPING
    examples = [
        Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
        "http://images.cocodataset.org/val2017/000000039769.jpg",
    ]

    @require_torch
    @require_vision
    @slow
    def test_pipeline(self, torch_dtype="float32"):
        model_id = "caidas/swin2SR-classical-sr-x2-64"
        upscaler = pipeline("image-to-image", model=model_id, torch_dtype=torch_dtype)
        upscaled_list = upscaler(self.examples)

        self.assertEqual(len(upscaled_list), len(self.examples))
        for output in upscaled_list:
            self.assertIsInstance(output, Image.Image)

        self.assertEqual(upscaled_list[0].size, (1296, 976))
        self.assertEqual(upscaled_list[1].size, (1296, 976))

    @require_torch
    @require_vision
    @slow
    def test_pipeline_fp16(self):
        self.test_pipeline(torch_dtype="float16")

    @require_torch
    @require_vision
    @slow
    def test_pipeline_model_processor(self):
        model_id = "caidas/swin2SR-classical-sr-x2-64"
        model = AutoModelForImageToImage.from_pretrained(model_id)
        image_processor = AutoImageProcessor.from_pretrained(model_id)

        upscaler = ImageToImagePipeline(model=model, image_processor=image_processor)
        upscaled_list = upscaler(self.examples)

        self.assertEqual(len(upscaled_list), len(self.examples))
        for output in upscaled_list:
            self.assertIsInstance(output, Image.Image)

        self.assertEqual(upscaled_list[0].size, (1296, 976))
        self.assertEqual(upscaled_list[1].size, (1296, 976))
