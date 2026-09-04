# Copyright 2026 Cohere Inc. and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

from transformers import (
    CohereCompassImageProcessor,
    CohereCompassProcessor,
    CohereCompassVideoProcessor,
    PreTrainedTokenizerFast,
)
from transformers.testing_utils import require_torch, require_vision

from ...test_processing_common import ProcessorTesterMixin


@require_torch
@require_vision
class CohereCompassProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = CohereCompassProcessor
    video_unstructured_max_length = 870
    video_text_kwargs_max_length = 870
    video_text_kwargs_override_max_length = 870

    @classmethod
    def _setup_tokenizer(cls):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace

        tokenizer = Tokenizer(
            WordLevel(
                {
                    "<unk>": 0,
                    "<bos>": 1,
                    "<eos>": 2,
                    "<pad>": 3,
                    "<|IMAGE_PAD|>": 4,
                    "<|VISION_START|>": 5,
                    "<|VISION_END|>": 6,
                    "<|VIDEO_PAD|>": 7,
                    "describe": 8,
                    "this": 9,
                    "image": 10,
                },
                unk_token="<unk>",
            )
        )
        tokenizer.pre_tokenizer = Whitespace()
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
            additional_special_tokens=[
                "<|IMAGE_PAD|>",
                "<|VIDEO_PAD|>",
                "<|VISION_START|>",
                "<|VISION_END|>",
            ],
        )

    @classmethod
    def _setup_image_processor(cls):
        return CohereCompassImageProcessor(
            min_pixels=56 * 56,
            max_pixels=56 * 56,
            patch_size=16,
        )

    @classmethod
    def _setup_video_processor(cls):
        return CohereCompassVideoProcessor(patch_size=16)

    def _image(self, height=56, width=56):
        from PIL import Image

        return Image.fromarray(np.full((height, width, 3), 127, dtype=np.uint8))

    def prepare_image_inputs(self, batch_size=None, nested=False):
        if batch_size is None:
            return self._image(64, 64)
        images = [self._image(64, 64) for _ in range(batch_size)]
        return [[image] for image in images] if nested else images

    def prepare_video_inputs(self, batch_size=None):
        video = np.random.randint(255, size=(8, 3, 64, 64), dtype=np.uint8)
        return video if batch_size is None else [video] * batch_size

    def test_image_placeholder_expansion(self):
        processor = self.get_processor()
        output = processor(
            images=self._image(),
            text="<|VISION_START|><|IMAGE_PAD|><|VISION_END|> describe this image",
            return_tensors="pt",
        )
        self.assertEqual(output.image_grid_thw.tolist(), [[1, 2, 2]])
        self.assertEqual((output.input_ids == processor.image_token_id).sum().item(), 1)
        self.assertTrue(output.mm_token_type_ids.equal((output.input_ids == processor.image_token_id).int()))

    def test_multiple_images_preserve_grid_order(self):
        processor = self.get_processor()
        output = processor(
            images=[self._image(56, 56), self._image(56, 112)],
            text=(
                "<|VISION_START|><|IMAGE_PAD|><|VISION_END|> "
                "<|VISION_START|><|IMAGE_PAD|><|VISION_END|> describe this image"
            ),
            return_tensors="pt",
        )
        self.assertEqual(output.image_grid_thw.tolist(), [[1, 2, 2], [1, 2, 4]])
        self.assertEqual((output.input_ids == processor.image_token_id).sum().item(), 3)

    def test_get_num_multimodal_tokens(self):
        processor = self.get_processor()
        output = processor._get_num_multimodal_tokens(image_sizes=[(56, 56), (56, 112)])
        self.assertEqual(output["num_image_patches"], [4, 8])
        self.assertEqual(output["num_image_tokens"], [1, 2])

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        processor = self.get_processor()
        image_sizes = [(64, 64), (64, 128), (128, 64)]
        images = [np.random.randint(255, size=(*size, 3), dtype=np.uint8) for size in image_sizes]
        output = processor(
            text=[processor.image_token] * len(images),
            images=images,
            padding=True,
            return_tensors="pt",
        )
        expected = processor._get_num_multimodal_tokens(image_sizes=image_sizes)["num_image_tokens"]
        actual = (output.input_ids == processor.image_token_id).sum(dim=1).tolist()
        self.assertEqual(actual, expected)
