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
from parameterized import parameterized

from transformers import AutoProcessor, MiniMaxVL01ImageProcessor, MiniMaxVL01Processor, PreTrainedTokenizerFast
from transformers.image_utils import PILImageResampling
from transformers.testing_utils import (
    require_tokenizers,
    require_torch,
    require_torchvision,
    require_vision,
    slow,
)
from transformers.utils import is_tokenizers_available

from ...test_processing_common import ProcessorTesterMixin


if is_tokenizers_available():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace


CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['role'] | upper + ':' }}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' %}<image>"
    "{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}"
    "{% endfor %}{% endfor %}"
    "{% if add_generation_prompt %}ASSISTANT:{% endif %}"
)


@require_vision
@require_torch
@require_torchvision
@require_tokenizers
class MiniMaxVL01ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MiniMaxVL01Processor

    @classmethod
    def _setup_tokenizer(cls):
        vocab = {
            "<unk>": 0,
            "<pad>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<image>": 4,
            "lower": 5,
            "newer": 6,
            "upper": 7,
            "older": 8,
            "longer": 9,
            "string": 10,
            "USER": 11,
            "ASSISTANT": 12,
            ":": 13,
            "Describe": 14,
            "this": 15,
            ".": 16,
            "Question": 17,
        }
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            extra_special_tokens={"image_token": "<image>"},
        )

    @classmethod
    def _setup_image_processor(cls):
        return MiniMaxVL01ImageProcessor(
            size={"height": 32, "width": 32},
            patch_size=16,
            image_grid_pinpoints=[[32, 32], [64, 32], [32, 64]],
            resample=PILImageResampling.NEAREST,
            do_center_crop=False,
        )

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token

    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": CHAT_TEMPLATE,
            "patch_size": 16,
            "vision_feature_select_strategy": "default",
            "num_additional_image_tokens": 1,
        }

    @staticmethod
    def _images():
        square = np.zeros((32, 32, 3), dtype=np.uint8)
        portrait = np.zeros((64, 32, 3), dtype=np.uint8)
        landscape = np.zeros((32, 64, 3), dtype=np.uint8)
        return square, portrait, landscape

    def test_exact_image_token_expansion_for_each_grid_shape(self):
        processor = self.get_processor()
        expected_counts = [10, 16, 14]

        for image, expected_count in zip(self._images(), expected_counts):
            with self.subTest(image_shape=image.shape):
                inputs = processor(text=["Question <image>"], images=[image], return_tensors="pt")
                observed_count = (inputs.input_ids == processor.image_token_id).sum().item()
                self.assertEqual(observed_count, expected_count)

    def test_multiple_images_expand_in_placeholder_order(self):
        processor = self.get_processor()
        images = self._images()
        inputs = processor(
            text=["<image> Question <image> Question <image>"],
            images=list(images),
            return_tensors="pt",
        )

        self.assertEqual((inputs.input_ids == processor.image_token_id).sum().item(), 10 + 16 + 14)
        self.assertEqual(tuple(inputs.pixel_values.shape), (8, 3, 32, 32))
        self.assertEqual(inputs.image_sizes.tolist(), [[32, 32], [64, 32], [32, 64]])

    def test_num_multimodal_tokens_matches_model_packing(self):
        processor = self.get_processor()
        output = processor._get_num_multimodal_tokens(image_sizes=[(32, 32), (64, 32), (32, 64)])

        self.assertEqual(output["num_image_tokens"], [10, 16, 14])
        self.assertEqual(output["num_image_patches"], [1, 1, 1])

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        processor = self.get_processor()
        image_sizes = [(32, 32), (64, 32), (32, 64)]
        images = list(self._images())
        inputs = processor(
            text=[f"Question {processor.image_token}"] * len(images),
            images=images,
            padding=True,
            return_mm_token_type_ids=True,
            return_tensors="pt",
        )

        observed_counts = inputs.mm_token_type_ids.sum(-1).tolist()
        helper_counts = processor._get_num_multimodal_tokens(image_sizes=image_sizes)["num_image_tokens"]
        self.assertEqual(observed_counts, helper_counts)

    def test_chat_template_render_then_exact_expansion(self):
        processor = self.get_processor()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this."},
                ],
            }
        ]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertEqual(rendered, "USER:<image>Describe this.ASSISTANT:")

        inputs = processor(text=[rendered], images=[self._images()[0]], return_tensors="pt")
        self.assertEqual((inputs.input_ids == processor.image_token_id).sum().item(), 10)

    @parameterized.expand([(1, "pt"), (2, "pt")])
    @unittest.skip("MiniMax-VL-01 packs variable numbers of image tiles across the batch")
    def test_apply_chat_template_image(self, batch_size: int, return_tensors: str):
        pass

    def test_save_reload_with_auto_processor_preserves_exact_outputs(self):
        processor = self.get_processor()
        image = self._images()[2]
        expected = processor(text=["Question <image>"], images=[image], return_tensors="pt")

        with tempfile.TemporaryDirectory() as tmpdirname:
            processor.save_pretrained(tmpdirname)
            reloaded = AutoProcessor.from_pretrained(tmpdirname, trust_remote_code=False)
            observed = reloaded(text=["Question <image>"], images=[image], return_tensors="pt")

        self.assertIsInstance(reloaded, MiniMaxVL01Processor)
        self.assertEqual(reloaded.patch_size, 16)
        self.assertEqual(reloaded.num_additional_image_tokens, 1)
        self.assertEqual(reloaded.vision_feature_select_strategy, "default")
        self.assertEqual(reloaded.chat_template, CHAT_TEMPLATE)
        for key in expected:
            self.assertTrue((expected[key] == observed[key]).all(), key)

    @slow
    def test_public_processor_loads_without_remote_code(self):
        processor = AutoProcessor.from_pretrained("MiniMaxAI/MiniMax-VL-01", trust_remote_code=False)

        self.assertIsInstance(processor, MiniMaxVL01Processor)
        self.assertEqual(processor.image_token, "<image>")
        self.assertEqual(processor.image_processor.process_image_mode, "anyres")


if __name__ == "__main__":
    unittest.main()
