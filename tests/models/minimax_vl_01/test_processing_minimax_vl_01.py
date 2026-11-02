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

import unittest
from itertools import groupby

import numpy as np

from transformers import MiniMaxVL01Config, MiniMaxVL01Processor, PreTrainedTokenizerFast
from transformers.image_utils import PILImageResampling
from transformers.testing_utils import (
    require_tokenizers,
    require_torch,
    require_torchvision,
    require_vision,
    slow,
)
from transformers.utils import is_tokenizers_available, is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_tokenizers_available():
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel


if is_torch_available():
    import torch


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
        }
        vocab.update({token: index for index, token in enumerate(sorted(ByteLevel.alphabet()), start=len(vocab))})
        tokenizer = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
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
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        return image_processor_class(
            size={"height": 32, "width": 32},
            patch_size=16,
            image_grid_pinpoints=[[32, 32], [64, 32], [32, 64]],
            resample=PILImageResampling.NEAREST,
            do_center_crop=False,
        )

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

    def test_image_token_filling(self):
        processor = self.get_processor()
        expected_counts = [10, 16, 14]

        for image, expected_count in zip(self._images(), expected_counts):
            with self.subTest(image_shape=image.shape):
                inputs = processor(text=["Question <image>"], images=[image], return_tensors="pt")
                observed_count = (inputs.input_ids == processor.image_token_id).sum().item()
                self.assertEqual(observed_count, expected_count)

    def test_image_token_filling_multiple_images(self):
        processor = self.get_processor()
        images = self._images()
        inputs = processor(
            text=["<image> Question <image> Question <image>"],
            images=list(images),
            return_tensors="pt",
        )

        image_token_mask = (inputs.input_ids[0] == processor.image_token_id).tolist()
        image_token_counts = [len(list(tokens)) for is_image, tokens in groupby(image_token_mask) if is_image]
        self.assertEqual(image_token_counts, [10, 16, 14])
        self.assertEqual(tuple(inputs.pixel_values.shape), (8, 3, 32, 32))
        self.assertEqual(inputs.image_sizes.tolist(), [[32, 32], [64, 32], [32, 64]])

    # Adapted from ProcessorTesterMixin._test_apply_chat_template for VL-01's flat crop stream.
    def _test_apply_chat_template(
        self,
        modality: str,
        batch_size: int,
        return_tensors: str,
        input_name: str,
        processor_name: str,
        input_data: list[str],
    ):
        if modality != "image":
            return super()._test_apply_chat_template(
                modality, batch_size, return_tensors, input_name, processor_name, input_data
            )

        processor = self.get_processor()
        batch_messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Describe this."}],
                },
            ]
            for _ in range(batch_size)
        ]

        formatted_prompt = processor.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), batch_size)

        formatted_prompt_tokenized = processor.apply_chat_template(
            batch_messages, add_generation_prompt=True, tokenize=True, return_tensors=return_tensors
        )
        add_special_tokens = True
        if processor.tokenizer.bos_token is not None and formatted_prompt[0].startswith(processor.tokenizer.bos_token):
            add_special_tokens = False
        tok_output = processor.tokenizer(
            formatted_prompt, return_tensors=return_tensors, add_special_tokens=add_special_tokens
        )
        self.assertListEqual(tok_output.input_ids.tolist(), formatted_prompt_tokenized.tolist())

        tokenized_prompt = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=return_tensors,
            processor_kwargs={
                "padding": "max_length",
                "truncation": True,
                "max_length": self.chat_template_max_length,
            },
        )
        self.assertEqual(len(tokenized_prompt[0]), self.chat_template_max_length)

        out_dict_text = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
        )
        self.assertTrue(all(key in out_dict_text for key in ["input_ids", "attention_mask"]))
        self.assertEqual(len(out_dict_text["input_ids"]), batch_size)
        self.assertEqual(len(out_dict_text["attention_mask"]), batch_size)

        for message, url in zip(batch_messages, input_data[:batch_size]):
            message[1]["content"].append({"type": modality, "url": url})

        out_dict = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
        )
        input_name = getattr(self, input_name)
        self.assertIn(input_name, out_dict)
        self.assertEqual(len(out_dict["input_ids"]), batch_size)
        self.assertEqual(len(out_dict["attention_mask"]), batch_size)
        # Image crops are concatenated across the batch; image_sizes retains one entry per image.
        self.assertEqual(out_dict[input_name].ndim, 4)
        self.assertEqual(len(out_dict["image_sizes"]), batch_size)

        return_tensor_to_type = {"pt": torch.Tensor, "np": np.ndarray, None: list}
        for value in out_dict.values():
            self.assertIsInstance(value, return_tensor_to_type[return_tensors])

        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "It is the sound of"}],
        }
        for message in batch_messages:
            message.append(assistant_message)
        continue_prompt = processor.apply_chat_template(batch_messages, continue_final_message=True, tokenize=False)
        for prompt in continue_prompt:
            self.assertTrue(prompt.endswith("It is the sound of"))

    @slow
    def test_public_processor_loads_without_remote_code(self):
        processor = MiniMaxVL01Processor.from_pretrained(
            "MiniMaxAI/MiniMax-VL-01", config=MiniMaxVL01Config(), trust_remote_code=False
        )

        self.assertIsInstance(processor, MiniMaxVL01Processor)
        self.assertEqual(processor.image_token, "<image>")
        self.assertEqual(processor.image_processor.process_image_mode, "anyres")


if __name__ == "__main__":
    unittest.main()
