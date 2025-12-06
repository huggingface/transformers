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

import math
import unittest

import numpy as np

from transformers import Lfm2VlProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torchvision_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from PIL import Image

    if is_torchvision_available():
        pass


@require_torch
@require_vision
class Lfm2VlProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Lfm2VlProcessor

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        return image_processor_class(
            tile_size=14,
            min_image_tokens=2,
            max_image_tokens=10,
            encoder_patch_size=2,
            do_image_splitting=False,
        )

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        processor_kwargs = cls.prepare_processor_dict()
        return tokenizer_class.from_pretrained("LiquidAI/LFM2-VL-1.6B", **processor_kwargs)

    @classmethod
    def _setup_test_attributes(cls, processor):
        # Create images with different sizes
        cls.small_image = Image.new("RGB", (256, 256))
        cls.large_image = Image.new("RGB", (512, 1024))
        cls.high_res_image = Image.new("RGB", (1024, 1024))
        cls.bos_token = processor.tokenizer.bos_token
        cls.image_token = processor.image_token

        cls.bos_token_id = processor.tokenizer.convert_tokens_to_ids(cls.bos_token)
        cls.image_token_id = processor.image_token_id
        cls.image_start_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_start_token)
        cls.image_end_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_end_token)
        cls.padding_token_id = processor.tokenizer.pad_token_id
        cls.image_thumbnail_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_thumbnail_token)

    @staticmethod
    def prepare_processor_dict():
        chat_template = (
            "{{bos_token}}{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n'}}"
            "{% if message['content'] is string %}"
            "{{ message['content'] }}"
            "{% else %}"
            "{% for content in message['content'] %}"
            "{% if content['type'] == 'image' %}"
            "{{ '<image>' }}"
            "{% elif content['type'] == 'text' %}"
            "{{ content['text'] }}"
            "{% endif %}"
            "{% endfor %}"
            "{% endif %}"
            "{{'<|im_end|>\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{'<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
        return {"chat_template": chat_template}

    @unittest.skip("Lfm2VlProcessor adds special tokens to the text")
    def test_tokenizer_defaults(self):
        pass

    # Override as Lfm2VL needs images/video to be an explicitly nested batch
    def prepare_image_inputs(self, batch_size=None):
        """This function prepares a list of PIL images for testing"""
        images = super().prepare_image_inputs(batch_size)
        if isinstance(images, (list, tuple)):
            images = [[image] for image in images]
        return images

    def get_split_image_expected_tokens(self, processor, image_rows, image_cols, add_thumbnail, image_seq_len):
        text_split_images = [self.image_start_token_id]
        num_patches_tile = processor.image_processor.tile_size // processor.image_processor.encoder_patch_size
        tile_seq_len = math.ceil(num_patches_tile / processor.image_processor.downsample_factor) ** 2
        for n_h in range(image_rows):
            for n_w in range(image_cols):
                text_split_images += (
                    processor.tokenizer(f"<|img_row_{n_h + 1}_col_{n_w + 1}|>", add_special_tokens=False)["input_ids"]
                    + [self.image_token_id] * tile_seq_len
                )
        if add_thumbnail:
            text_split_images += [self.image_thumbnail_token_id] + [self.image_token_id] * image_seq_len
        text_split_images += [self.image_end_token_id]
        return text_split_images

    def test_process_interleaved_images_prompts_no_image_splitting_single_image(self):
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding_side="left")
        processor_components["image_processor"] = self.get_component("image_processor", do_image_splitting=False)
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)
        image_str = "<image>"

        # Test that a single image is processed correctly
        inputs = processor(images=self.small_image, text=image_str)
        encoder_feature_dims = (
            3 * processor.image_processor.encoder_patch_size * processor.image_processor.encoder_patch_size
        )
        self.assertEqual(
            np.array(inputs["pixel_values"]).shape,
            (1, processor.image_processor.max_num_patches, encoder_feature_dims),
        )
        self.assertEqual(
            np.array(inputs["pixel_attention_mask"]).shape, (1, processor.image_processor.max_num_patches)
        )
        self.assertListEqual(inputs["spatial_shapes"].tolist(), [[6, 6]])
        # fmt: on

    def test_process_interleaved_images_prompts_no_image_splitting_single_image_with_text(self):
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding_side="left")
        processor_components["image_processor"] = self.get_component("image_processor", do_image_splitting=False)
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)

        image_str = "<image>"
        text_str = "In this image, we see"
        text = image_str + text_str
        inputs = processor(text=text, images=self.small_image)

        # fmt: off
        tokenized_sentence = processor.tokenizer(text_str, add_special_tokens=False)
        expected_input_ids = [[self.image_start_token_id] + [self.image_token_id] * 9 + [self.image_end_token_id] + tokenized_sentence["input_ids"]]
        self.assertEqual(inputs["input_ids"], expected_input_ids)
        self.assertEqual(inputs["attention_mask"], [[1] * len(expected_input_ids[0])])
        encoder_feature_dims = 3 * processor.image_processor.encoder_patch_size * processor.image_processor.encoder_patch_size
        self.assertEqual(np.array(inputs["pixel_values"]).shape, (1, processor.image_processor.max_num_patches, encoder_feature_dims))
        self.assertEqual(np.array(inputs["pixel_attention_mask"]).shape, (1, processor.image_processor.max_num_patches))
        self.assertListEqual(inputs["spatial_shapes"].tolist(), [[6, 6]])
        # fmt: on

    def test_process_interleaved_images_prompts_no_image_splitting_multiple_images(self):
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding_side="left")
        processor_components["image_processor"] = self.get_component("image_processor", do_image_splitting=False)
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)

        image_str = "<image>"
        text_str_1 = "In this image, we see"
        text_str_2 = "In this image, we see"

        text = [
            image_str + text_str_1,
            image_str + image_str + text_str_2,
        ]
        images = [[self.small_image], [self.small_image, self.small_image]]

        inputs = processor(text=text, images=images, padding=True)

        tokenized_sentence_1 = processor.tokenizer(text_str_1, add_special_tokens=False)
        tokenized_sentence_2 = processor.tokenizer(text_str_2, add_special_tokens=False)
        image_tokens = [self.image_start_token_id] + [self.image_token_id] * 9 + [self.image_end_token_id]
        expected_input_ids_1 = image_tokens + tokenized_sentence_1["input_ids"]
        expected_input_ids_2 = 2 * image_tokens + tokenized_sentence_2["input_ids"]
        # Pad the first input to match the second input
        pad_len = len(expected_input_ids_2) - len(expected_input_ids_1)
        padded_expected_input_ids_1 = [self.padding_token_id] * pad_len + expected_input_ids_1

        self.assertEqual(inputs["input_ids"], [padded_expected_input_ids_1, expected_input_ids_2])
        self.assertEqual(
            inputs["attention_mask"],
            [[0] * pad_len + [1] * len(expected_input_ids_1), [1] * len(expected_input_ids_2)],
        )
        encoder_feature_dims = (
            3 * processor.image_processor.encoder_patch_size * processor.image_processor.encoder_patch_size
        )
        self.assertEqual(
            np.array(inputs["pixel_values"]).shape,
            (3, processor.image_processor.max_num_patches, encoder_feature_dims),
        )
        self.assertEqual(
            np.array(inputs["pixel_attention_mask"]).shape, (3, processor.image_processor.max_num_patches)
        )
        self.assertListEqual(inputs["spatial_shapes"].tolist(), [[6, 6], [6, 6], [6, 6]])

    def test_process_interleaved_images_prompts_image_splitting(self):
        processor = self.get_processor()

        image_str = "<image>"
        text_str_1 = "In this image, we see"
        text_str_2 = "bla, bla"

        text = [image_str + text_str_1, text_str_2 + image_str + image_str]
        images = [[self.small_image], [self.high_res_image, self.high_res_image]]

        inputs = processor(
            text=text,
            images=images,
            padding=True,
            padding_side="left",
            max_pixels_tolerance=2.0,
            use_thumbnail=True,
            do_image_splitting=True,
        )

        tokenized_sentence_1 = processor.tokenizer(text_str_1, add_special_tokens=False)
        tokenized_sentence_2 = processor.tokenizer(text_str_2, add_special_tokens=False)

        small_image_tokens = self.get_split_image_expected_tokens(processor, 3, 3, True, 9)
        large_image_tokens = self.get_split_image_expected_tokens(processor, 3, 3, True, 9)
        high_res_image_tokens = self.get_split_image_expected_tokens(processor, 3, 3, True, 9)

        expected_input_ids_1 = small_image_tokens + tokenized_sentence_1["input_ids"]
        expected_input_ids_2 = tokenized_sentence_2["input_ids"] + large_image_tokens + high_res_image_tokens
        # Pad the first input to match the second input
        pad_len = len(expected_input_ids_2) - len(expected_input_ids_1)
        padded_expected_input_ids_1 = [self.padding_token_id] * pad_len + expected_input_ids_1

        self.assertEqual(inputs["input_ids"][0], padded_expected_input_ids_1)
        self.assertEqual(inputs["input_ids"][1], expected_input_ids_2)
        self.assertEqual(
            inputs["attention_mask"],
            [[0] * pad_len + [1] * len(expected_input_ids_1), [1] * len(expected_input_ids_2)],
        )
        self.assertEqual(np.array(inputs["pixel_values"]).shape, (30, 49, 12))
        self.assertEqual(np.array(inputs["pixel_attention_mask"]).shape, (30, 49))
        self.assertListEqual(inputs["spatial_shapes"].tolist(), ([[7, 7]] * 9 + [[6, 6]]) * 3)

    def test_add_special_tokens_processor_image_splitting(self):
        processor = self.get_processor()

        image_str = "<image>"
        text_str = "In this image, we see"
        text = text_str + image_str

        # fmt: off
        inputs = processor(text=text, images=self.high_res_image, add_special_tokens=False, do_image_splitting=True)
        tokenized_sentence = processor.tokenizer(text_str, add_special_tokens=False)
        split_high_res_image_tokens = self.get_split_image_expected_tokens(processor, 3, 3, True, 9)
        expected_input_ids = [tokenized_sentence["input_ids"] + split_high_res_image_tokens]
        self.assertEqual(inputs["input_ids"], expected_input_ids)
        # fmt: on

    def test_add_special_tokens_processor_image_splitting_large_image(self):
        processor = self.get_processor()

        image_str = "<image>"
        text_str = "In this image, we see"
        text = text_str + image_str

        # fmt: off
        inputs = processor(text=text, images=self.large_image, add_special_tokens=False, max_pixels_tolerance=2.0, do_image_splitting=True)
        tokenized_sentence = processor.tokenizer(text_str, add_special_tokens=False)
        large_image_tokens = self.get_split_image_expected_tokens(processor, 4, 2, True, 8)
        expected_input_ids = [tokenized_sentence["input_ids"] + large_image_tokens]
        self.assertEqual(inputs["input_ids"], expected_input_ids)
        # fmt: on

    def test_add_special_tokens_processor_image_no_splitting(self):
        processor = self.get_processor()

        image_str = "<image>"
        text_str = "In this image, we see"
        text = image_str + text_str

        # fmt: off
        inputs = processor(text=text, images=self.high_res_image, add_special_tokens=False, use_image_special_tokens=True, do_image_splitting=False)
        tokenized_sentence = processor.tokenizer(text_str, add_special_tokens=False)
        split_high_res_image_tokens = [self.image_start_token_id] + [self.image_token_id] * 9 + [self.image_end_token_id]
        expected_input_ids = [split_high_res_image_tokens + tokenized_sentence["input_ids"]]
        self.assertEqual(inputs["input_ids"], expected_input_ids)
        # fmt: on

    def test_process_interleaved_images_prompts_image_error(self):
        processor = self.get_processor()

        text = [
            "This is a test sentence.",
            "In this other sentence we try some good things",
        ]
        images = [[self.small_image], [self.large_image]]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)
        images = [[self.small_image], []]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)

        text = [
            "This is a test sentence.<image>",
            "In this other sentence we try some good things<image>",
        ]
        images = [[self.small_image], [self.large_image, self.high_res_image]]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)
        images = [[], [self.large_image]]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)
        images = [self.small_image, self.large_image, self.high_res_image]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)
        images = [self.small_image]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)

        text = [
            "This is a test sentence.",
            "In this other sentence we try some good things<image>",
        ]
        images = [[self.small_image], []]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)

        images = [[], [self.large_image]]
        processor(text=text, images=images, padding=True)

        images = [self.small_image, self.large_image]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)

        images = [self.small_image]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)

    def test_apply_chat_template(self):
        # Message contains content which a mix of lists with images and image urls and string
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do these images show?"},
                    {"type": "image"},
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The first image shows the statue of Liberty in New York. The second image picture depicts Idefix, the dog of Obelix in Asterix and Obelix.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "And who is that?"}]},
        ]
        processor = self.get_processor()
        # Make short sequence length to test that the fake tokens are added correctly
        rendered = processor.apply_chat_template(messages, add_generation_prompt=True)

        expected_rendered = (
            "<|startoftext|><|im_start|>user\nWhat do these images show?<image><image><|im_end|>\n"
            "<|im_start|>assistant\nThe first image shows the statue of Liberty in New York. The second image picture depicts Idefix, the dog of Obelix in Asterix and Obelix.<|im_end|>\n"
            "<|im_start|>user\nAnd who is that?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.assertEqual(rendered, expected_rendered)

    def test_text_only_inference(self):
        """Test that the processor works correctly with text-only input."""
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding_side="left")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)

        text = "This is a simple text without images."
        inputs = processor(text=text)

        tokenized_sentence = processor.tokenizer(text, add_special_tokens=False)
        expected_input_ids = [tokenized_sentence["input_ids"]]

        self.assertEqual(inputs["input_ids"], expected_input_ids)
        self.assertEqual(inputs["attention_mask"], [[1] * len(expected_input_ids[0])])
        self.assertTrue("pixel_values" not in inputs)
        self.assertTrue("pixel_attention_mask" not in inputs)

        # Test batch of texts without image tokens
        texts = ["First text.", "Second piece of text."]
        batch_inputs = processor(text=texts, padding=True)

        tokenized_1 = processor.tokenizer(texts[0], add_special_tokens=False)
        tokenized_2 = processor.tokenizer(texts[1], add_special_tokens=False)

        expected_1 = tokenized_1["input_ids"]
        expected_2 = tokenized_2["input_ids"]

        # Pad the shorter sequence
        pad_len = len(expected_2) - len(expected_1)
        if pad_len > 0:
            padded_expected_1 = [self.padding_token_id] * pad_len + expected_1
            expected_attention_1 = [0] * pad_len + [1] * len(expected_1)
            self.assertEqual(batch_inputs["input_ids"], [padded_expected_1, expected_2])
            self.assertEqual(batch_inputs["attention_mask"], [expected_attention_1, [1] * len(expected_2)])
        else:
            pad_len = -pad_len
            padded_expected_2 = [self.padding_token_id] * pad_len + expected_2
            expected_attention_2 = [0] * pad_len + [1] * len(expected_2)
            self.assertEqual(batch_inputs["input_ids"], [expected_1, padded_expected_2])
            self.assertEqual(batch_inputs["attention_mask"], [[1] * len(expected_1), expected_attention_2])

    def test_missing_images_error(self):
        """Test that appropriate error is raised when images are referenced but not provided."""
        processor = self.get_processor()

        # Test single text with image token but no image
        text = "Let me show you this image: <image> What do you think?"
        with self.assertRaises(ValueError) as context:
            processor(text=text)
        self.assertTrue("We detected 1 tokens in the text but no images were passed" in str(context.exception))

        # Test batch with image tokens but no images
        texts = [
            "First text with <image> token.",
            "Second text <image> with token.",
        ]
        with self.assertRaises(ValueError) as context:
            processor(text=texts)
        self.assertTrue("We detected 2 tokens in the text but no images were passed" in str(context.exception))

        # Test with None as Images
        with self.assertRaises(ValueError) as context:
            processor(text=text, images=None)
        self.assertTrue("We detected 1 tokens in the text but no images were passed" in str(context.exception))

        with self.assertRaises(ValueError) as context:
            processor(text=texts, images=None)
        self.assertTrue("We detected 2 tokens in the text but no images were passed" in str(context.exception))
