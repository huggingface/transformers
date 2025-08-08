# Copyright 2024 HuggingFace Inc.
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

import shutil
import tempfile
import unittest
from io import BytesIO
from typing import Optional

import numpy as np
import requests

from transformers import SmolVLMProcessor
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.testing_utils import require_av, require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from PIL import Image


@require_torch
@require_vision
class SmolVLMProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = SmolVLMProcessor
    videos_input_name = "pixel_values"

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        processor_kwargs = cls.prepare_processor_dict()
        processor = SmolVLMProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct", **processor_kwargs)
        processor.save_pretrained(cls.tmpdirname)
        cls.image1 = Image.open(
            BytesIO(
                requests.get(
                    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                ).content
            )
        )
        cls.image2 = Image.open(
            BytesIO(requests.get("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg").content)
        )
        cls.image3 = Image.open(
            BytesIO(
                requests.get(
                    "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg"
                ).content
            )
        )
        cls.bos_token = processor.tokenizer.bos_token
        cls.image_token = processor.image_token
        cls.video_token = processor.video_token
        cls.fake_image_token = processor.fake_image_token
        cls.global_img_token = processor.global_image_token

        cls.bos_token_id = processor.tokenizer.convert_tokens_to_ids(cls.bos_token)
        cls.image_token_id = processor.tokenizer.convert_tokens_to_ids(cls.image_token)
        cls.fake_image_token_id = processor.tokenizer.convert_tokens_to_ids(cls.fake_image_token)
        cls.global_img_tokens_id = processor.tokenizer(cls.global_img_token, add_special_tokens=False)["input_ids"]
        cls.padding_token_id = processor.tokenizer.pad_token_id
        cls.image_seq_len = processor.image_seq_len

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_video_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).video_processor

    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @staticmethod
    def prepare_processor_dict():
        return {
            "image_seq_len": 2,
            "chat_template": "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}",
        }

    def prepare_video_inputs(self, batch_size: Optional[int] = None):
        """This function prepares a list of numpy videos."""
        video_input = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8)] * 8
        if batch_size is None:
            return [[video_input]]
        return [[video_input]] * batch_size

    def get_split_image_expected_tokens(self, processor, image_rows, image_cols):
        text_split_images = []
        for n_h in range(image_rows):
            for n_w in range(image_cols):
                text_split_images += (
                    [self.fake_image_token_id]
                    + processor.tokenizer(f"<row_{n_h + 1}_col_{n_w + 1}>", add_special_tokens=False)["input_ids"]
                    + [self.image_token_id] * self.image_seq_len
                )
            text_split_images += processor.tokenizer("\n", add_special_tokens=False)["input_ids"]
        text_split_images = text_split_images[:-1]  # remove last newline
        # add double newline, as it gets its own token
        text_split_images += processor.tokenizer("\n\n", add_special_tokens=False)["input_ids"]
        text_split_images += (
            [self.fake_image_token_id]
            + self.global_img_tokens_id
            + [self.image_token_id] * self.image_seq_len
            + [self.fake_image_token_id]
        )
        return text_split_images

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def test_process_interleaved_images_prompts_no_image_splitting(self):
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding_side="left")
        processor_components["image_processor"] = self.get_component("image_processor", do_image_splitting=False)
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)

        # Test that a single image is processed correctly
        inputs = processor(images=self.image1)
        image1_expected_size = (512, 512)
        self.assertEqual(np.array(inputs["pixel_values"]).shape, (1, 1, 3, *image1_expected_size))
        self.assertEqual(np.array(inputs["pixel_attention_mask"]).shape, (1, 1, *image1_expected_size))
        # fmt: on

        # Test a single sample with image and text
        image_str = "<image>"
        text_str = "In this image, we see"
        text = image_str + text_str
        inputs = processor(text=text, images=self.image1)

        # fmt: off
        tokenized_sentence = processor.tokenizer(text_str, add_special_tokens=False)
        expected_input_ids = [[self.fake_image_token_id] + self.global_img_tokens_id + [self.image_token_id] * self.image_seq_len + [self.fake_image_token_id] + tokenized_sentence["input_ids"]]
        self.assertEqual(inputs["input_ids"], expected_input_ids)
        self.assertEqual(inputs["attention_mask"], [[1] * len(expected_input_ids[0])])
        self.assertEqual(np.array(inputs["pixel_values"]).shape, (1, 1, 3, *image1_expected_size))
        self.assertEqual(np.array(inputs["pixel_attention_mask"]).shape, (1, 1, *image1_expected_size))
        # fmt: on

        # Test that batch is correctly processed
        image_str = "<image>"
        text_str_1 = "In this image, we see"
        text_str_2 = "In this image, we see"

        text = [
            image_str + text_str_1,
            image_str + image_str + text_str_2,
        ]
        images = [[self.image1], [self.image2, self.image3]]

        inputs = processor(text=text, images=images, padding=True)

        # fmt: off
        tokenized_sentence_1 = processor.tokenizer(text_str_1, add_special_tokens=False)
        tokenized_sentence_2 = processor.tokenizer(text_str_2, add_special_tokens=False)
        image_tokens = [self.fake_image_token_id] + self.global_img_tokens_id + [self.image_token_id] * self.image_seq_len + [self.fake_image_token_id]
        expected_input_ids_1 = image_tokens + tokenized_sentence_1["input_ids"]
        expected_input_ids_2 = 2 * image_tokens + tokenized_sentence_2["input_ids"]
        # Pad the first input to match the second input
        pad_len = len(expected_input_ids_2) - len(expected_input_ids_1)
        padded_expected_input_ids_1 = [self.padding_token_id] * pad_len + expected_input_ids_1

        self.assertEqual(
            inputs["input_ids"], [padded_expected_input_ids_1, expected_input_ids_2]
        )
        self.assertEqual(
            inputs["attention_mask"],
            [[0] * pad_len + [1] * len(expected_input_ids_1), [1] * len(expected_input_ids_2)]
        )
        self.assertEqual(np.array(inputs['pixel_values']).shape, (2, 2, 3, 512, 512))
        self.assertEqual(np.array(inputs['pixel_attention_mask']).shape, (2, 2, 512, 512))
        # fmt: on

    def test_process_interleaved_images_prompts_image_splitting(self):
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding_side="left")
        processor_components["image_processor"] = self.get_component("image_processor", do_image_splitting=True)
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)

        # Test that a single image is processed correctly
        inputs = processor(images=self.image1)
        self.assertEqual(np.array(inputs["pixel_values"]).shape, (1, 13, 3, 512, 512))
        self.assertEqual(np.array(inputs["pixel_attention_mask"]).shape, (1, 13, 512, 512))
        # fmt: on
        self.maxDiff = None

        # Test a single sample with image and text
        image_str = "<image>"
        text_str = "In this image, we see"
        text = image_str + text_str
        inputs = processor(text=text, images=self.image1)

        # fmt: off
        tokenized_sentence = processor.tokenizer(text_str, add_special_tokens=False)
        split_image1_tokens = self.get_split_image_expected_tokens(processor, 3, 4)
        expected_input_ids_1 = [split_image1_tokens + tokenized_sentence["input_ids"]]
        self.assertEqual(inputs["input_ids"], expected_input_ids_1)
        self.assertEqual(inputs["attention_mask"], [[1] * len(expected_input_ids_1[0])])
        self.assertEqual(np.array(inputs["pixel_values"]).shape, (1, 13, 3, 512, 512))
        self.assertEqual(np.array(inputs["pixel_attention_mask"]).shape, (1, 13, 512, 512))
        # fmt: on

        # Test that batch is correctly processed
        image_str = "<image>"
        text_str_1 = "In this image, we see"
        text_str_2 = "bla, bla"

        text = [
            image_str + text_str_1,
            text_str_2 + image_str + image_str,
        ]
        images = [[self.image1], [self.image2, self.image3]]

        inputs = processor(text=text, images=images, padding=True)

        # fmt: off
        tokenized_sentence_1 = processor.tokenizer(text_str_1, add_special_tokens=False)
        tokenized_sentence_2 = processor.tokenizer(text_str_2, add_special_tokens=False)

        split_image1_tokens = self.get_split_image_expected_tokens(processor, 3, 4)
        split_image2_tokens = self.get_split_image_expected_tokens(processor, 4, 4)
        split_image3_tokens = self.get_split_image_expected_tokens(processor, 3, 4)
        expected_input_ids_1 = split_image1_tokens + tokenized_sentence_1["input_ids"]
        expected_input_ids_2 = tokenized_sentence_2["input_ids"] + split_image2_tokens + split_image3_tokens
        # Pad the first input to match the second input
        pad_len = len(expected_input_ids_2) - len(expected_input_ids_1)
        padded_expected_input_ids_1 = [self.padding_token_id] * pad_len + expected_input_ids_1

        self.assertEqual(
            inputs["input_ids"], [padded_expected_input_ids_1, expected_input_ids_2]
        )
        self.assertEqual(
            inputs["attention_mask"],
            [[0] * pad_len + [1] * len(expected_input_ids_1), [1] * len(expected_input_ids_2)]
        )
        self.assertEqual(np.array(inputs['pixel_values']).shape, (2, 30, 3, 512, 512))
        self.assertEqual(np.array(inputs['pixel_attention_mask']).shape, (2, 30, 512, 512))
        # fmt: on

    def test_add_special_tokens_processor(self):
        processor = self.get_processor()

        image_str = "<image>"
        text_str = "In this image, we see"
        text = text_str + image_str

        # fmt: off
        inputs = processor(text=text, images=self.image1, add_special_tokens=False)
        tokenized_sentence = processor.tokenizer(text_str, add_special_tokens=False)
        split_image1_tokens = self.get_split_image_expected_tokens(processor, 3, 4)
        expected_input_ids = [tokenized_sentence["input_ids"] + split_image1_tokens]
        self.assertEqual(inputs["input_ids"], expected_input_ids)

        inputs = processor(text=text, images=self.image1)
        expected_input_ids = [tokenized_sentence["input_ids"] + split_image1_tokens]
        self.assertEqual(inputs["input_ids"], expected_input_ids)
        # fmt: on

    @unittest.skip(reason="from @molbap @zucchini-nlp, passing non-nested images is error-prone and not recommended")
    def test_non_nested_images_with_batched_text(self):
        processor = self.get_processor()
        processor.image_processor.do_image_splitting = False

        image_str = "<image>"
        text_str_1 = "In this image, we see"
        text_str_2 = "In this image, we see"

        text = [
            image_str + text_str_1,
            image_str + image_str + text_str_2,
        ]
        images = [[self.image1], [self.image2, self.image3]]

        inputs = processor(text=text, images=images, padding=True)

        self.assertEqual(np.array(inputs["pixel_values"]).shape, (2, 2, 3, 512, 512))
        self.assertEqual(np.array(inputs["pixel_attention_mask"]).shape, (2, 2, 512, 512))

    # Copied from tests.models.idefics2.test_processing_idefics2.Idefics2ProcessorTest.test_process_interleaved_images_prompts_image_error
    def test_process_interleaved_images_prompts_image_error(self):
        processor = self.get_processor()

        text = [
            "This is a test sentence.",
            "In this other sentence we try some good things",
        ]
        images = [[self.image1], [self.image2]]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)
        images = [[self.image1], []]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)

        text = [
            "This is a test sentence.<image>",
            "In this other sentence we try some good things<image>",
        ]
        images = [[self.image1], [self.image2, self.image3]]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)
        images = [[], [self.image2]]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)
        images = [self.image1, self.image2, self.image3]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)
        images = [self.image1]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)

        text = [
            "This is a test sentence.",
            "In this other sentence we try some good things<image>",
        ]
        images = [[self.image1], []]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)
        images = [[], [self.image2]]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)
        images = [self.image1, self.image2]
        with self.assertRaises(ValueError):
            processor(text=text, images=images, padding=True)
        images = [self.image1]
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
            "<|im_start|>User: What do these images show?<image><image><end_of_utterance>\n"
            "Assistant: The first image shows the statue of Liberty in New York. The second image picture depicts Idefix, the dog of Obelix in Asterix and Obelix.<end_of_utterance>\n"
            "User: And who is that?<end_of_utterance>\n"
            "Assistant:"
        )
        self.assertEqual(rendered, expected_rendered)

    @require_av
    @require_torch
    def test_apply_chat_template_video_frame_sampling(self):
        # overridden because SmolVLM has special preprocessing for videos
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4",
                        },
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        num_frames = 3
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            num_frames=num_frames,
            return_tensors="pt",
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        # SmolVLM doesn't sample `num_frames` exactly, by uses other sampling method
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), 3)

        # Load with `fps` arg
        fps = 1
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            fps=fps,
            return_tensors="pt",
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
        # SmolVLM doesn't sample 1 frame per second exactly, by uses other sampling method
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), fps * 10)

        # NOTE: the last assert checks are removed
        # Loading video as a list of frames (i.e. images) is not supported in SmolVLM

    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        video_processor = self.get_component("video_processor")
        tokenizer = self.get_component("tokenizer")

        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(
            tokenizer=tokenizer, image_processor=image_processor, video_processor=video_processor, **processor_kwargs
        )
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=2, modality="image")
        image_input = self.prepare_image_inputs(batch_size=2)
        image_input = [[image_input[0]], [image_input[1]]]
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            padding="max_length",
            max_length=76,
            truncation=True,
            max_image_size={"longest_edge": 300},
        )

        self.assertEqual(inputs["pixel_values"].shape[2], 3)
        self.assertEqual(inputs["pixel_values"].shape[3], 300)
        self.assertEqual(len(inputs["input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched_video(self):
        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"video_processor attribute not present in {self.processor_class}")
        processor_components = self.prepare_components()
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(**processor_components, **processor_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=2, modality="video")
        video_input = self.prepare_video_inputs(batch_size=2)
        inputs = processor(
            text=input_str,
            videos=video_input,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1,
            padding="max_length",
            max_length=172,
        )

        self.assertLessEqual(inputs[self.videos_input_name][0].mean(), 0)
        self.assertEqual(len(inputs["input_ids"][0]), 172)

    @require_torch
    @require_vision
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

    @require_torch
    @require_vision
    def test_missing_images_error(self):
        """Test that appropriate error is raised when images are referenced but not provided."""
        processor = self.get_processor()

        # Test single text with image token but no image
        text = "Let me show you this image: <image> What do you think?"
        with self.assertRaises(ValueError) as context:
            processor(text=text)
        self.assertTrue("tokens in the text but no images/videos were passed" in str(context.exception))

        # Test batch with image tokens but no images
        texts = [
            "First text with <image> token.",
            "Second text <image> with token.",
        ]
        with self.assertRaises(ValueError) as context:
            processor(text=texts)
        self.assertTrue("tokens in the text but no images/videos were passed" in str(context.exception))

        # Test with None as Images
        with self.assertRaises(ValueError) as context:
            processor(text=text, images=None)
        self.assertTrue("tokens in the text but no images/videos were passed" in str(context.exception))

        with self.assertRaises(ValueError) as context:
            processor(text=texts, images=None)
        self.assertTrue("tokens in the text but no images/videos were passed" in str(context.exception))

    def test_special_mm_token_truncation(self):
        """Tests that special vision tokens do not get truncated when `truncation=True` is set."""

        processor = self.get_processor()

        input_str = self.prepare_text_inputs(batch_size=2, modality="image")
        image_input = self.prepare_image_inputs(batch_size=2)
        image_input = [[image_input[0]], [image_input[1]]]
        _ = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            truncation=None,
            padding=True,
        )

        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=20,
            )

    @unittest.skip("SmolVLM cannot accept image URL as video frames, because it needs to know video fps and duration")
    def test_apply_chat_template_video_1(self):
        pass

    @unittest.skip(
        "SmolVLM cannot accept list of decoded video frames, because it needs to know video fps and duration"
    )
    def test_apply_chat_template_video_2(self):
        pass
