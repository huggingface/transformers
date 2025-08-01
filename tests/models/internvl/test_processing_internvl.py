# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import shutil
import tempfile
import unittest

from parameterized import parameterized

from transformers import AutoProcessor, AutoTokenizer, InternVLProcessor
from transformers.testing_utils import require_av, require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import MODALITY_INPUT_DATA, ProcessorTesterMixin


if is_torch_available():
    import torch


if is_vision_available():
    from transformers import GotOcr2ImageProcessor, InternVLVideoProcessor


@require_vision
class InternVLProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = InternVLProcessor
    videos_input_name = "pixel_values"

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()

        image_processor = GotOcr2ImageProcessor(
            do_resize=True,
            size={"height": 20, "width": 20},
            max_patches=2,
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=True,
            do_center_crop=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            do_convert_rgb=True,
        )
        video_processor = InternVLVideoProcessor(
            do_resize=True,
            size={"height": 20, "width": 20},
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            do_convert_rgb=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B-hf", padding_side="left")
        processor_kwargs = cls.prepare_processor_dict()
        processor = InternVLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            **processor_kwargs,
        )
        processor.save_pretrained(cls.tmpdirname)
        cls.image_token = processor.image_token
        cls.video_token = processor.video_token

    @staticmethod
    def prepare_processor_dict():
        return {"image_seq_length": 2}

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_video_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).video_processor

    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    # Copied from tests.models.llava.test_processing_llava.LlavaProcessorTest.test_get_num_vision_tokens
    def test_get_num_vision_tokens(self):
        "Tests general functionality of the helper used internally in vLLM"

        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertTrue("num_image_tokens" in output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertTrue("num_image_patches" in output)
        self.assertEqual(len(output["num_image_patches"]), 3)

    @require_av
    @require_torch
    def test_process_interleaved_images_videos(self):
        processor = self.get_processor()

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
                        },
                        {
                            "type": "image",
                            "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
                        },
                        {"type": "text", "text": "What are the differences between these two images?"},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
                        },
                        {"type": "text", "text": "What type of shot is the man performing?"},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": "https://llava-vl.github.io/static/images/view.jpg",
                        },
                        {"type": "text", "text": "Write a haiku for this image"},
                    ],
                }
            ],
        ]

        inputs_batched = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            num_frames=8,
        )

        # Process non batched inputs to check if the pixel_values and input_ids are reconstructed in the correct order when batched together
        images_patches_index = 0
        for i, message in enumerate(messages):
            inputs = processor.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                num_frames=8,
            )
            # We slice with [-inputs["input_ids"].shape[1] :] as the input_ids are left padded
            torch.testing.assert_close(
                inputs["input_ids"][0], inputs_batched["input_ids"][i][-inputs["input_ids"].shape[1] :]
            )
            torch.testing.assert_close(
                inputs["pixel_values"],
                inputs_batched["pixel_values"][
                    images_patches_index : images_patches_index + inputs["pixel_values"].shape[0]
                ],
            )
            images_patches_index += inputs["pixel_values"].shape[0]

    @require_torch
    @require_av
    def test_apply_chat_template_video_frame_sampling(self):
        processor = self.get_processor()

        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        signature = inspect.signature(processor.__call__)
        if "videos" not in {*signature.parameters.keys()} or (
            signature.parameters.get("videos") is not None
            and signature.parameters["videos"].annotation == inspect._empty
        ):
            self.skipTest("Processor doesn't accept videos at input")

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
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), num_frames)

        # Load with `video_fps` arg is not possible with InternVL (skip)

        # Load without any arg should use the default loading method
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 300)

        # Load video as a list of frames (i.e. images). NOTE: each frame should have same size
        # because we assume they come from one video
        messages[0][0]["content"][0] = {
            "type": "video",
            "url": [
                "https://www.ilankelman.org/stopsigns/australia.jpg",
                "https://www.ilankelman.org/stopsigns/australia.jpg",
            ],
        }
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 2)

    @require_av
    @parameterized.expand([(1, "pt"), (2, "pt")])
    def test_apply_chat_template_video(self, batch_size: int, return_tensors: str):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        if "video_processor" not in self.processor_class.attributes:
            self.skipTest(f"`video_processor` attribute not present in {self.processor_class}")

        batch_messages = [
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Describe this."}],
                },
            ]
        ] * batch_size

        # Test that jinja can be applied
        formatted_prompt = processor.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), batch_size)

        # Test that tokenizing with template and directly with `self.tokenizer` gives same output
        formatted_prompt_tokenized = processor.apply_chat_template(
            batch_messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
        add_special_tokens = True
        if processor.tokenizer.bos_token is not None and formatted_prompt[0].startswith(processor.tokenizer.bos_token):
            add_special_tokens = False
        tok_output = processor.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
        expected_output = tok_output.input_ids
        self.assertListEqual(expected_output.tolist(), formatted_prompt_tokenized.tolist())

        # Test that kwargs passed to processor's `__call__` are actually used
        tokenized_prompt_100 = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=100,
        )
        self.assertEqual(len(tokenized_prompt_100[0]), 100)

        # Test that `return_dict=True` returns text related inputs in the dict
        out_dict_text = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        self.assertTrue(all(key in out_dict_text for key in ["input_ids", "attention_mask"]))
        self.assertEqual(len(out_dict_text["input_ids"]), batch_size)
        self.assertEqual(len(out_dict_text["attention_mask"]), batch_size)

        # Test that with modality URLs and `return_dict=True`, we get modality inputs in the dict
        for idx, url in enumerate(MODALITY_INPUT_DATA["videos"][:batch_size]):
            batch_messages[idx][0]["content"] = [batch_messages[idx][0]["content"][0], {"type": "video", "url": url}]

        out_dict = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            num_frames=2,  # by default no more than 2 frames, otherwise too slow
        )
        self.assertTrue(self.videos_input_name in out_dict)
        self.assertEqual(len(out_dict["input_ids"]), batch_size)
        self.assertEqual(len(out_dict["attention_mask"]), batch_size)

        video_len = 2 if batch_size == 1 else 3  # InternVL patches out and removes frames after processing
        self.assertEqual(len(out_dict[self.videos_input_name]), video_len)
        for k in out_dict:
            self.assertIsInstance(out_dict[k], torch.Tensor)

        # Test continue from final message
        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "It is the sound of"}],
        }
        for batch_idx in range(batch_size):
            batch_messages[batch_idx] = batch_messages[batch_idx] + [assistant_message]
        continue_prompt = processor.apply_chat_template(batch_messages, continue_final_message=True, tokenize=False)
        for prompt in continue_prompt:
            self.assertTrue(prompt.endswith("It is the sound of"))  # no `eos` token at the end
