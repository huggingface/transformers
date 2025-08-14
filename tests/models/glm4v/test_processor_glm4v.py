# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import numpy as np

from transformers import AutoProcessor
from transformers.testing_utils import require_av, require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Glm4vProcessor


if is_torch_available():
    import torch


@require_vision
@require_torch
class Glm4vProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Glm4vProcessor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        processor = Glm4vProcessor.from_pretrained(
            "THUDM/GLM-4.1V-9B-Thinking", patch_size=4, size={"shortest_edge": 12 * 12, "longest_edge": 18 * 18}
        )
        processor.save_pretrained(cls.tmpdirname)
        cls.image_token = processor.image_token

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

    @require_torch
    @require_av
    def _test_apply_chat_template(
        self,
        modality: str,
        batch_size: int,
        return_tensors: str,
        input_name: str,
        processor_name: str,
        input_data: list[str],
    ):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        if processor_name not in self.processor_class.attributes:
            self.skipTest(f"{processor_name} attribute not present in {self.processor_class}")

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
            batch_messages, add_generation_prompt=True, tokenize=True, return_tensors=return_tensors
        )
        add_special_tokens = True
        if processor.tokenizer.bos_token is not None and formatted_prompt[0].startswith(processor.tokenizer.bos_token):
            add_special_tokens = False
        tok_output = processor.tokenizer(
            formatted_prompt, return_tensors=return_tensors, add_special_tokens=add_special_tokens
        )
        expected_output = tok_output.input_ids
        self.assertListEqual(expected_output.tolist(), formatted_prompt_tokenized.tolist())

        # Test that kwargs passed to processor's `__call__` are actually used
        tokenized_prompt_100 = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
            max_length=100,
        )
        self.assertEqual(len(tokenized_prompt_100[0]), 100)

        # Test that `return_dict=True` returns text related inputs in the dict
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

        # Test that with modality URLs and `return_dict=True`, we get modality inputs in the dict
        for idx, url in enumerate(input_data[:batch_size]):
            batch_messages[idx][0]["content"] = [batch_messages[idx][0]["content"][0], {"type": modality, "url": url}]

        out_dict = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
            fps=2,  # by default no more than 2 frames per second, otherwise too slow
        )
        input_name = getattr(self, input_name)
        self.assertTrue(input_name in out_dict)
        self.assertEqual(len(out_dict["input_ids"]), batch_size)
        self.assertEqual(len(out_dict["attention_mask"]), batch_size)

        if modality == "video":
            # qwen pixels don't scale with bs same way as other models, calculate expected video token count based on video_grid_thw
            expected_video_token_count = 0
            for thw in out_dict["video_grid_thw"]:
                expected_video_token_count += thw[0] * thw[1] * thw[2]
            mm_len = expected_video_token_count
        else:
            mm_len = batch_size * 4
        self.assertEqual(len(out_dict[input_name]), mm_len)

        return_tensor_to_type = {"pt": torch.Tensor, "np": np.ndarray, None: list}
        for k in out_dict:
            self.assertIsInstance(out_dict[k], return_tensor_to_type[return_tensors])

    @require_av
    @unittest.skip("GLM4V can't sample frames from image frames")
    def test_apply_chat_template_video_1(self):
        pass

    @require_av
    @unittest.skip("GLM4V can't sample frames from image frames")
    def test_apply_chat_template_video_2(self):
        pass

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
                        {"type": "video"},
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ]
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 1)

        formatted_prompt_tokenized = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        expected_output = processor.tokenizer(formatted_prompt, return_tensors=None).input_ids
        self.assertListEqual(expected_output, formatted_prompt_tokenized)

        out_dict = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)
        self.assertListEqual(list(out_dict.keys()), ["input_ids", "attention_mask"])

        # Add video URL for return dict and load with `num_frames` arg
        messages[0][0]["content"][0] = {
            "type": "video",
            "url": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4",
        }

        # Load with `video_fps` arg
        video_fps = 1
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            video_fps=video_fps,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 40)

        # Load without any arg should load the whole video
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            do_sample_frames=False,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 600)

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
            do_sample_frames=False,
        )
        self.assertTrue(self.videos_input_name in out_dict_with_video)
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 4)
