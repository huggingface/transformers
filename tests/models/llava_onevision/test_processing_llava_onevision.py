# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import unittest

import torch

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import (
        LlavaOnevisionProcessor,
    )


@require_vision
@require_torch
class LlavaOnevisionProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = LlavaOnevisionProcessor

    @classmethod
    def setUpClass(cls):
        # Ensure local assets are used instead of remote URLs to avoid network access in tests
        from tests.test_processing_common import MODALITY_INPUT_DATA
        from transformers import video_processing_utils, video_utils

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        local_image = os.path.join(repo_root, "coco_sample.png")
        if not os.path.isfile(local_image):
            import numpy as np
            from PIL import Image

            Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8")).save(local_image)

        local_tiny_video = os.path.join(repo_root, "tiny_video.mp4")
        if not os.path.isfile(local_tiny_video):
            try:
                import torchvision

                frames = (torch.rand(8, 64, 64, 3) * 255).byte()
                torchvision.io.write_video(local_tiny_video, frames, fps=4)
            except Exception:
                local_tiny_video = None

        local_videos = [
            os.path.join(repo_root, "Big_Buck_Bunny_720_10s_10MB.mp4"),
            os.path.join(repo_root, "sample_demo_1.mp4"),
        ]
        cls.local_tiny_video = local_tiny_video
        MODALITY_INPUT_DATA["images"] = [local_image, local_image]
        MODALITY_INPUT_DATA["videos"] = local_videos

        # Force video decoding to use torchvision backend to avoid torchcodec dependency during tests
        video_processing_utils.is_torchcodec_available = lambda: False  # type: ignore
        video_utils.is_torchcodec_available = lambda: False  # type: ignore
        super().setUpClass()

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        vocab_tokens = [
            ("<unk>", 0.0),
            ("<s>", 0.0),
            ("</s>", 0.0),
            ("[PAD]", 0.0),
            ("<image>", 0.0),
            ("<video>", 0.0),
            ("Hello", 0.0),
            ("world", 0.0),
        ]
        tokenizer = tokenizer_class(vocab=vocab_tokens, add_bos_token=True, add_eos_token=False)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>", "<video>"]})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "[PAD]"
        return tokenizer

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor", use_fast=False)
        return image_processor_class()

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token
        cls.video_token = processor.video_token

    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + ' '}}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>' }}{% endfor %}{# Render all video then #}{% for content in message['content'] | selectattr('type', 'equalto', 'video') %}{{ '<video>' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ '\n' + content['text'] }}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ '\n' + content['text'] }}{% endgeneration %}{% endfor %}{% endif %}{{'<|im_end|>'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
            "num_image_tokens": 6,
            "vision_feature_select_strategy": "default"
        }  # fmt: skip

    # Copied from tests.models.llava.test_processing_llava.LlavaProcessorTest.test_get_num_vision_tokens
    def test_get_num_vision_tokens(self):
        "Tests general functionality of the helper used internally in vLLM"

        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertTrue("num_image_tokens" in output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertTrue("num_image_patches" in output)
        self.assertEqual(len(output["num_image_patches"]), 3)

    # Copied from tests.models.llava.test_processing_llava.LlavaProcessorTest.test_chat_template_is_saved
    def test_chat_template_is_saved(self):
        processor_loaded = self.processor_class.from_pretrained(self.tmpdirname)
        processor_dict_loaded = json.loads(processor_loaded.to_json_string())
        # chat templates aren't serialized to json in processors
        self.assertFalse("chat_template" in processor_dict_loaded)

        # they have to be saved as separate file and loaded back from that file
        # so we check if the same template is loaded
        processor_dict = self.prepare_processor_dict()
        self.assertTrue(processor_loaded.chat_template == processor_dict.get("chat_template", None))

    def test_image_token_filling(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        processor.patch_size = 14
        processor.vision_feature_select_strategy = "default"
        processor.image_processor.crop_size = {"height": 336, "width": 336}
        processor.image_processor.size = {"shortest_edge": 336}
        processor.image_processor.image_grid_pinpoints = [[672, 336]]
        processor.num_image_tokens = (processor.image_processor.size["shortest_edge"] // processor.patch_size) ** 2
        # Important to check with non square image
        image = torch.randint(0, 2, (3, 503, 316))
        expected_image_tokens = 1525
        image_token_index = processor.image_token_id

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        inputs = processor(
            text=[processor.apply_chat_template(messages)],
            images=[image],
            return_tensors="pt",
        )
        image_tokens = (inputs["input_ids"] == image_token_index).sum().item()
        self.assertEqual(expected_image_tokens, image_tokens)

    @require_torch
    def test_apply_chat_template_video_frame_sampling(self):
        processor = self.get_processor()

        if self.local_tiny_video is None:
            self.skipTest("Local tiny video unavailable for sampling test")

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "url": self.local_tiny_video,
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
        self.assertEqual(len(out_dict_with_video[self.videos_input_name][0]), num_frames)

        # Choose an fps high enough to avoid rounding down to zero sampled frames on short dummy videos
        fps = 4
        out_dict_with_video = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            fps=fps,
            return_tensors="pt",
        )
        self.assertEqual(len(out_dict_with_video[self.videos_input_name]), 1)
