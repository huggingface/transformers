# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Kimi2.6 model."""

import gc
import unittest

import requests

from transformers import (
    AutoProcessor,
    DeepseekV3Config,
    Kimi2_6Config,
    Kimi2_6VisionConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    backend_empty_cache,
    require_torch,
    slow,
    torch_device,
)

from ...test_modeling_common import (
    floats_tensor,
)
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import Kimi2_6ForConditionalGeneration, Kimi2_6Model


if is_vision_available():
    from PIL import Image


class Kimi2_6VisionText2TextModelTester(VLMModelTester):
    base_model_class = Kimi2_6Model
    config_class = Kimi2_6Config
    text_config_class = DeepseekV3Config
    vision_config_class = Kimi2_6VisionConfig
    conditional_generation_class = Kimi2_6ForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("image_size", 32)
        kwargs.setdefault("patch_size", 8)
        kwargs.setdefault("num_image_tokens", 16)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("num_heads", 4)
        kwargs.setdefault("pos_emb_height", 4)
        kwargs.setdefault("merge_kernel_size", (1, 1))
        kwargs.setdefault("pos_emb_width", 4)
        kwargs.setdefault("pos_emb_time", 1)
        kwargs.setdefault("kv_lora_rank", 16)
        kwargs.setdefault("q_lora_rank", 32)
        kwargs.setdefault("qk_rope_head_dim", 16)
        kwargs.setdefault("v_head_dim", 32)
        kwargs.setdefault("qk_nope_head_dim", 32)
        kwargs.setdefault(
            "rope_parameters",
            {
                "rope_type": "default",
                "rope_theta": 10000,
            },
        )
        super().__init__(parent, **kwargs)

        # These can be inferred from existing properties and don't get separate kwargs
        self.projection_hidden_size = self.hidden_size

    def create_pixel_values(self):
        return floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (self.patch_size**2),
                self.num_channels,
                self.patch_size,
                self.patch_size,
            ]
        )

    def place_image_tokens(self, input_ids, config):
        # Place image tokens with vision_start_token_id prefix
        input_ids = input_ids.clone()
        # Clear any accidental special tokens first
        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        # Place image tokens with vision_start_token_id prefix
        input_ids[:, : self.num_image_tokens] = self.image_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, pixel_values):
        return {
            "image_grid_thw": torch.tensor([[1, 4, 4]] * self.batch_size, device=torch_device),
        }


@require_torch
class Kimi2_6ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Kimi2_6VisionText2TextModelTester

    # Kimi has images shaped as (bs*patch_len, dim) so we can't slice to batches in generate
    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # We don't want a few model inputs in our model input dictionary for generation tests
        input_keys_to_ignore = [
            "decoder_input_ids",
            "decoder_attention_mask",
            "use_cache",
            "labels",
        ]

        # The diff from the general `prepare_config_and_inputs_for_generate` lies here
        patch_size = config.vision_config.patch_size
        filtered_image_length = batch_size * (self.model_tester.image_size**2) // (patch_size**2)
        filtered_inputs_dict = {
            k: v[:batch_size, ...] if isinstance(v, torch.Tensor) else v
            for k, v in inputs_dict.items()
            if k not in input_keys_to_ignore
        }
        filtered_inputs_dict["pixel_values"] = inputs_dict["pixel_values"][:filtered_image_length]

        # It is important set `eos_token_id` to `None` to avoid early stopping (would break for length-based checks)
        text_gen_config = config.get_text_config(decoder=True)
        if text_gen_config.eos_token_id is not None and text_gen_config.pad_token_id is None:
            text_gen_config.pad_token_id = (
                text_gen_config.eos_token_id
                if isinstance(text_gen_config.eos_token_id, int)
                else text_gen_config.eos_token_id[0]
            )
        text_gen_config.eos_token_id = None
        text_gen_config.forced_eos_token_id = None

        return config, filtered_inputs_dict


@require_torch
class Kimi26IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("todo")
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            }
        ]
        url = "https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/demo_small.jpg"
        self.image = Image.open(requests.get(url, stream=True).raw)

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    @slow
    def test_small_model_integration_test(self):
        model = Kimi2_6ForConditionalGeneration.from_pretrained("todo", dtype="auto", device_map="auto")

        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[self.image], return_tensors="pt")

        expected_input_ids = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 151652, 151655, 151655]  # fmt: skip
        assert expected_input_ids == inputs.input_ids[0].tolist()[:17]

        expected_pixel_slice = torch.tensor(
            [
                [0.8792, 0.8792, 0.9084],
                [1.1858, 1.1858, 1.2296],
                [1.2004, 1.2004, 1.2150],
                [1.4340, 1.4340, 1.4194],
                [1.3902, 1.4048, 1.4194],
                [1.5216, 1.5362, 1.5362],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        assert torch.allclose(expected_pixel_slice, inputs.pixel_values[:6, :3], atol=3e-3)

        # verify generation
        inputs = inputs.to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30)
        EXPECTED_DECODED_TEXT = "system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and intelligent nature, making them popular choices"

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch(self):
        model = Kimi2_6ForConditionalGeneration.from_pretrained("todo", dtype="auto", device_map="auto")
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text, text], images=[self.image, self.image], return_tensors="pt").to(
            torch_device
        )

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and intelligent nature, making them popular choices',
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and intelligent nature, making them popular choices',
        ]  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_expand(self):
        model = Kimi2_6ForConditionalGeneration.from_pretrained("todo", dtype="auto", device_map="auto")
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[self.image], return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, num_return_sequences=3)

        EXPECTED_DECODED_TEXT = [
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and intelligent nature, making them popular choices',
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and intelligent nature, making them popular choices',
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and intelligent nature, making them popular choices',
        ]  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch_wo_image(self):
        model = Kimi2_6ForConditionalGeneration.from_pretrained("todo", dtype="auto", device_map="auto")
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        messages2 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who are you?"},
        ]
        text2 = self.processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text, text2], images=[self.image], padding=True, return_tensors="pt").to(
            torch_device
        )

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            'system\nYou are a helpful assistant.\nuser\nWhat kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labradors are known for their friendly and intelligent nature, making them popular choices',
            'system\nYou are a helpful assistant.\nuser\nWho are you?\nassistant\nI am a large language model created by Alibaba Cloud. I am called Qwen.'
        ]  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
