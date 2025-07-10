# Copyright 2025 The Keye Team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Keye model."""

import unittest

import requests

from transformers import (
    AutoProcessor,
    KeyeConfig,
    KeyeForConditionalGeneration,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    torch_device,
)
from transformers.utils import is_cv2_available

from ...test_modeling_common import (
    floats_tensor,
    ids_tensor,
)


if is_cv2_available():
    pass

if is_torch_available():
    import torch

else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class KeyeVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        ignore_index=-100,
        image_size=14,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        vision_start_token_id=3,
        image_token_id=4,
        video_token_id=5,
        hidden_act="silu",
        hidden_size=32,
        vocab_size=99,
        intermediate_size=37,
        max_position_embeddings=512,
        max_window_layers=3,
        model_type="Keye",
        num_attention_heads=4,
        num_hidden_layers=4,
        num_key_value_heads=2,
        rope_theta=10000,
        tie_word_embeddings=True,
        is_training=True,
        vision_config={
            "depth": 2,
            "in_chans": 3,
            "hidden_act": "silu",
            "intermediate_size": 16,
            "out_hidden_size": 16,
            "hidden_size": 16,
            "num_heads": 4,
            "patch_size": 14,
            "spatial_patch_size": 14,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
        },
        rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.vision_start_token_id = vision_start_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.max_window_layers = max_window_layers
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.vision_config = vision_config
        self.rope_scaling = rope_scaling
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.num_image_tokens = 32
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return KeyeConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            vision_config=self.vision_config,
            model_type=self.model_type,
            max_window_layers=self.max_window_layers,
            rope_scaling=self.rope_scaling,
            tie_word_embeddings=self.tie_word_embeddings,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            vision_start_token_id=self.vision_start_token_id,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vocab_size=self.vocab_size,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        temporal_patch_size = config.vision_config.temporal_patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (patch_size**2),
                self.num_channels * (patch_size**2) * temporal_patch_size,
            ]
        )

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.vision_start_token_id] = self.pad_token_id
        input_ids[:, self.num_image_tokens] = self.image_token_id
        input_ids[:, self.num_image_tokens - 1] = self.vision_start_token_id
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class KeyeIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("Kwai-Keye/Keye-VL-8B-Preview", trust_remote_code=True)
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Take a look at the picture."},
                    {"type": "image"},
                    {"type": "text", "text": "What kind of cat is this?"},
                ],
            }
        ]
        url = "https://s1-11508.kwimgs.com/kos/nlav11508/mllm_all/ziran_jiafeimao_11.jpg"
        self.image = Image.open(requests.get(url, stream=True).raw).resize(size=(32, 32))

        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    # @slow
    def test_small_model_integration_test(self):
        model = KeyeForConditionalGeneration.from_pretrained(
            "Kwai-Keye/Keye-VL-8B-Preview", torch_dtype="auto", device_map="auto", trust_remote_code=True
        )

        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[self.image], return_tensors="pt")
        expected_input_ids = [151644, 872, 198, 17814, 264, 1401, 518, 279, 6802, 13, 151652, 151655, 151655, 151655, 151655, 151655, 151655]  # fmt: skip
        torch.testing.assert_close(expected_input_ids, inputs.input_ids[0].tolist()[:17])
        expected_pixel_slice = torch.tensor(
            [
                [
                    [[-0.0588, -0.0588, -0.0588], [-0.0588, -0.0588, -0.0588], [-0.0588, -0.0588, -0.0588]],
                    [[-0.0980, -0.0980, -0.0980], [-0.0980, -0.0980, -0.0980], [-0.0980, -0.0980, -0.0980]],
                    [[-0.2863, -0.2863, -0.2863], [-0.2863, -0.2863, -0.2863], [-0.2863, -0.2863, -0.2863]],
                ],
                [
                    [[-0.0588, -0.0588, -0.0588], [-0.0588, -0.0588, -0.0588], [-0.0588, -0.0588, -0.0588]],
                    [[-0.1059, -0.1059, -0.1059], [-0.1059, -0.1059, -0.1059], [-0.1059, -0.1059, -0.1059]],
                    [[-0.2863, -0.2863, -0.2863], [-0.2863, -0.2863, -0.2863], [-0.2863, -0.2863, -0.2863]],
                ],
                [
                    [[-0.0667, -0.0667, -0.0667], [-0.0667, -0.0667, -0.0667], [-0.0667, -0.0667, -0.0667]],
                    [[-0.1059, -0.1059, -0.1059], [-0.1059, -0.1059, -0.1059], [-0.1059, -0.1059, -0.1059]],
                    [[-0.2863, -0.2863, -0.2863], [-0.2863, -0.2863, -0.2863], [-0.2863, -0.2863, -0.2863]],
                ],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        torch.testing.assert_close(expected_pixel_slice, inputs.pixel_values[:3, :3, :3, :3], atol=5e-4, rtol=1e-5)

        # verify generation
        inputs = inputs.to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, top_k=1)
        EXPECTED_DECODED_TEXT = "user\nTake a look at the picture.What kind of cat is this?\nassistant\n<analysis>This question asks for the identification of the type of cat shown in the picture. The answer can be determined by visual characteristics, making it straightforward"
        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

