# Copyright 2025 Baidu and HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Ernie 4.5 VL model."""

import copy
import tempfile
import unittest

import requests

from transformers import (
    AutoProcessor,
    Ernie4_5_VLConfig,
    Ernie4_5_VLForConditionalGeneration,
    Ernie4_5_VLModel,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    cleanup,
    is_flaky,
    require_cv2,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)
from transformers.utils import is_cv2_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
)


if is_cv2_available():
    import cv2

if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class Ernie4_5_VLVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        ignore_index=-100,
        image_size=112,
        video_start_token_id=3,
        video_end_token_id=4,
        image_start_token_id=5,
        image_end_token_id=6,
        image_token_id=7,
        video_token_id=8,
        is_training=True,
        text_config=None,
        vision_config=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.ignore_index = ignore_index
        self.image_size = image_size
        self.bos_token_id = 0
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.is_training = is_training

        self.text_config = text_config
        if text_config is None:
            self.text_config = {
                "vocab_size": 99,
                "hidden_size": 16,
                "intermediate_size": 22,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "hidden_act": "silu",
                "max_position_embeddings": 512,
                "tie_word_embeddings": True,
                "rope_parameters": {"type": "ernie_3d", "freq_allocation": 2},
                "moe_intermediate_size":[22, 22],
                "moe_k": 2,
                "moe_num_experts": 8,
                "moe_num_shared_experts": 2,
                "moe_layer_start_index": 0,
                "moe_layer_interval": 1,
                "moe_norm_min": 1e-12,
            }

        self.vision_config = vision_config
        if vision_config is None:
            self.vision_config = {
                "depth": 2,
                "hidden_size": 32,
                "hidden_act": "silu",
                "intermediate_size": 22,
                "num_heads": 2,
                "text_hidden_size": 16,
                "spatial_merge_size": 1,
            }

        self.hidden_size = self.text_config["hidden_size"]
        self.num_hidden_layers = self.text_config["num_hidden_layers"]
        self.num_attention_heads = self.text_config["num_attention_heads"]
        self.vocab_size = self.text_config["vocab_size"]

        self.num_image_tokens = 64
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return Ernie4_5_VLConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            video_start_token_id=self.video_start_token_id,
            video_end_token_id=self.video_end_token_id,
            image_start_token_id=self.image_start_token_id,
            image_end_token_id=self.image_end_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (patch_size**2),
                self.num_channels * (patch_size**2)
            ]
        )

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.video_start_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_start_token_id] = self.pad_token_id
        input_ids[input_ids == self.video_end_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_end_token_id] = self.pad_token_id

        input_ids[:, 0] = self.image_start_token_id
        input_ids[:, 1 : 1 + self.num_image_tokens] = self.image_token_id
        input_ids[:, 1 + self.num_image_tokens] = self.image_end_token_id

        patch_size = config.vision_config.patch_size
        patches_per_side = self.image_size // patch_size

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, patches_per_side, patches_per_side]] * self.batch_size, device=torch_device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class Ernie4_5_VLModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Ernie4_5_VLModel,
            Ernie4_5_VLForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    model_split_percents = [0.7, 0.9]  # model too big to split at 0.5
    test_all_params_have_gradient = False  # e score correction bias + moe
    _is_composite = True

    def setUp(self):
        self.model_tester = Ernie4_5_VLVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Ernie4_5_VLConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        """
        Same as in GLM4V, see `tests/models/glm4v/test_modeling_glm4v.py` for reference
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # We don't want a few model inputs in our model input dictionary for generation tests
        input_keys_to_ignore = [
            # we don't want encoder-decoder models to start from filled decoder ids
            "decoder_input_ids",
            "decoder_attention_mask",
            # we'll set cache use in each test differently
            "use_cache",
            # ignore labels if it is in the input dict
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

    def test_inputs_embeds_matches_input_ids(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)
            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]
            del inputs["image_grid_thw"]

            inputs_embeds = model.get_input_embeddings()(input_ids)

            with torch.no_grad():
                out_ids = model(input_ids=input_ids, **inputs)[0]
                out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            torch.testing.assert_close(out_embeds, out_ids)


@require_torch
class Ernie4_5_VLIntegrationTest(unittest.TestCase):
    # TODO: vid, img
    pass
