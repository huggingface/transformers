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
"""Testing suite for the PyTorch MiniMax-M3-VL model."""

import copy
import unittest

from transformers import (
    AutoProcessor,
    MiniMaxM3VLConfig,
    MiniMaxM3VLForConditionalGeneration,
    MiniMaxM3VLModel,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


class MiniMaxM3VLVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        ignore_index=-100,
        image_token_index=4,
        is_training=True,
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "dense_intermediate_size": 128,
            "shared_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "rotary_dim": 16,
            "hidden_act": "silu",
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "vocab_size": 99,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "moe_layer_freq": [0, 1],
            "layer_types": [
                "full_attention",
                "minimax_m3_sparse",
            ],
            "use_routing_bias": True,
            "routed_scaling_factor": 2.0,
            "swiglu_alpha": 1.702,
            "swiglu_limit": 7.0,
            "tie_word_embeddings": False,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 5000000.0,
                "partial_rotary_factor": 0.5,
            },
            "sparse_attention_config": {
                "sparse_attention_freq": [0, 1],
                "sparse_block_size": 8,
                "sparse_disable_index_value": [0, 1],
                "sparse_init_block": 0,
                "sparse_index_dim": 16,
                "sparse_local_block": 1,
                "sparse_num_index_heads": 2,
                "sparse_score_type": "max",
                "sparse_topk_blocks": 4,
                "use_sparse_attention": True,
            },
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_channels": 3,
            "image_size": 14,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "spatial_merge_size": 1,
            "rope_theta": 10000.0,
        },
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.is_training = is_training
        self.text_config = text_config
        self.vision_config = vision_config

        self.pad_token_id = text_config["pad_token_id"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.hidden_size = text_config["hidden_size"]
        self.vocab_size = text_config["vocab_size"]

        self.num_channels = vision_config["num_channels"]
        self.image_size = vision_config["image_size"]
        self.patch_size = vision_config["patch_size"]
        self.temporal_patch_size = vision_config["temporal_patch_size"]
        self.spatial_merge_size = vision_config["spatial_merge_size"]

        # One patch per image (grid [1, 1, 1]) so that the generation common tests, which crop
        # all inputs along the batch dim, keep ``pixel_values`` and ``image_grid_thw`` consistent.
        self.num_patches = 1
        self.num_image_tokens = self.num_patches // (self.spatial_merge_size**2)
        self.seq_length = seq_length + self.num_image_tokens
        self.encoder_seq_length = self.seq_length

    def get_config(self):
        return MiniMaxM3VLConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_index=self.image_token_index,
            projector_hidden_size=self.text_config["hidden_size"],
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_dim = self.num_channels * (self.patch_size**2) * self.temporal_patch_size
        pixel_values = floats_tensor([self.batch_size * self.num_patches, patch_dim])
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()

        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 2
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        input_ids[input_ids == self.image_token_index] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = self.image_token_index

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class MiniMaxM3VLModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Model tester for `MiniMaxM3VLForConditionalGeneration`.
    """

    all_model_classes = (
        (
            MiniMaxM3VLModel,
            MiniMaxM3VLForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "image-text-to-text": MiniMaxM3VLForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    _is_composite = True

    def setUp(self):
        self.model_tester = MiniMaxM3VLVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MiniMaxM3VLConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs raise an explicit error when the number of images doesn't match the number
        of image tokens in the text, and that genuine multi-image cases are accepted.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        num_patches = self.model_tester.num_patches
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            curr_input_dict = copy.deepcopy(input_dict)
            _ = model(**curr_input_dict)  # successful forward with no modifications

            # remove one image but leave its image tokens in text
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][:-num_patches, ...]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][:-1, ...]
            with self.assertRaisesRegex(ValueError, "Image features and image tokens do not match"):
                _ = model(**curr_input_dict)

            # simulate multi-image case by concatenating inputs where each has exactly one image
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:num_patches]
            image_grid_thw = curr_input_dict["image_grid_thw"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # two image-token groups but one image raises an error
            with self.assertRaisesRegex(ValueError, "Image features and image tokens do not match"):
                _ = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

            # two images and two image-token groups don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            image_grid_thw = torch.cat([image_grid_thw, image_grid_thw], dim=0)
            _ = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)


@slow
@require_torch
class MiniMaxM3VLIntegrationTest(unittest.TestCase):
    model_id = "MiniMaxAI/MiniMax-M3-preview"

    def test_image_and_text_generation(self):
        from transformers import AutoConfig, AutoModelForImageTextToText, FineGrainedFP8Config

        cfg = AutoConfig.from_pretrained(self.model_id)
        quant = FineGrainedFP8Config(
            activation_scheme="dynamic",
            weight_block_size=tuple(cfg.quantization_config["weight_block_size"]),
            dequantize=True,
            modules_to_not_convert=cfg.quantization_config.get("ignored_layers"),
        )
        quant.quant_method = "mxfp8"
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            quantization_config=quant,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        model.model.vision_tower.to(torch.bfloat16)

        processor = AutoProcessor.from_pretrained(self.model_id)
        image = Image.new("RGB", (672, 672), (127, 127, 127))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image briefly."},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, thinking_mode="disabled"
        )
        inputs = processor(images=[image], text=text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
        self.assertIsInstance(decoded, str)
        self.assertGreater(len(decoded.strip()), 0)
