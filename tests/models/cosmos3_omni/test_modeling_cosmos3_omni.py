# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Cosmos3 Reasoner model.

The Reasoner tower is architecturally identical to Qwen3-VL, so this mirrors the Qwen3-VL
common-test setup. The custom regression tests in the Qwen3-VL suite (position-id / image /
video forward checks) are intentionally omitted here — they cover behavior inherited verbatim
from Qwen3-VL. The `test_mismatching_num_image_tokens` override and gradient-checkpointing
xfails are kept because the base `VLMModelTest` versions do not hold for this architecture.
"""

import copy
import unittest

import pytest

from transformers import (
    AutoProcessor,
    Cosmos3OmniConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_modeling_common import floats_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        Cosmos3OmniForConditionalGeneration,
        Cosmos3OmniModel,
    )


class Cosmos3OmniVisionText2TextModelTester(VLMModelTester):
    base_model_class = Cosmos3OmniModel
    config_class = Cosmos3OmniConfig
    text_config_class = Qwen3VLTextConfig
    vision_config_class = Qwen3VLVisionConfig
    conditional_generation_class = Cosmos3OmniForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("vision_start_token_id", 5)
        kwargs.setdefault("vision_end_token_id", 6)
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("num_image_tokens", 32)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("depth", 2)
        kwargs.setdefault("vision_hidden_act", "gelu_pytorch_tanh")
        kwargs.setdefault("num_heads", 4)
        kwargs.setdefault("spatial_merge_size", 1)
        kwargs.setdefault("temporal_patch_size", 2)
        kwargs.setdefault("num_position_embeddings", 16)
        kwargs.setdefault("deepstack_visual_indexes", [0, 1])
        kwargs.setdefault(
            "rope_parameters",
            {
                "rope_type": "default",
                "mrope_section": [16, 8, 8],
                "mrope_interleaved": True,
                "rope_theta": 10000,
            },
        )
        super().__init__(parent, **kwargs)

        # These can be inferred from existing properties and don't get separate kwargs
        self.out_hidden_size = self.hidden_size
        self.vision_hidden_size = self.hidden_size
        self.vision_intermediate_size = self.hidden_size

    def create_pixel_values(self):
        # Cosmos3 Reasoner (like Qwen3-VL) expects flattened patches:
        # (total_patches, channels * patch_size^2 * temporal_patch_size)
        return floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (self.patch_size**2),
                self.num_channels * (self.patch_size**2) * self.temporal_patch_size,
            ]
        )

    def place_image_tokens(self, input_ids, config):
        # Place image tokens with vision_start_token_id prefix
        input_ids = input_ids.clone()
        # Clear any accidental special tokens first
        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.vision_start_token_id] = self.pad_token_id
        # Place image tokens with vision_start_token_id prefix
        input_ids[:, 1] = self.image_token_id
        input_ids[:, 0] = self.vision_start_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, modality_inputs):
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        return {
            "image_grid_thw": torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device),
            "mm_token_type_ids": mm_token_type_ids,
        }

    def get_config(self):
        # Cosmos3OmniConfig expects text_config and vision_config as dicts, not config objects
        return self.config_class(
            text_config=self.get_text_config().to_dict(),
            vision_config=self.get_vision_config().to_dict(),
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            pad_token_id=self.pad_token_id,
        )


@require_torch
class Cosmos3OmniModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Cosmos3OmniVisionText2TextModelTester

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing(self):
        super().test_training_gradient_checkpointing()

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        super().test_training_gradient_checkpointing_use_reentrant_false()

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        super().test_training_gradient_checkpointing_use_reentrant_true()

    def test_reverse_loading_mapping(self):
        # The unified-checkpoint conversion for model_type "cosmos3_omni" (defined in
        # `conversion_mapping.py`) rewrites flat checkpoint keys into the nested
        # `model.language_model.*` / `model.visual.*` layout. That `model.` prefix is the
        # base-model prefix, so the mapping is only visible on the model-with-head, not on the
        # base `Cosmos3OmniModel` (whose keys lack the prefix). Skip the base-model check.
        super().test_reverse_loading_mapping(skip_base_model=True)

    def test_mismatching_num_image_tokens(self):
        # Override the base test because we need to slice image_grid_thw too
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            _ = model(**input_dict)  # successful forward with no modifications
            curr_input_dict = copy.deepcopy(input_dict)

            # remove one image but leave the image token in text
            patch_size = config.vision_config.patch_size
            one_img_length = (self.model_tester.image_size**2) // (patch_size**2)
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-one_img_length:, ...]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            model.base_model.rope_deltas = None
            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:one_img_length]
            image_grid_thw = curr_input_dict["image_grid_thw"][:1]
            mm_token_type_ids = curr_input_dict["mm_token_type_ids"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    mm_token_type_ids=torch.cat([mm_token_type_ids, mm_token_type_ids], dim=0),
                )

            model.base_model.rope_deltas = None
            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            image_grid_thw = torch.cat([image_grid_thw, image_grid_thw], dim=0)
            mm_token_type_ids = torch.cat(
                [curr_input_dict["mm_token_type_ids"][:1], curr_input_dict["mm_token_type_ids"][:1]], dim=0
            )
            _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )


@require_torch
@slow
class Cosmos3OmniForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("nvidia/Cosmos3-Nano")

        self.messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/demo_small.jpg"
                        ),
                    },
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            }
        ]
        self.messages_2 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": url_to_local_path("http://images.cocodataset.org/val2017/000000039769.jpg"),
                    },
                    {"type": "text", "text": "What do you see in this image?"},
                ],
            }
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_small_model_integration(self):
        # Let's make sure we test the preprocessing to replace what is used
        model = Cosmos3OmniForConditionalGeneration.from_pretrained(
            "nvidia/Cosmos3-Nano",
            dtype="bfloat16",
            device_map=torch_device,
        )

        inputs = self.processor.apply_chat_template(
            self.messages, tokenize=True, return_dict=True, add_generation_prompt=True, return_tensors="pt"
        ).to(torch_device, torch.bfloat16)

        output = model.generate(**inputs, do_sample=False, max_new_tokens=40)
        expected_decoded_texts = Expectations({
            ("cuda", None): 'user\nWhat kind of dog is this?\nassistant\nThe dog in the image appears to be a Labrador Retriever. It has a light brown or golden coat, which is characteristic of this breed. Labrador Retrievers are known for their friendly demeanor and',
        })  # fmt: skip
        EXPECTED_DECODED_TEXT = expected_decoded_texts.get_expectation()

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @require_torch_accelerator
    def test_small_model_integration_batched(self):
        model = Cosmos3OmniForConditionalGeneration.from_pretrained(
            "nvidia/Cosmos3-Nano", dtype="bfloat16", device_map=torch_device
        )

        inputs = self.processor.apply_chat_template(
            [self.messages, self.messages_2],
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(torch_device, torch.bfloat16)

        output = model.generate(**inputs, do_sample=False, max_new_tokens=40)

        expected_decoded_texts = Expectations(
            {
                ("cuda", None): [
                    "user\nWhat kind of dog is this?\nassistant\nThe dog in the image appears to be a Labrador Retriever. It has a light brown or golden coat, which is characteristic of this breed. Labrador Retrievers are known for their friendly demeanor and",
                    "user\nWhat do you see in this image?\nassistant\nIn this image, I see two cats sleeping on a pink couch. The cats appear to be of the same breed, with brown and black striped fur. They're both lying down in a relaxed position",
                ],
            }
        )
        EXPECTED_DECODED_TEXT = expected_decoded_texts.get_expectation()
        decoded_output = self.processor.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(decoded_output, EXPECTED_DECODED_TEXT)
