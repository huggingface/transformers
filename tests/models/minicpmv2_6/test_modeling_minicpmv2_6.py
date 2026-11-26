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
"""Testing suite for the PyTorch MiniCPM-V 2.6 model."""

import unittest

from transformers import AutoProcessor, MiniCPMV2_6Config, is_torch_available
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

    from transformers import MiniCPMV2_6ForConditionalGeneration, MiniCPMV2_6Model


class MiniCPMV2_6VisionText2TextModelTester(VLMModelTester):
    base_model_class = MiniCPMV2_6Model
    config_class = MiniCPMV2_6Config
    conditional_generation_class = MiniCPMV2_6ForConditionalGeneration

    small_config_overrides = {
        "batch_size": 2,
        "image_token_id": 3,  # low id, so it survives tests that shrink the embedding table
        "image_size": 32,
        "patch_size": 8,  # each test image is one 4x4-patch slice
        "resampler_n_latents": 4,  # ... which the resampler turns into 4 tokens
        "resampler_n_heads": 4,
        "num_image_tokens": 4,
        "hidden_size": 128,
        "intermediate_size": 37,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "hidden_act": "silu",
        "max_position_embeddings": 512,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "vision_hidden_act": "gelu_pytorch_tanh",
        "vision_intermediate_size": 37,
    }

    def create_pixel_values(self, batch_size: int | None = None):
        # One slice per image, as a single row of 4 x 4 = 16 patches: (num_slices, channels, patch_size, 16 * patch_size)
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_patches = (self.image_size // self.patch_size) ** 2
        return floats_tensor([batch_size, self.num_channels, self.patch_size, num_patches * self.patch_size])

    def get_additional_inputs(self, config, input_ids, pixel_values, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        side = self.image_size // self.patch_size
        # Every other slice is a smaller (side - 1) x side grid padded to side x side, so attention tests cover padding.
        pixel_attention_mask = torch.ones(batch_size, side * side, dtype=torch.bool, device=torch_device)
        pixel_attention_mask[1::2, (side - 1) * side :] = False
        rows, cols = torch.meshgrid(
            torch.arange(side, device=torch_device), torch.arange(side, device=torch_device), indexing="ij"
        )
        grid = torch.stack([rows, cols], dim=-1).reshape(1, side * side, 2).repeat(batch_size, 1, 1)
        patch_position_ids = grid * pixel_attention_mask[..., None]
        return {"pixel_attention_mask": pixel_attention_mask, "patch_position_ids": patch_position_ids}


@require_torch
class MiniCPMV2_6ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = MiniCPMV2_6VisionText2TextModelTester

    @unittest.skip(reason="SDPA can't dispatch on flash with the attention mask of padded slices")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def test_reverse_loading_mapping(self):
        # As in llava, the base model has no `model.` prefix to reverse the `^llm\.model\.` rename onto
        super().test_reverse_loading_mapping(skip_base_model=True)

    def _image_features_prepare_config_and_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict = {
            key: value for key, value in inputs_dict.items() if key.startswith("pixel") or key == "patch_position_ids"
        }
        return config, inputs_dict


@slow
@require_torch_accelerator
class MiniCPMV2_6IntegrationTest(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    # `revision`: the Hub PR with the converted config and processor files (the weights are the original ones)
    def _describe_image(self, model_id, revision):
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        model = MiniCPMV2_6ForConditionalGeneration.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16, device_map=torch_device
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": url_to_local_path("http://images.cocodataset.org/val2017/000000039769.jpg"),
                    },
                    {"type": "text", "text": "What is in this image?"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        return processor.post_process_image_text_to_text(output[:, inputs["input_ids"].shape[1] :])[0]

    def test_minicpm_v_4_vision_generation(self):
        decoded = self._describe_image("openbmb/MiniCPM-V-4", revision="refs/pr/6")
        expected_texts = Expectations(
            {
                ("cuda", None): (
                    "The image features two cats lying on a pink surface. There are also remote controls present in the "
                    "scene."
                ),
            }
        )
        self.assertEqual(decoded, expected_texts.get_expectation())

    def test_minicpm_v_2_6_vision_generation(self):
        decoded = self._describe_image("openbmb/MiniCPM-V-2_6", revision="refs/pr/60")
        expected_texts = Expectations(
            {
                ("cuda", None): (
                    "The image features two cats lying on a pink surface. One cat is on the left side, stretching "
                    "out with its paws up, while the other cat on the right is curled up, with its head resting on "
                    "its paw. There are also two remote controls in the image, one near the left cat and the other"
                ),
            }
        )
        self.assertEqual(decoded, expected_texts.get_expectation())
