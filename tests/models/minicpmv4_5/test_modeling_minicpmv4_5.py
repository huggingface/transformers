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
"""Testing suite for the PyTorch MiniCPM-V 4.5 model."""

import unittest

from transformers import AutoProcessor, MiniCPMV4_5Config, is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_accelerator,
    require_torchcodec,
    slow,
    torch_device,
)

from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import MiniCPMV4_5ForConditionalGeneration, MiniCPMV4_5Model


class MiniCPMV4_5VisionText2TextModelTester(VLMModelTester):
    base_model_class = MiniCPMV4_5Model
    config_class = MiniCPMV4_5Config
    conditional_generation_class = MiniCPMV4_5ForConditionalGeneration

    small_config_overrides = {
        "batch_size": 2,
        "image_token_id": 100,
        "video_token_id": 101,
        "image_size": 32,
        "patch_size": 8,  # each test image is one 4x4-patch slice
        "resampler_n_latents": 4,  # ... which the resampler turns into 4 tokens
        "resampler_n_heads": 4,
        "num_image_tokens": 4,
        "vocab_size": 256,
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
class MiniCPMV4_5ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = MiniCPMV4_5VisionText2TextModelTester

    def test_reverse_loading_mapping(self):
        # As in llava, the base model has no `model.` prefix to reverse the `^llm\.model\.` rename onto
        super().test_reverse_loading_mapping(skip_base_model=True)

    def _image_features_prepare_config_and_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict = {
            key: value for key, value in inputs_dict.items() if key.startswith("pixel") or key == "patch_position_ids"
        }
        return config, inputs_dict

    def _video_features_prepare_config_and_inputs(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        patch_size = self.model_tester.patch_size
        side = self.model_tester.image_size // patch_size
        # Two groups of frames (3 and 2), one slice each, every frame a single row of side x side patches
        rows, cols = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
        inputs_dict = {
            "pixel_values_videos": floats_tensor([5, 3, patch_size, side * side * patch_size]).to(torch_device),
            "pixel_attention_mask_videos": torch.ones(5, side * side, dtype=torch.bool, device=torch_device),
            "patch_position_ids_videos": torch.stack([rows, cols], -1)
            .reshape(1, -1, 2)
            .repeat(5, 1, 1)
            .to(torch_device),
            "temporal_ids": torch.tensor([[0, 5, 10], [15, 20, -1]], device=torch_device),
        }
        return config, inputs_dict

    def test_video_frames_attended_per_group(self):
        config, inputs_dict = self._video_features_prepare_config_and_inputs()
        model = MiniCPMV4_5Model(config).to(torch_device).eval()
        temporal_ids = inputs_dict.pop("temporal_ids")
        # Shifting the second group's timestamps changes only that group's tokens
        shifted_ids = torch.where(temporal_ids[1:] >= 0, temporal_ids[1:] + 3, -1)
        with torch.no_grad():
            features = model.get_video_features(**inputs_dict, temporal_ids=temporal_ids).pooler_output
            shifted = model.get_video_features(
                **inputs_dict, temporal_ids=torch.cat([temporal_ids[:1], shifted_ids])
            ).pooler_output
        self.assertEqual(features.shape, (2, config.resampler_n_latents, config.text_config.hidden_size))
        torch.testing.assert_close(features[0], shifted[0])
        self.assertFalse(torch.allclose(features[1], shifted[1]))


@slow
@require_torch_accelerator
class MiniCPMV4_5IntegrationTest(unittest.TestCase):
    model_id = "openbmb/MiniCPM-V-4_5"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @require_torchcodec
    def test_small_model_video_generation(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = MiniCPMV4_5ForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/sample_demo_1.mp4",
                    },
                    {"type": "text", "text": "Describe the video."},
                ],
            }
        ]
        # At most 20 groups, so the 10 s clip is packed 3 frames per group
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            max_num_frames=20,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(torch_device)
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        decoded = processor.post_process_image_text_to_text(output[:, inputs["input_ids"].shape[1] :])[0]
        expected_texts = Expectations(
            {
                ("cuda", None): (
                    "The video begins with a little girl sitting on a bed, wearing glasses and a green shirt, reading "
                    "a book. She is also wearing pink pants. The room is well-lit, and there is a crib in the "
                    "background. The girl is seen turning the pages of the book and occasionally looking up, possibly "
                    "to understand"
                ),
            }
        )
        self.assertEqual(decoded, expected_texts.get_expectation())
