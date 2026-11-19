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
"""Testing suite for the PyTorch MiniCPM-Llama3-V 2.5 model."""

import unittest

from transformers import AutoProcessor, MiniCPMV2_5Config, is_torch_available
from transformers.testing_utils import cleanup, require_torch, require_torch_accelerator, slow, torch_device

from ...test_modeling_common import floats_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import MiniCPMV2_5ForConditionalGeneration, MiniCPMV2_5Model


class MiniCPMV2_5VisionText2TextModelTester(VLMModelTester):
    base_model_class = MiniCPMV2_5Model
    config_class = MiniCPMV2_5Config
    conditional_generation_class = MiniCPMV2_5ForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 2)
        kwargs.setdefault("image_token_id", 100)
        kwargs.setdefault("image_size", 32)
        kwargs.setdefault("patch_size", 8)  # each test image is one 4x4-patch slice
        kwargs.setdefault("query_num", 4)  # ... which the resampler turns into 4 tokens
        kwargs.setdefault("num_image_tokens", 4)
        kwargs.setdefault("vocab_size", 256)
        kwargs.setdefault("hidden_size", 128)  # the resampler uses 128-dim heads
        kwargs.setdefault("intermediate_size", 37)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 32)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("max_position_embeddings", 512)
        kwargs.setdefault("bos_token_id", 0)
        kwargs.setdefault("eos_token_id", 1)
        kwargs.setdefault("pad_token_id", 2)
        kwargs.setdefault("vision_hidden_act", "gelu_pytorch_tanh")
        kwargs.setdefault("vision_intermediate_size", 37)
        super().__init__(parent, **kwargs)

    def create_pixel_values(self, batch_size: int | None = None):
        # One slice per image, as a single row of 4 x 4 = 16 patches: (num_slices, channels, patch_size, 16 * patch_size)
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_patches = (self.image_size // self.patch_size) ** 2
        return floats_tensor([batch_size, self.num_channels, self.patch_size, num_patches * self.patch_size])

    def get_additional_inputs(self, config, input_ids, pixel_values, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        side = self.image_size // self.patch_size
        patch_attention_mask = torch.ones(batch_size, side * side, dtype=torch.bool, device=torch_device)
        grid = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij"), dim=-1)
        patch_position_ids = grid.reshape(1, side * side, 2).expand(batch_size, -1, -1).to(torch_device)
        return {"patch_attention_mask": patch_attention_mask, "patch_position_ids": patch_position_ids}


@require_torch
class MiniCPMV2_5ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = MiniCPMV2_5VisionText2TextModelTester

    def test_reverse_loading_mapping(self):
        # As in llava, the base model has no `model.` prefix to reverse the `^llm\.model\.` rename onto
        super().test_reverse_loading_mapping(skip_base_model=True)

    def _image_features_prepare_config_and_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict = {
            key: value for key, value in inputs_dict.items() if key.startswith("pixel") or key.startswith("patch_")
        }
        return config, inputs_dict


@slow
@require_torch_accelerator
class MiniCPMV2_5IntegrationTest(unittest.TestCase):
    # TODO: pin the `revision` of the Hub PR that adds the converted config and processor files
    model_id = "openbmb/MiniCPM-Llama3-V-2_5"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_small_model_vision_generation(self):
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = MiniCPMV2_5ForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.float16, device_map=torch_device
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
        decoded = processor.post_process_image_text_to_text(output[:, inputs["input_ids"].shape[1] :])[0]
        expected = (
            "The image depicts two cats lying on a pink blanket. One cat is on the left side, and the other is on "
            "the right side. Both cats appear to be resting or sleeping. There are also two remote controls placed "
            "near the cats."
        )
        self.assertEqual(decoded, expected)
