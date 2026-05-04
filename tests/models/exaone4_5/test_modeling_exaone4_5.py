# Copyright 2026 The LG AI Research and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch EXAONE 4.5 model."""

import copy
import unittest

from transformers import (
    is_torch_available,
)
from transformers.image_utils import load_image
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        Exaone4_5_Config,
        Exaone4_5_ForConditionalGeneration,
        Exaone4_5_Model,
        Exaone4_5_Processor,
        Exaone4_5_VisionConfig,
        Exaone4Config,
    )


class Exaone4_5_ModelTester(VLMModelTester):
    base_model_class = Exaone4_5_Model
    config_class = Exaone4_5_Config
    text_config_class = Exaone4Config
    vision_config_class = Exaone4_5_VisionConfig
    conditional_generation_class = Exaone4_5_ForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("vision_start_token_id", 5)
        kwargs.setdefault("vision_end_token_id", 6)
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("num_image_tokens", 1)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("depth", 2)
        kwargs.setdefault("num_heads", 4)
        kwargs.setdefault("spatial_merge_size", 1)
        kwargs.setdefault("temporal_patch_size", 2)
        kwargs.setdefault("out_hidden_size", 32)
        super().__init__(parent, **kwargs)

        # Exaone4_5 vision config expects `in_channels` instead of `num_channels`.
        self.in_channels = self.num_channels

    def create_pixel_values(self):
        # EXAONE 4.5 vision tower expects flattened patches:
        # (total_patches, channels * patch_size^2 * temporal_patch_size)
        return torch.rand(
            self.batch_size * (self.image_size**2) // (self.patch_size**2),
            self.num_channels * (self.patch_size**2) * self.temporal_patch_size,
            device=torch_device,
        )

    def get_additional_inputs(self, config, input_ids, pixel_values):
        return {"image_grid_thw": torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device)}

    def get_config(self):
        config = super().get_config()
        # Some generic generation tests expect these attrs for VLMs.
        config.vision_start_token_id = self.vision_start_token_id
        config.vision_end_token_id = self.vision_end_token_id
        return config


@require_torch
class Exaone4_5_ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Exaone4_5_ModelTester
    test_all_params_have_gradient = False

    def test_reverse_loading_mapping(self):
        super().test_reverse_loading_mapping(skip_base_model=True)

    def test_mismatching_num_image_tokens(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            curr_input_dict = copy.deepcopy(input_dict)
            _ = model(**curr_input_dict)

            # Test 1: fewer images than image placeholders -> should raise.
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-1:, ...]
            if "image_grid_thw" in curr_input_dict:
                curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][-1:, ...]
            if "image_sizes" in curr_input_dict:
                curr_input_dict["image_sizes"] = curr_input_dict["image_sizes"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # Test 2: one image but two prompts with image placeholders -> should raise.
            curr_input_dict = {key: val[:1] for key, val in curr_input_dict.items()}
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                if key in curr_input_dict and curr_input_dict[key] is not None:
                    curr_input_dict[key] = torch.cat([curr_input_dict[key], curr_input_dict[key]], dim=0)
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # Test 3: two images and two image placeholders -> should pass.
            curr_input_dict["pixel_values"] = torch.cat(
                [curr_input_dict["pixel_values"], curr_input_dict["pixel_values"]], dim=0
            )
            if "image_grid_thw" in curr_input_dict:
                curr_input_dict["image_grid_thw"] = torch.cat(
                    [curr_input_dict["image_grid_thw"], curr_input_dict["image_grid_thw"]], dim=0
                )
            if "image_sizes" in curr_input_dict:
                curr_input_dict["image_sizes"] = torch.cat(
                    [curr_input_dict["image_sizes"], curr_input_dict["image_sizes"]], dim=0
                )
            _ = model(**curr_input_dict)

    @unittest.skip("Model parallel auto-sharding for EXAONE 4.5 VLM is not supported yet.")
    def test_model_parallelism(self):
        pass

    @unittest.skip("Beam search with model parallel auto device_map is not stable for EXAONE 4.5 VLM yet.")
    def test_model_parallel_beam_search(self):
        pass


@require_torch
class Exaone4_5_IntegrationTest(unittest.TestCase):
    model_id = "LGAI-EXAONE/EXAONE-4.5-33B"
    model = None
    processor = None

    @classmethod
    def setUpClass(cls):
        cleanup(torch_device, gc_collect=True)
        cls.model = Exaone4_5_ForConditionalGeneration.from_pretrained(cls.model_id, device_map="auto")
        cls.processor = Exaone4_5_Processor.from_pretrained(cls.model_id)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_logits(self):
        input_ids = [70045, 1109, 115406, 16943, 11697, 115365, 19816, 12137, 375]
        input_ids = torch.tensor([input_ids]).to(torch_device)

        with torch.no_grad():
            out = self.model(input_ids).logits.float().cpu()

        EXPECTED_MEAN = Expectations(
            {
                ("cuda", (8, 6)): torch.tensor(
                    [[44.8527, 45.7216, 71.1159, 36.9564, 44.3283, 22.0527, 28.3233, 62.5739, 46.0708]]
                ),
            }
        )
        EXPECTED_SLICE = Expectations(
            {
                ("cuda", (8, 6)): torch.tensor(
                    [42.2500, 43.0000, 42.5000, 44.7500, 49.5000, 46.0000, 46.5000, 46.5000, 45.7500, 46.2500]
                ),
            }
        )

        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN.get_expectation(), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[0, 0, :10], EXPECTED_SLICE.get_expectation(), atol=1e-4, rtol=1e-4)

    @slow
    def test_model_generation_text_only(self):
        EXPECTED_TEXT = Expectations(
            {
                ("cuda", 8): (
                    '\nTell me about the Miracle on the Han river.\n\n<think>\n\n</think>\n\nThe **"Miracle on the Han River"**'
                    " is a term used to describe the rapid economic development and industrialization that South Korea experienced"
                ),
            }
        )
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Tell me about the Miracle on the Han river."}]}
        ]
        input_ids = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(torch_device)

        generated_ids = self.model.generate(input_ids=input_ids, max_new_tokens=20, do_sample=False)
        text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(text, EXPECTED_TEXT.get_expectation())

    @slow
    def test_model_generation_image_text(self):
        IMAGE_URL = (
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        )
        EXPECTED_TEXT = Expectations(
            {
                ("cuda", 8): (
                    "\n\nDescribe the image.\n\n<think>\n\n</think>\n\nThe image captures a fluffy, young lynx kitten walking across a snowy surface, its thick"
                ),
            }
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image = load_image(IMAGE_URL).convert("RGB")

        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(torch_device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(text, EXPECTED_TEXT.get_expectation())
