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
"""Testing suite for the PyTorch Ovis2.5 model."""

import unittest

from parameterized import parameterized

from transformers import (
    Ovis2_5Config,
    Ovis2_5ForConditionalGeneration,
    Ovis2_5Model,
    Ovis2_5Processor,
    Ovis2_5VisionConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_image_processing_common import load_coco_image
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class Ovis2_5VisionText2TextModelTester(VLMModelTester):
    base_model_class = Ovis2_5Model
    config_class = Ovis2_5Config
    conditional_generation_class = Ovis2_5ForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 2)
        kwargs.setdefault("vocab_size", 32)
        kwargs.setdefault("hidden_size", 16)
        kwargs.setdefault("num_hidden_layers", 1)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("max_position_embeddings", 32)
        kwargs.setdefault("attention_dropout", 0.0)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("image_size", 8)
        kwargs.setdefault("patch_size", 2)
        kwargs.setdefault(
            "num_image_tokens",
            (kwargs["image_size"] // kwargs["patch_size"] // Ovis2_5VisionConfig.spatial_merge_size) ** 2,
        )
        kwargs.setdefault("image_token_id", 4)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("image_start_token_id", 5)
        kwargs.setdefault("image_end_token_id", 6)
        kwargs.setdefault("video_start_token_id", 7)
        kwargs.setdefault("video_end_token_id", 8)
        kwargs.setdefault("vision_vocab_size", 12)
        kwargs.setdefault("vision_window_size", kwargs["image_size"])
        kwargs.setdefault("vision_hidden_act", Ovis2_5VisionConfig.hidden_act)
        kwargs.setdefault("vision_max_position_embeddings", Ovis2_5VisionConfig.max_position_embeddings)
        super().__init__(parent, **kwargs)
        self.num_image_patches = (self.image_size // self.patch_size) ** 2

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {
            self.video_token_id,
            self.image_start_token_id,
            self.image_end_token_id,
            self.video_start_token_id,
            self.video_end_token_id,
        }

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return torch.randn(
            batch_size * self.num_image_patches,
            self.num_channels * self.patch_size**2,
            device=torch_device,
        )

    def place_image_tokens(self, input_ids, config):
        input_ids = super().place_image_tokens(input_ids, config)
        input_ids[:, 0] = config.image_start_token_id
        input_ids[:, 1 : self.num_image_tokens + 1] = config.image_token_id
        input_ids[:, self.num_image_tokens + 1] = config.image_end_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        grid_size = self.image_size // self.patch_size
        return {
            "image_grid_thw": torch.tensor(
                [[1, grid_size, grid_size]] * batch_size,
                dtype=torch.long,
                device=torch_device,
            )
        }


@require_torch
class Ovis2_5ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Ovis2_5VisionText2TextModelTester
    # The visual-tokenizer head consumes the encoder state before post_layernorm.
    test_all_params_have_gradient = False

    def test_reverse_loading_mapping(self):
        # The official-key mapping targets the conditional model's `model.*` subtree.
        super().test_reverse_loading_mapping(skip_base_model=True)

    @parameterized.expand([("image",), ("video",)])
    def test_visual_pooler_outputs(self, modality):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = Ovis2_5Model(config).to(torch_device).eval()
        if modality == "video":
            inputs["pixel_values_videos"] = inputs.pop("pixel_values")
            inputs["video_grid_thw"] = inputs.pop("image_grid_thw")
            inputs["input_ids"][inputs["input_ids"] == config.image_start_token_id] = config.video_start_token_id
            inputs["input_ids"][inputs["input_ids"] == config.image_end_token_id] = config.video_end_token_id

        text_inputs = []

        def capture_text_inputs(module, args, kwargs):
            text_inputs.append(kwargs["inputs_embeds"].detach())

        hook = model.language_model.register_forward_pre_hook(capture_text_inputs, with_kwargs=True)
        with torch.no_grad():
            outputs = model(**inputs)
        hook.remove()

        pooler_output = getattr(outputs, f"{modality}_pooler_output")
        visual_mask = inputs["input_ids"] == getattr(config, f"{modality}_token_id")
        torch.testing.assert_close(pooler_output, text_inputs[0][visual_mask], rtol=0, atol=0)
        other_modality = "video" if modality == "image" else "image"
        self.assertIsNone(getattr(outputs, f"{other_modality}_pooler_output"))
        self.assertNotIn(f"{modality}_hidden_states", outputs)

    def test_video_forward(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = Ovis2_5ForConditionalGeneration(config).to(torch_device).eval()
        num_frames = 2
        num_video_tokens = num_frames * self.model_tester.num_image_tokens
        inputs["input_ids"][:, : num_video_tokens + 2] = torch.tensor(
            [config.video_start_token_id] + [config.video_token_id] * num_video_tokens + [config.video_end_token_id],
            device=torch_device,
        )
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        inputs.pop("pixel_values")
        inputs["pixel_values_videos"] = self.model_tester.create_pixel_values(
            batch_size=self.model_tester.batch_size * num_frames
        )
        inputs["video_grid_thw"] = inputs.pop("image_grid_thw")
        inputs["video_grid_thw"][:, 0] = num_frames

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertEqual(outputs.logits.shape[:2], inputs["input_ids"].shape)
        self.assertTrue(torch.isfinite(outputs.logits).all())


@slow
@require_torch_accelerator
class Ovis2_5IntegrationTest(unittest.TestCase):
    model_id = "ovis-community/Ovis2.5-2B"
    smoke_pixels = 448 * 448

    @classmethod
    def setUpClass(cls):
        cls.processor = Ovis2_5Processor.from_pretrained(
            cls.model_id, size={"shortest_edge": cls.smoke_pixels, "longest_edge": cls.smoke_pixels}
        )
        cls.model, cls.loading_info = Ovis2_5ForConditionalGeneration.from_pretrained(
            cls.model_id,
            dtype="auto",
            device_map=torch_device,
            output_loading_info=True,
        )

    @classmethod
    def tearDownClass(cls):
        del cls.model
        del cls.processor
        cleanup(torch_device, gc_collect=True)

    @parameterized.expand(
        [
            ("image", 1, "What animals are visible? Answer in one short sentence.", "Two cats"),
            (
                "video",
                4,
                "What remains visible throughout all four frames? Answer in one short sentence.",
                "The two remote controls remain visible throughout all four frames.",
            ),
        ]
    )
    def test_generation(self, modality, num_frames, prompt, expected_text):
        image = load_coco_image("000000039769.jpg").convert("RGB")
        visual_input = image if modality == "image" else [image.copy() for _ in range(num_frames)]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": modality, modality: visual_input},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            self.assertFalse(self.loading_info[key])
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            processor_kwargs={"videos_kwargs": {"do_sample_frames": False}},
        ).to(torch_device, dtype=self.model.dtype)

        self.assertEqual(getattr(inputs, f"{modality}_grid_thw").tolist(), [[num_frames, 24, 32]])
        output = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
        generated_text = self.processor.decode(
            output[0, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        self.assertEqual(generated_text, expected_text)
