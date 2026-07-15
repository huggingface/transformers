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
"""Focused tests for the native Cosmos3 Edge reasoner implementation."""

import copy
import unittest

from transformers import (
    AutoProcessor,
    Cosmos3EdgeConfig,
    Cosmos3EdgeForConditionalGeneration,
    Cosmos3EdgeModel,
    Cosmos3EdgeProjectorConfig,
    Cosmos3EdgeTextConfig,
    Cosmos3EdgeVisionConfig,
    is_torch_available,
)
from transformers.testing_utils import cleanup, require_torch, require_torch_accelerator, slow, torch_device

from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers.models.cosmos3_edge.modeling_cosmos3_edge import Cosmos3EdgeTextRotaryEmbedding


class Cosmos3EdgeVisionText2TextModelTester(VLMModelTester):
    """Tiny packed-vision inputs for the shared VLM model-test suite."""

    base_model_class = Cosmos3EdgeModel
    config_class = Cosmos3EdgeConfig
    text_config_class = Cosmos3EdgeTextConfig
    vision_config_class = Cosmos3EdgeVisionConfig
    conditional_generation_class = Cosmos3EdgeForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("vocab_size", 97)
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("intermediate_size", 64)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("max_position_embeddings", 128)
        kwargs.setdefault("hidden_act", "relu2")
        kwargs.setdefault("rms_norm_eps", 1e-5)
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("vision_start_token_id", 5)
        kwargs.setdefault("vision_end_token_id", 6)
        kwargs.setdefault("image_size", 4)
        kwargs.setdefault("patch_size", 2)
        kwargs.setdefault("num_image_tokens", 1)
        kwargs.setdefault("num_channels", 3)
        kwargs.setdefault("spatial_merge_size", 2)
        kwargs.setdefault(
            "rope_parameters",
            {"rope_type": "default", "rope_theta": 100_000_000, "mrope_section": [2, 1, 1]},
        )
        super().__init__(parent, **kwargs)

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        }

    def get_vision_config(self):
        return self.vision_config_class(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_channels=self.num_channels,
            patch_size=self.patch_size,
            num_patches=(self.image_size // self.patch_size) ** 2,
            spatial_merge_size=self.spatial_merge_size,
        )

    def get_config(self):
        return self.config_class(
            text_config=self.get_text_config(),
            vision_config=self.get_vision_config(),
            projector_config=Cosmos3EdgeProjectorConfig(
                input_hidden_size=self.hidden_size,
                merger_intermediate_size=self.intermediate_size,
                out_hidden_size=self.hidden_size,
                spatial_merge_size=self.spatial_merge_size,
            ),
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            pad_token_id=self.pad_token_id,
        )

    def create_pixel_values(self):
        # Edge consumes flattened spatial patches. A 2 x 2 patch grid is merged into one language token.
        return floats_tensor(
            [
                self.batch_size * (self.image_size // self.patch_size) ** 2,
                self.num_channels * self.patch_size**2,
            ]
        )

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        input_ids[:, 0] = self.vision_start_token_id
        input_ids[:, 1] = self.image_token_id
        input_ids[:, 2] = self.vision_end_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, modality_inputs):
        patch_grid_size = self.image_size // self.patch_size
        return {
            "image_grid_thw": torch.tensor(
                [[1, patch_grid_size, patch_grid_size]] * self.batch_size,
                device=input_ids.device,
            ),
            "mm_token_type_ids": (input_ids == self.image_token_id).long(),
        }


@require_torch
class Cosmos3EdgeModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Cosmos3EdgeVisionText2TextModelTester
    test_torch_exportable = False  # packed patch spans require data-dependent shape handling

    def test_reverse_loading_mapping(self):
        # Native conversion mappings target the conditional model's `language_model` subtree, not the bare model.
        super().test_reverse_loading_mapping(skip_base_model=True)

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        """Keep packed visual patches aligned with the corresponding text batch during generation tests."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        patches_per_image = (self.model_tester.image_size // config.vision_config.patch_size) ** 2
        filtered_inputs_dict = {}

        for key, value in inputs_dict.items():
            if key == "pixel_values":
                filtered_inputs_dict[key] = value[: batch_size * patches_per_image]
            elif key == "image_grid_thw":
                filtered_inputs_dict[key] = value[:batch_size]
            elif isinstance(value, torch.Tensor):
                filtered_inputs_dict[key] = value[:batch_size, ...]
            else:
                filtered_inputs_dict[key] = value

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

    def test_mismatching_num_image_tokens(self):
        # The shared VLM test slices one image tensor at a time. Edge stores images as a packed sequence of patches,
        # so an image must be sliced as its full `grid_thw.prod()` span instead.
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        patches_per_image = (self.model_tester.image_size // config.vision_config.patch_size) ** 2

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            _ = model(**input_dict)
            curr_input_dict = copy.deepcopy(input_dict)

            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-patches_per_image:]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][-1:]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            model.base_model.rope_deltas = None
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:patches_per_image]
            image_grid_thw = curr_input_dict["image_grid_thw"][:1]
            mm_token_type_ids = curr_input_dict["mm_token_type_ids"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            with self.assertRaises(ValueError):
                _ = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    mm_token_type_ids=torch.cat([mm_token_type_ids, mm_token_type_ids], dim=0),
                )

            model.base_model.rope_deltas = None
            _ = model(
                input_ids=input_ids,
                pixel_values=torch.cat([pixel_values, pixel_values], dim=0),
                image_grid_thw=torch.cat([image_grid_thw, image_grid_thw], dim=0),
                mm_token_type_ids=torch.cat([mm_token_type_ids, mm_token_type_ids], dim=0),
            )

    def test_mrope_interleaving_is_precomputed_in_inverse_frequencies(self):
        config = self.model_tester.get_text_config()
        rotary_embedding = Cosmos3EdgeTextRotaryEmbedding(config).to(torch_device)
        position_ids = torch.tensor(
            [
                [[0, 1, 2]],
                [[0, 3, 4]],
                [[0, 5, 6]],
            ],
            dtype=torch.long,
            device=torch_device,
        )
        hidden_states = torch.zeros((1, 3, config.head_dim), device=torch_device)

        cos, sin = rotary_embedding(hidden_states, position_ids)

        base_inv_freq = 1.0 / (
            config.rope_parameters["rope_theta"]
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.int64, device=torch_device).float() / config.head_dim)
        )
        reference_freqs = position_ids[..., None].float() * base_inv_freq
        interleaved_freqs = reference_freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = config.rope_parameters["mrope_section"][dim] * 3
            interleaved_freqs[..., slice(offset, length, 3)] = reference_freqs[dim, ..., slice(offset, length, 3)]
        expected_embedding = torch.cat((interleaved_freqs, interleaved_freqs), dim=-1)

        self.assertEqual(tuple(rotary_embedding.inv_freq.shape), (3, config.head_dim // 2))
        axis_assignment = rotary_embedding.inv_freq.ne(0).int().argmax(dim=0)
        self.assertEqual(axis_assignment.bincount(minlength=3).tolist(), config.rope_parameters["mrope_section"])
        torch.testing.assert_close(cos, expected_embedding.cos())
        torch.testing.assert_close(sin, expected_embedding.sin())

    def test_mrope_positions_use_merged_image_grid(self):
        config = self.model_tester.get_config()
        model = Cosmos3EdgeModel(config).to(torch_device)
        input_ids = torch.tensor(
            [
                [
                    config.vision_start_token_id,
                    config.image_token_id,
                    config.image_token_id,
                    config.image_token_id,
                    config.image_token_id,
                    config.vision_end_token_id,
                ]
            ],
            dtype=torch.long,
            device=torch_device,
        )
        mm_token_type_ids = torch.tensor([[0, 1, 1, 1, 1, 0]], dtype=torch.long, device=torch_device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=torch_device)

        position_ids, rope_deltas = model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
        )

        expected_position_ids = torch.tensor(
            [
                [[0, 1, 1, 1, 1, 3]],
                [[0, 1, 1, 2, 2, 3]],
                [[0, 1, 2, 1, 2, 3]],
            ],
            dtype=torch.long,
            device=torch_device,
        )
        torch.testing.assert_close(position_ids, expected_position_ids)
        self.assertEqual(rope_deltas.tolist(), [[-2]])


@slow
@require_torch_accelerator
@unittest.skip(reason="Enable after release")
class Cosmos3EdgeForConditionalGenerationIntegrationTest(unittest.TestCase):
    model_id = "nvidia/Cosmos3-Edge"

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_image_generation(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                    },
                    {"type": "text", "text": "Identify the main subject of this image briefly."},
                ],
            }
        ]
        model = Cosmos3EdgeForConditionalGeneration.from_pretrained(
            self.model_id, dtype="auto", device_map=torch_device
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        generated_text = self.processor.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        expected_text = (
            "The main subject of this image is a large white flower, likely a daffodil, positioned centrally in the "
            "foreground."
        )
        self.assertEqual(generated_text, expected_text)

    def test_video_generation(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "url": "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
                    },
                    {"type": "text", "text": "Describe the main subject and action in this video briefly."},
                ],
            }
        ]
        model = Cosmos3EdgeForConditionalGeneration.from_pretrained(
            self.model_id, dtype="auto", device_map=torch_device
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            num_frames=4,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        generated_text = self.processor.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        expected_text = (
            "A man wearing a white t-shirt and black pants holds a baby wrapped in a white towel. The baby makes "
            "small movements, and the man rubs its head and back while smiling."
        )
        self.assertEqual(generated_text, expected_text)
