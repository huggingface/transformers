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
"""Testing suite for the PyTorch Xiaomi-Lens-0 model."""

import unittest

from transformers import AutoProcessor, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torch_accelerator,
    require_vision,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import floats_tensor, ids_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import XiaomiLens0Config, XiaomiLens0ForConditionalGeneration, XiaomiLens0Model
    from transformers.loss.loss_xiaomi_lens0 import XiaomiLens0ForConditionalGenerationLoss


class XiaomiLens0VisionText2TextModelTester(VLMModelTester):
    if is_torch_available():
        base_model_class = XiaomiLens0Model
        config_class = XiaomiLens0Config
        conditional_generation_class = XiaomiLens0ForConditionalGeneration

    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        image_size=64,
        text_config={
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "hidden_act": "silu",
            "head_dim": 32,
            "hidden_size": 32,
            "vocab_size": 99,
            "intermediate_size": 37,
            "max_position_embeddings": 512,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "linear_attention"],
            "num_key_value_heads": 2,
            "tie_word_embeddings": True,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
            "linear_conv_kernel_dim": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 8,
        },
        vision_config={
            "depth": 2,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 32,
            "out_hidden_size": 32,
            "hidden_size": 32,
            "num_heads": 4,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
        },
        mask_decoder_config={
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_attention_heads": 4,
            "vision_hidden_size": 32,
            "upscale_channels": 16,
            "mask_embed_dim": 8,
        },
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
        mask_token_id=7,
        tie_word_embeddings=True,
        is_training=True,
    ):
        self.parent = parent
        self.is_training = is_training

        self.text_config = text_config
        self.vision_config = vision_config
        self.mask_decoder_config = mask_decoder_config

        self.vocab_size = text_config["vocab_size"]
        self.pad_token_id = text_config["pad_token_id"]
        self.bos_token_id = text_config["bos_token_id"]
        self.eos_token_id = text_config["eos_token_id"]
        self.hidden_size = text_config["hidden_size"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.mask_token_id = mask_token_id
        self.tie_word_embeddings = tie_word_embeddings

        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.grid_size = image_size // vision_config["patch_size"]
        self.num_image_tokens = self.grid_size**2 // vision_config["spatial_merge_size"] ** 2
        self.seq_length = seq_length + self.num_image_tokens + 2

    def get_config(self):
        return XiaomiLens0Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            mask_decoder_config=self.mask_decoder_config,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            mask_token_id=self.mask_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
        )

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
            self.mask_token_id,
        }

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        patch_size = self.vision_config["patch_size"]
        temporal_patch_size = self.vision_config["temporal_patch_size"]
        return floats_tensor(
            [
                batch_size * self.grid_size**2,
                self.num_channels * (patch_size**2) * temporal_patch_size,
            ]
        )

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        input_ids[:, 0] = self.vision_start_token_id
        input_ids[:, 1 : self.num_image_tokens + 1] = self.image_token_id
        input_ids[:, self.num_image_tokens + 1] = self.vision_end_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return {
            "image_grid_thw": torch.tensor([[1, self.grid_size, self.grid_size]] * batch_size, device=torch_device),
            "mm_token_type_ids": (input_ids == self.image_token_id).long(),
        }

    def prepare_config_and_inputs(self):
        return self.get_config(), self.create_pixel_values()

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        special_token_ids = torch.tensor(list(self._special_token_ids), device=input_ids.device)
        input_ids[torch.isin(input_ids, special_token_ids)] = self.pad_token_id
        input_ids[:, -1] = self.pad_token_id
        input_ids, modality_inputs = self._prepare_modality_inputs(input_ids, config)

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(input_ids.shape, dtype=torch.long, device=torch_device),
            **modality_inputs,
        }
        return config, inputs_dict


@require_torch
class XiaomiLens0ModelTest(VLMModelTest, GenerationTesterMixin, unittest.TestCase):
    model_tester_class = XiaomiLens0VisionText2TextModelTester
    model_split_percents = [0.5, 0.8, 0.9]

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if return_labels and model_class is XiaomiLens0ForConditionalGeneration:
            # Add a mask token and mask labels so that training also covers the mask decoder
            inputs_dict["input_ids"][:, -2] = self.model_tester.mask_token_id
            image_size = self.model_tester.image_size
            mask_labels = torch.zeros(self.model_tester.batch_size, 1, image_size, image_size, device=torch_device)
            mask_labels[..., : image_size // 2, : image_size // 2] = 1
            inputs_dict["mask_labels"] = list(mask_labels)
        return inputs_dict

    def test_config(self):
        self.config_tester.run_common_tests()

    def _get_conv_state_shape(self, batch_size: int, config):
        num_v_heads = config.linear_num_value_heads
        num_k_heads = config.linear_num_key_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim
        intermediate_size = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim

        return (batch_size, intermediate_size, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        num_v_heads = config.linear_num_value_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim

        return (batch_size, num_v_heads, head_k_dim, head_v_dim)

    def test_attention_outputs(self):
        "Needs to be overwritten as Xiaomi-Lens-0 alternates between attention layers and gated deltanet layers (same as Qwen 3.5)."
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        seq_len = self.model_tester.seq_length
        num_attention_layers = sum(layer == "full_attention" for layer in config.text_config.layer_types)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class._from_config(config, attn_implementation="eager")
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), num_attention_layers)
            self.assertListEqual(
                list(attentions[0].shape[-3:]), [config.text_config.num_attention_heads, seq_len, seq_len]
            )

    def test_reverse_loading_mapping(self):
        # The mask decoder exists only on the conditional generation model.
        super().test_reverse_loading_mapping(skip_base_model=True)

    def test_loss(self):
        rng = torch.Generator().manual_seed(42)
        logits = torch.randn(2, 4, 5, generator=rng).to(torch_device)
        labels = torch.tensor([[-100, 1, 4, -100], [2, 0, -100, 3]], device=torch_device)
        mask_logits = torch.randn(4, 2, 3, generator=rng).to(torch_device)
        mask_labels = (torch.arange(24, device=torch_device).reshape(4, 2, 3) % 3 == 0).to(torch.uint8)
        valid_mask = torch.tensor(
            [
                [[1, 1, 0], [1, 0, 0]],
                [[0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [1, 1, 0]],
                [[0, 1, 0], [1, 0, 1]],
            ],
            dtype=torch.bool,
            device=torch_device,
        )
        # Unequal mask counts per image
        mask_logits = list(mask_logits.split([3, 1]))
        mask_labels = list(mask_labels.split([3, 1]))

        config = self.model_tester.get_config()
        config.text_config.vocab_size = 5

        loss, _, _ = XiaomiLens0ForConditionalGenerationLoss(
            logits=logits,
            labels=labels,
            mask_logits=mask_logits,
            mask_labels=mask_labels,
            mask_labels_valid_mask=list(valid_mask.split([3, 1])),
            config=config,
        )
        expected_loss = Expectations(
            {
                (None, None): 3.642270088195801,
                ("cuda", 8): 3.642270088195801,
            }
        ).get_expectation()
        torch.testing.assert_close(loss, torch.tensor(expected_loss, device=torch_device), atol=1e-5, rtol=1e-5)


@slow
@require_torch_accelerator
@require_vision
class XiaomiLens0IntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    # TODO: Update before merge
    model_id = "guarin/Xiaomi-Lens-0"  # "SeerRay-Lab/Xiaomi-Lens-0"
    image_url = (
        "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"
    )

    def setUp(self):
        super().setUp()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def get_messages(self, text):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": url_to_local_path(self.image_url)},
                    {"type": "text", "text": text},
                ],
            }
        ]

    def test_grounding_logits(self):
        model = XiaomiLens0ForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device
        ).eval()

        messages = self.get_messages("How many cats are there?\nReport bbox coordinates and masks.")
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        expected_logits = Expectations(
            {("cuda", 9): [5.09375, 6.4375, 7.0, 6.5, 0.92578125, 3.265625, 4.28125, 3.640625, 5.5625, 5.84375]}
        ).get_expectation()
        torch.testing.assert_close(
            outputs.logits[0, -1, :10].float(),
            torch.tensor(expected_logits, device=torch_device),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_grounding_generation(self):
        model = XiaomiLens0ForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device
        ).eval()

        messages = self.get_messages(
            'Locate all instances in the image that match the description "the cat on the left".'
            "\nReport bbox coordinates and masks."
        )
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
                output_hidden_states=[model.config.text_config.num_hidden_layers - 1],
            )
            hidden_states = torch.cat([step[-1] for step in outputs.hidden_states], dim=1)
            mask_logits, mask_token_indices = model.decode_masks(
                outputs.sequences,
                hidden_states,
                outputs.image_last_hidden_state,
                inputs.image_grid_thw,
                ignore_thinking=True,
            )
        results = self.processor.post_process_grounding(
            outputs.sequences, inputs.image_grid_thw, mask_logits, mask_token_indices, target_sizes=[(480, 640)]
        )
        self.assertEqual(results[0]["masks"].shape, (1, 480, 640))

        expected_logits = Expectations(
            {("cuda", 9): [6.34375, 8.125, 2.53125, 4.90625, 1.3515625, 4.28125, 7.21875, 2.265625, 11.625, 7.84375]}
        ).get_expectation()
        torch.testing.assert_close(
            outputs.logits[0][0, :10].float(), torch.tensor(expected_logits, device=torch_device), atol=1e-5, rtol=1e-5
        )

        expected_boxes = Expectations(
            {("cuda", 9): [[8.960000038146973, 53.76000213623047, 316.1600036621094, 474.239990234375]]}
        ).get_expectation()
        torch.testing.assert_close(results[0]["boxes"], torch.tensor(expected_boxes), atol=1e-5, rtol=1e-5)

        expected_foreground = Expectations({("cuda", 9): 52097}).get_expectation()
        self.assertAlmostEqual(results[0]["masks"].sum().item(), expected_foreground, delta=5e-4 * expected_foreground)

        expected_mask = Expectations(
            {("cuda", 9): [[[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]]}
        ).get_expectation()
        torch.testing.assert_close(
            results[0]["masks"][:, ::96, ::128], torch.tensor(expected_mask, dtype=torch.bool, device=torch_device)
        )

    def test_grounding_generation_with_reasoning(self):
        model = XiaomiLens0ForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device
        ).eval()

        messages = self.get_messages("What color is the blanket the cats are lying on?")
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
                output_hidden_states=[model.config.text_config.num_hidden_layers - 1],
            )
            hidden_states = torch.cat([step[-1] for step in outputs.hidden_states], dim=1)
            mask_logits, mask_token_indices = model.decode_masks(
                outputs.sequences,
                hidden_states,
                outputs.image_last_hidden_state,
                inputs.image_grid_thw,
                ignore_thinking=True,
            )

        results = self.processor.post_process_grounding(
            outputs.sequences, inputs.image_grid_thw, mask_logits, mask_token_indices, target_sizes=[(480, 640)]
        )
        self.assertEqual(results[0]["masks"].shape, (1, 480, 640))

        expected_logits = Expectations(
            {("cuda", 9): [6.28125, 9.9375, 2.84375, 6.40625, 3.078125, 3.5625, 7.1875, 2.703125, 7.5625, 7.46875]}
        ).get_expectation()
        torch.testing.assert_close(
            outputs.logits[0][0, :10].float(), torch.tensor(expected_logits, device=torch_device), atol=1e-5, rtol=1e-5
        )

        expected_boxes = Expectations(
            {("cuda", 9): [[0.0, 119.04000091552734, 640.0, 479.0400085449219]]}
        ).get_expectation()
        torch.testing.assert_close(results[0]["boxes"], torch.tensor(expected_boxes), atol=1e-5, rtol=1e-5)

        expected_foreground = Expectations({("cuda", 9): 132000}).get_expectation()
        self.assertAlmostEqual(results[0]["masks"].sum().item(), expected_foreground, delta=5e-4 * expected_foreground)

        expected_mask = Expectations(
            {("cuda", 9): [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 0, 1, 0, 0], [1, 1, 1, 1, 1]]]}
        ).get_expectation()
        torch.testing.assert_close(
            results[0]["masks"][:, ::96, ::128], torch.tensor(expected_mask, dtype=torch.bool, device=torch_device)
        )
