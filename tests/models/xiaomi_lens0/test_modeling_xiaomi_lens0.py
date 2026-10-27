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
from transformers.testing_utils import Expectations, require_torch, require_torch_accelerator, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch

    from transformers import XiaomiLens0Config, XiaomiLens0ForConditionalGeneration, XiaomiLens0Model


class XiaomiLens0VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        image_size=32,
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

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        temporal_patch_size = config.vision_config.temporal_patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size * self.grid_size**2,
                self.num_channels * (patch_size**2) * temporal_patch_size,
            ]
        )
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        special_token_ids = [
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
            self.mask_token_id,
        ]
        input_ids[torch.isin(input_ids, torch.tensor(special_token_ids, device=input_ids.device))] = self.pad_token_id
        input_ids[:, -1] = self.pad_token_id
        input_ids[:, 0] = self.vision_start_token_id
        input_ids[:, 1 : self.num_image_tokens + 1] = self.image_token_id
        input_ids[:, self.num_image_tokens + 1] = self.vision_end_token_id

        mm_token_type_ids = (input_ids == self.image_token_id).long()

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor(
                [[1, self.grid_size, self.grid_size]] * self.batch_size, device=torch_device
            ),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
        }
        return config, inputs_dict


@require_torch
class XiaomiLens0ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (XiaomiLens0Model, XiaomiLens0ForConditionalGeneration) if is_torch_available() else ()
    model_split_percents = [0.5, 0.8, 0.9]
    # The mask decoder only receives gradients from a mask loss, which is not part of the LM loss
    test_all_params_have_gradient = False

    def setUp(self):
        self.model_tester = XiaomiLens0VisionText2TextModelTester(self)

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
        "Needs to be overwritten as Xiaomi-Lens-0 alternates between attention layers and gated deltanet layers."
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

    @unittest.skip("The mask decoder is not part of `XiaomiLens0Model`, the base model of the checkpoint mapping")
    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        pass


@slow
@require_torch_accelerator
class XiaomiLens0IntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "SeerRay-Lab/Xiaomi-Lens-0"
    image_url = (
        "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"
    )
    grounding_suffix = "\nReport bbox coordinates and masks."

    def setUp(self):
        super().setUp()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def prepare_inputs(self, text, completion=None):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": url_to_local_path(self.image_url)},
                    {"type": "text", "text": text},
                ],
            }
        ]
        if completion is not None:
            messages.append({"role": "assistant", "content": completion})
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=completion is None,
            return_dict=True,
            return_tensors="pt",
        ).to(torch_device)

    def forward_and_decode_masks(self, model, inputs):
        inputs = dict(inputs)
        image_features = model.get_image_features(
            inputs.pop("pixel_values"), inputs["image_grid_thw"], return_dict=True
        )
        outputs = model(**inputs, mm_encoder_outputs={"image": image_features}, output_hidden_states=True)
        mask_logits, mask_token_indices = model.decode_masks(
            inputs["input_ids"],
            outputs.hidden_states[-1],
            image_features.last_hidden_state,
            inputs["image_grid_thw"],
        )
        return outputs, mask_logits, mask_token_indices

    def test_model_logits(self):
        # fp32 as bf16 logits differ by a few ulps from the original, too much for a meaningful comparison
        model = XiaomiLens0ForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.float32, device_map=torch_device
        )
        text = "How many cats are there?" + self.grounding_suffix
        completion = (
            "\n</think>\n\n<bbox>[541, 52, 1000, 775]</bbox><mask><|mask|></mask>"
            "<bbox>[14, 115, 494, 988]</bbox><mask><|mask|></mask>"
        )
        inputs = self.prepare_inputs(text, completion)
        outputs, mask_logits, _ = self.forward_and_decode_masks(model, inputs)

        prompt_length = self.prepare_inputs(text).input_ids.shape[1]
        expected_logits = Expectations(
            {
                ("cuda", 8): [5.1540, 6.4170, 7.1199, 6.4834, 1.0022, 3.2335, 4.3166, 3.6997, 5.5916, 5.8586],
            }
        ).get_expectation()
        torch.testing.assert_close(
            outputs.logits[0, prompt_length - 1, :10].cpu(), torch.tensor(expected_logits), atol=1e-3, rtol=1e-3
        )

        self.assertEqual(len(mask_logits), 1)
        self.assertEqual(mask_logits[0].shape, (2, 480, 640))
        # Mask logits inside each predicted cat mask, and the number of foreground pixels
        expected_mask_slices = Expectations(
            {
                ("cuda", 8): [[6.156, 6.09, 6.023, 5.957, 5.887], [8.711, 8.75, 8.797, 8.836, 8.844]],
            }
        ).get_expectation()
        mask_slices = torch.stack([mask_logits[0][0, 198, 493:498], mask_logits[0][1, 228, 146:151]])
        torch.testing.assert_close(mask_slices.cpu(), torch.tensor(expected_mask_slices), atol=5e-2, rtol=5e-2)
        expected_foreground = Expectations({("cuda", 8): [59138, 51824]}).get_expectation()
        foreground = (mask_logits[0] > 0).sum(dim=(-1, -2)).tolist()
        for pixels, expected_pixels in zip(foreground, expected_foreground):
            self.assertAlmostEqual(pixels, expected_pixels, delta=0.005 * expected_pixels)

    def test_model_vision_generation(self):
        model = XiaomiLens0ForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device
        )
        text = (
            'Locate all instances in the image that match the description "the cat on the left".'
            + self.grounding_suffix
        )
        inputs = self.prepare_inputs(text)
        generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        completion = self.processor.decode(generated_ids[0, inputs.input_ids.shape[1] :])

        expected_completion = Expectations(
            {
                ("cuda", 8): "\n</think>\n\n<bbox>[14, 115, 494, 988]</bbox><mask><|mask|></mask><|im_end|>",
            }
        ).get_expectation()
        self.assertEqual(completion, expected_completion)

        mask_inputs = self.prepare_inputs(text, completion.removesuffix("<|im_end|>"))
        _, mask_logits, mask_token_indices = self.forward_and_decode_masks(model, mask_inputs)
        results = self.processor.post_process_grounding(
            mask_inputs.input_ids,
            mask_inputs.image_grid_thw,
            mask_logits,
            mask_token_indices,
            target_sizes=[(480, 640)],
        )
        torch.testing.assert_close(results[0]["boxes"], torch.tensor([[8.96, 55.2, 316.16, 474.24]]))
        self.assertEqual(results[0]["masks"].shape, (1, 480, 640))
        expected_foreground = Expectations({("cuda", 8): 51985}).get_expectation()
        self.assertAlmostEqual(results[0]["masks"].sum().item(), expected_foreground, delta=0.01 * expected_foreground)

    def test_model_vision_generation_with_reasoning(self):
        model = XiaomiLens0ForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=torch_device
        )
        inputs = self.prepare_inputs("What color is the blanket the cats are lying on?")
        generated_ids = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        completion = self.processor.decode(generated_ids[0, inputs.input_ids.shape[1] :])

        expected_completion = Expectations(
            {
                ("cuda", 8): (
                    "The user wants to know the color of the blanket. I need to look at the image and identify the "
                    "blanket covering the couch where the cats are lying. The blanket is a bright pink color, covering "
                    "most of the lower portion of the image. I'll localize the blanket region to support my answer.\n"
                    "</think>\n\nThe blanket covering the couch where the cats are resting is a vivid pink color, as "
                    "seen across the lower portion of the image <bbox>[0, 248, 1000, 998]</bbox><mask><|mask|></mask>."
                    "\n\nPink<|im_end|>"
                ),
            }
        ).get_expectation()
        self.assertEqual(completion, expected_completion)
