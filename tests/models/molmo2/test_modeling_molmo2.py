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
"""Testing suite for the PyTorch Molmo2 model."""

import copy
import unittest

import requests

from transformers import (
    Molmo2Config,
    Molmo2ForConditionalGeneration,
    Molmo2Model,
    Molmo2Processor,
    is_torch_available,
    is_vision_available,
)
from transformers.models.molmo2.configuration_molmo2 import (
    Molmo2AdapterConfig,
    Molmo2TextConfig,
    Molmo2VitConfig,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.video_utils import load_video

from ...test_modeling_common import (
    _config_zero_init,
    floats_tensor,
)
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class Molmo2VisionText2TextModelTester(VLMModelTester):
    base_model_class = Molmo2Model
    config_class = Molmo2Config
    text_config_class = Molmo2TextConfig
    vision_config_class = Molmo2VitConfig
    conditional_generation_class = Molmo2ForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_size", 378)
        kwargs.setdefault("patch_size", 14)
        kwargs.setdefault("num_image_tokens", 32)
        kwargs.setdefault("seq_length", 7 + kwargs["num_image_tokens"])
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("intermediate_size", 37)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 128)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("max_position_embeddings", 512)
        kwargs.setdefault("bos_token_id", 0)
        kwargs.setdefault("eos_token_id", 1)
        kwargs.setdefault("pad_token_id", 2)
        kwargs.setdefault("image_start_token_id", 3)
        kwargs.setdefault("image_end_token_id", 4)
        kwargs.setdefault("image_patch_id", 5)
        kwargs.setdefault("image_col_id", 6)
        # Alias so base helpers (special-token clearing, mismatch tests) protect image patch tokens.
        kwargs.setdefault("image_token_id", kwargs["image_patch_id"])
        super().__init__(parent, **kwargs)

    def create_pixel_values(self):
        # Molmo2 expects flattened patches: (batch, num_crops, n_patches, pixels_per_patch).
        num_patches = (self.image_size // self.patch_size) ** 2
        return floats_tensor(
            [
                self.batch_size,
                1,
                num_patches,
                self.patch_size * self.patch_size * self.num_channels,
            ]
        )

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.image_patch_id] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = self.image_patch_id
        return input_ids

    def create_attention_mask(self, input_ids):
        # Molmo2 expects a standard 2D padding mask of ones, not the base's tril matrix.
        return torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

    def get_additional_inputs(self, config, input_ids, pixel_values):
        num_patches = (self.image_size // self.patch_size) ** 2
        # Mark image-patch positions; required by the training-mode mask path.
        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[input_ids == self.image_patch_id] = 1
        return {
            "image_token_pooling": torch.randint(
                -1,
                num_patches,
                (self.batch_size, self.num_image_tokens, 4),
                device=torch_device,
            ),
            "image_grids": torch.tensor([[[4, 4, 4, 4]]] * self.batch_size, device=torch_device),
            "image_num_crops": torch.ones(self.batch_size, 1, dtype=torch.long, device=torch_device),
            "token_type_ids": token_type_ids,
        }

    def get_config(self):
        text_config = Molmo2TextConfig(
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            hidden_act=self.hidden_act,
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            num_key_value_heads=self.num_key_value_heads,
            rope_theta=10000.0,
            tie_word_embeddings=self.tie_word_embeddings,
            layer_norm_eps=1e-6,
        )
        vit_config = Molmo2VitConfig(
            hidden_size=32,
            intermediate_size=37,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            hidden_act="gelu_pytorch_tanh",
            layer_norm_eps=1e-6,
            image_default_input_size=[self.image_size, self.image_size],
            image_patch_size=self.patch_size,
            image_num_pos=(self.image_size // self.patch_size) ** 2,
            attention_dropout=0.0,
            residual_dropout=0.0,
        )
        adapter_config = Molmo2AdapterConfig(
            vit_layers=[-1],
            pooling_attention_mask=False,
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            intermediate_size=37,
            text_hidden_size=32,
            hidden_act="silu",
        )
        return Molmo2Config(
            text_config=text_config,
            vit_config=vit_config,
            adapter_config=adapter_config,
            image_start_token_id=self.image_start_token_id,
            image_end_token_id=self.image_end_token_id,
            image_patch_id=self.image_patch_id,
            image_col_id=self.image_col_id,
            tie_word_embeddings=self.tie_word_embeddings,
        )


@require_torch
class Molmo2ModelTest(VLMModelTest, unittest.TestCase):
    """
    Model tester for `Molmo2ForConditionalGeneration`.
    """

    model_tester_class = Molmo2VisionText2TextModelTester
    pipeline_model_mapping = (
        {
            "image-to-text": Molmo2ForConditionalGeneration,
            "image-text-to-text": Molmo2ForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    test_torchscript = False
    test_pruning = False
    test_head_masking = False
    skip_test_image_features_output_shape = True
    skip_test_video_features_output_shape = True

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        config, inputs_dict = config_and_inputs
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            with torch.no_grad():
                _ = model(**inputs_dict)

    # overwrite inputs_embeds tests because we need to delete "pixel_values" for VLMs
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]
            del inputs["image_token_pooling"]
            del inputs["image_grids"]
            del inputs["image_num_crops"]

            wte = model.get_input_embeddings()
            inputs["inputs_embeds"] = wte(input_ids)

            with torch.no_grad():
                model(**inputs)

    # overwrite inputs_embeds tests because we need to delete "pixel_values" for VLMs
    def test_inputs_embeds_matches_input_ids(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)
            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]
            del inputs["image_token_pooling"]
            del inputs["image_grids"]
            del inputs["image_num_crops"]

            inputs_embeds = model.get_input_embeddings()(input_ids)

            with torch.no_grad():
                out_ids = model(input_ids=input_ids, **inputs)[0]
                out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            self.assertTrue(torch.allclose(out_embeds, out_ids))

    @unittest.skip(
        reason="This architecture does not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture does not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture does not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(
        reason="This architecture does not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    @unittest.skip(reason="VLMs have dynamic control flow in preparing inputs for generation")
    def test_generate_compile_1_end_to_end(self):
        pass

    @unittest.skip(reason="Molmo2 weights are not tied.")
    def test_tied_weights_keys(self):
        pass

    @unittest.skip(reason="Molmo2 uses a custom Molmo2Embedding class instead of nn.Embedding")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Molmo2 uses a custom Molmo2Embedding class that does not support standard resize")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Molmo2 uses a custom Molmo2Embedding class that does not support standard resize")
    def test_resize_embeddings_untied(self):
        pass

    @unittest.skip(
        reason="Molmo2 interleaves visual features in text hidden states, causing shape mismatches in equivalence checks"
    )
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @unittest.skip(
        reason="Supported only for text-only inputs (otherwise dynamic control flows for multimodal inputs)"
    )
    def test_generate_compile_model_forward(self):
        pass

    @unittest.skip("Molmo2 builds vision-aware embeddings; text-only get_input_embeddings bypasses image placement.")
    def test_generate_from_inputs_embeds_0_greedy(self):
        pass

    @unittest.skip("Molmo2 builds vision-aware embeddings; text-only get_input_embeddings bypasses image placement.")
    def test_generate_from_inputs_embeds_1_beam_search(self):
        pass

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad and "class_embedding" not in name:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )
                if "class_embedding" in name:
                    self.assertTrue(
                        -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs handle single-batch image inputs correctly.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            _ = model(**input_dict)  # successful forward with no modifications
            curr_input_dict = copy.deepcopy(input_dict)

            # Reduce to single batch item (all inputs sliced consistently)
            curr_input_dict["input_ids"] = curr_input_dict["input_ids"][:1, ...]
            curr_input_dict["attention_mask"] = curr_input_dict["attention_mask"][:1, ...]
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][:1, ...]
            curr_input_dict["image_token_pooling"] = curr_input_dict["image_token_pooling"][:1, ...]
            curr_input_dict["image_grids"] = curr_input_dict["image_grids"][:1, ...]
            curr_input_dict["image_num_crops"] = curr_input_dict["image_num_crops"][:1, ...]
            _ = model(**curr_input_dict)

    # Image features get cached in KV cache like other VLMs; no need to skip.

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = self.has_attentions
        self._set_subconfig_attributes(config, "output_hidden_states", True)
        self._set_subconfig_attributes(config, "output_attentions", self.has_attentions)

        for model_class in self.all_model_classes:
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device)
            outputs = model(**inputs_dict)

            output = outputs[0]
            hidden_states = outputs.hidden_states[0]
            hidden_states.retain_grad()

            if self.has_attentions:
                attentions = outputs.attentions[0]
                attentions.retain_grad()

            output.flatten()[0].backward(retain_graph=True)

            self.assertIsNotNone(hidden_states.grad)
            if self.has_attentions:
                self.assertIsNotNone(attentions.grad)


IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"


@slow
@require_torch
@require_vision
class Molmo2IntegrationTest(unittest.TestCase):
    model_id = "allenai/Molmo2-4B"

    def setUp(self):
        self.processor = Molmo2Processor.from_pretrained(self.model_id)
        self.image = Image.open(requests.get(IMAGE_URL, stream=True).raw)
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image", "image": self.image},
                ],
            }
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def build_inputs(self):
        return self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    def test_preprocessing(self):
        inputs = self.build_inputs()

        for key in ("input_ids", "pixel_values", "image_token_pooling", "image_grids", "image_num_crops", "token_type_ids"):
            self.assertIn(key, inputs)

        self.assertEqual(inputs["pixel_values"].shape, torch.Size([1, 7, 729, 588]))
        self.assertEqual(inputs["image_token_pooling"].shape, torch.Size([1, 955, 4]))
        self.assertEqual(inputs["image_grids"].shape, torch.Size([1, 1, 4]))
        self.assertEqual(inputs["input_ids"].shape[0], 1)

        expected_pixel_slice = torch.tensor(
            [
                [-0.0745098, -0.05098039, 0.0196079],
                [-0.7019608, -0.6784314, -0.60784316],
                [-0.8745098, -0.88235295, -0.84313726],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            inputs["pixel_values"][0, 0, :3, :3],
            expected_pixel_slice,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_forward_logits(self):
        inputs = self.build_inputs()

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**device_inputs)

        logits = outputs.logits
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], device_inputs["input_ids"].shape[1])

        expected_last_logits = torch.tensor(
            [
                -10.443631,
                -6.061066,
                -11.089376,
                -10.382135,
                -16.847486,
                -14.479667,
                -11.206448,
                -9.839166,
                -11.699806,
                -9.23956,
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            logits[0, -1, :10].cpu().float(),
            expected_last_logits,
            atol=1e-2,
            rtol=1e-2,
        )
        self.assertEqual(logits[0, -1].argmax().item(), 641)

    def test_generation(self):
        inputs = self.build_inputs()

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**device_inputs, max_new_tokens=10, do_sample=False)

        input_len = device_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        EXPECTED_TEXT = "In this captivating image, a large, chubby cat"  # fmt: skip
        self.assertEqual(generated_text.strip(), EXPECTED_TEXT)


@slow
@require_torch
@require_vision
class Molmo2O7BIntegrationTest(unittest.TestCase):
    model_id = "allenai/Molmo2-O-7B"

    def setUp(self):
        self.processor = Molmo2Processor.from_pretrained(self.model_id)
        self.image = Image.open(requests.get(IMAGE_URL, stream=True).raw)
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image", "image": self.image},
                ],
            }
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def build_inputs(self):
        return self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    def test_preprocessing(self):
        inputs = self.build_inputs()
        self.assertEqual(inputs["pixel_values"].shape, torch.Size([1, 7, 729, 588]))
        self.assertEqual(inputs["input_ids"].shape[0], 1)

    def test_forward_logits(self):
        inputs = self.build_inputs()

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**device_inputs)

        logits = outputs.logits
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], device_inputs["input_ids"].shape[1])

        expected_last_logits = torch.tensor(
            [-13.0625, -5.875, -11.6875, -11.0, -12.6875, -16.25, -10.3125, -12.25, -12.6875, -10.625],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            logits[0, -1, :10].cpu().float(),
            expected_last_logits,
            atol=3e-1,
            rtol=5e-2,
        )
        self.assertEqual(logits[0, -1].argmax().item(), 644)

    def test_generation(self):
        inputs = self.build_inputs()

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**device_inputs, max_new_tokens=10, do_sample=False)

        input_len = device_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        EXPECTED_TEXT = "In this captivating image, a small, chubby cat"  # fmt: skip
        self.assertEqual(generated_text.strip(), EXPECTED_TEXT)


@slow
@require_torch
@require_vision
class Molmo2_8BIntegrationTest(unittest.TestCase):
    model_id = "allenai/Molmo2-8B"

    def setUp(self):
        self.processor = Molmo2Processor.from_pretrained(self.model_id)
        self.image = Image.open(requests.get(IMAGE_URL, stream=True).raw)
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image", "image": self.image},
                ],
            }
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def build_inputs(self):
        return self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    def test_preprocessing(self):
        inputs = self.build_inputs()
        self.assertEqual(inputs["pixel_values"].shape, torch.Size([1, 7, 729, 588]))
        self.assertEqual(inputs["input_ids"].shape[0], 1)

    def test_forward_logits(self):
        inputs = self.build_inputs()

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**device_inputs)

        logits = outputs.logits
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], device_inputs["input_ids"].shape[1])

        expected_last_logits = torch.tensor(
            [-15.8125, -7.875, -15.5625, -14.9375, -16.5, -18.25, -14.4375, -15.6875, -15.375, -12.4375],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            logits[0, -1, :10].cpu().float(),
            expected_last_logits,
            atol=3e-1,
            rtol=5e-2,
        )
        self.assertEqual(logits[0, -1].argmax().item(), 641)

    def test_generation(self):
        inputs = self.build_inputs()

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**device_inputs, max_new_tokens=10, do_sample=False)

        input_len = device_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        EXPECTED_TEXT = "In this captivating image, a snow leopard is captured"  # fmt: skip
        self.assertEqual(generated_text.strip(), EXPECTED_TEXT)

    def test_generation_video_qa(self):
        """Test video question answering for Molmo2-8B."""
        video_url = "https://storage.googleapis.com/oe-training-public/demo_videos/many_penguins.mp4"
        video, metadata = load_video(video_url)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Which animal appears in the video?"},
                    {"type": "video", "video": video},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            processor_kwargs={"video_metadata": [metadata]},
        )

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**device_inputs, max_new_tokens=64, do_sample=False)

        input_len = device_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        EXPECTED_TEXT = "Penguins appear in the video."
        self.assertEqual(generated_text.strip(), EXPECTED_TEXT)
