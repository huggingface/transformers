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
            "image_grids": torch.tensor([[4, 4, 4, 4]] * self.batch_size, device=torch_device),
            "image_num_crops": torch.ones(self.batch_size, dtype=torch.long, device=torch_device),
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
        config.output_attentions = True

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            outputs = model(**inputs_dict)

            output = outputs[0]

            # Encoder-/Decoder-only models
            hidden_states = outputs.hidden_states[0]
            attentions = outputs.attentions[0]

            hidden_states.retain_grad()
            attentions.retain_grad()

            output.flatten()[0].backward(retain_graph=True)

            self.assertIsNotNone(hidden_states.grad)
            self.assertIsNotNone(attentions.grad)


@slow
@require_torch
@require_vision
class Molmo2IntegrationTest(unittest.TestCase):
    model_id = "allenai/Molmo2-4B"
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

    def setUp(self):
        self.processor = Molmo2Processor.from_pretrained(self.model_id)
        self.image = Image.open(requests.get(self.image_url, stream=True).raw)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_preprocessing(self):
        """Test that preprocessing produces expected shapes and values."""
        prompt = "<|image|>Describe this image."
        inputs = self.processor(images=self.image, text=prompt, return_tensors="pt")

        # Check output keys
        self.assertIn("input_ids", inputs)
        self.assertIn("pixel_values", inputs)
        self.assertIn("image_token_pooling", inputs)
        self.assertIn("image_grids", inputs)
        self.assertIn("image_num_crops", inputs)
        self.assertIn("token_type_ids", inputs)

        # Check shapes
        self.assertEqual(inputs["pixel_values"].shape, torch.Size([7, 729, 588]))
        self.assertEqual(inputs["image_token_pooling"].shape, torch.Size([955, 4]))
        self.assertEqual(inputs["image_grids"].shape, torch.Size([1, 4]))
        self.assertEqual(inputs["input_ids"].shape[0], 1)
        self.assertEqual(inputs["input_ids"].shape[1], 987)

        # Check pixel_values slice (preprocessing correctness)
        expected_pixel_slice = torch.tensor(
            [
                [-0.0745098, -0.05098039, 0.0196079],
                [-0.7019608, -0.6784314, -0.60784316],
                [-0.8745098, -0.88235295, -0.84313726],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            inputs["pixel_values"][0, :3, :3],
            expected_pixel_slice,
            atol=1e-4,
            rtol=1e-4,
        )

        # Check input_ids: BOS token, then image start token, then image patches, ending with text tokens
        input_ids = inputs["input_ids"][0]
        self.assertEqual(input_ids[0].item(), 151645)  # BOS token
        self.assertEqual(input_ids[1].item(), 151940)  # low_res_image_start token
        # Last tokens should be the text "Describe this image."
        EXPECTED_TAIL_IDS = [151939, 151937, 74785, 419, 2168, 13]  # <im_end> <im_start> Describe this image.
        self.assertEqual(input_ids[-6:].tolist(), EXPECTED_TAIL_IDS)

    def test_forward_logits(self):
        """Test that forward pass produces expected logits."""
        prompt = "<|image|>Describe this image."
        inputs = self.processor(images=self.image, text=prompt, return_tensors="pt")

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

        # Check logits shape: [batch=1, seq_len=987, vocab_size=151936]
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], 987)

        # Check logits at last position (first 10 vocab tokens)
        expected_last_logits = torch.tensor(
            [
                -10.781937,
                -10.9183,
                -10.77226,
                -10.607452,
                -11.623884,
                -14.052853,
                -11.137567,
                -9.903504,
                -9.405103,
                -13.061548,
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            logits[0, -1, :10].cpu().float(),
            expected_last_logits,
            atol=1e-2,
            rtol=1e-2,
        )

        # Check argmax at last position
        self.assertEqual(logits[0, -1].argmax().item(), 11379)

    def test_generation(self):
        """Test that generation produces non-empty output."""
        prompt = "<|image|>Describe this image."
        inputs = self.processor(images=self.image, text=prompt, return_tensors="pt")

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**device_inputs, max_new_tokens=20)

        # Generated sequence should be longer than input
        self.assertGreater(generated_ids.shape[1], device_inputs["input_ids"].shape[1])

        # Decode and check non-empty
        input_len = device_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        self.assertGreater(len(generated_text.strip()), 0)


@slow
@require_torch
@require_vision
class Molmo2O7BIntegrationTest(unittest.TestCase):
    model_id = "allenai/Molmo2-O-7B"
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

    def setUp(self):
        self.processor = Molmo2Processor.from_pretrained(self.model_id)
        self.image = Image.open(requests.get(self.image_url, stream=True).raw)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_preprocessing(self):
        """Test that preprocessing produces expected shapes and values for Molmo2-O-7B."""
        prompt = "<|image|>Describe this image."
        inputs = self.processor(images=self.image, text=prompt, return_tensors="pt")

        # Same image produces same pixel_values regardless of model variant
        self.assertEqual(inputs["pixel_values"].shape, torch.Size([7, 729, 588]))
        self.assertEqual(inputs["input_ids"].shape[1], 987)

        # Molmo2-O-7B uses a different tokenizer (OLMo-based, vocab_size ~100k)
        EXPECTED_TAIL_IDS = [100281, 100279, 75885, 420, 2217, 13]
        self.assertEqual(inputs["input_ids"][0, -6:].tolist(), EXPECTED_TAIL_IDS)

    def test_forward_logits(self):
        """Test forward pass logits for Molmo2-O-7B."""
        prompt = "<|image|>Describe this image."
        inputs = self.processor(images=self.image, text=prompt, return_tensors="pt")

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

        # Molmo2-O-7B has vocab_size=100278
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], 987)

        expected_last_logits = torch.tensor(
            [
                -18.260553,
                -19.018972,
                -18.696802,
                -18.284496,
                -16.284964,
                -19.856026,
                -19.706102,
                -20.052923,
                -17.303316,
                -21.92196,
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            logits[0, -1, :10].cpu().float(),
            expected_last_logits,
            atol=1e-2,
            rtol=1e-2,
        )

        self.assertEqual(logits[0, -1].argmax().item(), 578)

    def test_generation(self):
        """Test generation for Molmo2-O-7B."""
        prompt = "<|image|>Describe this image."
        inputs = self.processor(images=self.image, text=prompt, return_tensors="pt")

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**device_inputs, max_new_tokens=20, do_sample=False)

        self.assertGreater(generated_ids.shape[1], device_inputs["input_ids"].shape[1])

        input_len = device_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        self.assertGreater(len(generated_text.strip()), 0)


@slow
@require_torch
@require_vision
class Molmo2_8BIntegrationTest(unittest.TestCase):
    model_id = "allenai/Molmo2-8B"
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

    def setUp(self):
        self.processor = Molmo2Processor.from_pretrained(self.model_id)
        self.image = Image.open(requests.get(self.image_url, stream=True).raw)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_preprocessing(self):
        """Test that preprocessing produces expected shapes and values for Molmo2-8B."""
        prompt = "<|image|>Describe this image."
        inputs = self.processor(images=self.image, text=prompt, return_tensors="pt")

        self.assertEqual(inputs["pixel_values"].shape, torch.Size([7, 729, 588]))
        self.assertEqual(inputs["input_ids"].shape[1], 987)

        # Molmo2-8B uses the same tokenizer as Molmo2-4B (Qwen-based, vocab_size ~152k)
        EXPECTED_TAIL_IDS = [151939, 151937, 74785, 419, 2168, 13]
        self.assertEqual(inputs["input_ids"][0, -6:].tolist(), EXPECTED_TAIL_IDS)

    def test_forward_logits(self):
        """Test forward pass logits for Molmo2-8B."""
        prompt = "<|image|>Describe this image."
        inputs = self.processor(images=self.image, text=prompt, return_tensors="pt")

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
        self.assertEqual(logits.shape[1], 987)

        expected_last_logits = torch.tensor(
            [
                -19.064266,
                -21.253227,
                -20.791862,
                -19.417578,
                -16.480974,
                -20.062803,
                -20.178888,
                -19.560125,
                -17.375803,
                -21.136972,
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            logits[0, -1, :10].cpu().float(),
            expected_last_logits,
            atol=1e-2,
            rtol=1e-2,
        )

        self.assertEqual(logits[0, -1].argmax().item(), 25244)

    def test_generation(self):
        """Test generation produces expected text for Molmo2-8B."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in exactly 1 short sentence."},
                    {"type": "image", "image": self.image},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**device_inputs, max_new_tokens=30, do_sample=False)

        input_len = device_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        EXPECTED_TEXT = "A snow leopard is captured mid-stride in a snowy landscape, its thick fur dusted with snow as it moves gracefully through its natural habitat."  # fmt: skip
        self.assertEqual(generated_text.strip(), EXPECTED_TEXT)

    def test_generation_video_qa(self):
        """Test video question answering for Molmo2-8B."""
        video_url = "https://storage.googleapis.com/oe-training-public/demo_videos/many_penguins.mp4"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Which animal appears in the video?"},
                    {"type": "video", "video": video_url},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**device_inputs, max_new_tokens=100, do_sample=False)

        input_len = device_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        EXPECTED_TEXT = "Penguins appear in the video."
        self.assertEqual(generated_text.strip(), EXPECTED_TEXT)

    def test_generation_video_pointing(self):
        """Test video pointing for Molmo2-8B."""
        video_url = "https://storage.googleapis.com/oe-training-public/demo_videos/many_penguins.mp4"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Point to the penguins."},
                    {"type": "video", "video": video_url},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**device_inputs, max_new_tokens=2048, do_sample=False)

        input_len = device_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        # Should contain pointing coordinates
        self.assertIn("<points", generated_text)
        self.assertIn("penguins", generated_text.lower())

    def test_generation_multi_image(self):
        """Test multi-image question answering for Molmo2-8B."""
        image_urls = [
            "https://picsum.photos/id/237/536/354",
            "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg",
        ]
        images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images."},
                    *[{"type": "image", "image": img} for img in images],
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**device_inputs, max_new_tokens=2048, do_sample=False)

        input_len = device_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        # Should produce a comparison mentioning both images
        self.assertIn("first image", generated_text.lower())
        self.assertIn("second image", generated_text.lower())
