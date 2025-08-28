# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Janus model."""

import re
import tempfile
import unittest
from functools import reduce

import numpy as np
import requests

from transformers import (
    AutoProcessor,
    JanusConfig,
    JanusForConditionalGeneration,
    JanusModel,
    JanusVQVAE,
    JanusVQVAEConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import MODEL_FOR_BACKBONE_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.testing_utils import (
    Expectations,
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


class JanusVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        image_token_index=0,
        seq_length=25,
        initializer_range=0.02,
        text_config={
            "model_type": "llama",
            "seq_length": 7,
            "is_training": True,
            "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 1,
        },
        is_training=True,
        vision_config={
            "use_labels": True,
            "image_size": 20,
            "patch_size": 5,
            "num_image_tokens": 4,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "projection_dim": 32,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "mlp_ratio": 2,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
            "vision_feature_select_strategy": "default",
            "vision_feature_layer": -1,
        },
        use_cache=False,
        vq_num_embeds=12,
        vq_embed_dim=12,
        vq_channel_multiplier=[1, 1],
    ):
        self.parent = parent
        self.initializer_range = initializer_range
        # `image_token_index` is set to 0 to pass "resize_embeddings" test, do not modify
        self.image_token_index = image_token_index
        self.text_config = text_config
        self.vision_config = vision_config
        self.seq_length = seq_length
        self.pad_token_id = text_config["pad_token_id"]

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = vision_config["num_channels"]
        self.image_size = vision_config["image_size"]
        self.num_image_tokens = vision_config["num_image_tokens"]
        self.use_cache = use_cache

        # vq model params
        self.vq_num_embeds = vq_num_embeds
        self.vq_embed_dim = vq_embed_dim
        self.vq_channel_multiplier = vq_channel_multiplier

    def get_vq_config(self):
        return {
            "embed_dim": self.vq_embed_dim,
            "num_embeddings": self.vq_num_embeds,
            "latent_channels": self.vq_embed_dim,
            "in_channels": 3,
            "base_channels": 32,  # we have a GroupNorm of 32 groups, so can't do less
            "channel_multiplier": self.vq_channel_multiplier,
            "initializer_range": self.initializer_range,
            "projection_dim": 10,
            "image_token_embed_dim": 32,  # Same as text model hidden size
        }

    def get_config(self):
        return JanusConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            vq_config=self.get_vq_config(),
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = floats_tensor(
            [
                self.batch_size,
                3,
                self.image_size,
                self.image_size,
            ]
        )
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        # set the 16 first tokens to be image, and ensure that no other tokens are image tokens
        # do not change this unless you modified image size or patch size
        input_ids[input_ids == self.image_token_index] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = self.image_token_index
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
            "generation_mode": "text",  # Required to perform text generation instead of image generation.
        }
        return config, inputs_dict


@require_torch
class JanusVisionText2TextModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (JanusModel, JanusForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (JanusForConditionalGeneration,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = JanusVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=JanusConfig, has_text_modality=False)

    # overwrite inputs_embeds tests because we need to delete "pixel values" for LVLMs
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
            del inputs["generation_mode"]

            wte = model.get_input_embeddings()
            inputs["inputs_embeds"] = wte(input_ids)

            with torch.no_grad():
                model(**inputs)

    # Overwrite inputs_embeds tests because we need to delete "pixel values" for VLMs.
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
            del inputs["generation_mode"]

            inputs_embeds = model.get_input_embeddings()(input_ids)

            with torch.no_grad():
                out_ids = model(input_ids=input_ids, **inputs)[0]
                out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            torch.testing.assert_close(out_embeds, out_ids)

    def test_sdpa_can_dispatch_composite_models(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # Load the model with SDPA
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                # Load model with eager attention
                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    attn_implementation="eager",
                )
                model_eager = model_eager.eval().to(torch_device)

            # SigLip has one shared cls attr for all models, so we assign both submodels heer
            vision_attn = language_attn = "sdpa" if model._supports_sdpa else "eager"

            if hasattr(model_sdpa, "vision_model") and hasattr(model_sdpa, "language_model"):
                self.assertTrue(model_sdpa.vision_model.config._attn_implementation == vision_attn)
                self.assertTrue(model_sdpa.language_model.config._attn_implementation == language_attn)
                self.assertTrue(model_eager.vision_model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.language_model.config._attn_implementation == "eager")

            self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
            self.assertTrue(model_eager.config._attn_implementation == "eager")

            for name, submodule in model_eager.named_modules():
                class_name = submodule.__class__.__name__
                if any(re.finditer(r"Attention(?!Pool)", class_name)):
                    self.assertTrue(submodule.config._attn_implementation == "eager")

            for name, submodule in model_sdpa.named_modules():
                class_name = submodule.__class__.__name__
                if any(re.finditer(r"Attention(?!Pool)", class_name)):
                    self.assertTrue(submodule.config._attn_implementation == "sdpa")

    def check_training_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        if not self.model_tester.is_training:
            self.skipTest(reason="ModelTester is not configured to run training tests")
        """
        We skip some parameters when checking for gradient checkpointing:
        - VQ model, as its training is not supported.
        - A few other modules used for image generation.
        """
        skip_patterns = ["vqmodel", "generation_embeddings", "generation_aligner", "generation_head"]

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                if (
                    model_class.__name__
                    in [
                        *get_values(MODEL_MAPPING_NAMES),
                        *get_values(MODEL_FOR_BACKBONE_MAPPING_NAMES),
                    ]
                    or not model_class.supports_gradient_checkpointing
                ):
                    # TODO (ydshieh): use `skipTest` once pytest-dev/pytest-subtests/pull/169 is merged
                    # self.skipTest(reason=f"`supports_gradient_checkpointing` is False for {model_class.__name__}.")
                    continue

                config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
                config.use_cache = False
                config.return_dict = True
                model = model_class(config)

                model.to(torch_device)
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
                model.train()

                # unfreeze additional layers
                for p in model.parameters():
                    p.requires_grad_(True)

                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

                inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                loss = model(**inputs).loss
                loss.backward()
                optimizer.step()

                if self.test_all_params_have_gradient:
                    for k, v in model.named_parameters():
                        if v.requires_grad and not reduce(lambda t, s: t | (s in k), skip_patterns, False):
                            self.assertTrue(v.grad is not None, f"{k} in {model_class.__name__} has no gradient!")
                        else:
                            pass

    @unittest.skip("There are recompilations in Janus")  # TODO (joao, raushan): fix me
    def test_generate_compile_model_forward(self):
        pass


class JanusVQModelTester:
    def __init__(
        self,
        parent,
        batch_size=5,
        is_training=False,
        initializer_range=0.02,
        image_size=30,
        num_embeds=12,
        base_channels=32,  # we have a GroupNorm of 32 groups, so can't do less
        embed_dim=12,
        channel_multiplier=[1, 2],
        patch_size=2,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.initializer_range = initializer_range
        self.image_size = image_size
        self.base_channels = base_channels
        self.num_embeds = num_embeds
        self.embed_dim = embed_dim
        self.channel_multiplier = channel_multiplier
        self.num_patches = image_size // patch_size

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, 3, self.image_size, self.image_size])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return JanusVQVAEConfig(
            embed_dim=self.embed_dim,
            num_embeddings=self.num_embeds,
            latent_channels=self.embed_dim,
            in_channels=3,
            base_channels=self.base_channels,
            channel_multiplier=self.channel_multiplier,
            initializer_range=self.initializer_range,
            resolution=self.image_size,
            num_patches=self.num_patches,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class JanusVQModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (JanusVQVAE,) if is_torch_available() else ()
    test_head_masking = False
    test_pruning = False
    fx_compatible = False
    has_attentions = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = JanusVQModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=JanusVQVAEConfig,
            has_text_modality=False,
            common_properties=["embed_dim", "num_embeddings"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip("Janus VQ module cannot offload due to using `self.weight` directly")
    def test_cpu_offload(self):
        pass

    @unittest.skip("Janus VQ module cannot offload due to using `self.weight` directly")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip("Janus VQ module cannot offload due to using `self.weight` directly")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip("Janus VQ module has no hidden states")
    def test_hidden_states_output(self):
        pass

    @unittest.skip("Janus VQ module has no hidden states")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip("Janus VQ module has no get/set embeddings method")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip("Janus VQ module has no hidden states")
    def test_retain_grad_hidden_states_attentions(self):
        pass


class JanusIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_id = "deepseek-community/Janus-Pro-1B"

    @slow
    def test_model_text_generation(self):
        model = JanusForConditionalGeneration.from_pretrained(self.model_id, device_map="auto")
        model.eval()
        processor = AutoProcessor.from_pretrained(self.model_id)
        image = Image.open(
            requests.get("https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg", stream=True).raw
        )
        prompt = "<image_placeholder>\nDescribe what do you see here and tell me about the history behind it?"
        inputs = processor(images=image, text=prompt, generation_mode="text", return_tensors="pt").to(model.device)

        output = model.generate(**inputs, max_new_tokens=20, generation_mode="text", do_sample=False)
        EXPECTED_DECODED_TEXT = 'You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\n\nDescribe what do you see here and tell me about the history behind it?\n\nThe image depicts the constellation of Leo, which is often referred to as the "Lion"'  # fmt: skip
        text = processor.decode(output[0], skip_special_tokens=True)
        self.assertEqual(
            text,
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_model_text_generation_batched(self):
        model = JanusForConditionalGeneration.from_pretrained(self.model_id, device_map="auto")
        processor = AutoProcessor.from_pretrained(self.model_id)

        image_1 = Image.open(
            requests.get("https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg", stream=True).raw
        )
        image_2 = Image.open(
            requests.get("https://www.kxan.com/wp-content/uploads/sites/40/2020/10/ORION.jpg", stream=True).raw
        )
        prompts = [
            "<image_placeholder>\nDescribe what do you see here and tell me about the history behind it?",
            "What constellation is this image showing?<image_placeholder>\n",
        ]

        inputs = processor(
            images=[image_1, image_2], text=prompts, generation_mode="text", padding=True, return_tensors="pt"
        ).to(model.device, torch.float16)

        EXPECTED_TEXT_COMPLETION = [
            'You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\n\nDescribe what do you see here and tell me about the history behind it?\n\nThe image depicts the constellation of Leo, which is often referred to as the "Lion"',
            "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\nWhat constellation is this image showing?\n\nThe image shows a constellation that is shaped like a stylized figure with a long tail. This",
        ]
        generated_ids = model.generate(**inputs, max_new_tokens=20, generation_mode="text", do_sample=False)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    def test_model_text_generation_with_multi_image(self):
        model = JanusForConditionalGeneration.from_pretrained(self.model_id, device_map="auto")
        processor = AutoProcessor.from_pretrained(self.model_id)

        image_1 = Image.open(
            requests.get("https://nineplanets.org/wp-content/uploads/2020/12/the-big-dipper-1.jpg", stream=True).raw
        )
        image_2 = Image.open(
            requests.get("https://www.kxan.com/wp-content/uploads/sites/40/2020/10/ORION.jpg", stream=True).raw
        )
        prompt = "What do these two images <image_placeholder> and <image_placeholder> have in common?"

        inputs = processor(images=[image_1, image_2], text=prompt, generation_mode="text", return_tensors="pt").to(
            model.device, torch.float16
        )

        EXPECTED_TEXT_COMPLETION = ['You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\nWhat do these two images  and  have in common?\n\nThe two images you provided are of the same constellation. The first image shows the constellation of Leo, and the second image shows the constellation of Ursa Major. Both constellations are part of']  # fmt: skip
        generated_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    def test_model_generate_images(self):
        model = JanusForConditionalGeneration.from_pretrained(self.model_id, device_map="auto")
        processor = AutoProcessor.from_pretrained(self.model_id)

        inputs = processor(
            text=["A portrait of young girl. masterpiece, film grained, best quality."],
            padding=True,
            generation_mode="image",
            return_tensors="pt",
        ).to(model.device)

        self.assertTrue(inputs.input_ids.shape[1] == 17)

        out = model.generate(
            **inputs,
            generation_mode="image",
            do_sample=False,
        )

        # It should run for num_image_tokens in this case 576.
        self.assertTrue(out.shape[1] == 576)

        # fmt: off
        expected_tokens =  Expectations(
            {
                ("rocm", None): [
                    10367, 1380, 4841, 15155, 1224, 16361, 15834, 13722, 15258, 8321, 10496, 14532, 8770, 12353, 5481,
                    11484, 2585, 8587, 3201, 14292, 3356, 2037, 3077, 6107, 3758, 2572, 9376, 13219, 6007, 14292, 12696,
                    10666, 10046, 13483, 8282, 9101, 5208, 4260, 13886, 13335, 6135, 2316, 15423, 311, 5460, 12218,
                    14172, 8583, 14577, 3648
                ],
                ("rocm", (9, 5)): [
                    4484, 4015, 15750, 506, 3758, 11651, 8597, 5739, 4861, 971, 14985, 14834, 15438, 7548, 1820, 1465,
                    13529, 12761, 10503, 12761, 14303, 6155, 4015, 11766, 705, 15736, 14146, 10417, 1951, 7713, 14305,
                    15617, 6169, 2706, 8006, 14893, 3855, 10188, 15652, 6297, 1097, 12108, 15038, 311, 14998, 15165,
                    897, 4044, 1762, 4676
                ],
                ("cuda", None): [
                    4484, 4015, 15750, 506, 3758, 11651, 8597, 5739, 4861, 971, 14985, 14834, 15438, 7548, 1820, 1465,
                    13529, 12761, 10503, 12761, 14303, 6155, 4015, 11766, 705, 15736, 14146, 10417, 1951, 7713, 14305,
                    15617, 6169, 2706, 8006, 14893, 3855, 10188, 15652, 6297, 1097, 12108, 15038, 311, 14998, 15165,
                    897, 4044, 1762, 4676
                ],
            }
        )
        expected_tokens = torch.tensor(expected_tokens.get_expectation()).to(model.device)
        # fmt: on

        # Compare the first 50 generated tokens.
        self.assertTrue(torch.allclose(expected_tokens, out[0][:50]))

        # Decode generated tokens to pixel values and postprocess them.
        decoded_pixel_values = model.decode_image_tokens(out)
        images = processor.postprocess(list(decoded_pixel_values.float()), return_tensors="np")

        self.assertTrue(images["pixel_values"].shape == (1, 384, 384, 3))
        self.assertTrue(isinstance(images["pixel_values"], np.ndarray))
