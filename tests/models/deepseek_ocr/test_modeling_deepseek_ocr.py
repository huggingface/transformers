# Copyright 2025 HuggingFace Inc.
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

import re
import tempfile
import unittest

from transformers import (
    AutoProcessor,
    DeepseekOcrConfig,
    DeepseekOcrForConditionalGeneration,
    DeepseekOcrModel,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch


class DeepseekOcrModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=25,
        num_channels=3,
        initializer_range=0.02,
        is_training=True,
        use_cache=False,
        text_config={
            "model_type": "deepseek_v2",
            "num_hidden_layers": 2,
            "vocab_size": 99,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "pad_token_id": 1,
            "use_mla": False,
            "first_k_dense_replace": 1,
            "moe_intermediate_size": 32,
            "n_routed_experts": 2,
            "n_shared_experts": 0,
            "num_experts_per_tok": 1,
        },
        sam_config={
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "num_attention_heads": 2,
            "image_size": 128,
            "patch_size": 16,
            "hidden_act": "gelu",
            "output_channels": 8,
            "window_size": 2,
            "global_attn_indexes": [],
            "use_rel_pos": False,
            "downsample_channels": [12, 16],
        },
        clip_config={
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "image_size": 128,
            "patch_size": 16,
            "hidden_act": "quick_gelu",
        },
        projector_config=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.initializer_range = initializer_range
        self.is_training = is_training
        self.use_cache = use_cache

        self.text_config = text_config
        self.sam_config = sam_config
        self.clip_config = clip_config
        if projector_config is None:
            projector_hidden = text_config["hidden_size"]
            sam_projector_dim = (
                sam_config["downsample_channels"][-1]
                if sam_config["downsample_channels"]
                else sam_config["output_channels"]
            )
            projector_input_dim = clip_config["hidden_size"] + sam_projector_dim
            projector_config = {
                "input_dim": projector_input_dim,
                "n_embed": projector_hidden,
            }
        self.projector_config = projector_config

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.image_size = sam_config["image_size"]
        self.num_image_tokens = 7
        self.pad_token_id = text_config["pad_token_id"]
        self.image_token_id = 0
        self.projector_config = (
            projector_config
            if projector_config is not None
            else {
                "input_dim": 32,
                "n_embed": text_config["hidden_size"],
            }
        )

    def get_config(self):
        vision_config = {
            "sam_config": self.sam_config,
            "clip_config": self.clip_config,
        }
        return DeepseekOcrConfig(
            text_config=self.text_config,
            vision_config=vision_config,
            projector_config=self.projector_config,
            image_token_id=self.image_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()

        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 1
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.num_channels,
                self.image_size,
                self.image_size,
            ]
        )
        input_ids[input_ids == self.image_token_id] = config.text_config.pad_token_id
        input_ids[:, : self.num_image_tokens] = self.image_token_id

        return config, input_ids, attention_mask, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        image_sizes = torch.tensor([[self.image_size, self.image_size]] * self.batch_size)
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
        }
        return config, inputs_dict


@require_torch
class DeepseekOcrModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (DeepseekOcrModel, DeepseekOcrForConditionalGeneration) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": DeepseekOcrModel,
            "image-text-to-text": DeepseekOcrForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    _is_composite = True

    def setUp(self):
        self.model_tester = DeepseekOcrModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DeepseekOcrConfig, has_text_modality=False)

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
            if "image_sizes" in inputs:
                del inputs["image_sizes"]

            wte = model.get_input_embeddings()
            inputs["inputs_embeds"] = wte(input_ids)

            with torch.no_grad():
                model(**inputs)

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
            if "image_sizes" in inputs:
                del inputs["image_sizes"]

            inputs_embeds = model.get_input_embeddings()(input_ids)

            with torch.no_grad():
                out_ids = model(input_ids=input_ids, **inputs)[0]
                out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            torch.testing.assert_close(out_embeds, out_ids)

    @unittest.skip(reason="Composite model with external vision components")
    def test_can_init_all_missing_weights(self):
        pass

    def test_sdpa_can_dispatch_composite_models(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    attn_implementation="eager",
                )
                model_eager = model_eager.eval().to(torch_device)

            vision_attn = language_attn = "sdpa" if model._supports_sdpa else "eager"

            if hasattr(model_sdpa, "sam_model") and hasattr(model_sdpa, "language_model"):
                self.assertTrue(model_sdpa.sam_model.config._attn_implementation == vision_attn)
                self.assertTrue(model_sdpa.clip_model.config._attn_implementation == vision_attn)
                self.assertTrue(model_sdpa.language_model.config._attn_implementation == language_attn)
                self.assertTrue(model_eager.sam_model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.clip_model.config._attn_implementation == "eager")
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

    @require_torch
    @slow
    class DeepseekOcrIntegrationTest(unittest.TestCase):
        def setUp(self):
            self.model_id = "deepseek-ai/DeepSeek-OCR"

        def test_model_text_generation(self):
            processor = AutoProcessor.from_pretrained(self.model_id)
            model = DeepseekOcrForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)

            conversation = [
                {
                    "role": "<|User|>",
                    "content": [
                        {"type": "image", "path": "./handwritten_letter_small.png"},
                        {"type": "text", "text": "<|grounding|>Convert the document to markdown."},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                conversation, return_dict=True, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(device=torch_device, dtype=model.dtype)
            # inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=250)

            text = processor.batch_decode(generated, skip_special_tokens=False)[0]

            self.assertIn("<|grounding|>Convert the document to markdown.", text)
            self.assertIn("text", text)
            det_match = re.search(r"<\|det\|>\[\[([^\]]+)\]\]<\|/det\|>", text)
            self.assertIsNotNone(det_match, msg=f"Detection coordinates not found in generated text: {text}")
            det_values = [int(val.strip()) for val in det_match.group(1).split(",")]
            expected_values = [52, 50, 940, 950]
            for generated_val, expected_val in zip(det_values, expected_values):
                self.assertLessEqual(
                    abs(generated_val - expected_val),
                    10,
                    msg=f"Detection coordinate {generated_val} deviates too much from expected {expected_val}",
                )
