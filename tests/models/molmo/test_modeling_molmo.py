# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Molmo model."""

import gc
import unittest

import requests

from transformers import (
    AutoProcessor,
    MolmoConfig,
    MolmoForConditionalGeneration,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import require_torch, require_torch_gpu, require_vision, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class MolmoVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        image_token_index=0,
        projector_hidden_act="gelu",
        seq_length=7,
        vision_feature_select_strategy="default",
        vision_feature_layers=(0, 1),
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
            "intermediate_size": 38,
            "head_dim": 8,
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
            "image_size": 49,
            "num_image_positions": 50,
            "patch_size": 4,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "projection_dim": 32,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        pooling_config={
            "image_num_patches": 7,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "head_dim": 8,
            "pad_embed_dim": 64,
            "text_intermediate_size": 38,
            "text_hidden_size": 32,
        },
    ):
        self.parent = parent
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layers = vision_feature_layers
        self.text_config = text_config
        self.vision_config = vision_config
        self.pooling_config = pooling_config
        self.pad_token_id = text_config["pad_token_id"]

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_patches = 5
        self.image_size = 49
        self.num_image_tokens = 16
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return MolmoConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            pooling_config=self.pooling_config,
            image_token_index=self.image_token_index,
            projector_hidden_act=self.projector_hidden_act,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layers=self.vision_feature_layers,
            image_seq_length=self.num_image_tokens,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.num_patches,
                self.vision_config["image_size"],
                self.vision_config["patch_size"] ** 2 * 3,
            ]
        )
        image_token_indices = torch.arange(self.num_image_tokens, device=torch_device)
        image_token_indices = image_token_indices.unsqueeze(0).repeat(self.batch_size, self.num_patches, 1)
        image_masks = torch.ones_like(pixel_values)[..., 0]
        config = self.get_config()

        return config, pixel_values, image_token_indices, image_masks

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, image_token_indices, image_masks = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(1).to(torch_device)
        input_ids[input_ids == config.image_token_index] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = config.image_token_index
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_token_indices": image_token_indices,
            "image_masks": image_masks,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_molmo_model_fp16_forward(self, config, input_ids, pixel_values, attention_mask):
        model = MolmoForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values.to(torch.bfloat16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())


@require_torch
class MolmoForConditionalGenerationModelTest(
    ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    """
    Model tester for `MolmoForConditionalGeneration`.
    """

    all_model_classes = (MolmoForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (MolmoForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-to-text": MolmoForConditionalGeneration, "image-text-to-text": MolmoForConditionalGeneration}
        if is_torch_available()
        else {}
    )
    test_torchscript = False
    test_pruning = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = MolmoVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MolmoConfig, has_text_modality=False)

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

            wte = model.get_input_embeddings()
            inputs["inputs_embeds"] = wte(input_ids)

            with torch.no_grad():
                model(**inputs)

    # overwrite inputs_embeds tests because we need to delete "pixel values" for LVLMs
    # while some other models require pixel_values to be present
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

            inputs_embeds = model.get_input_embeddings()(input_ids)

            with torch.no_grad():
                out_ids = model(input_ids=input_ids, **inputs)[0]
                out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            self.assertTrue(torch.allclose(out_embeds, out_ids))

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="VLMs have dynamic control flow in preparing inputs for generation")
    def test_generate_compile_1_end_to_end(self):
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


@require_torch
@require_vision
class MolmoForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("Molbap/molmo-hf-7B-D")

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    @require_torch_gpu
    def test_7B_model_integration_test(self):
        model = MolmoForConditionalGeneration.from_pretrained(
            "Molbap/molmo-hf-7B-D", torch_dtype=torch.bfloat16, device_map="auto"
        )

        prompt = "<image> User: Describe this image. Assistant:"
        image_file = "https://picsum.photos/id/237/536/354"
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = self.processor(images=raw_image, text=prompt, return_tensors="pt").to(torch.bfloat16).to(model.device)
        EXPECTED_INPUT_IDS = torch.tensor([[151643, 152064, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152065, 152064, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152066, 152067, 152065, 2657, 25,  60785, 419, 2168, 13, 21388, 25]])  # fmt: skip
        self.assertTrue(torch.equal(inputs["input_ids"].cpu(), EXPECTED_INPUT_IDS))

        output = model.generate(**inputs, max_new_tokens=18)
        EXPECTED_DECODED_TEXT = "User: Describe this image. Assistant: This image captures a young black Labrador puppy, likely around 12 weeks old, sitting"  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
