# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch VipLlava model."""

import unittest

import requests
from parameterized import parameterized

from transformers import (
    AutoProcessor,
    VipLlavaConfig,
    VipLlavaForConditionalGeneration,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    cleanup,
    require_bitsandbytes,
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


# Copied from transformers.tests.models.llava.test_modeling_llava.LlavaVisionText2TextModelTester with Llava->VipLlava
class VipLlavaVisionText2TextModelTester:
    # Ignore copy
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=0,
        projector_hidden_act="gelu",
        seq_length=7,
        vision_feature_layers=[0, 0, 1, 1, 0],
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
            "batch_size": 12,
            "image_size": 8,
            "patch_size": 2,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "projection_dim": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_layers = vision_feature_layers
        self.text_config = text_config
        self.vision_config = vision_config
        self.pad_token_id = text_config["pad_token_id"]

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = 3
        self.image_size = 336
        self.num_image_tokens = (self.vision_config["image_size"] // self.vision_config["patch_size"]) ** 2
        self.seq_length = seq_length + self.num_image_tokens
        self.encoder_seq_length = self.seq_length

    def get_config(self):
        return VipLlavaConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
            projector_hidden_act=self.projector_hidden_act,
            vision_feature_layers=self.vision_feature_layers,
            image_seq_length=self.num_image_tokens,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(1).to(torch_device)

        input_ids[input_ids == config.image_token_index] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = config.image_token_index
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
# Copied from transformers.tests.models.llava.test_modeling_llava.LlavaForConditionalGenerationModelTest with Llava->VipLlava
class VipLlavaForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `VipLlavaForConditionalGeneration`.
    """

    all_model_classes = (VipLlavaForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-text-to-text": VipLlavaForConditionalGeneration} if is_torch_available() else {}
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = True
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = VipLlavaVisionText2TextModelTester(self)
        common_properties = ["image_token_index", "vision_feature_layers", "image_seq_length"]
        self.config_tester = ConfigTester(
            self, config_class=VipLlavaConfig, has_text_modality=False, common_properties=common_properties
        )

    def test_config(self):
        self.config_tester.run_common_tests()

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
            torch.testing.assert_close(out_embeds, out_ids)

    # Copied from tests.models.llava.test_modeling_llava.LlavaForConditionalGenerationModelTest.test_mismatching_num_image_tokens
    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs through an error with explicit message saying what is wrong
        when number of images doesn't match number of image tokens in the text.
        Also we need to test multi-image cases when one prompr has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            _ = model(**input_dict)  # successful forward with no modifications

            # remove one image but leave the image token in text
            input_dict["pixel_values"] = input_dict["pixel_values"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**input_dict)

            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = input_dict["input_ids"][:1]
            pixel_values = input_dict["pixel_values"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(input_ids=input_ids, pixel_values=pixel_values)

            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            _ = model(input_ids=input_ids, pixel_values=pixel_values)

    @parameterized.expand(
        [
            (-1,),
            ([-1],),
            ([-1, -2],),
        ],
    )
    def test_vision_feature_layers(self, vision_feature_layers):
        """
        Test that we can use either one vision feature layer, or a list of
        vision feature layers.
        """
        # NOTE: vipllava uses vision_feature_layers instead of vision_feature_layer as the
        # config key. The reason is that other llava classes supported one vision feature layer
        # and added support for a list of layers with granite vision support, while vipllava
        # originally supported multiple feature layers, and added support for a single layer for
        # for compatibility reasons.
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.vision_feature_layers = vision_feature_layers

        num_feature_layers = 1 if isinstance(vision_feature_layers, int) else len(vision_feature_layers)
        hidden_size = config.vision_config.hidden_size
        expected_features = hidden_size * num_feature_layers

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            # We should have the right number of input features,
            # and should be able to run a forward pass without exploding
            assert model.multi_modal_projector.linear_1.in_features == expected_features
            model(**input_dict)

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip("FlashAttention only support fp16 and bf16 data type")
    def test_flash_attn_2_fp32_ln(self):
        pass

    @unittest.skip(
        "VLMs need lots of steps to prepare images/mask correctly to get pad-free inputs. Can be tested as part of LLM test"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass


@require_torch
class VipLlavaForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("llava-hf/vip-llava-7b-hf")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test(self):
        model_id = "llava-hf/vip-llava-7b-hf"

        model = VipLlavaForConditionalGeneration.from_pretrained(model_id, load_in_4bit=True)
        processor = AutoProcessor.from_pretrained(model_id)

        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-neg.png"

        image = Image.open(requests.get(url, stream=True).raw)
        prompt = "USER: <image>\nCan you please describe this image?\nASSISTANT:"

        inputs = processor(prompt, image, return_tensors="pt").to(torch_device, torch.float16)

        outputs = model.generate(**inputs, max_new_tokens=10)

        EXPECTED_OUTPUT = "USER:  \nCan you please describe this image?\nASSISTANT: The image features a brown and white cat sitting on"
        self.assertEqual(processor.decode(outputs[0], skip_special_tokens=True), EXPECTED_OUTPUT)
