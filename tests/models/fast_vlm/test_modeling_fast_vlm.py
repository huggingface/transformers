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
"""Testing suite for the FastVLM model."""

import copy
import unittest

import requests

from transformers import (
    AutoProcessor,
    FastVlmConfig,
    FastVlmForConditionalGeneration,
    FastVlmModel,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_vision,
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


class FastVlmVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_id=0,
        projector_hidden_act="gelu",
        seq_length=7,
        vision_feature_select_strategy="full",
        vision_feature_layer=-1,
        text_config={
            "model_type": "qwen2",
            "is_training": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "initializer_range": 0.02,
            "pad_token_id": 1,
        },
        is_training=True,
        vision_config={
            "image_size": 16,
            "patch_size": 8,
            "num_channels": 3,
            "hidden_size": 32,
            "initializer_range": 0.02,
            "architecture": "fastvit_mci3",
            "do_pooling": True,
            "global_pool": "avg",
            "model_args": {
                "inference_mode": True,
                "layers": (2, 2),
                "embed_dims": (8, 16),
                "mlp_ratios": (4, 4),
                "se_downsamples": (False, False),
                "downsamples": (False, True),
                "pos_embs": (None, None),
                "token_mixers": ("repmixer", "repmixer"),
                "lkc_use_act": True,
                "stem_use_scale_branch": False,
            },
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.image_token_id = image_token_id
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.text_config = text_config
        self.vision_config = vision_config
        self.pad_token_id = text_config["pad_token_id"]

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_image_tokens = (self.vision_config["image_size"] // self.vision_config["patch_size"]) ** 2
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return FastVlmConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_id=self.image_token_id,
            projector_hidden_act=self.projector_hidden_act,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layer=self.vision_feature_layer,
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
        input_ids[input_ids == config.image_token_index] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = config.image_token_index
        attention_mask = input_ids.ne(1).to(torch_device)

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class FastVlmForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `FastVlmForConditionalGeneration`.
    """

    all_model_classes = (
        (
            FastVlmModel,
            FastVlmForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"image-to-text": FastVlmForConditionalGeneration, "image-text-to-text": FastVlmForConditionalGeneration}
        if is_torch_available()
        else {}
    )

    _is_composite = True

    def setUp(self):
        self.model_tester = FastVlmVisionText2TextModelTester(self)
        common_properties = ["image_token_id"]
        self.config_tester = ConfigTester(
            self, config_class=FastVlmConfig, has_text_modality=False, common_properties=common_properties
        )

    def test_enable_input_require_grads(self):
        self.skipTest("FastVLM relies on timm architectures unavailable in this test environment.")

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_mismatching_num_image_tokens(self):
        """
        Tests that an explicit error is thrown when the number of image tokens
        doesn't match the number of image placeholders in the text.
        We also test multi-image cases when one prompt has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            curr_input_dict = copy.deepcopy(input_dict)  # in-place modifications further
            _ = model(**curr_input_dict)  # successful forward with no modifications

            # remove one image but leave all the image tokens in text
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-2:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # simulate the multi-image/single set of placeholders case by concatenating
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:1]
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)

            # two images and one set of image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(input_ids=input_ids, pixel_values=pixel_values)

            # two images and two sets of image tokens don't raise an error
            input_ids = torch.cat([input_ids, input_ids], dim=0)
            _ = model(input_ids=input_ids, pixel_values=pixel_values)

    @unittest.skip("Timm can't be initialized on meta")
    def test_can_be_initialized_on_meta(self):
        pass


@require_torch
@slow
class FastVlmForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("KamilaMila/FastVLM-0.5B")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @require_vision
    def test_small_model_integration_test(self):
        model = FastVlmForConditionalGeneration.from_pretrained(
            "KamilaMila/FastVLM-0.5B", device_map=torch_device, dtype=torch.bfloat16
        )

        prompt = "user\n<image>\nWhat are the things I should be cautious about when I visit this place?\nassistant"
        image_file = "https://llava-vl.github.io/static/images/view.jpg"
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = self.processor(images=raw_image, text=prompt, return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20)
        expected_decoded_texts = "user\n\nWhat are the things I should be cautious about when I visit this place?\nassistant\n\nWhen visiting this place, there are a few things you should be cautious about:\n\n1. **"  # fmt: skip

        EXPECTED_DECODED_TEXT = expected_decoded_texts

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @require_vision
    def test_small_model_integration_test_batch(self):
        model = FastVlmForConditionalGeneration.from_pretrained(
            "KamilaMila/FastVLM-0.5B", device_map=torch_device, dtype=torch.bfloat16
        )

        prompts = [
            "user\n<image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nassistant",
            "user\n<image>\nWhat is this?\nassistant",
        ]
        image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
        image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = self.processor(images=[image1, image2], text=prompts, return_tensors="pt", padding=True).to(
            torch_device
        )

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = [
            "user\n\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nassistant\n\nWhen visiting this serene place, it's essential to be mindful of the following:\n\n1. **",
            "user\n\nWhat is this?\nassistant\nThe image depicts two cats lying on a pink surface, which could be a couch or a"
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_generation_no_images(self):
        model_id = "KamilaMila/FastVLM-0.5B"
        model = FastVlmForConditionalGeneration.from_pretrained(
            model_id, device_map=torch_device, dtype=torch.bfloat16
        )
        processor = AutoProcessor.from_pretrained(model_id)

        # Prepare inputs with no images
        inputs = processor(text="Hello, I am", return_tensors="pt").to(torch_device)

        # Make sure that `generate` works
        _ = model.generate(**inputs, max_new_tokens=20)
