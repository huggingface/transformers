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
"""Testing suite for the PyTorch Llava model."""

import unittest

import requests

from transformers import (
    AutoProcessor,
    Phi3VConfig,
    Phi3VForConditionalGeneration,
    Phi3VModel,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    Expectations,
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
    pass


if is_vision_available():
    from PIL import Image


class Phi3VModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=0,
        projector_hidden_act="gelu",
        seq_length=7,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
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
        self.num_channels = 3
        self.image_size = 336
        self.num_image_tokens = (self.vision_config["image_size"] // self.vision_config["patch_size"]) ** 2
        self.seq_length = seq_length + self.num_image_tokens
        self.encoder_seq_length = self.seq_length

    def get_config(self):
        return Phi3VConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
            projector_hidden_act=self.projector_hidden_act,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layer=self.vision_feature_layer,
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
class LlavaForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `LlavaForConditionalGeneration`.
    """

    all_model_classes = (
        (
            Phi3VModel,
            Phi3VForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"image-to-text": Phi3VForConditionalGeneration, "image-text-to-text": Phi3VForConditionalGeneration}
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Phi3VModelTester(self)
        common_properties = ["image_token_index", "vision_feature_layer", "image_seq_length"]
        self.config_tester = ConfigTester(
            self, config_class=Phi3VConfig, has_text_modality=False, common_properties=common_properties
        )

    def test_config(self):
        self.config_tester.run_common_tests()


@require_torch
class LlavaForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("llava-hf/bakLlava-v1-hf")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test(self):
        # Let's make sure we test the preprocessing to replace what is used
        model = Phi3VForConditionalGeneration.from_pretrained("llava-hf/bakLlava-v1-hf", load_in_4bit=True)

        prompt = "<image>\nUSER: What are the things I should be cautious about when I visit this place?\nASSISTANT:"
        image_file = "https://llava-vl.github.io/static/images/view.jpg"
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = self.processor(images=raw_image, text=prompt, return_tensors="pt").to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20)
        expected_decoded_texts = Expectations({
            ("cuda", None): "\nUSER: What are the things I should be cautious about when I visit this place?\nASSISTANT: When visiting this place, there are a few things one should be cautious about. Firstly,",
            ("rocm", (9, 5)): "\nUSER: What are the things I should be cautious about when I visit this place?\nASSISTANT: When visiting this place, there are a few things one should be cautious about. First, the",
        })  # fmt: skip
        EXPECTED_DECODED_TEXT = expected_decoded_texts.get_expectation()

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_llama_batched(self):
        # Let's make sure we test the preprocessing to replace what is used
        model_id = "llava-hf/llava-1.5-7b-hf"

        model = Phi3VForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", load_in_4bit=True)
        processor = AutoProcessor.from_pretrained(model_id)

        prompts = [
            "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me? ASSISTANT:",
            "USER: <image>\nWhat is this? ASSISTANT:",
        ]
        image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
        image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = processor(images=[image1, image2], text=prompts, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20)

        expected_decoded_texts = Expectations(
            {
                ("cuda", None): [
                    "USER:  \nWhat are the things I should be cautious about when I visit this place? What should I bring "
                    "with me? ASSISTANT: When visiting this place, which is a pier or dock extending over a body of water, "
                    "you",
                    "USER:  \nWhat is this? ASSISTANT: The image features two cats lying down on a pink couch. One cat "
                    "is located on",
                ],
                ("rocm", (9, 5)): [
                    "USER:  \nWhat are the things I should be cautious about when I visit this place? What should I bring "
                    "with me? ASSISTANT: When visiting this serene location, which features a wooden pier overlooking a "
                    "lake, you should",
                    "USER:  \nWhat is this? ASSISTANT: The image features two cats lying down on a pink couch. One cat "
                    "is located on",
                ],
            }
        )
        EXPECTED_DECODED_TEXT = expected_decoded_texts.get_expectation()

        decoded_output = processor.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(decoded_output, EXPECTED_DECODED_TEXT)

    @slow
    @require_bitsandbytes
    def test_generation_no_images(self):
        model_id = "llava-hf/llava-1.5-7b-hf"
        model = Phi3VForConditionalGeneration.from_pretrained(model_id, load_in_4bit=True)
        processor = AutoProcessor.from_pretrained(model_id)

        # Prepare inputs with no images
        inputs = processor(text="Hello, I am", return_tensors="pt").to(torch_device)

        # Make sure that `generate` works
        _ = model.generate(**inputs, max_new_tokens=20)
