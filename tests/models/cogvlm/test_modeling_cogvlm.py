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
""" Testing suite for the PyTorch CogVLM model. """


import inspect
import tempfile
import unittest

import requests

from transformers import CogVLMConfig, CogVLMVisionConfig
from transformers.testing_utils import (
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import CogVLMForCausalLM, CogVLMModel, CogVLMVisionConfig
    from transformers.models.cogvlm.modeling_cogvlm import COGVLM_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import CogVLMProcessor


class CogVLMModelTester:
    def __init__(self,
                parent,
                num_channels=3,
                image_size=32,
                patch_size=2,
                batch_size=1,
                text_seq_length=7,
                is_training=True,
                use_input_mask=True,
                use_labels=True,
                vocab_size=99,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=37,
                hidden_act="gelu",
                max_position_embeddings=512,
                initializer_range=0.02,
                num_labels=3,
            ):
        
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.text_seq_length = text_seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.parent = parent
        self.is_training = is_training

        self.vision_seq_length = (self.image_size // self.patch_size) ** 2 + 2
        self.seq_length = self.text_seq_length + self.vision_seq_length

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).to(torch_device)
        token_type_ids = torch.cat(
            [
                            torch.zeros(self.batch_size, self.text_seq_length, dtype=torch.long),
                            torch.ones(self.batch_size, self.vision_seq_length, dtype=torch.long),
            ],
            dim=1,
        ).to(torch_device)

        attention_mask = None
        if self.use_input_mask:
            attention_mask = random_attention_mask([self.batch_size, self.seq_length]).to(torch_device)

        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size]).to(torch_device)

        labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels).to(torch_device) if self.use_labels else None

        config = self.get_config()

        return config, input_ids, attention_mask, token_type_ids, pixel_values, labels
    
    def get_vision_config(self):
        return CogVLMVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
        )

    def get_config(self):
        return CogVLMConfig(
            vision_config=self.get_vision_config().to_dict(),
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            vocab_size=self.vocab_size,
        )

    def create_and_check_for_causal_lm(self, config, input_ids, attention_mask, token_type_ids, pixel_values, labels):
        model = CogVLMForCausalLM(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            token_type_ids,
            pixel_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
        return config, inputs_dict


@require_torch
class CogVLMModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (CogVLMForCausalLM, CogVLMModel) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": CogVLMModel, "image-to-text": CogVLMForCausalLM}
        if is_torch_available()
        else {}
    )

    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = CogVLMModelTester(self)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_cpu_offload(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in COGVLM_PRETRAINED_MODEL_ARCHIVE_LIST:
            model = CogVLMForCausalLM.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    url = "https://huggingface.co/hf-internal-testing/blip-test-image/resolve/main/demo.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


@require_vision
@require_torch
@slow
class CogVLMModelIntegrationTest(unittest.TestCase):
    def test_inference_opt(self):
        processor = CogVLMProcessor.from_pretrained("THUDM/cogvlm-chat-hf")
        model = CogVLMForCausalLM.from_pretrained("THUDM/cogvlm-chat-hf", torch_dtype=torch.float16).to(torch_device)

        # prepare image
        image = prepare_img()
        inputs = processor(images=image, return_tensors="pt").to(torch_device, dtype=torch.float16)

        predictions = model.generate(**inputs)
        generated_text = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip()

        # Test output
        self.assertEqual(predictions[0].tolist(), [2, 102, 693, 2828, 15, 5, 4105, 19, 10, 2335, 50118])
        self.assertEqual("a woman sitting on the beach with a dog", generated_text)

        # image and context
        prompt = "Question: which city is this? Answer:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch_device, dtype=torch.float16)

        predictions = model.generate(**inputs)
        generated_text = processor.batch_decode(predictions, skip_special_tokens=True)[0].strip()

        # Test output
        self.assertEqual(
            predictions[0].tolist(),
            [2, 24, 18, 45, 10, 343, 6, 24, 18, 10, 4105, 50118],
        )
        self.assertEqual(generated_text, "it's not a city, it's a beach")
