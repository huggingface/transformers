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
"""Testing suite for the PyTorch Mllama model."""

import gc
import unittest

import requests

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    MllamaConfig,
    MllamaForConditionalGeneration,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_bitsandbytes,
    require_torch,
    require_torch_gpu,
    require_vision,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch
else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class MllamaVisionText2TextModelTester:
    # TODO add correct dummy config
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
            "pad_token_id": 0,
        },
        is_training=True,
        vision_config={
            "image_size": 30,
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
        self.seq_length = seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = 3
        self.image_size = 336
        self.encoder_seq_length = 231

    def get_config(self):
        return MllamaConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
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
        attention_mask = input_ids.ne(1).to(torch_device)
        # we are giving 3 images let's make sure we pass in 3 image tokens
        input_ids[:, 1] = config.image_token_index
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_mllama_model_fp16_forward(self, config, input_ids, pixel_values, attention_mask):
        model = MllamaForConditionalGeneration(config=config)
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
class MllamaForConditionalGenerationModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `MllamaForConditionalGeneration`.
    """

    all_model_classes = (MllamaForConditionalGeneration,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = MllamaVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MllamaConfig, has_text_modality=False)

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


@require_torch
class MllamaForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.device = "cuda"
        self.small_model_checkpoint = "s0409/model-1"
        self.processor = AutoProcessor.from_pretrained(self.small_model_checkpoint)

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    @require_torch_gpu
    def test_11b_model_integration_generate(self):
        
        # Prepare inputs
        prompt = "<|image|><|begin_of_text|>If I had to write a haiku for this one"

        url = "https://llava-vl.github.io/static/images/view.jpg"
        raw_image = Image.open(requests.get(url, stream=True).raw)

        inputs = self.processor(prompt, raw_image, return_tensors="pt")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        input_ids = inputs["input_ids"]
        input_ids[input_ids == 128011] = 128256  # TODO: remove when tokenizer corrected

        # Check inputs ids
        expected_input_ids = torch.tensor([[128256, 128000, 2746, 358, 1047, 311, 3350, 264, 6520, 39342, 369, 420, 832]], device=self.device)  # fmt: skip
        self.assertTrue(torch.equal(input_ids, expected_input_ids))

        # Prepare model
        torch_dtype = torch.bfloat16
        model = MllamaForConditionalGeneration.from_pretrained(
            self.small_model_checkpoint, torch_dtype=torch_dtype, device_map=self.device
        )
        model.setup_cache(1, torch_dtype)  # TODO: remove when native cache is supported

        # Run generate
        position_ids = torch.arange(0, input_ids.shape[1]).to(self.device)
        model_kwargs = {
            "position_ids": position_ids,
            "pixel_values": inputs["pixel_values"],
            "aspect_ratios": inputs["aspect_ratios"],
            "num_tiles": inputs["num_tiles"],
            "cross_attention_token_mask": inputs["cross_attention_token_mask"],
            "use_cache": False,
        }
        output = model.generate(input_ids, **model_kwargs, do_sample=False, max_new_tokens=25)

        decoded_output = self.processor.decode(output[0], skip_special_tokens=True)
        expected_output = "If I had to write a haiku for this one, it would be:.\\nA dock on a lake.\\nA mountain in the distance.\\nA long exposure."  # fmt: skip
        
        self.assertEqual(decoded_output, expected_output)

    @slow
    @require_torch_gpu
    def test_11b_model_integration_forward(self):
        
        # Prepare inputs
        prompt = "<|image|><|begin_of_text|>If I had to write a haiku for this one"

        url = "https://llava-vl.github.io/static/images/view.jpg"
        raw_image = Image.open(requests.get(url, stream=True).raw)

        inputs = self.processor(prompt, raw_image, return_tensors="pt")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        input_ids = inputs["input_ids"]
        input_ids[input_ids == 128011] = 128256  # TODO: remove when tokenizer corrected

        # Check inputs ids
        expected_input_ids = torch.tensor([[128256, 128000, 2746, 358, 1047, 311, 3350, 264, 6520, 39342, 369, 420, 832]], device=self.device)  # fmt: skip
        self.assertTrue(torch.equal(input_ids, expected_input_ids))

        # Prepare model
        torch_dtype = torch.bfloat16
        model = MllamaForConditionalGeneration.from_pretrained(
            self.small_model_checkpoint, torch_dtype=torch_dtype, device_map=self.device
        )
        model.setup_cache(1, torch_dtype)  # TODO: remove when native cache is supported

        # Run generate
        position_ids = torch.arange(0, input_ids.shape[1]).to(self.device)
        model_kwargs = {
            "position_ids": position_ids,
            "input_ids": input_ids,
            "pixel_values": inputs["pixel_values"],
            "aspect_ratios": inputs["aspect_ratios"],
            "num_tiles": inputs["num_tiles"],
            "cross_attention_token_mask": inputs["cross_attention_token_mask"],
            "use_cache": False,
        }
        with torch.inference_mode():
            output = model(**model_kwargs)

        actual_cross_attention_key_value = output.cross_attention_key_value[0, 0, 0, 0, :5, 64].cpu()
        expected_cross_attention_key_value = torch.tensor([-0.0933, 0.2930, 1.2656, -0.9883, -0.2100], dtype=torch_dtype)
        self.assertTrue(torch.allclose(actual_cross_attention_key_value, expected_cross_attention_key_value, atol=1e-4))

        actual_logits = output.logits[0, -1, :5].cpu()
        expected_logits = torch.tensor([8.5000, 7.8750, 4.2812, 0.5000, 3.0312], dtype=torch_dtype)
        self.assertTrue(torch.allclose(actual_logits, expected_logits, atol=1e-4))
