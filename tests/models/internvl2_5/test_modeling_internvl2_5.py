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
"""Testing suite for the PyTorch Llava-NeXT model."""

import gc
import unittest

import requests

from transformers import (
    AutoProcessor,
    InternVL2_5Config,
    InternVL2_5ForConditionalGeneration,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch
else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class InternVL2_5VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        seq_length=7,
        ignore_id=-100,
        bos_token_id=0,
        eos_token_id=1,  # 151645
        pad_token_id=2,  # 151643
        image_token_id=3,  # 151667
        image_start_token_id=4,  # 151665
        image_end_token_id=5,  # 151666
        select_layer=-1,
        text_config={
            "model_type": "qwen2",
            "seq_length": 7,
            "is_training": True,
            "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 580,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 2,
            "image_token_id": 3,
        },
        is_training=True,
        vision_config={
            "image_size": 16,
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
        self.ignore_id = ignore_id
        self.image_token_id = image_token_id
        self.pad_token_id = pad_token_id
        self.select_layer = select_layer
        self.text_config = text_config
        self.vision_config = vision_config
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training
        self.batch_size = 3
        self.num_channels = 3
        self.image_size = 30
        self.num_image_tokens = 16
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return InternVL2_5Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            select_layer=self.select_layer,
            hidden_size=self.hidden_size,
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
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        # input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[:, 1 : self.num_image_tokens + 1] = self.image_token_id
        labels = torch.zeros(
            (self.batch_size, self.seq_length),
            dtype=torch.long,
            device=torch_device,
        )
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "num_patches": torch.tensor([1] * self.batch_size, device=torch_device),
        }
        return config, inputs_dict

    def create_and_check_internvl2_5_model_fp16_forward(
        self, config, input_ids, pixel_values, attention_mask, image_sizes
    ):
        model = InternVL2_5ForConditionalGeneration(config=config)
        model.to(torch_device)
        model.half()
        model.eval()
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_sizes=image_sizes,
            pixel_values=pixel_values.to(torch.bfloat16),
            return_dict=True,
        )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())

    def create_and_check_internvl2_5_model_fp16_autocast_forward(
        self, config, input_ids, pixel_values, attention_mask, image_sizes
    ):
        config.torch_dtype = torch.float16
        model = InternVL2_5ForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_sizes=image_sizes,
                pixel_values=pixel_values.to(torch.bfloat16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())


@require_torch
class InternVL2_5ForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `InternVL2_5ForConditionalGeneration`.
    """

    all_model_classes = (InternVL2_5ForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (InternVL2_5ForConditionalGeneration,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = InternVL2_5VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=InternVL2_5Config, has_text_modality=False)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if "vision_model" in name:
                    continue
                elif param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

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


@require_torch
class InternVL2_5ForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("thisisiron/InternVL2_5-1B")  # Use of temporary storage
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            }
        ]
        url = "https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/demo_small.jpg"
        self.image = Image.open(requests.get(url, stream=True).raw)

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    def test_small_model_integration_test(self):
        model = InternVL2_5ForConditionalGeneration.from_pretrained(
            "thisisiron/InternVL2_5-1B", torch_dtype="auto", device_map="auto"
        )

        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[self.image], return_tensors="pt")

        expected_input_ids = [
            151644,
            8948,
            198,
            2610,
            525,
            264,
            10950,
            17847,
            13,
            151645,
            198,
            151644,
            872,
            198,
            151665,
            151667,
            151667,
        ]
        assert expected_input_ids == inputs.input_ids[0].tolist()[:17]

        # verify generation
        inputs = inputs.to(torch_device, torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=30)
        EXPECTED_DECODED_TEXT = "system\nYou are a helpful assistant.\nuser\n\nWhat kind of dog is this?\nassistant\nThis is a Labrador Retriever."

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch(self):
        model = InternVL2_5ForConditionalGeneration.from_pretrained(
            "thisisiron/InternVL2_5-1B", torch_dtype="auto", device_map="auto"
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text, text], images=[self.image, self.image], return_tensors="pt").to(
            torch_device, torch.bfloat16
        )

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            'system\nYou are a helpful assistant.\nuser\n\nWhat kind of dog is this?\nassistant\nThis is a Labrador Retriever.',
            'system\nYou are a helpful assistant.\nuser\n\nWhat kind of dog is this?\nassistant\nThis is a Labrador Retriever.'
        ]  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch_different_resolutions(self):
        model = InternVL2_5ForConditionalGeneration.from_pretrained(
            "thisisiron/InternVL2_5-1B", torch_dtype="auto", device_map="auto"
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        text2 = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        image2 = self.image.resize((224, 224))
        inputs = self.processor(
            text=[text, text2],
            images=[self.image, image2],
            padding=True,
            return_tensors="pt",
        ).to(torch_device, torch.bfloat16)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            "system\nYou are a helpful assistant.\nuser\n\nWhat kind of dog is this?\nassistant\nThis is a Labrador Retriever.",
            "system\nYou are a helpful assistant.\nuser\n\nWhat kind of dog is this?\nassistant\nThe dog in the image appears to be a Labrador Retriever. Labrador Retrievers are known for their friendly and outgoing personalities, making them popular pets",
        ]
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_flash_attn
    @require_torch_gpu
    def test_small_model_integration_test_batch_flashatt2(self):
        model = InternVL2_5ForConditionalGeneration.from_pretrained(
            "thisisiron/InternVL2_5-1B",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text, text], images=[self.image, self.image], return_tensors="pt").to(
            torch_device, torch.bfloat16
        )

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30)

        EXPECTED_DECODED_TEXT = [
            "system\nYou are a helpful assistant.\nuser\n\nWhat kind of dog is this?\nassistant\nThis is a Labrador Retriever.",
            "system\nYou are a helpful assistant.\nuser\n\nWhat kind of dog is this?\nassistant\nThis is a Labrador Retriever.",
        ]

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True)[0],
            self.processor.batch_decode(output, skip_special_tokens=True)[1],
        )
