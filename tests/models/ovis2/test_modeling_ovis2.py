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

import unittest

import requests

from transformers import (
    AutoProcessor,
    Ovis2Config,
    Ovis2ForConditionalGeneration,
    Ovis2Model,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


class Ovis2VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        seq_length=7,
        text_config={
            "model_type": "qwen2",
            "seq_length": 7,
            "is_training": True,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 54,
            "hidden_act": "gelu",
            "max_position_embeddings": 580,
            "initializer_range": 0.02,
            "num_labels": 3,
            "pad_token_id": 0,
        },
        is_training=True,
        vision_config={
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
            "hidden_size": 64,
            "vocab_size": 99,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 54,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "qkv_bias": False,
            "hidden_stride": 2,
            "tokenize_function": "softmax",
        },
        image_token_id=1,
        visual_indicator_token_ids=[],
        vocab_size=99,
        hidden_size=64,
        ignore_id=-100,
    ):
        self.parent = parent
        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.visual_indicator_token_ids = visual_indicator_token_ids
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.image_seq_length = (
            vision_config["image_size"] // (vision_config["patch_size"] * vision_config["hidden_stride"])
        ) ** 2
        self.seq_length = seq_length + self.image_seq_length
        self.is_training = is_training
        self.num_attention_heads = text_config["num_attention_heads"]

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.pad_token_id = text_config["pad_token_id"]
        self.ignore_id = ignore_id

        self.batch_size = 3
        self.num_channels = 3

    def get_config(self):
        return Ovis2Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            visual_indicator_token_ids=self.visual_indicator_token_ids,
            vocab_size=self.vocab_size,
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

        vocab_range = self.vocab_size - 2
        input_ids = ids_tensor([self.batch_size, self.seq_length], vocab_range) + 2
        input_ids[:, : self.image_seq_length] = config.image_token_id

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)

        labels = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)
        labels[:, : self.image_seq_length] = self.ignore_id

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return config, inputs_dict


@require_torch
class Ovis2ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Ovis2ForConditionalGeneration`.
    """

    all_model_classes = (
        (
            Ovis2Model,
            Ovis2ForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = {"image-text-to-text": Ovis2ForConditionalGeneration} if is_torch_available() else {}
    _is_composite = True
    test_pruning = False
    test_torchscript = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = Ovis2VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Ovis2Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

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


@require_torch
@slow
class Ovis2IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(
            "thisisiron/Ovis2-2B-hf",
        )
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        self.image = Image.open(requests.get(url, stream=True).raw)
        self.prompt_image = ""
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What do you see in this image?"},
                ],
            }
        ]
        self.text = self.processor.apply_chat_template(self.messages, add_generation_prompt=True, tokenize=False)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_small_model_integration_test(self):
        model = Ovis2ForConditionalGeneration.from_pretrained(
            "thisisiron/Ovis2-2B-hf", dtype="bfloat16", device_map=torch_device
        )

        inputs = self.processor(images=self.image, text=self.text, return_tensors="pt").to(
            torch_device, torch.bfloat16
        )

        self.assertTrue(inputs.input_ids.shape[1] == 1314)  # should expand num-image-tokens times
        self.assertTrue(inputs.pixel_values.shape == torch.Size([5, 3, 448, 448]))

        inputs = inputs.to(torch_device)

        output = model.generate(**inputs, max_new_tokens=64)
        EXPECTED_DECODED_TEXT = 'system\nYou are a helpful assistant.\nuser\n\nWhat do you see in this image?\nassistant\nI see two cats lying on a pink blanket. There are also two remote controls on the blanket.'  # fmt: skip
        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_batch(self):
        model = Ovis2ForConditionalGeneration.from_pretrained(
            "thisisiron/Ovis2-2B-hf", dtype="bfloat16", device_map=torch_device
        )

        inputs = self.processor(
            text=[self.text],
            images=self.image,
            return_tensors="pt",
            padding=True,
        ).to(torch_device, torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ['system\nYou are a helpful assistant.\nuser\n\nWhat do you see in this image?\nassistant\nI see two cats lying on a pink blanket. There are also two remote controls on the blanket.']  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_multi_image(self):
        # related to (#29835)
        model = Ovis2ForConditionalGeneration.from_pretrained(
            "thisisiron/Ovis2-2B-hf",
            dtype="bfloat16",
            device_map=torch_device,
        )

        url = "http://images.cocodataset.org/val2014/COCO_val2014_000000537955.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "What do you see in these images?"},
                ],
            }
        ]
        text = self.processor.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, images=[self.image, image], return_tensors="pt").to(
            torch_device, torch.bfloat16
        )

        output = model.generate(**inputs, max_new_tokens=40)
        EXPECTED_DECODED_TEXT = 'system\nYou are a helpful assistant.\nuser\n\n\nWhat do you see in these images?\nassistant\nIn the first image, I see two cats lying on a pink blanket with remote controls nearby. The second image shows a dog standing on a wooden floor near a kitchen cabinet.'  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_batch_different_resolutions(self):
        model = Ovis2ForConditionalGeneration.from_pretrained(
            "thisisiron/Ovis2-2B-hf", dtype="bfloat16", device_map=torch_device
        )

        lowres_url = "http://images.cocodataset.org/val2014/COCO_val2014_000000537955.jpg"
        lowres_img = Image.open(requests.get(lowres_url, stream=True).raw).resize((320, 240))

        inputs = self.processor(
            text=[self.text, self.text],
            images=[lowres_img, self.image],
            return_tensors="pt",
            padding=True,
        ).to(torch_device, torch.bfloat16)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = [
            'system\nYou are a helpful assistant.\nuser\n\nWhat do you see in this image?\nassistant\nAnswer: I see a brown dog standing on a wooden floor in what appears to be a kitchen.',
            'system\nYou are a helpful assistant.\nuser\n\nWhat do you see in this image?\nassistant\nI see two cats lying on a pink blanket. There are also two remote controls on the blanket.'
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_batch_matches_single(self):
        model = Ovis2ForConditionalGeneration.from_pretrained(
            "thisisiron/Ovis2-2B-hf",
            dtype="bfloat16",
            device_map=torch_device,
        )

        lowres_url = "https://4.img-dpreview.com/files/p/TS560x560~forums/56876524/03975b28741443319e9a94615e35667e"
        lowres_img = Image.open(requests.get(lowres_url, stream=True).raw)

        inputs_batched = self.processor(
            text=[self.text, self.text],
            images=[self.image, lowres_img],
            return_tensors="pt",
            padding=True,
        ).to(torch_device, torch.bfloat16)

        inputs_single = self.processor(text=self.text, images=self.image, return_tensors="pt", padding=True).to(
            torch_device, torch.bfloat16
        )

        output_batched = model.generate(**inputs_batched, max_new_tokens=50)
        output_single = model.generate(**inputs_single, max_new_tokens=50)

        self.assertEqual(
            self.processor.decode(output_batched[0], skip_special_tokens=True),
            self.processor.decode(output_single[0], skip_special_tokens=True),
        )
