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
"""Testing suite for the PyTorch Florence2 model."""

import unittest

import requests

from transformers import (
    AutoProcessor,
    Florence2Config,
    Florence2ForConditionalGeneration,
    Florence2Model,
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
else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class Florence2VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        num_channels=3,
        image_size=8,
        seq_length=13,
        encoder_seq_length=18,
        is_training=True,
        vocab_size=99,
        max_position_embeddings=64,
        encoder_layers=1,
        encoder_ffn_dim=8,
        decoder_layers=1,
        decoder_ffn_dim=8,
        num_attention_heads=1,
        d_model=8,
        activation_function="gelu",
        dropout=0.1,
        eos_token_id=2,
        bos_token_id=0,
        pad_token_id=1,
        depths=[1],
        patch_size=[7],
        patch_stride=[4],
        patch_padding=[3],
        patch_prenorm=[False],
        embed_dim=[8],
        num_heads=[1],
        num_groups=[1],
        window_size=12,
        drop_path_rate=0.1,
        projection_dim=8,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.seq_length = seq_length
        self.encoder_seq_length = encoder_seq_length
        self.is_training = is_training
        self.num_hidden_layers = decoder_layers
        self.hidden_size = d_model

        # Language model configs
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.encoder_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_attention_heads = num_attention_heads
        self.d_model = d_model
        self.activation_function = activation_function
        self.dropout = dropout
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id

        # Vision model configs
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size
        self.depths = depths
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.patch_prenorm = patch_prenorm
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.window_size = window_size
        self.projection_dim = projection_dim

    def get_config(self):
        text_config = {
            "model_type": "bart",
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "encoder_layers": self.encoder_layers,
            "encoder_ffn_dim": self.encoder_ffn_dim,
            "encoder_attention_heads": self.num_attention_heads,
            "decoder_layers": self.decoder_layers,
            "decoder_ffn_dim": self.decoder_ffn_dim,
            "decoder_attention_heads": self.num_attention_heads,
            "d_model": self.d_model,
            "activation_function": self.activation_function,
            "dropout": self.dropout,
            "attention_dropout": self.dropout,
            "activation_dropout": self.dropout,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
        }

        vision_config = {
            "drop_path_rate": self.drop_path_rate,
            "patch_size": self.patch_size,
            "depths": self.depths,
            "patch_stride": self.patch_stride,
            "patch_padding": self.patch_padding,
            "patch_prenorm": self.patch_prenorm,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_groups": self.num_groups,
            "window_size": self.window_size,
            "activation_function": self.activation_function,
            "projection_dim": self.projection_dim,
        }

        return Florence2Config(text_config=text_config, vision_config=vision_config)

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.num_channels,
                self.image_size,
                self.image_size,
            ]
        )
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(
            3,
        )
        input_ids[:, -1] = self.eos_token_id
        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        inputs_dict = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
        }

        config = self.get_config()
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_florence2_model_fp16_forward(self, config, input_ids, pixel_values, attention_mask):
        model = Florence2ForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values.to(torch.float16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())

    @unittest.skip(
        reason="This architecture (bart) has tied weights by default and there is no way to remove it, check: https://github.com/huggingface/transformers/pull/31771#issuecomment-2210915245"
    )
    def test_load_save_without_tied_weights(self):
        pass


@require_torch
class Florence2ForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Florence2ForConditionalGeneration`.
    """

    all_model_classes = (Florence2ForConditionalGeneration, Florence2Model) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "image-to-text": Florence2ForConditionalGeneration,
            "image-text-to-text": Florence2ForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_head_masking = False
    test_attention_outputs = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Florence2VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Florence2Config, has_text_modality=False)

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

    @unittest.skip(
        reason="This architecture has tied weights by default and there is no way to remove it, check: https://github.com/huggingface/transformers/pull/31771#issuecomment-2210915245"
    )
    def test_load_save_without_tied_weights(self):
        pass


def prepare_img():
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


@require_vision
@require_torch
@slow
class Florence2ForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_name = "microsoft/Florence-2-base"
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.image1 = Image.open(
            requests.get(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg?download=true",
                stream=True,
            ).raw
        )
        self.image2 = Image.open(
            requests.get(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true",
                stream=True,
            ).raw
        )

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_inference_base(self):
        model = Florence2ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.float16).to(
            torch_device
        )

        prompt = "<DETAILED_CAPTION>"
        inputs = self.processor(images=self.image1, text=prompt, return_tensors="pt")
        inputs.to(device=torch_device, dtype=torch.float16)

        EXPECTED_INPUT_IDS = [[0, 47066, 21700, 11, 4617, 99, 16, 2343, 11, 5, 2274, 4, 2]]  # fmt: skip
        self.assertTrue(inputs["input_ids"].tolist(), EXPECTED_INPUT_IDS)

        predictions = model.generate(**inputs, max_new_tokens=100)

        EXPECTED_PREDICTION_IDS = [[2, 0, 133, 2274, 924, 10, 912, 1203, 2828, 15, 5, 526, 9, 10, 2014, 11, 35910, 6, 188, 469, 412, 4, 20, 2014, 16, 9321, 19, 3413, 6, 3980, 6, 8, 19638, 6, 8, 89, 32, 82, 3051, 15, 5, 2767, 22609, 4, 20, 6360, 16, 7097, 11, 5, 3618, 4, 2]]  # fmt: skip
        self.assertTrue(predictions.tolist(), EXPECTED_PREDICTION_IDS)

        generated_text = self.processor.batch_decode(predictions, skip_special_tokens=True)[0]

        EXPECTED_GENERATED_TEXT = "The image shows a stop sign sitting on the side of a street in Chinatown, New York City. The street is lined with buildings, trees, and statues, and there are people walking on the footpath. The sky is visible in the background."  # fmt: skip
        self.assertEqual(generated_text, EXPECTED_GENERATED_TEXT)

    def test_batch_inference_base(self):
        model = Florence2ForConditionalGeneration.from_pretrained(
            self.model_name, attn_implementation="eager", torch_dtype=torch.float16
        ).to(torch_device)

        images = [self.image1, self.image2]
        prompts = ["<CAPTION>", "<DETAILED_CAPTION>"]
        inputs = self.processor(images=images, text=prompts, padding="longest", return_tensors="pt")

        EXPECTED_INPUT_IDS = [
            [0,  2264,   473,  5, 2274, 6190, 116,    2,  1, 1,    1, 1, 1],
            [0, 47066, 21700, 11, 4617,   99,  16, 2343, 11, 5, 2274, 4, 2],
        ]  # fmt: skip
        self.assertTrue(inputs["input_ids"].tolist(), EXPECTED_INPUT_IDS)

        inputs.to(device=torch_device, dtype=torch.float16)
        print(inputs)
        predictions = model.generate(**inputs, max_new_tokens=100)

        EXPECTED_PREDICTION_IDS = [
            [2, 0, 250,  912, 1203, 2828,   15,     5,   526,    9, 10, 2014, 4,  2,    1,   1,  1,  1,    1,    1, 1,    1,  1,    1, 1,  1,   1,    1,    1, 1, 1],
            [2, 0, 133, 2274,  924,   10, 2272, 10685, 41537, 9181, 11,  760, 9, 10, 5718, 745, 19, 80, 6219, 4259, 6, 7501, 30, 3980, 8, 10, 699, 2440, 6360, 4, 2]
        ]  # fmt: skip
        self.assertTrue(predictions.tolist(), EXPECTED_PREDICTION_IDS)

        generated_texts = self.processor.batch_decode(predictions, skip_special_tokens=True)

        EXPECTED_GENERATED_TEXTS = [
            "A stop sign sitting on the side of a street.",
            "The image shows a green Volkswagen Beetle parked in front of a yellow building with two brown doors, surrounded by trees and a clear blue sky.",
        ]  # fmt: skip
        self.assertEqual(generated_texts, EXPECTED_GENERATED_TEXTS)
