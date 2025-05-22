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

import gc
import unittest

import requests

from transformers import (
    AutoProcessor,
    Florence2Config,
    Florence2ForConditionalGeneration,
    Florence2LanguageConfig,
    Florence2VisionConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
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
        use_labels=False,
        vocab_size=99,
        max_position_embeddings=64,
        encoder_layers=1,
        encoder_ffn_dim=8,
        decoder_layers=1,
        decoder_ffn_dim=8,
        num_attention_heads=1,
        d_model=8,
        hidden_act="gelu",
        dropout=0.1,
        eos_token_id=2,
        bos_token_id=0,
        pad_token_id=1,
        depths=[1],
        patch_size=[7],
        patch_stride=[4],
        patch_padding=[3],
        patch_prenorm=[False],
        dim_embed=[8],
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
        self.activation_function = hidden_act
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
        self.dim_embed = dim_embed
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.window_size = window_size
        self.projection_dim = projection_dim

    def get_config(self):
        text_config = Florence2LanguageConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            encoder_layers=self.encoder_layers,
            encoder_ffn_dim=self.encoder_ffn_dim,
            encoder_attention_heads=self.num_attention_heads,
            decoder_layers=self.decoder_layers,
            decoder_ffn_dim=self.decoder_ffn_dim,
            decoder_attention_heads=self.num_attention_heads,
            d_model=self.d_model,
            activation_function=self.activation_function,
            dropout=self.dropout,
            attention_dropout=self.dropout,
            activation_dropout=self.dropout,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
        )

        vision_config = Florence2VisionConfig(
            drop_path_rate=self.drop_path_rate,
            patch_size=self.patch_size,
            depths=self.depths,
            patch_stride=self.patch_stride,
            patch_padding=self.patch_padding,
            patch_prenorm=self.patch_prenorm,
            dim_embed=self.dim_embed,
            num_heads=self.num_heads,
            num_groups=self.num_groups,
            window_size=self.window_size,
            projection_dim=self.projection_dim,
        )

        return Florence2Config.from_text_vision_configs(text_config=text_config, vision_config=vision_config)

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

    additional_model_inputs = ["pixel_values"]
    all_model_classes = (Florence2ForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (Florence2ForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-to-text": Florence2ForConditionalGeneration} if is_torch_available() else {}
    test_pruning = False
    test_head_masking = False
    test_attention_outputs = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = Florence2VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Florence2Config, has_text_modality=False)

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

    # @unittest.skip(
    #     reason="This architecture has tied weights by default and there is no way to remove it, check: https://github.com/huggingface/transformers/pull/31771#issuecomment-2210915245"
    # )
    # def test_contrastive_generate_low_memory(self):
    #     pass

    @unittest.skip(
        reason="This architecture has tied weights by default and there is no way to remove it, check: https://github.com/huggingface/transformers/pull/31771#issuecomment-2210915245"
    )
    def test_load_save_without_tied_weights(self):
        pass


@require_torch
class Florence2ForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base")

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    def test_small_model_integration_test(self):
        model = Florence2ForConditionalGeneration.from_pretrained("microsoft/Florence-2-base")

        prompt = "<CAPTION>"
        image_file = (
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
        )
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = self.processor(images=raw_image, text=prompt, return_tensors="pt")

        EXPECTED_INPUT_IDS = torch.tensor([[0, 2264, 473, 5, 2274, 6190, 116, 2]])  # fmt: skip
        self.assertTrue(torch.equal(inputs["input_ids"], EXPECTED_INPUT_IDS))

        output = model.generate(**inputs, max_new_tokens=20)
        EXPECTED_DECODED_TEXT = "A green car parked in front of a yellow building."  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_florence_single(self):
        model_id = "microsoft/Florence-2-base"

        model = Florence2ForConditionalGeneration.from_pretrained("microsoft/Florence-2-base")
        processor = AutoProcessor.from_pretrained(model_id)

        prompt = "<CAPTION>"
        image_file = (
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
        )
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = processor(images=raw_image, text=prompt, return_tensors="pt")

        output = model.generate(**inputs, max_new_tokens=900, do_sample=False)
        EXPECTED_DECODED_TEXT = "A green car parked in front of a yellow building."  # fmt: skip

        self.assertEqual(
            processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch(self):
        model = Florence2ForConditionalGeneration.from_pretrained("microsoft/Florence-2-base")
        prompts = [
            "<CAPTION>",
            "<CAPTION>",
        ]
        image1 = Image.open(
            requests.get(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
                stream=True,
            ).raw
        )
        image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = self.processor(images=[image1, image2], text=prompts, return_tensors="pt", padding=True)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ['A green car parked in front of a yellow building.', 'Two cats laying on a pink couch next to a remote control.']  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_torch
    @require_vision
    def test_batched_generation(self):
        model = Florence2ForConditionalGeneration.from_pretrained("microsoft/Florence-2-base")

        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base")

        prompt1 = "<CAPTION>"
        prompt2 = "<CAPTION>"
        prompt3 = "<CAPTION>"
        url1 = "https://images.unsplash.com/photo-1552053831-71594a27632d?q=80&w=3062&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        url2 = "https://images.unsplash.com/photo-1617258683320-61900b281ced?q=80&w=3087&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        image1 = Image.open(requests.get(url1, stream=True).raw)
        image2 = Image.open(requests.get(url2, stream=True).raw)

        inputs = processor(
            images=[image1, image2, image1, image2],
            text=[prompt1, prompt1, prompt2, prompt3],
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        model = model.eval()

        EXPECTED_OUTPUT = [
            "A dog sitting on a patio holding a flower in its mouth.",
            "A baby llama standing on top of a hill.",
            "A dog sitting on a patio holding a flower in its mouth.",
            "A baby llama standing on top of a hill.",
        ]

        generate_ids = model.generate(**inputs, max_new_tokens=20)
        outputs = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.assertEqual(outputs, EXPECTED_OUTPUT)
