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
""" Testing suite for the PyTorch Llava-NeXT model. """

import copy
import gc
import unittest

import requests
from huggingface_hub import hf_hub_download

from transformers import (
    AutoProcessor,
    LlavaNextConfig,
    LlavaNextForConditionalGeneration,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import require_bitsandbytes, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor


if is_torch_available():
    import torch
else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class LlavaNextVisionText2TextModelTester:
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
            "max_position_embeddings": 580,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 0,
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
        self.image_size = 30
        self.encoder_seq_length = 342
        self.image_grid_pinpoints = [[32, 32]]

    def get_config(self):
        return LlavaNextConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
            projector_hidden_act=self.projector_hidden_act,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layer=self.vision_feature_layer,
            image_grid_pinpoints=self.image_grid_pinpoints,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                5,
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
            "image_sizes": torch.tensor(
                [[self.vision_config["image_size"], self.vision_config["image_size"]]] * self.batch_size
            ),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class LlavaNextForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `LlavaNextForConditionalGeneration`.
    """

    all_model_classes = (LlavaNextForConditionalGeneration,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = LlavaNextVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LlavaNextConfig, has_text_modality=False)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if "image_newline" in name:
                    continue
                elif param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

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

    @unittest.skip(reason="Feedforward chunking is not yet supported")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="CPU offload is not yet supported")
    def test_cpu_offload(self):
        pass

    # Copied from tests.test_modeling_common.ModelTesterMixin.test_resize_tokens_embeddings with config.vocab_size->config.text_config.vocab_size
    def test_resize_tokens_embeddings(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config.text_config.vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 1)

            # make sure that decoder_input_ids are resized as well
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            model_vocab_size = config.text_config.vocab_size
            model.resize_token_embeddings(model_vocab_size + 10, pad_to_multiple_of=1)
            self.assertTrue(model.config.text_config.vocab_size + 10, model_vocab_size)

            model_embed = model.resize_token_embeddings(model_vocab_size, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0] // 64, 0)

            self.assertTrue(model_embed.weight.shape[0], model.config.text_config.vocab_size)
            self.assertTrue(model.config.text_config.vocab_size, model.vocab_size)

            model_embed = model.resize_token_embeddings(model_vocab_size + 13, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0] // 64, 0)

            # Check that resizing a model to a multiple of pad_to_multiple leads to a model of exactly that size
            target_dimension = 128
            model_embed = model.resize_token_embeddings(target_dimension, pad_to_multiple_of=64)
            self.assertTrue(model_embed.weight.shape[0], target_dimension)

            with self.assertRaisesRegex(
                ValueError,
                "Asking to pad the embedding matrix to a multiple of `1.3`, which is not and integer. Please make sure to pass an integer",
            ):
                model.resize_token_embeddings(model_vocab_size, pad_to_multiple_of=1.3)

    # Copied from tests.test_modeling_common.ModelTesterMixin.test_resize_embeddings_untied with config.vocab_size->config.text_config.vocab_size
    def test_resize_embeddings_untied(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return

        original_config.tie_word_embeddings = False

        # if model cannot untied embeddings -> leave test
        if original_config.tie_word_embeddings:
            return

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config).to(torch_device)

            # if no output embeddings -> leave test
            if model.get_output_embeddings() is None:
                continue

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_vocab_size = config.text_config.vocab_size
            model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size + 10)
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size + 10)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.text_config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size - 15)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size - 15)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            if "decoder_input_ids" in inputs_dict:
                inputs_dict["decoder_input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

    # Copied from tests.test_modeling_common.ModelTesterMixin.test_tie_model_weights with config.vocab_size->config.text_config.vocab_size
    def test_tie_model_weights(self):
        if not self.test_torchscript:
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_same_values(layer_1, layer_2):
            equal = True
            for p1, p2 in zip(layer_1.weight, layer_2.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    equal = False
            return equal

        for model_class in self.all_model_classes:
            config.torchscript = True
            model_not_tied = model_class(config)
            if model_not_tied.get_output_embeddings() is None:
                continue

            config_tied = copy.deepcopy(config)
            config_tied.torchscript = False
            model_tied = model_class(config_tied)
            params_tied = list(model_tied.parameters())
            # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(check_same_values(embeddings, decoding))

            # Check that after resize they remain tied.
            model_tied.resize_token_embeddings(config.text_config.vocab_size + 10)
            params_tied_2 = list(model_tied.parameters())
            self.assertEqual(len(params_tied_2), len(params_tied))


@require_torch
class LlavaNextForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
        self.image = Image.open(requests.get(url, stream=True).raw)

        self.prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test(self):
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            load_in_4bit=True,
        )

        inputs = self.processor(self.prompt, self.image, return_tensors="pt")

        # verify inputs against original implementation
        filepath = hf_hub_download(repo_id="nielsr/test-image", filename="llava_1_6_input_ids.pt", repo_type="dataset")
        original_input_ids = torch.load(filepath, map_location="cpu")
        # replace -200 by image_token_index (since we use token ID = 32000 for the image token)
        original_input_ids[original_input_ids == -200] = model.config.image_token_index
        assert original_input_ids[0].tolist() == inputs.input_ids[0].tolist()

        filepath = hf_hub_download(
            repo_id="nielsr/test-image", filename="llava_1_6_pixel_values.pt", repo_type="dataset"
        )
        original_pixel_values = torch.load(filepath, map_location="cpu")
        assert torch.allclose(original_pixel_values, inputs.pixel_values.half())

        # verify single forward pass
        inputs = inputs.to(torch_device)
        with torch.no_grad():
            output = model(**inputs)

        expected_slice = torch.tensor(
            [[-4.7695, -4.5664, -0.2786], [-10.6250, -10.8906, -2.5254], [-6.7383, -7.2461, -0.6787]],
            dtype=torch.float32,
            device=torch_device,
        )
        assert torch.allclose(output.logits[0, :3, :3], expected_slice, atol=1e-3)

        # verify generation
        output = model.generate(**inputs, max_new_tokens=100)
        EXPECTED_DECODED_TEXT = '[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot that displays values for multiple quantitative variables represented on axes starting from the same point. This particular radar chart is showing the performance of various models or systems across different metrics or datasets.\n\nThe chart is divided into several sections, each representing a different model or dataset. The axes represent different metrics or datasets, such as "MMM-Vet," "MMM-Bench," "L'  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_bitsandbytes
    def test_small_model_integration_test_batch(self):
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", load_in_4bit=True
        )
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        cats_image = Image.open(requests.get(url, stream=True).raw)

        inputs = self.processor(
            [self.prompt, self.prompt], images=[self.image, cats_image], return_tensors="pt", padding=True
        ).to(torch_device)

        # make sure image_sizes are the same
        # as otherwise batched generation doesn't work
        inputs.image_sizes[1] = inputs.image_sizes[0]

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ['[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot that displays', '[INST]  \nWhat is shown in this image? [/INST] The image shows two cats lying on a pink surface, which appears to be a couch or a cush']  # fmt: skip
        self.assertEqual(self.processor.batch_decode(output, skip_special_tokens=True), EXPECTED_DECODED_TEXT)
