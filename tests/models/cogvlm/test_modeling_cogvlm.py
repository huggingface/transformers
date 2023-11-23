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

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import CogVLMForCausalLM, CogVLMModel
    from transformers.models.cogvlm.modeling_cogvlm import COGVLM_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import CogVLMProcessor


class CogVLMVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=1e-10,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return CogVLMVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values):
        model = CogVLMVisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class CogVLMVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as CogVLM's vision encoder does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (CogVLMVisionModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = CogVLMVisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=CogVLMVisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="CogVLM's vision encoder does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_training(self):
        pass

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

    @unittest.skip(reason="CogVLMVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="CogVLMVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in COGVLM_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CogVLMVisionModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# this model tester uses an encoder-decoder language model (T5)
class CogVLMModelTester:
    def __init__(self, parent, vision_kwargs=None, text_kwargs=None, is_training=True, num_query_tokens=10):
        if vision_kwargs is None:
            vision_kwargs = {}
        if text_kwargs is None:
            text_kwargs = {}

        self.parent = parent
        self.model_ster = CogVLMModelTester(parent, **vision_kwargs)
        self.is_training = is_training
        self.num_query_tokens = num_query_tokens

    def prepare_config_and_inputs(self):
        _, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        (
            _,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = self.text_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values, decoder_input_ids, decoder_attention_mask, lm_labels

    def get_config(self):
        return CogVLMConfig.from_vision_text_configs(
            vision_config=self.vision_model_tester.get_config(),
            text_config=self.text_model_tester.get_config(),
            num_query_tokens=self.num_query_tokens,
        )

    def create_and_check_for_causal_lm(
        self, config, input_ids, attention_mask, pixel_values, decoder_input_ids, decoder_attention_mask, labels
    ):
        model = CogVLMForCausalLM(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(pixel_values, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)

        self.parent.assertEqual(
            result.logits.shape,
            (
                self.vision_model_tester.batch_size,
                self.text_model_tester.seq_length,
                self.text_model_tester.vocab_size,
            ),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            pixel_values,
            decoder_input_ids,
            decoder_attention_mask,
            labels,
        ) = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }
        return config, inputs_dict


@require_torch
class CogVLMModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (CogVLMForCausalLM, CogVLMModel) if is_torch_available() else ()
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = CogVLMModelTester(self)

    def test_for_conditional_generation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_conditional_generation(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="CogVLMModel does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="There's no base CogVLMModel")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="There's no base CogVLMModel")
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_cpu_offload(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save CogVLMConfig and check if we can load CogVLMVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = CogVLMVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        for model_name in COGVLM_PRETRAINED_MODEL_ARCHIVE_LIST:
            model = CogVLMForCausalLM.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_get_text_features(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        inputs_dict = {
            "input_ids": torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(torch_device),
            "attention_mask": torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(torch_device),
            "decoder_input_ids": torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(torch_device),
        }

        model = CogVLMModel(config).to(torch_device)
        model.eval()
        text_features = model.get_text_features(**inputs_dict)
        self.assertEqual(text_features[0].shape, (1, 10, config.text_config.vocab_size))

    def test_get_image_features(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        keys_to_pop = ["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"]

        for key in keys_to_pop:
            inputs_dict.pop(key)

        model = CogVLMModel(config).to(torch_device)
        model.eval()
        image_features = model.get_image_features(**inputs_dict)
        self.assertEqual(
            image_features[0].shape,
            (
                self.model_tester.vision_model_tester.batch_size,
                self.model_tester.vision_model_tester.seq_length,
                config.vision_config.hidden_size,
            ),
        )

    # override from common to deal with nested configurations (`vision_config` and `text_config`)
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for key in ["vision_config", "text_config"]:
            setattr(configs_no_init, key, _config_zero_init(getattr(configs_no_init, key)))
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )


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
