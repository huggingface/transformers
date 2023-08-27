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
""" Testing suite for the PyTorch CLVP model. """


import copy
import inspect
import os
import tempfile
import unittest

import datasets

import numpy as np

from transformers import CLVPConfig, CLVPSpeechConfig, CLVPTextConfig, CLVPAutoRegressiveConfig, CLVPFeatureExtractor
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    ids_tensor,
    random_attention_mask,
)

from ...generation.test_utils import GenerationTesterMixin

if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        CLVPModel,
        CLVPTransformerWithProjection,
    )
    from transformers.models.clvp.modeling_clvp import CLVP_PRETRAINED_MODEL_ARCHIVE_LIST


class CLVPTransformerWithProjectionTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

    def get_config(self):
        # we are only checking with speech config though both of the configs have same attributes
        speech_config = CLVPSpeechConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

        return speech_config

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        speech_config = self.get_config()

        return speech_config, input_ids, input_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        speech_config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids.to(torch_device), "attention_mask": input_mask.to(torch_device)}
        return speech_config, inputs_dict

    def create_and_check_model(self, speech_config, input_ids, input_mask):
        # check the model with both type of inputs
        text_config = CLVPTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )
        text_model = CLVPTransformerWithProjection(config=text_config)
        text_model.to(torch_device)
        text_model.eval()
        with torch.no_grad():
            result = text_model(input_ids, attention_mask=input_mask)
            result = text_model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.text_embeds.shape, (self.batch_size, self.projection_dim))

        # now check with speech config
        speech_model = CLVPTransformerWithProjection(config=speech_config)
        speech_model.to(torch_device)
        speech_model.eval()
        with torch.no_grad():
            result = speech_model(input_ids, attention_mask=input_mask)
            result = speech_model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.speech_embeds.shape, (self.batch_size, self.projection_dim))


@require_torch
class CLVPTransformerWithProjectionTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (CLVPTransformerWithProjection, ) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = CLVPTransformerWithProjectionTester(self)
        self.text_config_tester = ConfigTester(self, config_class=CLVPTextConfig, hidden_size=64)
        self.speech_config_tester = ConfigTester(self, config_class=CLVPSpeechConfig, hidden_size=64)

    def test_config(self):
        self.text_config_tester.run_common_tests()
        self.speech_config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="CLVPTextModelWithProjection does not output loss")
    def test_training(self):
        pass

    @unittest.skip(reason="CLVPTextModelWithProjection does not output loss")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="CLVP does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLVP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLVPTransformerWithProjection.from_pretrained(model_name)
            self.assertIsNotNone(model)


class CLVPModelTester:
    def __init__(self, parent, is_training=True):
        self.parent = parent
        self.transformer_projection_model_tester = CLVPTransformerWithProjectionTester(parent)
        self.is_training = is_training

    def get_config(self):
        autoregressive_config = CLVPAutoRegressiveConfig(vocab_size=99,
                                                         max_mel_tokens=100,
                                                         max_text_tokens=100,
                                                         n_embd=32,
                                                         n_layer=2,
                                                         n_head=2,
                                                         bos_token_id=97,
                                                         eos_token_id=98,
                                                         relative_attention_num_buckets=4,
                                                         relative_attention_max_distance=16,
                                                         )

        return CLVPConfig.from_text_speech_autoregressive_configs(
            self.transformer_projection_model_tester.get_config(),# text config
            self.transformer_projection_model_tester.get_config(),# text config used as speech config as they have same attributes
            autoregressive_config, # autoregressive config
            projection_dim=64
        )

    def prepare_config_and_inputs(self):
        _, input_ids, attention_mask = self.transformer_projection_model_tester.prepare_config_and_inputs()

        ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", datasets.Audio(sampling_rate=22050))
        _, audio, sr = ds.sort("id").select(range(1))[:1]["audio"][0].values()

        feature_extractor = CLVPFeatureExtractor()
        input_features = feature_extractor(raw_speech=audio, sampling_rate=sr, return_tensors="pt")["input_features"].to(torch_device)

        config = self.get_config()

        return config, input_ids, attention_mask, input_features

    def create_and_check_model(self, config, input_ids, attention_mask, input_features):
        model = CLVPModel(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, input_features=input_features, attention_mask=attention_mask)

        self.parent.assertEqual(
            result.logits_per_speech.shape, (2, self.transformer_projection_model_tester.batch_size)
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, (self.transformer_projection_model_tester.batch_size, 2)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, input_features = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids.to(torch_device),
            "attention_mask": attention_mask.to(torch_device),
            "input_features": input_features.to(torch_device),
            "return_loss": False,
            # "return_dict": True,
        }
        return config, inputs_dict


@require_torch
class CLVPModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (CLVPModel,) if is_torch_available() else ()
    # all_generative_model_classes = (CLVPModel, ) if is_torch_available() else ()

    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = CLVPModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="CLVPModel does not output Hidden_states, since it has two types(text and speech) of them")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="CLVPModel does not take inputs_embeds as inputs")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="CLVPModel does not have input/output embeddings, since it has two types(text and speech) of them")
    def test_model_common_attributes(self):
        pass

    # override as the `logit_scale` parameter initilization is different for CLVP
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # check if `logit_scale` is initilized as per the original implementation
                    if name == "logit_scale":
                        self.assertAlmostEqual(
                            param.data.item(),
                            np.log(1 / 0.07),
                            delta=1e-3,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def test_load_speech_text_autoregressive_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save CLVPConfig and check if we can load CLVPSpeechConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            speech_config = CLVPSpeechConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.speech_config.to_dict(), speech_config.to_dict())

        # Save CLVPConfig and check if we can load CLVPTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = CLVPTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

        # Save CLVPConfig and check if we can load CLVPAutoRegressiveConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            autoregressive_config = CLVPAutoRegressiveConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.autoregressive_config.to_dict(), autoregressive_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLVP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLVPModel.from_pretrained(model_name)
            self.assertIsNotNone(model)



# Since CLVP has a lot of different models connected with each other it's better to test each of them individually
# along with a test_full_model_integration. If the model breaks in future, it could be of a great help to identify the
# broken part.

@require_torch
class CLVPModelIntegrationTest(unittest.TestCase):
    @slow
    def test_conditional_encoder(self):
        # model_name = "susnato/clvp_dev"
        # model = CLVPModel.from_pretrained(model_name).to(torch_device)
        # model.eval()
        #
        # text = torch.tensor([[5, 241, 41, 22, 39, 105, 98], [8, 95, 46, 45, 159, 54, 6]]).long().to(torch_device)
        # speech = torch.tensor([[11, 255, 25, 57, 10, 7, 41], [9, 20, 226, 15, 5, 97, 32]]).long().to(torch_device)
        # inputs = {"input_ids": text, "speech_ids": speech}
        #
        # # forward pass
        # with torch.no_grad():
        #     outputs = model(**inputs)
        #
        # # verify the logits
        # self.assertEqual(
        #     outputs.logits_per_speech.shape,
        #     torch.Size((inputs["speech_ids"].shape[0], inputs["input_ids"].shape[0])),
        # )
        # self.assertEqual(
        #     outputs.logits_per_text.shape,
        #     torch.Size((inputs["input_ids"].shape[0], inputs["speech_ids"].shape[0])),
        # )
        #
        # expected_logits = torch.tensor([[32.028324, 11.421426], [4.789056, 7.113933]], device=torch_device)
        #
        # self.assertTrue(torch.allclose(outputs.logits_per_speech, expected_logits, atol=1e-3))
        pass

    @slow
    def test_autoregressive_model_generate(self):
        pass

    @slow
    def test_speech_and_text_projection_models(self):
        pass

    @slow
    def test_full_model_integration(self):
        pass

