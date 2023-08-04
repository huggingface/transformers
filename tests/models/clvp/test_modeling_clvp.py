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

import numpy as np

from transformers import CLVPConfig, CLVPSpeechConfig, CLVPTextConfig
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


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        CLVPModel,
        CLVPSpeechModel,
        CLVPSpeechModelWithProjection,
        CLVPTextModel,
        CLVPTextModelWithProjection,
    )
    from transformers.models.clvp.modeling_clvp import CLVP_PRETRAINED_MODEL_ARCHIVE_LIST


class CLVPSpeechModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        vocab_size=30,
        seq_length=10,
        is_training=True,
        hidden_size=32,
        projection_dim=8,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.vocab_size = vocab_size
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

    def prepare_config_and_inputs(self):
        speech_ids = ids_tensor([self.batch_size, self.seq_length], vocab_size=self.vocab_size)
        config = self.get_config()

        return config, speech_ids

    def get_config(self):
        return CLVPSpeechConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, speech_ids):
        model = CLVPSpeechModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(speech_ids)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        seq_len = speech_ids.size()[1]
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, seq_len, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_model_with_projection(self, config, speech_ids):
        model = CLVPSpeechModelWithProjection(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(speech_ids)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        seq_len = speech_ids.size()[1]
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, seq_len, self.hidden_size))
        self.parent.assertEqual(result.speech_embeds.shape, (self.batch_size, self.projection_dim))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, speech_ids = config_and_inputs
        inputs_dict = {"speech_ids": speech_ids}
        return config, inputs_dict


@require_torch
class CLVPSpeechModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as CLVP does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (CLVPSpeechModel, CLVPSpeechModelWithProjection) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = CLVPSpeechModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CLVPSpeechConfig, hidden_size=32)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="CLVP does not use inputs_embeds")
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

            expected_arg_names = ["speech_ids"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_projection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_projection(*config_and_inputs)

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

            model_vocab_size = config.vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["speech_ids"].clamp_(max=model_vocab_size - 15 - 1)

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

    @unittest.skip(reason="CLVPSpeechModelWithProjection does not output loss")
    def test_training(self):
        pass

    @unittest.skip(reason="CLVPSpeechModelWithProjection does not output loss")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="CLVPSpeechModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="CLVPSpeechModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLVP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLVPSpeechModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @slow
    def test_model_with_projection_from_pretrained(self):
        for model_name in CLVP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLVPSpeechModelWithProjection.from_pretrained(model_name)
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, "speech_projection"))


class CLVPTextModelTester:
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
        num_hidden_layers=4,
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

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return CLVPTextConfig(
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

    def create_and_check_model(self, config, input_ids, input_mask):
        model = CLVPTextModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_model_with_projection(self, config, input_ids, input_mask):
        model = CLVPTextModelWithProjection(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.text_embeds.shape, (self.batch_size, self.projection_dim))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class CLVPTextModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (CLVPTextModel, CLVPTextModelWithProjection) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = CLVPTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CLVPTextConfig, hidden_size=64)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_projection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_projection(*config_and_inputs)

    @unittest.skip(reason="CLVPTextModelWithProjection does not output loss")
    def test_training(self):
        pass

    @unittest.skip(reason="CLVPTextModelWithProjection does not output loss")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="CLVP does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="CLVPTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="CLVPTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLVP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLVPTextModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @slow
    def test_model_with_projection_from_pretrained(self):
        for model_name in CLVP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLVPTextModelWithProjection.from_pretrained(model_name)
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, "text_projection"))


class CLVPModelTester:
    def __init__(self, parent, text_kwargs=None, speech_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if speech_kwargs is None:
            speech_kwargs = {}

        self.parent = parent
        self.text_model_tester = CLVPTextModelTester(parent, **text_kwargs)
        self.speech_model_tester = CLVPSpeechModelTester(parent, **speech_kwargs)
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, speech_ids = self.speech_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, speech_ids

    def get_config(self):
        return CLVPConfig.from_text_speech_configs(
            self.text_model_tester.get_config(), self.speech_model_tester.get_config(), projection_dim=64
        )

    def create_and_check_model(self, config, input_ids, text_attention_mask, speech_ids):
        model = CLVPModel(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids, speech_ids, text_attention_mask)
        self.parent.assertEqual(
            result.logits_per_speech.shape, (self.speech_model_tester.batch_size, self.text_model_tester.batch_size)
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, (self.text_model_tester.batch_size, self.speech_model_tester.batch_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, speech_ids = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "text_attention_mask": attention_mask,
            "speech_ids": speech_ids,
            "return_loss": True,
        }
        return config, inputs_dict


@require_torch
class CLVPModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (CLVPModel,) if is_torch_available() else ()
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = CLVPModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="CLVPModel does not have input/output embeddings")
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

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            return

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        configs_no_init.return_dict = False
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()

            try:
                input_ids = inputs_dict["input_ids"]
                speech_ids = inputs_dict["speech_ids"]  # CLVP needs speech_ids
                traced_model = torch.jit.trace(model, (input_ids, speech_ids))
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            model.to(torch_device)
            model.eval()

            loaded_model.to(torch_device)
            loaded_model.eval()

            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()

            non_persistent_buffers = {}
            for key in loaded_model_state_dict.keys():
                if key not in model_state_dict.keys():
                    non_persistent_buffers[key] = loaded_model_state_dict[key]

            loaded_model_state_dict = {
                key: value for key, value in loaded_model_state_dict.items() if key not in non_persistent_buffers
            }

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for i, model_buffer in enumerate(model_buffers):
                    if torch.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break

                self.assertTrue(found_buffer)
                model_buffers.pop(i)

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                p2 = loaded_model_state_dict[layer_name]
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def test_load_speech_text_config(self):
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

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLVP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLVPModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class CLVPModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "susnato/clvp_dev"
        model = CLVPModel.from_pretrained(model_name).to(torch_device)
        model.eval()

        text = torch.tensor([[5, 241, 41, 22, 39, 105, 98], [8, 95, 46, 45, 159, 54, 6]]).long().to(torch_device)
        speech = torch.tensor([[11, 255, 25, 57, 10, 7, 41], [9, 20, 226, 15, 5, 97, 32]]).long().to(torch_device)
        inputs = {"input_ids": text, "speech_ids": speech}

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        self.assertEqual(
            outputs.logits_per_speech.shape,
            torch.Size((inputs["speech_ids"].shape[0], inputs["input_ids"].shape[0])),
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((inputs["input_ids"].shape[0], inputs["speech_ids"].shape[0])),
        )

        expected_logits = torch.tensor([[32.028324, 11.421426], [4.789056, 7.113933]], device=torch_device)

        self.assertTrue(torch.allclose(outputs.logits_per_speech, expected_logits, atol=1e-3))
