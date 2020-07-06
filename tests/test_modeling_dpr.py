# coding=utf-8
# Copyright 2020 Huggingface
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


import os
import tempfile
import unittest

from transformers import is_torch_available

from .test_configuration_common import ConfigTester
from .test_modeling_common import _config_zero_init, ids_tensor
from .utils import require_torch, slow, torch_device


if is_torch_available():
    import torch
    import numpy as np
    from transformers import BertConfig, DPRConfig, DPRContextEncoder, DPRQuestionEncoder, DPRReader
    from transformers.modeling_dpr import (
        DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


class DPRModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
        projection_dim=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.projection_dim = projection_dim

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )
        config = DPRConfig(projection_dim=self.projection_dim, **config.to_dict())

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def create_and_check_dpr_context_encoder(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = DPRContextEncoder(config=config)
        model.to(torch_device)
        model.eval()
        embeddings = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        embeddings = model(input_ids, token_type_ids=token_type_ids)
        embeddings = model(input_ids)

        result = {
            "embeddings": embeddings,
        }
        self.parent.assertListEqual(
            list(result["embeddings"].size()), [self.batch_size, self.projection_dim or self.hidden_size]
        )

    def create_and_check_dpr_question_encoder(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = DPRQuestionEncoder(config=config)
        model.to(torch_device)
        model.eval()
        embeddings = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        embeddings = model(input_ids, token_type_ids=token_type_ids)
        embeddings = model(input_ids)

        result = {
            "embeddings": embeddings,
        }
        self.parent.assertListEqual(
            list(result["embeddings"].size()), [self.batch_size, self.projection_dim or self.hidden_size]
        )

    def create_and_check_dpr_reader(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = DPRReader(config=config)
        model.to(torch_device)
        model.eval()
        start_logits, end_logits, relevance_logits = model(input_ids, attention_mask=input_mask,)
        result = {
            "relevance_logits": relevance_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }
        self.parent.assertListEqual(list(result["start_logits"].size()), [self.batch_size, self.seq_length])
        self.parent.assertListEqual(list(result["end_logits"].size()), [self.batch_size, self.seq_length])
        self.parent.assertListEqual(list(result["relevance_logits"].size()), [self.batch_size])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids}
        return config, inputs_dict


@require_torch
class DPRModelTest(unittest.TestCase):

    all_model_classes = (DPRContextEncoder, DPRQuestionEncoder, DPRReader,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = DPRModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DPRConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_dpr_context_encoder_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_dpr_context_encoder(*config_and_inputs)

    def test_dpr_question_encoder_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_dpr_question_encoder(*config_and_inputs)

    def test_dpr_reader_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_dpr_reader(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = DPRContextEncoder.from_pretrained(model_name)
            self.assertIsNotNone(model)

        for model_name in DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = DPRContextEncoder.from_pretrained(model_name)
            self.assertIsNotNone(model)

        for model_name in DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = DPRQuestionEncoder.from_pretrained(model_name)
            self.assertIsNotNone(model)

        for model_name in DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = DPRReader.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def _prepare_for_class(self, inputs_dict, model_class):
        return inputs_dict

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)
                model.to(torch_device)
                with torch.no_grad():
                    after_outputs = model(**self._prepare_for_class(inputs_dict, model_class))

                # Make sure we don't have nans
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg="Parameter {} of model {} seems not properly initialized".format(name, model_class),
                    )

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]
                second = model(**self._prepare_for_class(inputs_dict, model_class))[0]
            out_1 = first.cpu().numpy()
            out_2 = second.cpu().numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

    def test_torchscript(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        self._create_and_check_torchscript(config, inputs_dict)

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            return

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class)["input_ids"]  # Let's keep only input_ids

            try:
                traced_gpt2 = torch.jit.trace(model, inputs)
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_gpt2, pt_file_name)
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

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                p2 = loaded_model_state_dict[layer_name]
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)
