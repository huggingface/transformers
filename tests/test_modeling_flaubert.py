# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from transformers import FlaubertConfig, is_torch_available
from transformers.testing_utils import require_torch, require_torch_gpu, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        FlaubertForMultipleChoice,
        FlaubertForQuestionAnswering,
        FlaubertForQuestionAnsweringSimple,
        FlaubertForSequenceClassification,
        FlaubertForTokenClassification,
        FlaubertModel,
        FlaubertWithLMHeadModel,
    )
    from transformers.models.flaubert.modeling_flaubert import FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST


class FlaubertModelTester(object):
    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_input_lengths = True
        self.use_token_type_ids = True
        self.use_labels = True
        self.gelu_activation = True
        self.sinusoidal_embeddings = False
        self.causal = False
        self.asm = False
        self.n_langs = 2
        self.vocab_size = 99
        self.n_special = 0
        self.hidden_size = 32
        self.num_hidden_layers = 5
        self.num_attention_heads = 4
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 12
        self.type_sequence_label_size = 2
        self.initializer_range = 0.02
        self.num_labels = 3
        self.num_choices = 4
        self.summary_type = "last"
        self.use_proj = None
        self.scope = None

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = random_attention_mask([self.batch_size, self.seq_length])

        input_lengths = None
        if self.use_input_lengths:
            input_lengths = (
                ids_tensor([self.batch_size], vocab_size=2) + self.seq_length - 2
            )  # small variation of seq_length

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.n_langs)

        sequence_labels = None
        token_labels = None
        is_impossible_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            is_impossible_labels = ids_tensor([self.batch_size], 2).float()
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return (
            config,
            input_ids,
            token_type_ids,
            input_lengths,
            sequence_labels,
            token_labels,
            is_impossible_labels,
            choice_labels,
            input_mask,
        )

    def get_config(self):
        return FlaubertConfig(
            vocab_size=self.vocab_size,
            n_special=self.n_special,
            emb_dim=self.hidden_size,
            n_layers=self.num_hidden_layers,
            n_heads=self.num_attention_heads,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            gelu_activation=self.gelu_activation,
            sinusoidal_embeddings=self.sinusoidal_embeddings,
            asm=self.asm,
            causal=self.causal,
            n_langs=self.n_langs,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            summary_type=self.summary_type,
            use_proj=self.use_proj,
        )

    def create_and_check_flaubert_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        choice_labels,
        input_mask,
    ):
        model = FlaubertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, lengths=input_lengths, langs=token_type_ids)
        result = model(input_ids, langs=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_flaubert_lm_head(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        choice_labels,
        input_mask,
    ):
        model = FlaubertWithLMHeadModel(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_flaubert_simple_qa(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        choice_labels,
        input_mask,
    ):
        model = FlaubertForQuestionAnsweringSimple(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids)

        result = model(input_ids, start_positions=sequence_labels, end_positions=sequence_labels)
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_flaubert_qa(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        choice_labels,
        input_mask,
    ):
        model = FlaubertForQuestionAnswering(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids)

        result_with_labels = model(
            input_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
            cls_index=sequence_labels,
            is_impossible=is_impossible_labels,
            p_mask=input_mask,
        )

        result_with_labels = model(
            input_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
            cls_index=sequence_labels,
            is_impossible=is_impossible_labels,
        )

        (total_loss,) = result_with_labels.to_tuple()

        result_with_labels = model(input_ids, start_positions=sequence_labels, end_positions=sequence_labels)

        (total_loss,) = result_with_labels.to_tuple()

        self.parent.assertEqual(result_with_labels.loss.shape, ())
        self.parent.assertEqual(result.start_top_log_probs.shape, (self.batch_size, model.config.start_n_top))
        self.parent.assertEqual(result.start_top_index.shape, (self.batch_size, model.config.start_n_top))
        self.parent.assertEqual(
            result.end_top_log_probs.shape, (self.batch_size, model.config.start_n_top * model.config.end_n_top)
        )
        self.parent.assertEqual(
            result.end_top_index.shape, (self.batch_size, model.config.start_n_top * model.config.end_n_top)
        )
        self.parent.assertEqual(result.cls_logits.shape, (self.batch_size,))

    def create_and_check_flaubert_sequence_classif(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        choice_labels,
        input_mask,
    ):
        model = FlaubertForSequenceClassification(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids)
        result = model(input_ids, labels=sequence_labels)

        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def create_and_check_flaubert_token_classif(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        choice_labels,
        input_mask,
    ):
        config.num_labels = self.num_labels
        model = FlaubertForTokenClassification(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_flaubert_multiple_choice(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        choice_labels,
        input_mask,
    ):
        config.num_choices = self.num_choices
        model = FlaubertForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        result = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
            token_type_ids=multiple_choice_token_type_ids,
            labels=choice_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_lengths,
            sequence_labels,
            token_labels,
            is_impossible_labels,
            choice_labels,
            input_mask,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "lengths": input_lengths,
            "attention_mask": input_mask,
        }
        return config, inputs_dict


@require_torch
class FlaubertModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            FlaubertModel,
            FlaubertWithLMHeadModel,
            FlaubertForQuestionAnswering,
            FlaubertForQuestionAnsweringSimple,
            FlaubertForSequenceClassification,
            FlaubertForTokenClassification,
            FlaubertForMultipleChoice,
        )
        if is_torch_available()
        else ()
    )

    # Flaubert has 2 QA models -> need to manually set the correct labels for one of them here
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ == "FlaubertForQuestionAnswering":
                inputs_dict["start_positions"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
                inputs_dict["end_positions"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )

        return inputs_dict

    def setUp(self):
        self.model_tester = FlaubertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=FlaubertConfig, emb_dim=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_flaubert_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_flaubert_model(*config_and_inputs)

    def test_flaubert_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_flaubert_lm_head(*config_and_inputs)

    def test_flaubert_simple_qa(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_flaubert_simple_qa(*config_and_inputs)

    def test_flaubert_qa(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_flaubert_qa(*config_and_inputs)

    def test_flaubert_sequence_classif(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_flaubert_sequence_classif(*config_and_inputs)

    def test_flaubert_token_classif(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_flaubert_token_classif(*config_and_inputs)

    def test_flaubert_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_flaubert_multiple_choice(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = FlaubertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @slow
    @require_torch_gpu
    def test_torchscript_device_change(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:

            # FlauBertForMultipleChoice behaves incorrectly in JIT environments.
            if model_class == FlaubertForMultipleChoice:
                return

            config.torchscript = True
            model = model_class(config=config)

            inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            traced_model = torch.jit.trace(
                model, (inputs_dict["input_ids"].to("cpu"), inputs_dict["attention_mask"].to("cpu"))
            )

            with tempfile.TemporaryDirectory() as tmp:
                torch.jit.save(traced_model, os.path.join(tmp, "traced_model.pt"))
                loaded = torch.jit.load(os.path.join(tmp, "traced_model.pt"), map_location=torch_device)
                loaded(inputs_dict["input_ids"].to(torch_device), inputs_dict["attention_mask"].to(torch_device))


@require_torch
class FlaubertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head_absolute_embedding(self):
        model = FlaubertModel.from_pretrained("flaubert/flaubert_base_cased")
        input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        output = model(input_ids)[0]
        expected_shape = torch.Size((1, 11, 768))
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor(
            [[[-2.6251, -1.4298, -0.0227], [-2.8510, -1.6387, 0.2258], [-2.8114, -1.1832, -0.3066]]]
        )

        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-4))
