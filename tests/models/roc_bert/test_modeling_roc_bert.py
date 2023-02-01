# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch RoCBert model. """

import unittest

from transformers import RoCBertConfig, is_torch_available
from transformers.models.auto import get_values
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        MODEL_FOR_PRETRAINING_MAPPING,
        RoCBertForCausalLM,
        RoCBertForMaskedLM,
        RoCBertForMultipleChoice,
        RoCBertForPreTraining,
        RoCBertForQuestionAnswering,
        RoCBertForSequenceClassification,
        RoCBertForTokenClassification,
        RoCBertModel,
    )
    from transformers.models.roc_bert.modeling_roc_bert import ROC_BERT_PRETRAINED_MODEL_ARCHIVE_LIST


class RoCBertModelTester:
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
        pronunciation_vocab_size=99,
        shape_vocab_size=99,
        pronunciation_embed_dim=32,
        shape_embed_dim=32,
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
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.pronunciation_vocab_size = pronunciation_vocab_size
        self.shape_vocab_size = shape_vocab_size
        self.pronunciation_embed_dim = pronunciation_embed_dim
        self.shape_embed_dim = shape_embed_dim
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

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_shape_ids = ids_tensor([self.batch_size, self.seq_length], self.shape_vocab_size)
        input_pronunciation_ids = ids_tensor([self.batch_size, self.seq_length], self.pronunciation_vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

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

        config = self.get_config()

        return (
            config,
            input_ids,
            input_shape_ids,
            input_pronunciation_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(self):
        return RoCBertConfig(
            vocab_size=self.vocab_size,
            shape_vocab_size=self.shape_vocab_size,
            pronunciation_vocab_size=self.pronunciation_vocab_size,
            shape_embed_dim=self.shape_embed_dim,
            pronunciation_embed_dim=self.pronunciation_embed_dim,
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

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            input_shape_ids,
            input_pronunciation_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        config.is_decoder = True
        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            input_shape_ids,
            input_pronunciation_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        input_shape_ids,
        input_pronunciation_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = RoCBertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
        )
        result = model(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            token_type_ids=token_type_ids,
        )
        result = model(input_ids, input_shape_ids=input_shape_ids, input_pronunciation_ids=input_pronunciation_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        input_shape_ids,
        input_pronunciation_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = RoCBertModel(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
        )
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        input_shape_ids,
        input_pronunciation_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = RoCBertForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_masked_lm(
        self,
        config,
        input_ids,
        input_shape_ids,
        input_pronunciation_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = RoCBertForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        input_shape_ids,
        input_pronunciation_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = RoCBertForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_shape_tokens = ids_tensor((self.batch_size, 3), config.shape_vocab_size)
        next_pronunciation_tokens = ids_tensor((self.batch_size, 3), config.pronunciation_vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_input_shape_ids = torch.cat([input_shape_ids, next_shape_tokens], dim=-1)
        next_input_pronunciation_ids = torch.cat([input_pronunciation_ids, next_pronunciation_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            input_shape_ids=next_input_shape_ids,
            input_pronunciation_ids=next_input_pronunciation_ids,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            input_shape_ids=next_shape_tokens,
            input_pronunciation_ids=next_pronunciation_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        input_shape_ids,
        input_pronunciation_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = RoCBertForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        input_shape_ids,
        input_pronunciation_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        config.num_labels = self.num_labels
        model = RoCBertForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=sequence_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        input_shape_ids,
        input_pronunciation_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        config.num_labels = self.num_labels
        model = RoCBertForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_multiple_choice(
        self,
        config,
        input_ids,
        input_shape_ids,
        input_pronunciation_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        config.num_choices = self.num_choices
        model = RoCBertForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_inputs_shape_ids = input_shape_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_inputs_pronunciation_ids = (
            input_pronunciation_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        )
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        result = model(
            multiple_choice_inputs_ids,
            input_shape_ids=multiple_choice_inputs_shape_ids,
            input_pronunciation_ids=multiple_choice_inputs_pronunciation_ids,
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
            input_shape_ids,
            input_pronunciation_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "input_shape_ids": input_shape_ids,
            "input_pronunciation_ids": input_pronunciation_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }
        return config, inputs_dict

    def create_and_check_for_pretraining(
        self,
        config,
        input_ids,
        input_shape_ids,
        input_pronunciation_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = RoCBertForPreTraining(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            input_shape_ids,
            input_pronunciation_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            attack_input_ids=input_ids,
            attack_input_shape_ids=input_shape_ids,
            attack_input_pronunciation_ids=input_pronunciation_ids,
            attack_attention_mask=input_mask,
            attack_token_type_ids=token_type_ids,
            labels_input_ids=token_labels,
            labels_input_shape_ids=input_shape_ids,
            labels_input_pronunciation_ids=input_pronunciation_ids,
            labels_attention_mask=input_mask,
            labels_token_type_ids=token_type_ids,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))


@require_torch
class RoCBertModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            RoCBertModel,
            RoCBertForMaskedLM,
            RoCBertForCausalLM,
            RoCBertForMultipleChoice,
            RoCBertForQuestionAnswering,
            RoCBertForSequenceClassification,
            RoCBertForTokenClassification,
            RoCBertForPreTraining,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (RoCBertForCausalLM,) if is_torch_available() else ()

    # special case for ForPreTraining model
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class in get_values(MODEL_FOR_PRETRAINING_MAPPING):
                inputs_dict["labels_input_ids"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
                inputs_dict["labels_input_shape_ids"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
                inputs_dict["labels_input_pronunciation_ids"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
                inputs_dict["attack_input_ids"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
                inputs_dict["attack_input_shape_ids"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
                inputs_dict["attack_input_pronunciation_ids"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
        return inputs_dict

    def setUp(self):
        self.model_tester = RoCBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=RoCBertConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs_relative_pos_emb(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        config_and_inputs[0].position_embedding_type = "relative_key"
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_model_as_decoder(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_model_as_decoder(*config_and_inputs)

    def test_model_as_decoder_with_default_input_mask(self):
        # This regression test was failing with PyTorch < 1.3
        (
            config,
            input_ids,
            input_shape_ids,
            input_pronunciation_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = self.model_tester.prepare_config_and_inputs_for_decoder()

        input_mask = None

        self.model_tester.create_and_check_model_as_decoder(
            config,
            input_ids,
            input_shape_ids,
            input_pronunciation_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    @slow
    def test_model_from_pretrained(self):
        for model_name in ROC_BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = RoCBertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class RoCBertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = RoCBertForMaskedLM.from_pretrained("weiweishi/roc-bert-base-zh")

        # input_text: ['[CLS]', 'b', 'a', '里', '系', '[MASK]', '国', '的', '首', '都', '[SEP]'] is the adversarial text
        # of ['[CLS]', '巴', '黎', '是', '[MASK]', '国', '的', '首', '都', '[SEP]'], means
        # "Paris is the [MASK] of France" in English
        input_ids = torch.tensor([[101, 144, 143, 7027, 5143, 103, 1744, 4638, 7674, 6963, 102]])
        input_shape_ids = torch.tensor([[2, 20324, 23690, 8740, 706, 1, 10900, 23343, 20205, 5850, 2]])
        input_pronunciation_ids = torch.tensor([[2, 718, 397, 52, 61, 1, 168, 273, 180, 243, 2]])

        output = model(input_ids, input_shape_ids, input_pronunciation_ids)
        output_ids = torch.argmax(output.logits, dim=2)

        # convert to tokens is: ['[CLS]', '巴', '*', '黎', '是', '法', '国', '的', '首', '都', '[SEP]']
        expected_output = torch.tensor([[101, 2349, 115, 7944, 3221, 3791, 1744, 4638, 7674, 6963, 102]])

        assert torch.allclose(output_ids, expected_output)
