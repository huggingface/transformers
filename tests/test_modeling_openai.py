# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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

from transformers import is_torch_available

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor
from .utils import CACHE_DIR, require_torch, slow, torch_device


if is_torch_available():
    import torch
    from transformers import (
        OpenAIGPTConfig,
        OpenAIGPTModel,
        OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP,
        OpenAIGPTLMHeadModel,
        OpenAIGPTDoubleHeadsModel,
    )


@require_torch
class OpenAIGPTModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (OpenAIGPTModel, OpenAIGPTLMHeadModel, OpenAIGPTDoubleHeadsModel) if is_torch_available() else ()
    )
    all_generative_model_classes = (
        (OpenAIGPTLMHeadModel,) if is_torch_available() else ()
    )  # TODO (PVP): Add Double HeadsModel when generate() function is changed accordingly

    class OpenAIGPTModelTester(object):
        def __init__(
            self,
            parent,
            batch_size=13,
            seq_length=7,
            is_training=True,
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
        ):
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.is_training = is_training
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

        def prepare_config_and_inputs(self):
            input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

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

            config = OpenAIGPTConfig(
                vocab_size=self.vocab_size,
                n_embd=self.hidden_size,
                n_layer=self.num_hidden_layers,
                n_head=self.num_attention_heads,
                # intermediate_size=self.intermediate_size,
                # hidden_act=self.hidden_act,
                # hidden_dropout_prob=self.hidden_dropout_prob,
                # attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                n_positions=self.max_position_embeddings,
                n_ctx=self.max_position_embeddings
                # type_vocab_size=self.type_vocab_size,
                # initializer_range=self.initializer_range
            )

            head_mask = ids_tensor([self.num_hidden_layers, self.num_attention_heads], 2)

            return config, input_ids, head_mask, token_type_ids, sequence_labels, token_labels, choice_labels

        def check_loss_output(self, result):
            self.parent.assertListEqual(list(result["loss"].size()), [])

        def create_and_check_openai_gpt_model(self, config, input_ids, head_mask, token_type_ids, *args):
            model = OpenAIGPTModel(config=config)
            model.to(torch_device)
            model.eval()

            model(input_ids, token_type_ids=token_type_ids, head_mask=head_mask)
            model(input_ids, token_type_ids=token_type_ids)
            (sequence_output,) = model(input_ids)

            result = {"sequence_output": sequence_output}
            self.parent.assertListEqual(
                list(result["sequence_output"].size()), [self.batch_size, self.seq_length, self.hidden_size]
            )

        def create_and_check_lm_head_model(self, config, input_ids, head_mask, token_type_ids, *args):
            model = OpenAIGPTLMHeadModel(config)
            model.to(torch_device)
            model.eval()

            loss, lm_logits = model(input_ids, token_type_ids=token_type_ids, labels=input_ids)

            result = {"loss": loss, "lm_logits": lm_logits}

            self.parent.assertListEqual(list(result["loss"].size()), [])
            self.parent.assertListEqual(
                list(result["lm_logits"].size()), [self.batch_size, self.seq_length, self.vocab_size]
            )

        def create_and_check_double_lm_head_model(self, config, input_ids, head_mask, token_type_ids, *args):
            model = OpenAIGPTDoubleHeadsModel(config)
            model.to(torch_device)
            model.eval()

            loss, lm_logits, mc_logits = model(input_ids, token_type_ids=token_type_ids, lm_labels=input_ids)

            result = {"loss": loss, "lm_logits": lm_logits}

            self.parent.assertListEqual(list(result["loss"].size()), [])
            self.parent.assertListEqual(
                list(result["lm_logits"].size()), [self.batch_size, self.seq_length, self.vocab_size]
            )

        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()
            (
                config,
                input_ids,
                head_mask,
                token_type_ids,
                sequence_labels,
                token_labels,
                choice_labels,
            ) = config_and_inputs
            inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "head_mask": head_mask}

            return config, inputs_dict

    def setUp(self):
        self.model_tester = OpenAIGPTModelTest.OpenAIGPTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OpenAIGPTConfig, n_embd=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_openai_gpt_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_openai_gpt_model(*config_and_inputs)

    def test_openai_gpt_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_openai_gpt_double_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_double_lm_head_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = OpenAIGPTModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.assertIsNotNone(model)


class OPENAIGPTModelLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_openai_gpt(self):
        model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt")
        input_ids = torch.Tensor([[481, 2585, 544, 4957]]).long()  # The dog is cute
        expected_output_ids = [
            481,
            2585,
            544,
            4957,
            669,
            512,
            761,
            5990,
            271,
            645,
            487,
            535,
            976,
            2479,
            240,
            487,
            804,
            1296,
            2891,
            512,
        ]  # the dog is cute when you're annoyed : if he's really stupid, he 'll stop fighting you
        torch.manual_seed(0)

        output_ids = model.generate(input_ids)
        self.assertListEqual(output_ids[0].tolist(), expected_output_ids)
