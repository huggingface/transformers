# coding=utf-8
# Copyright 2018 The Hugging Face team.
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
    from transformers import ElectraConfig, ElectraModel, ElectraDiscriminator, ElectraGenerator
    from transformers.modeling_electra import ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP


@require_torch
class ElectraModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            # ElectraModel,
            ElectraGenerator,
            ElectraDiscriminator,
        )
        if is_torch_available()
        else ()
    )

    class ElectraModelTester(object):
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
            embedding_size=16,
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

        def prepare_config_and_inputs(self, generator_or_discriminator=None):
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
                fake_labels = ids_tensor([self.batch_size, self.seq_length], self.num_choices)

            config = ElectraConfig(
                vocab_size=self.vocab_size,
                discriminator_hidden_size=self.hidden_size,
                discriminator_num_hidden_layers=self.num_hidden_layers,
                discriminator_num_attention_heads=self.num_attention_heads,
                discriminator_intermediate_size=self.intermediate_size,
                generator_hidden_size=self.hidden_size,
                generator_num_hidden_layers=self.num_hidden_layers,
                generator_num_attention_heads=self.num_attention_heads,
                generator_intermediate_size=self.intermediate_size,
                hidden_act=self.hidden_act,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                max_position_embeddings=self.max_position_embeddings,
                type_vocab_size=self.type_vocab_size,
                is_decoder=False,
                initializer_range=self.initializer_range,
            )

            if generator_or_discriminator == "discriminator":
                config = config.get_discriminator_config()

            if generator_or_discriminator == "generator":
                config = config.get_generator_config()

            return (
                config,
                input_ids,
                token_type_ids,
                input_mask,
                sequence_labels,
                token_labels,
                choice_labels,
                fake_labels,
            )

        def check_loss_output(self, result):
            self.parent.assertListEqual(list(result["loss"].size()), [])

        def create_and_check_electra_model(
            self,
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            fake_labels,
        ):
            model = ElectraModel(config=config)
            model.to(torch_device)
            model.eval()
            generator_sequence_output, generator_pooled_output, discriminator_sequence_output = model(
                input_ids, attention_mask=input_mask, token_type_ids=token_type_ids
            )
            generator_sequence_output, generator_pooled_output, discriminator_sequence_output = model(
                input_ids, token_type_ids=token_type_ids
            )
            generator_sequence_output, generator_pooled_output, discriminator_sequence_output = model(input_ids)

            result = {
                "generator_sequence_output": generator_sequence_output,
                "generator_pooled_output": generator_pooled_output,
                "discriminator_sequence_output": discriminator_sequence_output,
            }
            self.parent.assertListEqual(
                list(result["discriminator_sequence_output"].size()),
                [self.batch_size, self.seq_length, self.hidden_size],
            )
            self.parent.assertListEqual(
                list(result["generator_sequence_output"].size()), [self.batch_size, self.seq_length, self.hidden_size]
            )
            self.parent.assertListEqual(
                list(result["generator_pooled_output"].size()), [self.batch_size, self.hidden_size]
            )

        def create_and_check_electra_generator(
            self,
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            fake_labels,
        ):
            model = ElectraGenerator(config=config)
            model.to(torch_device)
            model.eval()
            generator_sequence_output, generator_pooled_output, logits, probs, preds, loss = model(
                input_ids,
                attention_mask=input_mask,
                token_type_ids=token_type_ids,
                masked_lm_positions=token_labels,
                masked_lm_ids=token_labels,
            )
            self.parent.assertListEqual(
                list(generator_sequence_output.size()), [self.batch_size, self.seq_length, self.hidden_size]
            )
            self.parent.assertListEqual(list(generator_pooled_output.size()), [self.batch_size, self.hidden_size])
            self.parent.assertListEqual(list(logits.size()), [self.batch_size, self.seq_length, self.vocab_size])
            self.parent.assertListEqual(list(probs.size()), [self.batch_size, token_labels.shape[1], self.vocab_size])
            self.parent.assertListEqual(list(preds.size()), [self.batch_size, token_labels.shape[1]])

        def create_and_check_electra_discriminator(
            self,
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            fake_labels,
        ):
            model = ElectraDiscriminator(config=config)
            model.to(torch_device)
            model.eval()
            discriminator_sequence_output, discrim_probs, discrim_preds, discrim_loss = model(
                input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, fake_token_labels=fake_labels
            )
            self.parent.assertListEqual(
                list(discriminator_sequence_output.size()), [self.batch_size, self.seq_length, self.hidden_size]
            )
            self.parent.assertListEqual(list(discrim_probs.size()), [self.batch_size, self.seq_length])
            self.parent.assertListEqual(list(discrim_preds.size()), [self.batch_size, self.seq_length])
            self.parent.assertListEqual(list(discrim_loss.size()), [])

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
                fake_labels,
            ) = config_and_inputs

            inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids}

            return config, inputs_dict

    def setUp(self):
        self.model_tester = ElectraModelTest.ElectraModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ElectraConfig, discriminator_hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_electra_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_electra_model(*config_and_inputs)

    def test_generator(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(generator_or_discriminator="generator")
        self.model_tester.create_and_check_electra_generator(*config_and_inputs)

    def test_discriminator(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(generator_or_discriminator="discriminator")
        self.model_tester.create_and_check_electra_discriminator(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in list(ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = ElectraModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.assertIsNotNone(model)
