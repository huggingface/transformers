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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import json
import random

import torch

from pytorch_pretrained_bert import (OpenAIGPTConfig, OpenAIGPTModel,
                                     OpenAIGPTLMHeadModel, OpenAIGPTDoubleHeadsModel)


class OpenAIGPTModelTest(unittest.TestCase):
    class OpenAIGPTModelTester(object):

        def __init__(self,
                     parent,
                     batch_size=13,
                     seq_length=7,
                     is_training=True,
                     use_position_ids=True,
                     use_token_type_ids=True,
                     use_labels=True,
                     vocab_size=99,
                     n_special=1,
                     n_positions=33,
                     n_embd=32,
                     n_layer=5,
                     n_head=4,
                     n_choices=3,
                     afn="gelu",
                     resid_pdrop=0.1,
                     attn_pdrop=0.1,
                     embd_pdrop=0.1,
                     type_sequence_label_size=2,
                     initializer_range=0.02,
                     num_labels=3,
                     scope=None):
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.is_training = is_training
            self.use_position_ids = use_position_ids
            self.use_token_type_ids = use_token_type_ids
            self.use_labels = use_labels
            self.vocab_size = vocab_size
            self.n_special = n_special
            self.n_positions = n_positions
            self.n_embd = n_embd
            self.n_layer = n_layer
            self.n_head = n_head
            self.afn = afn
            self.n_choices = n_choices
            self.resid_pdrop = resid_pdrop
            self.attn_pdrop = attn_pdrop
            self.embd_pdrop = embd_pdrop
            self.type_sequence_label_size = type_sequence_label_size
            self.initializer_range = initializer_range
            self.num_labels = num_labels
            self.scope = scope

        def prepare_config_and_inputs(self):
            input_ids = OpenAIGPTModelTest.ids_tensor([self.batch_size, self.n_choices, self.seq_length], self.vocab_size)

            position_ids = None
            if self.use_position_ids:
                position_ids = OpenAIGPTModelTest.ids_tensor([self.batch_size, self.n_choices, self.seq_length], self.n_positions)

            token_type_ids = None
            if self.use_token_type_ids:
                total_voc = self.vocab_size + self.n_special
                token_type_ids = OpenAIGPTModelTest.ids_tensor([self.batch_size, self.n_choices, self.seq_length], total_voc)

            mc_labels = None
            lm_labels = None
            mc_token_ids = None
            if self.use_labels:
                mc_labels = OpenAIGPTModelTest.ids_tensor([self.batch_size], self.type_sequence_label_size)
                lm_labels = OpenAIGPTModelTest.ids_tensor([self.batch_size, self.n_choices, self.seq_length], self.num_labels)
                mc_token_ids = OpenAIGPTModelTest.ids_tensor([self.batch_size, self.n_choices], self.seq_length)

            config = OpenAIGPTConfig(
                vocab_size_or_config_json_file=self.vocab_size,
                n_positions=self.n_positions,
                n_special=self.n_special,
                n_embd=self.n_embd,
                n_layer=self.n_layer,
                n_head=self.n_head,
                afn=self.afn,
                resid_pdrop=self.resid_pdrop,
                attn_pdrop=self.attn_pdrop,
                embd_pdrop=self.embd_pdrop,
                initializer_range=self.initializer_range)

            return (config, input_ids, token_type_ids, position_ids,
                    mc_labels, lm_labels, mc_token_ids)

        def create_openai_model(self, config, input_ids, token_type_ids, position_ids,
                                mc_labels, lm_labels, mc_token_ids):
            model = OpenAIGPTModel(config)
            model.eval()
            hidden_states = model(input_ids, position_ids, token_type_ids)
            outputs = {
                "hidden_states": hidden_states,
            }
            return outputs

        def check_openai_model_output(self, result):
            self.parent.assertListEqual(
                list(result["hidden_states"].size()),
                [self.batch_size, self.n_choices, self.seq_length, self.n_embd])


        def create_openai_lm_head(self, config, input_ids, token_type_ids, position_ids,
                                       mc_labels, lm_labels, mc_token_ids):
            model = OpenAIGPTLMHeadModel(config)
            model.eval()
            loss = model(input_ids, position_ids, token_type_ids, lm_labels)
            lm_logits = model(input_ids, position_ids, token_type_ids)
            outputs = {
                "loss": loss,
                "lm_logits": lm_logits,
            }
            return outputs

        def check_openai_lm_head_output(self, result):
            total_voc = self.n_special + self.vocab_size
            self.parent.assertListEqual(
                list(result["lm_logits"].size()),
                [self.batch_size, self.n_choices, self.seq_length, total_voc])

        def check_openai_lm_head_loss_output(self, result):
            self.parent.assertListEqual(
                list(result["loss"].size()),
                [])

        def create_openai_double_heads(self, config, input_ids, token_type_ids, position_ids,
                                       mc_labels, lm_labels, mc_token_ids):
            model = OpenAIGPTDoubleHeadsModel(config)
            model.eval()
            loss = model(input_ids, mc_token_ids,
                         lm_labels=lm_labels, mc_labels=mc_labels,
                         token_type_ids=token_type_ids, position_ids=position_ids)
            lm_logits, mc_logits = model(input_ids, mc_token_ids, position_ids=position_ids, token_type_ids=token_type_ids)
            outputs = {
                "loss": loss,
                "lm_logits": lm_logits,
                "mc_logits": mc_logits,
            }
            return outputs

        def check_openai_double_heads_output(self, result):
            total_voc = self.n_special + self.vocab_size
            self.parent.assertListEqual(
                list(result["lm_logits"].size()),
                [self.batch_size, self.n_choices, self.seq_length, total_voc])
            self.parent.assertListEqual(
                list(result["mc_logits"].size()),
                [self.batch_size, self.n_choices])

        def check_openai_double_heads_loss_output(self, result):
            self.parent.assertListEqual(
                [list(l.size()) for l in result["loss"]],
                [[], []])

    def test_default(self):
        self.run_tester(OpenAIGPTModelTest.OpenAIGPTModelTester(self))

    def test_config_to_json_string(self):
        config = OpenAIGPTConfig(vocab_size_or_config_json_file=99, n_embd=37)
        obj = json.loads(config.to_json_string())
        self.assertEqual(obj["vocab_size"], 99)
        self.assertEqual(obj["n_embd"], 37)

    def run_tester(self, tester):
        config_and_inputs = tester.prepare_config_and_inputs()
        output_result = tester.create_openai_model(*config_and_inputs)
        tester.check_openai_model_output(output_result)

        output_result = tester.create_openai_lm_head(*config_and_inputs)
        tester.check_openai_lm_head_output(output_result)
        tester.check_openai_lm_head_loss_output(output_result)

        output_result = tester.create_openai_double_heads(*config_and_inputs)
        tester.check_openai_double_heads_output(output_result)
        tester.check_openai_double_heads_loss_output(output_result)

    @classmethod
    def ids_tensor(cls, shape, vocab_size, rng=None, name=None):
        """Creates a random int32 tensor of the shape within the vocab size."""
        if rng is None:
            rng = random.Random()

        total_dims = 1
        for dim in shape:
            total_dims *= dim

        values = []
        for _ in range(total_dims):
            values.append(rng.randint(0, vocab_size - 1))

        return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


if __name__ == "__main__":
    unittest.main()
