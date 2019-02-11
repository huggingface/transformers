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

from pytorch_pretrained_bert import (TransfoXLConfig, TransfoXLModel, TransfoXLLMHeadModel)


class TransfoXLModelTest(unittest.TestCase):
    class TransfoXLModelTester(object):

        def __init__(self,
                     parent,
                     batch_size=13,
                     seq_length=7,
                     mem_len=30,
                     clamp_len=15,
                     is_training=True,
                     use_labels=True,
                     vocab_size=99,
                     cutoffs=[10, 50, 80],
                     d_model=32,
                     d_embed=32,
                     n_head=4,
                     d_head=8,
                     d_inner=128,
                     div_val=2,
                     n_layer=5,
                     scope=None,
                     seed=1):
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.mem_len = mem_len
            self.clamp_len = clamp_len
            self.is_training = is_training
            self.use_labels = use_labels
            self.vocab_size = vocab_size
            self.cutoffs = cutoffs
            self.d_model = d_model
            self.d_embed = d_embed
            self.n_head = n_head
            self.d_head = d_head
            self.d_inner = d_inner
            self.div_val = div_val
            self.n_layer = n_layer
            self.scope = scope
            self.seed = seed

        def prepare_config_and_inputs(self):
            input_ids_1 = TransfoXLModelTest.ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            input_ids_2 = TransfoXLModelTest.ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

            lm_labels = None
            if self.use_labels:
                lm_labels = TransfoXLModelTest.ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

            config = TransfoXLConfig(
                vocab_size_or_config_json_file=self.vocab_size,
                mem_len=self.mem_len,
                clamp_len=self.clamp_len,
                cutoffs=self.cutoffs,
                d_model=self.d_model,
                d_embed=self.d_embed,
                n_head=self.n_head,
                d_head=self.d_head,
                d_inner=self.d_inner,
                div_val=self.div_val,
                n_layer=self.n_layer)

            return (config, input_ids_1, input_ids_2, lm_labels)

        def set_seed(self):
            random.seed(self.seed)
            torch.manual_seed(self.seed)

        def create_transfo_xl_model(self, config, input_ids_1, input_ids_2, lm_labels):
            model = TransfoXLModel(config)
            model.eval()

            hidden_states_1, mems_1 = model(input_ids_1)
            hidden_states_2, mems_2 = model(input_ids_2, mems_1)
            outputs = {
                "hidden_states_1": hidden_states_1,
                "mems_1": mems_1,
                "hidden_states_2": hidden_states_2,
                "mems_2": mems_2,
            }
            return outputs

        def check_transfo_xl_model_output(self, result):
            self.parent.assertListEqual(
                list(result["hidden_states_1"].size()),
                [self.batch_size, self.seq_length, self.d_model])
            self.parent.assertListEqual(
                list(result["hidden_states_2"].size()),
                [self.batch_size, self.seq_length, self.d_model])
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems_1"]),
                [[self.mem_len, self.batch_size, self.d_model]] * self.n_layer)
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems_2"]),
                [[self.mem_len, self.batch_size, self.d_model]] * self.n_layer)


        def create_transfo_xl_lm_head(self, config, input_ids_1, input_ids_2, lm_labels):
            model = TransfoXLLMHeadModel(config)
            model.eval()

            loss_1, mems_1a = model(input_ids_1, target=lm_labels)
            lm_logits_1, mems_1b = model(input_ids_1)

            loss_2, mems_2a = model(input_ids_2, target=lm_labels, mems=mems_1a)
            lm_logits_2, mems_2b = model(input_ids_2, mems=mems_1b)

            outputs = {
                "loss_1": loss_1,
                "mems_1a": mems_1a,
                "lm_logits_1": lm_logits_1,
                "mems_1b": mems_1b,
                "loss_2": loss_2,
                "mems_2a": mems_2a,
                "lm_logits_2": lm_logits_2,
                "mems_2b": mems_2b,
            }
            return outputs

        def check_transfo_xl_lm_head_output(self, result):
            self.parent.assertListEqual(
                list(result["loss_1"].size()),
                [self.batch_size, self.seq_length])
            self.parent.assertListEqual(
                list(result["lm_logits_1"].size()),
                [self.batch_size, self.seq_length, self.vocab_size])
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems_1a"]),
                [[self.mem_len, self.batch_size, self.d_model]] * self.n_layer)
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems_1b"]),
                [[self.mem_len, self.batch_size, self.d_model]] * self.n_layer)
            self.parent.assertListEqual(
                list(mem[~torch.isnan(mem)].sum() for mem in result["mems_1a"]),
                list(mem[~torch.isnan(mem)].sum() for mem in result["mems_1b"]))

            self.parent.assertListEqual(
                list(result["loss_2"].size()),
                [self.batch_size, self.seq_length])
            self.parent.assertListEqual(
                list(result["lm_logits_2"].size()),
                [self.batch_size, self.seq_length, self.vocab_size])
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems_2a"]),
                [[self.mem_len, self.batch_size, self.d_model]] * self.n_layer)
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems_2b"]),
                [[self.mem_len, self.batch_size, self.d_model]] * self.n_layer)
            self.parent.assertListEqual(
                list(mem[~torch.isnan(mem)].sum() for mem in result["mems_2a"]),
                list(mem[~torch.isnan(mem)].sum() for mem in result["mems_2b"]))

    def test_default(self):
        self.run_tester(TransfoXLModelTest.TransfoXLModelTester(self))

    def test_config_to_json_string(self):
        config = TransfoXLConfig(vocab_size_or_config_json_file=96, d_embed=37)
        obj = json.loads(config.to_json_string())
        self.assertEqual(obj["n_token"], 96)
        self.assertEqual(obj["d_embed"], 37)

    def run_tester(self, tester):
        config_and_inputs = tester.prepare_config_and_inputs()

        tester.set_seed()
        output_result = tester.create_transfo_xl_model(*config_and_inputs)
        tester.check_transfo_xl_model_output(output_result)

        tester.set_seed()
        output_result = tester.create_transfo_xl_lm_head(*config_and_inputs)
        tester.check_transfo_xl_lm_head_output(output_result)

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
