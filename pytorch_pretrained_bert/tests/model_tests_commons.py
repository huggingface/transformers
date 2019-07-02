# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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

import os
import shutil
import json
import random

import torch

def create_and_check_for_headmasking(tester, model_classes, config, inputs_dict):
    for model_class in model_classes:
        config.output_hidden_states = True
        model = model_class(config=config)
        model.eval()
        head_mask = torch.zeros(tester.num_hidden_layers, tester.num_attention_heads)
        # Set that after having prepared the tensor to avoid error (leaf variable has been moved into the graph interior) 
        head_mask.requires_grad_(requires_grad=True)
        outputs = model(**inputs_dict, head_mask=head_mask)

        # Compute some gradients
        output = sum(t.sum() for t in outputs[0])
        output = output.sum()
        output.backward()
        multihead_outputs = head_mask.grad

        tester.parent.assertEqual(len(multihead_outputs), tester.num_hidden_layers)
        # self.parent.assertListEqual(
        #     list(multihead_outputs[0].size()),
        #     [self.batch_size, self.num_attention_heads,
        #      self.seq_length, self.hidden_size // self.num_attention_heads])
        # self.parent.assertEqual(
        #     len(multihead_outputs[0][:, 1:(self.num_attention_heads-1), :, :].nonzero()),
        #     0)
        # self.parent.assertEqual(
        #     len(multihead_outputs[0][:, 0, :, :].nonzero()),
        #     self.batch_size * self.seq_length * self.hidden_size // self.num_attention_heads)
        # self.parent.assertEqual(
        #     len(multihead_outputs[0][:, self.num_attention_heads-1, :, :].nonzero()),
        #     self.batch_size * self.seq_length * self.hidden_size // self.num_attention_heads)

        # self.parent.assertListEqual(
        #     list(multihead_outputs[1].size()),
        #     [self.batch_size, self.num_attention_heads,
        #      self.seq_length, self.hidden_size // self.num_attention_heads])
        # self.parent.assertEqual(
        #     len(multihead_outputs[1].nonzero()),
        #     multihead_outputs[1].numel())

        # self.parent.assertListEqual(
        #     list(multihead_outputs[-1].size()),
        #     [self.batch_size, self.num_attention_heads,
        #      self.seq_length, self.hidden_size // self.num_attention_heads])
        # self.parent.assertEqual(
        #     len(multihead_outputs[-1][:, 1:, :, :].nonzero()),
        #     0)
        # self.parent.assertEqual(
        #     len(multihead_outputs[-1][:, 0, :, :].nonzero()),
        #     self.batch_size * self.seq_length * self.hidden_size // self.num_attention_heads)


def create_and_check_for_head_pruning(tester, model_classes, config, inputs_dict):
    for model_class in model_classes:
        model = model_class(config=config)
        model.eval()
        heads_to_prune = {0: list(range(1, tester.num_attention_heads)),
                            -1: [0]}
        model.prune_heads(heads_to_prune)
        outputs = model(**inputs_dict)

        # output = sum(t.sum() for t in outputs[0])
        # output = output.sum()
        # output.backward()
        # multihead_outputs = bert_model.get_multihead_outputs()

        # self.parent.assertEqual(len(multihead_outputs), self.num_hidden_layers)
        # self.parent.assertListEqual(
        #     list(multihead_outputs[0].size()),
        #     [self.batch_size, 1,
        #      self.seq_length, self.hidden_size // self.num_attention_heads])
        # self.parent.assertListEqual(
        #     list(multihead_outputs[1].size()),
        #     [self.batch_size, self.num_attention_heads,
        #      self.seq_length, self.hidden_size // self.num_attention_heads])
        # self.parent.assertListEqual(
        #     list(multihead_outputs[-1].size()),
        #     [self.batch_size, self.num_attention_heads-1,
        #      self.seq_length, self.hidden_size // self.num_attention_heads])


def create_and_check_for_attentions(tester, model_classes, config, inputs_dict):
    for model_class in model_classes:
        config.output_attentions = True
        config.output_hidden_states = False
        model = model_class(config)
        model.eval()
        outputs = model(**inputs_dict)
        attentions = outputs[-1]
        tester.parent.assertEqual(model.config.output_attentions, True)
        tester.parent.assertEqual(model.config.output_hidden_states, False)
        tester.parent.assertEqual(len(attentions), tester.num_hidden_layers)
        tester.parent.assertListEqual(
            list(attentions[0].shape[-3:]),
            [tester.num_attention_heads,
             tester.seq_length,
             tester.key_len if hasattr(tester, 'key_len') else tester.seq_length])
        out_len = len(outputs)

        # Check attention is always last and order is fine
        config.output_attentions = True
        config.output_hidden_states = True
        model = model_class(config)
        model.eval()
        outputs = model(**inputs_dict)
        tester.parent.assertEqual(out_len+1, len(outputs))
        tester.parent.assertEqual(model.config.output_attentions, True)
        tester.parent.assertEqual(model.config.output_hidden_states, True)

        attentions = outputs[-1]
        tester.parent.assertEqual(len(attentions), tester.num_hidden_layers)
        tester.parent.assertListEqual(
            list(attentions[0].shape[-3:]),
            [tester.num_attention_heads,
             tester.seq_length,
             tester.key_len if hasattr(tester, 'key_len') else tester.seq_length])

def create_and_check_for_hidden_states(tester, model_classes, config, inputs_dict):
    for model_class in model_classes:
        config.output_hidden_states = True
        config.output_attentions = False
        model = model_class(config)
        model.eval()
        outputs = model(**inputs_dict)
        hidden_states = outputs[-1]
        tester.parent.assertEqual(model.config.output_attentions, False)
        tester.parent.assertEqual(model.config.output_hidden_states, True)
        tester.parent.assertEqual(len(hidden_states), tester.num_hidden_layers + 1)
        tester.parent.assertListEqual(
            list(hidden_states[0].shape[-2:]),
            [tester.seq_length, tester.hidden_size])


def create_and_check_commons(tester, config, inputs_dict):
    create_and_check_for_attentions(tester, tester.all_model_classes, config, inputs_dict)
    create_and_check_for_headmasking(tester, tester.all_model_classes, config, inputs_dict)
    create_and_check_for_head_pruning(tester, tester.all_model_classes, config, inputs_dict)
    create_and_check_for_hidden_states(tester, tester.all_model_classes, config, inputs_dict)


def ids_tensor(shape, vocab_size, rng=None, name=None):
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


class ConfigTester(object):
    def __init__(self, parent, config_class=None, **kwargs):
        self.parent = parent
        self.config_class = config_class
        self.inputs_dict = kwargs

    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, 'hidden_size'))
        self.parent.assertTrue(hasattr(config, 'num_attention_heads'))
        self.parent.assertTrue(hasattr(config, 'num_hidden_layers'))

    def create_and_test_config_to_json_string(self):
        config = self.config_class(**self.inputs_dict)
        obj = json.loads(config.to_json_string())
        for key, value in self.inputs_dict.items():
            self.parent.assertEqual(obj[key], value)

    def create_and_test_config_to_json_file(self):
        config_first = self.config_class(**self.inputs_dict)
        json_file_path = "/tmp/config.json"
        config_first.to_json_file(json_file_path)
        config_second = self.config_class.from_json_file(json_file_path)
        os.remove(json_file_path)
        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

    def run_common_tests(self):
        self.create_and_test_config_common_properties()
        self.create_and_test_config_to_json_string()
        self.create_and_test_config_to_json_file()


class GPTModelTester(object):
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
                    hidden_size=32,
                    num_hidden_layers=5,
                    num_attention_heads=4,
                    n_choices=3,
                    type_sequence_label_size=2,
                    initializer_range=0.02,
                    num_labels=3,
                    scope=None,
                    config_class=None,
                    base_model_class=None,
                    lm_head_model_class=None,
                    double_head_model_class=None,
                    ):
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
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_choices = n_choices
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope
        self.config_class = config_class
        self.base_model_class = base_model_class
        self.lm_head_model_class = lm_head_model_class
        self.double_head_model_class = double_head_model_class
        self.all_model_classes = (base_model_class, lm_head_model_class, double_head_model_class)

    def prepare_config_and_inputs(self):
        total_num_tokens = self.vocab_size + self.n_special
        input_ids = ids_tensor([self.batch_size, self.n_choices, self.seq_length], total_num_tokens)

        position_ids = None
        if self.use_position_ids:
            position_ids = ids_tensor([self.batch_size, self.n_choices, self.seq_length], self.n_positions)

        token_type_ids = None
        if self.use_token_type_ids:
            total_voc = self.vocab_size
            token_type_ids = ids_tensor([self.batch_size, self.n_choices, self.seq_length], total_voc)

        mc_labels = None
        lm_labels = None
        mc_token_ids = None
        if self.use_labels:
            mc_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            lm_labels = ids_tensor([self.batch_size, self.n_choices, self.seq_length], self.num_labels)
            mc_token_ids = ids_tensor([self.batch_size, self.n_choices], self.seq_length)

        config = self.config_class(
            vocab_size_or_config_json_file=self.vocab_size,
            n_special=self.n_special,
            n_positions=self.n_positions,
            n_embd=self.hidden_size,
            n_layer=self.num_hidden_layers,
            n_head=self.num_attention_heads,
            initializer_range=self.initializer_range)

        return (config, input_ids, token_type_ids, position_ids,
                mc_labels, lm_labels, mc_token_ids)

    def create_and_check_base_model(self, config, input_ids, token_type_ids, position_ids,
                            mc_labels, lm_labels, mc_token_ids):
        model = self.base_model_class(config)
        model.eval()
        outputs = model(input_ids, position_ids, token_type_ids)
        hidden_state = outputs[0]
        self.parent.assertListEqual(
            list(hidden_state.size()),
            [self.batch_size, self.n_choices, self.seq_length, self.hidden_size])


    def create_and_check_lm_head(self, config, input_ids, token_type_ids, position_ids,
                                    mc_labels, lm_labels, mc_token_ids):
        model = self.lm_head_model_class(config)
        model.eval()
        outputs = model(input_ids, position_ids, token_type_ids, lm_labels)
        loss, lm_logits = outputs[:2]

        total_voc = self.n_special + self.vocab_size
        self.parent.assertListEqual(
            list(lm_logits.size()),
            [self.batch_size, self.n_choices, self.seq_length, total_voc])
        self.parent.assertListEqual(
            list(loss.size()),
            [])

    def create_and_check_presents(self, config, input_ids, token_type_ids, position_ids,
                                    mc_labels, lm_labels, mc_token_ids):
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.eval()
            outputs = model(input_ids)
            presents = outputs[-1]
            self.parent.assertEqual(self.num_hidden_layers, len(presents))
            self.parent.assertListEqual(
                list(presents[0].size()),
                [2, self.batch_size * self.n_choices, self.num_attention_heads,
                    self.seq_length, self.hidden_size // self.num_attention_heads])

    def create_and_check_double_heads(self, config, input_ids, token_type_ids, position_ids,
                                    mc_labels, lm_labels, mc_token_ids):
        model = self.double_head_model_class(config)
        model.eval()
        outputs = model(input_ids, mc_token_ids, lm_labels=lm_labels, mc_labels=mc_labels,
                                                    token_type_ids=token_type_ids, position_ids=position_ids)
        lm_loss, mc_loss, lm_logits, mc_logits = outputs[:4]
        loss = [lm_loss, mc_loss]

        total_voc = self.n_special + self.vocab_size
        self.parent.assertListEqual(
            list(lm_logits.size()),
            [self.batch_size, self.n_choices, self.seq_length, total_voc])
        self.parent.assertListEqual(
            list(mc_logits.size()),
            [self.batch_size, self.n_choices])
        self.parent.assertListEqual(
            [list(l.size()) for l in loss],
            [[], []])

    def create_and_check_model_from_pretrained(self):
        cache_dir = "/tmp/pytorch_pretrained_bert_test/"
        for model_name in list(self.base_model_class.PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = self.base_model_class.from_pretrained(model_name, cache_dir=cache_dir)
            shutil.rmtree(cache_dir)
            self.parent.assertIsNotNone(model)

    def create_and_check_commons(self, config, input_ids, token_type_ids, position_ids,
                                    mc_labels, lm_labels, mc_token_ids):
        inputs_dict = {'input_ids': input_ids}
        create_and_check_commons(self, config, inputs_dict)

    def run_common_tests(self, test_presents=False):
        config_and_inputs = self.prepare_config_and_inputs()
        self.create_and_check_base_model(*config_and_inputs)

        config_and_inputs = self.prepare_config_and_inputs()
        self.create_and_check_lm_head(*config_and_inputs)

        config_and_inputs = self.prepare_config_and_inputs()
        self.create_and_check_double_heads(*config_and_inputs)

        if test_presents:
            config_and_inputs = self.prepare_config_and_inputs()
            self.create_and_check_presents(*config_and_inputs)

        config_and_inputs = self.prepare_config_and_inputs()
        self.create_and_check_commons(*config_and_inputs)

    def run_slow_tests(self):
        config_and_inputs = self.prepare_config_and_inputs()
        self.create_and_check_model_from_pretrained(*config_and_inputs)

