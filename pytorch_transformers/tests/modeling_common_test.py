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

import copy
import os
import shutil
import json
import random
import uuid

import unittest
import logging

import torch

from pytorch_transformers import (PretrainedConfig, PreTrainedModel,
                                  BertModel, BertConfig, BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
                                  GPT2LMHeadModel, GPT2Config, GPT2_PRETRAINED_MODEL_ARCHIVE_MAP)


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if '_range' in key or '_std' in key:
            setattr(configs_no_init, key, 0.0)
    return configs_no_init

class CommonTestCases:

    class CommonModelTester(unittest.TestCase):

        model_tester = None
        all_model_classes = ()
        test_torchscript = True
        test_pruning = True
        test_resize_embeddings = True
        test_head_masking = True

        def test_initialization(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            configs_no_init = _config_zero_init(config)
            for model_class in self.all_model_classes:
                model = model_class(config=configs_no_init)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        self.assertIn(param.data.mean().item(), [0.0, 1.0],
                        msg="Parameter {} of model {} seems not properly initialized".format(name, model_class))

        def test_attention_outputs(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            for model_class in self.all_model_classes:
                config.output_attentions = True
                config.output_hidden_states = False
                model = model_class(config)
                model.eval()
                outputs = model(**inputs_dict)
                attentions = outputs[-1]
                self.assertEqual(model.config.output_attentions, True)
                self.assertEqual(model.config.output_hidden_states, False)
                self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads,
                    self.model_tester.seq_length,
                    self.model_tester.key_len if hasattr(self.model_tester, 'key_len') else self.model_tester.seq_length])
                out_len = len(outputs)

                # Check attention is always last and order is fine
                config.output_attentions = True
                config.output_hidden_states = True
                model = model_class(config)
                model.eval()
                outputs = model(**inputs_dict)
                self.assertEqual(out_len+1, len(outputs))
                self.assertEqual(model.config.output_attentions, True)
                self.assertEqual(model.config.output_hidden_states, True)

                attentions = outputs[-1]
                self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads,
                    self.model_tester.seq_length,
                    self.model_tester.key_len if hasattr(self.model_tester, 'key_len') else self.model_tester.seq_length])

        def test_torchscript(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            self._create_and_check_torchscript(config, inputs_dict)

        def test_torchscript_output_attentions(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            config.output_attentions = True
            self._create_and_check_torchscript(config, inputs_dict)

        def test_torchscript_output_hidden_state(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            config.output_hidden_states = True
            self._create_and_check_torchscript(config, inputs_dict)

        def _create_and_check_torchscript(self, config, inputs_dict):
            if not self.test_torchscript:
                return

            configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
            configs_no_init.torchscript = True
            for model_class in self.all_model_classes:
                model = model_class(config=configs_no_init)
                model.eval()
                inputs = inputs_dict['input_ids']  # Let's keep only input_ids

                try:
                    torch.jit.trace(model, inputs)
                except RuntimeError:
                    self.fail("Couldn't trace module.")

                try:
                    traced_gpt2 = torch.jit.trace(model, inputs)
                    torch.jit.save(traced_gpt2, "traced_model.pt")
                except RuntimeError:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load("traced_model.pt")
                    os.remove("traced_model.pt")
                except ValueError:
                    self.fail("Couldn't load module.")

                model.eval()
                loaded_model.eval()

                model_params = model.parameters()
                loaded_model_params = loaded_model.parameters()

                models_equal = True
                for p1, p2 in zip(model_params, loaded_model_params):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False

                self.assertTrue(models_equal)


        def test_headmasking(self):
            if not self.test_head_masking:
                return

            torch.manual_seed(42)
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            config.output_attentions = True
            config.output_hidden_states = True
            configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
            for model_class in self.all_model_classes:
                model = model_class(config=configs_no_init)
                model.eval()

                # Prepare head_mask
                # Set require_grad after having prepared the tensor to avoid error (leaf variable has been moved into the graph interior) 
                head_mask = torch.ones(self.model_tester.num_hidden_layers, self.model_tester.num_attention_heads)
                head_mask[0, 0] = 0
                head_mask[-1, :-1] = 0
                head_mask.requires_grad_(requires_grad=True)
                inputs = inputs_dict.copy()
                inputs['head_mask'] = head_mask

                outputs = model(**inputs)

                # Test that we can get a gradient back for importance score computation
                output = sum(t.sum() for t in outputs[0])
                output = output.sum()
                output.backward()
                multihead_outputs = head_mask.grad

                attentions = outputs[-1]
                hidden_states = outputs[-2]

                # Remove Nan

                self.assertIsNotNone(multihead_outputs)
                self.assertEqual(len(multihead_outputs), self.model_tester.num_hidden_layers)
                self.assertAlmostEqual(
                    attentions[0][..., 0, :, :].flatten().sum().item(), 0.0)
                self.assertNotEqual(
                    attentions[0][..., -1, :, :].flatten().sum().item(), 0.0)
                self.assertNotEqual(
                    attentions[1][..., 0, :, :].flatten().sum().item(), 0.0)
                self.assertAlmostEqual(
                    attentions[-1][..., -2, :, :].flatten().sum().item(), 0.0)
                self.assertNotEqual(
                    attentions[-1][..., -1, :, :].flatten().sum().item(), 0.0)


        def test_head_pruning(self):
            if not self.test_pruning:
                return

            for model_class in self.all_model_classes:
                config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

                if "head_mask" in inputs_dict:
                    del inputs_dict["head_mask"]

                config.output_attentions = True
                config.output_hidden_states = False
                model = model_class(config=config)
                model.eval()
                heads_to_prune = {0: list(range(1, self.model_tester.num_attention_heads)),
                                -1: [0]}
                model.prune_heads(heads_to_prune)
                outputs = model(**inputs_dict)

                attentions = outputs[-1]

                self.assertEqual(
                    attentions[0].shape[-3], 1)
                self.assertEqual(
                    attentions[1].shape[-3], self.model_tester.num_attention_heads)
                self.assertEqual(
                    attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

        def test_head_pruning_save_load_from_pretrained(self):
            if not self.test_pruning:
                return

            for model_class in self.all_model_classes:
                config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

                if "head_mask" in inputs_dict:
                    del inputs_dict["head_mask"]

                config.output_attentions = True
                config.output_hidden_states = False
                model = model_class(config=config)
                model.eval()
                heads_to_prune = {0: list(range(1, self.model_tester.num_attention_heads)),
                                -1: [0]}
                model.prune_heads(heads_to_prune)
                directory = "pruned_model"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                model.save_pretrained(directory)
                model = model_class.from_pretrained(directory)

                outputs = model(**inputs_dict)
                attentions = outputs[-1]
                self.assertEqual(attentions[0].shape[-3], 1)
                self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads)
                self.assertEqual(attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

                shutil.rmtree(directory)

        def test_head_pruning_save_load_from_config_init(self):
            if not self.test_pruning:
                return

            for model_class in self.all_model_classes:
                config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

                if "head_mask" in inputs_dict:
                    del inputs_dict["head_mask"]

                config.output_attentions = True
                config.output_hidden_states = False

                heads_to_prune = {0: list(range(1, self.model_tester.num_attention_heads)),
                                 -1: [0]}
                config.pruned_heads = heads_to_prune

                model = model_class(config=config)
                model.eval()

                outputs = model(**inputs_dict)
                attentions = outputs[-1]

                self.assertEqual(attentions[0].shape[-3], 1)
                self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads)
                self.assertEqual(attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

        def test_head_pruning_integration(self):
            if not self.test_pruning:
                return

            for model_class in self.all_model_classes:
                config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

                if "head_mask" in inputs_dict:
                    del inputs_dict["head_mask"]

                config.output_attentions = True
                config.output_hidden_states = False

                heads_to_prune = {0: [0], 1: [1, 2]}
                config.pruned_heads = heads_to_prune

                model = model_class(config=config)
                model.eval()

                outputs = model(**inputs_dict)
                attentions = outputs[-1]

                self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads - 1)
                self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)
                self.assertEqual(attentions[2].shape[-3], self.model_tester.num_attention_heads)
                self.assertEqual(attentions[3].shape[-3], self.model_tester.num_attention_heads)

                directory = "pruned_model"

                if not os.path.exists(directory):
                    os.makedirs(directory)
                model.save_pretrained(directory)
                model = model_class.from_pretrained(directory)
                shutil.rmtree(directory)

                outputs = model(**inputs_dict)
                attentions = outputs[-1]

                self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads - 1)
                self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)
                self.assertEqual(attentions[2].shape[-3], self.model_tester.num_attention_heads)
                self.assertEqual(attentions[3].shape[-3], self.model_tester.num_attention_heads)

                heads_to_prune = {0: [0], 2: [1, 2]}
                model.prune_heads(heads_to_prune)

                outputs = model(**inputs_dict)
                attentions = outputs[-1]

                self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads -1)
                self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)
                self.assertEqual(attentions[2].shape[-3], self.model_tester.num_attention_heads - 2)
                self.assertEqual(attentions[3].shape[-3], self.model_tester.num_attention_heads)

                self.assertDictEqual(model.config.pruned_heads, {0: [0], 1: [1, 2], 2: [1, 2]})


        def test_hidden_states_output(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            for model_class in self.all_model_classes:
                config.output_hidden_states = True
                config.output_attentions = False
                model = model_class(config)
                model.eval()
                outputs = model(**inputs_dict)
                hidden_states = outputs[-1]
                self.assertEqual(model.config.output_attentions, False)
                self.assertEqual(model.config.output_hidden_states, True)
                self.assertEqual(len(hidden_states), self.model_tester.num_hidden_layers + 1)
                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [self.model_tester.seq_length, self.model_tester.hidden_size])

        def test_resize_tokens_embeddings(self):
            original_config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            if not self.test_resize_embeddings:
                return

            for model_class in self.all_model_classes:
                config = copy.deepcopy(original_config)
                model = model_class(config)

                model_vocab_size = config.vocab_size
                # Retrieve the embeddings and clone theme
                model_embed = model.resize_token_embeddings(model_vocab_size)
                cloned_embeddings = model_embed.weight.clone()

                # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
                model_embed = model.resize_token_embeddings(model_vocab_size + 10)
                self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
                # Check that it actually resizes the embeddings matrix
                self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)

                # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
                model_embed = model.resize_token_embeddings(model_vocab_size - 15)
                self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
                # Check that it actually resizes the embeddings matrix
                self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

                # Check that adding and removing tokens has not modified the first part of the embedding matrix.
                models_equal = True
                for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False

                self.assertTrue(models_equal)

        def test_tie_model_weights(self):
            if not self.test_torchscript:
                return

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            def check_same_values(layer_1, layer_2):
                equal = True
                for p1, p2 in zip(layer_1.weight, layer_2.weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        equal = False
                return equal

            for model_class in self.all_model_classes:
                if not hasattr(model_class, 'tie_weights'):
                    continue

                config.torchscript = True
                model_not_tied = model_class(config)
                params_not_tied = list(model_not_tied.parameters())

                config_tied = copy.deepcopy(config)
                config_tied.torchscript = False
                model_tied = model_class(config_tied)
                params_tied = list(model_tied.parameters())

                # Check that the embedding layer and decoding layer are the same in size and in value
                self.assertGreater(len(params_not_tied), len(params_tied))
                # self.assertTrue(check_same_values(embeddings, decoding))

                # # Check that after modification, they remain the same.
                # embeddings.weight.data.div_(2)
                # # Check that the embedding layer and decoding layer are the same in size and in value
                # self.assertTrue(embeddings.weight.shape, decoding.weight.shape)
                # self.assertTrue(check_same_values(embeddings, decoding))

                # # Check that after modification, they remain the same.
                # decoding.weight.data.div_(4)
                # # Check that the embedding layer and decoding layer are the same in size and in value
                # self.assertTrue(embeddings.weight.shape, decoding.weight.shape)
                # self.assertTrue(check_same_values(embeddings, decoding))

                # Check that after resize they remain tied.
                model_tied.resize_token_embeddings(config.vocab_size + 10)
                params_tied_2 = list(model_tied.parameters())
                self.assertGreater(len(params_not_tied), len(params_tied))
                self.assertEqual(len(params_tied_2), len(params_tied))

                # decoding.weight.data.mul_(20)
                # # Check that the embedding layer and decoding layer are the same in size and in value
                # self.assertTrue(model.transformer.wte.weight.shape, model.lm_head.weight.shape)
                # self.assertTrue(check_same_values(model.transformer.wte, model.lm_head))


    class GPTModelTester(CommonModelTester):

        def __init__(self,
                        parent,
                        batch_size=13,
                        seq_length=7,
                        is_training=True,
                        use_position_ids=True,
                        use_token_type_ids=True,
                        use_labels=True,
                        vocab_size=99,
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
            total_num_tokens = self.vocab_size
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
            outputs = model(input_ids, position_ids)
            outputs = model(input_ids)

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

            total_voc = self.vocab_size
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

            total_voc = self.vocab_size
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
            cache_dir = "/tmp/pytorch_transformers_test/"
            for model_name in list(self.base_model_class.pretrained_model_archive_map.keys())[:1]:
                model = self.base_model_class.from_pretrained(model_name, cache_dir=cache_dir)
                shutil.rmtree(cache_dir)
                self.parent.assertIsNotNone(model)

        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()
            (config, input_ids, token_type_ids, position_ids,
                mc_labels, lm_labels, mc_token_ids) = config_and_inputs
            inputs_dict = {'input_ids': input_ids}
            return config, inputs_dict

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

        def run_slow_tests(self):
            self.create_and_check_model_from_pretrained()


class ConfigTester(object):
    def __init__(self, parent, config_class=None, **kwargs):
        self.parent = parent
        self.config_class = config_class
        self.inputs_dict = kwargs

    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, 'vocab_size'))
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
        json_file_path = os.path.join(os.getcwd(), "config_" + str(uuid.uuid4()) + ".json")
        config_first.to_json_file(json_file_path)
        config_second = self.config_class.from_json_file(json_file_path)
        os.remove(json_file_path)
        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

    def run_common_tests(self):
        self.create_and_test_config_common_properties()
        self.create_and_test_config_to_json_string()
        self.create_and_test_config_to_json_file()




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


class ModelUtilsTest(unittest.TestCase):
    def test_model_from_pretrained(self):
        logging.basicConfig(level=logging.INFO)
        for model_name in list(BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            config = BertConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, PretrainedConfig)

            model = BertModel.from_pretrained(model_name)
            model, loading_info = BertModel.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, PreTrainedModel)
            for value in loading_info.values():
                self.assertEqual(len(value), 0)

            config = BertConfig.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
            model = BertModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
            self.assertEqual(model.config.output_attentions, True)
            self.assertEqual(model.config.output_hidden_states, True)
            self.assertEqual(model.config, config)


if __name__ == "__main__":
    unittest.main()
