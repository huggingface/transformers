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
from __future__ import absolute_import, division, print_function

import copy
import json
import logging
import importlib
import random
import shutil
import unittest
import uuid

import pytest
import sys

from transformers import is_tf_available, is_torch_available

if is_tf_available():
    import tensorflow as tf
    import numpy as np
    from transformers import TFPreTrainedModel
    # from transformers.modeling_bert import BertModel, BertConfig, BERT_PRETRAINED_MODEL_ARCHIVE_MAP
else:
    pytestmark = pytest.mark.skip("Require TensorFlow")


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if '_range' in key or '_std' in key:
            setattr(configs_no_init, key, 0.0)
    return configs_no_init

class TFCommonTestCases:

    class TFCommonModelTester(unittest.TestCase):

        model_tester = None
        all_model_classes = ()
        test_torchscript = True
        test_pruning = True
        test_resize_embeddings = True

        def test_initialization(self):
            pass
            # config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            # configs_no_init = _config_zero_init(config)
            # for model_class in self.all_model_classes:
            #     model = model_class(config=configs_no_init)
            #     for name, param in model.named_parameters():
            #         if param.requires_grad:
            #             self.assertIn(param.data.mean().item(), [0.0, 1.0],
            #             msg="Parameter {} of model {} seems not properly initialized".format(name, model_class))


        def test_pt_tf_model_equivalence(self):
            if not is_torch_available():
                return

            import transformers

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            for model_class in self.all_model_classes:
                pt_model_class_name = model_class.__name__[2:]  # Skip the "TF" at the beggining
                pt_model_class = getattr(transformers, pt_model_class_name)

                tf_model = model_class(config)
                pt_model = pt_model_class(config)

                tf_model = transformers.load_pytorch_model_in_tf2_model(tf_model, pt_model, tf_inputs=inputs_dict)
                pt_model = transformers.load_tf2_model_in_pytorch_model(pt_model, tf_model)


        def test_keyword_and_dict_args(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            for model_class in self.all_model_classes:
                model = model_class(config)
                outputs_dict = model(inputs_dict)

                inputs_keywords = copy.deepcopy(inputs_dict)
                input_ids = inputs_keywords.pop('input_ids')
                outputs_keywords = model(input_ids, **inputs_keywords)

                output_dict = outputs_dict[0].numpy()
                output_keywords = outputs_keywords[0].numpy()

                self.assertLess(np.sum(np.abs(output_dict - output_keywords)), 1e-6)

        def test_attention_outputs(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            for model_class in self.all_model_classes:
                config.output_attentions = True
                config.output_hidden_states = False
                model = model_class(config)
                outputs = model(inputs_dict)
                attentions = [t.numpy() for t in outputs[-1]]
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
                outputs = model(inputs_dict)
                self.assertEqual(out_len+1, len(outputs))
                self.assertEqual(model.config.output_attentions, True)
                self.assertEqual(model.config.output_hidden_states, True)

                attentions = [t.numpy() for t in outputs[-1]]
                self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads,
                    self.model_tester.seq_length,
                    self.model_tester.key_len if hasattr(self.model_tester, 'key_len') else self.model_tester.seq_length])

        def test_headmasking(self):
            pass
            # config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            # config.output_attentions = True
            # config.output_hidden_states = True
            # configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
            # for model_class in self.all_model_classes:
            #     model = model_class(config=configs_no_init)
            #     model.eval()

            #     # Prepare head_mask
            #     # Set require_grad after having prepared the tensor to avoid error (leaf variable has been moved into the graph interior) 
            #     head_mask = torch.ones(self.model_tester.num_hidden_layers, self.model_tester.num_attention_heads)
            #     head_mask[0, 0] = 0
            #     head_mask[-1, :-1] = 0
            #     head_mask.requires_grad_(requires_grad=True)
            #     inputs = inputs_dict.copy()
            #     inputs['head_mask'] = head_mask

            #     outputs = model(**inputs)

            #     # Test that we can get a gradient back for importance score computation
            #     output = sum(t.sum() for t in outputs[0])
            #     output = output.sum()
            #     output.backward()
            #     multihead_outputs = head_mask.grad

            #     attentions = outputs[-1]
            #     hidden_states = outputs[-2]

            #     # Remove Nan

            #     self.assertIsNotNone(multihead_outputs)
            #     self.assertEqual(len(multihead_outputs), self.model_tester.num_hidden_layers)
            #     self.assertAlmostEqual(
            #         attentions[0][..., 0, :, :].flatten().sum().item(), 0.0)
            #     self.assertNotEqual(
            #         attentions[0][..., -1, :, :].flatten().sum().item(), 0.0)
            #     self.assertNotEqual(
            #         attentions[1][..., 0, :, :].flatten().sum().item(), 0.0)
            #     self.assertAlmostEqual(
            #         attentions[-1][..., -2, :, :].flatten().sum().item(), 0.0)
            #     self.assertNotEqual(
            #         attentions[-1][..., -1, :, :].flatten().sum().item(), 0.0)


        def test_head_pruning(self):
            pass
            # if not self.test_pruning:
            #     return

            # config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            # for model_class in self.all_model_classes:
            #     config.output_attentions = True
            #     config.output_hidden_states = False
            #     model = model_class(config=config)
            #     model.eval()
            #     heads_to_prune = {0: list(range(1, self.model_tester.num_attention_heads)),
            #                     -1: [0]}
            #     model.prune_heads(heads_to_prune)
            #     outputs = model(**inputs_dict)

            #     attentions = outputs[-1]

            #     self.assertEqual(
            #         attentions[0].shape[-3], 1)
            #     self.assertEqual(
            #         attentions[1].shape[-3], self.model_tester.num_attention_heads)
            #     self.assertEqual(
            #         attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)


        def test_hidden_states_output(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            for model_class in self.all_model_classes:
                config.output_hidden_states = True
                config.output_attentions = False
                model = model_class(config)
                outputs = model(inputs_dict)
                hidden_states = [t.numpy() for t in outputs[-1]]
                self.assertEqual(model.config.output_attentions, False)
                self.assertEqual(model.config.output_hidden_states, True)
                self.assertEqual(len(hidden_states), self.model_tester.num_hidden_layers + 1)
                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [self.model_tester.seq_length, self.model_tester.hidden_size])


        def test_resize_tokens_embeddings(self):
            pass
            # original_config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            # if not self.test_resize_embeddings:
            #     return

            # for model_class in self.all_model_classes:
            #     config = copy.deepcopy(original_config)
            #     model = model_class(config)

            #     model_vocab_size = config.vocab_size
            #     # Retrieve the embeddings and clone theme
            #     model_embed = model.resize_token_embeddings(model_vocab_size)
            #     cloned_embeddings = model_embed.weight.clone()

            #     # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            #     model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            #     self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            #     # Check that it actually resizes the embeddings matrix
            #     self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)

            #     # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            #     model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            #     self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            #     # Check that it actually resizes the embeddings matrix
            #     self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            #     # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            #     models_equal = True
            #     for p1, p2 in zip(cloned_embeddings, model_embed.weight):
            #         if p1.data.ne(p2.data).sum() > 0:
            #             models_equal = False

            #     self.assertTrue(models_equal)


        def test_tie_model_weights(self):
            pass
            # config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            # def check_same_values(layer_1, layer_2):
            #     equal = True
            #     for p1, p2 in zip(layer_1.weight, layer_2.weight):
            #         if p1.data.ne(p2.data).sum() > 0:
            #             equal = False
            #     return equal

            # for model_class in self.all_model_classes:
            #     if not hasattr(model_class, 'tie_weights'):
            #         continue

            #     config.torchscript = True
            #     model_not_tied = model_class(config)
            #     params_not_tied = list(model_not_tied.parameters())

            #     config_tied = copy.deepcopy(config)
            #     config_tied.torchscript = False
            #     model_tied = model_class(config_tied)
            #     params_tied = list(model_tied.parameters())

            #     # Check that the embedding layer and decoding layer are the same in size and in value
            #     self.assertGreater(len(params_not_tied), len(params_tied))

            #     # Check that after resize they remain tied.
            #     model_tied.resize_token_embeddings(config.vocab_size + 10)
            #     params_tied_2 = list(model_tied.parameters())
            #     self.assertGreater(len(params_not_tied), len(params_tied))
            #     self.assertEqual(len(params_tied_2), len(params_tied))

        def test_determinism(self):
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            for model_class in self.all_model_classes:
                model = model_class(config)
                first, second = model(inputs_dict, training=False)[0], model(inputs_dict, training=False)[0]
                self.assertTrue(tf.math.equal(first, second).numpy().all())


def ids_tensor(shape, vocab_size, rng=None, name=None, dtype=None):
    """Creates a random int32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    output = tf.constant(values,
                         shape=shape,
                         dtype=dtype if dtype is not None else tf.int32)

    return output


class TFModelUtilsTest(unittest.TestCase):
    @pytest.mark.skipif('tensorflow' not in sys.modules, reason="requires TensorFlow")
    def test_model_from_pretrained(self):
        pass
        # logging.basicConfig(level=logging.INFO)
        # for model_name in list(BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
        #     config = BertConfig.from_pretrained(model_name)
        #     self.assertIsNotNone(config)
        #     self.assertIsInstance(config, PretrainedConfig)

        #     model = BertModel.from_pretrained(model_name)
        #     model, loading_info = BertModel.from_pretrained(model_name, output_loading_info=True)
        #     self.assertIsNotNone(model)
        #     self.assertIsInstance(model, PreTrainedModel)
        #     for value in loading_info.values():
        #         self.assertEqual(len(value), 0)

        #     config = BertConfig.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
        #     model = BertModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
        #     self.assertEqual(model.config.output_attentions, True)
        #     self.assertEqual(model.config.output_hidden_states, True)
        #     self.assertEqual(model.config, config)


if __name__ == "__main__":
    unittest.main()
