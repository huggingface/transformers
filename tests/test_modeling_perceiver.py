# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch Perceiver model. """


import unittest

import inspect
import copy
import numpy as np

from transformers import PerceiverConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch
    import torch.nn as nn

    from transformers import PerceiverForMaskedLM, PerceiverModel, PerceiverTokenizer
    from transformers.models.perceiver.modeling_perceiver import PERCEIVER_PRETRAINED_MODEL_ARCHIVE_LIST


class PerceiverModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        num_latents=10,
        d_latents=20,
        num_blocks=1,
        num_self_attends_per_block=2,
        num_self_attention_heads=1,
        num_cross_attention_heads=1,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        max_position_embeddings=7,
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_latents = num_latents
        self.d_latents = d_latents
        self.num_blocks = num_blocks
        self.num_self_attends_per_block = num_self_attends_per_block
        self.num_self_attention_heads = num_self_attention_heads
        self.num_cross_attention_heads = num_cross_attention_heads
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_act = hidden_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope

    def prepare_config_and_inputs(self):
        inputs = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.num_labels)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()

        return config, inputs, input_mask, sequence_labels, token_labels
    
    def prepare_config_and_inputs_masked_lm(self):
        inputs = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_labels = None
        if self.use_labels:
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()

        return config, inputs, input_mask, token_labels

    def get_config(self):
        return PerceiverConfig(
            num_latents=self.num_latents,
            d_latents=self.d_latents,
            num_blocks=self.num_blocks,
            num_self_attends_per_block=self.num_self_attends_per_block,
            num_self_attention_heads=self.num_self_attention_heads,
            num_cross_attention_heads=self.num_cross_attention_heads,
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            max_position_embeddings=self.max_position_embeddings,
        )

    def create_and_check_for_masked_lm(
        self,
        config,
        inputs,
        input_mask,
        token_labels,
    ):
        model = PerceiverForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(inputs, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            inputs,
            input_mask,
            sequence_labels,
            token_labels,
        ) = config_and_inputs
        inputs_dict = {"inputs": inputs, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class PerceiverModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            PerceiverModel,
            PerceiverForMaskedLM,
        )
        if is_torch_available()
        else ()
    )
    test_pruning = False
    test_head_masking = False
    test_torchscript = False

    maxDiff = None
    
    def setUp(self):
        self.model_tester = PerceiverModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PerceiverConfig, hidden_size=37)

    def test_config(self):
        # we don't test 
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_masked_lm()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_model_common_attributes(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            # we overwrite this, as the embeddings of Perceiver are an instance of nn.Parameter
            # and Perceiver doesn't support get_output_embeddings
            self.assertIsInstance(model.get_input_embeddings(), (nn.Parameter))
    
    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["inputs"]
            self.assertListEqual(arg_names[:1], expected_arg_names)
    
    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class.__name__ == "PerceiverModel":
                continue
            
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
    
    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "num_latents", None)
        
        for model_class in self.all_model_classes:
            if model_class.__name__ == "PerceiverModel":
                continue

            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self_attentions = outputs.attentions
            cross_attentions = outputs.cross_attentions

            # check expected number of attentions depending on model class
            if model.__class__.__name__ == "PerceiverForMaskedLM":
                expected_num_self_attentions = self.model_tester.num_blocks * self.model_tester.num_self_attends_per_block
                # TODO this should become 2, as we have one in the PerceiverEncoder, and one in PerceiverBasicDecoder
                # for now, PerceiverBasicDecoder doesn't support returning cross-attentions
                expected_num_cross_attentions = 1 
            else:
                # todo
                pass
            self.assertEqual(len(self_attentions), expected_num_self_attentions)
            self.assertEqual(len(cross_attentions), expected_num_cross_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self_attentions = outputs.attentions
            cross_attentions = outputs.cross_attentions
            self.assertEqual(len(self_attentions), expected_num_self_attentions)
            self.assertEqual(len(cross_attentions), expected_num_cross_attentions)

            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_self_attention_heads, seq_len, seq_len],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(out_len + 1, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), expected_num_self_attentions)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_self_attention_heads, seq_len, seq_len],
            )
    
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = self.model_tester.num_blocks * self.model_tester.num_self_attends_per_block + 1
            self.assertEqual(len(hidden_states), expected_num_layers)

            seq_length = self.model_tester.num_latents

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.d_latents],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class.__name__ == "PerceiverModel":
                continue

            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)
    
    def test_feed_forward_chunking(self):
        (
            original_config,
            inputs_dict,
        ) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            if model_class.__name__ == "PerceiverModel":
                continue
            torch.manual_seed(0)
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            hidden_states_no_chunk = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            torch.manual_seed(0)
            config.chunk_size_feed_forward = 1
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            hidden_states_with_chunk = model(**self._prepare_for_class(inputs_dict, model_class))[0]
            self.assertTrue(torch.allclose(hidden_states_no_chunk, hidden_states_with_chunk, atol=1e-3))

    @unittest.skip(reason="Perceiver doesn't support resize_token_embeddings")
    def test_resize_tokens_embeddings(self):
        pass
    
    @unittest.skip(reason="Perceiver doesn't support inputs_embeds")
    def test_inputs_embeds(self):
        pass
    
    @slow
    def test_model_from_pretrained(self):
        for model_name in PERCEIVER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = PerceiverModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class PerceiverModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):

        tokenizer = PerceiverTokenizer.from_pretrained("nielsr/language-perceiver")
        model = PerceiverForMaskedLM.from_pretrained("nielsr/language-perceiver")
        model.to(torch_device)

        # prepare inputs
        text = "This is an incomplete sentence where some words are missing."
        encoding = tokenizer(text, padding="max_length", return_tensors="pt")

        # mask " missing.".
        encoding.input_ids[0, 51:60] = tokenizer.mask_token_id
        inputs, input_mask = encoding.input_ids.to(torch_device), encoding.attention_mask.to(torch_device)

        # forward pass
        outputs = model(inputs=inputs, attention_mask=input_mask)
        logits = outputs.logits

        # verify logits
        expected_shape = torch.Size((1, tokenizer.model_max_length, tokenizer.vocab_size))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-11.8336, -11.6850, -11.8483], [-12.8149, -12.5863, -12.7904], [-12.8440, -12.6410, -12.8646]]
        )

        self.assertTrue(torch.allclose(logits[0, :3, :3], expected_slice, atol=1e-4))

        expected_greedy_predictions = [38, 115, 111, 121, 121, 111, 116, 109, 52]
        masked_tokens_predictions = logits[0, 51:60].argmax(dim=-1).tolist()
        self.assertListEqual(expected_greedy_predictions, masked_tokens_predictions)
