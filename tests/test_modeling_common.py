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

import copy
import logging
import os.path
import random
import tempfile
import unittest
from typing import List

from transformers import is_torch_available

from .utils import require_multigpu, require_torch, slow, torch_device


if is_torch_available():
    import torch
    import numpy as np

    from transformers import (
        AdaptiveEmbedding,
        PretrainedConfig,
        PreTrainedModel,
        BertModel,
        BertConfig,
        BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
        top_k_top_p_filtering,
    )


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key or "initializer_factor" in key:
            setattr(configs_no_init, key, 1e-10)
    return configs_no_init


@require_torch
class ModelTesterMixin:

    model_tester = None
    all_model_classes = ()
    all_generative_model_classes = ()
    test_torchscript = True
    test_pruning = True
    test_resize_embeddings = True
    test_head_masking = True
    test_missing_keys = True
    is_encoder_decoder = False

    def _prepare_for_class(self, inputs_dict, model_class):
        if model_class in MODEL_FOR_MULTIPLE_CHOICE_MAPPING.values():
            return {
                k: v.unsqueeze(1).expand(-1, self.model_tester.num_choices, -1).contiguous()
                if isinstance(v, torch.Tensor) and v.ndim != 0
                else v
                for k, v in inputs_dict.items()
            }
        return inputs_dict

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)
                model.to(torch_device)
                with torch.no_grad():
                    after_outputs = model(**self._prepare_for_class(inputs_dict, model_class))

                # Make sure we don't have nans
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg="Parameter {} of model {} seems not properly initialized".format(name, model_class),
                    )

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
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
        seq_len = getattr(self.model_tester, "seq_length", None)
        decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        decoder_key_length = getattr(self.model_tester, "key_length", decoder_seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)
        if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]
            self.assertEqual(model.config.output_hidden_states, False)
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]
            self.assertEqual(model.config.output_hidden_states, False)
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            if chunk_length is not None:
                self.assertListEqual(
                    list(attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )
            out_len = len(outputs)

            if self.is_encoder_decoder:
                correct_outlen = 4
                decoder_attention_idx = 1

                if "lm_labels" in inputs_dict:  # loss will come first
                    correct_outlen += 1  # compute loss
                    decoder_attention_idx += 1
                self.assertEqual(out_len, correct_outlen)

                decoder_attentions = outputs[decoder_attention_idx]
                self.assertIsInstance(decoder_attentions, (list, tuple))
                self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(decoder_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
                )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            config.output_hidden_states = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(out_len + (2 if self.is_encoder_decoder else 1), len(outputs))
            self.assertEqual(model.config.output_hidden_states, True)

            self_attentions = outputs[-1]
            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            if chunk_length is not None:
                self.assertListEqual(
                    list(self_attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )

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
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class)["input_ids"]  # Let's keep only input_ids

            try:
                traced_gpt2 = torch.jit.trace(model, inputs)
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_gpt2, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            model.to(torch_device)
            model.eval()

            loaded_model.to(torch_device)
            loaded_model.eval()

            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                p2 = loaded_model_state_dict[layer_name]
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def test_headmasking(self):
        if not self.test_head_masking:
            return

        global_rng.seed(42)
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        global_rng.seed()

        inputs_dict["output_attentions"] = True
        config.output_hidden_states = True
        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()

            # Prepare head_mask
            # Set require_grad after having prepared the tensor to avoid error (leaf variable has been moved into the graph interior)
            head_mask = torch.ones(
                self.model_tester.num_hidden_layers, self.model_tester.num_attention_heads, device=torch_device,
            )
            head_mask[0, 0] = 0
            head_mask[-1, :-1] = 0
            head_mask.requires_grad_(requires_grad=True)
            inputs = self._prepare_for_class(inputs_dict, model_class).copy()
            inputs["head_mask"] = head_mask

            outputs = model(**inputs)

            # Test that we can get a gradient back for importance score computation
            output = sum(t.sum() for t in outputs[0])
            output = output.sum()
            output.backward()
            multihead_outputs = head_mask.grad

            attentions = outputs[-1]

            # Remove Nan
            for t in attentions:
                self.assertLess(
                    torch.sum(torch.isnan(t)), t.numel() / 4
                )  # Check we don't have more than 25% nans (arbitrary)
            attentions = [
                t.masked_fill(torch.isnan(t), 0.0) for t in attentions
            ]  # remove them (the test is less complete)

            self.assertIsNotNone(multihead_outputs)
            self.assertEqual(len(multihead_outputs), self.model_tester.num_hidden_layers)
            self.assertAlmostEqual(attentions[0][..., 0, :, :].flatten().sum().item(), 0.0)
            self.assertNotEqual(attentions[0][..., -1, :, :].flatten().sum().item(), 0.0)
            self.assertNotEqual(attentions[1][..., 0, :, :].flatten().sum().item(), 0.0)
            self.assertAlmostEqual(attentions[-1][..., -2, :, :].flatten().sum().item(), 0.0)
            self.assertNotEqual(attentions[-1][..., -1, :, :].flatten().sum().item(), 0.0)

    def test_head_pruning(self):
        if not self.test_pruning:
            return

        for model_class in self.all_model_classes:
            (config, inputs_dict,) = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False
            model = model_class(config=config)
            model.to(torch_device)
            model.eval()
            heads_to_prune = {
                0: list(range(1, self.model_tester.num_attention_heads)),
                -1: [0],
            }
            model.prune_heads(heads_to_prune)
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

    def test_head_pruning_save_load_from_pretrained(self):
        if not self.test_pruning:
            return

        for model_class in self.all_model_classes:
            (config, inputs_dict,) = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False
            model = model_class(config=config)
            model.to(torch_device)
            model.eval()
            heads_to_prune = {
                0: list(range(1, self.model_tester.num_attention_heads)),
                -1: [0],
            }
            model.prune_heads(heads_to_prune)

            with tempfile.TemporaryDirectory() as temp_dir_name:
                model.save_pretrained(temp_dir_name)
                model = model_class.from_pretrained(temp_dir_name)
                model.to(torch_device)

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]
            self.assertEqual(attentions[0].shape[-3], 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

    def test_head_pruning_save_load_from_config_init(self):
        if not self.test_pruning:
            return

        for model_class in self.all_model_classes:
            (config, inputs_dict,) = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False

            heads_to_prune = {
                0: list(range(1, self.model_tester.num_attention_heads)),
                -1: [0],
            }
            config.pruned_heads = heads_to_prune

            model = model_class(config=config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

    def test_head_pruning_integration(self):
        if not self.test_pruning:
            return

        for model_class in self.all_model_classes:
            (config, inputs_dict,) = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            inputs_dict["output_attentions"] = True
            config.output_hidden_states = False

            heads_to_prune = {0: [0], 1: [1, 2]}
            config.pruned_heads = heads_to_prune

            model = model_class(config=config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads - 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)
            self.assertEqual(attentions[2].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[3].shape[-3], self.model_tester.num_attention_heads)

            with tempfile.TemporaryDirectory() as temp_dir_name:
                model.save_pretrained(temp_dir_name)
                model = model_class.from_pretrained(temp_dir_name)
                model.to(torch_device)

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads - 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)
            self.assertEqual(attentions[2].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[3].shape[-3], self.model_tester.num_attention_heads)

            heads_to_prune = {0: [0], 2: [1, 2]}
            model.prune_heads(heads_to_prune)

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads - 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)
            self.assertEqual(attentions[2].shape[-3], self.model_tester.num_attention_heads - 2)
            self.assertEqual(attentions[3].shape[-3], self.model_tester.num_attention_heads)

            self.assertDictEqual(model.config.pruned_heads, {0: [0], 1: [1, 2], 2: [1, 2]})

    def test_hidden_states_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            config.output_hidden_states = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            hidden_states = outputs[-1]
            self.assertEqual(model.config.output_hidden_states, True)
            self.assertEqual(len(hidden_states), self.model_tester.num_hidden_layers + 1)

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
                if hasattr(self.model_tester, "chunk_length") and self.model_tester.chunk_length > 1:
                    seq_length = seq_length * self.model_tester.chunk_length
            else:
                seq_length = self.model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]), [seq_length, self.model_tester.hidden_size],
            )

    def test_resize_tokens_embeddings(self):
        (original_config, inputs_dict,) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config.vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def test_model_common_attributes(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (torch.nn.Embedding, AdaptiveEmbedding))
            model.set_input_embeddings(torch.nn.Embedding(10, 10))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, torch.nn.Linear))

    def test_correct_missing_keys(self):
        if not self.test_missing_keys:
            return
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            base_model_prefix = model.base_model_prefix

            if hasattr(model, base_model_prefix):
                with tempfile.TemporaryDirectory() as temp_dir_name:
                    model.base_model.save_pretrained(temp_dir_name)
                    model, loading_info = model_class.from_pretrained(temp_dir_name, output_loading_info=True)

                    with self.subTest(msg="Missing keys for {}".format(model.__class__.__name__)):
                        self.assertGreater(len(loading_info["missing_keys"]), 0)

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
            config.torchscript = True
            model_not_tied = model_class(config)
            if model_not_tied.get_output_embeddings() is None:
                continue

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

    def test_inputs_embeds(self):

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.is_encoder_decoder:
            input_ids = inputs_dict["input_ids"]
            del inputs_dict["input_ids"]
        else:
            encoder_input_ids = inputs_dict["input_ids"]
            decoder_input_ids = inputs_dict.get("decoder_input_ids", encoder_input_ids)
            del inputs_dict["input_ids"]
            inputs_dict.pop("decoder_input_ids", None)

        for model_class in self.all_model_classes:
            if model_class in MODEL_FOR_MULTIPLE_CHOICE_MAPPING.values():
                continue
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            wte = model.get_input_embeddings()
            if not self.is_encoder_decoder:
                inputs_dict["inputs_embeds"] = wte(input_ids)
            else:
                inputs_dict["inputs_embeds"] = wte(encoder_input_ids)
                inputs_dict["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs_dict)

    def test_lm_head_model_random_no_beam_search_generate(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict["input_ids"] if "input_ids" in inputs_dict else inputs_dict["inputs"]

        # make sure that input_ids is at most of size 15
        input_ids = input_ids[..., :15]

        # iterate over all generative models
        for model_class in self.all_generative_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()

            if config.bos_token_id is None:
                # if bos token id is not defined, model needs input_ids
                with self.assertRaises(AssertionError):
                    model.generate(do_sample=True, max_length=5)
                # num_return_sequences = 1
                self._check_generated_ids(model.generate(input_ids, do_sample=True))
            else:
                # num_return_sequences = 1
                self._check_generated_ids(model.generate(do_sample=True, max_length=5))

            with self.assertRaises(AssertionError):
                # generating multiple sequences when no beam search generation
                # is not allowed as it would always generate the same sequences
                model.generate(input_ids, do_sample=False, num_return_sequences=2)

            # num_return_sequences > 1, sample
            self._check_generated_ids(model.generate(input_ids, do_sample=True, num_return_sequences=2))

            # check bad words tokens language generation
            # create list of 1-seq bad token and list of 2-seq of bad tokens
            bad_words_ids = [
                self._generate_random_bad_tokens(1, model.config),
                self._generate_random_bad_tokens(2, model.config),
            ]
            output_tokens = model.generate(
                input_ids, do_sample=True, bad_words_ids=bad_words_ids, num_return_sequences=2
            )
            # only count generated tokens
            generated_ids = output_tokens[:, input_ids.shape[-1] :]
            self.assertFalse(self._check_match_tokens(generated_ids.tolist(), bad_words_ids))

    def test_lm_head_model_random_beam_search_generate(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = (inputs_dict["input_ids"] if "input_ids" in inputs_dict else inputs_dict["inputs"]).to(
            torch_device
        )

        # make sure that input_ids is at most of size 15
        input_ids = input_ids[..., :15]

        for model_class in self.all_generative_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()

            if config.bos_token_id is None:
                # if bos token id is not defined mobel needs input_ids, num_return_sequences = 1
                self._check_generated_ids(model.generate(input_ids, do_sample=True, num_beams=2))
            else:
                # num_return_sequences = 1
                self._check_generated_ids(model.generate(do_sample=True, max_length=5, num_beams=2))

            with self.assertRaises(AssertionError):
                # generating more sequences than having beams leads is not possible
                model.generate(input_ids, do_sample=False, num_return_sequences=3, num_beams=2)

            # num_return_sequences > 1, sample
            self._check_generated_ids(model.generate(input_ids, do_sample=True, num_beams=2, num_return_sequences=2,))
            # num_return_sequences > 1, greedy
            self._check_generated_ids(model.generate(input_ids, do_sample=False, num_beams=2, num_return_sequences=2))

            # check bad words tokens language generation
            # create list of 1-seq bad token and list of 2-seq of bad tokens
            bad_words_ids = [
                self._generate_random_bad_tokens(1, model.config),
                self._generate_random_bad_tokens(2, model.config),
            ]
            output_tokens = model.generate(
                input_ids, do_sample=False, bad_words_ids=bad_words_ids, num_beams=2, num_return_sequences=2
            )
            # only count generated tokens
            generated_ids = output_tokens[:, input_ids.shape[-1] :]
            self.assertFalse(self._check_match_tokens(generated_ids.tolist(), bad_words_ids))

    def _generate_random_bad_tokens(self, num_bad_tokens: int, config) -> List[int]:
        # special tokens cannot be bad tokens
        special_tokens = [x for x in [config.bos_token_id, config.eos_token_id, config.pad_token_id] if x is not None]
        # create random bad tokens that are not special tokens
        bad_tokens = []
        while len(bad_tokens) < num_bad_tokens:
            token = ids_tensor((1, 1), self.model_tester.vocab_size).squeeze(0).cpu().numpy()[0]
            if token not in special_tokens:
                bad_tokens.append(token)
        return bad_tokens

    def _check_generated_ids(self, output_ids):
        for token_id in output_ids[0].tolist():
            self.assertGreaterEqual(token_id, 0)
            self.assertLess(token_id, self.model_tester.vocab_size)

    def _check_match_tokens(self, generated_ids, bad_words_ids):
        # for all bad word tokens
        for bad_word_ids in bad_words_ids:
            # for all slices in batch
            for generated_ids_slice in generated_ids:
                # for all word idx
                for i in range(len(bad_word_ids), len(generated_ids_slice)):
                    # if tokens match
                    if generated_ids_slice[i - len(bad_word_ids) : i] == bad_word_ids:
                        return True
        return False

    @require_multigpu
    def test_multigpu_data_parallel_forward(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # some params shouldn't be scattered by nn.DataParallel
        # so just remove them if they are present.
        blacklist_non_batched_params = ["head_mask"]
        for k in blacklist_non_batched_params:
            inputs_dict.pop(k, None)

        # move input tensors to cuda:O
        for k, v in inputs_dict.items():
            if torch.is_tensor(v):
                inputs_dict[k] = v.to(0)

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            model.to(0)
            model.eval()

            # Wrap model in nn.DataParallel
            model = torch.nn.DataParallel(model)
            with torch.no_grad():
                _ = model(**inputs_dict)


global_rng = random.Random()


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long, device=torch_device).view(shape).contiguous()


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return torch.tensor(data=values, dtype=torch.float, device=torch_device).view(shape).contiguous()


@require_torch
class ModelUtilsTest(unittest.TestCase):
    @slow
    def test_model_from_pretrained(self):
        logging.basicConfig(level=logging.INFO)
        for model_name in BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
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
            self.assertEqual(model.config.output_hidden_states, True)
            self.assertEqual(model.config, config)


@require_torch
class UtilsFunctionsTest(unittest.TestCase):

    # tests whether the top_k_top_p function behaves as expected
    def test_top_k_top_p_filtering(self):
        logits = torch.tensor(
            [
                [
                    8.2220991,  # 3rd highest value; idx. 0
                    -0.5620044,
                    5.23229752,
                    4.0386393,
                    -6.8798378,
                    -0.54785802,
                    -3.2012153,
                    2.92777176,
                    1.88171953,
                    7.35341276,  # 5th highest value; idx. 9
                    8.43207833,  # 2nd highest value; idx. 10
                    -9.85711836,
                    -5.96209236,
                    -1.13039161,
                    -7.1115294,
                    -0.8369633,
                    -5.3186408,
                    7.06427407,
                    0.81369344,
                    -0.82023817,
                    -5.9179796,
                    0.58813443,
                    -6.99778438,
                    4.71551189,
                    -0.18771637,
                    7.44020759,  # 4th highest value; idx. 25
                    9.38450987,  # 1st highest value; idx. 26
                    2.12662941,
                    -9.32562038,
                    2.35652522,
                ],  # cummulative prob of 5 highest values <= 0.6
                [
                    0.58425518,
                    4.53139238,
                    -5.57510464,
                    -6.28030699,
                    -7.19529503,
                    -4.02122551,
                    1.39337037,
                    -6.06707057,
                    1.59480517,
                    -9.643119,
                    0.03907799,
                    0.67231762,
                    -8.88206726,
                    6.27115922,  # 4th highest value; idx. 13
                    2.28520723,
                    4.82767506,
                    4.30421368,
                    8.8275313,  # 2nd highest value; idx. 17
                    5.44029958,  # 5th highest value; idx. 18
                    -4.4735794,
                    7.38579536,  # 3rd highest value; idx. 20
                    -2.91051663,
                    2.61946077,
                    -2.5674762,
                    -9.48959302,
                    -4.02922645,
                    -1.35416918,
                    9.67702323,  # 1st highest value; idx. 27
                    -5.89478553,
                    1.85370467,
                ],  # cummulative prob of 5 highest values <= 0.6
            ],
            dtype=torch.float,
            device=torch_device,
        )

        non_inf_expected_idx = torch.tensor(
            [[0, 0], [0, 9], [0, 10], [0, 25], [0, 26], [1, 13], [1, 17], [1, 18], [1, 20], [1, 27]],
            dtype=torch.long,
            device=torch_device,
        )  # expected non filtered idx as noted above

        non_inf_expected_output = torch.tensor(
            [
                8.2221,
                7.3534,
                8.4321,
                7.4402,
                9.3845,
                6.2712,
                8.8275,
                5.4403,
                7.3858,
                9.6770,
            ],  # expected non filtered values as noted above
            dtype=torch.float,
            device=torch_device,
        )

        output = top_k_top_p_filtering(logits, top_k=10, top_p=0.6, min_tokens_to_keep=4)
        non_inf_output = output[output != -float("inf")].to(device=torch_device)
        non_inf_idx = (output != -float("inf")).nonzero().to(device=torch_device)

        self.assertTrue(torch.allclose(non_inf_expected_output, non_inf_output, atol=1e-12))
        self.assertTrue(torch.all(torch.eq(non_inf_expected_idx, non_inf_idx)))
