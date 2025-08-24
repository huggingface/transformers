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
"""Testing suite for the PyTorch GPT Neo model."""

import unittest

from transformers import GPTNeoConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import cached_property

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        GPT2Tokenizer,
        GPTNeoForCausalLM,
        GPTNeoForQuestionAnswering,
        GPTNeoForSequenceClassification,
        GPTNeoForTokenClassification,
        GPTNeoModel,
    )


class GPTNeoModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_token_type_ids=True,
        use_input_mask=True,
        use_labels=True,
        use_mc_token_ids=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        attention_types=[[["global", "local"], 1]],
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        window_size=7,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_token_type_ids = use_token_type_ids
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.use_mc_token_ids = use_mc_token_ids
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.window_size = window_size
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1
        self.attention_types = attention_types

    def get_large_model_config(self):
        return GPTNeoConfig.from_pretrained("gpt-neo-125M")

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        mc_token_ids = None
        if self.use_mc_token_ids:
            mc_token_ids = ids_tensor([self.batch_size, self.num_choices], self.seq_length)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        head_mask = ids_tensor([self.num_hidden_layers, self.num_attention_heads], 2)

        return (
            config,
            input_ids,
            input_mask,
            head_mask,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(self):
        return GPTNeoConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            window_size=self.window_size,
            attention_types=self.attention_types,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.vocab_size = 300
        return config

    def create_and_check_gpt_neo_model(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        model = GPTNeoModel(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, token_type_ids=token_type_ids, head_mask=head_mask)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        # past_key_values is not implemented
        # self.parent.assertEqual(len(result.past_key_values), config.n_layer)

    def create_and_check_gpt_neo_model_past(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        model = GPTNeoModel(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(input_ids, token_type_ids=token_type_ids, use_cache=True)
        outputs_use_cache_conf = model(input_ids, token_type_ids=token_type_ids)
        outputs_no_past = model(input_ids, token_type_ids=token_type_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        output, past = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)
        next_token_types = ids_tensor([self.batch_size, 1], self.type_vocab_size)

        # append to next input_ids and token_type_ids
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_token_type_ids = torch.cat([token_type_ids, next_token_types], dim=-1)

        output_from_no_past = model(next_input_ids, token_type_ids=next_token_type_ids)["last_hidden_state"]
        output_from_past = model(next_tokens, token_type_ids=next_token_types, past_key_values=past)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_gpt_neo_model_attention_mask_past(
        self, config, input_ids, input_mask, head_mask, token_type_ids, *args
    ):
        model = GPTNeoModel(config=config)
        model.to(torch_device)
        model.eval()

        # create attention mask
        attn_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        half_seq_length = self.seq_length // 2
        attn_mask[:, half_seq_length:] = 0

        # first forward pass
        output, past = model(input_ids, attention_mask=attn_mask).to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # change a random masked slice from input_ids
        random_seq_idx_to_change = ids_tensor((1,), half_seq_length).item() + 1
        random_other_next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size).squeeze(-1)
        input_ids[:, -random_seq_idx_to_change] = random_other_next_tokens

        # append to next input_ids and attn_mask
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        attn_mask = torch.cat(
            [attn_mask, torch.ones((attn_mask.shape[0], 1), dtype=torch.long, device=torch_device)],
            dim=1,
        )

        # get two different outputs
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past, attention_mask=attn_mask)["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_gpt_neo_model_past_large_inputs(
        self, config, input_ids, input_mask, head_mask, token_type_ids, *args
    ):
        model = GPTNeoModel(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask, use_cache=True)

        output, past = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_token_types = ids_tensor([self.batch_size, 3], self.type_vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and token_type_ids
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_token_type_ids = torch.cat([token_type_ids, next_token_types], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids, token_type_ids=next_token_type_ids, attention_mask=next_attention_mask
        )["last_hidden_state"]
        output_from_past = model(
            next_tokens, token_type_ids=next_token_types, attention_mask=next_attention_mask, past_key_values=past
        )["last_hidden_state"]
        self.parent.assertTrue(output_from_past.shape[1] == next_tokens.shape[1])

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_lm_head_model(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        model = GPTNeoForCausalLM(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, token_type_ids=token_type_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_gpt_neo_for_question_answering(
        self, config, input_ids, input_mask, head_mask, token_type_ids, mc_token_ids, sequence_labels, *args
    ):
        config.num_labels = self.num_labels
        model = GPTNeoForQuestionAnswering(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_gpt_neo_for_sequence_classification(
        self, config, input_ids, input_mask, head_mask, token_type_ids, mc_token_ids, sequence_labels, *args
    ):
        config.num_labels = self.num_labels
        model = GPTNeoForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_gpt_neo_for_token_classification(
        self, config, input_ids, input_mask, head_mask, token_type_ids, mc_token_ids, sequence_labels, *args
    ):
        config.num_labels = self.num_labels
        model = GPTNeoForTokenClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_forward_and_backwards(
        self, config, input_ids, input_mask, head_mask, token_type_ids, *args, gradient_checkpointing=False
    ):
        model = GPTNeoForCausalLM(config)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.to(torch_device)

        result = model(input_ids, token_type_ids=token_type_ids, labels=input_ids)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        result.loss.backward()

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()

        (
            config,
            input_ids,
            input_mask,
            head_mask,
            token_type_ids,
            mc_token_ids,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "head_mask": head_mask,
        }

        return config, inputs_dict


@require_torch
class GPTNeoModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            GPTNeoModel,
            GPTNeoForCausalLM,
            GPTNeoForQuestionAnswering,
            GPTNeoForSequenceClassification,
            GPTNeoForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": GPTNeoModel,
            "question-answering": GPTNeoForQuestionAnswering,
            "text-classification": GPTNeoForSequenceClassification,
            "text-generation": GPTNeoForCausalLM,
            "token-classification": GPTNeoForTokenClassification,
            "zero-shot": GPTNeoForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    fx_compatible = True
    test_missing_keys = False
    test_pruning = False
    test_model_parallel = False

    # special case for DoubleHeads model
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        return inputs_dict

    def setUp(self):
        self.model_tester = GPTNeoModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GPTNeoConfig, n_embd=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_gpt_neo_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_neo_model(*config_and_inputs)

    def test_gpt_neo_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_neo_model_past(*config_and_inputs)

    def test_gpt_neo_model_att_mask_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_neo_model_attention_mask_past(*config_and_inputs)

    def test_gpt_neo_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_neo_model_past_large_inputs(*config_and_inputs)

    def test_gpt_neo_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_gpt_neo_question_answering_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_neo_for_question_answering(*config_and_inputs)

    def test_gpt_neo_sequence_classification_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_neo_for_sequence_classification(*config_and_inputs)

    def test_gpt_neo_token_classification_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gpt_neo_for_token_classification(*config_and_inputs)

    def test_gpt_neo_gradient_checkpointing(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_forward_and_backwards(*config_and_inputs, gradient_checkpointing=True)

    def _get_hidden_states(self):
        return torch.tensor(
            [
                [
                    [0.4983, -0.7584, -1.6944, 0.5440],
                    [2.6918, 0.4206, 0.4176, 0.2055],
                    [-0.0071, -0.0405, -1.4920, -0.3630],
                    [1.0492, 0.1599, -1.7648, 0.2419],
                    [-1.8348, 2.0514, -0.1946, 0.3203],
                    [0.7672, -1.1600, -1.7118, -0.9056],
                    [0.2986, 0.5372, 0.7729, -0.1927],
                    [0.0285, 0.2629, -1.1156, -1.1992],
                ]
            ],
            dtype=torch.float32,
            device=torch_device,
        )

    def test_local_attn_probs(self):
        model = GPTNeoModel.from_pretrained("valhalla/gpt-neo-random-tiny").eval()
        layer = model.h[1].attn.attention.to(torch_device)
        hidden_states = self._get_hidden_states()
        hidden_states = torch.cat([hidden_states, hidden_states - 0.5], dim=2)

        batch_size, seq_length, _ = hidden_states.shape
        mask_tokens = 2
        attention_mask = torch.ones(batch_size, seq_length, device=torch_device, dtype=torch.long)
        attention_mask[:, -mask_tokens:] = 0  # dont attend last mask_tokens

        attention_mask = attention_mask.view(batch_size, -1)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * -10000.0

        attn_probs = layer(hidden_states, attention_mask=attention_mask, output_attentions=True)[-1]

        # the last 2 tokens are masked, and should have 0 attn_probs
        self.assertTrue(torch.all(attn_probs[:, :, -mask_tokens:, -mask_tokens:] == 0))

        # in loacal attention each token can only attend to the previous window_size tokens (including itself)
        # here window_size is 4, so a token at index 5 can only attend to indcies [2, 3, 4, 5]
        # and the attn_probs should be 0 for token [0, 1]
        self.assertTrue(torch.all(attn_probs[:, :, 5, 2:6] != 0))
        self.assertTrue(torch.all(attn_probs[:, :, 5, :2] == 0))


@require_torch
class GPTNeoModelLanguageGenerationTest(unittest.TestCase):
    @cached_property
    def model(self):
        return GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(torch_device)

    @cached_property
    def tokenizer(self):
        return GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    @slow
    def test_lm_generate_gpt_neo(self):
        for checkpointing in [True, False]:
            model = self.model
            if checkpointing:
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_disable()
            input_ids = torch.tensor([[464, 3290]], dtype=torch.long, device=torch_device)  # The dog
            # The dog-eared copy of the book, which is a collection of essays by the late author,
            expected_output_ids = [464, 3290, 12, 3380, 4866, 286, 262, 1492, 11, 543, 318, 257, 4947, 286, 27126, 416, 262, 2739, 1772, 11]  # fmt: skip
            output_ids = model.generate(input_ids, do_sample=False)
            self.assertListEqual(output_ids[0].tolist(), expected_output_ids)

    @slow
    def test_gpt_neo_sample(self):
        model = self.model
        tokenizer = self.tokenizer

        torch.manual_seed(0)
        tokenized = tokenizer("Today is a nice day and", return_tensors="pt", return_token_type_ids=True)
        input_ids = tokenized.input_ids.to(torch_device)
        output_ids = model.generate(input_ids, do_sample=True)
        output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        EXPECTED_OUTPUT_STR = "Today is a nice day and if you don’t get the memo here is what you can"
        self.assertEqual(output_str, EXPECTED_OUTPUT_STR)

    @slow
    def test_batch_generation(self):
        model = self.model
        tokenizer = self.tokenizer

        tokenizer.padding_side = "left"

        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        # use different length sentences to test batching
        sentences = [
            "Hello, my dog is a little",
            "Today, I am",
        ]

        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(torch_device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].to(torch_device),
        )

        inputs_non_padded = tokenizer(sentences[0], return_tensors="pt").input_ids.to(torch_device)
        output_non_padded = model.generate(input_ids=inputs_non_padded)

        num_paddings = inputs_non_padded.shape[-1] - inputs["attention_mask"][-1].long().sum().item()
        inputs_padded = tokenizer(sentences[1], return_tensors="pt").input_ids.to(torch_device)
        output_padded = model.generate(input_ids=inputs_padded, max_length=model.config.max_length - num_paddings)

        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        non_padded_sentence = tokenizer.decode(output_non_padded[0], skip_special_tokens=True)
        padded_sentence = tokenizer.decode(output_padded[0], skip_special_tokens=True)

        expected_output_sentence = [
            "Hello, my dog is a little bit of a kitty. She is a very sweet and loving",
            "Today, I am going to talk about the best way to get a job in the",
        ]
        self.assertListEqual(expected_output_sentence, batch_out_sentence)
        self.assertListEqual(expected_output_sentence, [non_padded_sentence, padded_sentence])

    @slow
    def test_model_from_pretrained(self):
        model_name = "EleutherAI/gpt-neo-1.3B"
        model = GPTNeoModel.from_pretrained(model_name)
        self.assertIsNotNone(model)
