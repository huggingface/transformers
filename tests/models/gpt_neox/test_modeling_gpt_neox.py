# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch GPTNeoX model."""

import unittest

from transformers import AutoTokenizer, DynamicCache, GPTNeoXConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        GPTNeoXForCausalLM,
        GPTNeoXForQuestionAnswering,
        GPTNeoXForSequenceClassification,
        GPTNeoXForTokenClassification,
        GPTNeoXModel,
    )


class GPTNeoXModelTester:
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
        hidden_size=64,
        num_hidden_layers=2,
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
        self.pad_token_id = vocab_size - 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_labels = None
        if self.use_labels:
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()

        return config, input_ids, input_mask, token_labels

    def get_config(self):
        return GPTNeoXConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def prepare_config_and_inputs_for_decoder(self):
        config, input_ids, input_mask, token_labels = self.prepare_config_and_inputs()

        config.is_decoder = True

        return config, input_ids, input_mask, token_labels

    def create_and_check_model(self, config, input_ids, input_mask):
        model = GPTNeoXModel(config=config)
        model.to(torch_device)
        model.eval()
        _ = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(self, config, input_ids, input_mask):
        config.add_cross_attention = True
        model = GPTNeoXModel(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(self, config, input_ids, input_mask, token_labels):
        model = GPTNeoXForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_question_answering(self, config, input_ids, input_mask, token_labels):
        config.num_labels = self.num_labels
        model = GPTNeoXForQuestionAnswering(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_sequence_classification(self, config, input_ids, input_mask, token_labels):
        config.num_labels = self.num_labels
        model = GPTNeoXForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(self, config, input_ids, input_mask, token_labels):
        config.num_labels = self.num_labels
        model = GPTNeoXForTokenClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_decoder_model_past_large_inputs(self, config, input_ids, input_mask):
        config.is_decoder = True
        model = GPTNeoXForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(input_ids, attention_mask=input_mask, use_cache=True)
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask, output_hidden_states=True)
        output_from_no_past = output_from_no_past["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_cached_forward_with_and_without_attention_mask(self, config, input_ids, *args):
        # Relevant issue: https://github.com/huggingface/transformers/issues/31943
        model = GPTNeoXModel(config)
        model.to(torch_device)
        model.eval()

        # We want this for SDPA, eager works with a `None` attention mask
        assert model.config._attn_implementation == "sdpa", (
            "This test assumes the model to have the SDPA implementation for its attention calculations."
        )

        # Prepare cache and non_cache input, needs a full attention mask
        cached_len = input_ids.shape[-1] // 2
        input_mask = torch.ones(size=input_ids.size()).to(torch_device)
        cache_inputs = {"input_ids": input_ids[:, :cached_len], "attention_mask": input_mask[:, :cached_len]}
        non_cache_inputs = {"input_ids": input_ids[:, cached_len:], "attention_mask": input_mask}

        def copy_cache(cache: DynamicCache):
            """Deep copy a DynamicCache to reuse the same one multiple times."""
            new_cache = cache
            for i in range(len(cache)):
                new_cache.layers[i].keys = cache.layers[i].keys.clone()
                new_cache.layers[i].values = cache.layers[i].values.clone()

        # Cached forward once with the attention mask provided and the other time without it (which should assume full attention)
        # We need to run both on a copy of the cache, otherwise it is modified in-place
        cache_outputs = model(**cache_inputs)
        cache = cache_outputs.past_key_values
        full_outputs_with_attention_mask = model(
            **non_cache_inputs, past_key_values=copy_cache(cache)
        ).last_hidden_state
        full_outputs_without_attention_mask = model(
            non_cache_inputs["input_ids"], past_key_values=copy_cache(cache)
        ).last_hidden_state

        self.parent.assertTrue(
            torch.allclose(full_outputs_with_attention_mask, full_outputs_without_attention_mask, atol=1e-5)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask, token_labels = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class GPTNeoXModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            GPTNeoXModel,
            GPTNeoXForCausalLM,
            GPTNeoXForQuestionAnswering,
            GPTNeoXForSequenceClassification,
            GPTNeoXForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": GPTNeoXModel,
            "question-answering": GPTNeoXForQuestionAnswering,
            "text-classification": GPTNeoXForSequenceClassification,
            "text-generation": GPTNeoXForCausalLM,
            "token-classification": GPTNeoXForTokenClassification,
            "zero-shot": GPTNeoXForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    test_missing_keys = False

    def setUp(self):
        self.model_tester = GPTNeoXModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GPTNeoXConfig, hidden_size=64, num_attention_heads=8)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, input_ids, input_mask, token_labels = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, input_ids, input_mask)

    def test_model_as_decoder(self):
        config, input_ids, input_mask, token_labels = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_model_as_decoder(config, input_ids, input_mask)

    def test_model_as_decoder_with_default_input_mask(self):
        config, input_ids, input_mask, token_labels = self.model_tester.prepare_config_and_inputs_for_decoder()

        input_mask = None

        self.model_tester.create_and_check_model_as_decoder(config, input_ids, input_mask)

    def test_decoder_model_past_large_inputs(self):
        config, input_ids, input_mask, token_labels = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(config, input_ids, input_mask)

    def test_model_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_model_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_model_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_model_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_cached_forward_with_and_without_attention_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_cached_forward_with_and_without_attention_mask(*config_and_inputs)

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass


@require_torch
class GPTNeoXLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_gptneox(self):
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
        for checkpointing in [True, False]:
            model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-410m-deduped")

            if checkpointing:
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_disable()
            model.to(torch_device)

            inputs = tokenizer("My favorite food is", return_tensors="pt").to(torch_device)
            # The hub repo. is updated on 2023-04-04, resulting in poor outputs.
            # See: https://github.com/huggingface/transformers/pull/24193
            expected_output = "My favorite food is a good old-fashioned, old-fashioned, old-fashioned.\n\nI'm not sure"

            output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
            output_str = tokenizer.batch_decode(output_ids)[0]

            self.assertEqual(output_str, expected_output)

    @slow
    def test_lm_generate_flex_attn_gptneox(self):
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
        for checkpointing in [True, False]:
            model = GPTNeoXForCausalLM.from_pretrained(
                "EleutherAI/pythia-410m-deduped", attn_implementation="flex_attention"
            )
            self.assertTrue(model.config._attn_implementation == "flex_attention")

            if checkpointing:
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_disable()
            model.to(torch_device)

            inputs = tokenizer("My favorite food is", return_tensors="pt").to(torch_device)
            # The hub repo. is updated on 2023-04-04, resulting in poor outputs.
            # See: https://github.com/huggingface/transformers/pull/24193
            expected_output = "My favorite food is a good old-fashioned, old-fashioned, old-fashioned.\n\nI'm not sure"

            output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
            output_str = tokenizer.batch_decode(output_ids)[0]

            self.assertEqual(output_str, expected_output)

    def pythia_integration_test(self):
        model_name_or_path = "EleutherAI/pythia-70m"
        model = GPTNeoXForCausalLM.from_pretrained(model_name_or_path, dtype=torch.float16).to(torch_device)
        EXPECTED_LOGITS = torch.tensor([1069.0000,  228.7500, 1072.0000, 1072.0000, 1069.0000, 1068.0000, 1068.0000, 1071.0000, 1071.0000, 1071.0000, 1073.0000, 1070.0000, 1071.0000, 1075.0000, 1073.0000, 1075.0000, 1074.0000, 1069.0000, 1072.0000, 1071.0000, 1071.0000, 1071.0000, 1070.0000, 1069.0000, 1069.0000, 1069.0000, 1070.0000, 1075.0000, 1073.0000, 1074.0000])  # fmt: skip
        input_ids = [29, 93, 303, 64, 5478, 49651, 10394, 187, 34, 12939, 875]
        # alternative: tokenizer('<|im_start|>system\nA chat between')
        input_ids = torch.as_tensor(input_ids)[None].to(torch_device)
        outputs = model(input_ids)["logits"][:, -1][0, :30]
        torch.testing.assert_close(EXPECTED_LOGITS, outputs, rtol=1e-5, atol=1e-5)
