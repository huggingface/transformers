# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

import unittest

from transformers import AutoTokenizer, GPTJConfig, is_tf_available
from transformers.testing_utils import require_tf, slow, tooslow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin
from ...utils.test_modeling_tf_core import TFCoreModelTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers.models.gptj.modeling_tf_gptj import (
        TFGPTJForCausalLM,
        TFGPTJForQuestionAnswering,
        TFGPTJForSequenceClassification,
        TFGPTJModel,
        shape_list,
    )


class TFGPTJModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_token_type_ids = True
        self.use_input_mask = True
        self.use_labels = True
        self.use_mc_token_ids = True
        self.vocab_size = 99
        self.hidden_size = 32
        self.rotary_dim = 4
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.intermediate_size = 37
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 16
        self.type_sequence_label_size = 2
        self.initializer_range = 0.02
        self.num_labels = 3
        self.num_choices = 4
        self.scope = None
        self.bos_token_id = self.vocab_size - 1
        self.eos_token_id = self.vocab_size - 1
        self.pad_token_id = self.vocab_size - 1

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

        config = GPTJConfig(
            vocab_size=self.vocab_size,
            n_embd=self.hidden_size,
            n_layer=self.num_hidden_layers,
            n_head=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            n_positions=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            rotary_dim=self.rotary_dim,
            return_dict=True,
        )

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

    def create_and_check_gptj_model(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        model = TFGPTJModel(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }
        result = model(inputs)

        inputs = [input_ids, None, input_mask]  # None is the input for 'past'
        result = model(inputs)

        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_gptj_model_past(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        model = TFGPTJModel(config=config)

        # first forward pass
        outputs = model(input_ids, token_type_ids=token_type_ids, use_cache=True)
        outputs_use_cache_conf = model(input_ids, token_type_ids=token_type_ids)
        outputs_no_past = model(input_ids, token_type_ids=token_type_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)
        next_token_types = ids_tensor([self.batch_size, 1], self.type_vocab_size)

        # append to next input_ids and token_type_ids
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        next_token_type_ids = tf.concat([token_type_ids, next_token_types], axis=-1)

        output_from_no_past = model(next_input_ids, token_type_ids=next_token_type_ids)["last_hidden_state"]
        output_from_past = model(next_tokens, token_type_ids=next_token_types, past_key_values=past_key_values)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = int(ids_tensor((1,), shape_list(output_from_past)[-1]))
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
        output_from_past_slice = output_from_past[:, 0, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-6)

    def create_and_check_gptj_model_attention_mask_past(
        self, config, input_ids, input_mask, head_mask, token_type_ids, *args
    ):
        model = TFGPTJModel(config=config)

        # create attention mask
        half_seq_length = self.seq_length // 2
        attn_mask_begin = tf.ones((self.batch_size, half_seq_length), dtype=tf.int32)
        attn_mask_end = tf.zeros((self.batch_size, self.seq_length - half_seq_length), dtype=tf.int32)
        attn_mask = tf.concat([attn_mask_begin, attn_mask_end], axis=1)

        # first forward pass
        output, past_key_values = model(input_ids, attention_mask=attn_mask).to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # change a random masked slice from input_ids
        random_seq_idx_to_change = ids_tensor((1,), half_seq_length).numpy() + 1
        random_other_next_tokens = ids_tensor((self.batch_size, self.seq_length), config.vocab_size)
        vector_condition = tf.range(self.seq_length) == (self.seq_length - random_seq_idx_to_change)
        condition = tf.transpose(
            tf.broadcast_to(tf.expand_dims(vector_condition, -1), (self.seq_length, self.batch_size))
        )
        input_ids = tf.where(condition, random_other_next_tokens, input_ids)

        # append to next input_ids and attn_mask
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        attn_mask = tf.concat([attn_mask, tf.ones((shape_list(attn_mask)[0], 1), dtype=tf.int32)], axis=1)

        # get two different outputs
        output_from_no_past = model(next_input_ids, attention_mask=attn_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past_key_values, attention_mask=attn_mask)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = int(ids_tensor((1,), shape_list(output_from_past)[-1]))
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx]
        output_from_past_slice = output_from_past[:, 0, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-12)

    def create_and_check_gptj_model_past_large_inputs(
        self, config, input_ids, input_mask, head_mask, token_type_ids, *args
    ):
        model = TFGPTJModel(config=config)

        input_ids = input_ids[:1, :]
        input_mask = input_mask[:1, :]
        token_type_ids = token_type_ids[:1, :]
        self.batch_size = 1

        # first forward pass
        outputs = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)
        next_token_types = ids_tensor((self.batch_size, 3), self.type_vocab_size)

        # append to next input_ids and token_type_ids
        next_input_ids = tf.concat([input_ids, next_tokens], axis=-1)
        next_attention_mask = tf.concat([input_mask, next_attn_mask], axis=-1)
        next_token_type_ids = tf.concat([token_type_ids, next_token_types], axis=-1)

        output_from_no_past = model(
            next_input_ids, token_type_ids=next_token_type_ids, attention_mask=next_attention_mask
        )["last_hidden_state"]
        output_from_past = model(
            next_tokens,
            token_type_ids=next_token_types,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
        )["last_hidden_state"]
        self.parent.assertTrue(output_from_past.shape[1] == next_tokens.shape[1])

        # select random slice
        random_slice_idx = int(ids_tensor((1,), shape_list(output_from_past)[-1]))
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx]
        output_from_past_slice = output_from_past[:, :, random_slice_idx]

        # test that outputs are equal for slice
        tf.debugging.assert_near(output_from_past_slice, output_from_no_past_slice, rtol=1e-3)

    def create_and_check_gptj_lm_head_model(self, config, input_ids, input_mask, head_mask, token_type_ids, *args):
        model = TFGPTJForCausalLM(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        }
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

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
            "attention_mask": input_mask,
        }
        return config, inputs_dict


@require_tf
class TFGPTJModelTest(TFModelTesterMixin, TFCoreModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (TFGPTJForCausalLM, TFGPTJForSequenceClassification, TFGPTJForQuestionAnswering, TFGPTJModel)
        if is_tf_available()
        else ()
    )

    all_generative_model_classes = (TFGPTJForCausalLM,) if is_tf_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": TFGPTJModel,
            "question-answering": TFGPTJForQuestionAnswering,
            "text-classification": TFGPTJForSequenceClassification,
            "text-generation": TFGPTJForCausalLM,
            "zero-shot": TFGPTJForSequenceClassification,
        }
        if is_tf_available()
        else {}
    )
    test_onnx = False
    test_pruning = False
    test_missing_keys = False
    test_head_masking = False

    # TODO: Fix the failed tests
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        if (
            pipeline_test_casse_name == "QAPipelineTests"
            and tokenizer_name is not None
            and not tokenizer_name.endswith("Fast")
        ):
            # `QAPipelineTests` fails for a few models when the slower tokenizer are used.
            # (The slower tokenizers were never used for pipeline tests before the pipeline testing rework)
            # TODO: check (and possibly fix) the `QAPipelineTests` with slower tokenizer
            return True

        return False

    def setUp(self):
        self.model_tester = TFGPTJModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GPTJConfig, n_embd=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_gptj_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gptj_model(*config_and_inputs)

    def test_gptj_model_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gptj_model_past(*config_and_inputs)

    def test_gptj_model_att_mask_past(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gptj_model_attention_mask_past(*config_and_inputs)

    def test_gptj_model_past_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gptj_model_past_large_inputs(*config_and_inputs)

    def test_gptj_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_gptj_lm_head_model(*config_and_inputs)

    @slow
    @unittest.skipIf(
        not is_tf_available() or len(tf.config.list_physical_devices("GPU")) > 0,
        "skip testing on GPU for now to avoid GPU OOM.",
    )
    def test_model_from_pretrained(self):
        model = TFGPTJModel.from_pretrained("EleutherAI/gpt-j-6B", from_pt=True)
        self.assertIsNotNone(model)

    @unittest.skip(reason="Currently, model embeddings are going to undergo a major refactor.")
    def test_resize_token_embeddings(self):
        super().test_resize_token_embeddings()


@require_tf
@tooslow
# Marked as @tooslow due to GPU OOM -- but still useful to run locally. Requires ~39GB of RAM.
class TFGPTJModelLanguageGenerationTest(unittest.TestCase):
    def test_lm_generate_gptj(self):
        model = TFGPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", from_pt=True)
        input_ids = tf.convert_to_tensor([[464, 3290]], dtype=tf.int32)  # The dog
        # fmt: off
        # The dog is a man's best friend. It is a loyal companion, and it is a friend
        expected_output_ids = [464, 3290, 318, 257, 582, 338, 1266, 1545, 13, 632, 318, 257, 9112, 15185, 11, 290, 340, 318, 257, 1545]
        # fmt: on
        output_ids = model.generate(input_ids, do_sample=False)
        self.assertListEqual(output_ids[0].numpy().tolist(), expected_output_ids)

    def test_gptj_sample(self):
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", revision="float16")
        model = TFGPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", from_pt=True)

        tokenized = tokenizer("Today is a nice day and", return_tensors="tf")
        # forces the generation to happen on CPU, to avoid GPU-related quirks
        with tf.device(":/CPU:0"):
            output_ids = model.generate(**tokenized, do_sample=True, seed=[42, 0])
        output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        EXPECTED_OUTPUT_STR = "Today is a nice day and I’m going to go for a walk. I’"
        self.assertEqual(output_str, EXPECTED_OUTPUT_STR)

    def _get_beam_search_test_objects(self):
        model = TFGPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", from_pt=True)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", revision="float16")

        tokenizer.padding_side = "left"

        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        # use different length sentences to test batching
        sentences = [
            "Hello, my dog is a little",
            "Today, I",
        ]
        expected_output_sentences = [
            "Hello, my dog is a little over a year old and has been diagnosed with hip dysplasia",
            "Today, I’m going to be talking about a topic that’",
        ]
        return model, tokenizer, sentences, expected_output_sentences

    def test_batch_beam_search(self):
        # Confirms that we get the expected results with left-padded beam search
        model, tokenizer, sentences, expected_output_sentences = self._get_beam_search_test_objects()

        inputs = tokenizer(sentences, return_tensors="tf", padding=True)
        outputs = model.generate(**inputs, do_sample=False, num_beams=2)
        batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.assertListEqual(expected_output_sentences, batch_out_sentence)

    def test_batch_left_padding(self):
        # Confirms that left-padding is working properly
        model, tokenizer, sentences, expected_output_sentences = self._get_beam_search_test_objects()

        inputs = tokenizer(sentences, return_tensors="tf", padding=True)
        inputs_non_padded = tokenizer(sentences[0], return_tensors="tf")
        output_non_padded = model.generate(**inputs_non_padded, do_sample=False, num_beams=2)
        num_paddings = (
            shape_list(inputs_non_padded["input_ids"])[-1]
            - tf.reduce_sum(tf.cast(inputs["attention_mask"][-1], tf.int64)).numpy()
        )
        inputs_padded = tokenizer(sentences[1], return_tensors="tf")
        output_padded = model.generate(
            **inputs_padded, do_sample=False, num_beams=2, max_length=model.config.max_length - num_paddings
        )
        non_padded_sentence = tokenizer.decode(output_non_padded[0], skip_special_tokens=True)
        padded_sentence = tokenizer.decode(output_padded[0], skip_special_tokens=True)
        self.assertListEqual(expected_output_sentences, [non_padded_sentence, padded_sentence])

    def test_xla_beam_search(self):
        # Confirms that XLA is working properly
        model, tokenizer, sentences, expected_output_sentences = self._get_beam_search_test_objects()

        inputs = tokenizer(sentences, return_tensors="tf", padding=True)
        xla_generate = tf.function(model.generate, jit_compile=True)
        outputs_xla = xla_generate(**inputs, do_sample=False, num_beams=2)
        xla_sentence = tokenizer.batch_decode(outputs_xla, skip_special_tokens=True)
        self.assertListEqual(expected_output_sentences, xla_sentence)
