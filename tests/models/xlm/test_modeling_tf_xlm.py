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

from transformers import is_tf_available
from transformers.testing_utils import require_tf, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFXLMForMultipleChoice,
        TFXLMForQuestionAnsweringSimple,
        TFXLMForSequenceClassification,
        TFXLMForTokenClassification,
        TFXLMModel,
        TFXLMWithLMHeadModel,
        XLMConfig,
    )


class TFXLMModelTester:
    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_input_lengths = True
        self.use_token_type_ids = True
        self.use_labels = True
        self.gelu_activation = True
        self.sinusoidal_embeddings = False
        self.causal = False
        self.asm = False
        self.n_langs = 2
        self.vocab_size = 99
        self.n_special = 0
        self.hidden_size = 32
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 16
        self.type_sequence_label_size = 2
        self.initializer_range = 0.02
        self.num_labels = 3
        self.num_choices = 4
        self.summary_type = "last"
        self.use_proj = True
        self.scope = None
        self.bos_token_id = 0

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = random_attention_mask([self.batch_size, self.seq_length], dtype=tf.float32)

        input_lengths = None
        if self.use_input_lengths:
            input_lengths = (
                ids_tensor([self.batch_size], vocab_size=2) + self.seq_length - 2
            )  # small variation of seq_length

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.n_langs)

        sequence_labels = None
        token_labels = None
        is_impossible_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            is_impossible_labels = ids_tensor([self.batch_size], 2, dtype=tf.float32)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = XLMConfig(
            vocab_size=self.vocab_size,
            n_special=self.n_special,
            emb_dim=self.hidden_size,
            n_layers=self.num_hidden_layers,
            n_heads=self.num_attention_heads,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            gelu_activation=self.gelu_activation,
            sinusoidal_embeddings=self.sinusoidal_embeddings,
            asm=self.asm,
            causal=self.causal,
            n_langs=self.n_langs,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            summary_type=self.summary_type,
            use_proj=self.use_proj,
            bos_token_id=self.bos_token_id,
        )

        return (
            config,
            input_ids,
            token_type_ids,
            input_lengths,
            sequence_labels,
            token_labels,
            is_impossible_labels,
            choice_labels,
            input_mask,
        )

    def create_and_check_xlm_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        choice_labels,
        input_mask,
    ):
        model = TFXLMModel(config=config)
        inputs = {"input_ids": input_ids, "lengths": input_lengths, "langs": token_type_ids}
        result = model(inputs)

        inputs = [input_ids, input_mask]
        result = model(inputs)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_xlm_lm_head(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        choice_labels,
        input_mask,
    ):
        model = TFXLMWithLMHeadModel(config)

        inputs = {"input_ids": input_ids, "lengths": input_lengths, "langs": token_type_ids}
        outputs = model(inputs)

        result = outputs

        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_xlm_qa(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        choice_labels,
        input_mask,
    ):
        model = TFXLMForQuestionAnsweringSimple(config)

        inputs = {"input_ids": input_ids, "lengths": input_lengths}

        result = model(inputs)

        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_xlm_sequence_classif(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        choice_labels,
        input_mask,
    ):
        model = TFXLMForSequenceClassification(config)

        inputs = {"input_ids": input_ids, "lengths": input_lengths}

        result = model(inputs)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def create_and_check_xlm_for_token_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        choice_labels,
        input_mask,
    ):
        config.num_labels = self.num_labels
        model = TFXLMForTokenClassification(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_xlm_for_multiple_choice(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        choice_labels,
        input_mask,
    ):
        config.num_choices = self.num_choices
        model = TFXLMForMultipleChoice(config=config)
        multiple_choice_inputs_ids = tf.tile(tf.expand_dims(input_ids, 1), (1, self.num_choices, 1))
        multiple_choice_input_mask = tf.tile(tf.expand_dims(input_mask, 1), (1, self.num_choices, 1))
        multiple_choice_token_type_ids = tf.tile(tf.expand_dims(token_type_ids, 1), (1, self.num_choices, 1))
        inputs = {
            "input_ids": multiple_choice_inputs_ids,
            "attention_mask": multiple_choice_input_mask,
            "token_type_ids": multiple_choice_token_type_ids,
        }
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_lengths,
            sequence_labels,
            token_labels,
            is_impossible_labels,
            choice_labels,
            input_mask,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "langs": token_type_ids,
            "lengths": input_lengths,
        }
        return config, inputs_dict


@require_tf
class TFXLMModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            TFXLMModel,
            TFXLMWithLMHeadModel,
            TFXLMForSequenceClassification,
            TFXLMForQuestionAnsweringSimple,
            TFXLMForTokenClassification,
            TFXLMForMultipleChoice,
        )
        if is_tf_available()
        else ()
    )
    all_generative_model_classes = (
        (TFXLMWithLMHeadModel,) if is_tf_available() else ()
    )  # TODO (PVP): Check other models whether language generation is also applicable
    pipeline_model_mapping = (
        {
            "feature-extraction": TFXLMModel,
            "fill-mask": TFXLMWithLMHeadModel,
            "question-answering": TFXLMForQuestionAnsweringSimple,
            "text-classification": TFXLMForSequenceClassification,
            "text-generation": TFXLMWithLMHeadModel,
            "token-classification": TFXLMForTokenClassification,
            "zero-shot": TFXLMForSequenceClassification,
        }
        if is_tf_available()
        else {}
    )
    test_head_masking = False
    test_onnx = False

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
        self.model_tester = TFXLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=XLMConfig, emb_dim=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_xlm_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_model(*config_and_inputs)

    def test_xlm_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_lm_head(*config_and_inputs)

    def test_xlm_qa(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_qa(*config_and_inputs)

    def test_xlm_sequence_classif(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_sequence_classif(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_for_token_classification(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_for_multiple_choice(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFXLMModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_tf
class TFXLMModelLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_xlm_mlm_en_2048(self):
        model = TFXLMWithLMHeadModel.from_pretrained("FacebookAI/xlm-mlm-en-2048")
        input_ids = tf.convert_to_tensor([[14, 447]], dtype=tf.int32)  # the president
        expected_output_ids = [
            14,
            447,
            14,
            447,
            14,
            447,
            14,
            447,
            14,
            447,
            14,
            447,
            14,
            447,
            14,
            447,
            14,
            447,
            14,
            447,
        ]  # the president the president the president the president the president the president the president the president the president the president
        # TODO(PVP): this and other input_ids I tried for generation give pretty bad results. Not sure why. Model might just not be made for auto-regressive inference
        output_ids = model.generate(input_ids, do_sample=False)
        self.assertListEqual(output_ids[0].numpy().tolist(), expected_output_ids)
