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

import unittest

from transformers import XLMConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        XLMForMultipleChoice,
        XLMForQuestionAnswering,
        XLMForQuestionAnsweringSimple,
        XLMForSequenceClassification,
        XLMForTokenClassification,
        XLMModel,
        XLMWithLMHeadModel,
    )
    from transformers.models.xlm.modeling_xlm import XLM_PRETRAINED_MODEL_ARCHIVE_LIST


class XLMModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_lengths=True,
        use_token_type_ids=True,
        use_labels=True,
        gelu_activation=True,
        sinusoidal_embeddings=False,
        causal=False,
        asm=False,
        n_langs=2,
        vocab_size=99,
        n_special=0,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=2,
        num_choices=4,
        summary_type="last",
        use_proj=True,
        scope=None,
        bos_token_id=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_lengths = use_input_lengths
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.gelu_activation = gelu_activation
        self.sinusoidal_embeddings = sinusoidal_embeddings
        self.causal = causal
        self.asm = asm
        self.n_langs = n_langs
        self.vocab_size = vocab_size
        self.n_special = n_special
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.summary_type = summary_type
        self.use_proj = use_proj
        self.scope = scope
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = random_attention_mask([self.batch_size, self.seq_length])

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
            is_impossible_labels = ids_tensor([self.batch_size], 2).float()
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

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

    def get_config(self):
        return XLMConfig(
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
            num_labels=self.num_labels,
            bos_token_id=self.bos_token_id,
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
        model = XLMModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, lengths=input_lengths, langs=token_type_ids)
        result = model(input_ids, langs=token_type_ids)
        result = model(input_ids)
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
        model = XLMWithLMHeadModel(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_xlm_simple_qa(
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
        model = XLMForQuestionAnsweringSimple(config)
        model.to(torch_device)
        model.eval()

        outputs = model(input_ids)

        outputs = model(input_ids, start_positions=sequence_labels, end_positions=sequence_labels)
        result = outputs
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

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
        model = XLMForQuestionAnswering(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids)

        result_with_labels = model(
            input_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
            cls_index=sequence_labels,
            is_impossible=is_impossible_labels,
            p_mask=input_mask,
        )

        result_with_labels = model(
            input_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
            cls_index=sequence_labels,
            is_impossible=is_impossible_labels,
        )

        (total_loss,) = result_with_labels.to_tuple()

        result_with_labels = model(input_ids, start_positions=sequence_labels, end_positions=sequence_labels)

        (total_loss,) = result_with_labels.to_tuple()

        self.parent.assertEqual(result_with_labels.loss.shape, ())
        self.parent.assertEqual(result.start_top_log_probs.shape, (self.batch_size, model.config.start_n_top))
        self.parent.assertEqual(result.start_top_index.shape, (self.batch_size, model.config.start_n_top))
        self.parent.assertEqual(
            result.end_top_log_probs.shape, (self.batch_size, model.config.start_n_top * model.config.end_n_top)
        )
        self.parent.assertEqual(
            result.end_top_index.shape, (self.batch_size, model.config.start_n_top * model.config.end_n_top)
        )
        self.parent.assertEqual(result.cls_logits.shape, (self.batch_size,))

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
        model = XLMForSequenceClassification(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids)
        result = model(input_ids, labels=sequence_labels)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def create_and_check_xlm_token_classif(
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
        model = XLMForTokenClassification(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
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
        model = XLMForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        result = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
            token_type_ids=multiple_choice_token_type_ids,
            labels=choice_labels,
        )
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
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "lengths": input_lengths}
        return config, inputs_dict


@require_torch
class XLMModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            XLMModel,
            XLMWithLMHeadModel,
            XLMForQuestionAnswering,
            XLMForSequenceClassification,
            XLMForQuestionAnsweringSimple,
            XLMForTokenClassification,
            XLMForMultipleChoice,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (
        (XLMWithLMHeadModel,) if is_torch_available() else ()
    )  # TODO (PVP): Check other models whether language generation is also applicable
    pipeline_model_mapping = (
        {
            "feature-extraction": XLMModel,
            "fill-mask": XLMWithLMHeadModel,
            "question-answering": XLMForQuestionAnsweringSimple,
            "text-classification": XLMForSequenceClassification,
            "text-generation": XLMWithLMHeadModel,
            "token-classification": XLMForTokenClassification,
            "zero-shot": XLMForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    # TODO: Fix the failed tests
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        if pipeline_test_casse_name == "FillMaskPipelineTests":
            # Get `ValueError: AttributeError: 'NoneType' object has no attribute 'new_ones'` or `AssertionError`.
            # `XLMConfig` was never used in pipeline tests: cannot create a simple tokenizer
            return True
        elif (
            pipeline_test_casse_name == "QAPipelineTests"
            and tokenizer_name is not None
            and not tokenizer_name.endswith("Fast")
        ):
            # `QAPipelineTests` fails for a few models when the slower tokenizer are used.
            # (The slower tokenizers were never used for pipeline tests before the pipeline testing rework)
            # TODO: check (and possibly fix) the `QAPipelineTests` with slower tokenizer
            return True

        return False

    # XLM has 2 QA models -> need to manually set the correct labels for one of them here
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ == "XLMForQuestionAnswering":
                inputs_dict["start_positions"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
                inputs_dict["end_positions"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )

        return inputs_dict

    def setUp(self):
        self.model_tester = XLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=XLMConfig, emb_dim=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_xlm_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_model(*config_and_inputs)

    def test_xlm_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_lm_head(*config_and_inputs)

    def test_xlm_simple_qa(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_simple_qa(*config_and_inputs)

    def test_xlm_qa(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_qa(*config_and_inputs)

    def test_xlm_sequence_classif(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_sequence_classif(*config_and_inputs)

    def test_xlm_token_classif(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_token_classif(*config_and_inputs)

    def test_xlm_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_for_multiple_choice(*config_and_inputs)

    def _check_attentions_for_generate(
        self, batch_size, attentions, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [isinstance(iter_attentions, tuple) for iter_attentions in attentions], [True] * len(attentions)
        )
        self.assertEqual(len(attentions), (max_length - min_length) * num_beam_groups)

        for idx, iter_attentions in enumerate(attentions):
            # adds PAD dummy token
            tgt_len = min_length + idx + 1
            src_len = min_length + idx + 1

            expected_shape = (
                batch_size * num_beam_groups,
                config.num_attention_heads,
                tgt_len,
                src_len,
            )
            # check attn size
            self.assertListEqual(
                [layer_attention.shape for layer_attention in iter_attentions], [expected_shape] * len(iter_attentions)
            )

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [isinstance(iter_hidden_states, tuple) for iter_hidden_states in hidden_states],
            [True] * len(hidden_states),
        )
        self.assertEqual(len(hidden_states), (max_length - min_length) * num_beam_groups)

        for idx, iter_hidden_states in enumerate(hidden_states):
            # adds PAD dummy token
            seq_len = min_length + idx + 1
            expected_shape = (batch_size * num_beam_groups, seq_len, config.hidden_size)
            # check hidden size
            self.assertListEqual(
                [layer_hidden_states.shape for layer_hidden_states in iter_hidden_states],
                [expected_shape] * len(iter_hidden_states),
            )
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in XLM_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = XLMModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class XLMModelLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_xlm_mlm_en_2048(self):
        model = XLMWithLMHeadModel.from_pretrained("xlm-mlm-en-2048")
        model.to(torch_device)
        input_ids = torch.tensor([[14, 447]], dtype=torch.long, device=torch_device)  # the president
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
        self.assertListEqual(output_ids[0].cpu().numpy().tolist(), expected_output_ids)
