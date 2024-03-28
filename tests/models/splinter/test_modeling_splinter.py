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
""" Testing suite for the PyTorch Splinter model. """

import copy
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, require_torch_multi_gpu, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import SplinterConfig, SplinterForPreTraining, SplinterForQuestionAnswering, SplinterModel


class SplinterModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        num_questions=3,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        question_token_id=1,
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
        self.num_questions = num_questions
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.question_token_id = question_token_id
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

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids[:, 1] = self.question_token_id

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        start_positions = None
        end_positions = None
        question_positions = None
        if self.use_labels:
            start_positions = ids_tensor([self.batch_size, self.num_questions], self.type_sequence_label_size)
            end_positions = ids_tensor([self.batch_size, self.num_questions], self.type_sequence_label_size)
            question_positions = ids_tensor([self.batch_size, self.num_questions], self.num_labels)

        config = SplinterConfig(
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
            question_token_id=self.question_token_id,
        )

        return (config, input_ids, token_type_ids, input_mask, start_positions, end_positions, question_positions)

    def create_and_check_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        start_positions,
        end_positions,
        question_positions,
    ):
        model = SplinterModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        start_positions,
        end_positions,
        question_positions,
    ):
        model = SplinterForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions[:, 0],
            end_positions=end_positions[:, 0],
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_pretraining(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        start_positions,
        end_positions,
        question_positions,
    ):
        model = SplinterForPreTraining(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
            question_positions=question_positions,
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.num_questions, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.num_questions, self.seq_length))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            start_positions,
            end_positions,
            question_positions,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }
        return config, inputs_dict


@require_torch
class SplinterModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            SplinterModel,
            SplinterForQuestionAnswering,
            SplinterForPreTraining,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"feature-extraction": SplinterModel, "question-answering": SplinterForQuestionAnswering}
        if is_torch_available()
        else {}
    )

    # TODO: Fix the failed tests when this model gets more usage
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        if pipeline_test_casse_name == "QAPipelineTests":
            return True
        elif pipeline_test_casse_name == "FeatureExtractionPipelineTests" and tokenizer_name.endswith("Fast"):
            return True

        return False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)
        if return_labels:
            if issubclass(model_class, SplinterForPreTraining):
                inputs_dict["start_positions"] = torch.zeros(
                    self.model_tester.batch_size,
                    self.model_tester.num_questions,
                    dtype=torch.long,
                    device=torch_device,
                )
                inputs_dict["end_positions"] = torch.zeros(
                    self.model_tester.batch_size,
                    self.model_tester.num_questions,
                    dtype=torch.long,
                    device=torch_device,
                )
                inputs_dict["question_positions"] = torch.zeros(
                    self.model_tester.batch_size,
                    self.model_tester.num_questions,
                    dtype=torch.long,
                    device=torch_device,
                )
            elif issubclass(model_class, SplinterForQuestionAnswering):
                inputs_dict["start_positions"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
                inputs_dict["end_positions"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )

        return inputs_dict

    def setUp(self):
        self.model_tester = SplinterModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SplinterConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            wte = model.get_input_embeddings()
            if not self.is_encoder_decoder:
                inputs["inputs_embeds"] = wte(input_ids)
            else:
                inputs["inputs_embeds"] = wte(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                if isinstance(model, SplinterForPreTraining):
                    with self.assertRaises(TypeError):
                        # question_positions must not be None.
                        model(**inputs)[0]
                else:
                    model(**inputs)[0]

    @slow
    def test_model_from_pretrained(self):
        model_name = "tau/splinter-base"
        model = SplinterModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    # overwrite from common since `SplinterForPreTraining` could contain different number of question tokens in inputs.
    # When the batch is distributed to multiple devices, each replica could get different values for the maximal number
    # of question tokens (see `SplinterForPreTraining._prepare_question_positions()`), and the model returns different
    # shape along dimension 1 (i.e. `num_questions`) that could not be combined into a single tensor as an output.
    @require_torch_multi_gpu
    def test_multi_gpu_data_parallel_forward(self):
        from torch import nn

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # some params shouldn't be scattered by nn.DataParallel
        # so just remove them if they are present.
        blacklist_non_batched_params = ["head_mask", "decoder_head_mask", "cross_attn_head_mask"]
        for k in blacklist_non_batched_params:
            inputs_dict.pop(k, None)

        # move input tensors to cuda:O
        for k, v in inputs_dict.items():
            if torch.is_tensor(v):
                inputs_dict[k] = v.to(0)

        for model_class in self.all_model_classes:
            # Skip this case since it will fail sometimes, as described above.
            if model_class == SplinterForPreTraining:
                continue

            model = model_class(config=config)
            model.to(0)
            model.eval()

            # Wrap model in nn.DataParallel
            model = nn.DataParallel(model)
            with torch.no_grad():
                _ = model(**self._prepare_for_class(inputs_dict, model_class))


@require_torch
class SplinterModelIntegrationTest(unittest.TestCase):
    @slow
    def test_splinter_question_answering(self):
        model = SplinterForQuestionAnswering.from_pretrained("tau/splinter-base-qass")

        # Input: "[CLS] Brad was born in [QUESTION] . He returned to the United Kingdom later . [SEP]"
        # Output should be the span "the United Kingdom"
        input_ids = torch.tensor(
            [[101, 7796, 1108, 1255, 1107, 104, 119, 1124, 1608, 1106, 1103, 1244, 2325, 1224, 119, 102]]
        )
        output = model(input_ids)

        expected_shape = torch.Size((1, 16))
        self.assertEqual(output.start_logits.shape, expected_shape)
        self.assertEqual(output.end_logits.shape, expected_shape)

        self.assertEqual(torch.argmax(output.start_logits), 10)
        self.assertEqual(torch.argmax(output.end_logits), 12)

    @slow
    def test_splinter_pretraining(self):
        model = SplinterForPreTraining.from_pretrained("tau/splinter-base-qass")

        # Input: "[CLS] [QUESTION] was born in [QUESTION] . Brad returned to the United Kingdom later . [SEP]"
        # Output should be the spans "Brad" and "the United Kingdom"
        input_ids = torch.tensor(
            [[101, 104, 1108, 1255, 1107, 104, 119, 7796, 1608, 1106, 1103, 1244, 2325, 1224, 119, 102]]
        )
        question_positions = torch.tensor([[1, 5]], dtype=torch.long)
        output = model(input_ids, question_positions=question_positions)

        expected_shape = torch.Size((1, 2, 16))
        self.assertEqual(output.start_logits.shape, expected_shape)
        self.assertEqual(output.end_logits.shape, expected_shape)

        self.assertEqual(torch.argmax(output.start_logits[0, 0]), 7)
        self.assertEqual(torch.argmax(output.end_logits[0, 0]), 7)
        self.assertEqual(torch.argmax(output.start_logits[0, 1]), 10)
        self.assertEqual(torch.argmax(output.end_logits[0, 1]), 12)

    @slow
    def test_splinter_pretraining_loss_requires_question_positions(self):
        model = SplinterForPreTraining.from_pretrained("tau/splinter-base-qass")

        # Input: "[CLS] [QUESTION] was born in [QUESTION] . Brad returned to the United Kingdom later . [SEP]"
        # Output should be the spans "Brad" and "the United Kingdom"
        input_ids = torch.tensor(
            [[101, 104, 1108, 1255, 1107, 104, 119, 7796, 1608, 1106, 1103, 1244, 2325, 1224, 119, 102]]
        )
        start_positions = torch.tensor([[7, 10]], dtype=torch.long)
        end_positions = torch.tensor([7, 12], dtype=torch.long)
        with self.assertRaises(TypeError):
            model(
                input_ids,
                start_positions=start_positions,
                end_positions=end_positions,
            )

    @slow
    def test_splinter_pretraining_loss(self):
        model = SplinterForPreTraining.from_pretrained("tau/splinter-base-qass")

        # Input: "[CLS] [QUESTION] was born in [QUESTION] . Brad returned to the United Kingdom later . [SEP]"
        # Output should be the spans "Brad" and "the United Kingdom"
        input_ids = torch.tensor(
            [
                [101, 104, 1108, 1255, 1107, 104, 119, 7796, 1608, 1106, 1103, 1244, 2325, 1224, 119, 102],
                [101, 104, 1108, 1255, 1107, 104, 119, 7796, 1608, 1106, 1103, 1244, 2325, 1224, 119, 102],
            ]
        )
        start_positions = torch.tensor([[7, 10], [7, 10]], dtype=torch.long)
        end_positions = torch.tensor([[7, 12], [7, 12]], dtype=torch.long)
        question_positions = torch.tensor([[1, 5], [1, 5]], dtype=torch.long)
        output = model(
            input_ids,
            start_positions=start_positions,
            end_positions=end_positions,
            question_positions=question_positions,
        )
        self.assertAlmostEqual(output.loss.item(), 0.0024, 4)

    @slow
    def test_splinter_pretraining_loss_with_padding(self):
        model = SplinterForPreTraining.from_pretrained("tau/splinter-base-qass")

        # Input: "[CLS] [QUESTION] was born in [QUESTION] . Brad returned to the United Kingdom later . [SEP]"
        # Output should be the spans "Brad" and "the United Kingdom"
        input_ids = torch.tensor(
            [
                [101, 104, 1108, 1255, 1107, 104, 119, 7796, 1608, 1106, 1103, 1244, 2325, 1224, 119, 102],
            ]
        )
        start_positions = torch.tensor([[7, 10]], dtype=torch.long)
        end_positions = torch.tensor([7, 12], dtype=torch.long)
        question_positions = torch.tensor([[1, 5]], dtype=torch.long)
        start_positions_with_padding = torch.tensor([[7, 10, 0]], dtype=torch.long)
        end_positions_with_padding = torch.tensor([7, 12, 0], dtype=torch.long)
        question_positions_with_padding = torch.tensor([[1, 5, 0]], dtype=torch.long)
        output = model(
            input_ids,
            start_positions=start_positions,
            end_positions=end_positions,
            question_positions=question_positions,
        )
        output_with_padding = model(
            input_ids,
            start_positions=start_positions_with_padding,
            end_positions=end_positions_with_padding,
            question_positions=question_positions_with_padding,
        )

        self.assertAlmostEqual(output.loss.item(), output_with_padding.loss.item(), 4)

        # Note that the original code uses 0 to denote padded question tokens
        # and their start and end positions. As the pad_token_id of the model's
        # config is used for the losse's ignore_index in SplinterForPreTraining,
        # we add this test to ensure anybody making changes to the default
        # value of the config, will be aware of the implication.
        self.assertEqual(model.config.pad_token_id, 0)

    @slow
    def test_splinter_pretraining_prepare_question_positions(self):
        model = SplinterForPreTraining.from_pretrained("tau/splinter-base-qass")

        input_ids = torch.tensor(
            [
                [101, 104, 1, 2, 104, 3, 4, 102],
                [101, 1, 104, 2, 104, 3, 104, 102],
                [101, 1, 2, 104, 104, 3, 4, 102],
                [101, 1, 2, 3, 4, 5, 104, 102],
            ]
        )
        question_positions = torch.tensor([[1, 4, 0], [2, 4, 6], [3, 4, 0], [6, 0, 0]], dtype=torch.long)
        output_without_positions = model(input_ids)
        output_with_positions = model(input_ids, question_positions=question_positions)
        self.assertTrue((output_without_positions.start_logits == output_with_positions.start_logits).all())
        self.assertTrue((output_without_positions.end_logits == output_with_positions.end_logits).all())
