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
""" Testing suite for the PyTorch BigBird model. """


import unittest

from tests.test_modeling_common import floats_tensor
from transformers import BigBirdConfig, is_torch_available
from transformers.models.auto import get_values
from transformers.models.big_bird.tokenization_big_bird import BigBirdTokenizer
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        MODEL_FOR_PRETRAINING_MAPPING,
        BigBirdForCausalLM,
        BigBirdForMaskedLM,
        BigBirdForMultipleChoice,
        BigBirdForPreTraining,
        BigBirdForQuestionAnswering,
        BigBirdForSequenceClassification,
        BigBirdForTokenClassification,
        BigBirdModel,
    )
    from transformers.models.big_bird.modeling_big_bird import BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST


class BigBirdModelTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        seq_length=128,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu_fast",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=256,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        attention_type="block_sparse",
        use_bias=True,
        rescale_embeddings=False,
        block_size=16,
        num_rand_blocks=3,
        position_embedding_type="absolute",
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

        self.attention_type = attention_type
        self.use_bias = use_bias
        self.rescale_embeddings = rescale_embeddings
        self.block_size = block_size
        self.num_rand_blocks = num_rand_blocks
        self.position_embedding_type = position_embedding_type

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return BigBirdConfig(
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
            is_encoder_decoder=False,
            initializer_range=self.initializer_range,
            attention_type=self.attention_type,
            use_bias=self.use_bias,
            rescale_embeddings=self.rescale_embeddings,
            block_size=self.block_size,
            num_random_blocks=self.num_rand_blocks,
            position_embedding_type=self.position_embedding_type,
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        config.is_decoder = True
        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = BigBirdModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_pretraining(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = BigBirdForPreTraining(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
            next_sentence_label=sequence_labels,
        )
        self.parent.assertEqual(result.prediction_logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertEqual(result.seq_relationship_logits.shape, (self.batch_size, config.num_labels))

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = BigBirdModel(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = BigBirdForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = BigBirdForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = BigBirdForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
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

    def create_and_check_for_question_answering(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = BigBirdForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = BigBirdForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = BigBirdForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_multiple_choice(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = BigBirdForMultipleChoice(config=config)
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
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict

    def create_and_check_for_auto_padding(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = BigBirdModel(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_change_to_full_attn(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ):
        model = BigBirdModel(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        # the config should not be changed
        self.parent.assertTrue(model.config.attention_type == "block_sparse")


@require_torch
class BigBirdModelTest(ModelTesterMixin, unittest.TestCase):

    # head masking & pruning is currently not supported for big bird
    test_head_masking = False
    test_pruning = False
    test_sequence_classification_problem_types = True

    # torchscript should be possible, but takes prohibitively long to test.
    # Also torchscript is not an important feature to have in the beginning.
    test_torchscript = False

    all_model_classes = (
        (
            BigBirdModel,
            BigBirdForPreTraining,
            BigBirdForMaskedLM,
            BigBirdForCausalLM,
            BigBirdForMultipleChoice,
            BigBirdForQuestionAnswering,
            BigBirdForSequenceClassification,
            BigBirdForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (BigBirdForCausalLM,) if is_torch_available() else ()

    # special case for ForPreTraining model
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class in get_values(MODEL_FOR_PRETRAINING_MAPPING):
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
                inputs_dict["next_sentence_label"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
        return inputs_dict

    def setUp(self):
        self.model_tester = BigBirdModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BigBirdConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_model_as_decoder(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_model_as_decoder(*config_and_inputs)

    def test_model_as_decoder_with_default_input_mask(self):
        # This regression test was failing with PyTorch < 1.3
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = self.model_tester.prepare_config_and_inputs_for_decoder()

        input_mask = None

        self.model_tester.create_and_check_model_as_decoder(
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def test_retain_grad_hidden_states_attentions(self):
        # bigbird cannot keep gradients in attentions when `attention_type=block_sparse`

        if self.model_tester.attention_type == "original_full":
            super().test_retain_grad_hidden_states_attentions()

    @slow
    def test_model_from_pretrained(self):
        for model_name in BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = BigBirdForPreTraining.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_model_various_attn_type(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["original_full", "block_sparse"]:
            config_and_inputs[0].attention_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_fast_integration(self):
        # fmt: off
        input_ids = torch.tensor(
            [[6, 117, 33, 36, 70, 22, 63, 31, 71, 72, 88, 58, 109, 49, 48, 116, 92, 6, 19, 95, 118, 100, 80, 111, 93, 2, 31, 84, 26, 5, 6, 82, 46, 96, 109, 4, 39, 19, 109, 13, 92, 31, 36, 90, 111, 18, 75, 6, 56, 74, 16, 42, 56, 92, 69, 108, 127, 81, 82, 41, 106, 19, 44, 24, 82, 121, 120, 65, 36, 26, 72, 13, 36, 98, 43, 64, 8, 53, 100, 92, 51, 122, 66, 17, 61, 50, 104, 127, 26, 35, 94, 23, 110, 71, 80, 67, 109, 111, 44, 19, 51, 41, 86, 71, 76, 44, 18, 68, 44, 77, 107, 81, 98, 126, 100, 2, 49, 98, 84, 39, 23, 98, 52, 46, 10, 82, 121, 73],[6, 117, 33, 36, 70, 22, 63, 31, 71, 72, 88, 58, 109, 49, 48, 116, 92, 6, 19, 95, 118, 100, 80, 111, 93, 2, 31, 84, 26, 5, 6, 82, 46, 96, 109, 4, 39, 19, 109, 13, 92, 31, 36, 90, 111, 18, 75, 6, 56, 74, 16, 42, 56, 92, 69, 108, 127, 81, 82, 41, 106, 19, 44, 24, 82, 121, 120, 65, 36, 26, 72, 13, 36, 98, 43, 64, 8, 53, 100, 92, 51, 12, 66, 17, 61, 50, 104, 127, 26, 35, 94, 23, 110, 71, 80, 67, 109, 111, 44, 19, 51, 41, 86, 71, 76, 28, 18, 68, 44, 77, 107, 81, 98, 126, 100, 2, 49, 18, 84, 39, 23, 98, 52, 46, 10, 82, 121, 73]],  # noqa: E231
            dtype=torch.long,
            device=torch_device,
        )
        # fmt: on
        input_ids = input_ids % self.model_tester.vocab_size
        input_ids[1] = input_ids[1] - 1

        attention_mask = torch.ones((input_ids.shape), device=torch_device)
        attention_mask[:, :-10] = 0

        config, _, _, _, _, _, _ = self.model_tester.prepare_config_and_inputs()
        torch.manual_seed(0)
        model = BigBirdModel(config).eval().to(torch_device)

        with torch.no_grad():
            hidden_states = model(input_ids, attention_mask=attention_mask).last_hidden_state
            self.assertTrue(
                torch.allclose(
                    hidden_states[0, 0, :5],
                    torch.tensor([1.4943, 0.0928, 0.8254, -0.2816, -0.9788], device=torch_device),
                    atol=1e-3,
                )
            )

    def test_auto_padding(self):
        self.model_tester.seq_length = 241
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_auto_padding(*config_and_inputs)

    def test_for_change_to_full_attn(self):
        self.model_tester.seq_length = 9
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_change_to_full_attn(*config_and_inputs)


@require_torch
@slow
class BigBirdModelIntegrationTest(unittest.TestCase):
    # we can have this true once block_sparse attn_probs works accurately
    test_attention_probs = False

    def _get_dummy_input_ids(self):
        # fmt: off
        ids = torch.tensor(
            [[6, 117, 33, 36, 70, 22, 63, 31, 71, 72, 88, 58, 109, 49, 48, 116, 92, 6, 19, 95, 118, 100, 80, 111, 93, 2, 31, 84, 26, 5, 6, 82, 46, 96, 109, 4, 39, 19, 109, 13, 92, 31, 36, 90, 111, 18, 75, 6, 56, 74, 16, 42, 56, 92, 69, 108, 127, 81, 82, 41, 106, 19, 44, 24, 82, 121, 120, 65, 36, 26, 72, 13, 36, 98, 43, 64, 8, 53, 100, 92, 51, 122, 66, 17, 61, 50, 104, 127, 26, 35, 94, 23, 110, 71, 80, 67, 109, 111, 44, 19, 51, 41, 86, 71, 76, 44, 18, 68, 44, 77, 107, 81, 98, 126, 100, 2, 49, 98, 84, 39, 23, 98, 52, 46, 10, 82, 121, 73]],  # noqa: E231
            dtype=torch.long,
            device=torch_device,
        )
        # fmt: on
        return ids

    def test_inference_block_sparse_pretraining(self):
        model = BigBirdForPreTraining.from_pretrained("google/bigbird-roberta-base", attention_type="block_sparse")
        model.to(torch_device)

        input_ids = torch.tensor([[20920, 232, 328, 1437] * 1024], dtype=torch.long, device=torch_device)
        outputs = model(input_ids)
        prediction_logits = outputs.prediction_logits
        seq_relationship_logits = outputs.seq_relationship_logits

        self.assertEqual(prediction_logits.shape, torch.Size((1, 4096, 50358)))
        self.assertEqual(seq_relationship_logits.shape, torch.Size((1, 2)))

        expected_prediction_logits_slice = torch.tensor(
            [
                [-0.2420, -0.6048, -0.0614, 7.8422],
                [-0.0596, -0.0104, -1.8408, 9.3352],
                [1.0588, 0.7999, 5.0770, 8.7555],
                [-0.1385, -1.7199, -1.7613, 6.1094],
            ],
            device=torch_device,
        )
        self.assertTrue(
            torch.allclose(prediction_logits[0, 128:132, 128:132], expected_prediction_logits_slice, atol=1e-4)
        )

        expected_seq_relationship_logits = torch.tensor([[58.8196, 56.3629]], device=torch_device)
        self.assertTrue(torch.allclose(seq_relationship_logits, expected_seq_relationship_logits, atol=1e-4))

    def test_inference_full_pretraining(self):
        model = BigBirdForPreTraining.from_pretrained("google/bigbird-roberta-base", attention_type="original_full")
        model.to(torch_device)

        input_ids = torch.tensor([[20920, 232, 328, 1437] * 512], dtype=torch.long, device=torch_device)
        outputs = model(input_ids)
        prediction_logits = outputs.prediction_logits
        seq_relationship_logits = outputs.seq_relationship_logits

        self.assertEqual(prediction_logits.shape, torch.Size((1, 512 * 4, 50358)))
        self.assertEqual(seq_relationship_logits.shape, torch.Size((1, 2)))

        expected_prediction_logits_slice = torch.tensor(
            [
                [0.1499, -1.1217, 0.1990, 8.4499],
                [-2.7757, -3.0687, -4.8577, 7.5156],
                [1.5446, 0.1982, 4.3016, 10.4281],
                [-1.3705, -4.0130, -3.9629, 5.1526],
            ],
            device=torch_device,
        )
        self.assertTrue(
            torch.allclose(prediction_logits[0, 128:132, 128:132], expected_prediction_logits_slice, atol=1e-4)
        )

        expected_seq_relationship_logits = torch.tensor([[41.4503, 41.2406]], device=torch_device)
        self.assertTrue(torch.allclose(seq_relationship_logits, expected_seq_relationship_logits, atol=1e-4))

    def test_block_sparse_attention_probs(self):
        """
        Asserting if outputted attention matrix is similar to hard coded attention matrix
        """

        if not self.test_attention_probs:
            return

        model = BigBirdModel.from_pretrained(
            "google/bigbird-roberta-base", attention_type="block_sparse", num_random_blocks=3, block_size=16
        )
        model.to(torch_device)
        model.eval()
        config = model.config

        input_ids = self._get_dummy_input_ids()

        hidden_states = model.embeddings(input_ids)

        batch_size, seqlen, _ = hidden_states.size()
        attn_mask = torch.ones(batch_size, seqlen, device=torch_device, dtype=torch.float)
        to_seq_length = from_seq_length = seqlen
        from_block_size = to_block_size = config.block_size

        blocked_mask, band_mask, from_mask, to_mask = model.create_masks_for_block_sparse_attn(
            attn_mask, config.block_size
        )
        from_blocked_mask = to_blocked_mask = blocked_mask

        for i in range(config.num_hidden_layers):
            pointer = model.encoder.layer[i].attention.self

            query_layer = pointer.transpose_for_scores(pointer.query(hidden_states))
            key_layer = pointer.transpose_for_scores(pointer.key(hidden_states))
            value_layer = pointer.transpose_for_scores(pointer.value(hidden_states))

            context_layer, attention_probs = pointer.bigbird_block_sparse_attention(
                query_layer,
                key_layer,
                value_layer,
                band_mask,
                from_mask,
                to_mask,
                from_blocked_mask,
                to_blocked_mask,
                pointer.num_attention_heads,
                pointer.num_random_blocks,
                pointer.attention_head_size,
                from_block_size,
                to_block_size,
                batch_size,
                from_seq_length,
                to_seq_length,
                seed=pointer.seed,
                plan_from_length=None,
                plan_num_rand_blocks=None,
                output_attentions=True,
            )

            context_layer = context_layer.contiguous().view(batch_size, from_seq_length, -1)
            cl = torch.einsum("bhqk,bhkd->bhqd", attention_probs, value_layer)
            cl = cl.view(context_layer.size())

            self.assertTrue(torch.allclose(context_layer, cl, atol=0.001))

    def test_block_sparse_context_layer(self):
        model = BigBirdModel.from_pretrained(
            "google/bigbird-roberta-base", attention_type="block_sparse", num_random_blocks=3, block_size=16
        )
        model.to(torch_device)
        model.eval()
        config = model.config

        input_ids = self._get_dummy_input_ids()
        dummy_hidden_states = model.embeddings(input_ids)

        attn_mask = torch.ones_like(input_ids, device=torch_device)
        blocked_mask, band_mask, from_mask, to_mask = model.create_masks_for_block_sparse_attn(
            attn_mask, config.block_size
        )
        targeted_cl = torch.tensor(
            [
                [0.1874, 1.5260, 0.2335, -0.0473, -0.0961, 1.8384, -0.0141, 0.1250, 0.0085, -0.0048],
                [-0.0554, 0.0728, 0.1683, -0.1332, 0.1741, 0.1337, -0.2380, -0.1849, -0.0390, -0.0259],
                [-0.0419, 0.0767, 0.1591, -0.1399, 0.1789, 0.1257, -0.2406, -0.1772, -0.0261, -0.0079],
                [0.1860, 1.5172, 0.2326, -0.0473, -0.0953, 1.8291, -0.0147, 0.1245, 0.0082, -0.0046],
                [0.1879, 1.5296, 0.2335, -0.0471, -0.0975, 1.8433, -0.0136, 0.1260, 0.0086, -0.0054],
                [0.1854, 1.5147, 0.2334, -0.0480, -0.0956, 1.8250, -0.0149, 0.1222, 0.0082, -0.0060],
                [0.1859, 1.5184, 0.2334, -0.0474, -0.0955, 1.8297, -0.0143, 0.1234, 0.0079, -0.0054],
                [0.1885, 1.5336, 0.2335, -0.0467, -0.0979, 1.8481, -0.0130, 0.1269, 0.0085, -0.0049],
                [0.1881, 1.5305, 0.2335, -0.0471, -0.0976, 1.8445, -0.0135, 0.1262, 0.0086, -0.0053],
                [0.1852, 1.5148, 0.2333, -0.0480, -0.0949, 1.8254, -0.0151, 0.1225, 0.0079, -0.0055],
                [0.1877, 1.5292, 0.2335, -0.0470, -0.0972, 1.8431, -0.0135, 0.1259, 0.0084, -0.0052],
                [0.1874, 1.5261, 0.2334, -0.0472, -0.0968, 1.8393, -0.0140, 0.1251, 0.0084, -0.0052],
                [0.1853, 1.5151, 0.2331, -0.0478, -0.0948, 1.8256, -0.0154, 0.1228, 0.0086, -0.0052],
                [0.1867, 1.5233, 0.2334, -0.0475, -0.0965, 1.8361, -0.0139, 0.1247, 0.0084, -0.0054],
            ],
            device=torch_device,
        )

        context_layer = model.encoder.layer[0].attention.self(
            dummy_hidden_states,
            band_mask=band_mask,
            from_mask=from_mask,
            to_mask=to_mask,
            from_blocked_mask=blocked_mask,
            to_blocked_mask=blocked_mask,
        )
        context_layer = context_layer[0]

        self.assertEqual(context_layer.shape, torch.Size((1, 128, 768)))
        self.assertTrue(torch.allclose(context_layer[0, 64:78, 300:310], targeted_cl, atol=0.0001))

    def test_tokenizer_inference(self):
        tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        model = BigBirdModel.from_pretrained(
            "google/bigbird-roberta-base", attention_type="block_sparse", num_random_blocks=3, block_size=16
        )
        model.to(torch_device)

        text = [
            "Transformer-based models are unable to process long sequences due to their self-attention operation, which scales quadratically with the sequence length. To address this limitation, we introduce the Longformer with an attention mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or longer. Longformer’s attention mechanism is a drop-in replacement for the standard self-attention and combines a local windowed attention with a task motivated global attention. Following prior work on long-sequence transformers, we evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks. Our pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on WikiHop and TriviaQA."
        ]
        inputs = tokenizer(text)

        for k in inputs:
            inputs[k] = torch.tensor(inputs[k], device=torch_device, dtype=torch.long)

        prediction = model(**inputs)
        prediction = prediction[0]

        self.assertEqual(prediction.shape, torch.Size((1, 199, 768)))

        expected_prediction = torch.tensor(
            [
                [-0.0213, -0.2213, -0.0061, 0.0687],
                [0.0977, 0.1858, 0.2374, 0.0483],
                [0.2112, -0.2524, 0.5793, 0.0967],
                [0.2473, -0.5070, -0.0630, 0.2174],
                [0.2885, 0.1139, 0.6071, 0.2991],
                [0.2328, -0.2373, 0.3648, 0.1058],
                [0.2517, -0.0689, 0.0555, 0.0880],
                [0.1021, -0.1495, -0.0635, 0.1891],
                [0.0591, -0.0722, 0.2243, 0.2432],
                [-0.2059, -0.2679, 0.3225, 0.6183],
                [0.2280, -0.2618, 0.1693, 0.0103],
                [0.0183, -0.1375, 0.2284, -0.1707],
            ],
            device=torch_device,
        )
        self.assertTrue(torch.allclose(prediction[0, 52:64, 320:324], expected_prediction, atol=1e-4))

    def test_inference_question_answering(self):
        tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-base-trivia-itc")
        model = BigBirdForQuestionAnswering.from_pretrained(
            "google/bigbird-base-trivia-itc", attention_type="block_sparse", block_size=16, num_random_blocks=3
        )
        model.to(torch_device)

        context = "The BigBird model was proposed in Big Bird: Transformers for Longer Sequences by Zaheer, Manzil and Guruganesh, Guru and Dubey, Kumar Avinava and Ainslie, Joshua and Alberti, Chris and Ontanon, Santiago and Pham, Philip and Ravula, Anirudh and Wang, Qifan and Yang, Li and others. BigBird, is a sparse-attention based transformer which extends Transformer based models, such as BERT to much longer sequences. In addition to sparse attention, BigBird also applies global attention as well as random attention to the input sequence. Theoretically, it has been shown that applying sparse, global, and random attention approximates full attention, while being computationally much more efficient for longer sequences. As a consequence of the capability to handle longer context, BigBird has shown improved performance on various long document NLP tasks, such as question answering and summarization, compared to BERT or RoBERTa."

        question = [
            "Which is better for longer sequences- BigBird or BERT?",
            "What is the benefit of using BigBird over BERT?",
        ]
        inputs = tokenizer(
            question,
            [context, context],
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=256,
            truncation=True,
        )

        inputs = {k: v.to(torch_device) for k, v in inputs.items()}

        start_logits, end_logits = model(**inputs).to_tuple()

        # fmt: off
        target_start_logits = torch.tensor(
            [[-8.9304, -10.3849, -14.4997, -9.6497, -13.9469, -7.8134, -8.9687, -13.3585, -9.7987, -13.8869, -9.2632, -8.9294, -13.6721, -7.3198, -9.5434, -11.2641, -14.3245, -9.5705, -12.7367, -8.6168, -11.083, -13.7573, -8.1151, -14.5329, -7.6876, -15.706, -12.8558, -9.1135, 8.0909, -3.1925, -11.5812, -9.4822], [-11.5595, -14.5591, -10.2978, -14.8445, -10.2092, -11.1899, -13.8356, -10.5644, -14.7706, -9.9841, -11.0052, -14.1862, -8.8173, -11.1098, -12.4686, -15.0531, -11.0196, -13.6614, -10.0236, -11.8151, -14.8744, -9.5123, -15.1605, -8.6472, -15.4184, -8.898, -9.6328, -7.0258, -11.3365, -14.4065, -10.2587, -8.9103]],  # noqa: E231
            device=torch_device,
        )
        target_end_logits = torch.tensor(
            [[-12.4131, -8.5959, -15.7163, -11.1524, -15.9913, -12.2038, -7.8902, -16.0296, -12.164, -16.5017, -13.3332, -6.9488, -15.7756, -13.8506, -11.0779, -9.2893, -15.0426, -10.1963, -17.3292, -12.2945, -11.5337, -16.4514, -9.1564, -17.5001, -9.1562, -16.2971, -13.3199, -7.5724, -5.1175, 7.2168, -10.3804, -11.9873], [-10.8654, -14.9967, -11.4144, -16.9189, -14.2673, -9.7068, -15.0182, -12.8846, -16.8716, -13.665, -10.3113, -15.1436, -14.9069, -13.3364, -11.2339, -16.0118, -11.8331, -17.0613, -13.8852, -12.4163, -16.8978, -10.7772, -17.2324, -10.6979, -16.9811, -10.3427, -9.497, -13.7104, -11.1107, -13.2936, -13.855, -14.1264]],  # noqa: E231
            device=torch_device,
        )
        # fmt: on

        self.assertTrue(torch.allclose(start_logits[:, 64:96], target_start_logits, atol=1e-4))
        self.assertTrue(torch.allclose(end_logits[:, 64:96], target_end_logits, atol=1e-4))

        input_ids = inputs["input_ids"].tolist()
        answer = [
            input_ids[i][torch.argmax(start_logits, dim=-1)[i] : torch.argmax(end_logits, dim=-1)[i] + 1]
            for i in range(len(input_ids))
        ]
        answer = tokenizer.batch_decode(answer)

        self.assertTrue(answer == ["BigBird", "global attention"])

    def test_fill_mask(self):
        tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        model = BigBirdForMaskedLM.from_pretrained("google/bigbird-roberta-base")
        model.to(torch_device)

        input_ids = tokenizer("The goal of life is [MASK] .", return_tensors="pt").input_ids.to(torch_device)
        logits = model(input_ids).logits

        # [MASK] is token at 6th position
        pred_token = tokenizer.decode(torch.argmax(logits[0, 6:7], axis=-1))
        self.assertEqual(pred_token, "happiness")

    def test_auto_padding(self):
        model = BigBirdModel.from_pretrained(
            "google/bigbird-roberta-base", attention_type="block_sparse", num_random_blocks=3, block_size=16
        )
        model.to(torch_device)
        model.eval()

        input_ids = torch.tensor([200 * [10] + 40 * [2] + [1]], device=torch_device, dtype=torch.long)
        output = model(input_ids).to_tuple()[0]

        # fmt: off
        target = torch.tensor(
            [[-0.045136, -0.068013, 0.12246, -0.01356, 0.018386, 0.025333, -0.0044439, -0.0030996, -0.064031, 0.0006439], [-0.045018, -0.067638, 0.12317, -0.013998, 0.019216, 0.025695, -0.0043705, -0.0031895, -0.063153, 0.00088899], [-0.045042, -0.067305, 0.1234, -0.014512, 0.020057, 0.026084, -0.004615, -0.0031728, -0.062442, 0.0010263], [-0.044589, -0.067655, 0.12416, -0.014287, 0.019416, 0.026065, -0.0050958, -0.002702, -0.063158, 0.0004827], [-0.044627, -0.067535, 0.1239, -0.014319, 0.019491, 0.026213, -0.0059482, -0.0025906, -0.063116, 0.00014669], [-0.044899, -0.067704, 0.12337, -0.014231, 0.019256, 0.026345, -0.0065565, -0.0022938, -0.063433, -0.00011409], [-0.045599, -0.067764, 0.12235, -0.014151, 0.019206, 0.026417, -0.0068965, -0.0024494, -0.063313, -4.4499e-06], [-0.045557, -0.068372, 0.12199, -0.013747, 0.017962, 0.026103, -0.0070607, -0.0023552, -0.06447, -0.00048756], [-0.045334, -0.068913, 0.1217, -0.013566, 0.01693, 0.025745, -0.006311, -0.0024903, -0.065575, -0.0006719], [-0.045171, -0.068726, 0.12164, -0.013688, 0.017139, 0.025629, -0.005213, -0.0029412, -0.065237, -0.00020669], [-0.044411, -0.069267, 0.12206, -0.013645, 0.016212, 0.025589, -0.0044121, -0.002972, -0.066277, -0.00067963], [-0.043487, -0.069792, 0.1232, -0.013663, 0.015303, 0.02613, -0.0036294, -0.0030616, -0.067483, -0.0012642], [-0.042622, -0.069287, 0.12469, -0.013936, 0.016204, 0.026474, -0.0040534, -0.0027365, -0.066994, -0.0014148], [-0.041879, -0.070031, 0.12593, -0.014047, 0.015082, 0.027751, -0.0040683, -0.0027189, -0.068985, -0.0027146]],  # noqa: E231
            device=torch_device,
        )
        # fmt: on

        self.assertEqual(output.shape, torch.Size((1, 241, 768)))
        self.assertTrue(torch.allclose(output[0, 64:78, 300:310], target, atol=0.0001))
