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
from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        MODEL_FOR_PRETRAINING_MAPPING,
        BigBirdConfig,
        BigBirdForCausalLM,
        BigBirdForMaskedLM,
        BigBirdForMultipleChoice,
        BigBirdForQuestionAnswering,
        BigBirdForSequenceClassification,
        BigBirdForTokenClassification,
        BigBirdModel,
        BigBirdForPreTraining,
    )
    from transformers.models.big_bird.modeling_big_bird import (
        BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


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
            num_hidden_layers=5,
            num_attention_heads=4,
            intermediate_size=37,
            hidden_act="gelu_fast",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=4096,
            type_vocab_size=16,
            type_sequence_label_size=2,
            initializer_range=0.02,
            num_labels=3,
            num_choices=4,
            norm_type="postnorm",
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

        self.norm_type = norm_type
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

        config = BigBirdConfig(
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
            norm_type=self.norm_type,
            block_size=self.block_size,
            num_random_blocks=self.num_rand_blocks,
            position_embedding_type=self.position_embedding_type,
        )

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

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

@require_torch
class BigBirdModelTest(ModelTesterMixin, unittest.TestCase):

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
            if model_class in MODEL_FOR_PRETRAINING_MAPPING.values():
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

    # def test_model_various_embeddings(self):
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs()
    #     for type in ["absolute", "relative_key", "relative_key_query"]:
    #         config_and_inputs[0].position_embedding_type = type
    #         self.model_tester.create_and_check_model(*config_and_inputs)

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

    @slow
    def test_model_from_pretrained(self):
        for model_name in BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = BigBirdForPreTraining.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_retain_grad_hidden_states_attentions(self):
        # bigbird cannot keep gradients in attentions when `attention_type=block_sparse`

        if self.model_tester.attention_type == "original_full":
            super().test_retain_grad_hidden_states_attentions()


@require_torch
class BigBirdModelIntegrationTest(unittest.TestCase):

    def _get_dummy_input_ids(self):
        return torch.tensor([[  6, 117,  33,  36,  70,  22,  63,  31,  71,  72,  88,  58, 109,  49,
          48, 116,  92,   6,  19,  95, 118, 100,  80, 111,  93,   2,  31,  84,
          26,   5,   6,  82,  46,  96, 109,   4,  39,  19, 109,  13,  92,  31,
          36,  90, 111,  18,  75,   6,  56,  74,  16,  42,  56,  92,  69, 108,
         127,  81,  82,  41, 106,  19,  44,  24,  82, 121, 120,  65,  36,  26,
          72,  13,  36,  98,  43,  64,   8,  53, 100,  92,  51, 122,  66,  17,
          61,  50, 104, 127,  26,  35,  94,  23, 110,  71,  80,  67, 109, 111,
          44,  19,  51,  41,  86,  71,  76,  44,  18,  68,  44,  77, 107,  81,
          98, 126, 100,   2,  49,  98,  84,  39,  23,  98,  52,  46,  10,  82,
         121,  73]], dtype=torch.long, device=torch_device)

    @slow
    def test_inference_block_sparse_pretraining(self):
        model = BigBirdForPreTraining.from_pretrained("google/bigbird-base", attention_type="block_sparse")
        model.to(torch_device)

        input_ids = torch.tensor([[20920, 232, 328, 1437] * 1024], dtype=torch.long, device=torch_device)
        outputs = model(input_ids)
        prediction_logits = outputs.prediction_logits
        seq_relationship_logits = outputs.seq_relationship_logits

        self.assertEqual(prediction_logits.shape, torch.Size((1, 4096, 50358)))
        self.assertEqual(seq_relationship_logits.shape, torch.Size((1, 2)))

        expected_prediction_logits_slice = torch.tensor(
                                    [[-0.2420, -0.6048, -0.0614,  7.8422],
                                    [-0.0596, -0.0104, -1.8408,  9.3352],
                                    [ 1.0588,  0.7999,  5.0770,  8.7555],
                                    [-0.1385, -1.7199, -1.7613,  6.1094]],
                                    device=torch_device)
        self.assertTrue(torch.allclose(prediction_logits[0, 128:132, 128:132], expected_prediction_logits_slice, atol=1e-4))

        expected_seq_relationship_logits = torch.tensor([[58.8196, 56.3629]], device=torch_device)
        self.assertTrue(torch.allclose(seq_relationship_logits, expected_seq_relationship_logits, atol=1e-4))

    @slow
    def test_inference_full_pretraining(self):
        model = BigBirdForPreTraining.from_pretrained("google/bigbird-base", attention_type="original_full")
        model.to(torch_device)

        input_ids = torch.tensor([[20920, 232, 328, 1437] * 512], dtype=torch.long, device=torch_device)
        outputs = model(input_ids)
        prediction_logits = outputs.prediction_logits
        seq_relationship_logits = outputs.seq_relationship_logits

        self.assertEqual(prediction_logits.shape, torch.Size((1, 512*4, 50358)))
        self.assertEqual(seq_relationship_logits.shape, torch.Size((1, 2)))

        expected_prediction_logits_slice = torch.tensor(
                                    [[ 0.1499, -1.1217,  0.1990,  8.4499],
                                    [-2.7757, -3.0687, -4.8577,  7.5156],
                                    [ 1.5446,  0.1982,  4.3016, 10.4281],
                                    [-1.3705, -4.0130, -3.9629,  5.1526]],
                                    device=torch_device)
        self.assertTrue(torch.allclose(prediction_logits[0, 128:132, 128:132], expected_prediction_logits_slice, atol=1e-4))

        expected_seq_relationship_logits = torch.tensor([[41.4503, 41.2406]], device=torch_device)
        self.assertTrue(torch.allclose(seq_relationship_logits, expected_seq_relationship_logits, atol=1e-4))

    def test_block_sparse_attention_probs(self):
        """
        Asserting if outputted attention matrix is similar to hard coded attention matrix
        """

        model = BigBirdModel.from_pretrained(
            "google/bigbird-base", attention_type="block_sparse", 
            num_random_blocks=3, block_size=16)
        model.to(torch_device)
        model.eval()
        config = model.config

        input_ids = self._get_dummy_input_ids()

        hidden_states = model.embeddings(input_ids)

        batch_size, seqlen, _ = hidden_states.size()
        attn_mask = torch.ones(batch_size, seqlen, device=torch_device, dtype=torch.float)
        to_seq_length = from_seq_length = seqlen
        from_block_size = to_block_size = config.block_size

        blocked_mask, band_mask, from_mask, to_mask = model.create_masks_for_block_sparse_attn(attn_mask, config.block_size)
        from_blocked_mask = to_blocked_mask = blocked_mask

        for i in range(config.num_hidden_layers):
            pointer = model.encoder.layer[i].attention.self

            query_layer = pointer.transpose_for_scores(pointer.query(hidden_states))
            key_layer = pointer.transpose_for_scores(pointer.key(hidden_states))
            value_layer = pointer.transpose_for_scores(pointer.value(hidden_states))

            context_layer, attention_probs = pointer.bigbird_block_sparse_attention(
                                                    query_layer, key_layer, value_layer, band_mask,
                                                    from_mask, to_mask, from_blocked_mask, to_blocked_mask,
                                                    pointer.num_attention_heads, pointer.num_random_blocks, pointer.attention_head_size,
                                                    from_block_size, to_block_size, batch_size, from_seq_length,
                                                    to_seq_length, seed=pointer.seed, plan_from_length=None,
                                                    plan_num_rand_blocks=None, output_attentions=True
                                                    )

            context_layer = context_layer.contiguous().view(batch_size, from_seq_length, -1)
            cl = torch.einsum("bhqk,bhkd->bhqd", attention_probs, value_layer)
            cl = cl.view(context_layer.size())

            self.assertTrue(torch.allclose(context_layer, cl, atol=0.001))

    def test_block_sparse_context_layer(self):
        model = BigBirdModel.from_pretrained(
            "google/bigbird-base", attention_type="block_sparse", 
            num_random_blocks=3, block_size=16
            )
        model.to(torch_device)
        model.eval()
        config = model.config

        input_ids = self._get_dummy_input_ids()
        dummy_hidden_states = model.embeddings(input_ids)

        attn_mask = torch.ones_like(input_ids, device=torch_device)
        blocked_mask, band_mask, from_mask, to_mask = model.create_masks_for_block_sparse_attn(attn_mask, config.block_size)
        targeted_cl = torch.tensor([[ 0.0044,  0.0275,  0.0695, -0.0201,  0.0178,  0.0692, -0.0542, -0.0154,
            -0.0023,  0.0008],
            [ 0.0042,  0.0262,  0.0695, -0.0199,  0.0182,  0.0681, -0.0545, -0.0155,
            -0.0023,  0.0009],
            [ 0.0042,  0.0261,  0.0695, -0.0199,  0.0182,  0.0681, -0.0545, -0.0155,
            -0.0022,  0.0010],
            [ 0.0044,  0.0278,  0.0696, -0.0201,  0.0182,  0.0696, -0.0540, -0.0151,
            -0.0024,  0.0013],
            [ 0.0044,  0.0275,  0.0695, -0.0202,  0.0176,  0.0693, -0.0541, -0.0154,
            -0.0023,  0.0006],
            [ 0.0044,  0.0275,  0.0695, -0.0201,  0.0177,  0.0692, -0.0541, -0.0155,
            -0.0023,  0.0007],
            [ 0.0044,  0.0274,  0.0695, -0.0201,  0.0178,  0.0691, -0.0541, -0.0155,
            -0.0023,  0.0008],
            [ 0.0044,  0.0275,  0.0695, -0.0201,  0.0177,  0.0693, -0.0541, -0.0154,
            -0.0023,  0.0008],
            [ 0.0044,  0.0275,  0.0695, -0.0202,  0.0176,  0.0693, -0.0541, -0.0154,
            -0.0023,  0.0007],
            [ 0.0043,  0.0274,  0.0695, -0.0201,  0.0178,  0.0692, -0.0542, -0.0155,
            -0.0023,  0.0008],
            [ 0.0044,  0.0275,  0.0695, -0.0201,  0.0177,  0.0694, -0.0541, -0.0154,
            -0.0023,  0.0007],
            [ 0.0044,  0.0274,  0.0695, -0.0201,  0.0177,  0.0692, -0.0542, -0.0155,
            -0.0023,  0.0008],
            [ 0.0044,  0.0275,  0.0695, -0.0201,  0.0178,  0.0693, -0.0542, -0.0154,
            -0.0022,  0.0008],
            [ 0.0044,  0.0275,  0.0695, -0.0201,  0.0177,  0.0693, -0.0541, -0.0154,
            -0.0023,  0.0007]],
            device=torch_device
        )

        context_layer = model.encoder.layer[0].attention.self(
                                dummy_hidden_states, band_mask=band_mask, from_mask=from_mask, 
                                to_mask=to_mask, from_blocked_mask=blocked_mask, 
                                to_blocked_mask=blocked_mask
                                )
        context_layer = context_layer[0]

        self.assertEqual(context_layer.shape, torch.Size((1, 128, 768)))
        self.assertTrue(torch.allclose(context_layer[0, 64:78, 300:310], targeted_cl, atol=0.0001))
