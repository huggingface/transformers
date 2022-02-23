# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
""" Testing suite for the PyTorch Data2VecAudio model. """

import math
import unittest

import numpy as np
from datasets import load_dataset

from tests.test_modeling_common import floats_tensor, ids_tensor, random_attention_mask
from transformers import Data2VecAudioConfig, Data2VecTextConfig, is_torch_available
from transformers.testing_utils import (
    TestCasePlus,
    is_pt_flax_cross_test,
    require_soundfile,
    require_torch,
    slow,
    torch_device,
)

from .test_configuration_common import ConfigTester
from .test_generation_utils import GenerationTesterMixin
from .test_modeling_common import ModelTesterMixin, _config_zero_init


if is_torch_available():
    import torch

    from transformers import (
        Data2VecAudioForAudioFrameClassification,
        Data2VecAudioForCTC,
        Data2VecAudioForSequenceClassification,
        Data2VecAudioForXVector,
        Data2VecAudioModel,
        Data2VecTextForCausalLM,
        Data2VecTextForMaskedLM,
        Data2VecTextForMultipleChoice,
        Data2VecTextForQuestionAnswering,
        Data2VecTextForSequenceClassification,
        Data2VecTextForTokenClassification,
        Data2VecTextModel,
        Wav2Vec2Processor,
    )
    from transformers.models.data2vec.modeling_data2vec_audio import _compute_mask_indices
    from transformers.models.data2vec.modeling_data2vec_text import (
        DATA2VEC_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST,
        Data2VecTextForTextEmbeddings,
        create_position_ids_from_input_ids,
    )


class Data2VecTextModelTester:
    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_input_mask = True
        self.use_token_type_ids = True
        self.use_labels = True
        self.vocab_size = 99
        self.hidden_size = 32
        self.num_hidden_layers = 5
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
        return Data2VecTextConfig(
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
            initializer_range=self.initializer_range,
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
        model = Data2VecTextModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

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
        model = Data2VecTextModel(config)
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
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

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
        model = Data2VecTextForCausalLM(config=config)
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
        model = Data2VecTextForCausalLM(config=config).to(torch_device).eval()

        # make sure that ids don't start with pad token
        mask = input_ids.ne(config.pad_token_id).long()
        input_ids = input_ids * mask

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

        # make sure that ids don't start with pad token
        mask = next_tokens.ne(config.pad_token_id).long()
        next_tokens = next_tokens * mask
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

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = Data2VecTextForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = Data2VecTextForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_multiple_choice(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = Data2VecTextForMultipleChoice(config=config)
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

    def create_and_check_for_question_answering(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = Data2VecTextForQuestionAnswering(config=config)
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
class Data2VecTextModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Data2VecTextForCausalLM,
            Data2VecTextForMaskedLM,
            Data2VecTextModel,
            Data2VecTextForSequenceClassification,
            Data2VecTextForTokenClassification,
            Data2VecTextForMultipleChoice,
            Data2VecTextForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (Data2VecTextForCausalLM,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = Data2VecTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Data2VecTextConfig, hidden_size=37)

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

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in DATA2VEC_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = Data2VecTextModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_create_position_ids_respects_padding_index(self):
        """Ensure that the default position ids only assign a sequential . This is a regression
        test for https://github.com/huggingface/transformers/issues/1761

        The position ids should be masked with the embedding object's padding index. Therefore, the
        first available non-padding position index is Data2VecTextForTextEmbeddings.padding_idx + 1
        """
        config = self.model_tester.prepare_config_and_inputs()[0]
        model = Data2VecTextForTextEmbeddings(config=config)

        input_ids = torch.as_tensor([[12, 31, 13, model.padding_idx]])
        expected_positions = torch.as_tensor(
            [[0 + model.padding_idx + 1, 1 + model.padding_idx + 1, 2 + model.padding_idx + 1, model.padding_idx]]
        )

        position_ids = create_position_ids_from_input_ids(input_ids, model.padding_idx)
        self.assertEqual(position_ids.shape, expected_positions.shape)
        self.assertTrue(torch.all(torch.eq(position_ids, expected_positions)))

    def test_create_position_ids_from_inputs_embeds(self):
        """Ensure that the default position ids only assign a sequential . This is a regression
        test for https://github.com/huggingface/transformers/issues/1761

        The position ids should be masked with the embedding object's padding index. Therefore, the
        first available non-padding position index is Data2VecTextForTextEmbeddings.padding_idx + 1
        """
        config = self.model_tester.prepare_config_and_inputs()[0]
        embeddings = Data2VecTextForTextEmbeddings(config=config)

        inputs_embeds = torch.empty(2, 4, 30)
        expected_single_positions = [
            0 + embeddings.padding_idx + 1,
            1 + embeddings.padding_idx + 1,
            2 + embeddings.padding_idx + 1,
            3 + embeddings.padding_idx + 1,
        ]
        expected_positions = torch.as_tensor([expected_single_positions, expected_single_positions])
        position_ids = embeddings.create_position_ids_from_inputs_embeds(inputs_embeds)
        self.assertEqual(position_ids.shape, expected_positions.shape)
        self.assertTrue(torch.all(torch.eq(position_ids, expected_positions)))


@require_torch
class Data2VecTextModelIntegrationTest(TestCasePlus):
    @slow
    def test_inference_masked_lm(self):
        model = Data2VecTextForMaskedLM.from_pretrained("facebook/data2vec-text-base")

        input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        with torch.no_grad():
            output = model(input_ids)[0]
        expected_shape = torch.Size((1, 11, 50265))
        self.assertEqual(output.shape, expected_shape)
        # compare the actual values for a slice.
        expected_slice = torch.tensor([[[0.2328, 0.0000, 1.1710], [2.2525, 0.0000, 1.9937], [2.1280, 0.0000, 1.8691]]])

        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-4))

    @slow
    def test_inference_no_head(self):
        model = Data2VecTextModel.from_pretrained("facebook/data2vec-text-base")

        input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        with torch.no_grad():
            output = model(input_ids)[0]
        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [[[0.1998, -0.0379, 0.0024], [-0.0971, -0.2214, -0.1798], [-0.0789, -0.2400, -0.1898]]]
        )

        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-4))


class Data2VecAudioModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=1024,  # speech is longer
        is_training=False,
        hidden_size=16,
        feat_extract_dropout=0.0,
        feat_extract_activation="gelu",
        conv_dim=(32, 32, 32),
        conv_stride=(4, 4, 4),
        conv_kernel=(8, 8, 8),
        conv_bias=False,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        num_hidden_layers=4,
        num_attention_heads=2,
        hidden_dropout_prob=0.1,
        intermediate_size=20,
        layer_norm_eps=1e-5,
        hidden_act="gelu",
        initializer_range=0.02,
        mask_time_prob=0.5,
        mask_time_length=2,
        vocab_size=32,
        num_adapter_layers=1,
        adapter_stride=2,
        tdnn_dim=(32, 32),
        tdnn_kernel=(5, 3),
        tdnn_dilation=(1, 2),
        xvector_output_dim=32,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.feat_extract_dropout = feat_extract_dropout
        self.feat_extract_activation = feat_extract_activation
        self.conv_dim = conv_dim
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.num_adapter_layers = num_adapter_layers
        self.adapter_stride = adapter_stride
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.scope = scope
        self.tdnn_dim = tdnn_dim
        self.tdnn_kernel = tdnn_kernel
        self.tdnn_dilation = tdnn_dilation
        self.xvector_output_dim = xvector_output_dim

        output_seq_length = self.seq_length
        for kernel, stride in zip(self.conv_kernel, self.conv_stride):
            output_seq_length = (output_seq_length - (kernel - 1)) / stride
        self.output_seq_length = int(math.ceil(output_seq_length))
        self.encoder_seq_length = self.output_seq_length

        self.adapter_output_seq_length = (self.output_seq_length - 1) // adapter_stride + 1

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        return config, input_values, attention_mask

    def get_config(self):
        return Data2VecAudioConfig(
            hidden_size=self.hidden_size,
            feat_extract_dropout=self.feat_extract_dropout,
            feat_extract_activation=self.feat_extract_activation,
            conv_dim=self.conv_dim,
            conv_stride=self.conv_stride,
            conv_kernel=self.conv_kernel,
            conv_bias=self.conv_bias,
            mask_time_prob=self.mask_time_prob,
            mask_time_length=self.mask_time_length,
            num_conv_pos_embeddings=self.num_conv_pos_embeddings,
            num_conv_pos_embedding_groups=self.num_conv_pos_embedding_groups,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_dropout_prob=self.hidden_dropout_prob,
            intermediate_size=self.intermediate_size,
            layer_norm_eps=self.layer_norm_eps,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
            vocab_size=self.vocab_size,
            num_adapter_layers=self.num_adapter_layers,
            adapter_stride=self.adapter_stride,
            tdnn_dim=self.tdnn_dim,
            tdnn_kernel=self.tdnn_kernel,
            tdnn_dilation=self.tdnn_dilation,
            xvector_output_dim=self.xvector_output_dim,
        )

    def create_and_check_model(self, config, input_values, attention_mask):
        model = Data2VecAudioModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def create_and_check_model_with_adapter(self, config, input_values, attention_mask):
        config.add_adapter = True
        model = Data2VecAudioModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.adapter_output_seq_length, self.hidden_size)
        )

    def create_and_check_model_with_adapter_proj_dim(self, config, input_values, attention_mask):
        config.add_adapter = True
        config.output_hidden_size = 8
        model = Data2VecAudioModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.adapter_output_seq_length, config.output_hidden_size),
        )

    def create_and_check_batch_inference(self, config, input_values, *args):
        # test does not pass for models making use of `group_norm`
        # check: https://github.com/pytorch/fairseq/issues/3227
        model = Data2VecAudioModel(config=config)
        model.to(torch_device)
        model.eval()

        input_values = input_values[:3]
        attention_mask = torch.ones(input_values.shape, device=torch_device, dtype=torch.bool)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0.0

        batch_outputs = model(input_values, attention_mask=attention_mask).last_hidden_state

        for i in range(input_values.shape[0]):
            input_slice = input_values[i : i + 1, : input_lengths[i]]
            output = model(input_slice).last_hidden_state

            batch_output = batch_outputs[i : i + 1, : output.shape[1]]
            self.parent.assertTrue(torch.allclose(output, batch_output, atol=1e-3))

    def check_ctc_loss(self, config, input_values, *args):
        model = Data2VecAudioForCTC(config=config)
        model.to(torch_device)

        # make sure that dropout is disabled
        model.eval()

        input_values = input_values[:3]
        attention_mask = torch.ones(input_values.shape, device=torch_device, dtype=torch.long)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], min(max_length_labels) - 1), model.config.vocab_size)

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0

        model.config.ctc_loss_reduction = "sum"
        sum_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()

        model.config.ctc_loss_reduction = "mean"
        mean_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()

        self.parent.assertTrue(isinstance(sum_loss, float))
        self.parent.assertTrue(isinstance(mean_loss, float))

    def check_seq_classifier_loss(self, config, input_values, *args):
        model = Data2VecAudioForSequenceClassification(config=config)
        model.to(torch_device)

        # make sure that dropout is disabled
        model.eval()

        input_values = input_values[:3]
        attention_mask = torch.ones(input_values.shape, device=torch_device, dtype=torch.long)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        labels = ids_tensor((input_values.shape[0], 1), len(model.config.id2label))

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0

        masked_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()
        unmasked_loss = model(input_values, labels=labels).loss.item()

        self.parent.assertTrue(isinstance(masked_loss, float))
        self.parent.assertTrue(isinstance(unmasked_loss, float))
        self.parent.assertTrue(masked_loss != unmasked_loss)

    def check_ctc_training(self, config, input_values, *args):
        config.ctc_zero_infinity = True
        model = Data2VecAudioForCTC(config=config)
        model.to(torch_device)
        model.train()

        # freeze feature encoder
        model.freeze_feature_encoder()

        input_values = input_values[:3]

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], max(max_length_labels) - 2), model.config.vocab_size)

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0

            if max_length_labels[i] < labels.shape[-1]:
                # it's important that we make sure that target lenghts are at least
                # one shorter than logit lenghts to prevent -inf
                labels[i, max_length_labels[i] - 1 :] = -100

        loss = model(input_values, labels=labels).loss
        self.parent.assertFalse(torch.isinf(loss).item())

        loss.backward()

    def check_seq_classifier_training(self, config, input_values, *args):
        config.ctc_zero_infinity = True
        model = Data2VecAudioForSequenceClassification(config=config)
        model.to(torch_device)
        model.train()

        # freeze everything but the classification head
        model.freeze_base_model()

        input_values = input_values[:3]

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        labels = ids_tensor((input_values.shape[0], 1), len(model.config.id2label))

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0

        loss = model(input_values, labels=labels).loss
        self.parent.assertFalse(torch.isinf(loss).item())

        loss.backward()

    def check_xvector_training(self, config, input_values, *args):
        config.ctc_zero_infinity = True
        model = Data2VecAudioForXVector(config=config)
        model.to(torch_device)
        model.train()

        # freeze everything but the classification head
        model.freeze_base_model()

        input_values = input_values[:3]

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        labels = ids_tensor((input_values.shape[0], 1), len(model.config.id2label))

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0

        loss = model(input_values, labels=labels).loss
        self.parent.assertFalse(torch.isinf(loss).item())

        loss.backward()

    def check_labels_out_of_vocab(self, config, input_values, *args):
        model = Data2VecAudioForCTC(config)
        model.to(torch_device)
        model.train()

        input_values = input_values[:3]

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(torch.tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], max(max_length_labels) - 2), model.config.vocab_size + 100)

        with self.parent.assertRaises(ValueError):
            model(input_values, labels=labels)

    def prepare_config_and_inputs_for_common(self):
        config, input_values, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_values": input_values, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class Data2VecAudioModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Data2VecAudioForCTC,
            Data2VecAudioModel,
            Data2VecAudioForSequenceClassification,
            Data2VecAudioForAudioFrameClassification,
            Data2VecAudioForXVector,
        )
        if is_torch_available()
        else ()
    )
    test_pruning = False
    test_headmasking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = Data2VecAudioModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Data2VecAudioConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_adapter(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_adapter(*config_and_inputs)

    def test_model_with_adapter_proj_dim(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_adapter_proj_dim(*config_and_inputs)

    def test_ctc_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_loss(*config_and_inputs)

    def test_seq_classifier_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_seq_classifier_loss(*config_and_inputs)

    def test_ctc_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_training(*config_and_inputs)

    def test_seq_classifier_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_seq_classifier_training(*config_and_inputs)

    def test_xvector_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_xvector_training(*config_and_inputs)

    def test_labels_out_of_vocab(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_labels_out_of_vocab(*config_and_inputs)

    # Data2VecAudio has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # `input_ids` is renamed to `input_values`
    def test_forward_signature(self):
        pass

    # Data2VecAudio cannot resize token embeddings
    # since it has no tokens embeddings
    def test_resize_tokens_embeddings(self):
        pass

    # Data2VecAudio has no inputs_embeds
    # and thus the `get_input_embeddings` fn
    # is not implemented
    def test_model_common_attributes(self):
        pass

    @is_pt_flax_cross_test
    # non-robust architecture does not exist in Flax
    def test_equivalence_flax_to_pt(self):
        pass

    @is_pt_flax_cross_test
    # non-robust architecture does not exist in Flax
    def test_equivalence_pt_to_flax(self):
        pass

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        # set layer drop to 0
        model.config.layerdrop = 0.0

        input_values = inputs_dict["input_values"]

        input_lengths = torch.tensor(
            [input_values.shape[1] for _ in range(input_values.shape[0])], dtype=torch.long, device=torch_device
        )
        output_lengths = model._get_feat_extract_output_lengths(input_lengths)

        labels = ids_tensor((input_values.shape[0], output_lengths[0] - 2), self.model_tester.vocab_size)
        inputs_dict["attention_mask"] = torch.ones_like(inputs_dict["attention_mask"])
        inputs_dict["labels"] = labels

        outputs = model(**inputs_dict)

        output = outputs[0]

        # Encoder-/Decoder-only models
        hidden_states = outputs.hidden_states[0]
        attentions = outputs.attentions[0]

        hidden_states.retain_grad()
        attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)
        self.assertIsNotNone(attentions.grad)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = [
                    "conv.weight",
                    "masked_spec_embed",
                    "codevectors",
                    "quantizer.weight_proj.weight",
                    "project_hid.weight",
                    "project_hid.bias",
                    "project_q.weight",
                    "project_q.bias",
                    "feature_projection.projection.weight",
                    "feature_projection.projection.bias",
                    "objective.weight",
                ]
                if param.requires_grad:
                    if any([x in name for x in uniform_init_parms]):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    # overwrite from test_modeling_common
    def _mock_init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.fill_(3)
        if hasattr(module, "weight_g") and module.weight_g is not None:
            module.weight_g.data.fill_(3)
        if hasattr(module, "weight_v") and module.weight_v is not None:
            module.weight_v.data.fill_(3)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.fill_(3)
        if hasattr(module, "codevectors") and module.codevectors is not None:
            module.codevectors.data.fill_(3)
        if hasattr(module, "masked_spec_embed") and module.masked_spec_embed is not None:
            module.masked_spec_embed.data.fill_(3)

    def test_mask_feature_prob_ctc(self):
        model = Data2VecAudioForCTC.from_pretrained(
            "facebook/data2vec-audio-base-960h", mask_feature_prob=0.2, mask_feature_length=2
        )
        model.to(torch_device).train()
        processor = Wav2Vec2Processor.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2", return_attention_mask=True
        )

        batch_duration_in_seconds = [1, 3, 2, 6]
        input_features = [np.random.random(16_000 * s) for s in batch_duration_in_seconds]

        batch = processor(
            input_features, padding=True, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt"
        )

        logits = model(
            input_values=batch["input_values"].to(torch_device),
            attention_mask=batch["attention_mask"].to(torch_device),
        ).logits

        self.assertEqual(logits.shape, (4, 299, 32))

    def test_mask_time_prob_ctc(self):
        model = Data2VecAudioForCTC.from_pretrained(
            "facebook/data2vec-audio-base-960h", mask_time_prob=0.2, mask_time_length=2
        )
        model.to(torch_device).train()
        processor = Wav2Vec2Processor.from_pretrained(
            "hf-internal-testing/tiny-random-wav2vec2", return_attention_mask=True
        )

        batch_duration_in_seconds = [1, 3, 2, 6]
        input_features = [np.random.random(16_000 * s) for s in batch_duration_in_seconds]

        batch = processor(
            input_features, padding=True, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt"
        )

        logits = model(
            input_values=batch["input_values"].to(torch_device),
            attention_mask=batch["attention_mask"].to(torch_device),
        ).logits

        self.assertEqual(logits.shape, (4, 299, 32))

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")
        self.assertIsNotNone(model)


@require_torch
class Data2VecAudioUtilsTest(unittest.TestCase):
    def test_compute_mask_indices(self):
        batch_size = 4
        sequence_length = 60
        mask_prob = 0.5
        mask_length = 1

        mask = _compute_mask_indices((batch_size, sequence_length), mask_prob, mask_length)
        mask = torch.from_numpy(mask).to(torch_device)

        self.assertListEqual(mask.sum(axis=-1).tolist(), [mask_prob * sequence_length for _ in range(batch_size)])

    def test_compute_mask_indices_low_prob(self):
        # with these settings num_masked_spans=0.5, which means probabilistic rounding
        # ensures that in 5 out of 10 method calls, num_masked_spans=0, and in
        # the other 5 out of 10, cases num_masked_spans=1
        n_trials = 100
        batch_size = 4
        sequence_length = 100
        mask_prob = 0.05
        mask_length = 10

        count_dimensions_masked = 0
        count_dimensions_not_masked = 0

        for _ in range(n_trials):
            mask = _compute_mask_indices((batch_size, sequence_length), mask_prob, mask_length)
            mask = torch.from_numpy(mask).to(torch_device)

            num_masks = torch.sum(mask).item()

            if num_masks > 0:
                count_dimensions_masked += 1
            else:
                count_dimensions_not_masked += 1

        # as we test for at least 10 masked dimension and at least
        # 10 non-masked dimension, this test could fail with probability:
        # P(100 coin flips, at most 9 heads) = 1.66e-18
        self.assertGreater(count_dimensions_masked, int(n_trials * 0.1))
        self.assertGreater(count_dimensions_not_masked, int(n_trials * 0.1))

    def test_compute_mask_indices_overlap(self):
        batch_size = 4
        sequence_length = 80
        mask_prob = 0.5
        mask_length = 4

        mask = _compute_mask_indices((batch_size, sequence_length), mask_prob, mask_length)
        mask = torch.from_numpy(mask).to(torch_device)

        # because of overlap mask don't have to add up exactly to `mask_prob * sequence_length`, but have to be smaller or equal
        for batch_sum in mask.sum(axis=-1):
            self.assertTrue(int(batch_sum) <= mask_prob * sequence_length)

    def test_compute_mask_indices_attn_mask_overlap(self):
        batch_size = 4
        sequence_length = 80
        mask_prob = 0.5
        mask_length = 4

        attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long, device=torch_device)
        attention_mask[:2, sequence_length // 2 :] = 0

        mask = _compute_mask_indices(
            (batch_size, sequence_length), mask_prob, mask_length, attention_mask=attention_mask
        )
        mask = torch.from_numpy(mask).to(torch_device)

        for batch_sum in mask.sum(axis=-1):
            self.assertTrue(int(batch_sum) <= mask_prob * sequence_length)

        self.assertTrue(mask[:2, sequence_length // 2 :].sum() == 0)

    def test_compute_mask_indices_short_audio(self):
        batch_size = 4
        sequence_length = 100
        mask_prob = 0.05
        mask_length = 10

        attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long, device=torch_device)
        # force one example to be heavily padded
        attention_mask[0, 5:] = 0

        mask = _compute_mask_indices(
            (batch_size, sequence_length), mask_prob, mask_length, attention_mask=attention_mask, min_masks=2
        )

        # make sure that non-padded examples cannot be padded
        self.assertFalse(mask[0][attention_mask[0].to(torch.bool).cpu()].any())


@require_torch
@require_soundfile
@slow
class Data2VecAudioModelIntegrationTest(unittest.TestCase):
    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").filter(
            lambda x: x["id"] in [f"1272-141231-000{i}" for i in range(num_samples)]
        )[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def _load_superb(self, task, num_samples):
        ds = load_dataset("anton-l/superb_dummy", task, split="test")

        return ds[:num_samples]

    def test_inference_ctc_normal(self):
        model = Data2VecAudioForCTC.from_pretrained("facebook/data2vec-audio-base-960h")
        model.to(torch_device)
        processor = Wav2Vec2Processor.from_pretrained("hf-internal-testing/tiny-random-wav2vec2", do_lower_case=True)
        input_speech = self._load_datasamples(1)

        input_values = processor(input_speech, return_tensors="pt").input_values.to(torch_device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = ["a man said to the universe sir i exist"]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)

    def test_inference_ctc_batched(self):
        model = Data2VecAudioForCTC.from_pretrained("facebook/data2vec-audio-base-960h").to(torch_device)
        processor = Wav2Vec2Processor.from_pretrained("hf-internal-testing/tiny-random-wav2vec2", do_lower_case=True)

        input_speech = self._load_datasamples(4)

        inputs = processor(input_speech, return_tensors="pt", padding=True)

        input_values = inputs.input_values.to(torch_device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "sweat covered brion's body trickling into the tight loin cloth that was the only garment he wore",
            "the cut on his chest still dripping blood the ache of his overstrained eyes even the soaring arena around him with thousands of spectators were trivialities not worth thinking about",
            "his instant of panic was followed by a small sharp blow high on his chest",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)
