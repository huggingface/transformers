# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from transformers import MegaConfig, is_torch_available
from transformers.testing_utils import TestCasePlus, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        MegaForCausalLM,
        MegaForMaskedLM,
        MegaForMultipleChoice,
        MegaForQuestionAnswering,
        MegaForSequenceClassification,
        MegaForTokenClassification,
        MegaModel,
    )
    from transformers.models.mega.modeling_mega import MEGA_PRETRAINED_MODEL_ARCHIVE_LIST


class MegaModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        intermediate_size=37,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_positions=1024,
        bidirectional=False,  # needed for decoding, and can't modify common generation tests; test separately by overriding
        ema_projection_size=16,
        shared_representation_size=64,
        use_chunking=False,
        chunk_size=32,
        attention_activation="softmax",
        use_normalized_ffn=True,
        nffn_hidden_size=24,
        add_token_type_embeddings=True,
        type_vocab_size=2,
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
        self.add_token_type_embeddings = add_token_type_embeddings
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_positions = max_positions
        self.bidirectional = bidirectional
        self.ema_projection_size = ema_projection_size
        self.shared_representation_size = shared_representation_size
        self.use_chunking = use_chunking
        self.chunk_size = chunk_size
        self.attention_activation = attention_activation
        self.use_normalized_ffn = use_normalized_ffn
        self.nffn_hidden_size = nffn_hidden_size
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.num_attention_heads = 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.add_token_type_embeddings:
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
        return MegaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            # added args
            add_token_type_embeddings=self.add_token_type_embeddings,
            max_positions=self.max_positions,
            bidirectional=self.bidirectional,
            ema_projection_size=self.ema_projection_size,
            shared_representation_size=self.shared_representation_size,
            use_chunking=self.use_chunking,
            chunk_size=self.chunk_size,
            attention_activation=self.attention_activation,
            use_normalized_ffn=self.use_normalized_ffn,
            nffn_hidden_size=self.nffn_hidden_size,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.vocab_size = 300
        return config

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
        config.bidirectional = False
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
        model = MegaModel(config=config)
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
        model = MegaModel(config)
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
        model = MegaForCausalLM(config=config)
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
        config.bidirectional = False
        config.add_cross_attention = True
        model = MegaForCausalLM(config=config).to(torch_device).eval()

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
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # make sure that ids don't start with pad token
        mask = next_tokens.ne(config.pad_token_id).long()
        next_tokens = next_tokens * mask
        next_mask = ids_tensor((self.batch_size, 1), vocab_size=2)

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
            attention_mask=next_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = MegaForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = MegaForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_multiple_choice(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = MegaForMultipleChoice(config=config)
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
        model = MegaForQuestionAnswering(config=config)
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

    # extra checks for Mega-specific model functionality
    def create_and_check_bidirectionality(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.bidirectional = True
        model = MegaModel(config)
        model.to(torch_device)
        model.eval()
        # no mask
        result = model(input_ids)
        # with mask & token types
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)

        self.parent.assertEqual(result[0].shape, (self.batch_size, self.seq_length, self.hidden_size))

    def check_chunking_shorter_sequence(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.use_chunking = True
        config.chunk_size = input_ids.size(1) + 25
        model = MegaModel(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)

        self.parent.assertEqual(result[0].shape, (self.batch_size, self.seq_length, self.hidden_size))

    def check_chunking_longer_sequence(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.use_chunking = True

        # we want the chunk size to be < sequence length, and the sequence length to be a multiple of chunk size
        config.chunk_size = input_ids.size(1) * 2
        model = MegaModel(config)
        model.to(torch_device)
        model.eval()

        result = model(
            input_ids.repeat(1, 8),
        )

        self.parent.assertEqual(result[0].shape, (self.batch_size, self.seq_length * 8, self.hidden_size))

    def check_laplace_self_attention(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.attention_activation = "laplace"
        model = MegaModel(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)

        self.parent.assertEqual(result[0].shape, (self.batch_size, self.seq_length, self.hidden_size))

    def check_relu2_self_attention(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.attention_activation = "relu2"
        model = MegaModel(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)

        self.parent.assertEqual(result[0].shape, (self.batch_size, self.seq_length, self.hidden_size))

    def check_sequence_length_beyond_max_positions(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.max_positions = self.seq_length - 2
        model = MegaModel(config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)

        self.parent.assertEqual(result[0].shape, (self.batch_size, self.seq_length, self.hidden_size))

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
class MegaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            MegaForCausalLM,
            MegaForMaskedLM,
            MegaModel,
            MegaForSequenceClassification,
            MegaForTokenClassification,
            MegaForMultipleChoice,
            MegaForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (MegaForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": MegaModel,
            "fill-mask": MegaForMaskedLM,
            "question-answering": MegaForQuestionAnswering,
            "text-classification": MegaForSequenceClassification,
            "text-generation": MegaForCausalLM,
            "token-classification": MegaForTokenClassification,
            "zero-shot": MegaForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    fx_compatible = False
    test_head_masking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = MegaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MegaConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
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

    def test_for_bidirectionality(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_bidirectionality(*config_and_inputs)

    def test_for_chunking_shorter_sequence(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_chunking_shorter_sequence(*config_and_inputs)

    def test_for_chunking_longer_sequence(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_chunking_longer_sequence(*config_and_inputs)

    def test_for_laplace_attention(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_laplace_self_attention(*config_and_inputs)

    def test_for_relu2_attention(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_relu2_self_attention(*config_and_inputs)

    def test_for_sequence_length_beyond_max_positions(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_sequence_length_beyond_max_positions(*config_and_inputs)

    def test_generate_fp16(self):
        config, input_ids, _, attention_mask, *_ = self.model_tester.prepare_config_and_inputs_for_decoder()
        # attention_mask = torch.LongTensor(input_ids.ne(1)).to(torch_device)
        model = MegaForCausalLM(config).eval().to(torch_device)
        if torch_device == "cuda":
            model.half()
        model.generate(input_ids, attention_mask=attention_mask)
        model.generate(num_beams=4, do_sample=True, early_stopping=False, num_return_sequences=3)

    def test_sequence_classification_model(self):
        config, input_ids, _, attention_mask, *_ = self.model_tester.prepare_config_and_inputs()
        config.num_labels = self.model_tester.num_labels
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = MegaForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_sequence_classification_model_for_multi_label(self):
        config, input_ids, _, attention_mask, *_ = self.model_tester.prepare_config_and_inputs()
        config.num_labels = self.model_tester.num_labels
        config.problem_type = "multi_label_classification"
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = MegaForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    @slow
    def test_model_from_pretrained(self):
        for model_name in MEGA_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = MegaModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_cpu_offload(self):
        super().test_cpu_offload()

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_disk_offload(self):
        super().test_disk_offload()

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_model_parallelism(self):
        super().test_model_parallelism()

    @unittest.skip(
        reason=(
            "Calling `self.attention_function` in `MegaMovingAverageGatedAttention.forward` changes the submodules on "
            "device 1 to device 0 (also changes `requires_grad`). No idea how this could happen for now."
        )
    )
    def test_multi_gpu_data_parallel_forward(self):
        super().test_multi_gpu_data_parallel_forward()

    @unittest.skip(reason="Tracing of the dynamically computed `MegaMultiDimensionDampedEma._kernel` doesn't work.")
    def test_torchscript_simple(self):
        super().test_torchscript_simple()

    @unittest.skip(reason="Tracing of the dynamically computed `MegaMultiDimensionDampedEma._kernel` doesn't work.")
    def test_torchscript_output_hidden_state(self):
        super().test_torchscript_output_hidden_state()

    @unittest.skip(reason="Tracing of the dynamically computed `MegaMultiDimensionDampedEma._kernel` doesn't work.")
    def test_torchscript_output_attentions(self):
        super().test_torchscript_output_attentions()


@require_torch
class MegaModelIntegrationTest(TestCasePlus):
    @slow
    def test_inference_masked_lm(self):
        model = MegaForMaskedLM.from_pretrained("mnaylor/mega-base-wikitext")

        input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        with torch.no_grad():
            output = model(input_ids)[0]
        expected_shape = torch.Size((1, 11, 50265))
        self.assertEqual(output.shape, expected_shape)
        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [[[67.8389, 10.1470, -32.7148], [-11.1655, 29.1152, 23.1304], [-3.8015, 66.0397, 29.6733]]]
        )

        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-4))

    @slow
    def test_inference_no_head(self):
        model = MegaModel.from_pretrained("mnaylor/mega-base-wikitext")

        input_ids = torch.tensor([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        with torch.no_grad():
            output = model(input_ids)[0]
        expected_shape = torch.Size((1, 11, 128))
        self.assertEqual(output.shape, expected_shape)
        # compare the actual values for a slice. taken from output[:, :3, :3]
        expected_slice = torch.tensor(
            [[[1.1767, -0.6349, 2.8494], [-0.5109, -0.7745, 1.9495], [-0.3287, -0.2111, 3.3367]]]
        )

        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-4))
