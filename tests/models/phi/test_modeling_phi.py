# coding=utf-8
# Copyright 2023 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch Phi model. """


import unittest

from parameterized import parameterized
from pytest import mark

from transformers import PhiConfig, is_torch_available, set_seed
from transformers.testing_utils import require_flash_attn, require_torch, require_torch_gpu, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        PhiForCausalLM,
        PhiForSequenceClassification,
        PhiModel,
        CodeGenTokenizer,
    )


class PhiModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu_new",
        resid_pdrop=0.1,
        max_position_embeddings=512,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.resid_pdrop = resid_pdrop
        self.max_position_embeddings = max_position_embeddings
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return PhiConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            resid_pdrop=self.resid_pdrop,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_model(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = PhiModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        input_mask,
    ):
        config.add_cross_attention = True
        model = PhiModel(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
        )
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        input_mask,
        token_labels,
    ):
        model = PhiForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        input_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = False
        model = PhiForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
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
            output_hidden_states=True,
        )["hidden_states"][0]
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

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class PhiModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (PhiModel, PhiForCausalLM, PhiForSequenceClassification) if is_torch_available() else ()
    all_generative_model_classes = (PhiForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": PhiModel,
            # "question-answering": GPT2ForQuestionAnswering,
            "text-classification": PhiForSequenceClassification,
            "text-generation": PhiForCausalLM,
            # "token-classification": GPT2ForTokenClassification,
            "zero-shot": PhiForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = PhiModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PhiConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_phi_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = PhiForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_phi_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = PhiForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_phi_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = PhiForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))


# @require_torch
# class PhiIntegrationTest(unittest.TestCase):
#     @unittest.skip("Logits are not exactly the same, once we fix the instabalities somehow, will update!")
#     @slow
#     def test_model_7b_logits(self):
#         input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
#         model = PhiForCausalLM.from_pretrained("microsoft/phi-1-hf", device_map="auto")
#         out = model(torch.tensor([input_ids]))
#         # Expected mean on dim = -1
#         EXPECTED_MEAN = torch.tensor([[-6.6550, -4.1227, -4.9859, -3.2406, 0.8262, -3.0033, 1.2964, -3.3699]])
#         torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
#         # slicing logits[0, 0, 0:30]
#         # fmt: off
#         EXPECTED_SLICE = torch.tensor([-12.8281, -7.4453, -0.4639, -8.0625, -7.2500, -8.0000, -6.4883, -7.7695, -7.8438, -7.0312, -6.2188, -7.1328, -1.8496, 1.9961, -8.6250, -6.7227, -12.8281, -6.9492, -7.0742, -7.7852, -7.5820, -7.9062, -6.9375, -7.9805, -8.3438, -8.1562, -8.0469, -7.6250, -7.7422, -7.3398,])
#         # fmt: on
#         torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5)
#
#     @unittest.skip("Logits are not exactly the same, once we fix the instabalities somehow, will update!")
#     @slow
#     def test_model_13b_logits(self):
#         input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
#         model = PhiForCausalLM.from_pretrained("meta-phi/Phi-2-13b-hf", device_map="auto")
#         out = model(torch.tensor(input_ids))
#         # Expected mean on dim = -1
#         EXPECTED_MEAN = torch.tensor([[-2.0622, -1.2794, -1.1638, -0.9788, -1.4603, -1.0238, -1.7893, -1.4411]])
#         torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
#         # slicing logits[0, 0, 0:30]
#         # fmt: off
#         EXPECTED_SLICE = torch.tensor([-8.1406, -8.0547, 2.7461, -1.2344, -0.1448, -1.8262, -1.0020, -1.8154, -1.6895, -1.8516, -2.3574, -0.9277, 3.7598, 6.5742, -1.2998, -0.1177, -8.1406, -2.9688, -2.9199, -3.1699, -3.5254, -2.3555, -2.7988, -3.4141, -2.8262, -4.5195, -3.3379, -3.3164, -2.7832, -3.0273])
#         # fmt: on
#         torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5)
#
#     @unittest.skip("Logits are not exactly the same, once we fix the instabalities somehow, will update!")
#     @slow
#     def test_model_13bf_logits(self):
#         input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
#         model = PhiForCausalLM.from_pretrained("meta-phi/Phi-2-13b-chat-hf", device_map="auto")
#         out = model(torch.tensor(input_ids))
#         # Expected mean on dim = -1
#         EXPECTED_MEAN = torch.tensor([[-0.8562, -1.8520, -0.7551, -0.4162, -1.5161, -1.2038, -2.4823, -2.3254]])
#         torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
#         # slicing logits[0, 0, 0:30]
#         # fmt: off
#         EXPECTED_SLICE = torch.tensor([-2.2227, 4.8828, 0.9023, -0.4578, -0.7871, -0.1033, -0.6221, -0.5786, -0.7803, -1.0674, -1.2920, -0.1570, 0.8008, 2.0723, -0.9497, 0.2771, -2.2227, -0.7612, -1.4346, -1.2061, -1.6426, -0.3000, -0.7139, -1.1934, -1.8691, -1.6973, -1.5947, -1.2705, -0.3523, -0.5513])
#         # fmt: on
#         torch.testing.assert_close(out.mean(-1), EXPECTED_SLICE, atol=1e-2, rtol=1e-2)
#
#     @unittest.skip(
#         "Logits are not exactly the same, once we fix the instabalities somehow, will update! Also it is gonna be a `too_slow` test"
#     )
#     @slow
#     def test_model_70b_logits(self):
#         input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
#         model = PhiForCausalLM.from_pretrained("meta-phi/Phi-2-70b-hf", device_map="auto")
#         out = model(torch.tensor(input_ids))
#
#         EXPECTED_MEAN = torch.tensor(
#             [[-4.2327, -3.3360, -4.6665, -4.7631, -1.8180, -3.4170, -1.4211, -3.1810]], dtype=torch.float32
#         )
#         torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
#         # fmt: off
#         EXPECTED_SLICE = torch.tensor([-9.4922, -3.9551, 1.7998, -5.6758, -5.1055, -5.8984, -4.8320, -6.8086, -6.5391, -5.6172, -5.5820, -5.5352, 1.7881, 3.6289, -6.5117, -3.4785, -9.5000, -6.0352, -6.8125, -6.0195, -6.6836, -5.4727, -6.2812, -6.0391, -7.3398, -7.4297, -7.4844, -6.5820, -5.8789, -5.5312])
#         # fmt: on
#         torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5)
#
#     @unittest.skip("Model is curently gated")
#     @slow
#     def test_model_13b_greedy_generation(self):
#         EXPECTED_TEXT_COMPLETION = """Simply put, the theory of relativity states that 1) the laws of physics are the same everywhere in the universe and 2) the passage of time and the length of objects can vary depending on the observer\'s frame of reference.\n\nThe first part of the theory, that the laws of physics are the same everywhere, is known as the "princi"""
#         prompt = "Simply put, the theory of relativity states that "
#         tokenizer = CodeGenTokenizer.from_pretrained("meta-phi/Phi-2-13b-chat-hf")
#         input_ids = tokenizer.encode(prompt, return_tensors="pt")
#         model = PhiForCausalLM.from_pretrained(
#             "meta-phi/Phi-2-13b-chat-hf", device_map="sequential", use_safetensors=False
#         )
#
#         # greedy generation outputs
#         generated_ids = model.generate(input_ids, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
#         text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#         self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

