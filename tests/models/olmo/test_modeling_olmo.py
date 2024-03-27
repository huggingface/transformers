# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch OLMo model. """

import unittest

from transformers import OLMoConfig, is_torch_available
from transformers.testing_utils import (
    require_torch,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import OLMoForCausalLM, OLMoModel


class OLMoModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        d_model=32,
        mlp_hidden_size=None,
        n_layers=2,
        n_heads=2,
        use_python_sdpa=True,
        eos_token_id=42,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mlp_hidden_size = mlp_hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_sequence_length = seq_length
        self.use_python_sdpa = use_python_sdpa
        self.eos_token_id = eos_token_id
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices

        # `test_attention_outputs` and `test_hidden_states_output` don't seem to respect the attribute map
        self.num_hidden_layers = n_layers
        self.num_attention_heads = n_heads
        self.hidden_size = d_model

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones(self.batch_size, self.seq_length)).to(torch_device)

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
        return OLMoConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            mlp_hidden_size=self.mlp_hidden_size,
            max_sequence_length=self.max_sequence_length,
            use_python_sdpa=self.use_python_sdpa,
            eos_token_id=self.eos_token_id,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = OLMoModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.d_model))

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
        model = OLMoModel(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.d_model))

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
        model = OLMoForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
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
        model = OLMoForCausalLM(config=config)
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
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class OLMoModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (OLMoModel, OLMoForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (OLMoForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": OLMoModel,
            "text-generation": OLMoForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_torchscript = False
    fx_compatible = False

    def setUp(self):
        self.model_tester = OLMoModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=OLMoConfig, d_model=37, common_properties=["d_model", "n_heads", "n_layers"]
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip("OLMo does not support head pruning.")
    def test_headmasking(self):
        pass

    @require_torch_sdpa
    def test_no_sdpa_matches_sdpa(self):
        config = self.model_tester.get_config()

        batch = torch.randint(0, config.vocab_size, (2, config.max_sequence_length))

        model = OLMoForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        self.assertTrue(model.config.use_pytorch_sdpa)

        res_sdpa = model.forward(batch)

        model.config.use_pytorch_sdpa = False

        res_no_sdpa = model.forward(batch)

        torch.testing.assert_close(
            res_no_sdpa,
            res_sdpa,
        )


@require_torch
class OLMoIntegrationTest(unittest.TestCase):
    @slow
    def test_model_1b_logits(self):
        input_ids = [[1, 306, 4658, 278, 6593, 310, 2834, 338]]
        model = OLMoForCausalLM.from_pretrained("allenai/OLMo-1B")
        out = model(torch.tensor(input_ids)).logits
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[2.2869, 0.3315, 0.9876, 1.4146, 1.8804, 2.0430, 1.7055, 1.2065]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([2.5551, -1.1230, 11.0510, 12.4977, 7.9651, 7.2342, 6.1885, 7.8340, 9.9847, 12.6695, 12.2345, 10.7970, 8.4749, 14.2483, 12.9588, 13.9233, 11.0496, 5.5749, 7.4466, 7.7914, 6.8440, 5.8951, 4.8180, 4.1935, 4.5216, 4.7256, 3.9553, 12.2870, 12.4990, 8.1591])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-2, rtol=1e-2)

    @slow
    def test_model_7b_logits(self):
        input_ids = [[1, 306, 4658, 278, 6593, 310, 2834, 338]]
        model = OLMoForCausalLM.from_pretrained("allenai/OLMo-7B")
        out = model(torch.tensor(input_ids)).logits
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[0.0271, 0.0249, -0.0578, -0.0870, 0.0167, 0.0710, 0.1002, 0.0677]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([-1.7433, -1.6685, 7.4941, 6.1506, 0.1364, -0.1127, 1.3224, 4.5458, 4.2068, 5.8296, 7.4723, 2.7925, 3.1245, 10.8872, 10.0758, 10.6717, 7.0945, 1.2398, 3.6766, 4.2365, 2.5655, 2.2222, 1.7418, 0.5223, 0.7753, 1.0938, 0.6723, 6.2522, 6.2264, 1.8105])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-2, rtol=1e-2)

    # @unittest.skip("Logits are not exactly the same, once we fix the instabalities somehow, will update!")
    @slow
    def test_model_7b_twin_2t_logits(self):
        input_ids = [[1, 306, 4658, 278, 6593, 310, 2834, 338]]
        model = OLMoForCausalLM.from_pretrained("allenai/OLMo-7B-Twin-2T")
        out = model(torch.tensor(input_ids)).logits
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-0.3636, -0.3825, -0.4800, -0.3696, -0.8388, -0.9737, -0.9849, -0.8356]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([-2.0833, -1.9234, 8.7312, 7.8049, 1.0372, 0.8941, 3.1548, 1.8502, 5.5511, 5.5793, 8.1166, 4.5906, 1.8691, 11.6377, 8.9858, 11.6447, 7.4549, 1.4725, 2.8399, 2.7568, 1.4011, 1.6958, 0.5572, 0.5231, 0.3068, 0.5364, 0.6769, 7.9636, 8.2379, 1.7950])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-2, rtol=1e-2)
