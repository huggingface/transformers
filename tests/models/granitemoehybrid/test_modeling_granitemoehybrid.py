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
"""Testing suite for the PyTorch GraniteMoeHybrid model."""

import unittest

import pytest

from transformers import (
    AutoTokenizer,
    GraniteMoeHybridConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        GraniteMoeHybridForCausalLM,
        GraniteMoeHybridModel,
    )


class GraniteMoeHybridModelTester(CausalLMModelTester):
    config_class = GraniteMoeHybridConfig
    if is_torch_available():
        model_class = GraniteMoeHybridModel
        base_model_class = GraniteMoeHybridModel
        for_causal_lm_class = GraniteMoeHybridForCausalLM
        causal_lm_class = GraniteMoeHybridForCausalLM

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
        hidden_size=64,
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
        pad_token_id=0,
        scope=None,
        use_cache=False,
        shared_intermediate_size=174,
        layer_types=None,
        attn_layer_indices=None,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank="auto",
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        mamba_inner_layernorms=True,
        n_mamba_heads=2,
        layers_block_type=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        num_key_value_heads=None,
        tie_word_embeddings=False,
    ):
        super().__init__(parent)
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
        self.pad_token_id = pad_token_id
        self.scope = scope
        self.use_cache = use_cache
        self.shared_intermediate_size = shared_intermediate_size
        self.layer_types = layer_types
        self.attn_layer_indices = attn_layer_indices if attn_layer_indices is not None else [1]
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = mamba_dt_rank
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.mamba_inner_layernorms = mamba_inner_layernorms
        self.n_mamba_heads = n_mamba_heads
        self.mamba_n_heads = n_mamba_heads  # GraniteMoeHybrid uses mamba_n_heads
        self.layers_block_type = layers_block_type
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.num_key_value_heads = num_key_value_heads
        self.tie_word_embeddings = tie_word_embeddings

        self._update_layer_configs()

    def _update_layer_configs(self):
        # GraniteMoeHybrid uses layer_types instead of attn_layer_indices
        self.layer_types = ["mamba"] * self.num_hidden_layers
        for idx in self.attn_layer_indices:
            self.layer_types[idx] = "attention"

        # Also update layers_block_type for compatibility
        self.layers_block_type = self.layer_types

    def get_config(self):
        return self.config_class(
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
            use_cache=self.use_cache,
            shared_intermediate_size=self.shared_intermediate_size,
            layer_types=self.layer_types,
            mamba_d_state=self.mamba_d_state,
            mamba_d_conv=self.mamba_d_conv,
            mamba_expand=self.mamba_expand,
            mamba_dt_rank=self.mamba_dt_rank,
            mamba_conv_bias=self.mamba_conv_bias,
            mamba_proj_bias=self.mamba_proj_bias,
            mamba_inner_layernorms=self.mamba_inner_layernorms,
            mamba_n_heads=self.mamba_n_heads,
            layers_block_type=self.layers_block_type,
            attention_bias=self.attention_bias,
            attention_dropout=self.attention_dropout,
            mlp_bias=self.mlp_bias,
            num_key_value_heads=self.num_key_value_heads,
            tie_word_embeddings=self.tie_word_embeddings,
            pad_token_id=self.pad_token_id,
        )

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

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

    def create_and_check_decoder_model_past_large_inputs(self, config, input_ids, token_type_ids, input_mask, *args):
        config.is_decoder = True
        config.add_cross_attention = True
        model = GraniteMoeHybridForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
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
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
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


@require_torch
class GraniteMoeHybridModelTest(CausalLMModelTest, GenerationTesterMixin, unittest.TestCase):
    model_tester_class = GraniteMoeHybridModelTester
    all_model_classes = (
        (
            GraniteMoeHybridModel,
            GraniteMoeHybridForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": GraniteMoeHybridModel,
            "text-generation": GraniteMoeHybridForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = GraniteMoeHybridModelTester(self)
        # hidden_size must be chosen so that mamba_expand * hidden_size is divisible by mamba_n_heads
        # Default mamba_expand=2, mamba_n_heads=128, so hidden_size must be divisible by 64
        self.config_tester = ConfigTester(self, config_class=GraniteMoeHybridConfig, hidden_size=64)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_config_requires_mamba_or_attention_layers(self):
        """Ensure we can't create a config with disallowed layers."""
        with pytest.raises(ValueError):
            GraniteMoeHybridConfig(layer_types=["not allowed!"])

    @unittest.skip(reason="GraniteMoeHybrid has different cache handling")
    def test_decoder_model_past_with_large_inputs(self):
        pass

    @unittest.skip(reason="Stateful models don't support assisted generation")
    def test_assisted_decoding_matches_greedy_search(self):
        pass

    @unittest.skip(reason="Stateful models don't support assisted generation")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(reason="Bamba has its own special cache type")
    def test_past_key_values_format(self):
        pass

    @unittest.skip(reason="Mamba's A_log parameter is initialized differently")
    def test_initialization(self):
        pass

    @unittest.skip(reason="GraniteMoeHybrid has different attention output structure")
    def test_attention_outputs(self):
        pass


# TODO (@alex-jw-brooks) - update this once the model(s) are out
@unittest.skip(reason="GraniteMoeHybrid models are not yet released")
@require_torch_gpu
class GraniteMoeHybridIntegrationTest(unittest.TestCase):
    @slow
    def test_model_logits(self):
        input_ids = [31390, 631, 4162, 30, 322, 25342, 432, 1875, 43826, 10066, 688, 225]

        model = GraniteMoeHybridForCausalLM.from_pretrained("ibm-granite/granite-4.0-tiny", device_map="auto")

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        # fmt: off
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([
            [-2.9711, -2.2554, -1.0814, -1.6123, -0.8780, -1.0685, -0.6368, -1.9732, -3.3548, -2.6895, -2.3062, -2.6338]
        ])

        torch.testing.assert_close(EXPECTED_MEAN.to(torch_device), out.logits.float().mean(-1), rtol=1e-2, atol=1e-2)

        # slicing logits[0, 0, 0:15]
        EXPECTED_SLICE = torch.tensor([
            [4.0662, 5.9547, 3.5803, 3.1306, 4.3211, 3.8902, 4.6438, 8.5434, 7.5865, 5.1623, 5.2240, 9.2982, 5.9094, 6.8834, 5.7551],
        ])
        # fmt: on

        self.assertTrue(
            torch.allclose(
                EXPECTED_SLICE.to(torch_device),
                out.logits[0, 0, :15].float(),
                atol=1e-3,
                rtol=1e-3,
            )
        )

    @slow
    def test_model_generation(self):
        EXPECTED_TEXT_COMPLETION = (
            "Simply put, the theory of relativity states that 1) time is relative, and 2) space is relative. The first"
        )
        prompt = "Simply put, the theory of relativity states that "
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.0-tiny")
        model = GraniteMoeHybridForCausalLM.from_pretrained("ibm-granite/granite-4.0-tiny", device_map="auto")
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # greedy generation outputs
        generated_ids = model.generate(**model_inputs, max_new_tokens=16, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
