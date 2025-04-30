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

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        GraniteMoeHybridForCausalLM,
        GraniteMoeHybridModel,
    )
    from transformers.models.bamba.modeling_bamba import (
        HybridMambaAttentionDynamicCache,
    )


class GraniteMoeHybridModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        num_labels=3,
        vocab_size=99,
        hidden_size=32,
        intermediate_size=110,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        use_cache=False,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        attention_bias=False,
        embedding_multiplier=1.0,
        logits_scaling=1.0,
        residual_multiplier=1.0,
        attention_multiplier=1.0,
        num_local_experts=8,
        num_experts_per_tok=2,
        shared_intermediate_size=174,
        normalization_function=None,
        position_embedding_type=None,
        # layer types should be a List of str
        layer_types=None,
        # took defaults from bamba config
        mamba_n_heads=16,
        mamba_n_groups=1,
        mamba_d_state=16,
        mamba_d_head="auto",
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_chunk_size=16,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.embedding_multiplier = embedding_multiplier
        self.logits_scaling = logits_scaling
        self.residual_multiplier = residual_multiplier
        self.attention_multiplier = attention_multiplier
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.shared_intermediate_size = shared_intermediate_size
        self.normalization_function = normalization_function
        self.position_embedding_type = position_embedding_type
        # layer types should be a List of str
        self.layer_types = layer_types
        # took defaults from bamba config
        self.mamba_n_heads = mamba_n_heads
        self.mamba_n_groups = mamba_n_groups
        self.mamba_d_state = mamba_d_state
        self.mamba_d_head = mamba_d_head
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_chunk_size = mamba_chunk_size
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

        token_labels = None
        if self.use_labels:
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()

        return config, input_ids, input_mask, token_labels

    def get_config(self):
        # Similar to Bamba, setting some layers as attention
        if self.num_hidden_layers < 4:
            self.num_hidden_layers = 4
        if self.layer_types is None:
            d = [x for x in range(2, self.num_hidden_layers) if self.num_hidden_layers % x == 0]
            if len(d) == 0:
                raise ValueError("num_hidden_layers is prime, cannot automatically set attn_layer_indices.")
            d = d[-1]  # get the largest divisor
            self.attn_layer_indices = [x + 1 for x in range(0, self.num_hidden_layers, d)]
            self.layer_types = ["mamba"] * self.num_hidden_layers
            for idx in self.attn_layer_indices:
                self.layer_types[idx] = "attention"

        return GraniteMoeHybridConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            shared_intermediate_size=self.shared_intermediate_size,
            layer_types=self.layer_types,
            mamba_n_heads=self.mamba_n_heads,
            mamba_n_groups=self.mamba_n_groups,
            mamba_d_state=self.mamba_d_state,
            mamba_d_head=self.mamba_d_head,
            mamba_d_conv=self.mamba_d_conv,
            mamba_expand=self.mamba_expand,
            mamba_chunk_size=self.mamba_chunk_size,
            mamba_conv_bias=self.mamba_conv_bias,
            mamba_proj_bias=self.mamba_proj_bias,
        )

    def create_and_check_model(self, config, input_ids, input_mask, token_labels):
        model = GraniteMoeHybridModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (
                self.batch_size,
                self.seq_length,
                self.hidden_size,
            ),
        )

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        input_mask,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        # config.add_cross_attention = True
        model = GraniteMoeHybridModel(config)
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
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.seq_length, self.hidden_size),
        )

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        input_mask,
        token_labels,
    ):
        model = GraniteMoeHybridForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(
            result.logits.shape,
            (self.batch_size, self.seq_length, self.vocab_size),
        )

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        input_mask,
        token_labels,
    ):
        config.is_decoder = True
        model = GraniteMoeHybridForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # needs cache to be initialized, similar to Bamba models
        past_key_values = HybridMambaAttentionDynamicCache(
            config, input_ids.shape[0], model.dtype, device=model.device
        )

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            past_key_values=past_key_values,
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
            cache_position=torch.arange(
                input_ids.shape[1], input_ids.shape[1] + next_tokens.shape[1], device=model.device
            ),
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(
            torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            token_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class GraniteMoeHybridModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
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
    fx_compatible = False

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    def setUp(self):
        self.model_tester = GraniteMoeHybridModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GraniteMoeHybridConfig, hidden_size=64)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_casual_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_attention_outputs(self):
        r"""
        Overriding the test_attention_outputs test as hybrid state space models only output
        attentions for the attention layers.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        expected_num_attentions = self.model_tester.num_hidden_layers - len(self.model_tester.attn_layer_indices)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), expected_num_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), expected_num_attentions)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), expected_num_attentions)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )

    def test_initialization(self):
        r"""
        Overriding the test_initialization test as the A_log and D params of the Bamba mixer are initialized differently
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if "A_log" in name:
                        A = torch.arange(1, config.mamba_n_heads + 1, dtype=torch.float32)
                        torch.testing.assert_close(param.data, torch.log(A), rtol=1e-5, atol=1e-5)
                    elif "D" in name:
                        D = torch.ones(config.mamba_n_heads, dtype=torch.float32)
                        torch.testing.assert_close(param.data, D, rtol=1e-5, atol=1e-5)
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def test_config_requires_mamba_or_attention_layers(self):
        """Ensure we can't create a config with disallowed layers."""
        with pytest.raises(ValueError):
            GraniteMoeHybridConfig(layer_types=["not allowed!"])


# TODO (@alex-jw-brooks) - update this one the model(s) are out
@unittest.skip(reason="GraniteMoeHybrid models are not yet released")
@require_torch_gpu
class GraniteMoeHybridIntegrationTest(unittest.TestCase):
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    @slow
    def test_tiny_model_logits(self):
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
