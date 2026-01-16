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

import inspect
import tempfile
import unittest

import pytest
from parameterized import parameterized
from pytest import mark

from transformers import (
    AutoTokenizer,
    DataCollatorWithFlattening,
    GraniteMoeHybridConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...models.bamba.test_modeling_bamba import BambaModelTester
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        GraniteMoeHybridForCausalLM,
        GraniteMoeHybridModel,
    )
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import HybridMambaAttentionDynamicCache


class GraniteMoeHybridModelTester(BambaModelTester):
    config_class = GraniteMoeHybridConfig
    if is_torch_available():
        model_class = GraniteMoeHybridModel
        for_causal_lm_class = GraniteMoeHybridForCausalLM

    def __init__(
        self,
        parent,
        use_cache=False,
        shared_intermediate_size=174,
        layer_types=None,
    ):
        super().__init__(parent)
        self.shared_intermediate_size = shared_intermediate_size
        self.layer_types = layer_types
        self.use_cache = use_cache

    def _update_layer_configs(self):
        super()._update_layer_configs()
        # GraniteMoeHybrid uses layer_types instead of attn_layer_indices
        self.layer_types = ["mamba"] * self.num_hidden_layers
        for idx in self.attn_layer_indices:
            self.layer_types[idx] = "attention"

    def get_config(self):
        return super().get_config(
            shared_intermediate_size=self.shared_intermediate_size,
            layer_types=self.layer_types,
        )


@require_torch
class GraniteMoeHybridModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
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
            "text-embedding": GraniteMoeHybridModel,
            "text-generation": GraniteMoeHybridForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]
    test_torch_exportable = False  # uses custom kernels by default, not compatible with torch.export

    def _check_caches_are_equal(
        self, cache1: HybridMambaAttentionDynamicCache, cache2: HybridMambaAttentionDynamicCache
    ):
        if not isinstance(cache1, HybridMambaAttentionDynamicCache) or not isinstance(
            cache2, HybridMambaAttentionDynamicCache
        ):
            raise ValueError("The wrong cache is being used!")

        if not len(cache1) == len(cache2):
            raise ValueError("Both caches do not have the same number of layers.")

        num_layers = len(cache1)
        for idx in range(num_layers):
            torch.testing.assert_close(cache1.key_cache[idx], cache2.key_cache[idx])
            torch.testing.assert_close(cache1.value_cache[idx], cache2.value_cache[idx])
            torch.testing.assert_close(cache1.conv_states[idx], cache2.conv_states[idx])
            torch.testing.assert_close(cache1.ssm_states[idx], cache2.ssm_states[idx])

    def setUp(self):
        self.model_tester = self.model_tester_class(self)
        self.config_tester = ConfigTester(self, config_class=self.model_tester.config_class, hidden_size=64)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_attention_outputs(self):
        r"""
        Overriding the test_attention_outputs test as the Bamba model outputs attention only for its attention layers
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
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
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

    def test_batching_equivalence(self):
        # need to disable the tril input mask
        orig = self.model_tester.use_input_mask
        self.model_tester.use_input_mask = False
        super().test_batching_equivalence()
        self.model_tester.use_input_mask = orig

    @pytest.mark.generate
    def test_left_padding_compatibility(self):
        # TODO: document why a random attention mask causes this test to fail, but a full mask doesn't
        unpadded_custom_inputs = {"attention_mask": None}
        super().test_left_padding_compatibility(unpadded_custom_inputs=unpadded_custom_inputs)

    @unittest.skip(
        "Bamba requires additionally specifying position_ids, seq_idx, and FlashAttentionKwargs for padding-free training."
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        "Bamba requires additionally specifying position_ids, seq_idx, and FlashAttentionKwargs for padding-free training."
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids_and_fa_kwargs(self):
        pass

    @require_flash_attn
    @require_torch_accelerator
    @mark.flash_attn_test
    @slow
    @unittest.skip(
        "NotImplementedError: seq_idx support requires fast path support. Please install mamba_ssm and causal_conv1d"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids_seq_idx_and_fa_kwargs(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        max_new_tokens = 30

        for model_class in self.all_generative_model_classes:
            if not model_class._supports_flash_attn:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            if 0 not in inputs_dict.get("attention_mask", []) or "attention_mask" not in inputs_dict:
                self.skipTest("Model dummy inputs should contain padding in their attention mask")

            dummy_input = inputs_dict[model_class.main_input_name]
            if dummy_input.dtype in [torch.float32, torch.bfloat16]:
                dummy_input = dummy_input.to(torch.float16)

            # make sure that all models have enough positions for generation
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + dummy_input.shape[1] + 1

            model = model_class(config)
            if "position_ids" not in inspect.signature(model.forward).parameters:
                self.skipTest("Model does not support position_ids")

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # ensure left padding, to adapt for some models
                if 0 in inputs_dict["attention_mask"][:, -1]:
                    inputs_dict["attention_mask"] = inputs_dict["attention_mask"].flip(1)
                dummy_attention_mask = inputs_dict["attention_mask"]
                inputs_dict["input_ids"][~dummy_attention_mask.bool()] = config.get_text_config().pad_token_id
                # Ensure inputs_dict also has labels in it, as their presence/absence can induce
                # dtype conversions. This also lets us compare losses.
                labels = inputs_dict["input_ids"].clone()
                # Mask padding tokens
                labels[~dummy_attention_mask.bool()] = -100
                # Also need to mask the first non-trivial token to match the padding-free batch.
                first_nonneg_idx = (labels >= 0).int().argmax(dim=1)
                labels[torch.arange(labels.size(0), device=labels.device), first_nonneg_idx] = -100
                inputs_dict["labels"] = labels

                model = (
                    model_class.from_pretrained(
                        tmpdirname,
                        dtype=torch.float16,
                        attn_implementation="flash_attention_2",
                    )
                    .to(torch_device)
                    .eval()
                )

                # flatten
                features = [
                    {"input_ids": i[a.bool()].tolist()}
                    for i, a in zip(inputs_dict["input_ids"], inputs_dict["attention_mask"])
                ]

                # add position_ids + fa_kwargs + seq_idx
                data_collator = DataCollatorWithFlattening(
                    return_tensors="pt", return_seq_idx=True, return_flash_attn_kwargs=True
                )
                batch = data_collator(features)
                batch_accelerator = {k: t.to(torch_device) if torch.is_tensor(t) else t for k, t in batch.items()}

                res_padded = model(**inputs_dict)
                res_padfree = model(**batch_accelerator)

                logits_padded = res_padded.logits[inputs_dict["attention_mask"].bool()]
                logits_padfree = res_padfree.logits[0]

                torch.testing.assert_close(logits_padded.argmax(-1), logits_padfree.argmax(-1), rtol=0, atol=0)
                # acceptable numerical instability
                tol = torch.finfo(torch.float16).eps
                torch.testing.assert_close(logits_padded, logits_padfree, rtol=tol, atol=tol)

                loss_padded = res_padded.loss
                loss_padfree = res_padfree.loss
                torch.testing.assert_close(loss_padded, loss_padfree)

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        self.assertIsInstance(past_key_values, HybridMambaAttentionDynamicCache)

        # (batch, kv heads, seq_length, head_dim)
        num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        attention_shape = (batch_size, num_heads, seq_length, head_dim)

        conv_shape = (
            batch_size,
            config.mamba_expand * config.hidden_size + 2 * config.mamba_n_groups * config.mamba_d_state,
            config.mamba_d_conv,
        )
        ssm_shape = (batch_size, config.mamba_n_heads, config.mamba_d_head, config.mamba_d_state)

        self.assertTrue(config.num_hidden_layers, len(past_key_values))

        for idx in range(len(past_key_values)):
            if config.layers_block_type[idx] == "mamba":
                self.assertEqual(past_key_values.conv_states[idx].shape, conv_shape)
                self.assertEqual(past_key_values.ssm_states[idx].shape, ssm_shape)
            else:
                self.assertEqual(past_key_values.key_cache[idx].shape, attention_shape)
                self.assertEqual(past_key_values.value_cache[idx].shape, attention_shape)

    def test_config_requires_mamba_or_attention_layers(self):
        """Ensure we can't create a config with disallowed layers."""
        with pytest.raises(ValueError):
            GraniteMoeHybridConfig(layer_types=["not allowed!"])


@require_torch_accelerator
class GraniteMoeHybridIntegrationTest(unittest.TestCase):
    @slow
    @parameterized.expand([("cpu",)])  # runners crash with `cuda`, prob they have mamba kernels installed
    def test_model_logits(self, device):
        input_ids = [31390, 631, 4162, 30, 322, 25342, 432, 1875, 43826, 10066, 688, 225]

        model = GraniteMoeHybridForCausalLM.from_pretrained("ibm-granite/granite-4.0-h-tiny", device_map=device)

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(device))

        # fmt: off
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([
            [-0.3543, -1.0066, -0.5338, -0.8816, -0.7438,  0.0500, -1.3644, -0.0742, -1.7746, -1.6326, -1.4802, -0.4961]
        ], device=device)

        torch.testing.assert_close(EXPECTED_MEAN, out.logits.float().mean(-1), rtol=1e-2, atol=1e-2)

        # slicing logits[0, 0, 0:15]
        EXPECTED_SLICE = torch.tensor([
            [6.5938,  7.2500,  1.6484,  5.2188,  3.5781,  2.5469,  6.1250,  5.1875, 9.5000,  4.6875,  4.7188, 10.7500, 10.3125,  7.8438,  5.5312]
        ], device=device)
        # fmt: on

        self.assertTrue(
            torch.allclose(
                EXPECTED_SLICE,
                out.logits[0, 0, :15].float(),
                atol=1e-3,
                rtol=1e-3,
            )
        )

    @slow
    @parameterized.expand([("cpu",)])
    def test_model_generation(self, device):
        EXPECTED_TEXT_COMPLETION = "Simply put, the theory of relativity states that 1) the laws of physics are the same for all observers in uniform motion relative"
        prompt = "Simply put, the theory of relativity states that "
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.0-h-tiny")
        model = GraniteMoeHybridForCausalLM.from_pretrained("ibm-granite/granite-4.0-h-tiny", device_map=device)
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # greedy generation outputs
        generated_ids = model.generate(**model_inputs, max_new_tokens=16, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
