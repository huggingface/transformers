# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Olmo3_5Hybrid model."""

import tempfile
import unittest

from parameterized import parameterized

from transformers import Olmo3_5HybridConfig, is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_multi_gpu,
    slow,
    torch_device,
)
from transformers.utils.import_utils import is_flash_linear_attention_available

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    _test_eager_matches_sdpa_inference,
)


if is_torch_available():
    import torch

    from transformers import (
        Cache,
        Olmo3_5HybridForCausalLM,
        Olmo3_5HybridModel,
    )
    from transformers.models.olmo3_5_hybrid.modeling_olmo3_5_hybrid import (
        Olmo3_5HybridDynamicCache,
        Olmo3_5HybridRotaryEmbedding,
    )


class Olmo3_5HybridModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Olmo3_5HybridConfig
        base_model_class = Olmo3_5HybridModel
        causal_lm_class = Olmo3_5HybridForCausalLM

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
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        scope=None,
    ):
        super().__init__(
            parent=parent,
            batch_size=batch_size,
            seq_length=seq_length,
            is_training=is_training,
            use_input_mask=use_input_mask,
            use_token_type_ids=use_token_type_ids,
            use_labels=use_labels,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            type_sequence_label_size=type_sequence_label_size,
            initializer_range=initializer_range,
            num_labels=num_labels,
            num_choices=num_choices,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            scope=scope,
        )

        # Hybrid-specific config
        self.layer_types = ["full_attention", "linear_attention"]
        self.linear_num_key_heads = num_attention_heads
        self.linear_num_value_heads = num_attention_heads
        self.linear_key_head_dim = hidden_size // num_attention_heads
        self.linear_value_head_dim = hidden_size // num_attention_heads
        self.linear_conv_kernel_dim = 4
        self.linear_use_gate = True
        self.linear_allow_neg_eigval = False

    def get_config(self):
        return Olmo3_5HybridConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            layer_types=self.layer_types,
            linear_num_key_heads=self.linear_num_key_heads,
            linear_num_value_heads=self.linear_num_value_heads,
            linear_key_head_dim=self.linear_key_head_dim,
            linear_value_head_dim=self.linear_value_head_dim,
            linear_conv_kernel_dim=self.linear_conv_kernel_dim,
            linear_use_gate=self.linear_use_gate,
            linear_allow_neg_eigval=self.linear_allow_neg_eigval,
        )


@require_torch
class Olmo3_5HybridModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (Olmo3_5HybridModel, Olmo3_5HybridForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": Olmo3_5HybridModel, "text-generation": Olmo3_5HybridForCausalLM}
        if is_torch_available()
        else {}
    )
    model_tester_class = Olmo3_5HybridModelTester
    rotary_embedding_layer = Olmo3_5HybridRotaryEmbedding if is_torch_available() else None

    # === Cache helper methods (same pattern as Qwen3Next) ===
    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """Olmo3_5Hybrid has a special Cache as it alternates with gated deltanet layers"""
        self.assertIsInstance(past_key_values, Olmo3_5HybridDynamicCache)

        num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        expected_shape = (batch_size, num_heads, seq_length, head_dim)

        attention_layer_indices = past_key_values.transformer_layers
        self.assertListEqual(
            [past_key_values.key_cache[idx].shape for idx in attention_layer_indices],
            [expected_shape] * len(attention_layer_indices),
        )
        self.assertListEqual(
            [past_key_values.value_cache[idx].shape for idx in attention_layer_indices],
            [expected_shape] * len(attention_layer_indices),
        )

    def _check_caches_are_equal(self, cache1: Cache, cache2: Cache):
        """Olmo3_5Hybrid has a special Cache as it alternates with gated deltanet layers"""
        if not len(cache1) == len(cache2):
            raise ValueError("Both caches do not have the same number of layers.")

        num_layers = len(cache1)
        for idx in range(num_layers):
            if cache1.key_cache[idx] is not None:
                torch.testing.assert_close(cache1.key_cache[idx], cache2.key_cache[idx])
                torch.testing.assert_close(cache1.value_cache[idx], cache2.value_cache[idx])

    # === Override test_attention_outputs (same pattern as Qwen3Next) ===
    def test_attention_outputs(self):
        """Needs to be overwritten as Olmo3_5Hybrid alternates between attention layers and gated deltanet layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

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
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                self_attentions = outputs.attentions

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(self_attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])

    # === Override eager/sdpa test like Qwen3Next (skip fp16 due to fallback instability) ===
    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self,
        name,
        dtype,
        padding_side,
        use_attention_mask,
        output_attentions,
        enable_kernels,
    ):
        """
        Overwrite without fp16 because the slow path `torch_chunk_gated_delta_rule`
        is not robust enough in fp16 due to upscaling in fp32 and downscaling at the end
        """
        if dtype == "fp16":
            self.skipTest("Not robust in fp16")
        _test_eager_matches_sdpa_inference(
            self,
            name,
            dtype,
            padding_side,
            use_attention_mask,
            output_attentions,
            enable_kernels,
        )

    @unittest.skip("The specific cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    # Static cache tests - hybrid linear attention doesn't support static cache
    @unittest.skip("Static cache not supported for hybrid linear attention models")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("Static cache not supported for hybrid linear attention models")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("Static cache not supported for hybrid linear attention models")
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip("torch.compile fullgraph not supported with dynamic layer types")
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip("Hybrid model requires layer_types in config")
    def test_config(self):
        pass

    @unittest.skip("Hybrid model requires layer_types in config")
    def test_model_rope_scaling_frequencies(self):
        pass

    @unittest.skip("Hybrid model requires layer_types in config")
    def test_model_rope_scaling_from_config(self):
        pass

    @unittest.skip("Hybrid model has special A_log/dt_bias initialization")
    def test_can_init_all_missing_weights(self):
        pass

    @unittest.skip("Triton kernels require CUDA")
    def test_training_overfit(self):
        pass

    @unittest.skip("Hybrid model requires layer_types in config")
    def test_model_rope_scaling_from_config_1_dynamic(self, *args, **kwargs):
        pass

    def _skip_if_no_fla(self, reason="Fallback only supports bfloat16/float16, not float32"):
        if not is_flash_linear_attention_available():
            self.skipTest(reason)

    def test_resize_tokens_embeddings(self):
        self._skip_if_no_fla("Fallback path has numerical issues with small vocab sizes")
        super().test_resize_tokens_embeddings()

    def test_resize_embeddings_untied(self):
        self._skip_if_no_fla("Fallback path has numerical issues with small vocab sizes")
        super().test_resize_embeddings_untied()

    @require_torch_multi_gpu
    def test_can_use_device_map(self):
        """
        Test that this model can be dispatched on multiple gpus.
        """
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            inputs_dict = {k: v.to(0) if isinstance(v, torch.Tensor) else v for k, v in inputs_dict.items()}
            config.layer_types = ["full_attention", "linear_attention"]
            model = model_class(config).eval()

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                del model
                model = model_class.from_pretrained(
                    tmpdirname,
                    device_map={
                        "lm_head": 0,
                        "model.embed_tokens": 0,
                        "model.norm": 0,
                        "model.layers.0": 0,
                        "model.layers.1": 1,
                    },
                )

                self.assertTrue({param.device for param in model.model.layers[0].parameters()} == {torch.device(0)})
                self.assertTrue({param.device for param in model.model.layers[1].parameters()} == {torch.device(1)})

                _ = model.generate(**inputs_dict, max_new_tokens=5, min_new_tokens=5)


@require_torch
class Olmo3_5HybridIntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_logits(self):
        input_ids = [[1, 306, 4658, 278, 6593, 310, 2834, 338]]
        model = Olmo3_5HybridForCausalLM.from_pretrained("yanhong-l/olmo-3.5-test").to(
            torch_device, dtype=torch.bfloat16
        )
        out = model(torch.tensor(input_ids, device=torch_device)).logits.float()

        rtol = 3e-2
        atol = 5e-2

        expectations = Expectations(
            {
                ("cuda", 8): [
                    [
                        -3.819033145904541,
                        -3.795485734939575,
                        -2.975806951522827,
                        -2.7940011024475098,
                        -3.548236131668091,
                        -4.012556552886963,
                        -4.722480773925781,
                        -4.015453338623047,
                    ]
                ]
            }
        )
        EXPECTED_MEAN = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=rtol, atol=atol)

        expectations = Expectations(
            {
                ("cuda", 8): [
                    3.828125,
                    -0.546875,
                    -1.7578125,
                    -2.203125,
                    -2.25,
                    -2.890625,
                    -0.87109375,
                    -1.21875,
                    -1.65625,
                    -2.78125,
                    -1.2890625,
                    0.8359375,
                    -2.578125,
                    0.8125,
                    -2.1875,
                    2.921875,
                    3.671875,
                    3.5625,
                    3.109375,
                    2.78125,
                    2.703125,
                    1.7578125,
                    1.890625,
                    2.21875,
                    1.8984375,
                    -2.5,
                    -2.03125,
                    -4.03125,
                    1.2421875,
                    -1.1328125,
                ]
            }
        )
        EXPECTED_SLICE = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=rtol, atol=atol)
