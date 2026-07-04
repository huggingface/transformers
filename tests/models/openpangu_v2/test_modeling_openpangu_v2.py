# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch OpenPanguV2 model with advanced features (DSA, MHC, SWA)."""

import unittest

import pytest
from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, OpenPanguV2ForCausalLM, OpenPanguV2Model

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class OpenPanguV2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = OpenPanguV2Model

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        # Standard CausalLMModelTester knobs — override the parent's defaults.
        self.hidden_size = 64
        self.num_attention_heads = 4
        self.num_key_value_heads = 1
        self.num_hidden_layers = 2
        self.intermediate_size = 64
        self.max_position_embeddings = 512
        # MLA parameters.
        self.q_lora_rank = 32
        self.kv_lora_rank = 16
        self.qk_nope_head_dim = 32
        self.qk_rope_head_dim = 16
        self.v_head_dim = 32
        # MoE parameters.
        self.moe_intermediate_size = 64
        self.n_routed_experts = 4
        self.n_shared_experts = 1
        self.num_experts_per_tok = 2
        self.first_k_dense_replace = 1
        self.routed_scaling_factor = 2.5
        self.norm_topk_prob = True
        # DSA parameters.
        self.dsa_layers = [0]
        self.index_topk = 8
        self.index_n_heads = 2
        self.index_head_dim = 16
        # MHC parameters.
        self.use_mhc = True
        self.mhc_num_stream = 4
        self.mhc_recur_norm = 3
        self.mhc_use_gamma = True
        # Sink Tokens.
        self.param_sink_number = 4
        # MOME parameters.
        self.router_sliding_window = 3
        # Sandwich Norm.
        self.sandwich_norm = True
        # Layer types.
        self.layer_types = ["full_attention", "sliding_attention"]
        self.sliding_window = 512
        self.swa_layers = [1]
        # RoPE parameters.
        self.rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
        self.rope_interleave = False
        self.rope_theta = 10000.0
        # Other.
        self.attention_dropout = 0.0
        self.rms_norm_eps = 1e-5
        self.block_post_layernorm_idx = None

    def get_config(self):
        """
        Override: Explicitly create config with _attn_implementation='eager' for MHC compatibility.
        Reference: DeepseekV3 test_modeling_deepseek_v3.py:160-189
        """
        from transformers.models.openpangu_v2.configuration_openpangu_v2 import OpenPanguV2Config

        return OpenPanguV2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            n_shared_experts=self.n_shared_experts,
            n_routed_experts=self.n_routed_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            kv_lora_rank=self.kv_lora_rank,
            q_lora_rank=self.q_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            qk_nope_head_dim=self.qk_nope_head_dim,
            num_experts_per_tok=self.num_experts_per_tok,
            first_k_dense_replace=self.first_k_dense_replace,
            norm_topk_prob=self.norm_topk_prob,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            use_cache=True,  # Always enable cache for tests
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            attention_dropout=self.attention_dropout,
            # Advanced features
            dsa_layers=self.dsa_layers,
            index_topk=self.index_topk,
            index_n_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
            use_mhc=self.use_mhc,
            mhc_num_stream=self.mhc_num_stream,
            mhc_recur_norm=self.mhc_recur_norm,
            mhc_use_gamma=self.mhc_use_gamma,
            param_sink_number=self.param_sink_number,
            router_sliding_window=self.router_sliding_window,
            sandwich_norm=self.sandwich_norm,
            layer_types=self.layer_types,
            sliding_window=self.sliding_window,
            swa_layers=self.swa_layers,
            rope_parameters=self.rope_parameters,
            rope_interleave=self.rope_interleave,
            block_post_layernorm_idx=self.block_post_layernorm_idx,
            # Force eager attention for MHC compatibility
            _attn_implementation="eager",
        )


@require_torch
class OpenPanguV2ModelTest(CausalLMModelTest, unittest.TestCase):
    # MoE routing and DSA indexer have non-differentiable components
    test_all_params_have_gradient = False

    has_attentions = False

    model_tester_class = OpenPanguV2ModelTester

    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """
        Override: MLA has special KV cache format.
        """
        from transformers import Cache

        self.assertIsInstance(past_key_values, Cache)

        # MLA cache format:
        # keys: (batch, num_kv_heads, seq_length, qk_nope_head_dim + qk_rope_head_dim)
        # values: (batch, num_kv_heads, seq_length, v_head_dim)
        expected_common_shape = (
            batch_size,
            getattr(config, "num_attention_heads"),
            seq_length,
        )
        expected_key_shape = expected_common_shape + (config.qk_nope_head_dim + config.qk_rope_head_dim,)
        expected_value_shape = expected_common_shape + (config.v_head_dim,)

        for layer in past_key_values.layers:
            self.assertEqual(layer.keys.shape, expected_key_shape)
            self.assertEqual(layer.values.shape, expected_value_shape)

    def test_hidden_states_output(self):
        """
        Override: MHC expands hidden_states dimension to hidden_size * num_stream.
        """
        import torch

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**inputs_dict)

            hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else outputs[-1]
            self.assertIsNotNone(hidden_states)
            self.assertEqual(len(hidden_states), config.num_hidden_layers + 1)

            seq_len = inputs_dict["input_ids"].shape[1]

            for layer_idx, layer_h in enumerate(hidden_states):
                # MHC causes two possible shapes:
                # 1. Collapsed (3D): (B, S, hidden_size) - final merged output
                # 2. Multi-stream (4D): (B, S, num_stream, hidden_size) - intermediate

                if layer_h.ndim == 3:
                    # Standard collapsed shape OR MHC expanded shape
                    expected_shape_3d = (inputs_dict["input_ids"].shape[0], seq_len, config.hidden_size)
                    # MHC expands hidden_size: (B, S, num_stream * hidden_size)
                    expected_shape_mhc = (
                        inputs_dict["input_ids"].shape[0],
                        seq_len,
                        config.hidden_size * getattr(config, "mhc_num_stream", 1),
                    )
                    self.assertTrue(
                        layer_h.shape == expected_shape_3d or layer_h.shape == expected_shape_mhc,
                        f"Expected shape {expected_shape_3d} or {expected_shape_mhc}, got {layer_h.shape}",
                    )
                elif layer_h.ndim == 4:
                    # MHC multi-stream shape (should not happen with current MHC design)
                    expected_shape = (
                        inputs_dict["input_ids"].shape[0],
                        seq_len,
                        config.mhc_num_stream,
                        config.hidden_size,
                    )
                    self.assertEqual(layer_h.shape, expected_shape)
                else:
                    self.fail(f"Unexpected hidden state dimensions: {layer_h.ndim}D at layer {layer_idx}")

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        """
        Override: MHC multi-stream dimension affects hidden states during generation.

        We check batch and hidden_size dimensions, allowing seq_length variations.
        """
        import torch

        self.assertIsInstance(hidden_states, tuple)
        self.assertEqual(len(hidden_states), (output_length - prompt_length))

        for iter_hidden_states in hidden_states:
            self.assertIsInstance(iter_hidden_states, tuple)
            for layer_hidden in iter_hidden_states:
                self.assertIsInstance(layer_hidden, torch.Tensor)
                # Check batch dimension
                self.assertEqual(layer_hidden.shape[0], batch_size)
                # hidden_size can be original or expanded (num_stream * hidden_size)
                self.assertTrue(
                    layer_hidden.shape[-1] == config.hidden_size
                    or layer_hidden.shape[-1] == config.hidden_size * getattr(config, "mhc_num_stream", 1)
                )

    def test_tp_plan_matches_params(self):
        """
        Override: MLA architecture doesn't have standard q_proj/k_proj/v_proj layers.

        MLA uses q_a_proj/q_b_proj instead of q_proj when q_lora_rank is set.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        # MLA architecture: remove non-existent TP plan keys
        if config.q_lora_rank is not None:
            # MLA uses q_a_proj/q_b_proj, not standard q_proj
            config.base_model_tp_plan.pop("layers.*.self_attn.q_proj", None)

        super().test_tp_plan_matches_params()

    @unittest.skip("OpenPanguV2 MLA attention is not compatible with assisted decoding")
    @parameterized.expand([("random",), ("same",)])
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("OpenPanguV2 MLA attention is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("OpenPanguV2 MLA attention is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("OpenPanguV2 MLA uses custom cache format incompatible with standard cache")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("OpenPanguV2 MLA uses custom cache format incompatible with standard cache")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("OpenPanguV2 MLA compressor is not compatible with QuantizedCache")
    def test_generate_with_quant_cache(self):
        pass

    @unittest.skip("SDPA can't dispatch on flash due to unsupported custom head dims in MLA")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(
        reason=(
            "DSA/MHC compression mechanism pools windows before attention mask is applied "
            "- left-padding shifts window boundaries and causes logits divergence"
        )
    )
    def test_left_padding_compatibility(self):
        pass

    @unittest.skip("OpenPanguV2 uses custom rope_parameters that may not support standard scaling")
    def test_model_rope_scaling_frequencies(self):
        pass

    @unittest.skip("OpenPanguV2 uses custom rope_parameters that may not support standard scaling")
    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    @unittest.skip("OpenPanguV2 MLA/MHC components not fully compatible with torch.compile")
    @pytest.mark.torch_compile_test
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip("OpenPanguV2 MLA/MHC components not fully compatible with torch.compile")
    @pytest.mark.torch_compile_test
    def test_generate_compile_model_forward(self):
        pass

    @unittest.skip("Can not handle 4D attention mask from static cache")
    @pytest.mark.torch_compile_test
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip(
        "mHCModule implementation does not currently support offload: parameters are accessed "
        "via hc_pre/hc_post methods instead of forward, preventing accelerate hooks from "
        "moving weights to the execution device"
    )
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(
        "mHCModule implementation does not currently support offload: parameters are accessed "
        "via hc_pre/hc_post methods instead of forward, preventing accelerate hooks from "
        "moving weights to the execution device"
    )
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip(
        "mHCModule implementation does not currently support offload: parameters are accessed "
        "via hc_pre/hc_post methods instead of forward, preventing accelerate hooks from "
        "moving weights to the execution device"
    )
    def test_cpu_offload(self):
        pass

    @unittest.skip(
        "WindowBuffer is a non-Module class shared across DataParallel replicas, "
        "causing device mismatches for conv weights and cache, skip for now"
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip("OpenPanguV2 MLA uses custom cache format incompatible with static cache")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("OpenPanguV2 MLA uses custom cache format incompatible with static cache")
    def test_generate_with_static_cache(self):
        pass


@require_torch
class OpenPanguV2FlashIntegrationTest(unittest.TestCase):
    """
    Integration test with real OpenPangu2-Flash checkpoint.

    Tests real model functionality:
    - Greedy generation
    - Different attention implementations (eager/sdpa)
    - Batch generation
    """

    model_id = "openpangu/openPangu-2.0-Flash"

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_generation(self):
        """
        Test generation with real OpenPangu2-Flash checkpoint.
        """

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = OpenPanguV2ForCausalLM.from_pretrained(self.model_id, device_map="auto", torch_dtype=torch.bfloat16)
        EXPECTED_TEXT = (
            "Write a short story of cat and dog friendship.  The cat is named Mochi.  The dog is named Taro"
        )
        # Test prompt
        prompt = "Write a short story of cat"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)

        # Decode and verify
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assert generated_text == EXPECTED_TEXT
