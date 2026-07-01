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
"""Testing suite for the PyTorch OpenPanguV2 model."""

import unittest

import pytest
import torch

from transformers import is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)


if is_torch_available():
    from transformers import AutoTokenizer, OpenPanguV2ForCausalLM, OpenPanguV2Model

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class OpenPanguV2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = OpenPanguV2Model

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Basic dimensions
        self.hidden_size = 64
        self.num_attention_heads = 4
        self.num_key_value_heads = 1
        self.num_hidden_layers = 2
        self.vocab_size = 99
        self.intermediate_size = 64
        self.max_position_embeddings = 512
        
        # MLA parameters (Multi-Head Latent Attention)
        self.q_lora_rank = 32
        self.kv_lora_rank = 16
        self.qk_rope_head_dim = 16
        self.qk_nope_head_dim = 32
        self.v_head_dim = 32
        
        # MoE parameters (Mixture of Experts)
        self.moe_intermediate_size = 64
        self.n_routed_experts = 4
        self.n_shared_experts = 1
        self.num_experts_per_tok = 2
        self.first_k_dense_replace = 1
        self.routed_scaling_factor = 1.0
        self.norm_topk_prob = True
        
        # DSA parameters (Dynamic Sparse Attention) - optional, set to None for basic tests
        self.index_topk = None  # Disable DSA for basic tests
        self.index_head_dim = None
        self.index_n_heads = None
        self.dsa_layers = None
        
        # Other parameters
        self.rope_interleave = False
        self.layer_types = ["full_attention", "full_attention"]
        self.param_sink_number = 0
        self.router_sliding_window = 0
        self.sandwich_norm = False
        self.use_mhc = False
        self.mhc_num_stream = None
        self.mhc_recur_norm = None
        self.mhc_use_gamma = None
        
        # RoPE parameters
        self.rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}


@require_torch
class OpenPanguV2ModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
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