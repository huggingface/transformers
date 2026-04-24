# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    from transformers import DeepseekV4Model

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class DeepseekV4ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = DeepseekV4Model

    # V4-specific attributes — ``CausalLMModelTester.get_config`` pulls these by name
    # from the tester instance into the config kwargs.
    hidden_size = 64
    num_attention_heads = 4
    num_key_value_heads = 1
    head_dim = 32
    qk_rope_head_dim = 8
    q_lora_rank = 32
    o_groups = 2
    o_lora_rank = 16
    num_hidden_layers = 3
    moe_intermediate_size = 64
    n_routed_experts = 4
    n_shared_experts = 1
    num_experts_per_tok = 2
    num_hash_layers = 1
    compress_ratios = [0, 4, 128, 0]
    sliding_window = 8
    hc_mult = 2
    hc_sinkhorn_iters = 3
    hc_eps = 1.0e-6
    index_n_heads = 2
    index_head_dim = 16
    index_topk = 2
    num_nextn_predict_layers = 1
    max_position_embeddings = 64
    scoring_func = "sqrtsoftplus"
    routed_scaling_factor = 1.5
    swiglu_limit = 10.0
    rope_theta = 10000.0
    compress_rope_theta = 160000.0
    attention_bias = False
    attention_dropout = 0.0


@require_torch
class DeepseekV4ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = DeepseekV4ModelTester

    # No SequenceClassification / TokenClassification / QA heads on V4.
    def is_pipeline_test_to_skip(self, *args, **kwargs):
        return True
