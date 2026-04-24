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

    def __init__(self, parent, **kwargs):
        # ``CausalLMModelTester.__init__`` assigns a fixed set of attributes from its
        # keyword defaults (``hidden_size``, ``num_attention_heads`` and friends); those
        # overwrite any class-level attributes of the same name. Pass V4 defaults through
        # ``kwargs`` so the tester instance reflects V4's shape.
        kwargs.setdefault("hidden_size", 64)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 1)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_experts_per_tok", 2)
        kwargs.setdefault("moe_intermediate_size", 64)
        kwargs.setdefault("max_position_embeddings", 64)
        super().__init__(parent, **kwargs)
        # V4-only attributes that ``CausalLMModelTester.get_config`` will pull by name.
        self.head_dim = 32
        self.qk_rope_head_dim = 8
        self.q_lora_rank = 32
        self.o_groups = 2
        self.o_lora_rank = 16
        self.n_routed_experts = 4
        self.n_shared_experts = 1
        # ``num_hash_layers=0`` so the ``inputs_embeds``-only generation tests in
        # ``CausalLMModelTest`` can exercise the model without running into the hash
        # router's ``input_ids`` requirement. A dedicated test covers the hash path.
        self.num_hash_layers = 0
        self.compress_ratios = [0, 4]
        self.sliding_window = 8
        self.hc_mult = 2
        self.hc_sinkhorn_iters = 3
        self.hc_eps = 1.0e-6
        self.index_n_heads = 2
        self.index_head_dim = 16
        self.index_topk = 2
        self.num_nextn_predict_layers = 0
        self.scoring_func = "sqrtsoftplus"
        self.routed_scaling_factor = 1.5
        self.swiglu_limit = 10.0
        self.rope_theta = 10000.0
        self.compress_rope_theta = 160000.0
        self.attention_bias = False
        self.attention_dropout = 0.0


@require_torch
class DeepseekV4ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = DeepseekV4ModelTester

    # No SequenceClassification / TokenClassification / QA heads on V4.
    def is_pipeline_test_to_skip(self, *args, **kwargs):
        return True

    def _check_attentions_for_generate(
        self, batch_size, attentions, prompt_length, output_length, config, decoder_past_key_values
    ):
        # V4 layers with a Compressor attend to extra pooled positions, so the KV
        # length varies per layer. We only check the shape invariants: batched, same
        # number-of-heads and query-length; the KV-length axis may differ across layers.
        import torch  # noqa: PLC0415

        self.assertIsInstance(attentions, tuple)
        self.assertEqual(len(attentions), (output_length - prompt_length))
        for _, iter_attentions in enumerate(attentions):
            self.assertIsInstance(iter_attentions, tuple)
            for layer_attention in iter_attentions:
                self.assertIsInstance(layer_attention, torch.Tensor)
                self.assertEqual(layer_attention.shape[0], batch_size)
                self.assertEqual(layer_attention.shape[1], config.num_attention_heads)

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        # V4's per-layer hidden states carry an extra ``hc_mult`` dim (Hyper-Connection
        # parallel streams). Collapse it to the usual ``(batch, seq, hidden)`` convention
        # before running the standard shape checks from the generation tester.
        import torch  # noqa: PLC0415

        self.assertIsInstance(hidden_states, tuple)
        self.assertEqual(len(hidden_states), (output_length - prompt_length))
        for generated_length, iter_hidden_states in enumerate(hidden_states):
            self.assertIsInstance(iter_hidden_states, tuple)
            if use_cache and generated_length > 0:
                model_input_length = 1
            else:
                model_input_length = prompt_length + generated_length
            for layer_hidden in iter_hidden_states:
                self.assertIsInstance(layer_hidden, torch.Tensor)
                self.assertEqual(layer_hidden.shape[0], batch_size)
                self.assertEqual(layer_hidden.shape[1], model_input_length)
                self.assertEqual(layer_hidden.shape[-1], config.hidden_size)
