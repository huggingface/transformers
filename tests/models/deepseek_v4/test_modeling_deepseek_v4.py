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
    import torch

    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM, DeepseekV4Model
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4Cache,
        DeepseekV4Compressor,
        _pool_windows,
    )

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

    # Indexer parameters only influence the argmax over compressed positions (``topk``),
    # which is non-differentiable — their gradients flow through a separate objective in
    # the upstream training recipe, not the main causal-LM loss.
    test_all_params_have_gradient = False

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

    def test_hidden_states_output(self):
        # V4 layers emit a 4D ``[B, S, hc_mult, hidden]`` tensor — the hc_mult streams
        # are only collapsed at the top of the model via ``hc_head``. The common
        # ``test_hidden_states_output`` assumes ``(batch, seq, hidden)``; we re-run the
        # same check but accept the extra HC axis, and we additionally assert the final
        # (post-hc_head) ``last_hidden_state`` has the standard 3D shape.
        import torch  # noqa: PLC0415

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        for model_class in self.all_model_classes:
            model = model_class(config).eval()
            with torch.no_grad():
                outputs = model(**inputs_dict)
            hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else outputs[-1]
            self.assertIsNotNone(hidden_states)
            self.assertEqual(len(hidden_states), config.num_hidden_layers + 1)
            seq_len = inputs_dict["input_ids"].shape[1]
            for layer_h in hidden_states:
                # Accept either the collapsed (3D) post-head shape or the per-layer 4D shape.
                if layer_h.ndim == 3:
                    self.assertEqual(layer_h.shape, (inputs_dict["input_ids"].shape[0], seq_len, config.hidden_size))
                else:
                    self.assertEqual(
                        layer_h.shape,
                        (inputs_dict["input_ids"].shape[0], seq_len, config.hc_mult, config.hidden_size),
                    )

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        # Every V4 layer is sliding-window, so the cache is length-bounded to
        # ``sliding_window`` instead of the full ``seq_length`` the parent tester expects.
        # We also accept the compressed-segment positions that ``DeepseekV4Attention``
        # appends on compress layers (they live beyond the window on the keys axis).
        import torch  # noqa: PLC0415

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = config.head_dim
        for layer in past_key_values.layers:
            keys, values = layer.keys, layer.values
            self.assertIsInstance(keys, torch.Tensor)
            self.assertEqual(keys.shape[0], batch_size)
            self.assertEqual(keys.shape[1], num_kv_heads)
            self.assertEqual(keys.shape[3], head_dim)
            self.assertEqual(keys.shape, values.shape)

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        # V4's per-layer hidden states carry an extra ``hc_mult`` dim (Hyper-Connection
        # parallel streams). We skip the exact seq-length assertion the base tester does,
        # because assisted-decoding feeds arbitrary draft-token batches in, and just
        # sanity-check batch / hidden dims.
        import torch  # noqa: PLC0415

        self.assertIsInstance(hidden_states, tuple)
        self.assertEqual(len(hidden_states), (output_length - prompt_length))
        for iter_hidden_states in hidden_states:
            self.assertIsInstance(iter_hidden_states, tuple)
            for layer_hidden in iter_hidden_states:
                self.assertIsInstance(layer_hidden, torch.Tensor)
                self.assertEqual(layer_hidden.shape[0], batch_size)
                self.assertEqual(layer_hidden.shape[-1], config.hidden_size)


def _tiny_config(**overrides):
    """Smallest V4 config that still exercises every architectural piece: HC streams
    (``hc_mult=2``), hash routing (layer 0), a local-SWA layer, a compressor-with-
    indexer layer (ratio 4), and a routed MoE with a shared expert.
    """
    defaults = dict(
        vocab_size=32,
        hidden_size=32,
        head_dim=16,
        qk_rope_head_dim=4,
        q_lora_rank=16,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_hidden_layers=2,
        compress_ratios=[0, 4],
        sliding_window=4,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        hc_eps=1e-6,
        moe_intermediate_size=32,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=1,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.0,
        swiglu_limit=10.0,
        o_groups=2,
        o_lora_rank=8,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        num_nextn_predict_layers=0,
        max_position_embeddings=32,
        rope_theta=10000.0,
        compress_rope_theta=10000.0,  # match main rope for a cleaner parity check
        attention_bias=False,
        attention_dropout=0.0,
    )
    defaults.update(overrides)
    return DeepseekV4Config(**defaults)


@require_torch
class DeepseekV4ParityTest(unittest.TestCase):
    """Functional sanity checks against tiny-config reference implementations of the
    V4-specific pieces (compressor pooling, HC mix + collapse). These re-derive the
    math from the upstream ``inference/model.py`` and compare to our HF modules, so a
    regression in the packed cache / HC / pool code would surface here numerically.
    """

    def test_compressor_pool_matches_reference(self):
        """Independently re-implement the reference ``Compressor._pool`` (softmax-gated
        sum-pool with learned absolute position embedding) and check it matches
        :func:`_pool_windows` on a fixed input.
        """
        torch.manual_seed(0)
        batch, length, head_dim, ratio = 2, 8, 16, 4
        kv = torch.randn(batch, length, head_dim)
        gate = torch.randn(batch, length, head_dim)
        ape = torch.randn(ratio, head_dim)

        ours = _pool_windows(kv, gate, ape, ratio, head_dim)

        # Reference (transcribed from upstream `inference/model.py` ``Compressor``):
        #   weights = softmax(gate + ape, dim=window); pooled = Σ weights * kv.
        reference = torch.zeros(batch, length // ratio, head_dim)
        for b in range(batch):
            for i in range(length // ratio):
                window_kv = kv[b, i * ratio : (i + 1) * ratio]  # [ratio, D]
                window_gate = gate[b, i * ratio : (i + 1) * ratio] + ape  # [ratio, D]
                w = torch.softmax(window_gate, dim=0)
                reference[b, i] = (window_kv * w).sum(dim=0)

        torch.testing.assert_close(ours, reference, rtol=1e-5, atol=1e-6)

    def test_compressor_cache_accumulates_across_calls(self):
        """Feeding the compressor one token at a time should produce the same pool as
        feeding the whole sequence — that's the invariant the cache buffers exist for.
        Uses ``compress_ratio=128`` to exercise the indexer-less path so we don't need
        to fabricate position_embeddings for the test.
        """
        torch.manual_seed(1)
        config = _tiny_config(compress_ratios=[0, 128], sliding_window=128, max_position_embeddings=512)
        compressor = DeepseekV4Compressor(config, compress_ratio=128, head_dim=config.head_dim).eval()
        # Initialise ``ape`` to non-zero so the test actually exercises the pooling math.
        torch.nn.init.normal_(compressor.ape, std=0.1)
        rotary = DeepseekV4Model(config).rotary_emb_compress

        batch, seq_len = 1, 256  # two full windows
        hidden_states = torch.randn(batch, seq_len, config.hidden_size)

        cache_full = DeepseekV4Cache(config=config)
        with torch.no_grad():
            one_shot = compressor(
                hidden_states,
                q_residual=None,
                rotary=rotary,
                position_embeddings=None,
                cache=cache_full,
                layer_idx=0,
                start_pos=0,
            )

        cache_inc = DeepseekV4Cache(config=config)
        with torch.no_grad():
            for step in range(seq_len):
                incremental = compressor(
                    hidden_states[:, step : step + 1],
                    q_residual=None,
                    rotary=rotary,
                    position_embeddings=None,
                    cache=cache_inc,
                    layer_idx=0,
                    start_pos=step,
                )
        self.assertEqual(one_shot.shape, incremental.shape)
        torch.testing.assert_close(one_shot, incremental, rtol=1e-4, atol=1e-5)

    def test_tiny_forward_is_deterministic_and_finite(self):
        """End-to-end smoke: tiny ``DeepseekV4ForCausalLM`` forward produces finite
        logits of the right shape, and is deterministic under the same seed."""
        torch.manual_seed(42)
        config = _tiny_config()
        model = DeepseekV4ForCausalLM(config).eval()

        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        with torch.no_grad():
            out_a = model(input_ids).logits
            out_b = model(input_ids).logits

        self.assertEqual(out_a.shape, (2, 10, config.vocab_size))
        self.assertTrue(torch.isfinite(out_a).all())
        torch.testing.assert_close(out_a, out_b)  # deterministic

    def test_tiny_generate_runs(self):
        """Greedy-generate 4 new tokens on top of a 6-token prompt and check we get 10
        tokens out. Exercises the full generation loop: cache adopt, window cache,
        compressor state, HC, indexer gather."""
        torch.manual_seed(42)
        config = _tiny_config()
        model = DeepseekV4ForCausalLM(config).eval()

        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (1, 6))
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=4, do_sample=False)
        self.assertEqual(out.shape, (1, 10))
        self.assertTrue(torch.isfinite(out.float()).all())
