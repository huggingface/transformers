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
"""Testing suite for the PyTorch K2 Horizon model."""

import copy
import itertools
import math
import tempfile
import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import AutoConfig, K2HorizonConfig, is_torch_available
from transformers.testing_utils import require_accelerate, require_torch, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, K2HorizonForCausalLM, K2HorizonModel
    from transformers.models.k2_horizon.modeling_k2_horizon import (
        K2HorizonAttention,
        K2HorizonMLP,
        K2HorizonMoVAAttention,
        K2HorizonRMSNorm,
        K2HorizonSparseMoeBlock,
    )


class K2HorizonModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = K2HorizonModel

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 16)
        kwargs.setdefault("rope_head_dim", kwargs["head_dim"])
        kwargs.setdefault("rope_parameters", {"rope_type": "default", "rope_theta": 10000.0})
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("num_experts", 0)
        kwargs.setdefault("num_experts_per_tok", 0)
        kwargs.setdefault("moe_intermediate_size", 0)
        kwargs.setdefault("query_key_norm", False)
        super().__init__(parent, **kwargs)


@require_torch
class K2HorizonModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = K2HorizonModelTester

    def test_config_defaults_match_0p9b(self):
        config = K2HorizonConfig()
        self.assertEqual(config.vocab_size, 64256)
        self.assertEqual(config.hidden_size, 1536)
        self.assertEqual(config.intermediate_size, 5120)
        self.assertEqual(config.num_hidden_layers, 28)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.head_dim, 64)
        self.assertEqual(config.layernorm_num_groups, 1)
        self.assertFalse(config.query_key_norm)
        self.assertEqual(config.num_experts, 0)
        self.assertEqual(config.rope_parameters["rope_type"], "yarn")
        self.assertEqual(config.rope_parameters["factor"], 16.0)
        self.assertEqual(config.rope_parameters["rope_theta"], 1000000.0)
        self.assertEqual(config.rope_parameters["original_max_position_embeddings"], 8192)

    def test_config_legacy_rope_theta(self):
        # The published checkpoint stores theta at the top level, outside rope_parameters.
        config = K2HorizonConfig.from_dict(
            {
                "model_type": "k2_horizon",
                "rope_theta": 1000000.0,
                "rope_parameters": {
                    "rope_type": "yarn",
                    "factor": 16.0,
                    "attention_factor": 1.2772588722239782,
                    "beta_fast": 128.0,
                    "beta_slow": 4.0,
                    "original_max_position_embeddings": 8192,
                    "truncate": True,
                },
                "auto_map": {"AutoConfig": "configuration_k2_horizon.K2HorizonConfig"},
            }
        )
        self.assertEqual(config.rope_parameters["rope_theta"], 1000000.0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            restored = AutoConfig.from_pretrained(tmp_dir)
        self.assertIs(type(restored), K2HorizonConfig)
        self.assertEqual(restored.rope_parameters, config.rope_parameters)

    def test_output_router_logits_from_config(self):
        # The common tester is dense and has no routers, so run the check on a sparse configuration.
        self.model_tester = K2HorizonModelTester(self, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16)
        super().test_output_router_logits_from_config()

    def test_native_auto_classes_with_remote_code_metadata(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        # Published checkpoints retain these entries after native support is added.
        config.auto_map = {
            "AutoConfig": "configuration_k2_horizon.K2HorizonConfig",
            "AutoModel": "modeling_k2_horizon.K2HorizonModel",
            "AutoModelForCausalLM": "modeling_k2_horizon.K2HorizonForCausalLM",
        }
        for model_class, auto_class in ((K2HorizonModel, AutoModel), (K2HorizonForCausalLM, AutoModelForCausalLM)):
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                expected = model(**inputs)[0]
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)
                # There are no Python model files in this directory: either form must use the installed classes.
                for kwargs in ({}, {"trust_remote_code": False}):
                    with self.subTest(model_class=model_class.__name__, **kwargs):
                        loaded_config = AutoConfig.from_pretrained(tmp_dir, **kwargs)
                        self.assertIs(type(loaded_config), K2HorizonConfig)
                        reloaded = auto_class.from_pretrained(tmp_dir, **kwargs).to(torch_device).eval()
                        self.assertIs(type(reloaded), model_class)
                        with torch.no_grad():
                            actual = reloaded(**inputs)[0]
                        torch.testing.assert_close(actual, expected)

    def test_yarn_cached_forward_matches_full_sequence(self):
        config = self.model_tester.get_config()
        config.rope_parameters = {
            "rope_type": "yarn",
            "rope_theta": 1000000.0,
            "factor": 16.0,
            "attention_factor": 1.2772588722239782,
            "beta_fast": 128.0,
            "beta_slow": 4.0,
            "original_max_position_embeddings": 8192,
            "truncate": True,
        }
        input_ids = ids_tensor((2, 7), config.vocab_size)
        attention_mask = torch.ones_like(input_ids)
        # Exercise the scaled context without allocating an 8K-token attention matrix.
        position_ids = torch.arange(8190, 8197, device=torch_device).unsqueeze(0)
        for attention_implementation in ("eager", "sdpa"):
            with self.subTest(attention_implementation=attention_implementation):
                model = (
                    K2HorizonForCausalLM._from_config(config, attn_implementation=attention_implementation)
                    .to(torch_device)
                    .eval()
                )
                with torch.no_grad():
                    full = model(
                        input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                    ).logits
                    prefix = model(
                        input_ids[:, :4],
                        attention_mask=attention_mask[:, :4],
                        position_ids=position_ids[:, :4],
                        use_cache=True,
                    )
                    suffix = model(
                        input_ids[:, 4:],
                        attention_mask=attention_mask,
                        position_ids=position_ids[:, 4:],
                        past_key_values=prefix.past_key_values,
                        use_cache=True,
                    )
                torch.testing.assert_close(suffix.logits, full[:, 4:], rtol=1e-4, atol=1e-5)
                self.assertEqual(suffix.past_key_values.get_seq_length(), input_ids.shape[1])

    def test_grouped_rms_norm_groups_are_independent(self):
        norm = K2HorizonRMSNorm(hidden_size=8, n_groups=2).to(torch_device)
        hidden_states = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8]]], device=torch_device, dtype=torch.float32)
        changed_states = hidden_states.clone()
        changed_states[..., :4] *= 10
        with torch.no_grad():
            original = norm(hidden_states)
            changed = norm(changed_states)
        torch.testing.assert_close(original[..., 4:], changed[..., 4:], rtol=0, atol=0)
        torch.testing.assert_close(original[..., :4], changed[..., :4], rtol=1e-5, atol=1e-6)

    def test_rms_norm_preserves_input_dtype(self):
        norm = K2HorizonRMSNorm(hidden_size=8, n_groups=1).to(torch_device)
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.1, 1.0, 8, device=torch_device))
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                hidden_states = torch.arange(1, 9, device=torch_device, dtype=dtype).reshape(1, 1, 8)
                self.assertEqual(norm(hidden_states).dtype, dtype)


@require_torch
class K2HorizonSparseModelTest(unittest.TestCase):
    """Small numerical references do not require any published checkpoint weights."""

    def get_config(self, variant="moe", **kwargs):
        config_kwargs = {
            "vocab_size": 31,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rope_head_dim": 2,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
            "layernorm_num_groups": 2,
            "query_key_norm": True,
            "attention_gate_func": "softplus",
            "num_experts": 0 if variant == "dense" else 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 6,
            "num_shared_experts": 1,
            "mova_num_experts": 4 if variant == "mova" else 0,
            "mova_num_experts_per_tok": 2,
            "moe_gate_bias": True,
            "router_score_func": "sigmoid",
            "router_scaling_factor": 2.5,
            "norm_topk_prob": True,
            "mlp_only_layers": [],
            "decoder_sparse_step": 1,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": None,
        }
        config_kwargs.update(kwargs)
        return K2HorizonConfig(**config_kwargs)

    def fill_parameters(self, module):
        # Fixed nonuniform values make a changed expert, missing gate, or omitted scale observable.
        with torch.no_grad():
            for index, parameter in enumerate(module.parameters()):
                values = torch.arange(parameter.numel(), dtype=torch.float32, device=parameter.device)
                parameter.copy_((0.3 * torch.sin(values + index)).reshape(parameter.shape))

    def routing_reference(self, token, weight, bias, score_func, top_k, normalize, scale):
        # Sort a Python list, separately from the implementation's batched topk/dispatch.
        logits = weight.float() @ token.float()
        if score_func == "softmax":
            scores = (logits - logits.max()).exp()
            scores = scores / scores.sum()
        else:
            scores = 1 / (1 + (-logits).exp())
        choices = scores if bias is None else scores + bias.float()
        indices = sorted(range(len(scores)), key=lambda index: choices[index].item(), reverse=True)[:top_k]
        weights = scores[indices]
        if normalize:
            weights = weights / weights.sum()
        return indices, weights * scale

    def mlp_reference(self, token, mlp):
        gate = mlp.gate_proj.weight @ token
        up = mlp.up_proj.weight @ token
        return mlp.down_proj.weight @ (gate * torch.sigmoid(gate) * up)

    def test_published_sparse_configurations(self):
        # Architecture fields from the two published checkpoints; constructing a config allocates no weights.
        for num_experts, mova_experts, hidden_size, num_layers, num_heads, rope_head_dim, intermediate_size in (
            (192, 0, 6144, 61, 48, 64, 1792),
            (100, 64, 2560, 48, 32, 128, 768),
        ):
            with self.subTest(num_experts=num_experts):
                config = self.get_config(
                    num_experts=num_experts,
                    num_experts_per_tok=8,
                    mova_num_experts=mova_experts,
                    mova_num_experts_per_tok=4 if mova_experts else 0,
                    hidden_size=hidden_size,
                    num_hidden_layers=num_layers,
                    num_attention_heads=num_heads,
                    num_key_value_heads=8,
                    head_dim=128,
                    rope_head_dim=rope_head_dim,
                    moe_intermediate_size=intermediate_size,
                    mlp_only_layers=[0, 1, 2],
                )
                with tempfile.TemporaryDirectory() as directory:
                    config.save_pretrained(directory)
                    restored = AutoConfig.from_pretrained(directory, trust_remote_code=False)
                self.assertIs(type(restored), K2HorizonConfig)
                for field in (
                    "num_experts",
                    "num_experts_per_tok",
                    "mova_num_experts",
                    "mova_num_experts_per_tok",
                    "moe_intermediate_size",
                    "mlp_only_layers",
                ):
                    self.assertEqual(getattr(restored, field), getattr(config, field))

    def test_invalid_sparse_configurations(self):
        for kwargs, message in (
            ({"num_experts_per_tok": 5}, "num_experts_per_tok"),
            ({"moe_intermediate_size": 0}, "moe_intermediate_size"),
            ({"mova_num_experts": 3, "mova_num_experts_per_tok": 4}, "mova_num_experts_per_tok"),
            ({"decoder_sparse_step": 0}, "decoder_sparse_step"),
            ({"router_score_func": "relu"}, "router_score_func"),
            ({"mlp_only_layers": [2]}, "mlp_only_layers"),
        ):
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(StrictDataclassClassValidationError, message):
                    self.get_config(**kwargs)

    def test_sparse_moe_matches_token_reference(self):
        hidden_states = torch.linspace(-1.3, 1.7, 48, device=torch_device).reshape(2, 3, 8)
        cases = itertools.product(("softmax", "sigmoid"), (False, True), (1, 2), (0, 2))
        for score_func, normalize, top_k, shared_experts in cases:
            with self.subTest(score_func=score_func, normalize=normalize, top_k=top_k, shared=shared_experts):
                config = self.get_config(
                    router_score_func=score_func,
                    norm_topk_prob=normalize,
                    num_experts_per_tok=top_k,
                    num_shared_experts=shared_experts,
                )
                block = K2HorizonSparseMoeBlock(config).to(torch_device).eval()
                self.fill_parameters(block)
                with torch.no_grad():
                    block.gate.bias.copy_(torch.tensor([-2.0, 2.0, 1.0, -1.0], device=torch_device))
                    actual, router_logits = block(hidden_states)
                    expected = []
                    for token in hidden_states.reshape(-1, config.hidden_size):
                        indices, weights = self.routing_reference(
                            token,
                            block.gate.weight,
                            block.gate.bias,
                            score_func,
                            top_k,
                            normalize,
                            config.router_scaling_factor,
                        )
                        # The large correction bias forces these choices without entering the mixture weights.
                        self.assertEqual(indices, [1, 2][:top_k])
                        result = sum(
                            self.mlp_reference(token, block.experts[index]) * weights[route]
                            for route, index in enumerate(indices)
                        )
                        if shared_experts:
                            result = result + self.mlp_reference(token, block.shared_experts)
                        expected.append(result)
                torch.testing.assert_close(actual, torch.stack(expected).reshape_as(hidden_states))
                torch.testing.assert_close(
                    router_logits, hidden_states.reshape(-1, config.hidden_size) @ block.gate.weight.T
                )

    def test_mova_matches_value_and_attention_reference(self):
        hidden_states = torch.linspace(-1.3, 1.7, 24, device=torch_device).reshape(1, 3, 8)
        cases = itertools.product(("softmax", "sigmoid"), (1, 2), (None, "silu", "softplus"))
        for score_func, top_k, gate_func in cases:
            with self.subTest(score_func=score_func, top_k=top_k, gate_func=gate_func):
                config = self.get_config(
                    "mova",
                    router_score_func=score_func,
                    mova_num_experts_per_tok=top_k,
                    # MoVA normalizes iff top_k > 1, independently of this MoE setting.
                    norm_topk_prob=top_k == 1,
                    attention_gate_func=gate_func,
                )
                config._attn_implementation = "eager"
                attention = K2HorizonMoVAAttention(config, layer_idx=0).to(torch_device).eval()
                self.fill_parameters(attention)
                with torch.no_grad():
                    # Uniform causal attention gives a closed-form check of the entire routed value path.
                    attention.q_proj.weight.zero_()
                    attention.k_proj.weight.zero_()
                    attention.v_router.bias.copy_(torch.tensor([-2.0, 2.0, 1.0, -1.0], device=torch_device))
                    mask = torch.full((3, 3), float("-inf"), device=torch_device).triu(1)[None, None]
                    cos = torch.ones((1, 3, config.rope_head_dim), device=torch_device)
                    actual, _ = attention(hidden_states, (cos, torch.zeros_like(cos)), mask)
                    values = []
                    for token in hidden_states[0]:
                        indices, weights = self.routing_reference(
                            token,
                            attention.v_router.weight,
                            attention.v_router.bias,
                            score_func,
                            top_k,
                            top_k > 1,
                            config.router_scaling_factor,
                        )
                        self.assertEqual(indices, [1, 2][:top_k])
                        projected = [attention.v_experts[index].weight @ token for index in indices]
                        values.append(
                            sum(value * torch.sigmoid(value) * weights[route] for route, value in enumerate(projected))
                        )
                    expected = []
                    for position, token in enumerate(hidden_states[0]):
                        attended = torch.stack(values[: position + 1]).mean(dim=0).repeat(config.num_attention_heads)
                        if gate_func is not None:
                            gate = attention.gate_proj.weight @ token
                            if gate_func == "silu":
                                gate = gate * torch.sigmoid(gate)
                            else:
                                gate = torch.log1p(torch.exp(gate * math.log(2))) / math.log(2)
                            attended = attended * gate
                        expected.append(attention.o_proj.weight @ attended)
                torch.testing.assert_close(actual, torch.stack(expected).unsqueeze(0), rtol=1e-5, atol=1e-6)
                self.assertNotIn("v_proj.weight", attention.state_dict())

    def test_sparse_layer_schedule_and_checkpoint_names(self):
        for variant in ("dense", "moe", "mova"):
            with self.subTest(variant=variant):
                config = self.get_config(variant, num_hidden_layers=6, decoder_sparse_step=2, mlp_only_layers=[1])
                model = K2HorizonModel(config)
                self.assertEqual(model._can_compile_fullgraph, variant == "dense")
                for layer_index, layer in enumerate(model.layers):
                    sparse = variant != "dense" and layer_index in (3, 5)
                    self.assertIs(type(layer.mlp), K2HorizonSparseMoeBlock if sparse else K2HorizonMLP)
                    mova = sparse and variant == "mova"
                    self.assertIs(type(layer.self_attn), K2HorizonMoVAAttention if mova else K2HorizonAttention)
                keys = model.state_dict()
                if variant != "dense":
                    self.assertIn("layers.3.mlp.gate.bias", keys)
                    for expert in range(config.num_experts):
                        for projection in ("gate_proj", "up_proj", "down_proj"):
                            self.assertIn(f"layers.3.mlp.experts.{expert}.{projection}.weight", keys)
                    self.assertIn("layers.3.mlp.shared_experts.down_proj.weight", keys)
                if variant == "mova":
                    self.assertIn("layers.3.self_attn.v_router.bias", keys)
                    self.assertIn("layers.3.self_attn.v_experts.0.weight", keys)
                    self.assertNotIn("layers.3.self_attn.v_proj.weight", keys)
                    self.assertIn("layers.1.self_attn.v_proj.weight", keys)

    def test_sparse_routing_gradients_match_token_reference(self):
        for variant, use_bias in itertools.product(("moe", "mova"), (False, True)):
            with self.subTest(variant=variant, use_bias=use_bias):
                config = self.get_config(variant, moe_gate_bias=use_bias)
                module = (
                    K2HorizonSparseMoeBlock(config)
                    if variant == "moe"
                    else K2HorizonMoVAAttention(config, layer_idx=0)
                ).to(torch_device)
                self.fill_parameters(module)
                reference = copy.deepcopy(module)
                hidden_states = torch.linspace(-1.3, 1.7, 48, device=torch_device).reshape(2, 3, 8)
                hidden_states.requires_grad_()
                reference_states = hidden_states.detach().clone().requires_grad_()
                router = reference.gate if variant == "moe" else reference.v_router
                actual = module(hidden_states)[0] if variant == "moe" else module.compute_value_states(hidden_states)
                expected = []
                for token in reference_states.reshape(-1, config.hidden_size):
                    indices, weights = self.routing_reference(
                        token,
                        router.weight,
                        router.bias,
                        config.router_score_func,
                        2,
                        True,
                        config.router_scaling_factor,
                    )
                    if variant == "moe":
                        value = sum(
                            self.mlp_reference(token, reference.experts[index]) * weight
                            for index, weight in zip(indices, weights)
                        )
                        value = value + self.mlp_reference(token, reference.shared_experts)
                    else:
                        projected = [reference.v_experts[index].weight @ token for index in indices]
                        value = sum(value * torch.sigmoid(value) * weight for value, weight in zip(projected, weights))
                    expected.append(value)
                expected = torch.stack(expected).reshape_as(actual)
                torch.testing.assert_close(actual, expected)
                actual.square().sum().backward()
                expected.square().sum().backward()
                torch.testing.assert_close(hidden_states.grad, reference_states.grad, rtol=1e-4, atol=1e-6)
                reference_parameters = dict(reference.named_parameters())
                for name, parameter in module.named_parameters():
                    expected_grad = reference_parameters[name].grad
                    if expected_grad is None:
                        self.assertIsNone(parameter.grad, name)
                    else:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        torch.testing.assert_close(parameter.grad, expected_grad, rtol=1e-4, atol=1e-6, msg=name)
                # Correction biases select experts discretely; they must not enter differentiable mixture weights.
                if use_bias:
                    self.assertIsNone((module.gate if variant == "moe" else module.v_router).bias.grad)

    def test_sparse_cache_generation_and_attention_backends(self):
        input_ids = torch.tensor([[1, 5, 7, 8, 4], [2, 9, 3, 6, 2]], device=torch_device)
        attention_mask = torch.ones_like(input_ids)
        for variant, rope_head_dim in itertools.product(("moe", "mova"), (2, 4)):
            eager_logits = None
            eager_state = None
            for implementation in ("eager", "sdpa"):
                with self.subTest(variant=variant, implementation=implementation, rope_head_dim=rope_head_dim):
                    config = self.get_config(variant, rope_head_dim=rope_head_dim)
                    model = (
                        K2HorizonForCausalLM._from_config(config, attn_implementation=implementation)
                        .to(torch_device)
                        .eval()
                    )
                    if eager_state is not None:
                        model.load_state_dict(eager_state)
                    with torch.no_grad():
                        full = model(input_ids, attention_mask=attention_mask, use_cache=False).logits
                        prefix = model(input_ids[:, :3], attention_mask=attention_mask[:, :3], use_cache=True)
                        suffix = model(
                            input_ids[:, 3:],
                            attention_mask=attention_mask,
                            past_key_values=prefix.past_key_values,
                            use_cache=True,
                        )
                        cached = model.generate(
                            input_ids, attention_mask=attention_mask, max_new_tokens=3, do_sample=False, use_cache=True
                        )
                        uncached = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=3,
                            do_sample=False,
                            use_cache=False,
                        )
                    torch.testing.assert_close(suffix.logits, full[:, 3:], rtol=1e-4, atol=1e-5)
                    self.assertEqual(suffix.past_key_values.get_seq_length(), input_ids.shape[1])
                    self.assertEqual(cached.shape, (2, 8))
                    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)
                    if implementation == "eager":
                        eager_logits, eager_state = full, model.state_dict()
                    else:
                        torch.testing.assert_close(full, eager_logits, rtol=1e-4, atol=1e-5)

    def test_sparse_native_save_load_and_router_outputs(self):
        input_ids = torch.tensor([[1, 5, 7, 8, 4]], device=torch_device)
        for variant in ("moe", "mova"):
            for model_class, auto_class in ((K2HorizonModel, AutoModel), (K2HorizonForCausalLM, AutoModelForCausalLM)):
                with self.subTest(variant=variant, model_class=model_class.__name__):
                    config = self.get_config(variant, num_hidden_layers=3, mlp_only_layers=[0])
                    config.auto_map = {
                        "AutoConfig": "configuration_k2_horizon.K2HorizonConfig",
                        "AutoModel": "modeling_k2_horizon.K2HorizonModel",
                        "AutoModelForCausalLM": "modeling_k2_horizon.K2HorizonForCausalLM",
                    }
                    model = model_class(config).to(torch_device).eval()
                    with torch.no_grad():
                        expected = model(input_ids, output_router_logits=True)
                    self.assertEqual(len(expected.router_logits), 2)
                    for logits in expected.router_logits:
                        self.assertEqual(logits.shape, (5, config.num_experts))
                    if model_class is K2HorizonForCausalLM:
                        self.assertTrue(torch.isfinite(expected.aux_loss))
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        model.save_pretrained(tmp_dir)
                        reloaded, loading_info = auto_class.from_pretrained(
                            tmp_dir,
                            trust_remote_code=False,
                            output_loading_info=True,
                        )
                        self.assertIs(type(reloaded), model_class)
                        self.assertIs(
                            type(AutoConfig.from_pretrained(tmp_dir, trust_remote_code=False)), K2HorizonConfig
                        )
                        self.assertFalse(loading_info["missing_keys"])
                        self.assertFalse(loading_info["unexpected_keys"])
                        reloaded = reloaded.to(torch_device).eval()
                        with torch.no_grad():
                            actual = reloaded(input_ids, output_router_logits=True)
                        torch.testing.assert_close(actual[0], expected[0])
                        for actual_router, expected_router in zip(actual.router_logits, expected.router_logits):
                            torch.testing.assert_close(actual_router, expected_router)

    def test_sparse_bfloat16_sharded_checkpoint_roundtrip(self):
        input_ids = torch.tensor([[1, 5, 7, 8, 4]], device=torch_device)
        for variant in ("moe", "mova"):
            with self.subTest(variant=variant):
                config = self.get_config(variant, num_experts=12, mlp_only_layers=[0])
                config.auto_map = {"AutoModelForCausalLM": "modeling_k2_horizon.K2HorizonForCausalLM"}
                model = (
                    K2HorizonForCausalLM._from_config(config, dtype=torch.bfloat16, attn_implementation="eager")
                    .to(torch_device)
                    .eval()
                )
                with torch.no_grad():
                    expected = model(input_ids).logits
                with tempfile.TemporaryDirectory() as directory:
                    # Twelve individually named experts span multiple checkpoint shards.
                    model.save_pretrained(directory, max_shard_size="5KB")
                    loaded, info = AutoModelForCausalLM.from_pretrained(
                        directory,
                        dtype=torch.bfloat16,
                        trust_remote_code=False,
                        attn_implementation="eager",
                        output_loading_info=True,
                    )
                for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
                    self.assertFalse(info[key], f"{key}: {info[key]}")
                loaded = loaded.to(torch_device).eval()
                expected_state = model.state_dict()
                self.assertEqual(set(loaded.state_dict()), set(expected_state))
                for name, tensor in loaded.state_dict().items():
                    self.assertEqual(tensor.dtype, expected_state[name].dtype, name)
                    torch.testing.assert_close(tensor, expected_state[name], rtol=0, atol=0, msg=name)
                with torch.no_grad():
                    torch.testing.assert_close(loaded(input_ids).logits, expected, rtol=0, atol=0)

    @require_accelerate
    def test_sparse_cpu_offload_materializes_routers(self):
        from accelerate import cpu_offload

        input_ids = torch.tensor([[1, 5, 7, 8, 4]])
        for variant in ("moe", "mova"):
            with self.subTest(variant=variant):
                config = self.get_config(variant)
                model = K2HorizonForCausalLM._from_config(config, attn_implementation="eager").eval()
                with torch.no_grad():
                    expected = model(input_ids).logits
                # CPU execution still exercises Accelerate's meta-device hooks without requiring a GPU.
                cpu_offload(model, execution_device=torch.device("cpu"))
                self.assertEqual(model.model.layers[0].mlp.gate.weight.device.type, "meta")
                if variant == "mova":
                    self.assertEqual(model.model.layers[0].self_attn.v_router.weight.device.type, "meta")
                with torch.no_grad():
                    torch.testing.assert_close(model(input_ids).logits, expected)


@require_torch
@slow
class K2HorizonIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    def test_0p9b_native_chat_generation(self):
        model_id = "IFM/K2-Horizon-0.9B"
        revision = "40b7742db0e799df5d0bd3fb4fd5524511b1006d"
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, dtype=torch.float32, attn_implementation="eager"
            )
            .to(torch_device)
            .eval()
        )
        self.assertIs(type(model), K2HorizonForCausalLM)
        self.assertIs(type(model.config), K2HorizonConfig)

        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 2 + 2?"}],
            add_generation_prompt=True,
            reasoning_effort="low",
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(torch_device)
        inputs.pop("token_type_ids", None)
        # Captured from the unmodified checkpoint implementation with eager float32 attention.
        expected_logits = torch.tensor(
            [
                2.0341561,
                5.7932320,
                1.5396051,
                8.7653399,
                7.2184401,
                8.9973259,
                5.8357220,
                1.5522604,
                4.1253591,
                8.6460686,
            ],
            device=torch_device,
        )
        logits = model(**inputs).logits
        torch.testing.assert_close(logits[0, -1, :10], expected_logits, rtol=1e-4, atol=1e-4)
        cached = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True)
        uncached = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=False)
        self.assertEqual(cached[0, inputs.input_ids.shape[1] :].tolist(), [9300, 30838, 321, 3696, 394, 222, 19, 1085])
        torch.testing.assert_close(cached, uncached, rtol=0, atol=0)
