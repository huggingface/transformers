# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Real-model contracts for collecting router outputs one region at a time.

These tiny, randomly initialized CPU models exercise distinct capture contracts;
they are not performance benchmarks or a replacement for model integration tests.
"""

import copy
import importlib
from contextlib import nullcontext
from unittest.mock import patch

import pytest

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    import transformers


# One representative for each distinct source of router outputs: ordinary sparse
# decoders, fused expert modules, interleaved dense/sparse layers, attention plus
# MLP routing, encoder/decoder collectors, stochastic routing, and Mamba layers.
CASES = {
    "qwen3_moe": ("Qwen3Moe", {"moe_intermediate_size": 24, "num_experts": 3}, {"router_logits": 2}),
    "granitemoe": ("GraniteMoe", {"num_local_experts": 3}, {"router_logits": 2}),
    "olmoe": ("Olmoe", {"num_experts": 3, "eos_token_id": 2}, {"router_logits": 2}),
    # Only the text components are constructed: no vision inputs or encoders.
    "aria": (
        "AriaText",
        {"moe_num_experts": 3, "moe_topk": 2, "moe_num_shared_experts": 1},
        {"router_logits": 2},
    ),
    "deepseek_ocr2": (
        "DeepseekOcr2Text",
        {
            "num_hidden_layers": 3,
            "n_routed_experts": 3,
            "n_shared_experts": 1,
            "moe_intermediate_size": 24,
            "mlp_layer_types": ["dense", "sparse", "sparse"],
        },
        {"router_logits": 2},
    ),
    "deepseek_v3": (
        "DeepseekV3",
        {
            "num_hidden_layers": 3,
            "num_key_value_heads": 4,
            "moe_intermediate_size": 24,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "n_group": 2,
            "topk_group": 1,
            "first_k_dense_replace": 1,
            "q_lora_rank": 4,
            "kv_lora_rank": 4,
            "qk_rope_head_dim": 2,
            "qk_nope_head_dim": 2,
            "v_head_dim": 4,
        },
        {"router_logits": 2},
    ),
    "qwen2_moe": (
        "Qwen2Moe",
        {
            "num_hidden_layers": 3,
            "moe_intermediate_size": 24,
            "shared_expert_intermediate_size": 24,
            "num_experts": 3,
            "mlp_only_layers": [1],
        },
        {"router_logits": 2},
    ),
    # Its existing recorder captures both gates and the attention's returned
    # routing tensor: three events per decoder, including a repeated tensor.
    "jetmoe": ("JetMoe", {"kv_channels": 4, "num_local_experts": 3}, {"router_logits": 6}),
    "switch_transformers": (
        "SwitchTransformers",
        {
            "d_model": 16,
            "d_ff": 24,
            "d_kv": 4,
            "num_heads": 4,
            "num_layers": 2,
            "num_decoder_layers": 2,
            "num_sparse_encoder_layers": 1,
            "num_sparse_decoder_layers": 1,
            "num_experts": 3,
            "expert_capacity": 16,
            "dropout_rate": 0.0,
            "decoder_start_token_id": 0,
        },
        {"encoder_router_logits": 1, "decoder_router_logits": 1},
    ),
    "phimoe": ("Phimoe", {"num_local_experts": 3, "input_jitter_noise": 0.01}, {"router_logits": 2}),
    "jamba": (
        "Jamba",
        {
            "num_experts": 3,
            "expert_layer_period": 1,
            "expert_layer_offset": 0,
            "attn_layer_period": 2,
            "attn_layer_offset": 1,
            "use_mamba_kernels": False,
            "mamba_d_state": 4,
            "mamba_d_conv": 2,
            "mamba_expand": 1,
            "mamba_dt_rank": 2,
        },
        {"router_logits": 2},
    ),
}


def make_model(family, **config_overrides):
    prefix, extra, _ = CASES[family]
    config_kwargs = {
        "vocab_size": 31,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_experts_per_tok": 2,
        "max_position_embeddings": 32,
        "pad_token_id": 0,
        "use_cache": False,
    }
    if family == "switch_transformers":
        config_kwargs = {"vocab_size": 31, "pad_token_id": 0, "use_cache": False}
    config_kwargs.update(extra)
    config_kwargs.update(config_overrides)
    config = getattr(transformers, prefix + "Config")(**config_kwargs)
    config._attn_implementation = "eager"
    suffix = "ForConditionalGeneration" if family == "switch_transformers" else "ForCausalLM"
    if family == "deepseek_ocr2":
        suffix = "Model"  # This public text backbone has no separate causal-LM head.
    model_class = prefix + suffix
    torch.manual_seed(13)
    return getattr(transformers, model_class)(config).cpu().train()


def model_inputs(family):
    input_ids = torch.tensor([[2, 5, 9, 7, 3], [4, 8, 6, 3, 2]])
    inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": input_ids.clone()}
    if family == "switch_transformers":
        inputs["decoder_input_ids"] = input_ids.flip(1)
    if family == "deepseek_ocr2":
        inputs.pop("labels")
    return inputs


def capture_context(family, scoped):
    if scoped:
        return nullcontext()
    module = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    return patch.object(module, "_scope_router_output_collection", nullcontext)


def assert_same(actual, expected):
    """No rounding tolerance: the eager scope must not change any arithmetic."""
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_same(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_same(left, right)
    else:
        assert actual == expected


def router_gradient_objective(outputs, family):
    if family == "switch_transformers":
        return outputs.encoder_aux_loss + outputs.decoder_aux_loss
    if family in ("deepseek_v3", "aria", "deepseek_ocr2"):
        # These models expose scores but no model auxiliary loss. A test-only
        # scalar checks their gradient connection; the optimizer still uses the
        # separate model-output objective below.
        assert getattr(outputs, "aux_loss", None) is None
        return sum(scores.square().mean() for scores in outputs.router_logits)
    return outputs.aux_loss


@require_torch
@pytest.mark.parametrize("family", CASES)
@pytest.mark.parametrize("hidden_states", [False, True, [1]], ids=["router_only", "all_outputs", "selected_hidden"])
def test_router_collection_preserves_optional_outputs(family, hidden_states):
    model = make_model(family)
    inputs = model_inputs(family)
    outputs = []
    rng_states = []
    for scoped in (False, True):
        torch.manual_seed(29)
        with capture_context(family, scoped):
            outputs.append(
                model(
                    **inputs,
                    output_router_logits=True,
                    output_hidden_states=hidden_states,
                    output_attentions=hidden_states is True,
                )
            )
        rng_states.append(torch.get_rng_state())
    assert_same(outputs[1], outputs[0])
    assert_same(rng_states[1], rng_states[0])
    for name, count in CASES[family][2].items():
        assert len(outputs[1][name]) == count
        assert all(router.requires_grad for router in outputs[1][name])
    if hidden_states is True:
        attention_keys = [name for name in outputs[1] if name.endswith("attentions")]
        assert attention_keys
        for key in attention_keys:
            assert outputs[1][key]
            assert all(value is not None for value in outputs[1][key])
    if hidden_states:
        hidden_keys = [name for name in outputs[1] if name.endswith("hidden_states")]
        assert hidden_keys
        for key in hidden_keys:
            if isinstance(hidden_states, list):
                assert outputs[1][key][0] is None
                assert outputs[1][key][1] is not None
            else:
                assert all(value is not None for value in outputs[1][key])


@require_torch
@pytest.mark.parametrize(
    "family,checkpoint",
    [
        pytest.param(family, checkpoint, id=f"{family}-{'no_checkpoint' if checkpoint is None else 'non_reentrant'}")
        for family in CASES
        for checkpoint in (None, False)
        if family != "jetmoe" or checkpoint is None
    ],
)
def test_router_collection_preserves_router_gradients_and_update(family, checkpoint):
    check_gradients_and_update(family, checkpoint, require_aux_gradients=True)


@require_torch
@pytest.mark.parametrize("family", [family for family in CASES if family != "jetmoe"])
def test_router_collection_preserves_reentrant_checkpoint_behavior(family):
    # Reentrant checkpointing captures router outputs under no_grad in the
    # existing implementation. This checks parity, not auxiliary-loss support;
    # the non-reentrant cases above require nonzero auxiliary gradients.
    check_gradients_and_update(family, checkpoint=True, require_aux_gradients=False)


def check_gradients_and_update(family, checkpoint, require_aux_gradients):
    original = make_model(family)
    model = copy.deepcopy(original)
    inputs = model_inputs(family)
    results = []
    for current, scoped in ((original, False), (model, True)):
        if checkpoint is not None:
            current.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
            current.enable_input_require_grads()
        params = tuple(current.parameters())
        torch.manual_seed(29)
        with capture_context(family, scoped):
            outputs = current(**inputs, output_router_logits=True)
            aux = router_gradient_objective(outputs, family)
            aux_grads = None
            if require_aux_gradients or aux.requires_grad:
                aux_grads = torch.autograd.grad(aux, params, retain_graph=True, allow_unused=True)
                assert any(grad is not None and torch.count_nonzero(grad) for grad in aux_grads)
            if family == "deepseek_ocr2":
                # The public text backbone returns hidden states, not LM loss.
                # This test-only scalar checks backbone gradients and updates;
                # it does not qualify multimodal training or invent a task loss.
                outputs.last_hidden_state[..., 0].mean().backward()
            else:
                outputs.loss.backward()
        grads = tuple(None if parameter.grad is None else parameter.grad.clone() for parameter in params)
        assert any(grad is not None and torch.count_nonzero(grad) for grad in grads)
        assert all(grad is None or torch.isfinite(grad).all() for grad in grads)
        before = tuple(parameter.detach().clone() for parameter in params)
        torch.optim.SGD(params, lr=0.1).step()
        after = tuple(parameter.detach().clone() for parameter in params)
        assert any(not torch.equal(left, right) for left, right in zip(before, after))
        results.append((outputs, aux.requires_grad, aux_grads, grads, after, torch.get_rng_state()))
    assert_same(results[1], results[0])


@require_torch
@pytest.mark.torch_compile_test
def test_regional_compilation_reuses_graph_and_preserves_router_gradients():
    torch.compiler.reset()
    try:
        limit = getattr(torch._dynamo.config, "recompile_limit", None)
        if limit is None:
            limit = torch._dynamo.config.cache_size_limit
        model = make_model("qwen3_moe", num_hidden_layers=limit + 4)
        inputs = model_inputs("qwen3_moe")
        named_parameters = tuple(model.named_parameters())
        params = tuple(parameter for _, parameter in named_parameters)

        def forward_backward():
            model.zero_grad(set_to_none=True)
            outputs = model(**inputs, output_router_logits=True)
            assert len(outputs.router_logits) == model.config.num_hidden_layers
            aux_grads = torch.autograd.grad(outputs.aux_loss, params, allow_unused=True, retain_graph=True)
            router_grads = [
                grad for (name, _), grad in zip(named_parameters, aux_grads) if name.endswith("mlp.gate.weight")
            ]
            assert len(router_grads) == model.config.num_hidden_layers
            assert all(
                grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) for grad in router_grads
            )
            outputs.loss.backward()
            grads = tuple(None if parameter.grad is None else parameter.grad.clone() for parameter in params)
            assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
            return outputs, aux_grads, grads

        with capture_context("qwen3_moe", scoped=False):
            reference = forward_backward()  # Also installs the real capture hooks before tracing.

        graphs, calls = [], []

        def backend(graph, example_inputs):
            graphs.append(graph)

            def execute(*args):
                calls.append(None)
                return graph.forward(*args)

            return execute

        for layer in model.model.layers:
            layer.compile(backend=backend)
        # The normal model loop must supply the scope: adding one here would
        # conceal a missing production integration. This backend isolates graph
        # reuse from kernel numerics; it does not qualify Inductor performance.
        for _ in range(2):
            assert_same(forward_backward(), reference)
        assert len(graphs) == 1, "router collection specialized the graph for successive decoder regions"
        assert len(calls) == 2 * len(model.model.layers), "a decoder silently fell back to eager execution"
    finally:
        torch.compiler.reset()


@require_torch
@pytest.mark.torch_compile_test
def test_whole_model_compilation_preserves_router_capture_and_auxiliary_gradients():
    from torch._dynamo.testing import CompileCounterWithBackend

    # Exercise the actual Dynamo branch, not a mocked compilation predicate.
    # The eager backend tests full-graph capture compatibility; this deliberately
    # does not claim AOTAutograd/Inductor backward coverage on CPU expert kernels.
    results = []
    try:
        for scoped in (False, True):
            torch._dynamo.reset()
            model = make_model("qwen3_moe")
            inputs = model_inputs("qwen3_moe")
            params = tuple(model.parameters())
            with capture_context("qwen3_moe", scoped):
                model(**inputs, output_router_logits=True)  # Install capture hooks before tracing.
                counter = CompileCounterWithBackend("eager")
                compiled = torch.compile(model, backend=counter, fullgraph=True)
                outputs = compiled(**inputs, output_router_logits=True)
                assert_same(compiled(**inputs, output_router_logits=True), outputs)
                assert counter.frame_count == 1
                assert counter.op_count > 0
                assert len(outputs.router_logits) == 2
                aux_grads = torch.autograd.grad(outputs.aux_loss, params, allow_unused=True, retain_graph=True)
                assert any(grad is not None and torch.count_nonzero(grad) for grad in aux_grads)
                outputs.loss.backward()
                grads = tuple(None if parameter.grad is None else parameter.grad.clone() for parameter in params)
                results.append((outputs, aux_grads, grads))
        assert_same(results[1], results[0])
    finally:
        torch._dynamo.reset()
