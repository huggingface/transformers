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


# Representatives of distinct collection contracts, not a model support matrix:
# ordinary/mixed decoders, repeated router events, encoder/decoder outputs,
# stochastic routing, hybrid layers, and a backbone without an auxiliary loss.
CASES = {
    "qwen3_moe": ("Qwen3Moe", {"moe_intermediate_size": 24, "num_experts": 3}, {"router_logits": 2}),
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
    model_class = getattr(transformers, prefix + suffix)
    torch.manual_seed(13)
    return model_class(config).cpu().train()


def model_inputs(family):
    input_ids = torch.tensor([[2, 5, 9, 7, 3], [4, 8, 6, 3, 2]])
    inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": input_ids.clone()}
    if family == "switch_transformers":
        inputs["decoder_input_ids"] = input_ids.flip(1)
    if family == "deepseek_ocr2":
        inputs.pop("labels")
    return inputs


def without_output_scoping(family):
    """Use the existing collector without region-local buffers as the reference."""
    module = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    return patch.object(module, "_fresh_output_collection", nullcontext)


def assert_same(actual, expected):
    """The eager scope changes bookkeeping only: require exact numerical agreement."""
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def router_gradient_objective(outputs, family):
    if family == "switch_transformers":
        return outputs.encoder_aux_loss + outputs.decoder_aux_loss
    if family == "deepseek_ocr2":
        # These backbones expose router scores without an auxiliary loss.
        assert getattr(outputs, "aux_loss", None) is None
        return sum(scores.square().mean() for scores in outputs.router_logits)
    return outputs.aux_loss


@require_torch
@pytest.mark.parametrize("family", CASES)
def test_router_collection_preserves_outputs_and_gradients(family):
    """Region-local collection preserves each recorder's outputs and their learning signals."""
    reference = make_model(family)
    candidate = copy.deepcopy(reference)
    results = []
    for model, capture in ((reference, without_output_scoping(family)), (candidate, nullcontext())):
        torch.manual_seed(29)
        params = tuple(model.parameters())
        with capture:
            outputs = model(**model_inputs(family), output_router_logits=True)
            for name, count in CASES[family][2].items():
                assert isinstance(outputs[name], tuple)
                assert len(outputs[name]) == count
                assert all(router.requires_grad for router in outputs[name])
            aux_grads = torch.autograd.grad(
                router_gradient_objective(outputs, family), params, retain_graph=True, allow_unused=True
            )
            assert any(grad is not None and torch.count_nonzero(grad) for grad in aux_grads)
            # OCR2 exposes a backbone, without a task loss or a language-model head.
            loss = outputs.last_hidden_state[..., 0].mean() if family == "deepseek_ocr2" else outputs.loss
            loss.backward()
        grads = tuple(None if parameter.grad is None else parameter.grad.clone() for parameter in params)
        assert all(grad is None or torch.isfinite(grad).all() for grad in grads)
        assert any(grad is not None and torch.count_nonzero(grad) for grad in grads)
        rng_state = torch.get_rng_state() if family == "phimoe" else None
        results.append((outputs, aux_grads, grads, rng_state))
    assert_same(results[1], results[0])


@require_torch
def test_router_scoping_preserves_selected_hidden_states_and_attentions():
    """Collecting routers must not reset hidden-layer positions or lose attention outputs."""
    model = make_model("qwen3_moe")
    outputs = []
    for capture in (without_output_scoping("qwen3_moe"), nullcontext()):
        with capture:
            outputs.append(
                model(
                    **model_inputs("qwen3_moe"),
                    output_router_logits=True,
                    output_hidden_states=[1],
                    output_attentions=True,
                )
            )
    assert_same(outputs[1], outputs[0])
    assert isinstance(outputs[1].hidden_states, tuple)
    assert isinstance(outputs[1].attentions, tuple)
    assert outputs[1].hidden_states[0] is None
    assert outputs[1].hidden_states[1] is not None
    assert len(outputs[1].attentions) == model.config.num_hidden_layers
    assert all(value is not None for value in outputs[1].attentions)


@require_torch
def test_reentrant_checkpointing_preserves_existing_router_behavior():
    """Reentrant replay preserves the existing detached router outputs and task gradients."""
    model = make_model("qwen3_moe")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.enable_input_require_grads()
    results = []
    for capture in (without_output_scoping("qwen3_moe"), nullcontext()):
        model.zero_grad(set_to_none=True)
        with capture:
            outputs = model(**model_inputs("qwen3_moe"), output_router_logits=True)
            assert not any(router.requires_grad for router in outputs.router_logits)
            outputs.loss.backward()
        grads = tuple(p.grad.clone() if p.grad is not None else None for p in model.parameters())
        assert any(g is not None and torch.count_nonzero(g) for g in grads)
        results.append((outputs, grads))
    assert_same(results[1], results[0])


@require_torch
@pytest.mark.torch_compile_test
@pytest.mark.parametrize(
    "router_logits,attentions,checkpointing",
    [(True, False, False), (True, False, True), (False, True, False), (True, True, True)],
    ids=["routers", "checkpointed_routers", "attentions", "checkpointed_routers_and_attentions"],
)
def test_regional_compilation_reuses_graph_and_preserves_output_gradients(router_logits, attentions, checkpointing):
    """
    Given matching Qwen decoders with router or attention outputs requested,
    when forward and backward run with regional compilation,
    then every decoder stays compiled, graphs are reused, and outputs
    and gradients match eager execution. Checkpointed runs also replay layers.
    """
    from transformers.utils import output_capturing

    torch.compiler.reset()
    try:
        limit = getattr(torch._dynamo.config, "recompile_limit", None)
        if limit is None:
            limit = torch._dynamo.config.cache_size_limit
        model = make_model("qwen3_moe", num_hidden_layers=limit + 4)
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inputs = model_inputs("qwen3_moe")
        named_parameters = tuple(model.named_parameters())
        params = tuple(parameter for _, parameter in named_parameters)
        requested = tuple(
            key for key, enabled in (("router_logits", router_logits), ("attentions", attentions)) if enabled
        )
        graphs, calls, forward_buffers = [], [], []
        phase = "forward"

        def forward_backward():
            nonlocal phase
            calls.clear()
            forward_buffers.clear()
            model.zero_grad(set_to_none=True)
            phase = "forward"
            outputs = model(**inputs, output_router_logits=router_logits, output_attentions=attentions)
            for key in requested:
                assert isinstance(outputs[key], tuple)
                assert len(outputs[key]) == model.config.num_hidden_layers
                assert all(value.requires_grad for value in outputs[key])
            assert output_capturing._active_collector.get() is None
            buffer_lengths = [len(buffer) for buffer in forward_buffers]
            attention_grads = []
            if attentions:
                phase = "attention_backward"
                for attention, layer in zip(outputs.attentions, model.model.layers):
                    # A key-position weighting is nonconstant, unlike softmax row sums.
                    objective = (attention * torch.arange(attention.shape[-1], device=attention.device)).sum()
                    grads = torch.autograd.grad(
                        objective, (layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight), retain_graph=True
                    )
                    assert all(torch.isfinite(grad).all() and torch.count_nonzero(grad) for grad in grads)
                    attention_grads.append(grads)
            aux_grads = None
            if router_logits:
                phase = "auxiliary_backward"
                aux_grads = torch.autograd.grad(outputs.aux_loss, params, allow_unused=True, retain_graph=True)
                router_grads = [
                    grad for (name, _), grad in zip(named_parameters, aux_grads) if name.endswith("mlp.gate.weight")
                ]
                assert len(router_grads) == model.config.num_hidden_layers
                assert all(
                    grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad)
                    for grad in router_grads
                )
            phase = "total_backward"
            outputs.loss.backward()
            grads = tuple(None if parameter.grad is None else parameter.grad.clone() for parameter in params)
            assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
            assert [len(buffer) for buffer in forward_buffers] == buffer_lengths, "replay grew a forward output buffer"
            assert output_capturing._active_collector.get() is None
            return outputs, attention_grads, aux_grads, grads

        with without_output_scoping("qwen3_moe"):
            reference = forward_backward()  # Also installs the real capture hooks before tracing.

        def backend(graph, example_inputs):
            graphs.append(graph)

            def execute(*args):
                # Observe entry before checkpoint's intentional early-stop exception.
                calls.append(phase)
                collector = output_capturing._active_collector.get()
                if phase == "forward":
                    assert ("router_logits" in collector) == router_logits
                    assert ("attentions" in collector) == attentions
                    forward_buffers.extend(collector[key] for key in requested)
                else:
                    assert collector is None, "checkpoint replay reused the forward collector"
                return graph.forward(*args)

            return execute

        for layer in model.model.layers:
            layer.compile(backend=backend, fullgraph=True)
        # The normal model loop must supply the scope: adding one here would
        # conceal a missing production integration. This backend isolates graph
        # reuse from kernel numerics; it does not qualify Inductor performance.
        graph_counts = []
        for _ in range(2):
            assert_same(forward_backward(), reference)
            assert calls.count("forward") == len(model.model.layers), "a decoder silently fell back to eager execution"
            backward_phases = ["total_backward"]
            if attentions:
                backward_phases.append("attention_backward")
            if router_logits:
                backward_phases.append("auxiliary_backward")
            for backward_phase in backward_phases:
                assert (calls.count(backward_phase) > 0) == checkpointing, (
                    f"unexpected replay for {backward_phase} (checkpointing={checkpointing})"
                )
            graph_counts.append(len(graphs))
        assert graph_counts[1] == graph_counts[0], "repeated forward/backward created more graphs"
        # Replay has no active collector, so a separate reusable graph is legitimate.
        if not checkpointing:
            assert len(graphs) == 1, "output collection specialized the graph for successive decoder regions"
    finally:
        torch.compiler.reset()


@require_torch
@pytest.mark.torch_compile_test
def test_whole_model_compilation_preserves_output_capture_and_gradients():
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
            with nullcontext() if scoped else without_output_scoping("qwen3_moe"):
                model(
                    **inputs, output_router_logits=True, output_attentions=True
                )  # Install capture hooks before tracing.
                counter = CompileCounterWithBackend("eager")
                compiled = torch.compile(model, backend=counter, fullgraph=True)
                outputs = compiled(**inputs, output_router_logits=True, output_attentions=True)
                assert_same(compiled(**inputs, output_router_logits=True, output_attentions=True), outputs)
                assert counter.frame_count == 1
                assert counter.op_count > 0
                assert len(outputs.router_logits) == len(outputs.attentions) == 2
                attention_grads = []
                for attention, layer in zip(outputs.attentions, model.model.layers):
                    objective = (attention * torch.arange(attention.shape[-1], device=attention.device)).sum()
                    gradients = torch.autograd.grad(
                        objective, (layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight), retain_graph=True
                    )
                    assert all(torch.isfinite(grad).all() and torch.count_nonzero(grad) for grad in gradients)
                    attention_grads.append(gradients)
                aux_grads = torch.autograd.grad(outputs.aux_loss, params, allow_unused=True, retain_graph=True)
                assert any(grad is not None and torch.count_nonzero(grad) for grad in aux_grads)
                outputs.loss.backward()
                grads = tuple(None if parameter.grad is None else parameter.grad.clone() for parameter in params)
                results.append((outputs, attention_grads, aux_grads, grads))
        assert_same(results[1], results[0])
    finally:
        torch._dynamo.reset()


@require_torch
@pytest.mark.parametrize("decoder", [False, True])
@pytest.mark.parametrize("layerdrop", [0.5, 1.0])
def test_layerdrop_preserves_skips_empty_router_outputs_and_rng(decoder, layerdrop):
    from transformers.models.nllb_moe.modeling_nllb_moe import NllbMoeDecoder, NllbMoeEncoder

    config = transformers.NllbMoeConfig(
        vocab_size=31,
        d_model=16,
        encoder_layers=4,
        decoder_layers=4,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=24,
        decoder_ffn_dim=24,
        dropout=0.1,
        attention_dropout=0.0,
        encoder_layerdrop=layerdrop,
        decoder_layerdrop=layerdrop,
        max_position_embeddings=32,
        num_experts=4,
        encoder_sparse_step=2,
        decoder_sparse_step=2,
        expert_capacity=100,
        router_jitter_noise=0.01,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(13)
    model = (NllbMoeDecoder if decoder else NllbMoeEncoder)(config).train()
    input_ids = torch.tensor([[2, 5, 9, 7, 3], [4, 8, 6, 3, 2]])
    inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "use_cache": False}
    if decoder:
        inputs.update(
            encoder_hidden_states=torch.arange(160).reshape(2, 5, 16).float() / 100,
            encoder_attention_mask=torch.ones_like(input_ids),
        )
    results = []
    for enabled in (False, True):
        torch.manual_seed(0)
        with nullcontext() if enabled else without_output_scoping("nllb_moe"):
            outputs = model(**inputs, output_router_logits=True, output_hidden_states=True, output_attentions=True)
        results.append((outputs, torch.get_rng_state()))
    assert_same(results[0], results[1])
    if layerdrop == 1:
        assert results[1][0].router_logits == ()
    else:
        assert 0 < len(results[1][0].hidden_states) - 1 < 4


def _multimodal_or_hybrid_model(hybrid, layers):
    kwargs = {
        "vocab_size": 31,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "use_cache": False,
    }
    if hybrid:
        kwargs.update(
            layer_types=["linear_attention", "full_attention"] * (layers // 2),
            linear_conv_kernel_dim=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
        )
        config, cls = transformers.Qwen3NextConfig(**kwargs), transformers.Qwen3NextModel
    else:
        kwargs["rope_parameters"] = {"rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True}
        config, cls = transformers.Qwen3VLMoeTextConfig(**kwargs), transformers.Qwen3VLMoeTextModel
    config._attn_implementation = "eager"
    torch.manual_seed(13)
    return cls(config).eval()


@require_torch
@pytest.mark.torch_compile_test
@pytest.mark.parametrize("hybrid", [False, True], ids=["multimodal_text", "hybrid_attention"])
def test_distinct_production_loops_reuse_all_compiled_regions(hybrid):
    family = "qwen3_next" if hybrid else "qwen3_vl_moe"
    limit = getattr(torch._dynamo.config, "recompile_limit", None)
    if limit is None:
        limit = torch._dynamo.config.cache_size_limit
    layers = 2 * ((limit + 5) // 2)
    try:
        torch.compiler.reset()
        model = _multimodal_or_hybrid_model(hybrid, layers)
        input_ids = torch.tensor([[2, 5, 9, 7, 3], [4, 8, 6, 3, 2]])
        with without_output_scoping(family):
            reference = model(input_ids=input_ids, output_router_logits=True)
        graphs, calls = [], []

        def backend(graph, example_inputs):
            graphs.append(graph)

            def execute(*args):
                calls.append(None)
                return graph.forward(*args)

            return execute

        for layer in model.layers:
            layer.compile(backend=backend)
        # Require the production loop to keep every region compiled. The backend
        # executes the captured operations unchanged, isolating graph reuse from
        # kernel numerics. Alternating attention types legitimately use two graphs.
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            outputs = model(input_ids=input_ids, output_router_logits=True)
            assert_same(outputs, reference)
            assert len(outputs.router_logits) == layers
            outputs.last_hidden_state[..., 0].mean().backward()
        assert len(graphs) == (2 if hybrid else 1)
        assert len(calls) == 2 * layers
    finally:
        torch.compiler.reset()


@require_torch
@pytest.mark.parametrize("model_name", ["OpenAIPrivacyFilterModel", "OpenAIPrivacyFilterForTokenClassification"])
def test_encoder_router_capture_preserves_backbone_and_task_head_contracts(model_name):
    model_class = getattr(transformers, model_name)
    config = transformers.OpenAIPrivacyFilterConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        sliding_window=2,
        max_position_embeddings=32,
        pad_token_id=0,
        eos_token_id=2,
        num_labels=3,
        classifier_dropout=0.0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    config._attn_implementation = "eager"
    torch.manual_seed(13)
    original = model_class(config).cpu().train()
    candidate = copy.deepcopy(original)
    input_ids = torch.tensor([[2, 5, 9, 7, 3], [4, 8, 6, 3, 2]])
    is_task_head = model_name == "OpenAIPrivacyFilterForTokenClassification"
    kwargs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    if is_task_head:
        kwargs["labels"] = input_ids % config.num_labels
    results = []
    for model, scoped in ((original, False), (candidate, True)):
        torch.manual_seed(29)
        with nullcontext() if scoped else without_output_scoping("openai_privacy_filter"):
            outputs = model(**kwargs, output_router_logits=True, output_hidden_states=True)
            if is_task_head:
                # The classifier preserves hidden states but does not expose
                # router scores in its public TokenClassifierOutput contract.
                assert "router_logits" not in outputs
                router_gradients = None
                objective = outputs.loss
            else:
                assert len(outputs.router_logits) == config.num_hidden_layers
                router_weights = tuple(layer.mlp.router.weight for layer in model.layers)
                router_objective = sum(scores.square().mean() for scores in outputs.router_logits)
                router_gradients = torch.autograd.grad(router_objective, router_weights, retain_graph=True)
                assert all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in router_gradients)
                objective = outputs.last_hidden_state[..., 0].mean()
            objective.backward()
        params = tuple(model.parameters())
        grads = tuple(None if p.grad is None else p.grad.clone() for p in params)
        assert all(g is None or torch.isfinite(g).all() for g in grads)
        assert any(g is not None and torch.count_nonzero(g) for g in grads)
        results.append((outputs, router_gradients, grads))
    assert_same(results[1], results[0])
