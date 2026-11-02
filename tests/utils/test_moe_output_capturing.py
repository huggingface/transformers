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
"""CPU contracts for region-local model output collection.

Tiny production Qwen models isolate collection behavior from optimized kernel numerics.
"""

import pytest

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import Qwen3MoeForCausalLM

    from ..models.qwen3_moe.test_modeling_qwen3_moe import Qwen3MoeModelTester


@require_torch
@pytest.mark.torch_compile_test
@pytest.mark.parametrize(
    "output_router_logits,output_attentions,output_hidden_states,checkpointing",
    [
        (True, False, False, False),
        (False, True, False, False),
        (True, True, False, True),
        (False, False, True, False),
        (False, False, [0, 2], True),
    ],
    ids=[
        "routers",
        "attentions",
        "checkpointed_routers_and_attentions",
        "hidden_states",
        "checkpointed_selected_hidden_states",
    ],
)
def test_regional_compilation_preserves_outputs_and_reuses_graphs(
    output_router_logits, output_attentions, output_hidden_states, checkpointing, request
):
    """Given matching Qwen layers with requested router, attention, or hidden-state outputs,
    when forward and backward run with regional compilation,
    then outputs and gradients match eager execution, every forward layer call stays compiled,
    and repeated calls reuse graphs. Checkpointed calls also replay compiled code.
    """
    from transformers.utils import output_capturing

    torch.compiler.reset()
    request.addfinalizer(torch.compiler.reset)
    limit = getattr(torch._dynamo.config, "recompile_limit", None)
    if limit is None:
        limit = torch._dynamo.config.cache_size_limit
    # Exceed the normal limit so per-layer specialization cannot pass unnoticed.
    config = Qwen3MoeModelTester(None, num_hidden_layers=limit + 1, use_cache=False).get_config()
    config._attn_implementation = "eager"
    torch.manual_seed(13)
    model = Qwen3MoeForCausalLM(config).cpu().train()
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    input_ids = torch.tensor([[2, 5, 9, 7, 3], [4, 8, 6, 3, 2]])
    inputs = {"input_ids": input_ids, "labels": input_ids}
    requested = tuple(
        key for key, enabled in (("router_logits", output_router_logits), ("attentions", output_attentions)) if enabled
    )
    graphs, calls = [], []

    def forward_backward():
        calls.clear()
        model.zero_grad(set_to_none=True)
        outputs = model(
            **inputs,
            output_router_logits=output_router_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        forward_call_count = len(calls)
        for key in requested:
            assert len(outputs[key]) == len(model.model.layers)
            assert all(value.requires_grad for value in outputs[key])

        # Keep the collected-output objective separate from task loss: task
        # gradients alone could conceal detached collected outputs.
        objective = outputs.aux_loss if output_router_logits else 0
        output_parameters = []
        if output_router_logits:
            output_parameters.extend(layer.mlp.gate.weight for layer in model.model.layers)
        if output_attentions:
            objective = objective + sum(
                (attention * torch.arange(attention.shape[-1], device=attention.device)).sum()
                for attention in outputs.attentions
            )
            for layer in model.model.layers:
                output_parameters.extend((layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight))
        if output_hidden_states:
            hidden_states = outputs.hidden_states
            assert len(hidden_states) == len(model.model.layers) + (output_hidden_states is True)
            if isinstance(output_hidden_states, list):
                assert [i for i, value in enumerate(hidden_states) if value is not None] == output_hidden_states
                hidden_states = tuple(value for value in hidden_states if value is not None)
            assert all(value.requires_grad for value in hidden_states)
            objective = objective + sum(
                (value * torch.arange(value.shape[-1], device=value.device)).sum(-1).square().sum()
                for value in hidden_states
            )
            output_parameters.append(model.get_input_embeddings().weight)

        output_grads = torch.autograd.grad(objective, output_parameters, retain_graph=True)
        assert all(torch.isfinite(grad).all() and torch.count_nonzero(grad) for grad in output_grads)
        outputs.loss.backward()
        task_grads = tuple(None if p.grad is None else p.grad.clone() for p in model.parameters())
        assert all(grad is not None and torch.isfinite(grad).all() for grad in task_grads)
        assert torch.count_nonzero(model.get_input_embeddings().weight.grad)
        assert output_capturing._active_collector.get() is None
        return (outputs, output_grads, task_grads), forward_call_count

    reference, _ = forward_backward()  # Also installs the real recording hooks before tracing.

    def backend(graph, example_inputs):
        graphs.append(graph)

        def execute(*args):
            calls.append(output_capturing._active_collector.get())
            return graph.forward(*args)

        return execute

    for layer in model.model.layers:
        layer.compile(backend=backend, fullgraph=True)
    graph_counts = []
    for _ in range(2):
        actual, forward_call_count = forward_backward()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert forward_call_count == len(model.model.layers), "a decoder did not execute its compiled callable"
        replay_collectors = calls[forward_call_count:]
        assert bool(replay_collectors) == checkpointing, "checkpoint replay did not execute as requested"
        assert all(collector is None for collector in replay_collectors), (
            "checkpoint replay reused the forward collector"
        )
        graph_counts.append(len(graphs))
    assert graph_counts[0] == graph_counts[1], "repeated forward/backward created more graphs"
    if not checkpointing:
        assert len(graphs) == 1, "earlier decoder outputs caused graph specialization"


@require_torch
@pytest.mark.torch_compile_test
def test_whole_model_compilation_preserves_output_capture_and_gradients(request):
    """Given requested router, attention, and hidden-state outputs, when the complete model is compiled,
    then outputs and gradients match eager execution and repeated calls reuse the graph.
    """
    from torch._dynamo.testing import CompileCounterWithBackend

    torch.compiler.reset()
    request.addfinalizer(torch.compiler.reset)
    config = Qwen3MoeModelTester(None, use_cache=False).get_config()
    config._attn_implementation = "eager"
    torch.manual_seed(13)
    model = Qwen3MoeForCausalLM(config).cpu().train()
    input_ids = torch.tensor([[2, 5, 9, 7, 3], [4, 8, 6, 3, 2]])

    def forward_backward(call):
        model.zero_grad(set_to_none=True)
        outputs = call(
            input_ids, labels=input_ids, output_router_logits=True, output_attentions=True, output_hidden_states=True
        )
        output_loss = outputs.aux_loss + sum(
            (attention * torch.arange(attention.shape[-1])).sum() for attention in outputs.attentions
        )
        output_loss = output_loss + sum(
            (value * torch.arange(value.shape[-1])).sum(-1).square().sum() for value in outputs.hidden_states
        )
        params = tuple(model.parameters())
        output_grads = torch.autograd.grad(output_loss, params, allow_unused=True, retain_graph=True)
        outputs.loss.backward()
        task_grads = tuple(None if p.grad is None else p.grad.clone() for p in params)
        return outputs, output_grads, task_grads

    reference = forward_backward(model)
    counter = CompileCounterWithBackend("eager")
    compiled = torch.compile(model, backend=counter, fullgraph=True)
    for _ in range(2):
        torch.testing.assert_close(forward_backward(compiled), reference, rtol=0, atol=0)
    assert counter.frame_count == 1 and counter.op_count > 0
