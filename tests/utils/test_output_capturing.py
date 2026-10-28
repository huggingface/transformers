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

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from contextvars import copy_context
from unittest.mock import patch

from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available
from transformers.utils import output_capturing as capturing


if is_torch_available():
    import torch


@require_torch
class RouterOutputCollectionTest(unittest.TestCase):
    def setUp(self):
        self.collector = capturing.CompileableContextVar("test_output_collector")
        patcher = patch.object(capturing, "_active_collector", self.collector)
        patcher.start()
        self.addCleanup(patcher.stop)

    @contextmanager
    def collecting(self, outputs):
        token = self.collector.set(outputs)
        try:
            yield
        finally:
            self.collector.reset(token)

    def test_hooks_preserve_order_identity_and_gradients(self):
        inputs = torch.tensor([1.0, 2.0], requires_grad=True)
        expected = [inputs * scale for scale in (2, 3, 5, 7)]
        original = []
        outputs = {"router_logits": original}
        with self.collecting(outputs):
            # A local buffer belongs to the existing collector, not a second capture decorator.
            with patch.object(self.collector, "set", side_effect=AssertionError("second collector owner")):
                with patch.object(self.collector, "reset", side_effect=AssertionError("reset collector owner")):
                    with capturing._scope_router_output_collection():
                        self.assertIsNot(self.collector.get()["router_logits"], original)
                        self.assertIs(outputs["router_logits"], original)
                        for index, value in zip((None, 0, 2, -1), expected):
                            router = torch.nn.Identity()
                            capturing.install_output_capuring_hook(router, "router_logits", index or 0)
                            if index is None:
                                router(value)
                            elif index == 0:
                                router((value, None))
                            else:
                                router((None, None, value))
                        router((None, None, None))  # A recorder may have no value to append.
                    with capturing._scope_router_output_collection():
                        pass  # A dense region may emit no router outputs.
            self.assertIs(outputs["router_logits"], original)

        self.assertEqual(len(original), len(expected))
        for actual, reference in zip(original, expected):
            self.assertIs(actual, reference)
        sum(value.sum() for value in original).backward()
        torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 17.0), rtol=0, atol=0)

    def test_nested_scopes_and_exceptions_restore_the_parent(self):
        original = []
        outputs = {"router_logits": original}
        records = [object() for _ in range(5)]
        with self.collecting(outputs):
            with capturing._scope_router_output_collection():
                outer = self.collector.get()["router_logits"]
                outer.append(records[0])
                with self.assertRaisesRegex(RuntimeError, "inner"):
                    with capturing._scope_router_output_collection():
                        self.collector.get()["router_logits"].append(records[1])
                        raise RuntimeError("inner")
                self.assertIs(self.collector.get()["router_logits"], outer)
                outer.append(records[2])
            self.assertIs(outputs["router_logits"], original)
            with self.assertRaisesRegex(RuntimeError, "outer"):
                with capturing._scope_router_output_collection():
                    self.collector.get()["router_logits"].append(records[3])
                    raise RuntimeError("outer")
            self.assertIs(outputs["router_logits"], original)
            with capturing._scope_router_output_collection():
                self.collector.get()["router_logits"].append(records[4])
        self.assertEqual(original, records)
        self.assertIsNone(self.collector.get())

    def test_inactive_router_collection_is_unchanged(self):
        for outputs in (None, {"hidden_states": []}):
            with self.subTest(outputs=outputs), self.collecting(outputs):
                with patch.object(self.collector, "set", side_effect=AssertionError("unexpected collector")):
                    with capturing._scope_router_output_collection():
                        self.assertIs(self.collector.get(), outputs)
                if outputs is not None:
                    self.assertEqual(outputs, {"hidden_states": []})

    def test_hidden_state_indices_and_other_outputs_are_unchanged(self):
        hidden_states, attentions, router_logits = [], [], []
        outputs = {
            "hidden_states": hidden_states,
            "attentions": attentions,
            "router_logits": router_logits,
            "_hidden_states_layers": {1},
        }
        layer, attention, router = (torch.nn.Identity() for _ in range(3))
        capturing.install_output_capuring_hook(layer, "hidden_states", 0)
        capturing.install_output_capuring_hook(attention, "attentions", 0)
        capturing.install_output_capuring_hook(router, "router_logits", 0)
        values = [torch.tensor([float(i)]) for i in range(3)]
        with self.collecting(outputs):
            for value in values:
                with capturing._scope_router_output_collection():
                    self.assertIs(self.collector.get()["hidden_states"], hidden_states)
                    self.assertIs(self.collector.get()["attentions"], attentions)
                    layer(value)
                    attention(value)
                    router(value)
        self.assertEqual(len(hidden_states), 3)
        self.assertIsNone(hidden_states[0])
        self.assertIs(hidden_states[1], values[1])
        self.assertIsNone(hidden_states[2])
        for captured in (attentions, router_logits):
            self.assertEqual(len(captured), len(values))
            for actual, reference in zip(captured, values):
                self.assertIs(actual, reference)

    def test_whole_model_compilation_keeps_the_existing_buffer(self):
        # Tracing and a collector already activated by tracing both use the original path.
        for tracing, global_collector in ((True, False), (False, True)):
            with self.subTest(tracing=tracing, global_collector=global_collector):
                original = []
                outputs = {"router_logits": original}
                with self.collecting(outputs):
                    with (
                        patch.object(capturing, "is_torchdynamo_compiling", return_value=tracing),
                        patch.object(self.collector, "compiling", global_collector),
                        patch.object(self.collector, "global_var", outputs),
                    ):
                        with capturing._scope_router_output_collection():
                            self.assertIs(self.collector.get()["router_logits"], original)
                            original.append("record")
                self.assertEqual(original, ["record"])

    def test_regional_compile_reuses_graph_without_losing_router_gradients(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.router = torch.nn.Linear(8, 4)
                self.proj = torch.nn.Linear(8, 8)

            def forward(self, inputs):
                logits = self.router(inputs)
                return self.proj(inputs) + logits.sum(-1, keepdim=True) * 0.01

        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        limit = getattr(torch._dynamo.config, "recompile_limit", None)
        if limit is None:
            limit = torch._dynamo.config.cache_size_limit
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            blocks = [Block() for _ in range(limit + 1)]
            inputs = torch.randn(1, 5, 8)
        for block in blocks:
            capturing.install_output_capuring_hook(block.router, "router_logits", 0)
        parameters = [parameter for block in blocks for parameter in block.parameters()]
        router_parameters = [parameter for block in blocks for parameter in block.router.parameters()]

        def forward_backward(scoped):
            for block in blocks:
                block.zero_grad(set_to_none=True)
            hidden = inputs.clone().requires_grad_()
            outputs = {"router_logits": []}
            with self.collecting(outputs):
                for block in blocks:
                    with capturing._scope_router_output_collection() if scoped else nullcontext():
                        hidden = block(hidden)
            self.assertIsNone(self.collector.get())
            self.assertEqual(len(outputs["router_logits"]), len(blocks))
            auxiliary_loss = sum(value.square().mean() for value in outputs["router_logits"]) * 0.01
            auxiliary_gradients = torch.autograd.grad(auxiliary_loss, router_parameters, retain_graph=True)
            (hidden.square().mean() + auxiliary_loss).backward()
            for parameter in parameters:
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
            return (
                hidden.detach(),
                tuple(value.detach() for value in outputs["router_logits"]),
                auxiliary_loss.detach(),
                auxiliary_gradients,
                tuple(parameter.grad.clone() for parameter in parameters),
            )

        reference = forward_backward(scoped=False)
        graphs, calls = [], []

        def backend(graph, example_inputs):
            graphs.append(graph)

            def execute(*args):
                calls.append(None)
                return graph.forward(*args)

            return execute

        for block in blocks:
            block.compile(backend=backend)
        # A counting backend isolates Dynamo's list-length specialization from kernel numerics.
        # A second invocation also checks that capture does not leak across model calls/backward.
        for _ in range(2):
            torch.testing.assert_close(forward_backward(scoped=True), reference, rtol=0, atol=0)
        self.assertEqual(len(graphs), 1, "router capture specialized the graph for successive regions")
        self.assertEqual(len(calls), 2 * len(blocks), "a region silently fell back to eager execution")

    def test_threads_keep_independent_capture_contexts(self):
        barrier = threading.Barrier(2, timeout=10)

        def collect(marker):
            self.assertIsNone(self.collector.get())
            original = []
            outputs = {"router_logits": original}
            with self.collecting(outputs):
                with capturing._scope_router_output_collection():
                    barrier.wait()
                    self.collector.get()["router_logits"].append(marker)
                    barrier.wait()
                self.assertIs(outputs["router_logits"], original)
            self.assertIsNone(self.collector.get())
            return original

        parent = {"router_logits": ["parent"]}
        with self.collecting(parent), ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(collect, marker) for marker in ("first", "second")]
            self.assertEqual([future.result(timeout=15) for future in futures], [["first"], ["second"]])
            self.assertIs(self.collector.get(), parent)
            self.assertEqual(parent["router_logits"], ["parent"])

    def test_copied_contexts_do_not_replace_each_others_router_buffers(self):
        first_entered, second_entered, first_exited = (threading.Event() for _ in range(3))

        def collect_first():
            try:
                with capturing._scope_router_output_collection():
                    local = self.collector.get()["router_logits"]
                    first_entered.set()
                    self.assertTrue(second_entered.wait(timeout=10))
                    self.collector.get()["router_logits"].append("first")
                return local
            finally:
                first_exited.set()

        def collect_second():
            self.assertTrue(first_entered.wait(timeout=10))
            with capturing._scope_router_output_collection():
                local = self.collector.get()["router_logits"]
                second_entered.set()
                self.assertTrue(first_exited.wait(timeout=10))
                self.collector.get()["router_logits"].append("second")
            return local

        original = []
        parent = {"router_logits": original}
        with self.collecting(parent), ThreadPoolExecutor(max_workers=2) as pool:
            # copy_context copies bindings, not the dictionary to which they refer.
            first = pool.submit(copy_context().run, collect_first)
            second = pool.submit(copy_context().run, collect_second)
            self.assertEqual(first.result(timeout=15), ["first"])
            self.assertEqual(second.result(timeout=15), ["second"])
            self.assertIs(self.collector.get(), parent)
            self.assertIs(parent["router_logits"], original)
            self.assertEqual(original, ["first", "second"])
