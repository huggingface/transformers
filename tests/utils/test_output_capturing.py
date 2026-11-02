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
from contextlib import contextmanager
from contextvars import copy_context
from unittest.mock import patch

from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available
from transformers.utils import output_capturing as capturing


if is_torch_available():
    import torch


@require_torch
class OutputCollectionTest(unittest.TestCase):
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

    def test_hooks_preserve_order_and_gradient_connections(self):
        for key in ("router_logits", "attentions"):
            with self.subTest(key=key):
                inputs = torch.tensor([1.0, 2.0], requires_grad=True)
                expected = [inputs * scale for scale in (2, 3, 5, 7)]
                original = []
                outputs = {key: original}
                with self.collecting(outputs):
                    # A local buffer belongs to the existing collector, not a second capture decorator.
                    with capturing._fresh_output_collection():
                        recorder = torch.nn.Identity()
                        capturing.install_output_capuring_hook(recorder, key, 0)
                        for value in expected:
                            recorder((value, None))
                        recorder((None, None))  # A recorder may have no value to append.
                    with capturing._fresh_output_collection():
                        pass  # A region may emit no requested outputs.

                self.assertEqual(len(original), len(expected))
                for actual, reference in zip(original, expected):
                    self.assertIs(actual, reference)
                sum(value.sum() for value in original).backward()
                torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 17.0), rtol=0, atol=0)

    def test_nested_scopes_and_exceptions_restore_the_parent(self):
        for key in ("router_logits", "attentions"):
            with self.subTest(key=key):
                original = []
                outputs = {key: original}
                records = [object() for _ in range(5)]
                with self.collecting(outputs):
                    with capturing._fresh_output_collection():
                        outer = self.collector.get()[key]
                        outer.append(records[0])
                        with self.assertRaises(RuntimeError):
                            with capturing._fresh_output_collection():
                                self.collector.get()[key].append(records[1])
                                raise RuntimeError("inner")
                        self.assertIs(self.collector.get()[key], outer)
                        outer.append(records[2])
                    with self.assertRaises(RuntimeError):
                        with capturing._fresh_output_collection():
                            self.collector.get()[key].append(records[3])
                            raise RuntimeError("outer")
                    with capturing._fresh_output_collection():
                        self.collector.get()[key].append(records[4])
                    self.assertIs(self.collector.get(), outputs)
                self.assertEqual(original, records)

    def test_inactive_output_collection_is_unchanged(self):
        for outputs in (None, {"hidden_states": []}, {"cross_attentions": []}):
            expected = None if outputs is None else {key: [] for key in outputs}
            with self.subTest(outputs=outputs), self.collecting(outputs):
                with capturing._fresh_output_collection():
                    self.assertIs(self.collector.get(), outputs)
                self.assertEqual(outputs, expected)

    def test_hidden_state_indices_and_other_outputs_are_unchanged(self):
        hidden_states, attentions, router_logits, cross_attentions = [], [], [], []
        outputs = {
            "hidden_states": hidden_states,
            "cross_attentions": cross_attentions,
            "attentions": attentions,
            "router_logits": router_logits,
            "_hidden_states_layers": {1},
        }
        layer, attention, router, cross_attention = (torch.nn.Identity() for _ in range(4))
        capturing.install_output_capuring_hook(layer, "hidden_states", 0)
        capturing.install_output_capuring_hook(attention, "attentions", 0)
        capturing.install_output_capuring_hook(router, "router_logits", 0)
        capturing.install_output_capuring_hook(cross_attention, "cross_attentions", 0)
        values = [torch.tensor([float(i)]) for i in range(3)]
        attention_values = [value + 10 for value in values]
        router_values = [value + 20 for value in values]
        cross_attention_values = [value + 30 for value in values]
        with self.collecting(outputs):
            for value, attention_value, router_value, cross_attention_value in zip(
                values, attention_values, router_values, cross_attention_values
            ):
                with capturing._fresh_output_collection():
                    layer(value)
                    attention(attention_value)
                    router(router_value)
                    cross_attention(cross_attention_value)
        self.assertEqual(len(hidden_states), 3)
        self.assertIsNone(hidden_states[0])
        self.assertIs(hidden_states[1], values[1])
        self.assertIsNone(hidden_states[2])
        for captured, expected in (
            (attentions, attention_values),
            (router_logits, router_values),
            (cross_attentions, cross_attention_values),
        ):
            self.assertEqual(len(captured), len(expected))
            for actual, reference in zip(captured, expected):
                self.assertIs(actual, reference)

    def test_threads_keep_independent_capture_contexts(self):
        barrier = threading.Barrier(2, timeout=10)

        def collect(marker):
            original = []
            outputs = {"router_logits": original}
            with self.collecting(outputs):
                with capturing._fresh_output_collection():
                    barrier.wait()
                    self.collector.get()["router_logits"].append(marker)
                    barrier.wait()
                self.assertIs(self.collector.get(), outputs)
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
                with capturing._fresh_output_collection():
                    local = self.collector.get()["router_logits"]
                    first_entered.set()
                    self.assertTrue(second_entered.wait(timeout=10))
                    self.collector.get()["router_logits"].append("first")
                return local
            finally:
                first_exited.set()

        def collect_second():
            self.assertTrue(first_entered.wait(timeout=10))
            with capturing._fresh_output_collection():
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
