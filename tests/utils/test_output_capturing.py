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

import unittest
from contextlib import ExitStack, contextmanager
from contextvars import copy_context

from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available
from transformers.utils import output_capturing as capturing


if is_torch_available():
    import torch


@require_torch
class OutputCollectionTest(unittest.TestCase):
    def setUp(self):
        self.collector = capturing._active_collector

    @contextmanager
    def collecting(self, outputs):
        token = self.collector.set(outputs)
        try:
            yield
        finally:
            self.collector.reset(token)

    def test_nested_scopes_and_exceptions_restore_the_parent(self):
        """Given nested collections, when inner and outer calls raise,
        then both errors propagate, outputs merge once, and each caller regains its collector.
        """
        for key in ("router_logits", "attentions"):
            records = [object() for _ in range(3)]
            outputs = {key: []}
            with self.subTest(key=key), self.collecting(outputs):
                with self.assertRaisesRegex(RuntimeError, "outer"), capturing._fresh_output_collection():
                    outer = self.collector.get()[key]
                    outer.append(records[0])
                    with self.assertRaisesRegex(RuntimeError, "inner"), capturing._fresh_output_collection():
                        self.collector.get()[key].append(records[1])
                        raise RuntimeError("inner")
                    self.assertIs(self.collector.get()[key], outer)
                    outer.append(records[2])
                    raise RuntimeError("outer")
                self.assertIs(self.collector.get(), outputs)
                self.assertEqual(outputs[key], records)

    def test_inactive_output_collection_is_unchanged(self):
        """Given no router or attention collection, when a region runs,
        then the active collector and its contents stay unchanged.
        """
        for outputs in (None, {"hidden_states": [], "cross_attentions": []}):
            expected = None if outputs is None else {key: [] for key in outputs}
            with self.subTest(outputs=outputs), self.collecting(outputs):
                with capturing._fresh_output_collection():
                    self.assertIs(self.collector.get(), outputs)
                self.assertEqual(outputs, expected)

    def test_regions_preserve_outputs_gradients_and_other_channels(self):
        """Given a collector, when successive regions record router and attention tensors,
        then each starts fresh, results retain order and gradients, and other channels retain their history.
        """
        outputs = {"router_logits": [], "attentions": [], "hidden_states": [], "_hidden_states_layers": {1}}
        values = [torch.tensor(float(i), requires_grad=True) for i in range(3)]
        with self.collecting(outputs):
            for index, value in enumerate(values):
                with capturing._fresh_output_collection():
                    current = self.collector.get()
                    self.assertEqual(current["router_logits"], [])
                    self.assertEqual(current["attentions"], [])
                    self.assertEqual(current["hidden_states"], values[:index])
                    self.assertEqual(current["_hidden_states_layers"], {1})
                    current["router_logits"].extend([value, value])
                    current["attentions"].append(value)
                    current["hidden_states"].append(value)
            with capturing._fresh_output_collection():
                pass  # A layer may emit no requested outputs.
        self.assertEqual(outputs["router_logits"], [value for value in values for _ in range(2)])
        self.assertEqual(outputs["attentions"], values)
        self.assertEqual(outputs["hidden_states"], values)
        sum(outputs["router_logits"] + outputs["attentions"]).backward()
        for value in values:
            torch.testing.assert_close(value.grad, torch.tensor(3.0), rtol=0, atol=0)

    def test_copied_contexts_do_not_replace_each_others_router_buffers(self):
        """Given two contexts sharing a collector, when their regions overlap,
        then each collects its own outputs and both append to the original buffer.
        """

        def append(marker):
            buffer = self.collector.get()["router_logits"]
            buffer.append(marker)
            return buffer

        original = []
        parent = {"router_logits": original}
        with self.collecting(parent):
            # Copy bindings, not their shared dictionary. Exit in entry order,
            # as overlapping requests can, rather than nesting the two scopes.
            first_context, second_context = copy_context(), copy_context()
            first_scope, second_scope = ExitStack(), ExitStack()
            try:
                first_context.run(first_scope.enter_context, capturing._fresh_output_collection())
                second_context.run(second_scope.enter_context, capturing._fresh_output_collection())
                first = first_context.run(append, "first")
                first_context.run(first_scope.close)
                self.assertIs(first_context.run(self.collector.get), parent)
                second = second_context.run(append, "second")
                second_context.run(second_scope.close)
                self.assertIs(second_context.run(self.collector.get), parent)
            finally:
                first_context.run(first_scope.close)
                second_context.run(second_scope.close)
            self.assertEqual(first, ["first"])
            self.assertEqual(second, ["second"])
            self.assertIs(self.collector.get(), parent)
            self.assertIs(parent["router_logits"], original)
            self.assertEqual(original, ["first", "second"])
