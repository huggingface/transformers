# coding=utf-8
# Copyright 2020 The Hugging Face Team.
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
from dataclasses import dataclass
from typing import Optional

from transformers.testing_utils import require_torch
from transformers.utils import ModelOutput


@dataclass
class ModelOutputTest(ModelOutput):
    a: float
    b: Optional[float] = None
    c: Optional[float] = None


class ModelOutputTester(unittest.TestCase):
    def test_get_attributes(self):
        x = ModelOutputTest(a=30)
        self.assertEqual(x.a, 30)
        self.assertIsNone(x.b)
        self.assertIsNone(x.c)
        with self.assertRaises(AttributeError):
            _ = x.d

    def test_index_with_ints_and_slices(self):
        x = ModelOutputTest(a=30, b=10)
        self.assertEqual(x[0], 30)
        self.assertEqual(x[1], 10)
        self.assertEqual(x[:2], (30, 10))
        self.assertEqual(x[:], (30, 10))

        x = ModelOutputTest(a=30, c=10)
        self.assertEqual(x[0], 30)
        self.assertEqual(x[1], 10)
        self.assertEqual(x[:2], (30, 10))
        self.assertEqual(x[:], (30, 10))

    def test_index_with_strings(self):
        x = ModelOutputTest(a=30, b=10)
        self.assertEqual(x["a"], 30)
        self.assertEqual(x["b"], 10)
        with self.assertRaises(KeyError):
            _ = x["c"]

        x = ModelOutputTest(a=30, c=10)
        self.assertEqual(x["a"], 30)
        self.assertEqual(x["c"], 10)
        with self.assertRaises(KeyError):
            _ = x["b"]

    def test_dict_like_properties(self):
        x = ModelOutputTest(a=30)
        self.assertEqual(list(x.keys()), ["a"])
        self.assertEqual(list(x.values()), [30])
        self.assertEqual(list(x.items()), [("a", 30)])
        self.assertEqual(list(x), ["a"])

        x = ModelOutputTest(a=30, b=10)
        self.assertEqual(list(x.keys()), ["a", "b"])
        self.assertEqual(list(x.values()), [30, 10])
        self.assertEqual(list(x.items()), [("a", 30), ("b", 10)])
        self.assertEqual(list(x), ["a", "b"])

        x = ModelOutputTest(a=30, c=10)
        self.assertEqual(list(x.keys()), ["a", "c"])
        self.assertEqual(list(x.values()), [30, 10])
        self.assertEqual(list(x.items()), [("a", 30), ("c", 10)])
        self.assertEqual(list(x), ["a", "c"])

        with self.assertRaises(Exception):
            x = x.update({"d": 20})
        with self.assertRaises(Exception):
            del x["a"]
        with self.assertRaises(Exception):
            _ = x.pop("a")
        with self.assertRaises(Exception):
            _ = x.setdefault("d", 32)

    def test_set_attributes(self):
        x = ModelOutputTest(a=30)
        x.a = 10
        self.assertEqual(x.a, 10)
        self.assertEqual(x["a"], 10)

    def test_set_keys(self):
        x = ModelOutputTest(a=30)
        x["a"] = 10
        self.assertEqual(x.a, 10)
        self.assertEqual(x["a"], 10)

    def test_instantiate_from_dict(self):
        x = ModelOutputTest({"a": 30, "b": 10})
        self.assertEqual(list(x.keys()), ["a", "b"])
        self.assertEqual(x.a, 30)
        self.assertEqual(x.b, 10)

    def test_instantiate_from_iterator(self):
        x = ModelOutputTest([("a", 30), ("b", 10)])
        self.assertEqual(list(x.keys()), ["a", "b"])
        self.assertEqual(x.a, 30)
        self.assertEqual(x.b, 10)

        with self.assertRaises(ValueError):
            _ = ModelOutputTest([("a", 30), (10, 10)])

        x = ModelOutputTest(a=(30, 30))
        self.assertEqual(list(x.keys()), ["a"])
        self.assertEqual(x.a, (30, 30))

    @require_torch
    def test_torch_pytree(self):
        # ensure torch.utils._pytree treats ModelOutput subclasses as nodes (and not leaves)
        # this is important for DistributedDataParallel gradient synchronization with static_graph=True
        import torch
        import torch.utils._pytree

        x = ModelOutputTest(a=1.0, c=2.0)
        self.assertFalse(torch.utils._pytree._is_leaf(x))

        expected_flat_outs = [1.0, 2.0]
        expected_tree_spec = torch.utils._pytree.TreeSpec(
            ModelOutputTest, ["a", "c"], [torch.utils._pytree.LeafSpec(), torch.utils._pytree.LeafSpec()]
        )

        actual_flat_outs, actual_tree_spec = torch.utils._pytree.tree_flatten(x)
        self.assertEqual(expected_flat_outs, actual_flat_outs)
        self.assertEqual(expected_tree_spec, actual_tree_spec)

        unflattened_x = torch.utils._pytree.tree_unflatten(actual_flat_outs, actual_tree_spec)
        self.assertEqual(x, unflattened_x)


class ModelOutputTestNoDataclass(ModelOutput):
    """Invalid test subclass of ModelOutput where @dataclass decorator is not used"""

    a: float
    b: Optional[float] = None
    c: Optional[float] = None


class ModelOutputSubclassTester(unittest.TestCase):
    def test_direct_model_output(self):
        # Check that direct usage of ModelOutput instantiates without errors
        ModelOutput({"a": 1.1})

    def test_subclass_no_dataclass(self):
        # Check that a subclass of ModelOutput without @dataclass is invalid
        # A valid subclass is inherently tested other unit tests above.
        with self.assertRaises(TypeError):
            ModelOutputTestNoDataclass(a=1.1, b=2.2, c=3.3)
