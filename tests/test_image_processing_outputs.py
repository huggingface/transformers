# Copyright 2025 HuggingFace Inc.
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

import copy
import unittest
from dataclasses import dataclass
from typing import TYPE_CHECKING

from transformers.image_processing_outputs import PostProcessOutput
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if TYPE_CHECKING:
    import torch

if is_torch_available():
    import torch


@dataclass
class SubclassPostProcessOutput(PostProcessOutput):
    scores: "torch.Tensor"
    labels: "torch.Tensor"


class PostProcessOutputTest(unittest.TestCase):
    def test_base_class_direct_instantiation(self):
        obj = PostProcessOutput({"a": 1, "b": 2})
        self.assertEqual(obj["a"], 1)
        self.assertEqual(obj["b"], 2)

    def test_subclass_without_dataclass_raises(self):
        class BadSubclass(PostProcessOutput):
            pass

        with self.assertRaises(TypeError):
            BadSubclass()

    def test_subclass_with_dataclass_does_not_raise(self):
        @dataclass
        class GoodSubclass(PostProcessOutput):
            value: int

        obj = GoodSubclass(value=42)
        self.assertEqual(obj.value, 42)


@require_torch
class SubclassPostProcessOutputTest(unittest.TestCase):
    def _make_output(self, batch_size=2, num_labels=3):
        scores = torch.rand(batch_size, num_labels)
        labels = torch.randint(0, num_labels, (batch_size,))
        return SubclassPostProcessOutput(scores=scores, labels=labels)

    def test_attribute_access(self):
        obj = self._make_output()
        self.assertIsInstance(obj.scores, torch.Tensor)
        self.assertIsInstance(obj.labels, torch.Tensor)

    def test_dict_access(self):
        obj = self._make_output()
        self.assertIsInstance(obj["scores"], torch.Tensor)
        self.assertIsInstance(obj["labels"], torch.Tensor)

    def test_attribute_and_dict_access_return_same_object(self):
        obj = self._make_output()
        self.assertIs(obj.scores, obj["scores"])
        self.assertIs(obj.labels, obj["labels"])

    def test_len(self):
        obj = self._make_output()
        self.assertEqual(len(obj), 2)

    def test_keys(self):
        obj = self._make_output()
        self.assertEqual(set(obj.keys()), {"scores", "labels"})

    def test_contains(self):
        obj = self._make_output()
        self.assertIn("scores", obj)
        self.assertIn("labels", obj)
        self.assertNotIn("data", obj)
        self.assertNotIn("nonexistent_field", obj)

    def test_iteration_yields_keys(self):
        obj = self._make_output()
        self.assertEqual(set(obj), {"scores", "labels"})

    def test_values(self):
        obj = self._make_output()
        values = list(obj.values())
        self.assertEqual(len(values), 2)
        self.assertTrue(all(isinstance(v, torch.Tensor) for v in values))

    def test_items(self):
        obj = self._make_output()
        items = dict(obj.items())
        self.assertIn("scores", items)
        self.assertIn("labels", items)

    def test_setattr_updates_dict(self):
        obj = self._make_output()
        new_scores = torch.zeros(2, 3)
        obj.scores = new_scores
        self.assertIs(obj["scores"], new_scores)
        self.assertIs(obj.scores, new_scores)

    def test_setattr_does_not_corrupt_other_fields(self):
        obj = self._make_output()
        original_labels = obj.labels
        obj.scores = torch.ones(2, 3)
        self.assertIs(obj.labels, original_labels)
        self.assertIs(obj["labels"], original_labels)

    def test_setattr_data_raises(self):
        obj = self._make_output()
        with self.assertRaises(AttributeError):
            obj.data = {"scores": torch.zeros(2, 3), "labels": torch.zeros(2, dtype=torch.long)}

    def test_delitem(self):
        obj = self._make_output()
        del obj["scores"]
        self.assertNotIn("scores", obj)
        self.assertFalse(hasattr(obj, "scores"))

    def test_delitem_does_not_affect_other_fields(self):
        obj = self._make_output()
        original_labels = obj.labels
        del obj["scores"]
        self.assertIn("labels", obj)
        self.assertIs(obj.labels, original_labels)

    def test_delitem_nonexistent_key_raises(self):
        obj = self._make_output()
        with self.assertRaises(KeyError):
            del obj["nonexistent"]

    def test_delitem_data_raises(self):
        obj = self._make_output()
        with self.assertRaises(KeyError):
            del obj["data"]

    def test_delattr(self):
        obj = self._make_output()
        del obj.scores
        self.assertFalse(hasattr(obj, "scores"))
        self.assertNotIn("scores", obj)

    def test_delattr_does_not_affect_other_fields(self):
        obj = self._make_output()
        original_labels = obj.labels
        del obj.scores
        self.assertIn("labels", obj)
        self.assertIs(obj.labels, original_labels)

    def test_delattr_nonexistent_attr_raises(self):
        obj = self._make_output()
        with self.assertRaises(AttributeError):
            del obj.nonexistent

    def test_delattr_data_raises(self):
        obj = self._make_output()
        with self.assertRaises(AttributeError):
            del obj.data

    def test_copy(self):
        obj = self._make_output()
        obj_copy = copy.copy(obj)
        self.assertIsInstance(obj_copy, SubclassPostProcessOutput)
        self.assertIs(obj_copy.scores, obj.scores)
        self.assertIs(obj_copy.labels, obj.labels)

    def test_deepcopy(self):
        obj = self._make_output()
        obj_copy = copy.deepcopy(obj)
        self.assertIsInstance(obj_copy, SubclassPostProcessOutput)
        self.assertIsNot(obj_copy.scores, obj.scores)
        self.assertIsNot(obj_copy.labels, obj.labels)
        self.assertTrue(torch.equal(obj_copy.scores, obj.scores))
        self.assertTrue(torch.equal(obj_copy.labels, obj.labels))
