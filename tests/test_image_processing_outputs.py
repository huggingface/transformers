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

import unittest
from dataclasses import dataclass
from typing import TYPE_CHECKING

from transformers.image_processing_outputs import PostProcessorOutput
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if TYPE_CHECKING:
    import torch

if is_torch_available():
    import torch


@dataclass
class SubclassPostProcessorOutput(PostProcessorOutput):
    scores: "torch.Tensor"
    labels: "torch.Tensor"


class PostProcessorOutputTest(unittest.TestCase):
    def test_subclass_without_dataclass_raises(self):
        class BadSubclass(PostProcessorOutput):
            pass

        with self.assertRaises(TypeError):
            BadSubclass()

    def test_subclass_with_dataclass_does_not_raise(self):
        @dataclass
        class GoodSubclass(PostProcessorOutput):
            value: int

        obj = GoodSubclass(value=42)
        self.assertEqual(obj.value, 42)


@require_torch
class SubclassPostProcessorOutputTest(unittest.TestCase):
    def _make_output(self, batch_size=2, num_labels=3):
        scores = torch.rand(batch_size, num_labels)
        labels = torch.randint(0, num_labels, (batch_size,))
        return SubclassPostProcessorOutput(scores=scores, labels=labels)

    def test_attribute_and_dict_access_return_same_object(self):
        obj = self._make_output()
        self.assertIs(obj.scores, obj["scores"])
        self.assertIs(obj.labels, obj["labels"])

    def test_setattr_updates_dict(self):
        obj = self._make_output()
        new_scores = torch.zeros(2, 3)
        obj.scores = new_scores
        self.assertIs(obj["scores"], new_scores)
        self.assertIs(obj.scores, new_scores)

    def test_setattr_does_not_affect_other_fields(self):
        obj = self._make_output()
        original_labels = obj.labels
        obj.scores = torch.ones(2, 3)
        self.assertIs(obj.labels, original_labels)
        self.assertIs(obj["labels"], original_labels)

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

    def test_to_dtype(self):
        # `.to` is inherited from BatchFeature; it mutates the underlying dict in place. Because fields are
        # stored only in that dict, attribute and dict-style access must still return the same casted object.
        obj = self._make_output()
        returned = obj.to(torch.float64)
        self.assertIs(returned, obj)
        self.assertEqual(obj.scores.dtype, torch.float64)
        self.assertIs(obj.scores, obj["scores"])
        # Integer labels are not floating point, so they are left untouched but still accessible.
        self.assertFalse(torch.is_floating_point(obj.labels.dtype))
