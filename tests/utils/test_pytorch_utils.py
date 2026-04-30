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

import gc
import unittest
import weakref

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    from transformers.pytorch_utils import compile_compatible_method_lru_cache


@require_torch
class CompileCompatibleMethodLruCacheTest(unittest.TestCase):
    """
    Tests for `compile_compatible_method_lru_cache`.

    Regression coverage for https://github.com/huggingface/transformers/issues/45412
    -- decorating an instance method with `@compile_compatible_method_lru_cache`
    must not keep a strong reference to `self`, otherwise models that use it
    (e.g. RT-DETR, RT-DETRv2, MaskFormer, Conditional DETR, EdgeTAM)
    cannot release their parameters after `del model`.
    """

    def _make_class(self, call_counter):
        class _Module:
            @compile_compatible_method_lru_cache(maxsize=4)
            def forward(self, x: int, y: int) -> int:
                call_counter.append((id(self), x, y))
                return x + y

        return _Module

    def test_cache_returns_correct_value(self):
        call_counter = []
        cls = self._make_class(call_counter)
        instance = cls()
        self.assertEqual(instance.forward(1, 2), 3)
        self.assertEqual(instance.forward(3, 4), 7)

    def test_cache_hits_skip_underlying_call(self):
        call_counter = []
        cls = self._make_class(call_counter)
        instance = cls()

        instance.forward(1, 2)
        instance.forward(1, 2)
        instance.forward(1, 2)

        # Three calls, but the underlying function should only run once.
        self.assertEqual(len(call_counter), 1)

    def test_cache_is_per_instance(self):
        call_counter = []
        cls = self._make_class(call_counter)
        instance_a = cls()
        instance_b = cls()

        instance_a.forward(1, 2)
        instance_b.forward(1, 2)

        # Per-instance caches: instance_b cannot reuse instance_a's hit.
        self.assertEqual(len(call_counter), 2)
        self.assertEqual({entry[0] for entry in call_counter}, {id(instance_a), id(instance_b)})

    def test_instance_is_garbage_collected_after_delete(self):
        """
        Regression test for issue #45412.

        Before the fix, `lru_cache` was applied at class level and `self` was
        part of the cache key, so every instance ever passed through the
        decorated method was kept alive by the cache. After `del instance`,
        a `weakref` to it must resolve to `None` after a GC pass.
        """
        call_counter = []
        cls = self._make_class(call_counter)

        instance = cls()
        instance.forward(1, 2)  # populate the cache
        instance.forward(3, 4)

        ref = weakref.ref(instance)
        self.assertIsNotNone(ref())

        del instance
        gc.collect()

        self.assertIsNone(
            ref(),
            "Instance should be garbage collected after `del`, but the lru_cache "
            "decorator is still holding a strong reference to `self`.",
        )

    def test_many_instances_release_independently(self):
        """
        Stress test: build many instances, run the cached method on each,
        drop them, and verify all are collected. Mirrors the real-world
        pattern of repeatedly loading and freeing detection models.
        """
        call_counter = []
        cls = self._make_class(call_counter)

        refs = []
        for _ in range(50):
            instance = cls()
            instance.forward(1, 2)
            refs.append(weakref.ref(instance))
            del instance

        gc.collect()
        alive = sum(1 for ref in refs if ref() is not None)
        self.assertEqual(alive, 0, f"{alive}/50 instances leaked through the lru_cache decorator.")


if __name__ == "__main__":
    unittest.main()
