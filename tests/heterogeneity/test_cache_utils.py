# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import itertools
import unittest

from transformers.testing_utils import is_torch_available, require_torch


if is_torch_available():
    from transformers.cache_utils import DYNAMIC_LAYER_TYPE_MAPPING, STATIC_LAYER_TYPE_MAPPING
    from transformers.integrations.heterogeneity.cache_utils import (
        _HYBRID_LAYER_TYPE_TO_REDUCED_LAYER_TYPES,
        _get_layer_type_caches,
    )


# `_HYBRID_LAYER_TYPE_TO_REDUCED_LAYER_TYPES` is written by hand, so these tests check it against the cache layer
# classes in the cache mappings
@require_torch
class TestCacheUtils(unittest.TestCase):
    def test_every_hybrid_layer_type_has_an_entry(self):
        # A hybrid layer type added to the cache mappings without an entry would fail when skips drop some of its
        # caches
        for layer_type_mapping in (DYNAMIC_LAYER_TYPE_MAPPING, STATIC_LAYER_TYPE_MAPPING):
            hybrid_layer_types = {
                layer_type for layer_type in layer_type_mapping if len(_get_layer_type_caches(layer_type)) > 1
            }
            self.assertEqual(hybrid_layer_types, set(_HYBRID_LAYER_TYPE_TO_REDUCED_LAYER_TYPES))

    def test_every_set_of_remaining_caches_has_a_reduced_layer_type(self):
        # Skips can drop any of a hybrid layer type's caches, so any non-empty set of them except all of them can
        # remain, and each needs a reduced layer type
        for hybrid_layer_type, reduced_layer_types in _HYBRID_LAYER_TYPE_TO_REDUCED_LAYER_TYPES.items():
            caches = _get_layer_type_caches(hybrid_layer_type)
            possible_remaining_caches = {
                frozenset(subset) for size in range(1, len(caches)) for subset in itertools.combinations(caches, size)
            }
            self.assertEqual(set(reduced_layer_types), possible_remaining_caches, hybrid_layer_type)

    def test_reduced_layer_types_are_parts_of_their_hybrid_layer_type(self):
        # A hybrid cache layer class combines one class per cache. For example, "hybrid_sliding" uses
        # `LinearAttentionAndSlidingWindowAttentionLayer(LinearAttentionLayer, DynamicSlidingWindowLayer)`, so when
        # only its key/value cache remains, the reduced layer type must be "sliding_attention", which uses
        # `DynamicSlidingWindowLayer`.
        for layer_type_mapping in (DYNAMIC_LAYER_TYPE_MAPPING, STATIC_LAYER_TYPE_MAPPING):
            for hybrid_layer_type, reduced_layer_types in _HYBRID_LAYER_TYPE_TO_REDUCED_LAYER_TYPES.items():
                for remaining_caches, reduced_layer_type in reduced_layer_types.items():
                    with self.subTest(hybrid_layer_type=hybrid_layer_type, remaining_caches=sorted(remaining_caches)):
                        # The reduced layer type holds exactly the remaining caches
                        self.assertEqual(_get_layer_type_caches(reduced_layer_type), remaining_caches)
                        # Its class is one that the hybrid class combines. This checks direct bases rather than any
                        # ancestor, because `DynamicSlidingWindowLayer` subclasses `DynamicLayer`, so "full_attention"
                        # would wrongly pass for "hybrid_sliding".
                        self.assertIn(
                            layer_type_mapping[reduced_layer_type], layer_type_mapping[hybrid_layer_type].__bases__
                        )
