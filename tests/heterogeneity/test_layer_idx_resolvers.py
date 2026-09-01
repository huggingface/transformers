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

import unittest

from parameterized import parameterized

from transformers.integrations.heterogeneity import LayerIdxFromArgument, LayerIdxFromModelInitStack


def _argument_layer_init(self, config, layer_idx):
    pass


_MODEL_INIT_STACK_RESOLVER = LayerIdxFromModelInitStack("layer_idx")


class _ModelInitStackResolvingLayer:
    def __init__(self, model):
        self.resolved_layer_idx = _MODEL_INIT_STACK_RESOLVER.resolve(
            layer_init=_ModelInitStackResolvingLayer.__init__,
            args=(self, model),
            kwargs={},
            model=model,
        )


class TestLayerIdxResolvers(unittest.TestCase):
    @parameterized.expand(
        [
            ("positional", (object(), object(), 2), {}),
            ("keyword", (object(), object()), {"layer_idx": 2}),
        ]
    )
    def test_argument_resolver_resolves_argument(self, _, args, kwargs):
        resolver = LayerIdxFromArgument("layer_idx")
        resolved_layer_idx = resolver.resolve(
            layer_init=_argument_layer_init,
            args=args,
            kwargs=kwargs,
            model=object(),
        )

        self.assertEqual(resolved_layer_idx, 2)

    def test_model_init_stack_resolver_resolves_list_comprehension_layer_idx(self):
        class Model:
            def __init__(self):
                self.layers = [_ModelInitStackResolvingLayer(self) for layer_idx in range(3)]

        model = Model()

        self.assertEqual([layer.resolved_layer_idx for layer in model.layers], [0, 1, 2])

    def test_model_init_stack_resolver_does_not_search_beyond_model_init(self):
        class Model:
            def __init__(self):
                self.layer = _ModelInitStackResolvingLayer(self)

        def initialize_model_with_outer_layer_idx(layer_idx):
            return Model()

        with self.assertRaisesRegex(RuntimeError, "model initialization stack up to and including"):
            initialize_model_with_outer_layer_idx(2)
