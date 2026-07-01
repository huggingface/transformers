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
from unittest.mock import patch

from parameterized import parameterized

from transformers.testing_utils import cleanup, is_torch_available, require_torch, torch_device


if is_torch_available():
    import torch

    from tests.heterogeneity.model_fixtures import MODEL_FIXTURES
    from tests.heterogeneity.testing_utils import (
        _build_model,
        _dummy_input_ids,
        _forward_logits,
        _hetero_context,
        _tiny_llama_config,
    )
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedModel
    from transformers.integrations.heterogeneity import HeterogeneousModelingSpec


if is_torch_available():
    # Toy composite models for testing the nested model construction paths.
    class _DefaultContextMarker(torch.nn.Module):
        pass

    class _SameLayerContextMarker(torch.nn.Module):
        pass

    class _OuterContextMarker(torch.nn.Module):
        pass

    class _InnerContextMarker(torch.nn.Module):
        pass

    def _init_toy_layer(layer, config, layer_idx):
        torch.nn.Module.__init__(layer)
        layer.layer_idx = layer_idx
        layer.intermediate_size = config.intermediate_size
        layer.context_marker = _DefaultContextMarker()

    class _SameLayerToyLayer(torch.nn.Module):
        def __init__(self, config, layer_idx):
            _init_toy_layer(self, config, layer_idx)

    class _OuterToyLayer(torch.nn.Module):
        def __init__(self, config, layer_idx):
            _init_toy_layer(self, config, layer_idx)

    class _InnerToyLayer(torch.nn.Module):
        def __init__(self, config, layer_idx):
            _init_toy_layer(self, config, layer_idx)

    class _StackLookupToyLayer(torch.nn.Module):
        def __init__(self, config):
            torch.nn.Module.__init__(self)
            self.intermediate_size = config.intermediate_size

    def _toy_modeling_spec(layer_cls, marker_replacement_cls):
        return HeterogeneousModelingSpec(
            layer_cls=layer_cls,
            layer_idx_variable_name="layer_idx",
            skip_descriptors={"context_marker": {"context_marker": marker_replacement_cls}},
        )

    def _toy_config(intermediate_size, replace_context_marker=False):
        layer_config = {"intermediate_size": intermediate_size}
        if replace_context_marker:
            layer_config["skip"] = ["context_marker"]
        return _tiny_llama_config(per_layer_config={0: layer_config})

    class _NestedSameLayerToyModel(PreTrainedModel):
        config_class = LlamaConfig
        _heterogeneous_modeling_spec = _toy_modeling_spec(_SameLayerToyLayer, _SameLayerContextMarker)

        def __init__(self, config, inner_config=None):
            super().__init__(config)
            if inner_config is not None:
                self.inner_model = _NestedSameLayerToyModel(inner_config)
            self.layer = _SameLayerToyLayer(config, layer_idx=0)

    class _ToyPreTrainedModel(PreTrainedModel):
        config_class = LlamaConfig

    class _LayerIdxStackLookupToyModel(_ToyPreTrainedModel):
        _heterogeneous_modeling_spec = HeterogeneousModelingSpec(
            layer_cls=_StackLookupToyLayer, layer_idx_variable_name="layer_idx"
        )

        def __init__(self, config, layer_idx=0):
            super().__init__(config)
            self.layer = _StackLookupToyLayer(config)

    class _InterleavedOuterToyModel(_ToyPreTrainedModel):
        _heterogeneous_modeling_spec = _toy_modeling_spec(_OuterToyLayer, _OuterContextMarker)

        def __init__(self, config, inner_config):
            super().__init__(config)
            self.inner_model = _InterleavedInnerToyModel(
                inner_config,
                outer_layer_factory=lambda: _OuterToyLayer(config, layer_idx=0),
            )

    class _InterleavedInnerToyModel(_ToyPreTrainedModel):
        _heterogeneous_modeling_spec = _toy_modeling_spec(_InnerToyLayer, _InnerContextMarker)

        def __init__(self, config, outer_layer_factory):
            super().__init__(config)
            # Build an outer-owned layer while the inner model context is active.
            self.outer_layer_from_inner_init = outer_layer_factory()
            self.layer = _InnerToyLayer(config, layer_idx=0)


@require_torch
class TestHeterogeneousModeling(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_layer_configs_reflect_model_init_attention_implementation(self):
        config = _tiny_llama_config(per_layer_config={0: {"intermediate_size": 64}})
        self.assertIsNone(config._attn_implementation)

        with _hetero_context("llama"):
            model = _build_model(config, LlamaForCausalLM)

        expected_attn_implementation = model.config._attn_implementation
        self.assertIsNotNone(expected_attn_implementation)
        for layer in model.model.layers:
            self.assertEqual(layer.self_attn.config._attn_implementation, expected_attn_implementation)

    def test_no_leaked_state_after_heterogeneous_init(self):
        """After heterogeneous model init, no context token should remain and non-heterogeneous models should work."""
        with _hetero_context("llama"):
            config = _tiny_llama_config(per_layer_config={1: {"skip": ["attention"]}})
            model = _build_model(config, LlamaForCausalLM)

        self.assertFalse(hasattr(model, "_layer_init_context_token"))

        # A new non-heterogeneous model should work fine
        plain_model = _build_model(_tiny_llama_config(), LlamaForCausalLM)
        _forward_logits(plain_model, _dummy_input_ids())

    def test_error_missing_skip_descriptor(self):
        """Requesting a skip type without a matching descriptor should raise ValueError."""
        config = _tiny_llama_config(per_layer_config={1: {"skip": ["attention"]}})
        fixture = MODEL_FIXTURES["llama"]
        modeling_spec = fixture.spec_factory()
        modeling_spec = HeterogeneousModelingSpec(
            layer_cls=modeling_spec.layer_cls,
            layer_idx_variable_name=modeling_spec.layer_idx_variable_name,
            skip_descriptors={},
        )
        with patch.object(fixture.pretrained_cls, "_heterogeneous_modeling_spec", modeling_spec, create=True):
            with self.assertRaisesRegex(ValueError, "No-op descriptors are missing"):
                _build_model(config, LlamaForCausalLM)

    def test_model_with_same_layer_class_submodel_initializes_each_model_layers(self):
        """A model containing a same-layer-class submodel should initialize each model's layers correctly."""
        model = _NestedSameLayerToyModel(
            _toy_config(intermediate_size=32),
            inner_config=_toy_config(intermediate_size=64, replace_context_marker=True),
        )

        self.assertEqual(model.layer.intermediate_size, 32)
        self.assertIsInstance(model.layer.context_marker, _DefaultContextMarker)
        self.assertEqual(model.inner_model.layer.intermediate_size, 64)
        self.assertIsInstance(model.inner_model.layer.context_marker, _SameLayerContextMarker)

    def test_submodel_can_construct_outer_model_layer_during_initialization(self):
        """A submodel should be able to construct an outer-model layer during its own initialization."""
        model = _InterleavedOuterToyModel(
            _toy_config(intermediate_size=32, replace_context_marker=True),
            inner_config=_toy_config(intermediate_size=64, replace_context_marker=True),
        )

        outer_layer = model.inner_model.outer_layer_from_inner_init
        inner_layer = model.inner_model.layer

        self.assertEqual(outer_layer.intermediate_size, 32)
        self.assertIsInstance(outer_layer.context_marker, _OuterContextMarker)
        self.assertEqual(inner_layer.intermediate_size, 64)
        self.assertIsInstance(inner_layer.context_marker, _InnerContextMarker)

    def test_layer_index_can_be_resolved_from_stack(self):
        model = _LayerIdxStackLookupToyModel(_toy_config(intermediate_size=64))

        self.assertEqual(model.layer.intermediate_size, 64)

    @parameterized.expand(
        [
            ("bool", True, TypeError, "call stack must be an integer"),
            ("negative", -1, IndexError, "call stack is out of range"),
        ]
    )
    def test_invalid_layer_index_from_stack_fails_clearly(self, _, layer_idx, error_type, message):
        with self.assertRaisesRegex(error_type, message):
            _LayerIdxStackLookupToyModel(_toy_config(intermediate_size=64), layer_idx)

    def test_sequential_heterogeneous_models_no_interference(self):
        """Two heterogeneous models built sequentially should each have correct per-layer weights."""
        per_layer_a = {0: {"intermediate_size": 64}}
        per_layer_b = {0: {"intermediate_size": 96}}

        with _hetero_context("llama"):
            model_a = _build_model(_tiny_llama_config(per_layer_config=per_layer_a), LlamaForCausalLM)
            model_b = _build_model(_tiny_llama_config(per_layer_config=per_layer_b), LlamaForCausalLM, seed=123)

        self.assertEqual(model_a.model.layers[0].mlp.gate_proj.weight.shape[0], 64)
        self.assertEqual(model_b.model.layers[0].mlp.gate_proj.weight.shape[0], 96)

        input_ids = _dummy_input_ids()
        _forward_logits(model_a, input_ids)
        _forward_logits(model_b, input_ids)
