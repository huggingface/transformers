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
        build_model,
        dummy_input_ids,
        forward_logits,
        hetero_context,
        tiny_llama_config,
    )
    from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM, PreTrainedModel
    from transformers.integrations.heterogeneity import (
        HeterogeneousModelingSpec,
        SkipDescriptor,
    )
    from transformers.modeling_layers import MtpModel


if is_torch_available():
    # Toy composite models for testing the nested model construction paths.
    class _ToyAttention(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class _ToyNoOpAttention(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class _CompositeNoOpAttention(_ToyNoOpAttention):
        pass

    class _BackboneNoOpAttention(_ToyNoOpAttention):
        pass

    class _ToyDecoderLayer(torch.nn.Module):
        def __init__(self, config, layer_idx):
            super().__init__()
            self.layer_idx = layer_idx
            self.intermediate_size = config.intermediate_size
            self.self_attn = _ToyAttention()

    class _StackLookupToyLayer(torch.nn.Module):
        def __init__(self, config):
            torch.nn.Module.__init__(self)
            self.intermediate_size = config.intermediate_size

    def _toy_modeling_spec(layer_cls, attention_replacement_cls):
        return HeterogeneousModelingSpec(
            layer_cls=layer_cls,
            layer_idx_variable_name="layer_idx",
            skip_descriptors={
                "attention": SkipDescriptor(
                    replacements={"self_attn": attention_replacement_cls},
                    replaces_kv_cache_updater=True,
                )
            },
        )

    def _toy_config(intermediate_size, skip_attention=False):
        layer_config = {"intermediate_size": intermediate_size}
        if skip_attention:
            layer_config["skip"] = ["attention"]
        return tiny_llama_config(per_layer_config={0: layer_config})

    class _ToyPreTrainedModel(PreTrainedModel):
        config_class = LlamaConfig

        def __init__(self, config):
            super().__init__(config)

    class _NestedSameLayerToyModel(_ToyPreTrainedModel):
        _heterogeneous_modeling_spec = _toy_modeling_spec(_ToyDecoderLayer, _ToyNoOpAttention)

        def __init__(self, config, inner_config=None):
            super().__init__(config)
            if inner_config is not None:
                self.inner_model = _NestedSameLayerToyModel(inner_config)
            self.layer = _ToyDecoderLayer(config, layer_idx=0)

    class _LayerIdxStackLookupToyModel(_ToyPreTrainedModel):
        _heterogeneous_modeling_spec = HeterogeneousModelingSpec(
            layer_cls=_StackLookupToyLayer, layer_idx_variable_name="layer_idx"
        )

        def __init__(self, config, layer_idx=0):
            super().__init__(config)
            self.layer = _StackLookupToyLayer(config)

    class _CompositeToyModel(_ToyPreTrainedModel):
        _heterogeneous_modeling_spec = _toy_modeling_spec(_ToyDecoderLayer, _CompositeNoOpAttention)

        def __init__(self, config, backbone_config):
            super().__init__(config)
            self.backbone = _ToyBackboneModel(
                backbone_config,
                parent_layer_factory=lambda: _ToyDecoderLayer(config, layer_idx=0),
            )

    class _ToyBackboneModel(_ToyPreTrainedModel):
        _heterogeneous_modeling_spec = _toy_modeling_spec(_ToyDecoderLayer, _BackboneNoOpAttention)

        def __init__(self, config, parent_layer_factory):
            super().__init__(config)
            # Build a parent-owned layer while the backbone model context is active.
            self.parent_layer = parent_layer_factory()
            self.layer = _ToyDecoderLayer(config, layer_idx=0)

    class _ToyModel(PreTrainedModel):
        config_class = LlamaConfig
        _heterogeneous_modeling_spec = HeterogeneousModelingSpec(
            layer_cls=_StackLookupToyLayer, layer_idx_variable_name="layer_idx"
        )


@require_torch
class TestHeterogeneousModeling(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_layer_configs_reflect_model_init_attention_implementation(self):
        config = tiny_llama_config(per_layer_config={0: {"intermediate_size": 64}})
        self.assertIsNone(config._attn_implementation)

        with hetero_context("llama"):
            model = build_model(config, LlamaForCausalLM)

        expected_attn_implementation = model.config._attn_implementation
        self.assertIsNotNone(expected_attn_implementation)
        for layer in model.model.layers:
            self.assertEqual(layer.self_attn.config._attn_implementation, expected_attn_implementation)

    def test_no_leaked_state_after_heterogeneous_init(self):
        """After heterogeneous model init, no context token should remain and non-heterogeneous models should work."""
        with hetero_context("llama"):
            config = tiny_llama_config(per_layer_config={1: {"skip": ["attention"]}})
            model = build_model(config, LlamaForCausalLM)

        self.assertFalse(hasattr(model, "_layer_init_context_token"))

        inherited_init_model = _ToyModel(_toy_config(intermediate_size=64))
        self.assertFalse(hasattr(inherited_init_model, "_layer_init_context_token"))

        # A new non-heterogeneous model should work fine
        plain_model = build_model(tiny_llama_config(), LlamaForCausalLM)
        forward_logits(plain_model, dummy_input_ids())

    def test_error_missing_skip_descriptor(self):
        """Requesting a skip type without a matching descriptor should raise ValueError."""
        config = tiny_llama_config(per_layer_config={1: {"skip": ["attention"]}})
        fixture = MODEL_FIXTURES["llama"]
        modeling_spec = fixture.spec_factory()
        modeling_spec = HeterogeneousModelingSpec(
            layer_cls=modeling_spec.layer_cls,
            layer_idx_variable_name=modeling_spec.layer_idx_variable_name,
            skip_descriptors={},
        )
        with patch.object(fixture.pretrained_cls, "_heterogeneous_modeling_spec", modeling_spec, create=True):
            with self.assertRaisesRegex(ValueError, "No-op descriptors are missing"):
                build_model(config, LlamaForCausalLM)

    def test_model_and_submodel_initialize_shared_layer_class_independently(self):
        """A model containing a same-layer-class submodel should initialize each model's layers correctly."""
        model = _NestedSameLayerToyModel(
            _toy_config(intermediate_size=32),
            inner_config=_toy_config(intermediate_size=64, skip_attention=True),
        )

        self.assertEqual(model.layer.intermediate_size, 32)
        self.assertIsInstance(model.layer.self_attn, _ToyAttention)
        self.assertEqual(model.inner_model.layer.intermediate_size, 64)
        self.assertIsInstance(model.inner_model.layer.self_attn, _ToyNoOpAttention)
        self.assertFalse(hasattr(model, "_layer_init_context_token"))
        self.assertFalse(hasattr(model.inner_model, "_layer_init_context_token"))

    def test_nested_model_layers_use_their_own_config(self):
        model = _CompositeToyModel(
            _toy_config(intermediate_size=32, skip_attention=True),
            backbone_config=_toy_config(intermediate_size=64, skip_attention=True),
        )

        parent_layer = model.backbone.parent_layer
        backbone_layer = model.backbone.layer

        self.assertEqual(parent_layer.intermediate_size, 32)
        self.assertIsInstance(parent_layer.self_attn, _CompositeNoOpAttention)
        self.assertEqual(backbone_layer.intermediate_size, 64)
        self.assertIsInstance(backbone_layer.self_attn, _BackboneNoOpAttention)

    def test_layer_index_can_be_resolved_from_stack(self):
        model = _LayerIdxStackLookupToyModel(_toy_config(intermediate_size=64))

        self.assertEqual(model.layer.intermediate_size, 64)

    def test_mtp_model_applies_per_layer_config_and_skips(self):
        config = tiny_llama_config(num_hidden_layers=2)
        config.num_mtp_layers = 2
        config.mtp_per_layer_config = {
            0: {"intermediate_size": 64, "rms_norm_eps": 1e-5},
            1: {"intermediate_size": 96, "skip": ["attention"]},
        }
        main_model = build_model(config, LlamaForCausalLM)

        mtp_model = MtpModel(main_model, num_mtp_layers=2)

        self.assertTrue(mtp_model.config.is_heterogeneous)
        self.assertEqual(mtp_model.layers[0].mtp_block.mlp.gate_proj.out_features, 64)
        self.assertEqual(mtp_model.layers[1].mtp_block.mlp.gate_proj.out_features, 96)
        self.assertEqual(mtp_model.layers[0].enorm.variance_epsilon, 1e-5)
        self.assertIsNot(
            type(mtp_model.layers[1].mtp_block.self_attn),
            type(main_model.model.layers[1].self_attn),
        )
        self.assertEqual(mtp_model.config.get_disabled_kv_layer_indices(), (1,))

    def test_mtp_mask_creation_uses_per_layer_config(self):
        config = tiny_llama_config(num_hidden_layers=2)
        config.num_mtp_layers = 2
        config.mtp_per_layer_config = {
            0: {"is_causal": True},
            1: {"is_causal": False},
        }
        config._attn_implementation = "eager"
        main_model = build_model(config, LlamaForCausalLM)
        mtp_model = MtpModel(main_model, num_mtp_layers=2)

        inputs_embeds = torch.randn(1, 2, config.hidden_size)
        position_ids = torch.arange(2).unsqueeze(0)
        mtp_cache = DynamicCache(config=mtp_model.config)
        causal_mask = mtp_model.create_masks_for_mtp_layer(0, inputs_embeds, mtp_cache, position_ids)["attention_mask"]
        bidirectional_mask = mtp_model.create_masks_for_mtp_layer(1, inputs_embeds, mtp_cache, position_ids)[
            "attention_mask"
        ]
        self.assertIsNotNone(causal_mask)
        self.assertIsNone(bidirectional_mask)

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

        with hetero_context("llama"):
            model_a = build_model(tiny_llama_config(per_layer_config=per_layer_a), LlamaForCausalLM)
            model_b = build_model(tiny_llama_config(per_layer_config=per_layer_b), LlamaForCausalLM, seed=123)

        self.assertEqual(model_a.model.layers[0].mlp.gate_proj.weight.shape[0], 64)
        self.assertEqual(model_b.model.layers[0].mlp.gate_proj.weight.shape[0], 96)

        input_ids = dummy_input_ids()
        forward_logits(model_a, input_ids)
        forward_logits(model_b, input_ids)
