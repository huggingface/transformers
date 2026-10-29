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

import dataclasses
import tempfile
import unittest
from unittest.mock import patch

import pytest
from parameterized import parameterized

from transformers.testing_utils import cleanup, is_torch_available, require_torch, torch_device


if is_torch_available():
    import torch
    from torch._dynamo.testing import CompileCounter

    from tests.heterogeneity.model_fixtures import MODEL_FIXTURES
    from tests.heterogeneity.testing_utils import (
        build_model,
        dummy_input_ids,
        forward_logits,
        tiny_gpt_oss_config,
        tiny_llama4_config,
        tiny_llama_config,
        tiny_nemotron_h_config,
    )
    from transformers import (
        BambaConfig,
        BambaForCausalLM,
        DynamicCache,
        FalconH1Config,
        FalconH1ForCausalLM,
        GptOssForCausalLM,
        JambaConfig,
        JambaForCausalLM,
        Llama4ForCausalLM,
        LlamaConfig,
        LlamaForCausalLM,
        LlamaModel,
        NemotronHForCausalLM,
        PreTrainedConfig,
        PreTrainedModel,
        StaticCache,
    )
    from transformers.configuration_utils import get_head_shapes
    from transformers.integrations.heterogeneity import (
        HeterogeneousModelingSpec,
        LayerIdxFromArgument,
        ReturnEntry,
        get_skip_replacement_factory,
    )
    from transformers.integrations.heterogeneity.masking_utils import AttentionMasksByLayerIdx
    from transformers.modeling_layers import MtpModel
    from transformers.models.bamba.modeling_bamba import (
        BambaAttention,
        BambaDecoderLayer,
        BambaMixer,
        BambaPreTrainedModel,
        BambaRMSNorm,
    )
    from transformers.models.falcon_h1.modeling_falcon_h1 import (
        FalconH1Attention,
        FalconH1DecoderLayer,
        FalconH1PreTrainedModel,
    )
    from transformers.models.jamba.modeling_jamba import (
        JambaAttention,
        JambaAttentionDecoderLayer,
        JambaMambaDecoderLayer,
        JambaMambaMixer,
        JambaMLP,
        JambaPreTrainedModel,
        JambaRMSNorm,
        JambaSparseMoeBlock,
    )
    from transformers.models.llama.modeling_llama import LlamaRMSNorm


if is_torch_available():

    class _ToyAttention(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class _ToyNoOpAttention(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class _SelectedNoOpAttention(_ToyNoOpAttention):
        pass

    class _ToyDecoderLayer(torch.nn.Module):
        def __init__(self, config, layer_idx):
            super().__init__()
            self.layer_idx = layer_idx
            self.intermediate_size = config.intermediate_size
            self.self_attn = _ToyAttention()

    def _toy_modeling_spec(layer_cls, attention_replacement_cls):
        return HeterogeneousModelingSpec(
            layer_cls=layer_cls,
            layer_idx_resolver=LayerIdxFromArgument("layer_idx"),
            skip_descriptors={"attention": {"self_attn": attention_replacement_cls}},
            attention_mask_consumers=[],
            cache_writers={},
        )

    def _toy_config(intermediate_size, skip_attention=False):
        layer_config = {"intermediate_size": intermediate_size}
        if skip_attention:
            layer_config["skip"] = ["attention"]
        return tiny_llama_config(per_layer_config={0: layer_config})

    class _ToyPreTrainedModel(PreTrainedModel):
        config_class = LlamaConfig

        def forward(self, hidden_states, past_key_values=None, use_cache=False):
            return hidden_states

    class _ToyModel(_ToyPreTrainedModel):
        _heterogeneous_modeling_spec = _toy_modeling_spec(_ToyDecoderLayer, _ToyNoOpAttention)

        def __init__(self, config):
            super().__init__(config)
            self.layers = torch.nn.ModuleList(
                _ToyDecoderLayer(config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)
            )

    class _MaskSelectingToyLayer(torch.nn.Module):
        def __init__(self, config, layer_idx):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, hidden_states, position_ids=None, attention_mask=None):
            return hidden_states * self.scale, attention_mask

    class _MaskSelectingToyModel(_ToyPreTrainedModel):
        _heterogeneous_modeling_spec = HeterogeneousModelingSpec(
            layer_cls=_MaskSelectingToyLayer,
            layer_idx_resolver=LayerIdxFromArgument("layer_idx"),
        )

        def __init__(self, config, layer_indices=None):
            super().__init__(config)
            if layer_indices is None:
                layer_indices = range(config.num_hidden_layers)
            self.layers = torch.nn.ModuleList(
                _MaskSelectingToyLayer(config, layer_idx=layer_idx) for layer_idx in layer_indices
            )

    class _TwoTowerConfig(PreTrainedConfig):
        model_type = "two_tower_test"
        sub_configs = {"text_config": LlamaConfig}

        def __init__(self, text_config=None, **kwargs):
            self.text_config = text_config
            super().__init__(**kwargs)

    class _TwoTowerModel(PreTrainedModel):
        config_class = _TwoTowerConfig

        def __init__(self, config):
            super().__init__(config)
            self.first = LlamaModel(config.text_config)
            self.second = LlamaModel(config.text_config)
            self.post_init()

    def _jamba_modeling_specs():
        """Jamba builds its attention and Mamba layers from different classes, so it has a spec for each."""
        passthrough_norm = get_skip_replacement_factory(
            JambaRMSNorm, ReturnEntry(arg_name="hidden_states", transform=lambda hidden_states: hidden_states)
        )
        mlp_skip_descriptor = {
            "pre_ff_layernorm": passthrough_norm,
            ("feed_forward", JambaMLP): get_skip_replacement_factory(
                JambaMLP, ReturnEntry(arg_name="x", transform=torch.zeros_like)
            ),
            ("feed_forward", JambaSparseMoeBlock): get_skip_replacement_factory(
                JambaSparseMoeBlock, ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like)
            ),
        }
        attention_layer_spec = HeterogeneousModelingSpec(
            layer_cls=JambaAttentionDecoderLayer,
            layer_idx_resolver=LayerIdxFromArgument("layer_idx"),
            skip_descriptors={
                "mixer": {
                    "input_layernorm": passthrough_norm,
                    "self_attn": get_skip_replacement_factory(
                        JambaAttention, [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None]
                    ),
                },
                "mlp": mlp_skip_descriptor,
            },
            attention_mask_consumers=["self_attn"],
            cache_writers={"self_attn": ["key_value"]},
        )
        mamba_layer_spec = HeterogeneousModelingSpec(
            layer_cls=JambaMambaDecoderLayer,
            layer_idx_resolver=LayerIdxFromArgument("layer_idx"),
            skip_descriptors={
                "mixer": {
                    "input_layernorm": passthrough_norm,
                    "mamba": get_skip_replacement_factory(
                        JambaMambaMixer, ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like)
                    ),
                },
            },
            attention_mask_consumers=["mamba"],
            cache_writers={"mamba": ["linear_attention"]},
        )
        return attention_layer_spec, mamba_layer_spec

    def _tiny_jamba_config(per_layer_config):
        # Layers 0 and 2 are Mamba layers, and layers 1 and 3 are attention layers with a mixture of experts
        return JambaConfig(
            vocab_size=99,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            attn_layer_period=2,
            attn_layer_offset=1,
            expert_layer_period=2,
            expert_layer_offset=1,
            num_experts=2,
            per_layer_config=per_layer_config,
        )

    def _bamba_modeling_spec():
        """Bamba builds `self_attn` in attention layers and `mamba` in Mamba layers, from one layer class, so its spec
        has a skip type for each, and its declarations list both with their classes."""
        passthrough_norm = get_skip_replacement_factory(
            BambaRMSNorm, ReturnEntry(arg_name="hidden_states", transform=lambda hidden_states: hidden_states)
        )
        return HeterogeneousModelingSpec(
            layer_cls=BambaDecoderLayer,
            layer_idx_resolver=LayerIdxFromArgument("layer_idx"),
            skip_descriptors={
                "attention": {
                    "input_layernorm": passthrough_norm,
                    "self_attn": get_skip_replacement_factory(
                        BambaAttention, [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None]
                    ),
                },
                "mamba": {
                    "input_layernorm": passthrough_norm,
                    "mamba": get_skip_replacement_factory(
                        BambaMixer, ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like)
                    ),
                },
            },
            attention_mask_consumers=[("self_attn", BambaAttention), ("mamba", BambaMixer)],
            cache_writers={("self_attn", BambaAttention): ["key_value"], ("mamba", BambaMixer): ["linear_attention"]},
        )

    def _tiny_bamba_config(per_layer_config):
        # Layers 1 and 3 are attention layers, and layers 0 and 2 are Mamba layers
        return BambaConfig(
            vocab_size=99,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            attn_layer_indices=[1, 3],
            mamba_n_heads=8,
            mamba_d_state=16,
            mamba_chunk_size=16,
            per_layer_config=per_layer_config,
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

        model = build_model(config, LlamaForCausalLM)

        expected_attn_implementation = model.config._attn_implementation
        self.assertIsNotNone(expected_attn_implementation)
        for layer in model.model.layers:
            self.assertEqual(layer.self_attn.config._attn_implementation, expected_attn_implementation)

    def test_models_built_with_one_config_share_its_layer_configs(self):
        config = tiny_llama_config(per_layer_config={1: {"intermediate_size": 64}})
        first_model = build_model(config, LlamaForCausalLM)

        second_model = build_model(config, LlamaForCausalLM)

        self.assertEqual(second_model.model.layers[1].mlp.gate_proj.out_features, 64)
        for first_layer, second_layer in zip(first_model.model.layers, second_model.model.layers):
            self.assertIs(second_layer.self_attn.config, first_layer.self_attn.config)

    def test_models_sharing_a_config_must_agree_on_skipped_mask_consumers_and_cache_writers(self):
        config = tiny_llama_config(per_layer_config={1: {"skip": ["attention"]}})
        build_model(config, LlamaForCausalLM)
        fixture = MODEL_FIXTURES["llama"]
        spec = dataclasses.replace(fixture.spec_factory(), attention_mask_consumers=[], cache_writers={})

        with patch.object(fixture.pretrained_cls, "_heterogeneous_modeling_spec", spec, create=True):
            with self.assertRaisesRegex(
                ValueError, "differ in which layers are left without attention mask consumers"
            ):
                build_model(config, LlamaForCausalLM)

    def test_towers_built_from_one_sub_config_share_its_layer_configs(self):
        text_config = tiny_llama_config(per_layer_config={1: {"intermediate_size": 64}})

        model = _TwoTowerModel(_TwoTowerConfig(text_config=text_config))

        for layer_idx, (first_layer, second_layer) in enumerate(zip(model.first.layers, model.second.layers)):
            self.assertIs(first_layer.self_attn.config, second_layer.self_attn.config)
            self.assertIs(text_config.per_layer_config[layer_idx], first_layer.self_attn.config)

    def test_error_missing_skip_descriptor(self):
        config = tiny_llama_config(per_layer_config={1: {"skip": ["attn"]}})
        with self.assertRaisesRegex(ValueError, "Skip descriptors are missing.*'attn'"):
            build_model(config, LlamaForCausalLM)

    def test_error_when_spec_layer_cls_is_not_constructed_for_every_layer(self):
        """If some layers are not built from the spec's layer class, their overrides would be ignored."""
        config = tiny_llama_config(per_layer_config={1: {"intermediate_size": 64}})
        with self.assertRaisesRegex(ValueError, r"for layers \[1, 3\]"):
            _MaskSelectingToyModel(config, layer_indices=[0, 2])

    @parameterized.expand(
        [
            ("member_class_key_listed_last", ["plain", "base_class", "member_class"]),
            ("member_class_key_listed_first", ["member_class", "base_class"]),
        ]
    )
    def test_skip_replacement_selection(self, _, key_names):
        # The cases name the keys because the toy classes exist only when torch is available
        member_keys = {
            "plain": "self_attn",
            "member_class": ("self_attn", _ToyAttention),
            "base_class": ("self_attn", torch.nn.Module),
        }
        spec = _toy_modeling_spec(_ToyDecoderLayer, _ToyNoOpAttention)
        # The key with the member's own class is the most specific, so it wins wherever it's listed
        spec.skip_descriptors["attention"] = {
            member_keys[name]: _SelectedNoOpAttention if name == "member_class" else _ToyNoOpAttention
            for name in key_names
        }
        with patch.object(_ToyModel, "_heterogeneous_modeling_spec", spec):
            model = _ToyModel(_toy_config(intermediate_size=32, skip_attention=True))

        self.assertIs(type(model.layers[0].self_attn), _SelectedNoOpAttention)

    def test_unmatched_skip_member_raises_without_breaking_later_builds(self):
        config = tiny_llama_config(per_layer_config={3: {"skip": ["attention"]}})
        fixture = MODEL_FIXTURES["llama"]
        spec = fixture.spec_factory()
        spec.skip_descriptors["attention"] = {
            ("self_attn", torch.nn.Linear): torch.nn.Identity,
            ("self_attn", torch.nn.Conv1d): torch.nn.Identity,
        }

        with patch.object(fixture.pretrained_cls, "_heterogeneous_modeling_spec", spec, create=True):
            with self.assertRaisesRegex(
                ValueError,
                "Layer 3.*'attention'.*no replacement.*'self_attn'.*LlamaAttention",
            ):
                build_model(config, LlamaForCausalLM)

        # The build failed partway through its layers, and what it left behind must not affect the next build
        model = build_model(tiny_llama_config(per_layer_config={1: {"intermediate_size": 64}}), LlamaForCausalLM)
        self.assertEqual(model.model.layers[1].mlp.gate_proj.out_features, 64)

    @parameterized.expand([("no_main_skips", []), ("main_layers_skipped", ["attention", "mlp"])])
    def test_mtp_model_applies_per_layer_config(self, _, main_skips):
        config = tiny_llama_config(
            num_hidden_layers=2,
            per_layer_config={layer_idx: {"skip": main_skips} for layer_idx in range(2)} if main_skips else None,
        )
        config.num_mtp_layers = 2
        config.mtp_per_layer_config = {
            0: {"intermediate_size": 64, "rms_norm_eps": 1e-5},
            1: {"intermediate_size": 96},
        }
        main_model = build_model(config, LlamaForCausalLM)

        mtp_model = MtpModel(main_model, num_mtp_layers=2)

        self.assertTrue(mtp_model.config.is_heterogeneous)
        self.assertTrue(mtp_model.config.generic_heterogeneous_modeling_applied)
        self.assertEqual(mtp_model.layers[0].mtp_block.mlp.gate_proj.out_features, 64)
        self.assertEqual(mtp_model.layers[1].mtp_block.mlp.gate_proj.out_features, 96)
        self.assertEqual(mtp_model.layers[0].enorm.variance_epsilon, 1e-5)
        for layer in mtp_model.layers:
            for norm in (layer.enorm, layer.hnorm, layer.post_norm):
                self.assertIsInstance(norm, LlamaRMSNorm)

    def test_mtp_model_rejects_skip(self):
        config = tiny_llama_config(num_hidden_layers=2)
        config.num_mtp_layers = 2
        config.mtp_per_layer_config = {0: {"intermediate_size": 64}, 1: {"skip": ["attention"]}}
        main_model = build_model(config, LlamaForCausalLM)

        with self.assertRaisesRegex(ValueError, r"MTP layers don't support `skip`.*layers \[1\]"):
            MtpModel(main_model, num_mtp_layers=2)

    def test_mtp_mask_creation_uses_per_layer_config(self):
        config = tiny_gpt_oss_config(num_hidden_layers=2, layer_types=["sliding_attention"] * 2, sliding_window=4)
        config.num_mtp_layers = 2
        config.mtp_layer_types = ["sliding_attention"] * 2
        config.mtp_per_layer_config = {
            0: {"sliding_window": 2},
            1: {"sliding_window": 3},
        }
        config._attn_implementation = "eager"
        main_model = build_model(config, GptOssForCausalLM)
        self.assertFalse(main_model.config.generic_heterogeneous_modeling_applied)
        mtp_model = MtpModel(main_model, num_mtp_layers=2)

        inputs_embeds = torch.randn(1, 4, config.hidden_size)
        position_ids = torch.arange(4).unsqueeze(0)
        mtp_cache = DynamicCache(config=mtp_model.config)
        min_dtype = torch.finfo(inputs_embeds.dtype).min
        expected_last_rows = ([min_dtype, min_dtype, 0.0, 0.0], [min_dtype, 0.0, 0.0, 0.0])
        for layer_idx, expected_last_row in enumerate(expected_last_rows):
            mask = mtp_model.create_masks_for_mtp_layer(layer_idx, inputs_embeds, mtp_cache, position_ids)[
                "attention_mask"
            ]
            torch.testing.assert_close(mask[0, 0, -1], torch.tensor(expected_last_row))

    def test_layer_built_without_layer_idx_fails_clearly(self):
        with self.assertRaisesRegex(TypeError, "must be an integer.*None"):
            _MaskSelectingToyModel(_toy_config(intermediate_size=64), layer_indices=[None])

    def test_layer_forward_selects_attention_mask_by_layer_idx(self):
        model = _MaskSelectingToyModel(_toy_config(intermediate_size=64))
        masks = AttentionMasksByLayerIdx({0: "layer-zero-mask", 2: "layer-two-mask"})

        self.assertEqual(model.layers[2](torch.zeros(1), attention_mask=masks)[1], "layer-two-mask")
        self.assertEqual(model.layers[2](torch.zeros(1), None, masks)[1], "layer-two-mask")

    def test_data_parallel_replica_uses_its_own_parameters(self):
        model = _MaskSelectingToyModel(_toy_config(intermediate_size=64))
        # Exercise DataParallel's shallow replication without requiring GPUs.
        copied_layer = model.layers[2]._replicate_for_data_parallel()
        copied_layer.scale = torch.nn.Parameter(torch.tensor(7.0))
        masks = AttentionMasksByLayerIdx({0: "layer-zero-mask", 2: "layer-two-mask"})

        output, selected_mask = copied_layer(torch.tensor(3.0), attention_mask=masks)

        torch.testing.assert_close(output, torch.tensor(21.0))
        self.assertEqual(selected_mask, "layer-two-mask")

    def test_sequential_heterogeneous_models_no_interference(self):
        per_layer_a = {0: {"intermediate_size": 64}}
        per_layer_b = {0: {"intermediate_size": 96}}

        model_a = build_model(tiny_llama_config(per_layer_config=per_layer_a), LlamaForCausalLM)
        input_ids = dummy_input_ids()
        expected_logits = forward_logits(model_a, input_ids)
        model_b = build_model(tiny_llama_config(per_layer_config=per_layer_b), LlamaForCausalLM, seed=123)

        self.assertEqual(model_a.model.layers[0].mlp.gate_proj.weight.shape[0], 64)
        self.assertEqual(model_b.model.layers[0].mlp.gate_proj.weight.shape[0], 96)

        torch.testing.assert_close(forward_logits(model_a, input_ids), expected_logits)
        forward_logits(model_b, input_ids)

    def test_save_pretrained_model_round_trip(self):
        """Full model save/load: skips, weight shapes, and forward output should survive."""
        per_layer = {
            0: {"intermediate_size": 64},
            1: {"skip": ["attention"]},
            2: {"intermediate_size": 96},
        }
        hetero_config = tiny_llama_config(per_layer_config=per_layer)
        hetero_model = build_model(hetero_config, LlamaForCausalLM)

        input_ids = dummy_input_ids()
        expected_logits = forward_logits(hetero_model, input_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            hetero_model.save_pretrained(tmpdir)
            loaded_model = LlamaForCausalLM.from_pretrained(tmpdir)

        loaded_model.eval()
        self.assertEqual(list(loaded_model.model.layers[1].self_attn.parameters()), [])
        for layer_idx in range(hetero_config.num_hidden_layers):
            orig_shape = hetero_model.model.layers[layer_idx].mlp.gate_proj.weight.shape
            loaded_shape = loaded_model.model.layers[layer_idx].mlp.gate_proj.weight.shape
            self.assertEqual(orig_shape, loaded_shape, f"Layer {layer_idx} weight shape mismatch")

        torch.testing.assert_close(forward_logits(loaded_model, input_ids), expected_logits)


@require_torch
class TestHeterogeneousCache(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @parameterized.expand(
        [
            (
                "llama",
                "llama",
                {0: {"num_key_value_heads": 2}, 1: {"skip": ["attention"]}, 2: {"num_key_value_heads": 1}},
            ),
            # Caches read positions from layer 0 by default, and here nothing writes to its cache
            ("llama_first_attention_skipped", "llama", {0: {"skip": ["attention"]}}),
            ("gpt_oss", "gpt_oss", {0: {"sliding_window": 3}, 1: {"skip": ["attention"]}, 2: {"sliding_window": 4}}),
            (
                "llama4",
                "llama4",
                {0: {"attention_chunk_size": 2}, 1: {"attention_chunk_size": 3}, 2: {"skip": ["attention"]}},
            ),
            ("nemotron_h", "nemotron_h", {1: {"skip": ["mixer"]}, 3: {"skip": ["mixer"]}}),
        ]
    )
    def test_cached_decoding_matches_uncached(self, _name, model_name, overrides):
        factory, model_class = {
            "llama": (tiny_llama_config, LlamaForCausalLM),
            "gpt_oss": (tiny_gpt_oss_config, GptOssForCausalLM),
            "llama4": (tiny_llama4_config, Llama4ForCausalLM),
            "nemotron_h": (tiny_nemotron_h_config, NemotronHForCausalLM),
        }[model_name]
        config = factory(per_layer_config=overrides)
        model = build_model(config, model_class)
        input_ids = torch.tensor([[0, 0, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
        attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
        # Like the static cache of a chunked prefill
        early_initialized_static_cache = StaticCache(config=config, max_cache_len=input_ids.shape[1])
        early_initialized_static_cache.early_initialization(
            input_ids.shape[0], *get_head_shapes(config), model.dtype, model.device
        )
        caches = {
            "dynamic": DynamicCache(config=config),
            "static": StaticCache(config=config, max_cache_len=input_ids.shape[1]),
            "early_initialized_static": early_initialized_static_cache,
        }

        with torch.no_grad():
            expected_logits = model(input_ids, attention_mask=attention_mask, use_cache=False).logits[:, -2:]
            for cache_name, cache in caches.items():
                with self.subTest(cache=cache_name):
                    model(
                        input_ids[:, :-2], attention_mask=attention_mask[:, :-2], past_key_values=cache, use_cache=True
                    )
                    actual_logits = model(
                        input_ids[:, -2:], attention_mask=attention_mask, past_key_values=cache, use_cache=True
                    ).logits
                    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-4, atol=1e-5)

    def test_cache_not_built_from_the_config_raises(self):
        config = tiny_llama_config(per_layer_config={0: {"skip": ["attention"]}})
        model = build_model(config, LlamaForCausalLM)

        # Without the config, the cache fills layer 0 with a key/value cache that nothing writes, and positions would be
        # read from it
        with self.assertRaisesRegex(ValueError, "holds a key_value cache for layer 0"):
            model(dummy_input_ids(), past_key_values=DynamicCache(), use_cache=True)

    def test_model_with_several_layer_classes(self):
        config = _tiny_jamba_config(
            {
                0: {"skip": ["mixer"]},
                1: {"intermediate_size": 32},
                2: {"intermediate_size": 48},
                3: {"skip": ["mixer", "mlp"]},
            }
        )
        with patch.object(JambaPreTrainedModel, "_heterogeneous_modeling_spec", _jamba_modeling_specs(), create=True):
            model = build_model(config, JambaForCausalLM)
        layers = model.model.layers

        self.assertEqual(layers[1].feed_forward.experts.down_proj.shape[-1], 32)
        self.assertEqual(layers[2].feed_forward.down_proj.in_features, 48)
        self.assertEqual(list(layers[0].mamba.parameters()), [])
        self.assertEqual(list(layers[3].self_attn.parameters()), [])
        self.assertEqual(list(layers[3].feed_forward.parameters()), [])
        # Both skipped mixers were their layer's only cache writer
        self.assertEqual(
            [type(layer).__name__ for layer in DynamicCache(config=config).layers],
            ["NoCacheLayer", "DynamicLayer", "LinearAttentionLayer", "NoCacheLayer"],
        )

        input_ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
        cache = DynamicCache(config=config)
        with torch.no_grad():
            expected_logits = model(input_ids, use_cache=False).logits[:, -2:]
            model(input_ids[:, :-2], past_key_values=cache, use_cache=True)
            actual_logits = model(input_ids[:, -2:], past_key_values=cache, use_cache=True).logits

        torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-4, atol=1e-3)

    def test_skip_type_missing_from_the_spec_of_a_layer_class_raises(self):
        # Only the attention layer spec describes "mlp", and layer 0 is a Mamba layer
        config = _tiny_jamba_config({0: {"skip": ["mlp"]}})

        with patch.object(JambaPreTrainedModel, "_heterogeneous_modeling_spec", _jamba_modeling_specs(), create=True):
            with self.assertRaisesRegex(ValueError, r"Layer 0 skips \['mlp'\].*`JambaMambaDecoderLayer`"):
                build_model(config, JambaForCausalLM)

    def test_declarations_can_list_members_that_only_some_layers_build(self):
        config = _tiny_bamba_config({0: {"skip": ["mamba"]}, 1: {"skip": ["attention"]}})
        with patch.object(BambaPreTrainedModel, "_heterogeneous_modeling_spec", _bamba_modeling_spec(), create=True):
            build_model(config, BambaForCausalLM)

        # Each skipped mixer was its layer's only cache writer
        self.assertEqual(
            [type(layer).__name__ for layer in DynamicCache(config=config).layers],
            ["NoCacheLayer", "NoCacheLayer", "LinearAttentionLayer", "DynamicLayer"],
        )

    def test_skip_type_with_a_member_the_layer_doesnt_build_raises(self):
        spec = _bamba_modeling_spec()
        attention_descriptor = spec.skip_descriptors["attention"]
        attention_descriptor[("self_attn", BambaAttention)] = attention_descriptor.pop("self_attn")
        # Unlike the declarations, a skip descriptor needs its members even with a class key. Layer 0 is a Mamba
        # layer, so it has no `self_attn`, and the skip would replace only its norm.
        config = _tiny_bamba_config({0: {"skip": ["attention"]}})

        with patch.object(BambaPreTrainedModel, "_heterogeneous_modeling_spec", spec, create=True):
            with self.assertRaisesRegex(AttributeError, "Layer 0 skips 'attention'.*has no submodule 'self_attn'"):
                build_model(config, BambaForCausalLM)

    def test_declaration_key_without_a_class_must_exist_in_the_layer(self):
        spec = dataclasses.replace(_bamba_modeling_spec(), attention_mask_consumers=["self_attn"])
        # Layer 0 is a Mamba layer, so it has no `self_attn`
        config = _tiny_bamba_config({0: {"skip": ["mamba"]}})

        with patch.object(BambaPreTrainedModel, "_heterogeneous_modeling_spec", spec, create=True):
            with self.assertRaisesRegex(AttributeError, "`attention_mask_consumers`.*has no submodule 'self_attn'"):
                build_model(config, BambaForCausalLM)

    def test_cached_decoding_with_attention_skipped_in_a_hybrid_layer(self):
        # Falcon-H1 runs attention and Mamba in parallel in every layer, and each layer's cache entry holds the
        # key/value cache of the attention and the linear-attention cache of the Mamba
        spec = HeterogeneousModelingSpec(
            layer_cls=FalconH1DecoderLayer,
            layer_idx_resolver=LayerIdxFromArgument("layer_idx"),
            skip_descriptors={
                "attention": {
                    "self_attn": get_skip_replacement_factory(
                        FalconH1Attention, [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None]
                    )
                }
            },
            attention_mask_consumers=["self_attn"],
            cache_writers={"self_attn": ["key_value"], "mamba": ["linear_attention"]},
        )
        config = FalconH1Config(
            vocab_size=99,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            mamba_n_heads=16,
            mamba_d_state=16,
            mamba_chunk_size=16,
            # Caches read positions from layer 0 by default, and here only the Mamba writes to its cache entry
            per_layer_config={0: {"skip": ["attention"]}},
        )
        with patch.object(FalconH1PreTrainedModel, "_heterogeneous_modeling_spec", spec, create=True):
            model = build_model(config, FalconH1ForCausalLM)
        input_ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
        cache = DynamicCache(config=config)

        with torch.no_grad():
            expected_logits = model(input_ids, use_cache=False).logits[:, -2:]
            model(input_ids[:, :-2], past_key_values=cache, use_cache=True)
            actual_logits = model(input_ids[:, -2:], past_key_values=cache, use_cache=True).logits

        torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-4, atol=1e-5)

    @parameterized.expand(
        [
            # Llama has no layer types, so its forward passes the masks that `generate` built to `create_causal_mask`
            ("llama", "llama", {1: {"skip": ["attention"]}}),
            ("gpt_oss", "gpt_oss", {0: {"sliding_window": 3}, 1: {"skip": ["attention"]}, 2: {"sliding_window": 4}}),
        ]
    )
    def test_static_cache_generation_with_skipped_attention(self, _name, model_name, per_layer_config):
        factory, model_class = {
            "llama": (tiny_llama_config, LlamaForCausalLM),
            "gpt_oss": (tiny_gpt_oss_config, GptOssForCausalLM),
        }[model_name]
        config = factory(per_layer_config=per_layer_config, pad_token_id=0, eos_token_id=None)
        config._attn_implementation = "eager"
        model = build_model(config, model_class)
        input_ids = torch.tensor([[1, 3, 4, 5]])
        generation_kwargs = {
            "max_new_tokens": 3,
            "do_sample": False,
            "return_dict_in_generate": True,
            "output_logits": True,
        }
        with torch.no_grad():
            expected = model.generate(input_ids, use_cache=False, **generation_kwargs)
            actual = model.generate(
                input_ids, cache_implementation="static", disable_compile=True, **generation_kwargs
            )

        torch.testing.assert_close(actual.sequences, expected.sequences)
        torch.testing.assert_close(actual.logits, expected.logits)

    @pytest.mark.torch_compile_test
    def test_static_cache_decoding_compiles_fullgraph(self):
        config = tiny_llama_config(num_hidden_layers=3, per_layer_config={1: {"skip": ["attention"]}})
        # sdpa never skips the mask when decoding with a static cache, so the masks are always built
        config._attn_implementation = "sdpa"
        model = build_model(config, LlamaForCausalLM)
        input_ids = torch.tensor([[1, 3, 4, 5]])
        cache = StaticCache(config=config, max_cache_len=input_ids.shape[1])
        # Unlike with `generate`, compiling the model itself also builds the masks inside the graph
        compile_counter = CompileCounter()
        compiled_model = torch.compile(model, backend=compile_counter, fullgraph=True)

        with torch.no_grad():
            expected_logits = model(input_ids, use_cache=False).logits[:, -2:]
            # Prefill without compiling, like `generate` does, so the compiled steps see an initialized cache
            model(input_ids[:, :-2], past_key_values=cache, use_cache=True)
            torch.compiler.reset()
            with torch._dynamo.config.patch(error_on_recompile=True):
                actual_logits = torch.cat(
                    [
                        compiled_model(input_ids[:, [position]], past_key_values=cache, use_cache=True).logits
                        for position in (2, 3)
                    ],
                    dim=1,
                )

        # Both decoding steps ran through one compiled graph
        self.assertEqual(compile_counter.frame_count, 1)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-4, atol=1e-5)

    def test_assisted_generation_attention_outputs_with_skipped_attention(self):
        # Skipping layer 0 means the first layer returns no attentions, while the others still do
        config = tiny_llama_config(
            num_hidden_layers=3, per_layer_config={0: {"skip": ["attention"]}}, pad_token_id=0, eos_token_id=None
        )
        config._attn_implementation = "eager"
        model = build_model(config, LlamaForCausalLM)
        model.generation_config.num_assistant_tokens = 2
        model.generation_config.num_assistant_tokens_schedule = "constant"
        model.generation_config.assistant_confidence_threshold = 0.0
        input_ids = torch.tensor([[1, 3, 4, 5]])

        with torch.no_grad():
            actual = model.generate(
                input_ids,
                assistant_model=model,
                max_new_tokens=5,
                do_sample=False,
                return_dict_in_generate=True,
                output_attentions=True,
            )

        self.assertEqual(len(actual.attentions), 5)
        for step, attentions in enumerate(actual.attentions):
            self.assertEqual(len(attentions), 3)
            self.assertIsNone(attentions[0])
            query_length = input_ids.shape[1] if step == 0 else 1
            expected_shape = (1, config.num_attention_heads, query_length, input_ids.shape[1] + step)
            for layer_idx in (1, 2):
                self.assertEqual(attentions[layer_idx].shape, expected_shape)
