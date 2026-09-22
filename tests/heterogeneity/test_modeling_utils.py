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
        tiny_gpt_oss_config,
        tiny_llama4_config,
        tiny_llama_config,
        tiny_nemotron_h_config,
    )
    from transformers import (
        DynamicCache,
        GptOssForCausalLM,
        Llama4ForCausalLM,
        LlamaConfig,
        LlamaForCausalLM,
        NemotronHForCausalLM,
        PreTrainedModel,
        StaticCache,
    )
    from transformers.integrations.heterogeneity import (
        HeterogeneousModelingSpec,
        LayerIdxFromArgument,
    )
    from transformers.integrations.heterogeneity.masking_utils import AttentionMasksByLayerIdx
    from transformers.modeling_layers import MtpModel
    from transformers.models.llama.modeling_llama import LlamaRMSNorm


if is_torch_available():

    class _ToyAttention(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class _ToyNoOpAttention(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class _ClassSpecificNoOpAttention(_ToyNoOpAttention):
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

    class _SingleLayerToyModel(_ToyPreTrainedModel):
        _heterogeneous_modeling_spec = _toy_modeling_spec(_ToyDecoderLayer, _ToyNoOpAttention)

        def __init__(self, config):
            super().__init__(config)
            self.layer = _ToyDecoderLayer(config, layer_idx=0)

    class _MaskSelectingToyLayer(torch.nn.Module):
        def __init__(self, config, layer_idx):
            super().__init__()

        def forward(self, hidden_states, position_ids=None, attention_mask=None):
            return attention_mask

    class _MaskSelectingToyModel(_ToyPreTrainedModel):
        _heterogeneous_modeling_spec = HeterogeneousModelingSpec(
            layer_cls=_MaskSelectingToyLayer,
            layer_idx_resolver=LayerIdxFromArgument("layer_idx"),
        )

        def __init__(self, config, layer_idx=0):
            super().__init__(config)
            self.layer = _MaskSelectingToyLayer(config, layer_idx=layer_idx)


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

    def test_error_missing_skip_descriptor(self):
        """Requesting a skip type without a matching descriptor should raise ValueError."""
        config = tiny_llama_config(per_layer_config={1: {"skip": ["attention"]}})
        fixture = MODEL_FIXTURES["llama"]
        base_spec = fixture.spec_factory()
        modeling_spec = HeterogeneousModelingSpec(
            layer_cls=base_spec.layer_cls,
            layer_idx_resolver=base_spec.layer_idx_resolver,
            skip_descriptors={},
        )
        with patch.object(fixture.pretrained_cls, "_heterogeneous_modeling_spec", modeling_spec, create=True):
            with self.assertRaisesRegex(ValueError, "No-op descriptors are missing"):
                build_model(config, LlamaForCausalLM)

    def test_class_specific_skip_replacement_takes_precedence(self):
        spec = _toy_modeling_spec(_ToyDecoderLayer, _ToyNoOpAttention)
        spec.skip_descriptors["attention"] = {
            "self_attn": _ToyNoOpAttention,
            ("self_attn", _ToyAttention): _ClassSpecificNoOpAttention,
        }
        with patch.object(_SingleLayerToyModel, "_heterogeneous_modeling_spec", spec):
            model = _SingleLayerToyModel(_toy_config(intermediate_size=32, skip_attention=True))

        self.assertIsInstance(model.layer.self_attn, _ClassSpecificNoOpAttention)

    @parameterized.expand([("no_main_skips", []), ("main_layers_skipped", ["attention", "mlp"])])
    def test_mtp_model_applies_per_layer_config_and_skips(self, _, main_skips):
        config = tiny_llama_config(
            num_hidden_layers=2,
            per_layer_config={layer_idx: {"skip": main_skips} for layer_idx in range(2)} if main_skips else None,
        )
        config.num_mtp_layers = 2
        config.mtp_per_layer_config = {
            0: {"intermediate_size": 64, "rms_norm_eps": 1e-5},
            1: {"intermediate_size": 96, "skip": ["attention"]},
        }
        main_model = build_model(config, LlamaForCausalLM)

        mtp_model = MtpModel(main_model, num_mtp_layers=2)

        self.assertTrue(mtp_model.config.is_heterogeneous)
        self.assertTrue(mtp_model.config.generic_modeling_applied)
        self.assertEqual(mtp_model.layers[0].mtp_block.mlp.gate_proj.out_features, 64)
        self.assertEqual(mtp_model.layers[1].mtp_block.mlp.gate_proj.out_features, 96)
        self.assertEqual(mtp_model.layers[0].enorm.variance_epsilon, 1e-5)
        self.assertEqual(list(mtp_model.layers[1].mtp_block.self_attn.parameters()), [])
        for layer in mtp_model.layers:
            for norm in (layer.enorm, layer.hnorm, layer.post_norm):
                self.assertIsInstance(norm, LlamaRMSNorm)

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
        self.assertFalse(main_model.config.generic_modeling_applied)
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

    @parameterized.expand(
        [
            ("bool", True, TypeError, "must be an integer.*True"),
            ("negative", -1, IndexError, "out of range.*-1"),
        ]
    )
    def test_invalid_resolved_layer_idx_fails_clearly(self, _, layer_idx, error_type, message):
        with self.assertRaisesRegex(error_type, message):
            _MaskSelectingToyModel(_toy_config(intermediate_size=64), layer_idx=layer_idx)

    def test_layer_forward_selects_attention_mask_by_layer_idx(self):
        model = _MaskSelectingToyModel(_toy_config(intermediate_size=64), layer_idx=2)
        masks = AttentionMasksByLayerIdx({0: "layer-zero-mask", 2: "layer-two-mask"})

        self.assertEqual(model.layer(torch.zeros(1), attention_mask=masks), "layer-two-mask")
        self.assertEqual(model.layer(torch.zeros(1), None, masks), "layer-two-mask")

    def test_sequential_heterogeneous_models_no_interference(self):
        """Two heterogeneous models built sequentially should each have correct per-layer weights."""
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


@require_torch
class TestHeterogeneousCache(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @parameterized.expand(
        [
            ("llama", {0: {"num_key_value_heads": 2}, 1: {"skip": ["attention"]}, 2: {"num_key_value_heads": 1}}),
            ("gpt_oss", {0: {"sliding_window": 3}, 1: {"skip": ["attention"]}, 2: {"sliding_window": 4}}),
            ("llama4", {0: {"attention_chunk_size": 2}, 1: {"attention_chunk_size": 3}, 2: {"skip": ["attention"]}}),
            ("nemotron_h", {1: {"skip": ["mixer"]}, 3: {"skip": ["mixer"]}}),
        ]
    )
    def test_cached_decoding_matches_uncached(self, name, overrides):
        factory, model_class = {
            "llama": (tiny_llama_config, LlamaForCausalLM),
            "gpt_oss": (tiny_gpt_oss_config, GptOssForCausalLM),
            "llama4": (tiny_llama4_config, Llama4ForCausalLM),
            "nemotron_h": (tiny_nemotron_h_config, NemotronHForCausalLM),
        }[name]
        config = factory(per_layer_config=overrides)
        model = build_model(config, model_class)
        input_ids = torch.tensor([[0, 0, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
        attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
        caches = (DynamicCache(config=config), StaticCache(config=config, max_cache_len=input_ids.shape[1]))

        with torch.no_grad():
            expected_logits = model(input_ids, attention_mask=attention_mask, use_cache=False).logits[:, -2:]
            for cache in caches:
                with self.subTest(cache_type=type(cache).__name__):
                    model(
                        input_ids[:, :-2], attention_mask=attention_mask[:, :-2], past_key_values=cache, use_cache=True
                    )
                    actual_logits = model(
                        input_ids[:, -2:], attention_mask=attention_mask, past_key_values=cache, use_cache=True
                    ).logits
                    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-4, atol=1e-5)

    def test_static_cache_generation_with_skipped_attention(self):
        config = tiny_gpt_oss_config(
            per_layer_config={0: {"sliding_window": 3}, 1: {"skip": ["attention"]}, 2: {"sliding_window": 4}},
            pad_token_id=0,
            eos_token_id=None,
        )
        config._attn_implementation = "eager"
        model = build_model(config, GptOssForCausalLM)
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
