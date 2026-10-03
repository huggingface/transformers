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

import copy
import pickle
import tempfile
import unittest
from unittest.mock import Mock, patch

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
        DynamicCache,
        GptOssForCausalLM,
        Llama4ForCausalLM,
        LlamaConfig,
        LlamaForCausalLM,
        LlamaForSequenceClassification,
        LlamaModel,
        NemotronHForCausalLM,
        PreTrainedConfig,
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
            self.scale = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, hidden_states, position_ids=None, attention_mask=None):
            return hidden_states * self.scale, attention_mask

    class _MaskSelectingToyModel(_ToyPreTrainedModel):
        _heterogeneous_modeling_spec = HeterogeneousModelingSpec(
            layer_cls=_MaskSelectingToyLayer,
            layer_idx_resolver=LayerIdxFromArgument("layer_idx"),
        )

        def __init__(self, config, layer_idx=0):
            super().__init__(config)
            self.layer = _MaskSelectingToyLayer(config, layer_idx=layer_idx)

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

    def test_global_config_changes_propagate_to_layer_configs(self):
        config = tiny_llama_config(per_layer_config={1: {"attention_dropout": 0.1}})
        model = build_model(config, LlamaForCausalLM)
        layer_configs = [layer.self_attn.config for layer in model.model.layers]
        # Changes made directly on one layer's config after the model is built
        layer_configs[3].attention_dropout = 0.3
        layer_configs[3].rope_parameters = {"rope_type": "default", "rope_theta": 1234.0}

        # Change the global config after the model is built
        model.config.is_causal = False
        model.config.attention_dropout = 0.2
        # `rope_scaling` is a property that sets `rope_parameters` under the hood
        model.config.rope_scaling = {"rope_type": "default", "rope_theta": 9999.0}

        self.assertEqual([c.is_causal for c in layer_configs], [False, False, False, False])
        # Layer 1 overrides `attention_dropout` and layer 3 was changed directly, so they keep their own values
        self.assertEqual([c.attention_dropout for c in layer_configs], [0.2, 0.1, 0.2, 0.3])
        # Layer 3 was changed directly, so it keeps its own `rope_parameters`
        self.assertEqual([c.rope_parameters["rope_theta"] for c in layer_configs], [9999.0, 9999.0, 9999.0, 1234.0])

    def test_inherited_init_finalizes_layer_configs(self):
        config = tiny_llama_config(per_layer_config={1: {"intermediate_size": 64}})

        model = build_model(config, LlamaForSequenceClassification)

        self.assertTrue(config.generic_modeling_applied)
        self.assertIs(config._heterogeneity_spec.model_layer_configs[1], model.model.layers[1].self_attn.config)

    def test_failed_outer_init_does_not_publish_layer_configs(self):
        config = tiny_llama_config(per_layer_config={0: {"intermediate_size": 64}})

        def fail_post_init(model):
            self.assertEqual(len(model.model.layers), config.num_hidden_layers)
            self.assertIsNone(config._heterogeneity_spec.model_layer_configs)
            raise RuntimeError("Outer model initialization failed")

        with patch.object(LlamaForCausalLM, "post_init", fail_post_init):
            with self.assertRaisesRegex(RuntimeError, "Outer model initialization failed"):
                build_model(config, LlamaForCausalLM)

        self.assertFalse(config.generic_modeling_applied)

        model = build_model(config, LlamaForCausalLM)
        self.assertTrue(config.generic_modeling_applied)
        for layer_idx, layer in enumerate(model.model.layers):
            self.assertIs(config._heterogeneity_spec.model_layer_configs[layer_idx], layer.self_attn.config)

    def test_used_config_cannot_construct_another_model(self):
        config = tiny_llama_config(per_layer_config={1: {"intermediate_size": 64}})
        build_model(config, LlamaForCausalLM)
        copied_config = copy.deepcopy(config)

        with self.assertRaisesRegex(ValueError, "was already used to construct a model"):
            LlamaForCausalLM(copied_config)

    def test_composite_publishes_sub_config_layer_configs_only_after_success(self):
        text_config = tiny_llama_config(per_layer_config={1: {"intermediate_size": 64}})
        config = _TwoTowerConfig(text_config=text_config)

        def fail_post_init(model):
            raise RuntimeError("Composite model initialization failed")

        with patch.object(_TwoTowerModel, "post_init", fail_post_init):
            with self.assertRaisesRegex(RuntimeError, "Composite model initialization failed"):
                _TwoTowerModel(config)

        self.assertFalse(text_config.generic_modeling_applied)

        model = _TwoTowerModel(config)
        for layer_idx, (first_layer, second_layer) in enumerate(zip(model.first.layers, model.second.layers)):
            self.assertIs(first_layer.self_attn.config, second_layer.self_attn.config)
            self.assertIs(text_config.per_layer_config[layer_idx], first_layer.self_attn.config)

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

    def test_error_when_spec_layer_cls_is_not_constructed(self):
        """If the model never builds the spec's layer class, init should fail."""
        config = tiny_llama_config(per_layer_config={0: {"skip": ["attention"]}})
        # The model builds a `_ToyDecoderLayer`, but the spec is looking for `_MaskSelectingToyLayer`, so it never
        # catches anything. Without the check, layer 0 would just quietly keep its attention.
        spec = _toy_modeling_spec(_MaskSelectingToyLayer, _ToyNoOpAttention)
        with patch.object(_SingleLayerToyModel, "_heterogeneous_modeling_spec", spec):
            with self.assertRaisesRegex(ValueError, "no `_MaskSelectingToyLayer` layer was constructed"):
                _SingleLayerToyModel(config)

    @parameterized.expand([("class_specific_match", True), ("generic_match", False)])
    def test_skip_replacement_selection(self, _, class_matches):
        spec = _toy_modeling_spec(_ToyDecoderLayer, _ToyNoOpAttention)
        member_class = _ToyAttention if class_matches else torch.nn.Linear
        spec.skip_descriptors["attention"] = {
            "self_attn": _ToyNoOpAttention,
            ("self_attn", member_class): _ClassSpecificNoOpAttention,
        }
        with patch.object(_SingleLayerToyModel, "_heterogeneous_modeling_spec", spec):
            model = _SingleLayerToyModel(_toy_config(intermediate_size=32, skip_attention=True))

        expected_class = _ClassSpecificNoOpAttention if class_matches else _ToyNoOpAttention
        self.assertIs(type(model.layer.self_attn), expected_class)

    def test_unmatched_skip_member_raises_before_replacing_any_members(self):
        config = tiny_llama_config(per_layer_config={3: {"skip": ["attention"]}})
        fixture = MODEL_FIXTURES["llama"]
        spec = fixture.spec_factory()
        norm_replacement = Mock(wraps=torch.nn.Identity)
        spec.skip_descriptors["attention"] = {
            "input_layernorm": norm_replacement,
            ("self_attn", torch.nn.Linear): torch.nn.Identity,
            ("self_attn", torch.nn.Conv1d): torch.nn.Identity,
        }

        with patch.object(fixture.pretrained_cls, "_heterogeneous_modeling_spec", spec, create=True):
            with self.assertRaisesRegex(
                ValueError,
                "Layer 3.*'attention'.*no replacement.*'self_attn'.*LlamaAttention",
            ):
                build_model(config, LlamaForCausalLM)

        norm_replacement.assert_not_called()

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

        self.assertEqual(model.layer(torch.zeros(1), attention_mask=masks)[1], "layer-two-mask")
        self.assertEqual(model.layer(torch.zeros(1), None, masks)[1], "layer-two-mask")

    @parameterized.expand([("deepcopy",), ("data_parallel",), ("pickle",)])
    def test_copied_layer_uses_its_own_parameters(self, copy_method):
        model = _MaskSelectingToyModel(_toy_config(intermediate_size=64), layer_idx=2)
        if copy_method == "deepcopy":
            copied_layer = copy.deepcopy(model.layer)
        elif copy_method == "data_parallel":
            # Exercise DataParallel's shallow replication without requiring GPUs.
            copied_layer = model.layer._replicate_for_data_parallel()
        else:
            copied_layer = pickle.loads(pickle.dumps(model.layer))
        copied_layer.scale = torch.nn.Parameter(torch.tensor(7.0))
        masks = AttentionMasksByLayerIdx({0: "layer-zero-mask", 2: "layer-two-mask"})

        output, selected_mask = copied_layer(torch.tensor(3.0), attention_mask=masks)

        torch.testing.assert_close(output, torch.tensor(21.0))
        self.assertEqual(selected_mask, "layer-two-mask")

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

    def test_attention_outputs_with_custom_skipped_attention(self):
        class CustomSkippedAttention(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                return torch.zeros_like(hidden_states), None

        config = tiny_llama_config(num_hidden_layers=3, per_layer_config={1: {"skip": ["attention"]}})
        config._attn_implementation = "eager"
        fixture = MODEL_FIXTURES["llama"]
        spec = fixture.spec_factory()
        spec.skip_descriptors["attention"]["self_attn"] = CustomSkippedAttention
        with patch.object(fixture.pretrained_cls, "_heterogeneous_modeling_spec", spec, create=True):
            model = build_model(config, LlamaForCausalLM)
        input_ids = dummy_input_ids()

        with torch.no_grad():
            actual = model(input_ids, use_cache=False, output_attentions=True)

        self.assertEqual(len(actual.attentions), 3)
        self.assertIsNone(actual.attentions[1])
        batch_size, seq_length = input_ids.shape
        expected_shape = (batch_size, config.num_attention_heads, seq_length, seq_length)
        for layer_idx in (0, 2):
            self.assertEqual(actual.attentions[layer_idx].shape, expected_shape)

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

    @parameterized.expand(
        [
            ("no_skips", {1: {"intermediate_size": 96}}),
            ("skip_attention", {1: {"skip": ["attention"]}}),
        ]
    )
    @pytest.mark.torch_compile_test
    def test_static_cache_decoding_compiles_fullgraph(self, _name, per_layer_config):
        config = tiny_llama_config(num_hidden_layers=3, per_layer_config=per_layer_config)
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
        config = tiny_llama_config(
            num_hidden_layers=3, per_layer_config={1: {"skip": ["attention"]}}, pad_token_id=0, eos_token_id=None
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
            self.assertIsNone(attentions[1])
            query_length = input_ids.shape[1] if step == 0 else 1
            expected_shape = (1, config.num_attention_heads, query_length, input_ids.shape[1] + step)
            for layer_idx in (0, 2):
                self.assertEqual(attentions[layer_idx].shape, expected_shape)
