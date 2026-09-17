# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import itertools
import sys
import unittest
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from unittest import mock

from transformers.testing_utils import require_executorch, require_torch, require_torch_greater_or_equal
from transformers.utils.import_utils import is_torch_available


if is_torch_available():
    import torch


def _text_config():
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
    )


def _model():
    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _text_config()
            self.generation_config = SimpleNamespace(
                use_cache=False, cache_implementation=None, cache_config={"max_cache_len": 99}
            )
            self.embed = torch.nn.Embedding(17, 8)
            self.lm_head = torch.nn.Linear(8, 11, bias=False)
            self.last_kwargs = None

        @property
        def dtype(self):
            return self.embed.weight.dtype

        def get_output_embeddings(self):
            return self.lm_head

        def set_attn_implementation(self, name):
            self.config._attn_implementation = name
            self.config._attn_was_changed = True

        def forward(self, input_ids, **kwargs):
            self.last_kwargs = kwargs
            indices = kwargs.get("logits_to_keep", 0)
            indices = slice(-indices, None) if isinstance(indices, int) else indices
            return SimpleNamespace(logits=self.lm_head(self.embed(input_ids)[:, indices]))

    return TinyLM()


def _dependencies():
    # Synthetic writer results test HF's merging, not ET's metadata encoding.
    metadata = SimpleNamespace(model_vocab_size=mock.Mock(return_value=11))
    for name in ("max_context_len", "max_seq_len", "vocab_size", "activation_dtype", "logits_to_keep_mode"):
        setattr(metadata, "write_" + name, mock.Mock(return_value={name: object()}))
    metadata.write_cache_geometry = mock.Mock(return_value={"geometry": "written"})
    return SimpleNamespace(
        metadata=metadata,
        replace_hf_cache_with_mlx_in_graph_cache=mock.Mock(),
    )


@require_torch
class MLXPreparationTest(unittest.TestCase):
    def setUp(self):
        # No exporter/MLX imports at discovery time, including when torch/ET is absent.
        from transformers.exporters import exporter_executorch
        from transformers.exporters.configs import ExecutorchConfig
        from transformers.integrations import executorch

        self.shared = executorch
        self.mlx = exporter_executorch
        self.config = ExecutorchConfig(backend="mlx", dtype="fp32", max_context_len=8, max_seq_len=None)

    def test_cache_and_output_combinations(self):
        for cache_mode, mode, window in itertools.product(
            ("in-graph", "off-graph"), ("full", "last", "selected"), (None, 4)
        ):
            with self.subTest(cache_mode=cache_mode, mode=mode, window=window):
                model, deps = _model(), _dependencies()
                self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
                model.config.sliding_window = window
                cache_config = model.generation_config.cache_config
                config = replace(self.config, cache_mode=cache_mode, logits_to_keep=mode)
                snapshot = copy.deepcopy(config)
                with mock.patch.object(self.mlx, "_load_mlx_dependencies", return_value=deps):
                    prepared = self.mlx._prepare_mlx(model, {}, config)
                self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
                self.assertEqual(config, snapshot)
                self.assertIs(model.generation_config.cache_config, cache_config)
                self.assertEqual(cache_config, {"max_cache_len": 99})
                self.assertFalse(model.training)
                self.assertIsNone(model.last_kwargs)
                self.assertIs(prepared.attention_target, model)
                deps.metadata.model_vocab_size.assert_called_once_with(model)
                expected_metadata = {}
                for name, value in (
                    ("max_context_len", 8),
                    ("max_seq_len", window or 8),
                    ("vocab_size", 11),
                    ("activation_dtype", "fp32"),
                    ("logits_to_keep_mode", mode),
                ):
                    writer = getattr(deps.metadata, "write_" + name)
                    writer.assert_called_once_with(value)
                    expected_metadata.update(writer.return_value)
                if cache_mode == "off-graph":
                    deps.metadata.write_cache_geometry.assert_called_once_with([1, 1], [4, 4], [window or 0] * 2)
                    expected_metadata.update(deps.metadata.write_cache_geometry.return_value)
                    deps.replace_hf_cache_with_mlx_in_graph_cache.assert_not_called()
                else:
                    deps.metadata.write_cache_geometry.assert_not_called()
                self.assertEqual(prepared.constant_methods, expected_metadata)
                expected_type = (
                    self.shared._OffGraphWrapper if cache_mode == "off-graph" else self.shared._InGraphCacheAndOutput
                )
                self.assertIs(type(prepared.model), expected_type)
                self.assertEqual(prepared.cache_layout, self.shared._resolve_cache_layout(model.config))
                self.assertEqual(prepared.model.logits_to_keep_mode, mode)
                if cache_mode == "off-graph":
                    self.assertFalse(model.generation_config.use_cache)
                else:
                    self.assertIsNone(prepared.model.cache)
                    self.assertIs(prepared.model._cache_buffer_bindings, self.mlx._mlx_cache_buffer_bindings)
                    self.assertTrue(model.generation_config.use_cache)
                    self.assertTrue(model.config.use_cache)
                    self.assertEqual(model.generation_config.cache_implementation, "static")
                    deps.replace_hf_cache_with_mlx_in_graph_cache.assert_called_once_with(
                        prepared.model,
                        prepared.cache_layout.layer_configs,
                        max_batch_size=1,
                        max_cache_len=8,
                        max_write_len=window or 8,
                        dtype=torch.float32,
                    )

    def test_recipe_selects_capture_context(self):
        for cache_mode, strict in itertools.product(("in-graph", "off-graph"), (False, True)):
            with self.subTest(cache_mode=cache_mode, strict=strict):
                private = self.mlx._MLXRecipeState(
                    mock.sentinel.wrapper, {}, {}, {}, None, cache_mode, mock.sentinel.layout
                )
                config = replace(self.config, cache_mode=cache_mode, strict=strict)
                with (
                    mock.patch.object(self.mlx, "_prepare_mlx", return_value=private),
                    mock.patch.object(
                        self.mlx, "_in_graph_cache_capture_scope", return_value=mock.sentinel.context
                    ) as capture_scope,
                ):
                    prepared = self.mlx.prepare_for_mlx(mock.sentinel.model, {}, config)
                self.assertIsInstance(prepared, self.mlx._BackendPreparation)
                if cache_mode == "in-graph":
                    self.assertIs(prepared.capture_context, mock.sentinel.context)
                    capture_scope.assert_called_once_with(mock.sentinel.wrapper, strict=strict)
                else:
                    capture_scope.assert_not_called()
                    self.assertIsInstance(prepared.capture_context, nullcontext)

    def test_mlx_cache_buffer_bindings(self):
        wrapper = torch.nn.Module()
        cache = SimpleNamespace(layers=[], kv_cache=[])
        expected = []
        for index in range(2):
            layer = SimpleNamespace(cumulative_length=torch.tensor(0))
            kv = torch.nn.Module()
            kv.register_buffer("k_cache", torch.zeros(1))
            kv.register_buffer("v_cache", torch.zeros(1))
            cache.layers.append(layer)
            cache.kv_cache.append(kv)
            for mapping, slot, prefix in (
                (vars(layer), "cumulative_length", "cumulative_length"),
                (kv._buffers, "k_cache", "key_cache"),
                (kv._buffers, "v_cache", "value_cache"),
            ):
                name = f"{prefix}_{index}"
                wrapper.register_buffer(name, mapping[slot])
                expected.append((mapping, slot, name, mapping[slot]))
        for _, _, name, _ in expected:
            setattr(wrapper, name, torch.ones(1))

        bindings = self.mlx._mlx_cache_buffer_bindings(wrapper, cache)
        self.assertEqual(len(bindings), len(expected))
        for (mapping, slot, tensor), (destination, key, name, original) in zip(bindings, expected):
            self.assertIs(mapping, destination)
            self.assertEqual(slot, key)
            self.assertIs(tensor, getattr(wrapper, name))
            self.assertIs(destination[key], original)

        cache.kv_cache.pop()
        with self.assertRaisesRegex(ValueError, "layer counts do not match"):
            self.mlx._mlx_cache_buffer_bindings(wrapper, cache)

    def test_rejections_precede_model_mutation(self):
        cases = [(name, value) for name in ("max_context_len", "max_seq_len") for value in (0, -1, True, 1.5)]
        cases += [
            ("max_seq_len", 9),
            ("cache_mode", "invalid"),
            ("logits_to_keep", "invalid"),
            ("dtype", "fp64"),
            ("dynamic_shapes", {"input_ids": None}),
            ("dynamic_shapes", []),
        ]
        for name, value in cases:
            with self.subTest(name=name, value=value):
                model = _model()
                with mock.patch.object(self.mlx, "_load_mlx_dependencies") as load, self.assertRaises(ValueError):
                    self.mlx._prepare_mlx(model, {}, replace(self.config, **{name: value}))
                load.assert_not_called()
                self.assertTrue(model.training)
                self.assertFalse(model.generation_config.use_cache)
        mutations = (
            lambda m: m.to(dtype=torch.float16),
            lambda m: m.to(device="meta"),
            lambda m: m.lm_head.half(),
            lambda m: m.register_buffer("bad", torch.zeros(1, dtype=torch.float64)),
            lambda m: setattr(m.config, "sliding_window", 0),
            lambda m: setattr(m.config, "is_encoder_decoder", True),
            lambda m: setattr(m, "generation_config", None),
            lambda m: setattr(m, "set_attn_implementation", None),
        )
        for mutate in mutations:
            model = _model()
            mutate(model)
            with mock.patch.object(self.mlx, "_load_mlx_dependencies") as load, self.assertRaises(ValueError):
                self.mlx._prepare_mlx(model, {}, self.config)
            load.assert_not_called()
            self.assertTrue(model.training)
        for inputs in (None, [], {"input_ids": torch.tensor([[1]])}):
            with self.assertRaisesRegex(ValueError, "empty sample_inputs"):
                self.mlx._prepare_mlx(_model(), inputs, self.config)
        model = _model()
        model.config.sliding_window = 4
        with self.assertRaisesRegex(ValueError, "sliding-window limit"):
            self.mlx._prepare_mlx(model, {}, replace(self.config, max_seq_len=5))
        self.assertTrue(model.training)
        with mock.patch.object(self.mlx, "_load_mlx_dependencies", side_effect=ImportError("missing ET")):
            with self.assertRaisesRegex(ImportError, "missing ET"):
                self.mlx._prepare_mlx(model, {}, self.config)
        self.assertTrue(model.training)
        self.assertFalse(model.generation_config.use_cache)

    def test_off_graph_max_write_validation_precedes_model_mutation(self):
        int32_max = torch.iinfo(torch.int32).max
        for window, max_seq_len in ((4, 5), (None, int32_max + 1)):
            with self.subTest(window=window, max_seq_len=max_seq_len):
                model = _model()
                model.config.sliding_window = window
                config = replace(
                    self.config,
                    cache_mode="off-graph",
                    max_context_len=max_seq_len,
                    max_seq_len=max_seq_len,
                )
                with mock.patch.object(self.mlx, "_load_mlx_dependencies") as load:
                    with self.assertRaisesRegex(ValueError, "sliding-window limit|int32"):
                        self.mlx._prepare_mlx(model, {}, config)
                load.assert_not_called()
                self.assertTrue(model.training)
                self.assertFalse(model.generation_config.use_cache)

    def test_attention_callback_selection_and_layout_delegation(self):
        layout = self.shared._CacheLayout((1, 1), (4, 4), (0, 4, 0, 4), (0, 1, 0, 1))
        for cache_mode in ("in-graph", "off-graph"):
            with self.subTest(cache_mode=cache_mode):
                prepared = self.mlx._MLXRecipeState(
                    mock.sentinel.wrapper, {}, {}, {}, mock.sentinel.target, cache_mode, layout
                )
                with mock.patch.object(self.mlx, "_attention_scope", return_value=mock.sentinel.context) as scope:
                    context = self.mlx._mlx_attention_scope(prepared)
                self.assertIs(context, mock.sentinel.context)
                scope.assert_called_once()
                target, name, callback, mask = scope.call_args.args
                self.assertIs(target, mock.sentinel.target)
                self.assertEqual(name, "mlx" if cache_mode == "in-graph" else "executorch_off_graph")
                self.assertIs(mask, self.shared._attention_mask)
                self.assertEqual(scope.call_args.kwargs, {})
                self.assertIsInstance(callback, partial)
                self.assertIs(
                    callback.func,
                    self.mlx._mlx_in_graph_attention_forward
                    if cache_mode == "in-graph"
                    else self.shared._off_graph_attention_forward,
                )
                self.assertEqual(callback.args, ())
                expected_kwargs = (
                    {"_cache_windows": layout.windows}
                    if cache_mode == "in-graph"
                    else {"_cache_ids": layout.cache_ids}
                )
                self.assertEqual(callback.keywords, expected_kwargs)
                for key, value in expected_kwargs.items():
                    self.assertIs(callback.keywords[key], value)

    def test_adapter_preparation_does_not_require_lowering_apis(self):
        from transformers.exporters import exporter_executorch as et

        for cache_mode, strict in itertools.product(("in-graph", "off-graph"), (False, True)):
            with self.subTest(cache_mode=cache_mode, strict=strict):
                model = _model()
                config = replace(self.config, cache_mode=cache_mode, strict=strict)
                snapshot = copy.deepcopy(config)
                with mock.patch.object(self.mlx, "_load_mlx_dependencies", return_value=_dependencies()) as load:
                    prepared = et.prepare_for_mlx(model, {}, config)
                load.assert_called_once_with(cache_mode)
                self.assertIsInstance(prepared, et._BackendPreparation)
                self.assertIs(prepared.model.model, model)
                self.assertTrue(callable(prepared.make_lowering_settings))
                self.assertIsNot(prepared.capture_config, config)
                self.assertEqual(prepared.capture_config.strict, strict)
                self.assertEqual(set(prepared.sample_inputs), set(prepared.capture_config.dynamic_shapes))
                self.assertEqual(config, snapshot)

    def test_lowering_passes_metadata_and_allocation_flags(self):
        from transformers.exporters import exporter_executorch as et

        ep = object()
        for cache_mode, flags in itertools.product(
            ("in-graph", "off-graph"), ((True, True, True), (True, False, True), (False, True, False))
        ):
            with self.subTest(cache_mode=cache_mode, flags=flags):
                deps = SimpleNamespace(
                    **{
                        name: mock.Mock()
                        for name in (
                            "get_default_passes",
                            "MLXPartitioner",
                            "EdgeCompileConfig",
                            "MemoryPlanningPass",
                            "ExecutorchBackendConfig",
                        )
                    }
                )
                private = self.mlx._MLXRecipeState(
                    None,
                    {},
                    {},
                    {"get_vocab_size": 11},
                    None,
                    cache_mode,
                    self.shared._resolve_cache_layout(_text_config()),
                )
                config = replace(
                    self.config,
                    cache_mode=cache_mode,
                    alloc_graph_input=flags[0],
                    alloc_graph_output=flags[1],
                    alloc_mutable_buffers=flags[2],
                )
                model, inputs, edge = object(), {}, mock.Mock()

                def lower(*args, **kwargs):
                    deps.MemoryPlanningPass.assert_not_called()
                    deps.ExecutorchBackendConfig.assert_not_called()
                    return edge

                with (
                    mock.patch.object(self.mlx, "_prepare_mlx", return_value=private) as prepare,
                    mock.patch.object(self.mlx, "_load_mlx_dependencies", return_value=deps) as load,
                    mock.patch.object(et, "to_edge_transform_and_lower", side_effect=lower, create=True) as lower_edge,
                ):
                    prepared = et.prepare_for_mlx(model, inputs, config)
                    prepare.assert_called_once_with(model, inputs, config)
                    load.assert_not_called()
                    for constructor in vars(deps).values():
                        constructor.assert_not_called()
                    settings = prepared.make_lowering_settings()
                    load.assert_called_once_with(cache_mode)
                    self.assertIsInstance(settings, et._LoweringSettings)
                    self.assertEqual(settings.method_name, "forward")
                    self.assertIs(settings.constant_methods, private.constant_methods)
                    deps.MemoryPlanningPass.assert_not_called()
                    deps.ExecutorchBackendConfig.assert_not_called()
                    result = et._lower_to_executorch(ep, settings)
                lower_edge.assert_called_once_with(
                    {"forward": ep},
                    transform_passes=deps.get_default_passes.return_value,
                    partitioner=[deps.MLXPartitioner.return_value],
                    compile_config=deps.EdgeCompileConfig.return_value,
                    constant_methods=private.constant_methods,
                )
                deps.EdgeCompileConfig.assert_called_once_with(_check_ir_validity=False, _skip_dim_order=True)
                deps.get_default_passes.assert_called_once_with()
                deps.MLXPartitioner.assert_called_once_with()
                deps.MemoryPlanningPass.assert_called_once_with(
                    alloc_graph_input=flags[0], alloc_graph_output=flags[1], alloc_mutable_buffers=flags[2]
                )
                deps.ExecutorchBackendConfig.assert_called_once_with(
                    extract_delegate_segments=True, memory_planning_pass=deps.MemoryPlanningPass.return_value
                )
                edge.to_executorch.assert_called_once_with(config=deps.ExecutorchBackendConfig.return_value)
                self.assertIs(result, edge.to_executorch.return_value)


def _initialize_experts(experts, dtype):
    # Standalone HF experts allocate torch.empty parameters; no model post_init runs.
    generator = torch.Generator().manual_seed(43)
    with torch.no_grad():
        for parameter in experts.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.2)
    return experts.to(dtype=dtype).eval().requires_grad_(False)


def _qwen_experts(dtype, num_experts=4):
    from transformers import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

    config = Qwen3_5MoeTextConfig(
        hidden_size=16, moe_intermediate_size=24, num_experts=num_experts, num_experts_per_tok=2, num_hidden_layers=1
    )
    config._experts_implementation = "eager"
    return _initialize_experts(Qwen3_5MoeExperts(config), dtype)


def _expert_inputs(dtype, rows=4, num_experts=4):
    generator = torch.Generator().manual_seed(51)
    hidden = (torch.randn(rows, 16, generator=generator) * 0.5).to(dtype)
    indices = torch.randint(num_experts, (rows, 2), generator=generator)
    weights = torch.rand(rows, 2, generator=generator, dtype=torch.float32)
    return hidden, indices, weights / weights.sum(dim=-1, keepdim=True)


def _expert_target(experts):
    from transformers import PreTrainedModel

    class ExpertTarget(PreTrainedModel):
        # Exercise HF's real public setter, without constructing a full decoder.
        _can_set_experts_implementation_cached_value = True

        def __init__(self):
            super().__init__(experts.config)
            self.experts = experts

    return ExpertTarget().eval()


def _variant_experts(*, has_gate, has_bias, is_transposed, custom_gate=False):
    from transformers.integrations.moe import use_experts_implementation

    @use_experts_implementation(has_gate=has_gate, has_bias=has_bias, is_transposed=is_transposed)
    class VariantExperts(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.num_experts, self.hidden_dim, self.intermediate_dim = 4, 16, 24
            up_shape = (4, 24 * (2 if has_gate else 1), 16)
            down_shape = (4, 16, 24)
            if is_transposed:
                up_shape = (up_shape[0], up_shape[2], up_shape[1])
                down_shape = (down_shape[0], down_shape[2], down_shape[1])
            up_name = "gate_up_proj" if has_gate else "up_proj"
            setattr(self, up_name, torch.nn.Parameter(torch.empty(up_shape)))
            self.down_proj = torch.nn.Parameter(torch.empty(down_shape))
            if has_bias:
                setattr(self, up_name + "_bias", torch.nn.Parameter(torch.empty(4, 24 * (2 if has_gate else 1))))
                self.down_proj_bias = torch.nn.Parameter(torch.empty(4, 16))
            self.act_fn = torch.nn.functional.silu

        def forward(self, hidden_states, top_k_index, top_k_weights):
            raise AssertionError("Tests must invoke the HF MLX callback directly")

    if custom_gate:

        def apply_gate(self, projected):
            # HF's contract is 2D. Deliberately differ from the default SiLU gate.
            if projected.ndim != 2:
                raise AssertionError("HF _apply_gate expects a 2D projection")
            gate, up = projected.chunk(2, dim=-1)
            return torch.sigmoid(gate + 0.3) * (up - 0.2)

        VariantExperts._apply_gate = apply_gate
    return _initialize_experts(VariantExperts(SimpleNamespace(_experts_implementation="eager")), torch.float32)


@require_torch
class MLXExpertsScopeTest(unittest.TestCase):
    def setUp(self):
        from transformers.exporters import exporter_executorch
        from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

        self.mlx = exporter_executorch
        self.registry = ALL_EXPERTS_FUNCTIONS

    def test_public_setter_and_registry_restore_on_success_error_and_nesting(self):
        name = "executorch_mlx"
        for existing, shadow, failure in itertools.product((False, True), (False, True), (None, "select", "body")):
            with self.subTest(existing=existing, shadow=shadow, failure=failure), ExitStack() as stack:
                target = _expert_target(_qwen_experts(torch.float32))
                nested = SimpleNamespace(_experts_implementation_internal="batched_mm")
                target.config.sub_configs = {"nested": object}
                target.config.nested = nested
                target.extra = torch.nn.Module()
                target.extra.config = SimpleNamespace(_experts_implementation="eager")
                configs = (target.config, nested, target.extra.config)
                snapshots = [dict(vars(config)) for config in configs]
                mappings = (self.registry._global_mapping, self.registry._local_mapping)
                for mapping, present in zip(mappings, (existing, shadow)):
                    stack.enter_context(mock.patch.dict(mapping))
                    mapping.pop(name, None)
                    if present:
                        mapping[name] = mock.Mock()
                before = [dict(mapping) for mapping in mappings]
                original_select = target.set_experts_implementation

                def select(selected):
                    original_select(selected)
                    # Model a public setter propagating to submodel configs.
                    nested._experts_implementation_internal = selected
                    target.extra.config._experts_implementation = selected
                    if failure == "select":
                        raise RuntimeError("select")

                setter = stack.enter_context(
                    mock.patch.object(target, "set_experts_implementation", side_effect=select)
                )
                with self.assertRaisesRegex(RuntimeError, failure) if failure else nullcontext():
                    with self.mlx._mlx_experts_scope(target):
                        setter.assert_called_once_with(name)
                        self.assertEqual(target.config._experts_implementation, name)
                        self.assertIs(self.registry[name], self.mlx._mlx_experts_forward)
                        outer_configs = [dict(vars(config)) for config in configs]
                        outer_mappings = [dict(mapping) for mapping in mappings]
                        with self.assertRaisesRegex(RuntimeError, "inner"):
                            with self.mlx._mlx_experts_scope(target):
                                self.assertIs(self.registry[name], self.mlx._mlx_experts_forward)
                                raise RuntimeError("inner")
                        self.assertEqual([dict(vars(config)) for config in configs], outer_configs)
                        self.assertEqual([dict(mapping) for mapping in mappings], outer_mappings)
                        if failure == "body":
                            raise RuntimeError("body")
                self.assertEqual([dict(vars(config)) for config in configs], snapshots)
                self.assertEqual([dict(mapping) for mapping in mappings], before)

    def test_dense_and_missing_capability_are_noops(self):
        from transformers.integrations.moe import ExpertsInterface

        for target in (None, _model(), torch.nn.Linear(3, 2), _expert_target(_qwen_experts(torch.float32))):
            with self.subTest(target=type(target).__name__), ExitStack() as stack:
                if hasattr(target, "_can_set_experts_implementation"):
                    stack.enter_context(
                        mock.patch.object(target, "_can_set_experts_implementation", return_value=False)
                    )
                    setter = stack.enter_context(mock.patch.object(target, "set_experts_implementation"))
                else:
                    setter = None
                before = dict(vars(target.config)) if hasattr(target, "config") else None
                register = stack.enter_context(mock.patch.object(ExpertsInterface, "register"))
                with self.mlx._mlx_experts_scope(target):
                    register.assert_not_called()
                if setter is not None:
                    setter.assert_not_called()
                if before is not None:
                    self.assertEqual(dict(vars(target.config)), before)

    def test_recipe_composes_attention_and_experts_and_restores_both(self):
        from transformers.exporters.configs import ExecutorchConfig
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        for cache_mode, fail in itertools.product(("in-graph", "off-graph"), (False, True)):
            with self.subTest(cache_mode=cache_mode, fail=fail):
                target = _expert_target(_qwen_experts(torch.float32))
                # No attention layers are needed to exercise the shared scope's selection.
                target.set_attn_implementation = lambda name: setattr(target.config, "_attn_implementation", name)
                snapshot = dict(vars(target.config))
                registries = (self.registry, ALL_ATTENTION_FUNCTIONS)
                mappings = [
                    mapping
                    for registry in registries
                    for mapping in (registry._global_mapping, registry._local_mapping)
                ]
                before = [dict(mapping) for mapping in mappings]
                private = self.mlx._MLXRecipeState(
                    target, {}, {}, {}, target, cache_mode, self.mlx._resolve_cache_layout(_text_config())
                )
                with mock.patch.object(self.mlx, "_prepare_mlx", return_value=private):
                    prepared = self.mlx.prepare_for_mlx(
                        target, {}, ExecutorchConfig(backend="mlx", cache_mode=cache_mode)
                    )
                self.assertEqual(dict(vars(target.config)), snapshot)
                with self.assertRaisesRegex(RuntimeError, "export body") if fail else nullcontext():
                    with prepared.export_context:
                        self.assertEqual(target.config._experts_implementation, "executorch_mlx")
                        self.assertIs(self.registry["executorch_mlx"], self.mlx._mlx_experts_forward)
                        self.assertEqual(
                            target.config._attn_implementation,
                            "mlx" if cache_mode == "in-graph" else "executorch_off_graph",
                        )
                        if fail:
                            raise RuntimeError("export body")
                self.assertEqual(dict(vars(target.config)), snapshot)
                self.assertEqual([dict(mapping) for mapping in mappings], before)


@require_torch
@require_executorch
@require_torch_greater_or_equal("2.11")
class MLXExpertsTest(unittest.TestCase):
    def setUp(self):
        from transformers.exporters import exporter_executorch

        self.mlx = exporter_executorch
        self.routing, self.mm, self.scatter = mock.Mock(), mock.Mock(), mock.Mock()
        ops = SimpleNamespace(
            moe_gather_inputs=SimpleNamespace(default=self.routing),
            gather_mm=SimpleNamespace(default=self.mm),
            moe_scatter_outputs=SimpleNamespace(default=self.scatter),
        )
        self.enterContext(mock.patch.object(torch.ops, "mlx", ops))

    def _assert_expert_boundary(self, experts, inputs):
        hidden, routes, weights = inputs
        rows, width = hidden.shape
        top_k = routes.shape[1]
        up_name = "gate_up_proj" if experts.has_gate else "up_proj"
        up_weight, down_weight = getattr(experts, up_name), experts.down_proj
        selected = hidden.new_full((rows * top_k, 1, width), 0.7)
        indices = torch.tensor([1, 0], dtype=torch.int32).repeat(rows)
        sorted_indices, inverse = mock.sentinel.sorted_indices, mock.sentinel.inverse
        up = torch.linspace(-0.8, 0.9, rows * top_k * 24 * (2 if experts.has_gate else 1), dtype=hidden.dtype)
        up = up.reshape(rows * top_k, 1, -1)
        down = hidden.new_full((rows * top_k, 1, width), 0.3)
        scattered = torch.linspace(-0.4, 0.6, rows * top_k * width, dtype=hidden.dtype).reshape(rows, top_k, width)
        for op in (self.routing, self.mm, self.scatter):
            op.reset_mock()
        self.routing.return_value = selected, indices, sorted_indices, inverse
        self.mm.side_effect = [up, down]
        self.scatter.return_value = scattered
        projected, expected_down = up.squeeze(1), down
        if experts.has_bias:
            projected = projected + getattr(experts, up_name + "_bias")[indices.long()]
            expected_down = down + experts.down_proj_bias[indices.long()].unsqueeze(1)
        gate_name = "_apply_gate" if experts.has_gate else "act_fn"
        apply_gate = getattr(experts, gate_name)
        expected_gate = apply_gate(projected)
        with mock.patch.object(experts, gate_name, wraps=apply_gate) as gate:
            actual = self.mlx._mlx_experts_forward(experts, *inputs)
        gate.assert_called_once()
        self.assertEqual(gate.call_args.args[0].ndim, 2)
        torch.testing.assert_close(gate.call_args.args[0], projected)
        self.routing.assert_called_once()
        routed_hidden, routed_ids, *routing_args = self.routing.call_args.args
        self.assertIs(routed_hidden, hidden)
        self.assertEqual(routed_ids.dtype, torch.int64)
        torch.testing.assert_close(routed_ids, routes.to(torch.int64))
        self.assertEqual(routing_args, [top_k, 1])
        self.assertEqual(self.routing.call_args.kwargs, {})
        self.assertEqual(self.mm.call_count, 2)
        self.assertIs(self.mm.call_args_list[0].args[0], selected)
        torch.testing.assert_close(self.mm.call_args_list[1].args[0], expected_gate.unsqueeze(1))
        for call, parameter in zip(self.mm.call_args_list, (up_weight, down_weight)):
            _, weight, rhs_ids, lhs_ids, order = call.args
            expected_weight = parameter if experts.is_transposed else parameter.transpose(-2, -1)
            torch.testing.assert_close(weight, expected_weight, rtol=0, atol=0)
            self.assertEqual(weight.stride(), expected_weight.stride())
            self.assertEqual(weight.data_ptr(), parameter.data_ptr())
            if experts.is_transposed:
                self.assertIs(weight, parameter)
            self.assertIs(rhs_ids, indices)
            self.assertIsNone(lhs_ids)
            self.assertIs(order, sorted_indices)
            self.assertEqual(call.kwargs, {})
        self.scatter.assert_called_once()
        scatter_input, order, inverse_order, scatter_k = self.scatter.call_args.args
        torch.testing.assert_close(scatter_input, expected_down)
        self.assertIs(order, sorted_indices)
        self.assertIs(inverse_order, inverse)
        self.assertEqual(scatter_k, top_k)
        self.assertEqual(self.scatter.call_args.kwargs, {})
        expected = (scattered * weights.unsqueeze(-1)).sum(dim=1).to(hidden.dtype)
        self.assertEqual(actual.dtype, hidden.dtype)
        torch.testing.assert_close(actual, expected)

    def test_qwen_callback_preserves_parameters_inputs_and_dtype(self):
        for dtype, count, rows, route_dtype in itertools.product(
            (torch.float32, torch.float16, torch.bfloat16), (4, 8), (1, 4, 8), (torch.int32, torch.int64)
        ):
            with self.subTest(dtype=dtype, experts=count, rows=rows, route_dtype=route_dtype):
                experts = _qwen_experts(dtype, count)
                hidden, indices, weights = _expert_inputs(dtype, rows, count)
                inputs = hidden, indices.to(route_dtype), weights
                parameters = dict(experts.named_parameters())
                values = {name: parameter.clone() for name, parameter in parameters.items()}
                before = tuple(value.clone() for value in inputs)
                self._assert_expert_boundary(experts, inputs)
                for value, snapshot in zip(inputs, before):
                    torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
                for name, parameter in experts.named_parameters():
                    self.assertIs(parameter, parameters[name])
                    torch.testing.assert_close(parameter, values[name], rtol=0, atol=0)

    def test_hf_gating_bias_and_orientation_variants(self):
        for has_gate, has_bias, transposed, rows in itertools.product(
            (False, True), (False, True), (False, True), (1, 4)
        ):
            with self.subTest(gate=has_gate, bias=has_bias, transposed=transposed, rows=rows):
                experts = _variant_experts(has_gate=has_gate, has_bias=has_bias, is_transposed=transposed)
                self._assert_expert_boundary(experts, _expert_inputs(torch.float32, rows))

    def test_custom_apply_gate_receives_2d_projection(self):
        for rows in (1, 4):
            with self.subTest(rows=rows):
                experts = _variant_experts(has_gate=True, has_bias=True, is_transposed=False, custom_gate=True)
                self._assert_expert_boundary(experts, _expert_inputs(torch.float32, rows))

    def test_expert_parallel_sentinels_and_unsupported_weights_rejected(self):
        experts = _qwen_experts(torch.float32)
        hidden, indices, weights = _expert_inputs(torch.float32)
        experts._is_expert_parallel = True
        indices[0, 0] = experts.num_experts
        weights[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "[Ee]xpert.parallel|sentinel"):
            self.mlx._mlx_experts_forward(experts, hidden, indices, weights)
        for bad in (torch.empty(4, 48), torch.empty(4, 48, 17)):
            with self.subTest(shape=bad.shape):
                experts = _qwen_experts(torch.float32)
                experts.gate_up_proj = torch.nn.Parameter(bad)
                with self.assertRaisesRegex(ValueError, "[Ll]ayout|[Ss]hape|[Ww]eight|[Pp]rojection"):
                    self.mlx._mlx_experts_forward(experts, *_expert_inputs(torch.float32))
        for op in (self.routing, self.mm, self.scatter):
            op.assert_not_called()


@require_torch
@require_executorch
@require_torch_greater_or_equal("2.11")
class MLXIntegrationTest(unittest.TestCase):
    def setUp(self):
        from transformers.exporters import exporter_dynamo, exporter_executorch
        from transformers.exporters.configs import ExecutorchConfig
        from transformers.integrations import executorch

        self.shared = executorch
        self.mlx = self.et = exporter_executorch
        self.dynamo = exporter_dynamo
        self.config = ExecutorchConfig(
            backend="mlx",
            dtype="fp32",
            max_context_len=8,
            max_seq_len=4,
            strict=True,
            alloc_graph_input=False,
            alloc_mutable_buffers=False,
        )

    def test_full_decoder_window_dispatch_not_cache_prefix(self):
        config = _text_config()
        config.num_hidden_layers, config.num_kv_shared_layers = 4, 2
        config.layer_types = ["full_attention", "sliding_attention", "full_attention", "sliding_attention"]
        config.sliding_window = 4
        q, k, positions = torch.randn(1, 2, 2, 4), torch.randn(1, 1, 6, 4), torch.tensor([[2, 3]])
        for index in (2, 3):
            module = SimpleNamespace(config=config, layer_idx=index, is_causal=True)
            mask = mock.Mock(return_value=mock.sentinel.mask)
            op = mock.Mock(return_value=torch.randn_like(q))
            with (
                mock.patch.dict(
                    sys.modules, {"executorch.backends.mlx.llm.cache": SimpleNamespace(sliding_window_mask=mask)}
                ),
                mock.patch.object(torch.ops, "mlx", SimpleNamespace(custom_sdpa=op)),
            ):
                output, weights = self.mlx._mlx_in_graph_attention_forward(
                    module, q, k, k, None, positions, scaling=0.5
                )
            torch.testing.assert_close(output, op.return_value.transpose(1, 2).contiguous())
            self.assertTrue(output.is_contiguous())
            self.assertIsNone(weights)
            expected_mask = mask.return_value if index == 3 else None
            op.assert_called_once_with(
                q,
                k,
                k,
                start_pos=2 if index == 2 else 4,
                attn_mask=expected_mask,
                dropout_p=0.0,
                is_causal=index == 2,
                scale=0.5,
            )
            self.assertIs(op.call_args.kwargs["attn_mask"], expected_mask)
            if index == 3:
                mask.assert_called_once_with(2, 2, 4, 6, k.dtype)
            else:
                mask.assert_not_called()

    def test_orchestration_normalizes_before_capture_and_restores_on_failure(self):
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        stages = ["normalize", "capture", "program", "nodes", "settings", "lower"]
        for cache_mode, strict, failure in itertools.product(
            ("in-graph", "off-graph"), (False, True), (None, "normalize", "capture", "lower")
        ):
            with self.subTest(cache_mode=cache_mode, strict=strict, failure=failure):
                config = replace(self.config, cache_mode=cache_mode, strict=strict)
                model, events = _model(), []
                model._bind_cache_buffers = False
                inputs, shapes = self.shared._export_inputs(4, "full")
                inputs["use_cache"] = True  # The real normalizer must remove this output flag.
                private = self.mlx._MLXRecipeState(
                    model,
                    inputs,
                    shapes,
                    {"get_vocab_size": 11},
                    model,
                    cache_mode,
                    self.shared._resolve_cache_layout(model.config),
                )
                ep, result = SimpleNamespace(graph_module=object()), object()
                snapshot, attention = copy.deepcopy(config), model.config._attn_implementation
                bucketize, detach, tensor_detach = torch.bucketize, torch.detach, torch.Tensor.detach
                mappings = [
                    mapping
                    for registry in (ALL_ATTENTION_FUNCTIONS, ALL_MASK_ATTENTION_FUNCTIONS)
                    for mapping in (registry._global_mapping, registry._local_mapping)
                ]
                registry_snapshots = [dict(mapping) for mapping in mappings]
                attention_name = "mlx" if cache_mode == "in-graph" else "executorch_off_graph"

                def stage(name, result=None):
                    self.assertEqual(model.config._attn_implementation, attention_name)
                    self.assertIsNot(torch.bucketize, bucketize)
                    self.assertIs(torch.detach, detach)
                    self.assertIs(torch.Tensor.detach, tensor_detach)
                    self.assertEqual(
                        model._bind_cache_buffers,
                        cache_mode == "in-graph" and not strict and name in ("normalize", "capture"),
                    )
                    events.append(name)
                    if name == failure:
                        raise RuntimeError(name)
                    return result

                settings_factory = mock.Mock(side_effect=lambda: stage("settings", mock.sentinel.settings))

                def normalize(*args):
                    stage("normalize")
                    return original_normalize(*args)

                def capture(module, **kwargs):
                    stage("capture")
                    self.assertIs(module, model)
                    self.assertNotIn("use_cache", kwargs["kwargs"])
                    self.assertIs(kwargs["dynamic_shapes"], shapes)
                    self.assertEqual(kwargs["strict"], strict)
                    return ep

                def prepare(module, sample_inputs, export_config):
                    prepared = self.et.prepare_for_mlx(module, sample_inputs, export_config)
                    self.assertIs(prepared.model, model)
                    self.assertIs(prepared.sample_inputs, inputs)
                    self.assertIsNot(prepared.capture_config, config)
                    self.assertIs(prepared.capture_config.dynamic_shapes, shapes)
                    self.assertEqual(prepared.capture_config, replace(config, dynamic_shapes=shapes))
                    return replace(prepared, make_lowering_settings=settings_factory)

                original_normalize = self.dynamo.prepare_for_export
                self.assertIs(self.et._BACKEND_PREPARE["mlx"], self.et.prepare_for_mlx)
                prepare_mock = mock.Mock(side_effect=prepare)
                sample_inputs = {}
                with (
                    mock.patch.dict(self.et._BACKEND_PREPARE, {"mlx": prepare_mock}),
                    mock.patch.object(self.mlx, "_prepare_mlx", return_value=private),
                    mock.patch.object(self.mlx, "_load_mlx_dependencies") as load,
                    mock.patch.object(self.dynamo, "prepare_for_export", side_effect=normalize),
                    mock.patch.object(torch.export, "export", side_effect=capture),
                    mock.patch.object(self.et, "apply_fx_program_fixes", side_effect=lambda *a: stage("program")),
                    mock.patch.object(self.et, "apply_fx_node_fixes", side_effect=lambda *a: stage("nodes")),
                    mock.patch.object(
                        self.et, "_lower_to_executorch", side_effect=lambda *a: stage("lower", result)
                    ) as lower,
                    self.assertRaisesRegex(RuntimeError, failure) if failure else nullcontext(),
                ):
                    self.assertIs(self.et.ExecutorchExporter().export(model, sample_inputs, config), result)
                prepare_mock.assert_called_once_with(model, sample_inputs, config)
                expected = stages if failure is None else stages[: stages.index(failure) + 1]
                self.assertEqual(events, expected)
                load.assert_not_called()
                if "settings" in expected:
                    settings_factory.assert_called_once_with()
                    lower.assert_called_once_with(ep, mock.sentinel.settings)
                else:
                    settings_factory.assert_not_called()
                    lower.assert_not_called()
                self.assertEqual(config, snapshot)
                self.assertFalse(model._bind_cache_buffers)
                self.assertEqual(model.config._attn_implementation, attention)
                self.assertEqual([dict(mapping) for mapping in mappings], registry_snapshots)
                self.assertIs(torch.bucketize, bucketize)
                self.assertIs(torch.detach, detach)
                self.assertIs(torch.Tensor.detach, tensor_detach)

    def test_mlx_detach_semantics(self):
        detach, tensor_detach = torch.detach, torch.Tensor.detach
        value = torch.ones(2, requires_grad=True)
        with self.et.apply_patches("executorch"), self.et.apply_patches("executorch.mlx"):
            for output in (torch.detach(value), value.detach()):
                self.assertFalse(output.requires_grad)
                self.assertIsNot(output, value)
                self.assertEqual(output.data_ptr(), value.data_ptr())

        self.assertIs(torch.detach, detach)
        self.assertIs(torch.Tensor.detach, tensor_detach)
        self.assertFalse(torch.detach(value).requires_grad)
        self.assertFalse(value.detach().requires_grad)
