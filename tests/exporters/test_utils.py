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
"""Unit tests for the ``transformers.exporters`` pieces the per-model export tests don't reach.

The per-model exporter mixins in ``tests/exporters/test_export.py`` end-to-end-exercise
``prepare_for_export``, ``apply_patches`` / ``apply_fx_*_fixes``, the leaf-tensor helpers,
and the bundled input preparers — so those get real coverage on every CI run. What they
DON'T touch:

- The **auto factory** (``AutoExportConfig`` / ``AutoHfExporter``) — models bypass it and
  instantiate concrete exporters directly.
- **Config dict round-trips** — configs are built via constructor calls, never serialised.
- **Registration edge cases** — collision warnings and type-check rejections in
  ``register_exporter`` / ``register_export_config``.
- **`patch_attributes` restore-on-exception** — the happy path is exercised but the exception
  branch never fires in real exports.
- The **`decompose_prefill_decode` guard** against generators that bypass the top-level
  forward — real generators call ``forward`` many times, so the guard is dead code without a
  targeted test.
- **`register_patch`** unresolvable-path fallback — real registrations point at real paths.
- **Shared ExecuTorch helpers and import isolation** — backend-independent input/cache
  validation, nested attention restoration, and lazy optional dependencies.

Everything below targets one of those gaps.
"""

import builtins
import copy
import itertools
import os
import subprocess
import sys
import textwrap
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest import mock

from transformers.exporters import utils as exporter_utils
from transformers.exporters.auto import (
    AUTO_EXPORT_CONFIG_MAPPING,
    AUTO_EXPORTER_MAPPING,
    AutoExportConfig,
    AutoHfExporter,
    register_export_config,
    register_exporter,
)
from transformers.exporters.base import HfExporter
from transformers.exporters.configs import DynamoConfig, ExecutorchConfig, ExportFormat, OnnxConfig
from transformers.testing_utils import (
    require_executorch,
    require_onnx,
    require_onnxscript,
    require_torch,
    require_torch_greater_or_equal,
)
from transformers.utils.import_utils import is_torch_available


if is_torch_available():
    import torch
    from torch import nn

    from transformers import GenerationConfig
    from transformers.exporters.utils import (
        cast_leaf_tensors,
        decompose_prefill_decode,
        duplicate_leaf_tensors,
        patch_attributes,
        register_patch,
    )


CONCRETE_CONFIGS = [
    (OnnxConfig, ExportFormat.ONNX),
    (DynamoConfig, ExportFormat.DYNAMO),
    (ExecutorchConfig, ExportFormat.EXECUTORCH),
]


# ─────────────────────────────────────────────────────────────────────────────
# Auto factory + config serialisation
# ─────────────────────────────────────────────────────────────────────────────


class ExportConfigMixinTest(unittest.TestCase):
    def test_to_dict_from_dict_roundtrip(self):
        for config_cls, export_format in CONCRETE_CONFIGS:
            with self.subTest(config_cls.__name__):
                original = config_cls(dynamic=True)
                restored = config_cls.from_dict(original.to_dict())
                self.assertEqual(restored, original)
                self.assertIs(restored.export_format, export_format)


class ExecutorchConfigTest(unittest.TestCase):
    def test_causal_lm_defaults(self):
        config = ExecutorchConfig()
        self.assertEqual(config.cache_mode, "in-graph")
        self.assertEqual(config.max_context_len, 1024)
        self.assertEqual(config.max_seq_len, 512)
        self.assertEqual(config.dtype, "bf16")
        self.assertEqual(config.logits_to_keep, "full")

    def test_causal_lm_options_roundtrip(self):
        for cache_mode, max_seq_len, dtype, logits_to_keep in (
            ("in-graph", 256, "fp16", "last"),
            ("off-graph", None, "fp32", "selected"),
        ):
            with self.subTest(cache_mode=cache_mode, logits_to_keep=logits_to_keep):
                original = ExecutorchConfig(
                    cache_mode=cache_mode,
                    max_context_len=4096,
                    max_seq_len=max_seq_len,
                    dtype=dtype,
                    logits_to_keep=logits_to_keep,
                    strict=True,
                    alloc_mutable_buffers=False,
                )
                serialized = original.to_dict()
                self.assertEqual(ExecutorchConfig.from_dict(serialized), original)
                self.assertEqual(AutoExportConfig.from_dict(serialized), original)


class AutoExportConfigTest(unittest.TestCase):
    def test_from_dict_dispatches_to_concrete_config(self):
        for config_cls, export_format in CONCRETE_CONFIGS:
            with self.subTest(config_cls.__name__):
                self.assertIsInstance(AutoExportConfig.from_dict({"export_format": export_format.value}), config_cls)
                # Enum inputs also work — serialised configs may hold either form.
                self.assertIsInstance(AutoExportConfig.from_dict({"export_format": export_format}), config_cls)

    def test_from_dict_missing_export_format_raises(self):
        with self.assertRaisesRegex(ValueError, "export_format"):
            AutoExportConfig.from_dict({})

    def test_from_dict_unknown_format_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown exporter type"):
            AutoExportConfig.from_dict({"export_format": "not_a_real_backend"})


class AutoHfExporterTest(unittest.TestCase):
    def _check_dispatch(self, config):
        expected_cls = AUTO_EXPORTER_MAPPING[config.export_format.value]
        self.assertIsInstance(AutoHfExporter.from_config(config), expected_cls)
        # Same dispatch works when starting from a plain dict.
        self.assertIsInstance(AutoHfExporter.from_config(config.to_dict()), expected_cls)

    @require_torch
    def test_from_config_dispatches_dynamo(self):
        self._check_dispatch(DynamoConfig())

    @require_torch
    @require_onnx
    @require_onnxscript
    def test_from_config_dispatches_onnx(self):
        self._check_dispatch(OnnxConfig())

    @require_torch
    @require_executorch
    def test_from_config_dispatches_executorch(self):
        self._check_dispatch(ExecutorchConfig())

    def test_from_config_raises_on_unknown_format(self):
        with self.assertRaisesRegex(ValueError, "Unsupported export config"):
            AutoHfExporter.from_config({"export_format": "not_a_real_backend"})
        with self.assertRaisesRegex(ValueError, "Unsupported export config"):
            AutoHfExporter.from_config({})


class RegistrationTest(unittest.TestCase):
    """Cover the edge cases of `register_exporter` / `register_export_config` that normal
    registrations at module load don't hit — the type-check rejection paths. The mappings are
    temporarily patched so registrations never leak into other tests."""

    def test_register_exporter_rejects_non_subclass(self):
        with mock.patch.dict(AUTO_EXPORTER_MAPPING):
            with self.assertRaisesRegex(TypeError, "HfExporter"):

                @register_exporter("bad")
                class _NotAnExporter:
                    pass

    def test_register_export_config_rejects_non_subclass(self):
        with mock.patch.dict(AUTO_EXPORT_CONFIG_MAPPING):
            with self.assertRaisesRegex(TypeError, "ExportConfigMixin"):

                @register_export_config("bad_config")
                class _NotAConfig:
                    pass

    def test_register_exporter_installs_stub(self):
        # Sanity check that a legit registration is wired through — protects against a future
        # refactor that would break the decorator without breaking any real export test.
        with mock.patch.dict(AUTO_EXPORTER_MAPPING):

            @register_exporter("stub_exporter")
            class _StubExporter(HfExporter):
                required_packages = []

                def export(self, model, sample_inputs, config):
                    return None

            self.assertIs(AUTO_EXPORTER_MAPPING["stub_exporter"], _StubExporter)
        self.assertNotIn("stub_exporter", AUTO_EXPORTER_MAPPING)


# ─────────────────────────────────────────────────────────────────────────────
# Registry edge cases the happy-path exports don't exercise
# ─────────────────────────────────────────────────────────────────────────────


@require_torch
class SharedExecutorchHelpersTest(unittest.TestCase):
    def test_capture_scope_is_gated_and_restores_nested_flags(self):
        from transformers.integrations import executorch as shared

        self.assertTrue(issubclass(shared._InGraphCacheAndOutput, shared._CacheAndOutputMixin))
        self.assertNotIn("_forward_with_cache", vars(shared._InGraphCacheAndOutput))
        for strict, fail in itertools.product((False, True), (False, True)):
            with self.subTest(strict=strict, fail=fail):
                wrapper = shared._InGraphCacheAndOutput(nn.Identity(), "full")
                self.assertFalse(wrapper._bind_cache_buffers)
                with self.assertRaisesRegex(RuntimeError, "outer") if fail else nullcontext():
                    with shared._in_graph_cache_capture_scope(wrapper, strict=strict):
                        self.assertEqual(wrapper._bind_cache_buffers, not strict)
                        for inner_strict in (False, True):
                            with self.assertRaisesRegex(RuntimeError, "inner") if fail else nullcontext():
                                with shared._in_graph_cache_capture_scope(wrapper, strict=inner_strict):
                                    self.assertEqual(wrapper._bind_cache_buffers, not strict or not inner_strict)
                                    if fail:
                                        raise RuntimeError("inner")
                            self.assertEqual(wrapper._bind_cache_buffers, not strict)
                        if fail:
                            raise RuntimeError("outer")
                self.assertFalse(wrapper._bind_cache_buffers)

    def test_nonstrict_forward_restores_cache_references_without_reverting_writes(self):
        from transformers import LlamaConfig, StaticCache
        from transformers.integrations import executorch as shared

        config = LlamaConfig(hidden_size=8, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
        attributes = (("keys", "key_cache"), ("values", "value_cache"), ("cumulative_length", "cumulative_length"))
        for fail in (False, True):
            with self.subTest(fail=fail):
                cache = StaticCache(config=config, max_cache_len=8)
                cache.early_initialization(1, 1, 4, torch.float32, torch.device("cpu"))
                model = nn.Module()
                wrapper = shared._InGraphCacheAndOutput(model, "full", cache=cache)
                original = dict(wrapper.named_buffers())
                replacements = {name: torch.zeros_like(tensor) for name, tensor in original.items()}

                def forward(**kwargs):
                    self.assertIs(kwargs["past_key_values"], cache)
                    for index, layer in enumerate(cache.layers):
                        for attribute, prefix in attributes:
                            self.assertIs(getattr(layer, attribute), replacements[f"{prefix}_{index}"])
                        self.assertEqual(layer.cumulative_length.item(), 3)
                        layer.keys.add_(7)
                        layer.values.add_(9)
                    if fail:
                        raise RuntimeError("forward")
                    return SimpleNamespace(logits=torch.ones(1, 1, 11))

                with (
                    mock.patch.object(model, "forward", side_effect=forward),
                    self.assertRaisesRegex(RuntimeError, "forward") if fail else nullcontext(),
                ):
                    with shared._in_graph_cache_capture_scope(wrapper, strict=False):
                        # Model the registered-buffer substitution used by non-strict capture.
                        output = torch.func.functional_call(
                            wrapper,
                            replacements,
                            (),
                            {"input_ids": torch.tensor([[1]]), "cache_position": torch.tensor([3])},
                        )
                        torch.testing.assert_close(output, torch.ones(1, 1, 11))
                self.assertFalse(wrapper._bind_cache_buffers)
                for index, layer in enumerate(cache.layers):
                    for attribute, prefix in attributes:
                        name = f"{prefix}_{index}"
                        self.assertIs(getattr(layer, attribute), original[name])
                        self.assertIs(getattr(wrapper, name), original[name])
                        self.assertEqual(torch.count_nonzero(original[name]).item(), 0)
                        expected = 3 if prefix == "cumulative_length" else 7 if prefix == "key_cache" else 9
                        torch.testing.assert_close(replacements[name], torch.full_like(replacements[name], expected))

        # A missing late destination must fail before even temporarily rebinding earlier ones.
        writes = []

        class RecordingDestinations(dict):
            def __setitem__(self, key, value):
                writes.append(key)
                super().__setitem__(key, value)

        for layer in cache.layers:
            layer.__dict__ = RecordingDestinations(vars(layer))
        del vars(cache.layers[-1])["cumulative_length"]
        with mock.patch.dict(wrapper._buffers, replacements), self.assertRaisesRegex(KeyError, "cumulative_length"):
            with shared._cache_buffer_scope(wrapper, cache):
                self.fail("Invalid cache entered the binding scope")
        self.assertEqual(writes, [])
        for index, layer in enumerate(cache.layers):
            for attribute, prefix in attributes:
                if attribute in vars(layer):
                    self.assertIs(getattr(layer, attribute), original[f"{prefix}_{index}"])

    def test_helper_ownership_and_import_isolation(self):
        code = textwrap.dedent(
            """
            import sys
            from types import SimpleNamespace
            from unittest import mock

            import torch

            # Existing model utilities load TorchAO when installed; exercise its absence.
            with (
                mock.patch("transformers.utils.import_utils.is_torchao_available", return_value=False),
                mock.patch("transformers.utils.is_torchao_available", return_value=False),
            ):
                from transformers.integrations import executorch as shared

            for name in (
                "_positive_int", "_cache_layout", "_export_inputs", "_LogitsToKeepMixin", "_CacheAndOutputMixin",
                "_OffGraphWrapper", "_attention_mask", "_attention_scope", "_check_attention_options",
                "_off_graph_cache_id", "_off_graph_attention_forward",
            ):
                assert getattr(shared, name).__module__ == shared.__name__, name

            class DummyLM(torch.nn.Module):
                def forward(self, input_ids=None, **kwargs):
                    self.last_input_ids = input_ids
                    self.last_kwargs = kwargs
                    indices = kwargs.get("logits_to_keep", 0)
                    indices = slice(-indices, None) if isinstance(indices, int) else indices
                    values = input_ids.float().unsqueeze(-1) if input_ids is not None else kwargs["inputs_embeds"]
                    return SimpleNamespace(logits=values[:, indices])

            for mode in ("full", "last", "selected"):
                model = DummyLM()
                wrapper = shared._OffGraphWrapper(model, mode)
                inputs, _ = shared._export_inputs(4, mode)
                inputs["input_ids"] = torch.tensor([[1, 2, 3]])
                if mode == "selected":
                    inputs["logits_to_keep"] = torch.tensor([2, 0])
                output = wrapper(**inputs)
                expected = {"full": [1, 2, 3], "last": [3], "selected": [3, 1]}[mode]
                torch.testing.assert_close(output, torch.tensor(expected).float().reshape(1, -1, 1))
                assert model.last_kwargs["use_cache"] is False
                assert model.last_kwargs["past_key_values"] is None
                assert model.last_kwargs["cache_position"] is inputs["cache_position"]
                torch.testing.assert_close(model.last_kwargs["position_ids"], inputs["cache_position"][None])
                if mode == "full":
                    assert "logits_to_keep" not in model.last_kwargs
                elif mode == "last":
                    assert model.last_kwargs["logits_to_keep"] == 1
                else:
                    assert model.last_kwargs["logits_to_keep"] is inputs["logits_to_keep"]

            assert issubclass(shared._CacheAndOutputMixin, shared._LogitsToKeepMixin)
            assert not hasattr(shared._CacheAndOutputMixin, "_bind_cache_buffers")

            class CacheWrapper(shared._CacheAndOutputMixin, torch.nn.Module):
                def __init__(self, model, mode, cache):
                    super().__init__()
                    self.model = model
                    self.logits_to_keep_mode = mode
                    self._get_cache = mock.Mock(return_value=cache)

            for mode in ("full", "last", "selected"):
                model = DummyLM()
                lengths = [torch.tensor(17), torch.tensor(23)]
                cache = SimpleNamespace(layers=[SimpleNamespace(cumulative_length=t) for t in lengths])
                cache.layers.append(SimpleNamespace())  # Layers without a length remain supported.
                wrapper = CacheWrapper(model, mode, cache)
                selection = torch.tensor([2, 0]) if mode == "selected" else None
                for start in (0, 9, None):
                    positions = torch.arange(start, start + 3) if start is not None else None
                    values = torch.tensor([[1, 2, 3]])
                    inputs = {"input_ids": values} if start is not None else {
                        "inputs_embeds": values.float().unsqueeze(-1)
                    }
                    wrapper._get_cache.reset_mock()
                    output = wrapper(**inputs, cache_position=positions, logits_to_keep=selection)
                    wrapper._get_cache.assert_called_once_with()
                    expected = {"full": [1, 2, 3], "last": [3], "selected": [3, 1]}[mode]
                    torch.testing.assert_close(output, torch.tensor(expected).float().reshape(1, -1, 1))
                    assert model.last_input_ids is inputs.get("input_ids")
                    assert model.last_kwargs["inputs_embeds"] is inputs.get("inputs_embeds")
                    assert model.last_kwargs["past_key_values"] is cache
                    assert model.last_kwargs["use_cache"] is True
                    assert model.last_kwargs["attention_mask"] is None
                    assert model.last_kwargs["cache_position"] is positions
                    if positions is None:
                        assert model.last_kwargs["position_ids"] is None
                    else:
                        torch.testing.assert_close(model.last_kwargs["position_ids"], positions[None])
                    for layer, length in zip(cache.layers, lengths):
                        assert layer.cumulative_length is length
                        assert length.item() == (start if start is not None else 9)
                    if mode == "full":
                        assert "logits_to_keep" not in model.last_kwargs
                    elif mode == "last":
                        assert model.last_kwargs["logits_to_keep"] == 1
                    else:
                        assert model.last_kwargs["logits_to_keep"] is selection

            forbidden = (
                "transformers.exporters.exporter_executorch",
                "executorch.backends.mlx",
                "executorch.extension.llm.cache",
                "executorch.extension.llm.export.kv_cache",
                "mlx",
                "torchao",
            )
            loaded = [name for name in sys.modules if any(
                name == prefix or name.startswith(prefix + ".") for prefix in forbidden
            )]
            assert not loaded, loaded
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_generated_inputs_and_independent_selection_dimension(self):
        from transformers.integrations import executorch as shared

        for limit in (1, 2, 8):
            inputs, shapes = shared._export_inputs(limit, "selected")
            self.assertEqual(inputs["input_ids"].shape, (1, min(3, limit)))
            torch.testing.assert_close(inputs["cache_position"], torch.arange(min(3, limit)))
            self.assertTrue(all(t.dtype == torch.int64 and t.device.type == "cpu" for t in inputs.values()))
            if limit == 1:
                self.assertEqual(shapes, dict.fromkeys(inputs))
            else:
                seq, selected = shapes["input_ids"][1], shapes["logits_to_keep"][0]
                self.assertIs(seq, shapes["cache_position"][0])
                self.assertIsNot(seq, selected)
                self.assertNotEqual(seq.__name__, selected.__name__)
                self.assertEqual((seq.min, seq.max, selected.min, selected.max), (1, limit, 1, limit))
        wrapper = shared._OffGraphWrapper(nn.Identity(), "selected")
        for bad in (None, torch.tensor([0.0]), torch.tensor([[0]])):
            with self.subTest(indices=bad), self.assertRaisesRegex(ValueError, r"int64\[K\]"):
                wrapper(torch.tensor([[1]]), torch.tensor([0]), bad)

    def test_shared_kv_prefix_and_donor_mapping(self):
        from transformers import LlamaConfig
        from transformers.integrations import executorch as shared

        config = LlamaConfig(hidden_size=8, num_attention_heads=2, num_key_value_heads=1)
        config.num_hidden_layers, config.num_kv_shared_layers = 5, 2
        config.layer_types = [
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ]
        config.sliding_window = 4
        self.assertEqual(shared._cache_layout(config), ([1, 1, 1], [4, 4, 4], [0, 4, 0]))
        for index, donor in enumerate((0, 1, 2, 1, 2)):
            module = SimpleNamespace(config=config, layer_idx=index, is_kv_shared_layer=index >= 3)
            self.assertEqual(shared._off_graph_cache_id(module), donor)
        for field, value in (
            ("num_kv_shared_layers", True),
            ("num_kv_shared_layers", 5),
            ("sliding_window", None),
            ("layer_types", ["full_attention"]),
            ("layer_types", ["full_attention"] * 3 + ["sliding_attention"] * 2),
        ):
            invalid = copy.deepcopy(config)
            setattr(invalid, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                shared._cache_layout(invalid)

    def test_shared_attention_scope_restores_nested_custom_callbacks_on_exception(self):
        from transformers import LlamaConfig
        from transformers.integrations import executorch as shared
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

        for existing in (False, True):
            with self.subTest(existing=existing), ExitStack() as stack:
                model = nn.Module()
                model.config = LlamaConfig()

                def set_attn_implementation(name):
                    model.config._attn_implementation = name
                    model.config._attn_was_changed = True

                model.set_attn_implementation = set_attn_implementation
                model.local_attention, model.local_mask = AttentionInterface(), AttentionMaskInterface()
                model.config.sub_configs = {"nested": object}
                model.config.nested = SimpleNamespace(_attn_implementation_internal="eager")
                configs = (model.config, model.config.nested)
                snapshots = [dict(vars(config)) for config in configs]
                name = "custom_export_attention"
                registries = (
                    ALL_ATTENTION_FUNCTIONS,
                    ALL_MASK_ATTENTION_FUNCTIONS,
                    model.local_attention,
                    model.local_mask,
                )
                mappings = {id(m): m for r in registries for m in (r._global_mapping, r._local_mapping)}
                for mapping in mappings.values():
                    stack.enter_context(mock.patch.dict(mapping))
                    mapping.pop(name, None)
                    if existing:
                        mapping[name] = mock.Mock()
                before = [dict(mapping) for mapping in mappings.values()]
                attention, mask, inner_attention, inner_mask = (mock.Mock() for _ in range(4))
                original_select = model.set_attn_implementation

                def select(selected):
                    original_select(selected)
                    model.config.nested._attn_implementation_internal = selected
                    model.config.nested._attn_was_changed = True

                stack.enter_context(mock.patch.object(model, "set_attn_implementation", side_effect=select))
                with self.assertRaisesRegex(RuntimeError, "outer"):
                    with shared._attention_scope(model, name, attention, mask):
                        self.assertEqual(model.config._attn_implementation, name)
                        self.assertEqual(model.config.nested._attn_implementation_internal, name)
                        outer_configs = [dict(vars(config)) for config in configs]
                        outer_mappings = [dict(mapping) for mapping in mappings.values()]
                        for registry, callback in zip(registries, (attention, mask, attention, mask)):
                            self.assertIs(registry[name], callback)
                        with self.assertRaisesRegex(RuntimeError, "inner"):
                            with shared._attention_scope(model, name, inner_attention, inner_mask):
                                for registry, callback in zip(
                                    registries, (inner_attention, inner_mask, inner_attention, inner_mask)
                                ):
                                    self.assertIs(registry[name], callback)
                                model.config.nested._attn_was_changed = False
                                raise RuntimeError("inner")
                        self.assertEqual([dict(vars(config)) for config in configs], outer_configs)
                        self.assertEqual([dict(mapping) for mapping in mappings.values()], outer_mappings)
                        raise RuntimeError("outer")
                self.assertEqual([dict(vars(config)) for config in configs], snapshots)
                self.assertEqual([dict(mapping) for mapping in mappings.values()], before)


@require_torch
class ExecutorchImportIsolationTest(unittest.TestCase):
    def test_exporter_import_keeps_mlx_dependencies_lazy(self):
        code = textwrap.dedent(
            """
            import builtins
            import sys
            from unittest import mock

            # Core ExecuTorch imports generic LLM fallback ops, but not these recipe dependencies.
            forbidden = (
                "executorch.backends.mlx",
                "executorch.extension.llm.cache",
                "executorch.extension.llm.export.kv_cache",
                "executorch.extension.llm.export.model_metadata",
                "mlx",
            )
            original_import = builtins.__import__

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                names = (name, *(f"{name}.{item}" for item in (fromlist or ())))
                if any(item == prefix or item.startswith(prefix + ".") for item in names for prefix in forbidden):
                    raise AssertionError(f"Non-MLX import attempted an MLX/cache import: {name}")
                return original_import(name, globals, locals, fromlist, level)

            with mock.patch("builtins.__import__", side_effect=guarded_import):
                from transformers.exporters import exporter_executorch

            assert callable(exporter_executorch.prepare_for_xnnpack)
            assert callable(exporter_executorch.prepare_for_cuda)
            loaded = [name for name in sys.modules if any(
                name == prefix or name.startswith(prefix + ".") for prefix in forbidden
            )]
            assert not loaded, loaded
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@require_torch
@require_executorch
class ExecutorchTensorReshapePatchTest(unittest.TestCase):
    @contextmanager
    def _patches_with_restoration(self):
        from transformers.exporters import exporter_executorch

        patches = [
            patch
            for patch in exporter_utils._PATCHES["executorch"]
            if patch[0] is torch.Tensor and patch[1] == "reshape"
        ]
        self.assertEqual(len(patches), 1)
        self.assertIs(patches[0][2], exporter_executorch._patch_tensor_reshape)
        original = torch.Tensor.reshape
        try:
            with patch_attributes(patches):
                self.assertIsNot(torch.Tensor.reshape, original)
                yield
        finally:
            self.assertIs(torch.Tensor.reshape, original)

    def test_shape_argument_forms(self):
        operations = {
            "varargs": lambda x: x.reshape(2, -1),
            "tuple": lambda x: x.reshape((2, -1)),
            "list": lambda x: x.reshape([2, -1]),
            "keyword_tuple": lambda x: x.reshape(shape=(2, -1)),
            "keyword_list": lambda x: x.reshape(shape=[2, -1]),
            "size": lambda x: x.reshape(x.shape),
            "single_dim": lambda x: x.reshape(-1),
        }
        for x in (torch.randn(3, 4), torch.randn(4, 3).t()):
            expected = {name: operation(x) for name, operation in operations.items()}
            with self._patches_with_restoration():
                for name, operation in operations.items():
                    with self.subTest(operation=name, contiguous=x.is_contiguous()):
                        actual = operation(x)
                        torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)
                        self.assertTrue(actual.is_contiguous())
                        self.assertEqual(actual.data_ptr() == x.data_ptr(), x.is_contiguous())

    def test_scalar_shape_forms(self):
        x = torch.tensor(1.0)
        operations = (
            lambda: x.reshape([]),
            lambda: x.reshape(()),
            lambda: x.reshape(shape=()),
            lambda: x.reshape(1),
        )
        expected = [operation() for operation in operations]
        with self._patches_with_restoration():
            for operation, reference in zip(operations, expected):
                torch.testing.assert_close(operation(), reference)

    def test_invalid_arguments(self):
        x = torch.randn(3, 4)
        invalid_arguments = (
            lambda: x.reshape(),
            lambda: x.reshape(2, shape=(2, 6)),
            lambda: x.reshape(size=(2, 6)),
            lambda: x.reshape(shape=(2, 6), unknown=True),
        )
        with self._patches_with_restoration():
            for operation in invalid_arguments:
                with self.assertRaises(TypeError):
                    operation()
            with self.assertRaisesRegex(RuntimeError, "invalid for input"):
                x.reshape(5)

    def _check_strict_export(self, operation, sample, expected_clones):
        class Model(nn.Module):
            def forward(self, x):
                return operation(x)

        with self._patches_with_restoration():
            program = torch.export.export(
                Model(),
                (sample[:3],),
                dynamic_shapes={"x": {0: torch.export.Dim("rows", min=2, max=6)}},
                strict=True,
            )
        self.assertTrue(program.range_constraints)
        clones = [node for node in program.graph.nodes if node.target == torch.ops.aten.clone.default]
        self.assertEqual(len(clones), expected_clones)
        for node in clones:
            self.assertEqual(node.kwargs.get("memory_format"), torch.contiguous_format)
        exported = program.module()
        for rows in (2, 5):
            x = sample[:rows]
            actual = exported(x)
            torch.testing.assert_close(actual, operation(x), rtol=0, atol=0)
            self.assertTrue(actual.is_contiguous())
            self.assertEqual(actual.data_ptr() == x.data_ptr(), expected_clones == 0)

    def test_strict_symbolic_shape_forms(self):
        operations = {
            "varargs": lambda x: x.reshape(x.shape[0], -1),
            "tuple": lambda x: x.reshape((x.shape[0], -1)),
            "list": lambda x: x.reshape([x.shape[0], -1]),
            "keyword": lambda x: x.reshape(shape=(x.shape[0], -1)),
            "size": lambda x: x.reshape(x.shape),
        }
        sample = torch.randn(5, 2, 4)
        for name, operation in operations.items():
            for contiguous in (True, False):
                with self.subTest(operation=name, contiguous=contiguous):
                    self._check_strict_export(
                        operation, sample if contiguous else sample.transpose(1, 2), int(not contiguous)
                    )

    def test_strict_internal_transpose(self):
        self._check_strict_export(lambda x: x.transpose(1, 2).reshape(shape=(x.shape[0], -1)), torch.randn(5, 2, 4), 1)

    def test_restoration_after_strict_export_failure(self):
        class InvalidShape(nn.Module):
            def forward(self, x):
                return x.reshape(5)

        with self.assertRaisesRegex(RuntimeError, "invalid for input"):
            with self._patches_with_restoration():
                torch.export.export(InvalidShape(), (torch.randn(3, 4),), strict=True)


@require_torch
@require_executorch
class ExecutorchBackendPipelineTest(unittest.TestCase):
    def setUp(self):
        from transformers.exporters import exporter_dynamo, exporter_executorch

        self.dynamo, self.et = exporter_dynamo, exporter_executorch

    def test_legacy_preparation_and_shared_lowering(self):
        for backend in ("xnnpack", "cuda"):
            for custom_allocation in (False, True):
                with self.subTest(backend=backend, custom_allocation=custom_allocation):
                    model = nn.Linear(2, 2)
                    inputs = {"input": torch.randn(2, 3).t(), "nested": [torch.ones(1), "keep"]}
                    config = ExecutorchConfig(backend=backend, alloc_graph_input=not custom_allocation)
                    ep, events = SimpleNamespace(graph_module=object()), []
                    edge = mock.Mock()

                    def stage(name, result=None):
                        events.append(name)
                        return result

                    def capture(prepared_model, prepared_inputs, *, config):
                        stage("capture")
                        self.assertIs(prepared_model, model)
                        self.assertFalse(any(p.requires_grad for p in model.parameters()))
                        self.assertEqual(model.weight.device.type, "cpu")
                        self.assertEqual(model.weight.dtype, torch.bfloat16 if backend == "cuda" else torch.float32)
                        self.assertTrue(prepared_inputs["input"].is_contiguous())
                        torch.testing.assert_close(prepared_inputs["input"], inputs["input"])
                        self.assertIs(prepared_inputs["nested"][0], inputs["nested"][0])
                        self.assertEqual(prepared_inputs["nested"][1], "keep")
                        self.assertEqual(config.backend, backend)
                        return ep

                    with (
                        mock.patch.object(torch.cuda, "is_available", return_value=True),
                        mock.patch.object(model, "to", wraps=model.to) as move,
                        mock.patch.object(
                            self.et, "XnnpackPartitioner", side_effect=lambda: stage("partitioner", mock.sentinel.xnn)
                        ) as xnn,
                        # These names are deliberately absent on CPU-only imports.
                        mock.patch.object(self.et, "CudaBackend", create=True) as cuda_backend,
                        mock.patch.object(
                            self.et,
                            "CudaPartitioner",
                            create=True,
                            side_effect=lambda specs: stage("partitioner", mock.sentinel.cuda),
                        ) as cuda,
                        mock.patch.object(self.dynamo.DynamoExporter, "export", side_effect=capture),
                        mock.patch.object(self.et, "apply_fx_program_fixes", side_effect=lambda *a: stage("program")),
                        mock.patch.object(self.et, "apply_fx_node_fixes", side_effect=lambda *a: stage("nodes")),
                        mock.patch.object(
                            self.et,
                            "EdgeCompileConfig",
                            side_effect=lambda **kw: stage("compile", mock.sentinel.compile),
                        ) as compile_config,
                        mock.patch.object(self.et, "MemoryPlanningPass") as memory,
                        mock.patch.object(
                            self.et,
                            "ExecutorchBackendConfig",
                            side_effect=lambda **kw: stage("backend", mock.sentinel.backend),
                        ) as backend_config,
                        mock.patch.object(
                            self.et, "to_edge_transform_and_lower", side_effect=lambda *a, **kw: stage("edge", edge)
                        ) as lower,
                        mock.patch.object(
                            edge, "to_executorch", side_effect=lambda **kw: stage("executorch", mock.sentinel.result)
                        ) as to_executorch,
                    ):
                        result = self.et.ExecutorchExporter().export(model, inputs, config)
                    self.assertIs(result, mock.sentinel.result)
                    expected = ["partitioner", "capture", "program", "nodes", "compile", "edge"]
                    self.assertEqual(events, expected + (["backend"] if custom_allocation else []) + ["executorch"])
                    lower.assert_called_once_with(
                        ep,
                        partitioner=[mock.sentinel.xnn if backend == "xnnpack" else mock.sentinel.cuda],
                        compile_config=mock.sentinel.compile,
                    )
                    self.assertIn(
                        torch.ops.aten._fft_c2c.default,
                        compile_config.call_args.kwargs["_core_aten_ops_exception_list"],
                    )
                    if backend == "xnnpack":
                        xnn.assert_called_once_with()
                        cuda.assert_not_called()
                        move.assert_called_once_with(device="cpu")
                    else:
                        xnn.assert_not_called()
                        cuda_backend.generate_method_name_compile_spec.assert_called_once_with("Linear")
                        cuda.assert_called_once_with([cuda_backend.generate_method_name_compile_spec.return_value])
                        move.assert_called_once_with(dtype=torch.bfloat16)
                    if custom_allocation:
                        memory.assert_called_once_with(
                            alloc_graph_input=False, alloc_graph_output=True, alloc_mutable_buffers=True
                        )
                        backend_config.assert_called_once_with(memory_planning_pass=memory.return_value)
                        to_executorch.assert_called_once_with(config=mock.sentinel.backend)
                    else:
                        memory.assert_not_called()
                        backend_config.assert_not_called()
                        to_executorch.assert_called_once_with(config=None)
                    self.assertFalse(inputs["input"].is_contiguous())

    def test_xnnpack_experts_and_cuda_preparation_guards(self):
        model = mock.Mock(spec=self.et.PreTrainedModel)
        model.to.return_value = model
        model._can_set_experts_implementation.return_value = True
        with mock.patch.object(self.et, "XnnpackPartitioner"):
            prepared = self.et.prepare_for_xnnpack(model, {}, ExecutorchConfig())
        self.assertIsInstance(prepared, self.et._BackendPreparation)
        self.assertIs(prepared.model, model)
        model.requires_grad_.assert_called_once_with(False)
        model.to.assert_called_once_with(device="cpu")
        model.set_experts_implementation.assert_called_once_with("batched_mm")

        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = nn.Linear(2, 2).to(dtype=dtype)
                with (
                    mock.patch.object(torch.cuda, "is_available", return_value=False),
                    mock.patch.object(model, "to", wraps=model.to) as move,
                    mock.patch.object(self.et, "CudaPartitioner", create=True) as partitioner,
                    self.assertRaisesRegex(RuntimeError, "CUDA is not available"),
                ):
                    self.et.prepare_for_cuda(model, {}, ExecutorchConfig(backend="cuda"))
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                self.assertEqual(model.weight.dtype, dtype)
                move.assert_not_called()
                partitioner.assert_not_called()
        model = nn.Linear(2, 2).to(dtype=torch.bfloat16)
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(model, "to", wraps=model.to) as move,
            mock.patch.object(self.et, "CudaBackend", create=True),
            mock.patch.object(self.et, "CudaPartitioner", create=True),
        ):
            self.et.prepare_for_cuda(model, {}, ExecutorchConfig(backend="cuda"))
        move.assert_not_called()
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_backend_config_allocation_flags(self):
        for flags in (
            (True, True, True),
            (False, True, True),
            (True, False, True),
            (True, True, False),
            (False, False, False),
        ):
            with self.subTest(flags=flags):
                kwargs = dict(zip(("alloc_graph_input", "alloc_graph_output", "alloc_mutable_buffers"), flags))
                with (
                    mock.patch.object(self.et, "MemoryPlanningPass") as memory,
                    mock.patch.object(self.et, "ExecutorchBackendConfig") as backend,
                ):
                    result = self.et._get_backend_config(ExecutorchConfig(**kwargs))
                if all(flags):
                    self.assertIsNone(result)
                    memory.assert_not_called()
                    backend.assert_not_called()
                else:
                    memory.assert_called_once_with(**kwargs)
                    backend.assert_called_once_with(memory_planning_pass=memory.return_value)
                    self.assertIs(result, backend.return_value)

    def test_uniform_dispatch_uses_prepared_record_and_deferred_settings(self):
        self.assertEqual(set(self.et._BACKEND_PREPARE), {"xnnpack", "cuda", "mlx"})
        for backend in ("xnnpack", "cuda", "mlx"):
            with self.subTest(backend=backend):
                config = ExecutorchConfig(backend=backend)
                capture_config = ExecutorchConfig(backend=backend, strict=True, dynamic_shapes={})
                model, inputs = object(), {"original": object()}
                settings = self.et._LoweringSettings([], object(), mock.Mock())
                factory = mock.Mock(return_value=settings)
                record = self.et._BackendPreparation(object(), {}, capture_config, factory)
                preparer = mock.Mock(return_value=record)
                with (
                    mock.patch.dict(self.et._BACKEND_PREPARE, {backend: preparer}),
                    mock.patch.object(self.dynamo.DynamoExporter, "export") as capture,
                    mock.patch.object(self.et, "apply_fx_program_fixes") as program_fix,
                    mock.patch.object(self.et, "apply_fx_node_fixes") as node_fix,
                    mock.patch.object(self.et, "_lower_to_executorch") as lower,
                ):
                    result = self.et.ExecutorchExporter().export(model, inputs, config)
                preparer.assert_called_once_with(model, inputs, config)
                capture.assert_called_once_with(record.model, record.sample_inputs, config=capture_config)
                program_fix.assert_called_once_with("executorch", capture.return_value)
                node_fix.assert_called_once_with("executorch", capture.return_value.graph_module)
                factory.assert_called_once_with()
                lower.assert_called_once_with(capture.return_value, settings)
                self.assertIs(result, lower.return_value)

    def test_strict_dynamic_and_dict_config_passthrough(self):
        shapes = {"input": {0: torch.export.Dim("rows", min=1, max=4)}}
        for backend in ("xnnpack", "cuda"):
            for strict, dynamic, explicit_shapes in (
                (False, False, None),
                (True, True, None),
                (False, True, {}),
                (True, True, shapes),
            ):
                with self.subTest(backend=backend, strict=strict, dynamic=dynamic, shapes=explicit_shapes):
                    config = ExecutorchConfig(
                        backend=backend,
                        strict=strict,
                        dynamic=dynamic,
                        dynamic_shapes=explicit_shapes,
                        prefer_deferred_runtime_asserts_over_guards=True,
                    )
                    snapshot = vars(config).copy()
                    model, inputs = nn.Linear(2, 2), {"input": torch.ones(3, 2)}
                    with (
                        mock.patch.object(torch.cuda, "is_available", return_value=True),
                        mock.patch.object(self.et, "XnnpackPartitioner"),
                        mock.patch.object(self.et, "CudaBackend", create=True),
                        mock.patch.object(self.et, "CudaPartitioner", create=True),
                        mock.patch.object(torch.export, "export") as capture,
                        mock.patch.object(self.et, "apply_fx_program_fixes"),
                        mock.patch.object(self.et, "apply_fx_node_fixes"),
                        mock.patch.object(self.et, "to_edge_transform_and_lower"),
                    ):
                        # Exercise both public config forms through real Dynamo normalization.
                        self.et.ExecutorchExporter().export(model, inputs, snapshot if strict else config)
                    kwargs = capture.call_args.kwargs
                    self.assertEqual(kwargs["strict"], strict)
                    self.assertTrue(kwargs["prefer_deferred_runtime_asserts_over_guards"])
                    if dynamic and explicit_shapes is None:
                        self.assertEqual(
                            kwargs["dynamic_shapes"], {"input": {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO}}
                        )
                    else:
                        self.assertEqual(kwargs["dynamic_shapes"], explicit_shapes)
                    self.assertEqual(vars(config), snapshot)

    def test_lowering_optional_arguments_preserve_empty_values(self):
        for method_name, passes, metadata in (
            (None, None, None),
            ("forward", [], {}),
            (None, [], None),
            ("forward", None, {}),
        ):
            with self.subTest(method_name=method_name, passes=passes, metadata=metadata):
                ep, partitioners, compile_config = object(), [object()], object()
                backend_factory = mock.Mock(return_value=mock.sentinel.backend_config)
                settings = self.et._LoweringSettings(
                    partitioners,
                    compile_config,
                    backend_factory,
                    transform_passes=passes,
                    constant_methods=metadata,
                    method_name=method_name,
                )
                with mock.patch.object(self.et, "to_edge_transform_and_lower") as lower:
                    result = self.et._lower_to_executorch(ep, settings)
                expected = {"partitioner": partitioners, "compile_config": compile_config}
                if passes is not None:
                    expected["transform_passes"] = passes
                if metadata is not None:
                    expected["constant_methods"] = metadata
                lower.assert_called_once_with({method_name: ep} if method_name else ep, **expected)
                backend_factory.assert_called_once_with()
                lower.return_value.to_executorch.assert_called_once_with(config=mock.sentinel.backend_config)
                self.assertIs(result, lower.return_value.to_executorch.return_value)

    @require_torch_greater_or_equal("2.11")
    def test_non_mlx_rejects_causal_lm_options_before_preparation(self):
        options = (
            ("cache_mode", "off-graph"),
            ("logits_to_keep", "last"),
            ("logits_to_keep", "selected"),
            ("max_context_len", 2048),
            ("max_seq_len", 128),
            ("max_seq_len", None),
            ("dtype", "fp32"),
            ("dtype", "fp16"),
        )
        for backend, (name, value), as_dict in itertools.product(("xnnpack", "cuda"), options, (False, True)):
            with self.subTest(backend=backend, option=name, value=value, as_dict=as_dict):
                config = ExecutorchConfig(backend=backend, **{name: value})
                model, prepare = nn.Linear(2, 2), mock.Mock()
                with (
                    mock.patch.dict(self.et._BACKEND_PREPARE, {backend: prepare}),
                    mock.patch.object(self.dynamo.DynamoExporter, "export") as capture,
                    mock.patch.object(self.et, "apply_patches") as patches,
                    self.assertRaisesRegex(ValueError, f"{backend!r} does not support causal-LM options:.*{name}"),
                ):
                    self.et.ExecutorchExporter().export(model, {}, config.to_dict() if as_dict else config)
                prepare.assert_not_called()
                capture.assert_not_called()
                patches.assert_not_called()
                self.assertTrue(model.training)
                self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_scopes_and_deferred_factories_restore_after_pipeline_failures(self):
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if "mlx" in name or any("mlx" in item for item in (fromlist or ())):
                raise AssertionError("Non-MLX export attempted an MLX import")
            return original_import(name, globals, locals, fromlist, level)

        stages = ["capture", "program", "nodes", "settings", "edge", "backend", "executorch"]
        for backend in ("xnnpack", "cuda"):
            for failure in (None, *stages):
                with self.subTest(backend=backend, failure=failure):
                    events, state = [], SimpleNamespace(outer=False, capture=False)
                    config = ExecutorchConfig(
                        backend=backend, strict=True, alloc_graph_input=False, alloc_mutable_buffers=False
                    )
                    ep, edge = SimpleNamespace(graph_module=object()), mock.Mock()
                    bucketize, detach, tensor_detach = torch.bucketize, torch.detach, torch.Tensor.detach
                    value = torch.ones(2, requires_grad=True)

                    def stage(name, result=None):
                        self.assertTrue(state.outer)
                        self.assertEqual(state.capture, name == "capture")
                        self.assertIsNot(torch.bucketize, bucketize)
                        self.assertIsNot(torch.detach, detach)
                        self.assertIsNot(torch.Tensor.detach, tensor_detach)
                        self.assertIs(torch.detach(value), value)
                        self.assertIs(value.detach(), value)
                        events.append(name)
                        if name == failure:
                            raise RuntimeError(name)
                        return result

                    backend_factory = mock.Mock(side_effect=lambda: stage("backend", None))
                    settings = self.et._LoweringSettings([], object(), backend_factory)
                    settings_factory = mock.Mock(side_effect=lambda: stage("settings", settings))
                    record = self.et._BackendPreparation(
                        object(),
                        {},
                        config,
                        settings_factory,
                        export_context=patch_attributes([(state, "outer", lambda _: True)]),
                        capture_context=patch_attributes([(state, "capture", lambda _: True)]),
                    )
                    with (
                        mock.patch.dict(self.et._BACKEND_PREPARE, {backend: mock.Mock(return_value=record)}),
                        mock.patch.object(
                            self.dynamo.DynamoExporter, "export", side_effect=lambda *a, **kw: stage("capture", ep)
                        ),
                        mock.patch.object(self.et, "apply_fx_program_fixes", side_effect=lambda *a: stage("program")),
                        mock.patch.object(self.et, "apply_fx_node_fixes", side_effect=lambda *a: stage("nodes")),
                        mock.patch.object(
                            self.et, "to_edge_transform_and_lower", side_effect=lambda *a, **kw: stage("edge", edge)
                        ) as lower,
                        mock.patch.object(
                            edge, "to_executorch", side_effect=lambda **kw: stage("executorch", mock.sentinel.result)
                        ),
                        mock.patch("builtins.__import__", side_effect=guarded_import),
                        self.assertRaisesRegex(RuntimeError, failure) if failure else nullcontext(),
                    ):
                        self.assertIs(self.et.ExecutorchExporter().export(object(), {}, config), mock.sentinel.result)
                    self.assertEqual(events, stages if failure is None else stages[: stages.index(failure) + 1])
                    self.assertFalse(state.outer)
                    self.assertFalse(state.capture)
                    self.assertIs(torch.bucketize, bucketize)
                    self.assertIs(torch.detach, detach)
                    self.assertIs(torch.Tensor.detach, tensor_detach)
                    self.assertFalse(torch.detach(value).requires_grad)
                    self.assertFalse(value.detach().requires_grad)
                    if "edge" in events:
                        self.assertIs(lower.call_args.args[0], ep)
                    else:
                        lower.assert_not_called()
                    if "settings" in events:
                        settings_factory.assert_called_once_with()
                    else:
                        settings_factory.assert_not_called()
                    if "backend" in events:
                        backend_factory.assert_called_once_with()
                    else:
                        backend_factory.assert_not_called()


class _Owner:
    def method(self):
        return "original"


@require_torch
class PatchRegistryEdgeCasesTest(unittest.TestCase):
    def test_patch_attributes_roll_back_on_exception(self):
        # Real exports never exit the trace via exception, so this rollback path is untested by
        # integration. If it ever regressed to leave already-installed patches in place when a
        # later factory raises, the *next* export would run against a leaked patch and fail in
        # a way that looks unrelated. Only this test would catch that.
        a, b = _Owner(), _Owner()

        def _bad_factory(original):
            raise RuntimeError("factory boom")

        with self.assertRaisesRegex(RuntimeError, "factory boom"):
            with patch_attributes(
                [
                    (a, "method", lambda original: (lambda: "a-patched")),
                    (b, "method", _bad_factory),
                ]
            ):
                pass
        self.assertEqual(a.method(), "original")
        self.assertEqual(b.method(), "original")

    def test_register_patch_skips_unresolvable_path(self):
        # Real backends only register paths that resolve; the silent-skip fallback is what lets
        # `exporter_onnx.py` and `exporter_executorch.py` co-exist when only one backend is
        # installed. If it ever started raising, one of the two backends would fail to import.
        backend = "_test_unresolvable"

        @register_patch(backend, "does.not.exist.at.all")
        def _patch(original):
            return original

        try:
            self.assertEqual(exporter_utils._PATCHES.get(backend, []), [])
        finally:
            exporter_utils._PATCHES.pop(backend, None)


# ─────────────────────────────────────────────────────────────────────────────
# Leaf-tensor invariants that integration tests wouldn't visibly catch
# ─────────────────────────────────────────────────────────────────────────────


@require_torch
class LeafTensorInvariantsTest(unittest.TestCase):
    def test_duplicate_leaf_tensors_only_clones_repeats(self):
        # If this ever regressed to ``.clone()``-everything, ONNX exports would still succeed
        # and just get a bit bigger — no integration test would notice. Similarly, if it
        # stopped cloning the second occurrence, ONNX's output-node dedup would rename ports
        # in a way that only manifests as a stale name mapping.
        shared = torch.zeros(2)
        distinct = torch.ones(3)
        result = duplicate_leaf_tensors({"a": shared, "b": shared, "c": distinct})
        self.assertIs(result["a"], shared)
        self.assertIsNot(result["b"], shared)
        self.assertTrue(torch.equal(result["b"], shared))
        self.assertIs(result["c"], distinct)

    def test_cast_leaf_tensors_preserves_integer_dtypes(self):
        # ``prepare_for_export`` casts input trees to the model's dtype. If this ever started
        # coercing integer tensors (``input_ids``, indices, positions) to float, most exports
        # would still trace but embedding-lookup / bincount / index-select paths would fail
        # far downstream with confusing errors. Only this test would attribute it to the cast.
        out = cast_leaf_tensors(
            {
                "input_ids": torch.zeros(2, dtype=torch.int64),
                "attention_mask": torch.ones(2, dtype=torch.int32),
                "hidden": torch.zeros(2, dtype=torch.float32),
            },
            dtype=torch.float16,
            device=torch.device("cpu"),
        )
        self.assertEqual(out["input_ids"].dtype, torch.int64)
        self.assertEqual(out["attention_mask"].dtype, torch.int32)
        self.assertEqual(out["hidden"].dtype, torch.float16)


# ─────────────────────────────────────────────────────────────────────────────
# decompose_prefill_decode guard (dead code without this test — no real generator
# calls forward < 2 times, so the branch would rot silently)
# ─────────────────────────────────────────────────────────────────────────────


@require_torch
class DecomposePrefillDecodeGuardTest(unittest.TestCase):
    def test_raises_when_generate_bypasses_forward(self):
        # Guards against generators that delegate to an inner model — the top-level ``forward``
        # captures at most one call, so the ``calls[0] / calls[1]`` indexing would raise a
        # confusing IndexError instead of the helpful RuntimeError below.
        class _FakeGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)
                # `decompose_prefill_decode` bases its capture config on the model's own (mimics a
                # real `PreTrainedModel`); the guard under test fires afterwards on the capture count.
                self.generation_config = GenerationConfig()

            def forward(self, input_ids=None, **kwargs):
                return input_ids

            def generate(self, input_ids=None, max_new_tokens=None, min_new_tokens=None, **kwargs):
                return self.forward(input_ids=input_ids)  # a single top-level forward call

        with self.assertRaisesRegex(RuntimeError, "captured 1"):
            decompose_prefill_decode(_FakeGenerator(), {"input_ids": torch.zeros(1, 1, dtype=torch.long)})
