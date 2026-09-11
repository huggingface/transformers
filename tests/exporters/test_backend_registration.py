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
"""Backend-neutral contract tests for external ExecuTorch recipes."""

import subprocess
import sys
import textwrap
import threading
import unittest
from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest import mock

from transformers.exporters import (
    ExecutorchAttention,
    ExecutorchBackendPreparation,
    ExecutorchBackendRecipe,
    ExecutorchCapture,
    ExecutorchCompatibilityPolicy,
    ExecutorchConfig,
    ExecutorchExporter,
    ExecutorchExportPatch,
    register_executorch_backend,
    scoped_executorch_attention,
    utils,
)
from transformers.exporters import exporter_executorch as et
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    from torch import nn

    class _Identity(nn.Module):
        def forward(self, x):
            return x + 1

    class _AttentionModel(_Identity):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_attn_implementation_internal="original")
            self.setter_calls = 0

        def set_attn_implementation(self, implementation):
            self.setter_calls += 1
            self.config._attn_implementation_internal = implementation
            self.config._attn_was_changed = True


_PATCH_OWNER = SimpleNamespace(explicit="original", common="original", backend="original")
_PATCH_ALIAS = _PATCH_OWNER
if is_torch_available():
    _TORCH_ALIAS = torch


class BackendRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.dict(et._EXECUTORCH_BACKEND_RECIPES, {}, clear=True))

    def test_registration_is_public_idempotent_and_requires_explicit_overwrite(self):
        self.assertIs(register_executorch_backend, et.register_executorch_backend)
        first, second = mock.Mock(), mock.Mock()
        register_executorch_backend("test", first)
        register_executorch_backend("test", first)
        self.assertIs(et._EXECUTORCH_BACKEND_RECIPES["test"], first)
        with self.assertRaisesRegex(ValueError, "overwrite=True"):
            register_executorch_backend("test", second)
        self.assertIs(et._EXECUTORCH_BACKEND_RECIPES["test"], first)
        register_executorch_backend("test", second, overwrite=True)
        self.assertIs(et._EXECUTORCH_BACKEND_RECIPES["test"], second)

    def test_registration_rejects_invalid_arguments_and_builtin_names(self):
        for name in (None, 1, [], "", "   "):
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                register_executorch_backend(name, mock.Mock())
        with self.assertRaisesRegex(TypeError, "must be callable"):
            register_executorch_backend("test", object())
        for backend in ("xnnpack", "cuda"):
            for overwrite in (False, True):
                with (
                    self.subTest(backend=backend, overwrite=overwrite),
                    self.assertRaisesRegex(ValueError, "built in"),
                ):
                    register_executorch_backend(backend, mock.Mock(), overwrite=overwrite)

    def test_backend_options_roundtrip_and_default_isolation(self):
        first = ExecutorchConfig(backend="test", backend_options={"mode": "fast"})
        restored = ExecutorchConfig.from_dict(first.to_dict())
        self.assertEqual(restored.backend_options, {"mode": "fast"})
        restored.backend_options["mode"] = "slow"
        self.assertEqual(first.backend_options, {"mode": "fast"})
        self.assertEqual(ExecutorchConfig().backend_options, {})


@require_torch
class _BackendTestCase(unittest.TestCase):
    def setUp(self):
        # These contract tests need torch, but do not invoke ExecuTorch lowering.
        self.exporter = object.__new__(ExecutorchExporter)
        self.model, self.inputs = _Identity(), {}
        self.config = ExecutorchConfig(backend="test")
        self.prepared = ExecutorchBackendPreparation(_Identity(), {"x": torch.ones(2)}, normalize_inputs=False)
        self.program = mock.Mock(spec=torch.export.ExportedProgram)
        self.enterContext(mock.patch.dict(et._EXECUTORCH_BACKEND_RECIPES, {}, clear=True))
        self.enterContext(mock.patch.dict(utils._PATCHES, {}, clear=True))
        self.enterContext(mock.patch.dict(utils._FX_PROGRAM_FIXES, {}, clear=True))
        self.enterContext(mock.patch.dict(utils._FX_NODE_FIXES, {}, clear=True))
        self.program_fixes = self.enterContext(mock.patch.object(et, "apply_fx_program_fixes"))
        self.node_fixes = self.enterContext(mock.patch.object(et, "apply_fx_node_fixes"))

    def _register(self, *, prepared=None, **kwargs):
        """Register an ordinary mock-capture recipe; integration tests register theirs directly."""
        kwargs.setdefault("prepare", mock.Mock(return_value=self.prepared if prepared is None else prepared))
        kwargs.setdefault("lower", mock.Mock(return_value="artifact"))
        recipe = ExecutorchBackendRecipe(**kwargs)
        self.factory = mock.Mock(return_value=recipe)
        register_executorch_backend("test", self.factory, overwrite=True)
        self.trace = self.enterContext(mock.patch.object(self.exporter, "_export_prepared", return_value=self.program))
        return recipe


class BackendLifecycleTest(_BackendTestCase):
    def test_immediate_lifecycle_preserves_order_arguments_and_capture_only_contexts(self):
        steps = mock.Mock()
        scope = mock.MagicMock()
        transformed = mock.Mock(spec=torch.export.ExportedProgram)
        self.prepared.dynamic_shapes = {"x": None}
        self.prepared.capture_contexts = (scope,)
        self.config.backend_options = {"mode": "fast"}
        steps.prepare.return_value = self.prepared
        steps.transform.return_value = transformed
        steps.lower.return_value = "artifact"
        self._register(prepare=steps.prepare, lower=steps.lower, transform_exported_program=steps.transform)
        steps.attach_mock(scope, "context")
        steps.attach_mock(self.trace, "capture")
        steps.attach_mock(self.program_fixes, "program_fixes")
        steps.attach_mock(self.node_fixes, "node_fixes")

        self.assertEqual(self.exporter.export(self.model, self.inputs, self.config), "artifact")
        self.factory.assert_called_once_with({"mode": "fast"})
        self.assertIs(steps.prepare.call_args.args[1], self.inputs)
        self.assertIs(self.trace.call_args.args[1], self.prepared.sample_inputs)
        self.assertEqual(
            steps.mock_calls,
            [
                mock.call.prepare(self.model, self.inputs, self.config),
                mock.call.context.__enter__(),
                mock.call.capture(
                    self.prepared.model,
                    self.prepared.sample_inputs,
                    self.config,
                    output_flags={},
                    dynamic_shapes={"x": None},
                    patch_exclusions=(),
                ),
                mock.call.context.__exit__(None, None, None),
                mock.call.transform(self.program, self.prepared),
                mock.call.program_fixes("executorch", transformed),
                mock.call.node_fixes("executorch", transformed.graph_module),
                mock.call.lower(transformed, self.prepared, self.config),
            ],
        )

    def test_common_policy_is_independent_of_recipe_and_backend_patches(self):
        utils._PATCHES["executorch"] = [(_PATCH_OWNER, "common", lambda _: "patched")]
        utils._PATCHES["executorch.test"] = [(_PATCH_OWNER, "backend", lambda _: "patched")]

        def prepare(*_):
            self.assertEqual(vars(_PATCH_OWNER), {"explicit": "patched", "common": "original", "backend": "original"})
            return self.prepared

        def check_patches(*_args, **_kwargs):
            self.assertEqual(
                vars(_PATCH_OWNER),
                {"explicit": "patched", "common": "patched" if common_patches else "original", "backend": "patched"},
            )
            return self.program

        for common_patches, common_fixes in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(patches=common_patches, fixes=common_fixes):
                patch_factory = mock.Mock(return_value="patched")
                self._register(
                    prepare=prepare,
                    lower=check_patches,
                    patches=(ExecutorchExportPatch((f"{__name__}._PATCH_OWNER.explicit",), patch_factory),),
                    compatibility=ExecutorchCompatibilityPolicy(common_patches, common_fixes),
                )
                self.trace.side_effect = check_patches
                self.program_fixes.reset_mock()
                self.node_fixes.reset_mock()
                self.assertIs(self.exporter.export(self.model, self.inputs, self.config), self.program)
                patch_factory.assert_called_once_with("original")
                self.assertEqual(self.program_fixes.call_count, int(common_fixes))
                self.assertEqual(self.node_fixes.call_count, int(common_fixes))
                self.assertEqual(vars(_PATCH_OWNER), dict.fromkeys(("explicit", "common", "backend"), "original"))

    def test_deferred_lower_retains_config_recipe_and_attention_without_recapture(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        model, prepared_model = _AttentionModel(), _AttentionModel()
        self.prepared.model = prepared_model
        self.config.backend_options = {"nested": {"value": 1}}
        scope = mock.MagicMock()
        self.prepared.capture_contexts = (scope,)
        attention = ExecutorchAttention("test_scope_attention", lambda: None, None)
        patch_factory = mock.Mock(return_value="patched")
        transform = mock.Mock(return_value=mock.Mock(spec=torch.export.ExportedProgram))

        def prepare(*_):
            model.config._attn_implementation_internal = "prepared-selection"
            prepared_model.config._attn_implementation_internal = "new-model-selection"
            return self.prepared

        def lower(program, prepared, config):
            self.assertEqual(config.backend_options, {"nested": {"value": 1}})
            self.assertEqual(model.config._attn_implementation_internal, "prepared-selection")
            self.assertEqual(prepared_model.config._attn_implementation_internal, "new-model-selection")
            self.assertIs(ALL_ATTENTION_FUNCTIONS[attention.implementation], attention.attention_function)
            self.assertEqual(_PATCH_OWNER.explicit, "patched")
            return "lowered"

        recipe = self._register(
            prepare=mock.Mock(side_effect=prepare),
            lower=mock.Mock(side_effect=lower),
            attention=attention,
            patches=(ExecutorchExportPatch((f"{__name__}._PATCH_OWNER.explicit",), patch_factory),),
            transform_exported_program=transform,
        )
        captured = self.exporter.capture(model, self.inputs, self.config)
        self.assertIsInstance(captured, ExecutorchCapture)
        self.assertIs(captured.preparation, self.prepared)
        self.assertIs(captured.exported_program, transform.return_value)
        recipe.lower.assert_not_called()
        self.assertEqual(model.config._attn_implementation_internal, "prepared-selection")
        self.assertEqual(_PATCH_OWNER.explicit, "original")
        self.config.backend_options["nested"]["value"] = 2
        replacement = mock.Mock(side_effect=AssertionError("must use captured recipe"))
        register_executorch_backend("test", replacement, overwrite=True)
        model.config._attn_implementation_internal = "between-phases"
        prepared_model.config._attn_implementation_internal = "between-prepared-phases"
        self.assertEqual(self.exporter.lower(captured), "lowered")
        with self.assertRaisesRegex(RuntimeError, "one lowering attempt"):
            self.exporter.lower(captured)
        recipe.lower.assert_called_once_with(transform.return_value, self.prepared, captured.config)
        recipe.prepare.assert_called_once()
        self.trace.assert_called_once()
        self.factory.assert_called_once()
        replacement.assert_not_called()
        transform.assert_called_once_with(self.program, self.prepared)
        self.program_fixes.assert_called_once_with("executorch", transform.return_value)
        self.node_fixes.assert_called_once_with("executorch", transform.return_value.graph_module)
        self.assertEqual(scope.mock_calls, [mock.call.__enter__(), mock.call.__exit__(None, None, None)])
        self.assertEqual(patch_factory.call_args_list, [mock.call("original"), mock.call("original")])
        self.assertEqual(model.setter_calls, 0)
        self.assertEqual(prepared_model.setter_calls, 0)
        self.assertFalse(hasattr(model.config, "_attn_was_changed"))
        self.assertEqual(model.config._attn_implementation_internal, "between-phases")
        self.assertEqual(prepared_model.config._attn_implementation_internal, "between-prepared-phases")
        self.assertEqual(_PATCH_OWNER.explicit, "original")

    def test_prepared_inputs_preserve_mixed_dtypes_and_output_named_kwargs(self):
        class MixedInputs(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(2, dtype=torch.float32))

            def forward(self, x, y, return_dict):
                return x + self.weight, y + 1 if return_dict else y

        inputs = {
            "x": torch.ones(2, dtype=torch.float32),
            "y": torch.ones(2, dtype=torch.float64),
            "return_dict": True,
        }
        prepared = ExecutorchBackendPreparation(model=MixedInputs(), sample_inputs=inputs, normalize_inputs=False)
        recipe = ExecutorchBackendRecipe(
            prepare=mock.Mock(return_value=prepared),
            lower=mock.Mock(),
            compatibility=ExecutorchCompatibilityPolicy(False, False),
        )
        register_executorch_backend("test", mock.Mock(return_value=recipe))
        with mock.patch.object(et, "prepare_for_export", side_effect=AssertionError("must not normalize")):
            captured = self.exporter.capture(self.model, self.inputs, ExecutorchConfig(backend="test", strict=True))
        self.assertIs(captured.preparation.sample_inputs, inputs)
        result = captured.exported_program.module()(**inputs)
        self.assertEqual(result[0].dtype, torch.float32)
        self.assertEqual(result[1].dtype, torch.float64)
        self.assertIn("return_dict", inputs)
        torch.testing.assert_close(result[1], torch.full((2,), 2.0, dtype=torch.float64))

    def test_default_preparation_still_normalizes_inputs(self):
        model = nn.Linear(2, 2).float()
        prepared = ExecutorchBackendPreparation(model=model, sample_inputs={"input": torch.ones(2).double()})
        self._register(prepared=prepared)
        self.exporter.capture(self.model, self.inputs, self.config)
        self.assertEqual(self.trace.call_args.args[1]["input"].dtype, torch.float32)

    def test_exports_without_attention_are_serialized_before_patch_installation(self):
        attempted, second_patched = threading.Event(), threading.Event()
        errors = []

        def prepare_first(*_):
            thread.start()
            self.assertTrue(attempted.wait(5))
            self.assertFalse(second_patched.wait(0.1))
            self.assertEqual(_PATCH_OWNER.explicit, "first")
            return self.prepared

        def second_factory(_):
            second_patched.set()
            return "second"

        first = self._register(
            prepare=prepare_first,
            patches=(ExecutorchExportPatch((f"{__name__}._PATCH_OWNER.explicit",), lambda _: "first"),),
        )
        second = replace(
            first,
            prepare=mock.Mock(return_value=self.prepared),
            patches=(ExecutorchExportPatch((f"{__name__}._PATCH_OWNER.explicit",), second_factory),),
        )
        register_executorch_backend("second", mock.Mock(return_value=second))

        def run_second():
            attempted.set()
            try:
                self.exporter.export(self.model, self.inputs, ExecutorchConfig(backend="second"))
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=run_second, daemon=True)
        try:
            self.exporter.export(self.model, self.inputs, self.config)
        finally:
            if thread.ident is not None:
                thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(second_patched.is_set())
        self.assertEqual(_PATCH_OWNER.explicit, "original")

    def test_invalid_transform_result_is_rejected_before_lowering(self):
        recipe = self._register(transform_exported_program=mock.Mock(return_value=object()))
        with self.assertRaisesRegex(TypeError, "ExportedProgram"):
            self.exporter.export(self.model, self.inputs, self.config)
        recipe.lower.assert_not_called()

    def test_phase_failures_restore_recipe_patches_and_attention(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        model = _AttentionModel()
        attention = ExecutorchAttention("test_failed_phase_attention", lambda: None, None)
        self.prepared.model = self.prepared.attention_target = model
        for phase in ("prepare", "context", "capture", "transform", "fixes"):
            with self.subTest(phase=phase):
                scope = mock.MagicMock()
                self.prepared.capture_contexts = (scope,)
                recipe = self._register(
                    attention=attention,
                    transform_exported_program=mock.Mock(return_value=self.program),
                    patches=(ExecutorchExportPatch((f"{__name__}._PATCH_OWNER.explicit",), lambda _: "patched"),),
                )
                failing = {
                    "prepare": recipe.prepare,
                    "context": scope.__enter__,
                    "capture": self.trace,
                    "transform": recipe.transform_exported_program,
                    "fixes": self.program_fixes,
                }[phase]
                with (
                    mock.patch.object(failing, "side_effect", RuntimeError(f"failed {phase}")),
                    self.assertRaisesRegex(RuntimeError, f"failed {phase}"),
                ):
                    self.exporter.export(model, self.inputs, self.config)
                recipe.lower.assert_not_called()
                self.assertEqual(_PATCH_OWNER.explicit, "original")
                self.assertEqual(model.config._attn_implementation_internal, "original")
                self.assertNotIn(attention.implementation, ALL_ATTENTION_FUNCTIONS)


class BackendValidationTest(_BackendTestCase):
    def test_invalid_recipe_members_fail_before_preparation(self):
        valid = self._register()
        for recipe in (
            object(),
            replace(valid, prepare=None),
            replace(valid, lower=None),
            replace(valid, transform_exported_program=object()),
            replace(valid, compatibility=None),
            replace(valid, patches=(ExecutorchExportPatch(("torch.add",), None),)),
            replace(valid, attention=ExecutorchAttention("test", None, None)),
        ):
            with self.subTest(recipe=recipe), self.assertRaises(TypeError):
                self.factory.return_value = recipe
                self.exporter.capture(self.model, self.inputs, self.config)
        valid.prepare.assert_not_called()

    def test_unknown_backend_lists_available_backends(self):
        with self.assertRaisesRegex(ValueError, "available backends.*cuda.*xnnpack"):
            self.exporter.export(self.model, self.inputs, ExecutorchConfig(backend="missing"))

    def test_invalid_preparation_contract_is_rejected(self):
        for prepared in (
            object(),
            ExecutorchBackendPreparation(model=object(), sample_inputs={}),
            ExecutorchBackendPreparation(model=_Identity(), sample_inputs=[]),
        ):
            with self.subTest(prepared=type(prepared)):
                recipe = self._register(prepared=prepared)
                with self.assertRaises(TypeError):
                    self.exporter.capture(self.model, self.inputs, self.config)
                recipe.lower.assert_not_called()


class BackendAttentionTest(_BackendTestCase):
    def test_public_scope_restores_registries_on_body_or_registration_failure(self):
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        attention = ExecutorchAttention("test_public_attention", lambda: None, lambda: None)
        for registry in (ALL_ATTENTION_FUNCTIONS, ALL_MASK_ATTENTION_FUNCTIONS):
            self.enterContext(mock.patch.dict(registry._global_mapping))
        for preexisting in (False, True):
            for failure in ("body", "registration"):
                with self.subTest(preexisting=preexisting, failure=failure):
                    prior = mock.Mock(return_value="prior")
                    if preexisting:
                        for registry in (ALL_ATTENTION_FUNCTIONS, ALL_MASK_ATTENTION_FUNCTIONS):
                            registry.register(attention.implementation, prior)
                    register = ALL_MASK_ATTENTION_FUNCTIONS.register
                    if failure == "registration":
                        register = mock.Mock(side_effect=RuntimeError("registration failed"))
                    with (
                        mock.patch.object(ALL_MASK_ATTENTION_FUNCTIONS, "register", register),
                        self.assertRaisesRegex(RuntimeError, f"{failure} failed"),
                        scoped_executorch_attention(attention),
                    ):
                        self.assertIs(ALL_ATTENTION_FUNCTIONS[attention.implementation], attention.attention_function)
                        self.assertIs(ALL_MASK_ATTENTION_FUNCTIONS[attention.implementation], attention.mask_function)
                        raise RuntimeError("body failed")
                    for registry in (ALL_ATTENTION_FUNCTIONS, ALL_MASK_ATTENTION_FUNCTIONS):
                        if preexisting:
                            self.assertIs(registry[attention.implementation], prior)
                        else:
                            self.assertNotIn(attention.implementation, registry)

    def test_public_scope_restores_module_and_config_only_children(self):
        for kind in ("module", "config"):
            for fail in (False, True):
                with self.subTest(kind=kind, fail=fail):
                    model, child = _AttentionModel(), _AttentionModel()
                    child.config._attn_implementation_internal = "child"
                    child.config._attn_was_changed = False
                    if kind == "module":
                        model.child = child
                    else:
                        model.config.sub_configs = {"child": object}
                        model.config.child = child.config

                    def select(implementation):
                        for config in (model.config, child.config):
                            config._attn_implementation_internal = implementation
                            config._attn_was_changed = True
                        if fail:
                            raise RuntimeError("setter failed")

                    attention = ExecutorchAttention("test_child_attention", lambda: None, None)
                    outcome = self.assertRaisesRegex(RuntimeError, "setter failed") if fail else nullcontext()
                    with (
                        mock.patch.object(model, "set_attn_implementation", side_effect=select),
                        outcome,
                        scoped_executorch_attention(attention, model),
                    ):
                        self.assertFalse(fail, "scope entered after setter failure")
                        self.assertEqual(child.config._attn_implementation_internal, attention.implementation)
                    self.assertEqual(model.config._attn_implementation_internal, "original")
                    self.assertEqual(child.config._attn_implementation_internal, "child")
                    self.assertFalse(child.config._attn_was_changed)
                    self.assertFalse(hasattr(model.config, "_attn_was_changed"))

    def test_public_scope_serializes_registry_ownership(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        first = ExecutorchAttention("test_concurrent_attention", lambda: "first", None)
        second = ExecutorchAttention(first.implementation, lambda: "second", None)
        attempted, entered = threading.Event(), threading.Event()
        errors = []

        def worker():
            attempted.set()
            try:
                with scoped_executorch_attention(second):
                    self.assertIs(ALL_ATTENTION_FUNCTIONS[first.implementation], second.attention_function)
                    entered.set()
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=worker, daemon=True)
        try:
            with scoped_executorch_attention(first):
                thread.start()
                self.assertTrue(attempted.wait(5))
                self.assertFalse(entered.wait(0.1))
                self.assertIs(ALL_ATTENTION_FUNCTIONS[first.implementation], first.attention_function)
        finally:
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(entered.is_set())
        self.assertNotIn(first.implementation, ALL_ATTENTION_FUNCTIONS)

    def test_explicit_attention_targets_are_selected_after_prepare_and_restored(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        for kind in ("original", "wrapped", "replacement", "external"):
            with self.subTest(kind=kind):
                original = _AttentionModel()
                target = original if kind in ("original", "wrapped") else _AttentionModel()
                prepared_model = nn.Sequential(target) if kind == "wrapped" else target
                if kind == "external":
                    prepared_model = _Identity()
                attention = ExecutorchAttention("test_explicit_target", lambda: None, None)

                def prepare(*_):
                    self.assertIs(ALL_ATTENTION_FUNCTIONS[attention.implementation], attention.attention_function)
                    self.assertEqual(target.setter_calls, 0)
                    self.assertEqual(target.config._attn_implementation_internal, "original")
                    return ExecutorchBackendPreparation(
                        model=prepared_model, sample_inputs={"x": torch.ones(2)}, attention_target=target
                    )

                def normalize(model, inputs):
                    self.assertEqual(target.config._attn_implementation_internal, attention.implementation)
                    return model, inputs, {}

                def lower(*_):
                    self.assertEqual(target.config._attn_implementation_internal, attention.implementation)
                    self.assertTrue(target.config._attn_was_changed)
                    self.assertEqual(target.setter_calls, 1)
                    return "lowered"

                self._register(prepare=prepare, lower=lower, attention=attention)
                with mock.patch.object(et, "prepare_for_export", side_effect=normalize):
                    captured = self.exporter.capture(original, self.inputs, self.config)
                self.assertEqual(target.config._attn_implementation_internal, "original")
                self.assertFalse(hasattr(target.config, "_attn_was_changed"))
                target.config._attn_implementation_internal = "between"
                self.assertEqual(self.exporter.lower(captured), "lowered")
                self.assertEqual(target.config._attn_implementation_internal, "between")
                self.assertFalse(hasattr(target.config, "_attn_was_changed"))
                self.assertEqual(target.setter_calls, 1)

    def test_attention_target_validation_and_setter_failure(self):
        class FailingTarget(_AttentionModel):
            def set_attn_implementation(self, implementation):
                super().set_attn_implementation(implementation)
                raise RuntimeError("partial setter")

        attention = ExecutorchAttention("test_target_validation", lambda: None, None)
        for target, selected_attention, error in (
            (_AttentionModel(), None, ValueError),
            (_Identity(), attention, TypeError),
            (object(), attention, TypeError),
            (FailingTarget(), attention, RuntimeError),
        ):
            with self.subTest(target=type(target), attention=selected_attention):
                self.prepared.attention_target = target
                recipe = self._register(attention=selected_attention)
                with self.assertRaises(error):
                    self.exporter.capture(self.model, self.inputs, self.config)
                recipe.lower.assert_not_called()
                if isinstance(target, _AttentionModel):
                    self.assertEqual(target.config._attn_implementation_internal, "original")
                    self.assertFalse(hasattr(target.config, "_attn_was_changed"))

    def test_capture_retains_selected_target_when_transform_replaces_preparation_target(self):
        target = _AttentionModel()
        attention = ExecutorchAttention("test_saved_target", lambda: None, None)

        def transform(program, prepared):
            prepared.attention_target = None
            return program

        def lower(*_):
            self.assertEqual(target.config._attn_implementation_internal, attention.implementation)
            self.assertEqual(target.setter_calls, 1)

        self.prepared.attention_target = target
        self._register(lower=lower, attention=attention, transform_exported_program=transform)
        captured = self.exporter.capture(self.model, self.inputs, self.config)
        self.assertEqual(target.config._attn_implementation_internal, "original")
        self.exporter.lower(captured)
        self.assertEqual(target.config._attn_implementation_internal, "original")
        self.assertEqual(target.setter_calls, 1)

    def test_failed_lower_consumes_capture_and_restores_attention_and_patches(self):
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        target = _AttentionModel()
        attention = ExecutorchAttention("test_failed_lower_attention", lambda: None, None)

        def lower(*_):
            self.assertEqual(target.config._attn_implementation_internal, attention.implementation)
            self.assertEqual(target.setter_calls, 1)
            self.assertEqual(_PATCH_OWNER.explicit, "patched")
            raise RuntimeError("lower failed")

        self.prepared.attention_target = target
        recipe = self._register(
            lower=mock.Mock(side_effect=lower),
            attention=attention,
            patches=(ExecutorchExportPatch((f"{__name__}._PATCH_OWNER.explicit",), lambda _: "patched"),),
        )
        captured = self.exporter.capture(self.model, self.inputs, self.config)
        target.config._attn_implementation_internal = "between"
        with self.assertRaisesRegex(RuntimeError, "lower failed"):
            self.exporter.lower(captured)
        self.assertEqual(target.config._attn_implementation_internal, "between")
        self.assertEqual(target.setter_calls, 1)
        self.assertFalse(hasattr(target.config, "_attn_was_changed"))
        self.assertNotIn(attention.implementation, ALL_ATTENTION_FUNCTIONS)
        self.assertNotIn(attention.implementation, ALL_MASK_ATTENTION_FUNCTIONS)
        with self.assertRaisesRegex(RuntimeError, "one lowering attempt"):
            self.exporter.lower(captured)
        recipe.lower.assert_called_once()
        self.assertEqual(_PATCH_OWNER.explicit, "original")

    def test_preparation_can_temporarily_select_attention_with_public_helper(self):
        target = _AttentionModel()
        attention = ExecutorchAttention("test_prepare_selection", lambda: None, None)

        def prepare(*_):
            self.assertEqual(target.config._attn_implementation_internal, "original")
            with scoped_executorch_attention(attention, target):
                self.assertEqual(target.config._attn_implementation_internal, attention.implementation)
            self.assertEqual(target.config._attn_implementation_internal, "original")
            return self.prepared

        self.prepared.model = target
        self._register(prepare=prepare, attention=attention)
        captured = self.exporter.capture(target, self.inputs, self.config)
        self.exporter.lower(captured)
        self.assertEqual(target.setter_calls, 1)
        self.assertEqual(target.config._attn_implementation_internal, "original")

    def test_attention_requires_explicit_mask_policy(self):
        with self.assertRaises(TypeError):
            ExecutorchAttention("missing_mask", lambda: None)

    def test_none_mask_removes_stale_mapping_and_preserves_supplied_4d_mask(self):
        from transformers import LlamaConfig
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, create_causal_mask
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        name = "test_none_mask"
        old_attention, old_mask = lambda: None, mock.Mock(side_effect=AssertionError("stale mask ran"))
        self.enterContext(mock.patch.dict(ALL_MASK_ATTENTION_FUNCTIONS._global_mapping, {name: old_mask}))
        self.enterContext(mock.patch.dict(ALL_ATTENTION_FUNCTIONS._global_mapping, {name: old_attention}))
        config = LlamaConfig()
        config._attn_implementation = name
        embeds = torch.ones(1, 3, 8)
        supplied = torch.zeros(1, 1, 3, 3)
        for fail in (False, True):
            with self.subTest(fail=fail):
                outcome = self.assertRaisesRegex(RuntimeError, "body failure") if fail else nullcontext()
                with outcome, scoped_executorch_attention(ExecutorchAttention(name, lambda: None, None)):
                    self.assertNotIn(name, ALL_MASK_ATTENTION_FUNCTIONS)
                    self.assertIsNone(create_causal_mask(config, embeds, torch.ones(1, 3), None))
                    self.assertIs(create_causal_mask(config, embeds, supplied, None), supplied)
                    if fail:
                        raise RuntimeError("body failure")
                self.assertIs(ALL_MASK_ATTENTION_FUNCTIONS[name], old_mask)
                self.assertIs(ALL_ATTENTION_FUNCTIONS[name], old_attention)
        old_mask.assert_not_called()

    def test_real_hf_attention_and_mask_capture_preserves_causal_and_padding_numerics(self):
        self._check_real_hf_attention_and_mask_capture(strict=False)

    def test_strict_real_hf_attention_and_mask_capture_preserves_causal_and_padding_numerics(self):
        self._check_real_hf_attention_and_mask_capture(strict=True)

    def _check_real_hf_attention_and_mask_capture(self, *, strict):
        from transformers import LlamaConfig, LlamaForCausalLM
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask, sdpa_mask
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.llama.modeling_llama import eager_attention_forward

        torch.manual_seed(31)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=24,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=16,
                attention_dropout=0.0,
                use_cache=False,
                attn_implementation="eager",
            )
        ).eval()
        model.requires_grad_(False)

        class Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.inner = model

            def forward(self, input_ids, attention_mask):
                return self.inner(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

        wrapper = Wrapper(model)
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.tensor([[1, 0, 1, 1]])}
        expected = wrapper(**inputs)
        calls = {"attention": 0, "mask": 0}

        def attention_function(module, query, key, value, attention_mask, **kwargs):
            if not strict:
                calls["attention"] += 1
            assert attention_mask is not None
            assert attention_mask.ndim == 4
            return eager_attention_forward(module, query, key, value, attention_mask, **kwargs)

        def mask_function(dtype=torch.float32, **kwargs):
            if not strict:
                calls["mask"] += 1
                return eager_mask(dtype=dtype, **kwargs)
            # Stock eager_mask constructs torch.tensor(0.0). Strict capture through the
            # existing forward-signature closure gives it an invalid lifted-constant path.
            # Keep HF's causal/padding builder, but construct the additive mask without
            # a lifted literal. This is a recipe mask implementation, not a global patch.
            kwargs["allow_is_causal_skip"] = False
            kwargs["allow_is_bidirectional_skip"] = False
            mask = sdpa_mask(**kwargs)
            return torch.zeros_like(mask, dtype=dtype).masked_fill(~mask, torch.finfo(dtype).min)

        if strict:
            for padding in (inputs["attention_mask"], torch.ones(1, 4, dtype=torch.long)):
                kwargs = {"batch_size": 1, "q_length": 4, "kv_length": 4, "attention_mask": padding.bool()}
                torch.testing.assert_close(mask_function(**kwargs), eager_mask(**kwargs), rtol=0, atol=0)

        attention = ExecutorchAttention("test_real_eager", attention_function, mask_function)
        recipe = ExecutorchBackendRecipe(
            prepare=mock.Mock(return_value=ExecutorchBackendPreparation(wrapper, inputs, attention_target=model)),
            lower=mock.Mock(),
            attention=attention,
            compatibility=ExecutorchCompatibilityPolicy(False, False),
        )
        register_executorch_backend("test", mock.Mock(return_value=recipe))
        captured = self.exporter.capture(wrapper, inputs, ExecutorchConfig(backend="test", strict=strict))
        if strict:
            traces = [node.meta.get("stack_trace", "") for node in captured.exported_program.graph.nodes]
            self.assertTrue(any("in attention_function" in trace for trace in traces))
            self.assertTrue(any("in mask_function" in trace for trace in traces))
        else:
            self.assertGreater(calls["attention"], 0)
            self.assertGreater(calls["mask"], 0)
        exported = captured.exported_program.module()
        torch.testing.assert_close(exported(**inputs), expected)
        for index in (1, 3):
            changed = {**inputs, "input_ids": inputs["input_ids"].clone()}
            changed["input_ids"][0, index] = 7
            actual = exported(**changed)
            torch.testing.assert_close(actual, wrapper(**changed))
            # A padded token cannot affect later tokens; a future token cannot affect earlier ones.
            positions = slice(2, None) if index == 1 else slice(None, 3)
            torch.testing.assert_close(actual[:, positions], expected[:, positions])
        unpadded = {**inputs, "attention_mask": torch.ones_like(inputs["attention_mask"])}
        torch.testing.assert_close(exported(**unpadded), wrapper(**unpadded))
        self.assertFalse(torch.allclose(exported(**unpadded)[:, 2:], expected[:, 2:]))
        self.assertEqual(model.config._attn_implementation, "eager")
        self.assertNotIn(attention.implementation, ALL_ATTENTION_FUNCTIONS)
        self.assertNotIn(attention.implementation, ALL_MASK_ATTENTION_FUNCTIONS)


class BackendConfigTest(_BackendTestCase):
    def test_config_subclasses_are_accepted(self):
        @dataclass
        class Config(ExecutorchConfig):
            label: str = "custom"

        recipe = self._register()
        self.exporter.capture(self.model, self.inputs, Config(backend="test"))
        received = recipe.prepare.call_args.args[2]
        self.assertIsInstance(received, Config)
        self.assertEqual(received.label, "custom")

    def test_invalid_backend_options_are_rejected(self):
        for options in ([], {1: "value"}):
            with self.subTest(options=options), self.assertRaises((TypeError, ValueError)):
                config = ExecutorchConfig(backend="test", backend_options=options)
                self.exporter.capture(self.model, self.inputs, config)
        for backend in ("xnnpack", "cuda"):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                self.exporter.capture(
                    self.model, self.inputs, ExecutorchConfig(backend=backend, backend_options={"typo": True})
                )

    def test_snapshot_errors_are_contextual_chained_and_restore_scopes(self):
        class Uncopyable:
            def __deepcopy__(self, memo):
                raise RuntimeError("uncopyable resource")

        snapshot = et._snapshot_config
        self._register(patches=(ExecutorchExportPatch((f"{__name__}._PATCH_OWNER.explicit",), lambda _: "patched"),))
        for phase in (
            "initial config",
            "recipe factory backend_options",
            "retained capture config",
            "capture.config access",
        ):
            with self.subTest(phase=phase):

                def fail_snapshot(value, context):
                    return snapshot(Uncopyable() if context == phase else value, context)

                with (
                    mock.patch.object(et, "_snapshot_config", side_effect=fail_snapshot),
                    self.assertRaisesRegex(TypeError, phase) as raised,
                ):
                    captured = self.exporter.capture(self.model, self.inputs, self.config)
                    _ = captured.config
                self.assertIsInstance(raised.exception.__cause__, RuntimeError)
                self.assertIn("preparation.state", str(raised.exception))
                self.assertEqual(_PATCH_OWNER.explicit, "original")

    def test_copyable_options_are_isolated_and_resources_stay_in_state(self):
        @dataclass
        class Options:
            values: list[int]

        resource = threading.Lock()
        self.prepared.state = resource
        self.config.backend_options = {"custom": Options([1]), "dtype": torch.float32}
        recipe = self._register()

        def factory(options):
            options["custom"].values.append(9)
            return recipe

        self.factory.side_effect = factory
        captured = self.exporter.capture(self.model, self.inputs, self.config)
        self.assertIs(captured.preparation.state, resource)
        self.assertEqual(self.config.backend_options["custom"].values, [1])
        self.config.backend_options["custom"].values.append(2)
        captured.config.backend_options["custom"].values.append(3)
        self.exporter.lower(captured)
        _, prepared, saved = recipe.lower.call_args.args
        self.assertIs(prepared.state, resource)
        self.assertEqual(saved.backend_options, {"custom": Options([1]), "dtype": torch.float32})


class BackendPatchTest(_BackendTestCase):
    def test_registry_collisions_and_exclusions_affect_real_capture(self):
        class Negate(nn.Module):
            def forward(self, x):
                return torch.neg(x)

        original = torch.neg
        self.config.strict = True
        for namespace in ("executorch", "executorch.test", "dynamo"):
            for excluded in (False, True):
                with self.subTest(namespace=namespace, excluded=excluded):
                    prepared = replace(self.prepared, model=Negate())
                    incoming = mock.Mock(return_value=lambda x: x + 100)
                    policy = ExecutorchCompatibilityPolicy(
                        excluded_patch_targets=(f"{__name__}._TORCH_ALIAS.neg",) if excluded else ()
                    )
                    recipe = ExecutorchBackendRecipe(
                        prepare=mock.Mock(return_value=prepared),
                        lower=lambda program, *_: program,
                        patches=(ExecutorchExportPatch(("torch.neg",), lambda _: lambda x: x + 7),),
                        compatibility=policy,
                    )
                    register_executorch_backend("test", mock.Mock(return_value=recipe), overwrite=True)
                    with mock.patch.dict(utils._PATCHES, {namespace: [(torch, "neg", incoming)]}, clear=True):
                        if excluded:
                            captured = self.exporter.capture(self.model, self.inputs, self.config)
                            torch.testing.assert_close(
                                captured.exported_program.module()(x=torch.ones(2)), torch.full((2,), 8.0)
                            )
                            self.assertIs(self.exporter.lower(captured), captured.exported_program)
                        else:
                            with self.assertRaisesRegex(RuntimeError, "recipe"):
                                self.exporter.capture(self.model, self.inputs, self.config)
                    incoming.assert_not_called()
                    self.assertIs(torch.neg, original)

    def test_duplicate_recipe_aliases_fail_before_factories(self):
        factory = mock.Mock()
        self._register(
            patches=(
                ExecutorchExportPatch(
                    (f"{__name__}._PATCH_OWNER.explicit", f"{__name__}._PATCH_ALIAS.explicit"), factory
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "Duplicate recipe patch targets"):
            self.exporter.capture(self.model, self.inputs, self.config)
        factory.assert_not_called()
        self.assertEqual(_PATCH_OWNER.explicit, "original")

    def test_exclusions_require_tuple_of_valid_dotted_paths(self):
        for targets in (["torch.neg"], ("torch",), ("torch..neg",), ("torch.neg ",), (None,)):
            with self.subTest(targets=targets):
                recipe = self._register(compatibility=ExecutorchCompatibilityPolicy(excluded_patch_targets=targets))
                with self.assertRaises((TypeError, ValueError)):
                    self.exporter.capture(self.model, self.inputs, self.config)
                recipe.prepare.assert_not_called()


@require_torch
class BackendImportTest(unittest.TestCase):
    def test_public_spi_and_external_capture_do_not_import_builtin_backends(self):
        script = textwrap.dedent("""
            import importlib.abc
            import sys
            class BlockBuiltins(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.startswith(("executorch.backends.xnnpack", "executorch.backends.cuda")):
                        raise AssertionError("unexpected builtin import: " + fullname)
            sys.meta_path.insert(0, BlockBuiltins())
            import torch
            from transformers.exporters import (
                ExecutorchBackendPreparation, ExecutorchBackendRecipe, ExecutorchCompatibilityPolicy,
                ExecutorchConfig, ExecutorchExporter, register_executorch_backend,
            )
            class Model(torch.nn.Module):
                def forward(self, x):
                    return x + 1
            register_executorch_backend("external", lambda _: ExecutorchBackendRecipe(
                prepare=lambda model, inputs, config: ExecutorchBackendPreparation(
                    model=model, sample_inputs=inputs, normalize_inputs=False),
                lower=lambda *_: None,
                compatibility=ExecutorchCompatibilityPolicy(False, False),
            ))
            exporter = object.__new__(ExecutorchExporter)
            capture = exporter.capture(Model(), {"x": torch.ones(2)}, ExecutorchConfig(backend="external", strict=True))
            torch.testing.assert_close(capture.exported_program.module()(x=torch.ones(2)), torch.full((2,), 2.0))
        """)
        # Run only the literal test script with the current interpreter, without a shell or untrusted input.
        result = subprocess.run(  # nosec B603
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=90
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
