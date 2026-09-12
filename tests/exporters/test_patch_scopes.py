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

import inspect
import sys
import threading
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

from transformers.exporters import utils
from transformers.testing_utils import require_torch
from transformers.utils.import_utils import is_torch_available


if is_torch_available():
    import torch

    from transformers.exporters import exporter_dynamo as dynamo
    from transformers.exporters import exporter_onnx as onnx
    from transformers.exporters.configs import DynamoConfig, OnnxConfig


class _ScopeAssertions:
    def assert_serialized(self, operation, *, scope=utils.export_patch_scope, before_release=None):
        attempted = threading.Event()
        finished = threading.Event()
        errors = []

        def worker():
            attempted.set()
            try:
                operation()
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        thread = threading.Thread(target=worker, daemon=True)
        try:
            with scope():
                thread.start()
                self.assertTrue(attempted.wait(5))
                self.assertFalse(finished.wait(0.05))
                if before_release is not None:
                    before_release()
        finally:
            thread.join(5)
        self.assertFalse(thread.is_alive(), "cooperating operation deadlocked")
        self.assertFalse(errors, errors)
        self.assertTrue(finished.is_set())


class PatchOwnershipTest(_ScopeAssertions, unittest.TestCase):
    def test_empty_patch_collections_hold_shared_scope(self):
        for scope in (lambda: utils.patch_attributes([]), lambda: utils.apply_patches("_absent_patch_scope")):
            with self.subTest(scope=scope):
                self.assert_serialized(lambda: self._enter_scope(), scope=scope)

    @staticmethod
    def _enter_scope():
        with utils.export_patch_scope():
            pass

    def test_attribute_snapshot_waits_for_scope_and_sees_restored_value(self):
        owner = SimpleNamespace(value="original")
        seen = []

        def operation():
            with utils.patch_attribute(owner, "value", lambda original: seen.append(original) or "second"):
                self.assertEqual(owner.value, "second")

        self.assert_serialized(
            operation,
            scope=lambda: utils.patch_attribute(owner, "value", lambda original: "first"),
            before_release=lambda: self.assertEqual(seen, []),
        )
        self.assertEqual(seen, ["original"])
        self.assertEqual(owner.value, "original")

    def test_nested_attribute_scopes_restore_in_reverse_order(self):
        owner = SimpleNamespace(value="original")
        with utils.export_patch_scope(), utils.patch_attribute(owner, "value", lambda value: value + "-outer"):
            with utils.patch_attributes([(owner, "value", lambda value: value + "-inner")]):
                self.assertEqual(owner.value, "original-outer-inner")
            self.assertEqual(owner.value, "original-outer")
        self.assertEqual(owner.value, "original")

    def test_exclusive_conflicts_in_both_orders_before_incoming_factory(self):
        owner = SimpleNamespace(value="original")
        for outer_exclusive, inner_exclusive in ((True, False), (False, True), (True, True)):
            with self.subTest(outer=outer_exclusive, inner=inner_exclusive):
                incoming = mock.Mock(return_value="inner")
                with utils.patch_attribute(
                    owner, "value", lambda value: "outer", exclusive=outer_exclusive, source="outer recipe"
                ):
                    with self.assertRaisesRegex(RuntimeError, "inner recipe.*outer recipe.*excluded_patch_targets"):
                        with utils.patch_attribute(
                            owner, "value", incoming, exclusive=inner_exclusive, source="inner recipe"
                        ):
                            self.fail("conflicting patch entered")
                    incoming.assert_not_called()
                    self.assertEqual(owner.value, "outer")
                self.assertEqual(owner.value, "original")

    def test_exclusive_sees_all_legacy_records_after_nested_scope_exits(self):
        owner = SimpleNamespace(value="original")
        with utils.patch_attribute(owner, "value", lambda value: "outer", source="outer"):
            with utils.patch_attribute(owner, "value", lambda value: "inner", source="inner"):
                self.assertEqual(owner.value, "inner")
            with self.assertRaisesRegex(RuntimeError, "exclusive.*outer"):
                with utils.patch_attribute(owner, "value", mock.Mock(), exclusive=True, source="exclusive"):
                    self.fail("remaining outer patch must conflict")
            self.assertEqual(owner.value, "outer")
        with utils.patch_attribute(owner, "value", lambda value: "exclusive", exclusive=True):
            self.assertEqual(owner.value, "exclusive")
        self.assertEqual(owner.value, "original")

    def test_exclusive_duplicate_aliases_are_preflighted(self):
        owner = SimpleNamespace(value="original")
        alias = owner
        factory = mock.Mock(return_value="patched")
        with self.assertRaisesRegex(RuntimeError, "Duplicate exclusive patch target.*recipe"):
            with utils.patch_attributes(
                [(owner, "value", factory), (alias, "".join(["val", "ue"]), factory)],
                exclusive=True,
                source="recipe",
            ):
                self.fail("duplicate patch collection entered")
        factory.assert_not_called()
        self.assertEqual(owner.value, "original")
        with utils.patch_attributes([(owner, "value", factory)], exclusive=True):
            self.assertEqual(owner.value, "patched")

    def test_same_callable_in_distinct_slots_does_not_conflict(self):
        def original():
            return "original"

        first = SimpleNamespace(value=original, other=original)
        second = SimpleNamespace(value=original)
        with utils.patch_attributes(
            [
                (first, "value", lambda value: lambda: "first"),
                (first, "other", lambda value: lambda: "other"),
                (second, "value", lambda value: lambda: "second"),
            ],
            exclusive=True,
        ):
            self.assertEqual((first.value(), first.other(), second.value()), ("first", "other", "second"))
        self.assertIs(first.value, original)
        self.assertIs(first.other, original)
        self.assertIs(second.value, original)

    def test_reentrant_factory_reserves_slot_before_calling_user_code(self):
        owner = SimpleNamespace(value="original")
        for outer_exclusive, inner_exclusive in ((True, False), (False, True)):
            with self.subTest(outer=outer_exclusive, inner=inner_exclusive):
                incoming = mock.Mock()

                def factory(original):
                    with utils.patch_attribute(owner, "value", incoming, exclusive=inner_exclusive):
                        self.fail("reentrant conflicting factory entered")

                with self.assertRaisesRegex(RuntimeError, "Patch conflict"):
                    with utils.patch_attribute(owner, "value", factory, exclusive=outer_exclusive):
                        self.fail("failed factory entered")
                incoming.assert_not_called()
                with utils.patch_attribute(owner, "value", lambda value: "retry", exclusive=True):
                    self.assertEqual(owner.value, "retry")
                self.assertEqual(owner.value, "original")

    def test_patch_records_are_cleared_on_all_failures(self):
        for exclusive in (False, True):
            for failure in ("get", "factory", "set", "body", "restore"):
                with self.subTest(exclusive=exclusive, failure=failure):

                    class Owner:
                        value = "original"
                        enabled = True

                        def __getattribute__(self, name):
                            if name == "value" and self.enabled and failure == "get":
                                raise RuntimeError("get failed")
                            return object.__getattribute__(self, name)

                        def __setattr__(self, name, value):
                            object.__setattr__(self, name, value)
                            if name == "value" and self.enabled:
                                if failure == "set" and value == "patched":
                                    raise RuntimeError("set failed")
                                if failure == "restore" and value == "original":
                                    raise RuntimeError("restore failed")

                    owner = Owner()
                    first = SimpleNamespace(value="original")

                    def factory(original):
                        if failure == "factory":
                            raise RuntimeError("factory failed")
                        return "patched"

                    with self.assertRaisesRegex(RuntimeError, f"{failure} failed"):
                        with utils.patch_attributes(
                            [(first, "value", lambda value: "patched"), (owner, "value", factory)], exclusive=exclusive
                        ):
                            if failure in ("get", "factory", "set"):
                                self.fail("failed installation must not enter the body")
                            self.assertEqual((first.value, owner.value), ("patched", "patched"))
                            if failure == "body":
                                raise RuntimeError("body failed")
                    owner.enabled = False
                    self.assertEqual(owner.value, "original")
                    self.assertEqual(first.value, "original")
                    with utils.patch_attributes(
                        [(first, "value", lambda value: "retry"), (owner, "value", lambda value: "retry")],
                        exclusive=True,
                    ):
                        self.assertEqual((first.value, owner.value), ("retry", "retry"))
                    self.assertNotIn((id(owner), "value"), utils._ACTIVE_PATCHES)
                    self.assertNotIn((id(first), "value"), utils._ACTIVE_PATCHES)
                    self.assert_serialized(self._enter_scope)

    def test_restoration_remains_inside_shared_scope(self):
        observations = []

        class Owner:
            value = "original"

            def __setattr__(self, name, value):
                if value == "original":
                    acquired = []

                    def try_lock():
                        locked = utils._EXPORT_PATCH_LOCK.acquire(blocking=False)
                        acquired.append(locked)
                        if locked:
                            utils._EXPORT_PATCH_LOCK.release()

                    thread = threading.Thread(target=try_lock, daemon=True)
                    thread.start()
                    thread.join(5)
                    observations.extend(acquired)
                object.__setattr__(self, name, value)

        owner = Owner()
        with utils.patch_attribute(owner, "value", lambda original: "patched"):
            self.assertEqual(owner.value, "patched")
        self.assertEqual(observations, [False])
        self.assertEqual(owner.value, "original")


class PatchRegistryTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.dict(utils._PATCHES, {"test": []}, clear=True))
        self.enterContext(mock.patch.dict(sys.modules))
        self.module = ModuleType("_scope_target")
        self.module.value = "original"
        sys.modules[self.module.__name__] = self.module

    def test_lazy_targets_preserve_eager_and_direct_append_order(self):
        module = self.module
        utils.register_patch("test", "_scope_target.value")(lambda value: value + "-eager")
        eager = utils._PATCHES["test"][0]
        utils.register_patch("test", "_scope_target.value", lazy=True)(lambda value: value + "-lazy")
        utils._PATCHES["test"].append((module, "value", lambda value: value + "-direct"))
        with utils.apply_patches("test"):
            self.assertEqual(module.value, "original-eager-lazy-direct")
        self.assertEqual(module.value, "original")
        self.assertIs(utils._PATCHES["test"][0], eager)
        self.assertIs(eager[0], module)

    def test_lazy_missing_import_and_attribute_are_retried(self):
        module = self.module
        del sys.modules[module.__name__]
        del module.value
        with mock.patch.object(utils, "_resolve_dotted_path", wraps=utils._resolve_dotted_path) as resolve:
            utils.register_patch("test", "_scope_target.value", lazy=True)(lambda value: value + "-patch")
            resolve.assert_not_called()
            with self.assertLogs(utils.logger.name, level="DEBUG") as logs, utils.apply_patches("test"):
                pass
            self.assertIn(module.__name__, " ".join(logs.output))
            sys.modules[module.__name__] = module
            with utils.apply_patches("test"):
                self.assertFalse(hasattr(module, "value"))
            module.value = "original"
            with utils.apply_patches("test"):
                self.assertEqual(module.value, "original-patch")
        self.assertEqual(module.value, "original")

    def test_registry_conflicts_and_alias_exclusions_match_physical_slots(self):
        module = self.module
        original = mock.Mock(return_value="original")
        module.owner = module.alias = SimpleNamespace(value=original)
        module.other = SimpleNamespace(value=original)
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                utils._PATCHES["test"] = []
                factory = mock.Mock(return_value=lambda: "registry")
                utils.register_patch("test", "_scope_target.owner.value", lazy=lazy)(factory)
                utils._PATCHES["test"].append((module.other, "value", lambda _: lambda: "other"))
                with utils.patch_attribute(module.alias, "value", lambda _: lambda: "recipe", exclusive=True):
                    with self.assertRaisesRegex(RuntimeError, "Patch conflict"), utils.apply_patches("test"):
                        self.fail("registry shadowed exclusive recipe")
                    with utils.apply_patches("test", exclude=("_scope_target.alias.value",)):
                        self.assertEqual(module.owner.value(), "recipe")
                        self.assertEqual(module.other.value(), "other")
                factory.assert_not_called()
                self.assertIs(module.owner.value, original)
                self.assertIs(module.other.value, original)

    def test_exact_lazy_exclusion_does_not_resolve_or_import_owner(self):
        utils.register_patch("test", "_unimported.owner.value", lazy=True)(mock.Mock())
        with (
            mock.patch.object(utils, "_resolve_dotted_path") as resolve,
            mock.patch("importlib.import_module") as import_module,
            utils.apply_patches("test", exclude=("_unimported.owner.value",)),
        ):
            resolve.assert_not_called()
            import_module.assert_not_called()

    def test_exact_lazy_exclusion_does_not_import_during_other_slot_checks(self):
        self.module.__getattr__ = mock.Mock(side_effect=AssertionError("lazy module attribute imported"))
        other = SimpleNamespace(value="original")
        utils._PATCHES["test"].append((other, "value", lambda _: "patched"))
        utils.register_patch("test", "_scope_target.owner.value", lazy=True)(mock.Mock())
        with mock.patch("importlib.import_module") as import_module:
            with utils.apply_patches("test", exclude=("_scope_target.owner.value",)):
                self.assertEqual(other.value, "patched")
                import_module.assert_not_called()
                self.module.__getattr__.assert_not_called()
        self.assertEqual(other.value, "original")

    def test_exclusions_do_not_leak_into_nested_or_later_scopes(self):
        utils.register_patch("test", "_scope_target.value", lazy=True)(lambda value: value + "-patched")
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with utils.apply_patches("test", exclude=("_scope_target.value",)):
                self.assertEqual(self.module.value, "original")
                with utils.apply_patches("test"):
                    self.assertEqual(self.module.value, "original-patched")
                self.assertEqual(self.module.value, "original")
                raise RuntimeError("body failed")
        with utils.apply_patches("test"):
            self.assertEqual(self.module.value, "original-patched")
        self.assertEqual(self.module.value, "original")

    def test_lazy_resolution_remains_interleaved_with_installation_and_exclusions(self):
        module = self.module
        module.owner = original = SimpleNamespace(value="original")
        module.alias = replacement = SimpleNamespace(value="replacement")
        utils._PATCHES["test"].append((module, "owner", lambda _: replacement))
        utils.register_patch("test", "_scope_target.owner.value", lazy=True)(lambda value: value + "-patch")
        for exclusions, expected in (((), "replacement-patch"), (("_scope_target.alias.value",), "replacement")):
            with self.subTest(exclusions=exclusions), utils.apply_patches("test", exclude=exclusions):
                self.assertIs(module.owner, replacement)
                self.assertEqual(module.owner.value, expected)
                self.assertEqual(original.value, "original")
            self.assertIs(module.owner, original)
            self.assertEqual(replacement.value, "replacement")

    def test_unavailable_exclusion_alias_is_retried_on_next_scope(self):
        owner = SimpleNamespace(value="original")
        utils._PATCHES["test"].append((owner, "value", lambda _: "patched"))
        del sys.modules[self.module.__name__]
        with utils.apply_patches("test", exclude=("_scope_target.owner.value",)):
            self.assertEqual(owner.value, "patched")
        self.module.owner = owner
        sys.modules[self.module.__name__] = self.module
        with utils.apply_patches("test", exclude=("_scope_target.owner.value",)):
            self.assertEqual(owner.value, "original")
        with utils.apply_patches("test"):
            self.assertEqual(owner.value, "patched")
        self.assertEqual(owner.value, "original")

    def test_registration_does_not_acquire_export_lock(self):
        errors = []

        def register():
            try:
                for lazy in (False, True):
                    utils.register_patch("test", "_scope_target.value", lazy=lazy)(lambda value: value)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=register, daemon=True)
        with utils.export_patch_scope():
            thread.start()
            thread.join(5)
            blocked = thread.is_alive()
        thread.join(5)
        self.assertFalse(blocked, "import-time registration waited for the export lock")
        self.assertFalse(errors, errors)
        self.assertEqual(len(utils._PATCHES["test"]), 2)


@require_torch
class MutableExportHelperTest(_ScopeAssertions, unittest.TestCase):
    def make_model(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(use_cache=True, use_mamba_kernels=True)
                self.cached_sequence_length = 7

            def forward(self, x=None, **kwargs):
                return x

        return Model()

    def test_mutable_helpers_wait_before_reading_model_state(self):
        model = self.make_model()
        original_forward = model.forward
        scopes = (
            lambda: dynamo.patch_model_config(model, {"use_cache": False}),
            lambda: dynamo.patch_forward_signature(model, {"x": None}),
            lambda: dynamo.reset_model_state(model),
            lambda: onnx.patch_model_outputs(model),
            lambda: utils._capture_forward(model),
        )
        for scope in scopes:
            with self.subTest(scope=scope):

                def operation():
                    with scope():
                        pass

                def unchanged():
                    self.assertEqual(model.forward, original_forward)
                    self.assertEqual(model.cached_sequence_length, 7)
                    self.assertTrue(model.config.use_cache)
                    self.assertTrue(model.config.use_mamba_kernels)

                self.assert_serialized(operation, before_release=unchanged)
                unchanged()

    def test_nested_signature_config_and_reset_scopes_restore_on_failure(self):
        model = self.make_model()
        original_forward = model.forward
        with utils.export_patch_scope(), dynamo.patch_forward_signature(model, {"x": None}):
            outer_forward = model.forward
            with dynamo.reset_model_state(model), dynamo.patch_model_config(model, {"use_cache": False}):
                self.assertIsNone(model.cached_sequence_length)
                model.cached_sequence_length = 9
                with self.assertRaisesRegex(RuntimeError, "body failed"):
                    with (
                        dynamo.patch_forward_signature(model, {"x": None, "extra": None}),
                        dynamo.reset_model_state(model),
                        dynamo.patch_model_config(model, {"use_cache": True}),
                    ):
                        self.assertEqual(list(inspect.signature(model.forward).parameters), ["x", "extra"])
                        self.assertIsNone(model.cached_sequence_length)
                        self.assertTrue(model.config.use_cache)
                        raise RuntimeError("body failed")
                self.assertIs(model.forward, outer_forward)
                self.assertEqual(model.cached_sequence_length, 9)
                self.assertFalse(model.config.use_cache)
                self.assertFalse(model.config.use_mamba_kernels)
        self.assertEqual(model.forward, original_forward)
        self.assertEqual(model.cached_sequence_length, 7)
        self.assertTrue(model.config.use_cache)
        self.assertTrue(model.config.use_mamba_kernels)

    def test_reset_partial_setter_failure_restores_state(self):
        class Model(torch.nn.Module):
            cached_sequence_length = 7
            cached_rotary_positional_embedding = "original"

            def __setattr__(self, name, value):
                super().__setattr__(name, value)
                if name == "cached_sequence_length" and value is None:
                    raise RuntimeError("reset setter failed")

        model = Model()
        with self.assertRaisesRegex(RuntimeError, "reset setter failed"):
            with dynamo.reset_model_state(model):
                self.fail("reset must not enter after failure")
        self.assertEqual(model.cached_sequence_length, 7)
        self.assertEqual(model.cached_rotary_positional_embedding, "original")

    def test_forward_wrappers_share_reentrant_scope_and_restore(self):
        model = self.make_model()
        original_forward = model.forward
        value = torch.ones(2)
        with utils.export_patch_scope(), utils._capture_forward(model) as calls:
            with onnx.patch_model_outputs(model) as (input_names, output_names):
                with dynamo.patch_forward_signature(model, {"x": value}):
                    result = model(x=value)
            self.assertEqual(input_names, ["x"])
            self.assertEqual(output_names, ["output"])
        torch.testing.assert_close(result["output"], value)
        torch.testing.assert_close(calls[0]["x"], value)
        self.assertIsNot(calls[0]["x"], value)
        self.assertEqual(model.forward, original_forward)

    def test_preparation_and_pytree_registration_share_scope(self):
        model = self.make_model()
        with mock.patch.object(torch.utils._pytree, "register_pytree_node") as register:
            operations = (
                lambda: utils.prepare_for_export(model, {}),
                lambda: utils.precompute_export_inputs(model, {}),
                lambda: dynamo.register_pytree_node(type(model)),
                lambda: dynamo.register_cache_pytrees_for_model(model),
            )
            for operation in operations:
                with self.subTest(operation=operation):
                    register.reset_mock()
                    self.assert_serialized(operation, before_release=register.assert_not_called)
            self.assertTrue(register.called)

    def test_export_entry_points_lock_before_helpers(self):
        model = self.make_model()
        cases = (
            (dynamo.DynamoExporter(), "export", DynamoConfig(), dynamo, "prepare_for_export"),
            (dynamo.DynamoExporter(), "_export_prepared", DynamoConfig(), dynamo, "register_cache_pytrees_for_model"),
            (object.__new__(onnx.OnnxExporter), "export", OnnxConfig(), onnx, "patch_model_outputs"),
        )
        for exporter, method, config, module, helper in cases:
            with self.subTest(method=method):
                with mock.patch.object(module, helper, side_effect=RuntimeError("stopped after lock")) as entry:

                    def operation():
                        with self.assertRaisesRegex(RuntimeError, "stopped after lock"):
                            getattr(exporter, method)(model, {}, config)

                    self.assert_serialized(operation, before_release=entry.assert_not_called)
                entry.assert_called_once()

    def test_prepared_dynamo_forwards_exclusions_only_to_its_registry(self):
        model = self.make_model()
        module = ModuleType("_scope_dynamo_exclusion_target")
        module.value = "original"
        original_forward = model.forward
        sample = torch.ones(2)
        exporter = dynamo.DynamoExporter()
        with (
            mock.patch.dict(utils._PATCHES, {"dynamo": [(module, "value", lambda value: "registry")]}),
            mock.patch.dict(sys.modules, {module.__name__: module}),
            utils.patch_attribute(module, "value", lambda value: "recipe", exclusive=True),
        ):

            def capture(*args, **kwargs):
                self.assertEqual(module.value, "recipe")
                self.assertIsNone(model.cached_sequence_length)
                self.assertFalse(model.config.use_cache)
                self.assertEqual(list(inspect.signature(model.forward).parameters), ["x"])
                return "program"

            with mock.patch.object(torch.export, "export", side_effect=capture) as export:
                result = exporter._export_prepared(
                    model,
                    {"x": sample},
                    DynamoConfig(),
                    output_flags={"use_cache": False},
                    patch_exclusions=(f"{module.__name__}.value",),
                )
                self.assertEqual(result, "program")
                export.assert_called_once()
                with self.assertRaisesRegex(RuntimeError, "Patch conflict"):
                    exporter._export_prepared(model, {"x": sample}, DynamoConfig())
                export.assert_called_once()
        self.assertEqual(module.value, "original")
        self.assertEqual(model.forward, original_forward)
        self.assertEqual(model.cached_sequence_length, 7)
        self.assertTrue(model.config.use_cache)

    def test_real_dynamo_capture_still_works(self):
        model = self.make_model()
        sample = torch.ones(2)
        program = dynamo.DynamoExporter().export(model, {"x": sample}, DynamoConfig())
        torch.testing.assert_close(program.module()(x=sample), sample)
        self.assertEqual(model.cached_sequence_length, 7)
