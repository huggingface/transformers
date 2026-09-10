# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import functools
import gc
import io
import os
import re
import unittest
import warnings
import weakref
from unittest.mock import patch

from transformers import testing_utils
from transformers.testing_utils import (
    cap_psutil_cpu_memory,
    get_ci_cpu_memory_budget_gib,
    get_cpu_ram_total_gib,
    require_torch,
)
from transformers.utils import is_torch_available

from .. import test_memory_cleanup_mixin
from ..test_memory_cleanup_mixin import MemoryCleanupMixin, MemoryCleanupTestCase, with_grad, with_no_grad


if is_torch_available():
    import torch


GIB = 1024**3


class Payload:
    """Stand-in for a model, with a lifetime observable through a `weakref`."""


class GetCiCpuMemoryBudgetTest(unittest.TestCase):
    """`CI_CPU_MEMORY_LIMIT_GB` is a per-accelerator budget, so it scales with the accelerator count."""

    def test_returns_none_outside_ci(self):
        with patch.dict("os.environ", {}, clear=False):
            testing_utils.os.environ.pop("CI_CPU_MEMORY_LIMIT_GB", None)
            self.assertIsNone(get_ci_cpu_memory_budget_gib())

    def test_scales_with_accelerator_count(self):
        with (
            patch.dict("os.environ", {"CI_CPU_MEMORY_LIMIT_GB": "60"}),
            patch.object(testing_utils, "torch_device", "cuda"),
            patch.object(testing_utils, "backend_device_count", return_value=2),
        ):
            self.assertEqual(get_ci_cpu_memory_budget_gib(), 120.0)

    def test_single_accelerator_is_the_bare_budget(self):
        with (
            patch.dict("os.environ", {"CI_CPU_MEMORY_LIMIT_GB": "60"}),
            patch.object(testing_utils, "torch_device", "cuda"),
            patch.object(testing_utils, "backend_device_count", return_value=1),
        ):
            self.assertEqual(get_ci_cpu_memory_budget_gib(), 60.0)

    def test_ignores_a_malformed_value(self):
        with patch.dict("os.environ", {"CI_CPU_MEMORY_LIMIT_GB": "not-a-number"}):
            self.assertIsNone(get_ci_cpu_memory_budget_gib())


class GetCpuRamTotalTest(unittest.TestCase):
    """
    The guard has to hold both inside a pod (where physical RAM reports the whole node) and on a bare runner
    (where there is no cgroup limit), so it prefers measurements and treats the CI budget as a fallback.
    """

    def _resolve(self, cgroup_gib=None, physical_gib=None, ci_budget_gib=None):
        with (
            patch.object(
                testing_utils,
                "get_cgroup_memory_limit_bytes",
                return_value=None if cgroup_gib is None else int(cgroup_gib * GIB),
            ),
            patch.object(testing_utils, "get_physical_cpu_ram_gib", return_value=physical_gib),
            patch.object(testing_utils, "get_ci_cpu_memory_budget_gib", return_value=ci_budget_gib),
        ):
            return get_cpu_ram_total_gib()

    def test_inside_a_pod_the_cgroup_limit_wins(self):
        # Physical RAM is the whole node here; the cgroup limit is what the OOM killer enforces.
        self.assertEqual(self._resolve(cgroup_gib=60, physical_gib=750, ci_budget_gib=120), 60.0)

    def test_on_a_bare_runner_physical_ram_wins_over_the_ci_budget(self):
        # A 2-accelerator A10 runner: no cgroup limit, 180 GiB real. The 120 GiB budget is a device_map planning
        # number, and using it here would make every guard on this runner over-skip.
        self.assertEqual(self._resolve(cgroup_gib=None, physical_gib=180, ci_budget_gib=120), 180.0)

    def test_falls_back_to_the_ci_budget_when_nothing_is_measurable(self):
        self.assertEqual(self._resolve(cgroup_gib=None, physical_gib=None, ci_budget_gib=120), 120.0)

    def test_is_infinite_when_nothing_can_answer(self):
        # An ordinary local setup, not a broken one: callers should run their test rather than skip it.
        self.assertEqual(self._resolve(), float("inf"))

    def test_takes_the_smaller_measurement(self):
        self.assertEqual(self._resolve(cgroup_gib=90, physical_gib=180), 90.0)
        self.assertEqual(self._resolve(cgroup_gib=180, physical_gib=90), 90.0)


class GetPhysicalCpuRamTest(unittest.TestCase):
    def test_reads_past_the_device_map_cap(self):
        """
        `conftest.py` caps `psutil.virtual_memory` to a `device_map="auto"` planning budget. A guard asking whether
        an allocation will get the container OOM-killed needs the machine's real RAM, not that budget.
        """
        import psutil

        real_total = psutil.virtual_memory().total
        original_virtual_memory = psutil.virtual_memory
        original_unpatched = testing_utils._UNPATCHED_VIRTUAL_MEMORY
        try:
            testing_utils._UNPATCHED_VIRTUAL_MEMORY = None
            testing_utils.patch_psutil_cpu_memory(8 * GIB)

            self.assertEqual(psutil.virtual_memory().total, 8 * GIB)
            self.assertAlmostEqual(testing_utils.get_physical_cpu_ram_gib(), real_total / GIB, places=3)
        finally:
            psutil.virtual_memory = original_virtual_memory
            testing_utils._UNPATCHED_VIRTUAL_MEMORY = original_unpatched

    def test_returns_none_without_psutil(self):
        with patch.object(testing_utils, "is_psutil_available", return_value=False):
            self.assertIsNone(testing_utils.get_physical_cpu_ram_gib())


class CapPsutilCpuMemoryTest(unittest.TestCase):
    def setUp(self):
        import psutil

        self._original_virtual_memory = psutil.virtual_memory
        self._original_unpatched = testing_utils._UNPATCHED_VIRTUAL_MEMORY
        testing_utils._UNPATCHED_VIRTUAL_MEMORY = None

    def tearDown(self):
        import psutil

        psutil.virtual_memory = self._original_virtual_memory
        testing_utils._UNPATCHED_VIRTUAL_MEMORY = self._original_unpatched

    def test_caps_and_restores_on_exit(self):
        import psutil

        # `before` may already be conftest's session-wide cap, not the true original — that's fine,
        # we only assert the context manager restores whatever it found on entry.
        before = psutil.virtual_memory
        with cap_psutil_cpu_memory(int(0.5 * GIB)):
            self.assertEqual(psutil.virtual_memory().total, int(0.5 * GIB))
        self.assertIs(psutil.virtual_memory, before)

    def test_restores_on_exception(self):
        import psutil

        before = psutil.virtual_memory
        try:
            with cap_psutil_cpu_memory(int(0.5 * GIB)):
                raise RuntimeError("deliberate test error")
        except RuntimeError:
            pass
        self.assertIs(psutil.virtual_memory, before)

    def test_nested_unwinds_in_order(self):
        import psutil

        original = psutil.virtual_memory
        with cap_psutil_cpu_memory(int(0.5 * GIB)):
            self.assertEqual(psutil.virtual_memory().total, int(0.5 * GIB))
            with cap_psutil_cpu_memory(int(0.2 * GIB)):
                self.assertEqual(psutil.virtual_memory().total, int(0.2 * GIB))
            # Inner block exited: should be back to the outer cap
            self.assertEqual(psutil.virtual_memory().total, int(0.5 * GIB))
        # Outer block exited: should be back to the original callable
        self.assertIs(psutil.virtual_memory, original)


def _run_inner_test_class(cls):
    """Run every test in `cls` through unittest and return the result, keeping the runner silent."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(cls)
    runner = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0)
    return runner.run(suite)


class MemoryCleanupMixinTest(unittest.TestCase):
    """What matters is that the references are really gone, not that `cleanup()` was called."""

    def test_instance_attributes_are_dropped_after_the_test(self):
        seen = {}

        class Inner(MemoryCleanupMixin, unittest.TestCase):
            def test_leaks(self):
                self.payload = Payload()
                seen["ref"] = weakref.ref(self.payload)
                seen["case"] = self

        result = _run_inner_test_class(Inner)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        # The instance is still alive, which is why the attribute has to go.
        self.assertNotIn("payload", vars(seen["case"]))
        gc.collect()
        self.assertIsNone(seen["ref"](), "the object the test parked on `self` was not released")

    def test_cached_property_cache_is_dropped(self):
        seen = {}

        class Inner(MemoryCleanupMixin, unittest.TestCase):
            @functools.cached_property
            def model(self):
                return Payload()

            def test_uses_the_cached_property(self):
                seen["ref"] = weakref.ref(self.model)
                seen["case"] = self

        result = _run_inner_test_class(Inner)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertNotIn("model", vars(seen["case"]))
        gc.collect()
        self.assertIsNone(seen["ref"](), "the `@cached_property` value outlived the test")

    def test_class_attributes_are_dropped_after_the_class(self):
        seen = {}

        class Inner(MemoryCleanupMixin, unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                super().setUpClass()
                cls.shared_model = Payload()
                seen["ref"] = weakref.ref(cls.shared_model)

            def test_uses_the_class_attribute(self):
                self.assertIsNotNone(self.shared_model)

        result = _run_inner_test_class(Inner)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertNotIn("shared_model", vars(Inner))
        gc.collect()
        self.assertIsNone(seen["ref"](), "the model parked on the class outlived the class")

    def test_attributes_from_the_class_body_survive(self):
        class Inner(MemoryCleanupMixin, unittest.TestCase):
            checkpoint = "hf-internal-testing/tiny-random-gpt2"

            def test_reads_the_class_attribute(self):
                self.assertEqual(self.checkpoint, "hf-internal-testing/tiny-random-gpt2")

        result = _run_inner_test_class(Inner)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertEqual(Inner.checkpoint, "hf-internal-testing/tiny-random-gpt2")

    def test_setupclass_needs_no_particular_super_ordering(self):
        seen = {}

        class Inner(MemoryCleanupMixin, unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                cls.shared_model = Payload()
                seen["ref"] = weakref.ref(cls.shared_model)
                super().setUpClass()  # called last on purpose

            def test_uses_the_class_attribute(self):
                self.assertIsNotNone(self.shared_model)

        result = _run_inner_test_class(Inner)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        # The snapshot is taken at class creation, so `setUpClass` cannot accidentally protect what it assigns.
        self.assertNotIn("shared_model", vars(Inner))
        gc.collect()
        self.assertIsNone(seen["ref"]())

    def test_teardown_still_runs_when_the_test_fails(self):
        seen = {}

        class Inner(MemoryCleanupMixin, unittest.TestCase):
            def test_fails(self):
                self.payload = Payload()
                seen["case"] = self
                raise RuntimeError("deliberate test error")

        result = _run_inner_test_class(Inner)
        self.assertEqual(len(result.errors), 1)
        self.assertNotIn("payload", vars(seen["case"]))

    def test_setup_attributes_are_per_test_state(self):
        seen = {}

        class Inner(MemoryCleanupMixin, unittest.TestCase):
            def setUp(self):
                super().setUp()
                self.fixture = Payload()

            def test_uses_the_fixture(self):
                seen["case"] = self
                self.assertIsNotNone(self.fixture)

        result = _run_inner_test_class(Inner)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        # `setUp` runs after the snapshot, so what it loads is dropped too.
        self.assertNotIn("fixture", vars(seen["case"]))

    def test_a_setup_that_skips_super_is_an_error(self):
        class Inner(MemoryCleanupMixin, unittest.TestCase):
            def setUp(self):  # deliberately does not call super()
                self.fixture = Payload()

            def test_runs(self):
                pass

        result = _run_inner_test_class(Inner)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("must call super().setUp()", str(result.errors[0][1]))


@require_torch
class MemoryCleanupNoGradTest(unittest.TestCase):
    def test_test_methods_run_under_no_grad_by_default(self):
        seen = {}

        class Inner(MemoryCleanupMixin, unittest.TestCase):
            def test_grad_is_off(self):
                seen["grad_enabled"] = torch.is_grad_enabled()

        result = _run_inner_test_class(Inner)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertFalse(seen["grad_enabled"])
        # The surrounding grad mode is restored.
        self.assertTrue(torch.is_grad_enabled())

    def test_cleanup_no_grad_false_leaves_autograd_on(self):
        seen = {}

        class Inner(MemoryCleanupMixin, unittest.TestCase):
            cleanup_no_grad = False

            def test_grad_is_on(self):
                seen["grad_enabled"] = torch.is_grad_enabled()

        result = _run_inner_test_class(Inner)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertTrue(seen["grad_enabled"])

    def test_the_decorators_override_the_class_default(self):
        seen = {}

        class Inner(MemoryCleanupMixin, unittest.TestCase):
            @with_grad
            def test_opts_into_grad(self):
                seen["on_a_no_grad_class"] = torch.is_grad_enabled()

        class InnerTraining(MemoryCleanupMixin, unittest.TestCase):
            cleanup_no_grad = False

            @with_no_grad
            def test_opts_out_of_grad(self):
                seen["on_a_grad_class"] = torch.is_grad_enabled()

        for cls in (Inner, InnerTraining):
            result = _run_inner_test_class(cls)
            self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertTrue(seen["on_a_no_grad_class"])
        self.assertFalse(seen["on_a_grad_class"])


@require_torch
class MemoryCleanupUnderPytestTest(MemoryCleanupMixin, unittest.TestCase):
    """Guards the private `_callTestMethod` hook. Unlike the tests above, this class is collected and run by
    pytest itself, so it catches a runner that stops routing through the hook."""

    def test_grad_is_off_under_the_real_runner(self):
        self.assertFalse(torch.is_grad_enabled())

    def test_the_private_hook_is_still_there(self):
        self.assertTrue(
            hasattr(unittest.TestCase, "_callTestMethod"),
            "`unittest.TestCase._callTestMethod` is gone; `MemoryCleanupMixin.cleanup_no_grad` needs a new seam",
        )

    def test_attributes_are_dropped_under_the_real_runner(self):
        # `doCleanups` runs after `tearDown`, so this sees what the mixin left behind.
        self.payload = Payload()
        self.addCleanup(lambda: self.assertNotIn("payload", vars(self)))


class MemoryLeakCheckTest(unittest.TestCase):
    """Opt-in: unset means never measured, `warn` reports, `error` fails."""

    MIB = 1024**2

    def _run_leaking_class(self, leaked_mib):
        # `setUp` reads the baseline, `tearDown` reads it again after cleanup.
        allocations = iter([0, int(leaked_mib * self.MIB)])

        class Inner(MemoryCleanupMixin, unittest.TestCase):
            def test_noop(self):
                pass

        with patch.object(
            test_memory_cleanup_mixin, "_device_memory_allocated", side_effect=lambda: next(allocations)
        ):
            return _run_inner_test_class(Inner)

    def test_off_by_default(self):
        with patch.dict("os.environ", {}, clear=False):
            testing_utils.os.environ.pop("TRANSFORMERS_TEST_MEMORY_LEAK_MIB", None)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = self._run_leaking_class(leaked_mib=512)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertEqual([str(w.message) for w in caught if "MiB allocated" in str(w.message)], [])

    def test_warns_above_the_threshold(self):
        with patch.dict("os.environ", {"TRANSFORMERS_TEST_MEMORY_LEAK_MIB": "10"}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = self._run_leaking_class(leaked_mib=512)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        messages = [str(w.message) for w in caught if "MiB allocated" in str(w.message)]
        self.assertEqual(len(messages), 1, caught)
        self.assertIn("512.0 MiB", messages[0])
        # The CPU figure is a delta, not the whole RSS.
        rss_delta = float(re.search(r"CPU RSS ([-+][\d.]+) MiB", messages[0]).group(1))
        self.assertLess(abs(rss_delta), 100, messages[0])

    def test_silent_below_the_threshold(self):
        with patch.dict("os.environ", {"TRANSFORMERS_TEST_MEMORY_LEAK_MIB": "10"}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = self._run_leaking_class(leaked_mib=1)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertEqual([str(w.message) for w in caught if "MiB allocated" in str(w.message)], [])

    def test_error_mode_fails_the_test(self):
        env = {"TRANSFORMERS_TEST_MEMORY_LEAK_MIB": "10", "TRANSFORMERS_TEST_MEMORY_LEAK_MODE": "error"}
        with patch.dict("os.environ", env):
            result = self._run_leaking_class(leaked_mib=512)
        self.assertEqual(len(result.errors) + len(result.failures), 1)

    def test_rejects_a_malformed_threshold(self):
        with patch.dict("os.environ", {"TRANSFORMERS_TEST_MEMORY_LEAK_MIB": "lots"}):
            with self.assertRaises(ValueError):
                test_memory_cleanup_mixin._memory_leak_settings()

    def test_rejects_an_unknown_mode(self):
        env = {"TRANSFORMERS_TEST_MEMORY_LEAK_MIB": "10", "TRANSFORMERS_TEST_MEMORY_LEAK_MODE": "explode"}
        with patch.dict("os.environ", env):
            with self.assertRaises(ValueError):
                test_memory_cleanup_mixin._memory_leak_settings()


class MemoryCleanupTestCaseTest(unittest.TestCase):
    def test_it_keeps_the_test_case_plus_features(self):
        seen = {}

        class Inner(MemoryCleanupTestCase):
            def test_uses_a_tmp_dir(self):
                seen["tmp_dir"] = self.get_auto_remove_tmp_dir()
                seen["case"] = self
                self.payload = Payload()

        result = _run_inner_test_class(Inner)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertFalse(os.path.exists(seen["tmp_dir"]), "TestCasePlus no longer removes its tmp dirs")
        self.assertNotIn("payload", vars(seen["case"]))
