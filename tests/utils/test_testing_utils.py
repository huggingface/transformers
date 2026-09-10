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

import unittest
from unittest.mock import patch

from transformers import testing_utils
from transformers.testing_utils import cap_psutil_cpu_memory, get_ci_cpu_memory_budget_gib, get_cpu_ram_total_gib


GIB = 1024**3


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
