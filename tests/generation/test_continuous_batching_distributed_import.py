# Copyright 2025 The HuggingFace Team Inc.
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
"""Regression tests for transformers#46170.

Some PyTorch builds (notably AMD ROCm on Windows) ship an incomplete `torch.distributed`
package. Until the fix in `generation/continuous_batching/distributed.py`, the unconditional
`from torch.distributed.tensor.device_mesh import DeviceMesh` import propagated up to
`from transformers import ...`, breaking unrelated models such as CLIPSeg.

These tests run in a child interpreter so we can patch the failing import without
interfering with the parent test process.
"""

import subprocess
import sys
import textwrap
import unittest


def _run_in_subprocess(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )


# Shared preamble: monkeypatch `__import__` so any attempt to import
# `torch.distributed.tensor.device_mesh` raises ImportError, mirroring the failure
# reported on AMD ROCm / Windows in #46170.
_PREAMBLE = textwrap.dedent(
    """
    import builtins

    _real_import = builtins.__import__

    def _failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch.distributed.tensor.device_mesh" or name.startswith("torch.distributed.tensor.device_mesh."):
            raise ImportError("cannot import name 'FileStore' from 'torch.distributed'")
        if name == "torch.distributed.tensor" and fromlist and "device_mesh" in fromlist:
            raise ImportError("cannot import name 'FileStore' from 'torch.distributed'")
        return _real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = _failing_import
    """
)


class TestContinuousBatchingDistributedImportGuard(unittest.TestCase):
    """The distributed helper module must import even when `torch.distributed.tensor.device_mesh` is broken."""

    def test_distributed_module_imports_when_device_mesh_unavailable(self):
        script = _PREAMBLE + textwrap.dedent(
            """
            from transformers.generation.continuous_batching import distributed as cb_dist
            assert cb_dist.DeviceMesh is None, "DeviceMesh should be None when its import fails"
            # The class must be defined; the `DeviceMesh | None` annotation must not be evaluated.
            assert cb_dist.DistributedHelper is not None
            print("OK")
            """
        )
        result = _run_in_subprocess(script)
        self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertIn("OK", result.stdout)

    def test_distributed_helper_constructs_with_none_device_mesh(self):
        # The common single-process path passes `device_mesh=None`; this must succeed even when DeviceMesh is unavailable.
        script = _PREAMBLE + textwrap.dedent(
            """
            from transformers.generation.continuous_batching.distributed import DistributedHelper
            helper = DistributedHelper(device_mesh=None)
            assert helper.device_mesh is None
            assert helper.tp_size == 1
            assert helper.world_size == 1
            print("OK")
            """
        )
        result = _run_in_subprocess(script)
        self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertIn("OK", result.stdout)

    def test_distributed_helper_raises_on_real_device_mesh_when_class_unavailable(self):
        # If a caller somehow supplies a non-None device_mesh while DeviceMesh itself is unavailable, surface an
        # actionable ImportError rather than a confusing AttributeError later.
        script = _PREAMBLE + textwrap.dedent(
            """
            from transformers.generation.continuous_batching.distributed import DistributedHelper
            class _StubMesh:
                pass
            try:
                DistributedHelper(device_mesh=_StubMesh())
            except ImportError as e:
                msg = str(e)
                assert "torch.distributed.tensor.device_mesh" in msg, msg
                print("OK")
            else:
                raise AssertionError("expected ImportError")
            """
        )
        result = _run_in_subprocess(script)
        self.assertEqual(result.returncode, 0, msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
