# Copyright 2025 ModelCloud.ai team and The HuggingFace Inc. team.
##
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

"""Regression harness that reproduces the raw `regex` crash when the GIL is off."""

import os
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(
    not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled(),
    reason="Crash regression only runs when the GIL is disabled",
)


@pytest.mark.xfail(strict=False, reason="Raw regex crashes under PYTHON_GIL=0")
def test_raw_regex_thread_safety_crashes_under_gil0():
    script = r"""
import threading
import regex

pattern_text = r"(?P<prefix>test)(?P<number>\d+)"
ready_group = threading.Barrier(33)
run_event = threading.Event()


def worker():
    ready_group.wait()
    run_event.wait()
    for _ in range(100):
        match = regex.match(pattern_text, "test123")
        assert match is not None
        assert match.group("prefix") == "test"
        assert match.group("number") == "123"


threads = [threading.Thread(target=worker) for _ in range(32)]
for thread in threads:
    thread.start()

ready_group.wait()
run_event.set()

for thread in threads:
    thread.join()
"""

    env = dict(os.environ, PYTHON_GIL="0")
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    message_parts = [
        "raw regex subprocess run",
        f"return code: {result.returncode}",
    ]
    if stdout:
        message_parts.append(f"stdout:\n{stdout}")
    if stderr:
        message_parts.append(f"stderr:\n{stderr}")
    message = "\n".join(message_parts)

    if result.returncode == 0:
        pytest.fail("raw regex unexpectedly behaved thread-safely\n" + message)

    if result.returncode == -11:
        message += "\nProcess terminated with SIGSEGV (Segmentation fault)."

    if message:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()

    pytest.fail(message)
