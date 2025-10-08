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
import textwrap
from typing import Tuple

import pytest


pytestmark = pytest.mark.skipif(
    not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled(),
    reason="Crash regression only runs when the GIL is disabled",
)


def _thread_stress_script(*, imports: str, setup_code: str) -> str:
    setup_block = textwrap.indent(
        textwrap.dedent(setup_code).strip(),
        " " * 8,
    )
    return textwrap.dedent(
        f"""
        import threading
        {imports}

        pattern_text = r"(?P<prefix>test)(?P<number>\\d+)"
        text_to_match = "test123"
        ready_group = threading.Barrier(33)
        run_event = threading.Event()

{setup_block}

        def worker():
            ready_group.wait()
            run_event.wait()
            for _ in range(100):
                match = match_once()
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
    )


def _run_thread_stress_script(label: str, script: str) -> Tuple[subprocess.CompletedProcess, str]:
    env = dict(os.environ, PYTHON_GIL="0")
    result = subprocess.run(  # noqa: PLW1510 intentionally check return code manually
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
    )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    message_parts = [
        f"{label} subprocess run",
        f"return code: {result.returncode}",
    ]
    if stdout:
        message_parts.append(f"stdout:\n{stdout}")
    if stderr:
        message_parts.append(f"stderr:\n{stderr}")
    return result, "\n".join(message_parts)


@pytest.mark.xfail(strict=False, reason="Raw regex crashes under PYTHON_GIL=0")
def test_raw_regex_thread_safety_crashes_under_gil0():
    script = _thread_stress_script(
        imports="import regex",
        setup_code="""
        def match_once():
            return regex.match(pattern_text, text_to_match)
        """,
    )

    result, message = _run_thread_stress_script("raw regex", script)

    if result.returncode == 0:
        pytest.fail("raw regex unexpectedly behaved thread-safely\n" + message)

    if result.returncode == -11:
        message += "\nProcess terminated with SIGSEGV (Segmentation fault)."

    if message:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()

    pytest.fail(message)


@pytest.mark.parametrize(
    "label, imports, setup_code",
    (
        (
            "stdlib re",
            "import re",
            """
            def match_once():
                return re.match(pattern_text, text_to_match)
            """,
        ),
        (
            "pcre2",
            "import pcre2",
            """
            compiled = pcre2.compile(pattern_text, jit=False)

            def match_once():
                return compiled.match(text_to_match)
            """,
        ),
        (
            "pcre2 (jit)",
            "import pcre2",
            """
            compiled = pcre2.compile(pattern_text, jit=True)

            def match_once():
                return compiled.match(text_to_match)
            """,
        ),
    ),
)

# re.match is thread safe even with internal caching compiled regex
# pcre does not have caching so it is thread safe
def test_threaded_engine_regressions(label: str, imports: str, setup_code: str):
    if "pcre2" in imports:
        pytest.importorskip("pcre2")

    script = _thread_stress_script(imports=imports, setup_code=setup_code)
    result, message = _run_thread_stress_script(label, script)

    if result.returncode != 0:
        pytest.fail(f"{label} thread stress failed, return code = {result}\n " + message)
