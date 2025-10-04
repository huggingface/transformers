# Copyright 2025 ModelCloud.ai team and The HuggingFace Inc. team.
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

import sys
import threading

import pytest

from transformers.utils.safe import regex


pytestmark = pytest.mark.skipif(
    not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled(),
    reason="Safe regex test only runs when the GIL is disabled",
)


def _exercise_pattern(pattern_factory):
    expected = {"group0": "test123", "prefix": "test", "number": "123"}

    num_threads = 32
    errors = []
    errors_lock = threading.Lock()
    ready_group = threading.Barrier(num_threads + 1)
    run_event = threading.Event()

    def worker():
        try:
            ready_group.wait()
            run_event.wait()
            for _ in range(100):
                pattern = pattern_factory()
                match = pattern.match("test123")
                assert match is not None
                assert match.group(0) == expected["group0"]
                assert match.group("prefix") == expected["prefix"]
                assert match.group("number") == expected["number"]
        except Exception as exc:
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for thread in threads:
        thread.start()

    ready_group.wait()
    run_event.set()

    for thread in threads:
        thread.join()

    return errors


def test_regex_thread_safety_under_gil0():
    def factory():
        return regex.compile(r"(?P<prefix>test)(?P<number>\d+)")

    errors = _exercise_pattern(factory)
    assert not errors


def test_regex_thread_safety_shared_pattern_under_gil0():
    shared_pattern = regex.compile(r"(?P<prefix>test)(?P<number>\d+)")

    def factory():
        return shared_pattern

    errors = _exercise_pattern(factory)
    assert not errors


def test_regex_thread_safety_direct_match_under_gil0():
    def factory():
        class _DirectMatcher:
            def match(self, text):
                return regex.match(r"(?P<prefix>test)(?P<number>\d+)", text)

        return _DirectMatcher()

    errors = _exercise_pattern(factory)
    assert not errors
