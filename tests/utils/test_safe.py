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

import builtins
import sys
import threading
import types

import pytest
import regex as _regex

from transformers.utils.safe import ThreadSafe, regex


requires_no_gil = pytest.mark.skipif(
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

    ready_group.wait()  # Ensure every thread is ready before triggering execution
    run_event.set()

    for thread in threads:
        thread.join()

    return errors


@requires_no_gil
def test_regex_thread_safety_under_gil0():
    def factory():
        return regex.compile(r"(?P<prefix>test)(?P<number>\d+)")

    errors = _exercise_pattern(factory)
    assert not errors


@requires_no_gil
def test_regex_thread_safety_shared_pattern_under_gil0():
    shared_pattern = regex.compile(r"(?P<prefix>test)(?P<number>\d+)")

    def factory():
        return shared_pattern

    errors = _exercise_pattern(factory)
    assert not errors


@requires_no_gil
def test_regex_thread_safety_direct_match_under_gil0():
    def factory():
        class _DirectMatcher:
            def match(self, text):
                return regex.match(r"(?P<prefix>test)(?P<number>\d+)", text)

        return _DirectMatcher()

    errors = _exercise_pattern(factory)
    assert not errors


@requires_no_gil
def test_regex_threadsafe_allows_reentrant_calls_under_gil0():
    pattern = regex.compile(r"(?P<prefix>test)(?P<number>\d+)")
    completed = threading.Event()

    def target():
        def replace(match):
            inner = regex.match(r"(?P<prefix>test)(?P<number>\d+)", match.group(0))
            return inner.group(0)

        result = pattern.sub(replace, "test123")
        assert result == "test123"
        completed.set()

    worker = threading.Thread(target=target)
    worker.start()
    worker.join(timeout=2)

    assert completed.is_set(), "Re-entrant call should not deadlock"


@requires_no_gil
def test_threadsafe_callable_cache_is_serialized_under_gil0():
    module = types.ModuleType("_threadsafe_test_module")

    counter_lock = threading.Lock()
    call_counter = {"count": 0}

    def increment(value: int):
        with counter_lock:
            call_counter["count"] += 1
        return value + 1

    module.increment = increment
    thread_safe_module = ThreadSafe(module)

    num_threads = 32
    errors = []
    errors_lock = threading.Lock()
    ready_group = threading.Barrier(num_threads + 1)
    run_event = threading.Event()

    def worker():
        try:
            ready_group.wait()
            run_event.wait()
            for _ in range(256):
                func = thread_safe_module.increment
                assert func(1) == 2
        except Exception as exc:
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for thread in threads:
        thread.start()

    ready_group.wait()  # Synchronize thread start to mimic high contention scenarios
    run_event.set()

    for thread in threads:
        thread.join()

    assert not errors
    # Each invocation should be recorded; the exact count confirms every worker executed.
    assert call_counter["count"] == num_threads * 256


def test_regex_compile_returns_threadsafe_proxy():
    pattern = regex.compile(r"(?P<prefix>test)(?P<number>\d+)")
    assert hasattr(pattern, "__wrapped__")
    assert isinstance(pattern.__wrapped__, _regex.Pattern)
    assert pattern.match("test123").group(0) == "test123"


def test_regex_constructor_returns_threadsafe_proxy():
    pattern = regex.Regex(r"(?P<prefix>test)(?P<number>\d+)")
    assert hasattr(pattern, "__wrapped__")
    assert isinstance(pattern.__wrapped__, _regex.Pattern)
    assert pattern.fullmatch("test123").group("number") == "123"


def test_regex_shared_pattern_is_thread_safe_with_proxy():
    pattern = regex.compile(r"(?P<prefix>test)(?P<number>\d+)")
    ready_group = threading.Barrier(9)
    start_event = threading.Event()
    errors = []
    errors_lock = threading.Lock()

    def worker():
        try:
            ready_group.wait()
            start_event.wait()
            for _ in range(128):
                match = pattern.match("test123")
                assert match.group("prefix") == "test"
                assert match.group("number") == "123"
        except Exception as exc:
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()

    ready_group.wait()
    start_event.set()

    for thread in threads:
        thread.join()

    assert not errors


def test_threadsafe_copies_existing_module_metadata():
    thread_safe_regex = ThreadSafe(regex)

    for attr in ("__package__", "__file__", "__spec__"):
        if hasattr(regex, attr):
            assert hasattr(thread_safe_regex, attr)
            assert getattr(thread_safe_regex, attr) == getattr(regex, attr)


def test_threadsafe_skips_missing_module_metadata():
    thread_safe_builtins = ThreadSafe(builtins)

    for attr in ("__package__", "__file__", "__spec__"):
        if hasattr(builtins, attr):
            assert getattr(thread_safe_builtins, attr) == getattr(builtins, attr)
        else:
            assert not hasattr(thread_safe_builtins, attr)
