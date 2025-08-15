# Copyright 2020 The HuggingFace Team. All rights reserved.
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

# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

import doctest
import sys
import warnings
from os.path import abspath, dirname, join

import _pytest
import pytest

from transformers.testing_utils import HfDoctestModule, HfDocTestParser, is_torch_available


NOT_DEVICE_TESTS = {
    "test_tokenization",
    "test_tokenization_mistral_common",
    "test_processing",
    "test_beam_constraints",
    "test_configuration_utils",
    "test_data_collator",
    "test_trainer_callback",
    "test_trainer_utils",
    "test_feature_extraction",
    "test_image_processing",
    "test_image_processor",
    "test_image_transforms",
    "test_optimization",
    "test_retrieval",
    "test_config",
    "test_from_pretrained_no_checkpoint",
    "test_keep_in_fp32_modules",
    "test_gradient_checkpointing_backward_compatibility",
    "test_gradient_checkpointing_enable_disable",
    "test_torch_save_load",
    "test_initialization",
    "test_forward_signature",
    "test_model_get_set_embeddings",
    "test_model_main_input_name",
    "test_correct_missing_keys",
    "test_tie_model_weights",
    "test_can_use_safetensors",
    "test_load_save_without_tied_weights",
    "test_tied_weights_keys",
    "test_model_weights_reload_no_missing_tied_weights",
    "test_mismatched_shapes_have_properly_initialized_weights",
    "test_matched_shapes_have_loaded_weights_when_some_mismatched_shapes_exist",
    "test_model_is_small",
    "test_tf_from_pt_safetensors",
    "test_flax_from_pt_safetensors",
    "ModelTest::test_pipeline_",  # None of the pipeline tests from PipelineTesterMixin (of which XxxModelTest inherits from) are running on device
    "ModelTester::test_pipeline_",
    "/repo_utils/",
    "/utils/",
}

# allow having multiple repository checkouts and not needing to remember to rerun
# `pip install -e '.[dev]'` when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(__file__), "src"))
sys.path.insert(1, git_repo_path)

# silence FutureWarning warnings in tests since often we can't act on them until
# they become normal warnings - i.e. the tests still need to test the current functionality
warnings.simplefilter(action="ignore", category=FutureWarning)


def pytest_configure(config):
    config.addinivalue_line("markers", "is_pipeline_test: mark test to run only when pipelines are tested")
    config.addinivalue_line("markers", "is_staging_test: mark test to run only in the staging environment")
    config.addinivalue_line("markers", "accelerate_tests: mark test that require accelerate")
    config.addinivalue_line("markers", "not_device_test: mark the tests always running on cpu")
    config.addinivalue_line("markers", "torch_compile_test: mark test which tests torch compile functionality")
    config.addinivalue_line("markers", "torch_export_test: mark test which tests torch export functionality")


def pytest_collection_modifyitems(items):
    for item in items:
        if any(test_name in item.nodeid for test_name in NOT_DEVICE_TESTS):
            item.add_marker(pytest.mark.not_device_test)


def pytest_addoption(parser):
    from transformers.testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)


def pytest_terminal_summary(terminalreporter):
    from transformers.testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)


def pytest_sessionfinish(session, exitstatus):
    # If no tests are collected, pytest exists with code 5, which makes the CI fail.
    if exitstatus == 5:
        session.exitstatus = 0


# Doctest custom flag to ignore output.
IGNORE_RESULT = doctest.register_optionflag("IGNORE_RESULT")

OutputChecker = doctest.OutputChecker


class CustomOutputChecker(OutputChecker):
    def check_output(self, want, got, optionflags):
        if IGNORE_RESULT & optionflags:
            return True
        return OutputChecker.check_output(self, want, got, optionflags)


doctest.OutputChecker = CustomOutputChecker
_pytest.doctest.DoctestModule = HfDoctestModule
doctest.DocTestParser = HfDocTestParser

if is_torch_available():
    import torch

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    # We set it to `False` for CI. See https://github.com/pytorch/pytorch/issues/157274#issuecomment-3090791615
    torch.backends.cudnn.allow_tf32 = False


import faulthandler
import signal
import sys
import threading
import time

import subprocess
import os
import sys
import threading
import time


def dump_cpp_stack():
    pid = os.getpid()
    try:
        # Create a gdb script
        gdb_script = f"""
set pagination off
set logging file /tmp/gdb_trace_{pid}.txt
set logging on
attach {pid}
thread apply all bt
py-bt
detach
quit
"""

        with open(f'/tmp/gdb_script_{pid}.gdb', 'w') as f:
            f.write(gdb_script)

        # Run gdb non-interactively
        result = subprocess.run([
            'gdb', '-batch', '-x', f'/tmp/gdb_script_{pid}.gdb'
        ], capture_output=True, text=True, timeout=30)

        # Print the output
        if os.path.exists(f'/tmp/gdb_trace_{pid}.txt'):
            with open(f'/tmp/gdb_trace_{pid}.txt', 'r') as f:
                print("=== GDB C++ Stack Trace ===", file=sys.stderr)
                print(f.read(), file=sys.stderr)

        print("=== GDB stderr ===", file=sys.stderr)
        print(result.stderr, file=sys.stderr)

    except Exception as e:
        print(f"Failed to get gdb trace: {e}", file=sys.stderr)


def dump_stacks_and_exit():
    print("\n=== TIMEOUT: Dumping all stacks ===", file=sys.stderr)
    faulthandler.dump_traceback(sys.stderr, all_threads=True)
    print("=== Active threads ===", file=sys.stderr)
    for thread in threading.enumerate():
        print(f"Thread: {thread.name}, daemon: {thread.daemon}, alive: {thread.is_alive()}", file=sys.stderr)

    # dump_cpp_stack()

    sys.stderr.flush()

    # Force exit after showing debug info
    import os
    os._exit(1)


def setup_timeout_debug(timeout_seconds=600):  # 5 minutes
    def timeout_handler1():
        time.sleep(timeout_seconds)
        dump_stacks_and_exit()

    timeout_thread = threading.Thread(target=timeout_handler1, daemon=True)
    timeout_thread.start()








# In conftest.py

# In your conftest.py or at the start of your test
faulthandler.enable()
setup_timeout_debug(480)  # Adjust timeout as needed

import atexit
import threading
import time
import sys
import ctypes


class ThreadTimeoutCleanup:
    def __init__(self, cleanup_timeout=30):
        self.cleanup_timeout = cleanup_timeout
        self.hook_cleanup_registration()

    def hook_cleanup_registration(self):
        original_register = atexit.register

        def timeout_register(func, *args, **kwargs):
            module = getattr(func, '__module__', 'unknown')
            name = getattr(func, '__name__', str(func))
            func_id = f"{module}.{name}"

            print(f"CLEANUP REGISTERED: {func_id}", file=sys.stderr)

            def timeout_wrapper(*wrapper_args, **wrapper_kwargs):
                print(f"CLEANUP START: {func_id}", file=sys.stderr)
                start_time = time.time()

                # Create a container for the result
                result_container = {'result': None, 'exception': None, 'completed': False}

                def run_cleanup():
                    try:
                        result_container['result'] = func(*wrapper_args, **wrapper_kwargs)
                        result_container['completed'] = True
                    except Exception as e:
                        result_container['exception'] = e
                        result_container['completed'] = True

                # Run cleanup in a separate thread
                cleanup_thread = threading.Thread(target=run_cleanup)
                cleanup_thread.daemon = True  # Don't block exit
                cleanup_thread.start()

                # Wait for completion or timeout
                cleanup_thread.join(timeout=self.cleanup_timeout)

                elapsed = time.time() - start_time

                if cleanup_thread.is_alive():
                    # Cleanup is hanging
                    print(f"CLEANUP HANGING: {func_id} ({elapsed:.1f}s) - abandoning and continuing", file=sys.stderr)

                    # Dump stack trace
                    import faulthandler
                    faulthandler.dump_traceback(sys.stderr, all_threads=True)

                    print(f"CLEANUP ABANDONED: {func_id} - thread left running", file=sys.stderr)
                    return None  # Continue to next cleanup

                elif result_container['completed']:
                    if result_container['exception']:
                        print(f"CLEANUP FAILED: {func_id} ({elapsed:.3f}s) - {result_container['exception']}",
                              file=sys.stderr)
                    else:
                        print(f"CLEANUP DONE: {func_id} ({elapsed:.3f}s)", file=sys.stderr)
                    return result_container['result']

                else:
                    print(f"CLEANUP UNKNOWN STATE: {func_id} ({elapsed:.3f}s)", file=sys.stderr)
                    return None

            return original_register(timeout_wrapper, *args, **kwargs)

        atexit.register = timeout_register


# Use it:
thread_timeout_cleanup = ThreadTimeoutCleanup(cleanup_timeout=120)

import atexit
import threading
import time
import sys
import signal
import faulthandler
import gc
import os


class ComprehensiveExitDebugger:
    def __init__(self, cleanup_timeout=30, post_atexit_timeout=60):
        self.cleanup_timeout = cleanup_timeout
        self.post_atexit_timeout = post_atexit_timeout
        self.cleanup_count = 0
        self.skipped_cleanups = []
        self.exit_start_time = None

        # Enable faulthandler
        faulthandler.enable()

        # Hook atexit cleanup functions
        self.hook_cleanup_registration()

        # Hook sys.exit to monitor post-atexit hangs
        self.hook_system_exit()

        # Register our own final monitoring function
        atexit.register(self.final_exit_monitoring)

        print("=== Comprehensive Exit Debugger Initialized ===", file=sys.stderr)
        print(f"Cleanup timeout: {cleanup_timeout}s, Post-atexit timeout: {post_atexit_timeout}s", file=sys.stderr)

    def hook_cleanup_registration(self):
        """Hook atexit.register to monitor individual cleanup functions"""
        original_register = atexit.register

        def monitored_register(func, *args, **kwargs):
            module = getattr(func, '__module__', 'unknown')
            name = getattr(func, '__name__', str(func))
            func_id = f"{module}.{name}"

            # Skip known problematic cleanups
            if any(bad in func_id.lower() for bad in ['threadpoolexecutor', 'concurrent.futures']):
                print(f"SKIPPING KNOWN HANGER: {func_id}", file=sys.stderr)
                self.skipped_cleanups.append(func_id)
                return lambda: None

            print(f"CLEANUP REGISTERED: {func_id}", file=sys.stderr)

            def monitored_wrapper(*wrapper_args, **wrapper_kwargs):
                self.cleanup_count += 1
                print(f"CLEANUP START: {func_id}", file=sys.stderr)
                start_time = time.time()

                # Run cleanup with timeout in separate thread
                completed = threading.Event()
                result = [None]
                exception = [None]

                def run_cleanup():
                    try:
                        result[0] = func(*wrapper_args, **wrapper_kwargs)
                    except Exception as e:
                        exception[0] = e
                    finally:
                        completed.set()

                thread = threading.Thread(target=run_cleanup, daemon=True)
                thread.start()

                if completed.wait(timeout=self.cleanup_timeout):
                    elapsed = time.time() - start_time
                    if exception[0]:
                        print(f"CLEANUP FAILED: {func_id} ({elapsed:.3f}s) - {exception[0]}", file=sys.stderr)
                    else:
                        print(f"CLEANUP DONE: {func_id} ({elapsed:.3f}s)", file=sys.stderr)
                    return result[0]
                else:
                    elapsed = time.time() - start_time
                    print(f"CLEANUP TIMEOUT: {func_id} ({elapsed:.1f}s) - skipping and continuing", file=sys.stderr)
                    self.skipped_cleanups.append(func_id)

                    # Dump stack trace for hanging cleanup
                    print(f"  Stack trace of hanging cleanup:", file=sys.stderr)
                    faulthandler.dump_traceback(sys.stderr, all_threads=True)
                    return None

            return original_register(monitored_wrapper, *args, **kwargs)

        atexit.register = monitored_register

    def hook_system_exit(self):
        """Hook sys.exit to start post-atexit monitoring"""
        original_exit = sys.exit
        original_excepthook = sys.excepthook

        def monitored_exit(code=0):
            self.exit_start_time = time.time()
            print(f"\n=== sys.exit({code}) called ===", file=sys.stderr)

            # Start post-atexit monitoring thread
            def post_atexit_monitor():
                time.sleep(self.post_atexit_timeout)
                print(f"\n=== POST-ATEXIT HANG DETECTED ===", file=sys.stderr)
                print("All atexit cleanups completed, but process still hanging", file=sys.stderr)
                self.analyze_hang_causes()
                print("=== FORCE EXIT ===", file=sys.stderr)
                os._exit(1)

            monitor = threading.Thread(target=post_atexit_monitor, daemon=True)
            monitor.start()

            # Call original exit
            return original_exit(code)

        def monitored_excepthook(exc_type, exc_value, exc_traceback):
            if exc_type == SystemExit:
                print(f"=== SystemExit in excepthook: {exc_value} ===", file=sys.stderr)
            return original_excepthook(exc_type, exc_value, exc_traceback)

        sys.exit = monitored_exit
        sys.excepthook = monitored_excepthook

    def final_exit_monitoring(self):
        """Final monitoring function - runs as last atexit"""
        print(f"\n=== FINAL EXIT MONITORING START ===", file=sys.stderr)
        print(f"Completed {self.cleanup_count} cleanups", file=sys.stderr)
        if self.skipped_cleanups:
            print(f"Skipped cleanups: {self.skipped_cleanups}", file=sys.stderr)

        # Start detailed monitoring
        self.monitor_exit_process()

    def monitor_exit_process(self):
        """Monitor the exit process in detail"""
        start_time = time.time()

        # Monitor for up to post_atexit_timeout seconds
        for i in range(self.post_atexit_timeout // 5):
            time.sleep(5)
            elapsed = time.time() - start_time

            print(f"\n--- Exit monitoring at {elapsed:.1f}s ---", file=sys.stderr)

            # 1. Check threads
            non_daemon_threads = self.analyze_threads()

            # 2. If no threads blocking, check other causes
            if not non_daemon_threads:
                print("No non-daemon threads found - checking other causes", file=sys.stderr)
                self.analyze_other_causes()
                break
            else:
                print(f"Still waiting for {len(non_daemon_threads)} non-daemon threads", file=sys.stderr)

        # If we complete the loop, something else is hanging
        final_elapsed = time.time() - start_time
        if final_elapsed >= self.post_atexit_timeout - 5:
            print(f"\n=== TIMEOUT after {final_elapsed:.1f}s ===", file=sys.stderr)
            self.analyze_hang_causes()

    def analyze_threads(self):
        """Analyze thread status"""
        print("  Thread analysis:", file=sys.stderr)
        main_thread = threading.main_thread()
        all_threads = threading.enumerate()
        non_daemon_threads = []

        for thread in all_threads:
            is_main = thread == main_thread
            is_non_daemon = not thread.daemon and not is_main and thread.is_alive()

            if is_non_daemon:
                non_daemon_threads.append(thread)
                print(f"    NON-DAEMON: {thread.name} (ident: {thread.ident})", file=sys.stderr)

                # Try to get more info about the thread
                if hasattr(thread, '_target') and thread._target:
                    target_name = getattr(thread._target, '__name__', str(thread._target))
                    print(f"      Target: {target_name}", file=sys.stderr)
            elif not is_main:
                print(f"    daemon: {thread.name}", file=sys.stderr)

        print(f"    Total threads: {len(all_threads)}, Non-daemon: {len(non_daemon_threads)}", file=sys.stderr)
        return non_daemon_threads

    def analyze_other_causes(self):
        """Analyze other potential hang causes"""
        print("  Other analysis:", file=sys.stderr)

        # Garbage collection
        print(f"    GC enabled: {gc.isenabled()}, counts: {gc.get_count()}", file=sys.stderr)

        # Objects with finalizers
        try:
            finalizers = [obj for obj in gc.get_objects() if hasattr(obj, '__del__')]
            print(f"    Objects with __del__: {len(finalizers)}", file=sys.stderr)
        except:
            print("    Could not count finalizers", file=sys.stderr)

        # Loaded modules that might cause issues
        problematic_modules = ['multiprocessing', 'concurrent.futures', 'threading',
                               'asyncio', 'subprocess', 'queue', 'signal']
        loaded_problematic = [name for name in problematic_modules if name in sys.modules]
        if loaded_problematic:
            print(f"    Problematic modules loaded: {loaded_problematic}", file=sys.stderr)

        # File descriptors (Linux/Mac only)
        try:
            fd_count = len(os.listdir('/proc/self/fd'))
            print(f"    Open file descriptors: {fd_count}", file=sys.stderr)

            self.analyze_file_descriptors()

        except:
            pass

    def analyze_hang_causes(self):
        """Comprehensive analysis when hang is detected"""
        print("\n=== COMPREHENSIVE HANG ANALYSIS ===", file=sys.stderr)

        # 1. Thread stacks
        print("\n--- ALL THREAD STACKS ---", file=sys.stderr)
        faulthandler.dump_traceback(sys.stderr, all_threads=True)

        # 2. Detailed thread analysis
        print("\n--- DETAILED THREAD ANALYSIS ---", file=sys.stderr)
        self.analyze_threads()

        # 3. Other causes
        print("\n--- OTHER POTENTIAL CAUSES ---", file=sys.stderr)
        self.analyze_other_causes()

        # 4. Summary
        print(f"\n--- SUMMARY ---", file=sys.stderr)
        if self.exit_start_time:
            total_time = time.time() - self.exit_start_time
            print(f"Total exit time: {total_time:.1f}s", file=sys.stderr)
        print(f"Completed cleanups: {self.cleanup_count}", file=sys.stderr)
        print(f"Skipped cleanups: {len(self.skipped_cleanups)}", file=sys.stderr)
        if self.skipped_cleanups:
            print(f"Skipped: {self.skipped_cleanups}", file=sys.stderr)

        sys.stderr.flush()


    def analyze_file_descriptors(self):
        """Simple FD analysis using /proc"""
        print("  File descriptors:", file=sys.stderr)

        try:
            import os
            fd_dir = '/proc/self/fd'

            if not os.path.exists(fd_dir):
                print("    /proc/self/fd not available (not Linux)", file=sys.stderr)
                return

            fds = sorted(os.listdir(fd_dir), key=lambda x: int(x) if x.isdigit() else 999)
            print(f"    Total: {len(fds)}", file=sys.stderr)

            categories = {
                'stdin/stdout/stderr': [],
                'regular_files': [],
                'pipes': [],
                'sockets': [],
                'other': []
            }

            for fd in fds:
                try:
                    fd_path = os.path.join(fd_dir, fd)
                    target = os.readlink(fd_path)

                    # Categorize
                    if fd in ['0', '1', '2']:
                        categories['stdin/stdout/stderr'].append(f"FD {fd}: {target}")
                    elif target.startswith('pipe:'):
                        categories['pipes'].append(f"FD {fd}: {target}")
                    elif target.startswith('socket:'):
                        categories['sockets'].append(f"FD {fd}: {target}")
                    elif target.startswith('/'):
                        categories['regular_files'].append(f"FD {fd}: {target}")
                    else:
                        categories['other'].append(f"FD {fd}: {target}")

                except (OSError, IOError):
                    categories['other'].append(f"FD {fd}: (cannot read)")

            # Print categorized results
            for category, items in categories.items():
                if items:
                    print(f"    {category}:", file=sys.stderr)
                    for item in items:
                        print(f"      {item}", file=sys.stderr)

        except Exception as e:
            print(f"    FD analysis failed: {e}", file=sys.stderr)


# One-line setup for conftest.py:
exit_debugger = ComprehensiveExitDebugger(cleanup_timeout=120, post_atexit_timeout=180)

import gc
import sys
import weakref
import atexit
import time
import threading
import os

import gc
import sys
import weakref
import atexit
import time
import threading


class NonThreadingHangDebugger:
    def __init__(self):
        self.module_refs = weakref.WeakSet()
        self.large_objects = []

        # Track module imports
        self.original_import = __builtins__.__import__
        __builtins__.__import__ = self.tracked_import

        atexit.register(self.debug_non_threading_hang)

    def tracked_import(self, name, *args, **kwargs):
        """Track imported modules"""
        module = self.original_import(name, *args, **kwargs)
        if hasattr(module, '__file__') and module.__file__:
            if any(pattern in module.__file__ for pattern in ['.so', 'torch', 'numpy']):
                print(f"Imported C extension: {name} from {module.__file__}", file=sys.stderr)
        return module

    def debug_non_threading_hang(self):
        """Debug non-threading hang causes"""
        print("\n=== NON-THREADING HANG DEBUG ===", file=sys.stderr)

        # 1. Check for objects with finalizers
        self.check_finalizers()

        # 2. Check garbage collection state
        self.check_gc_state()

        # 3. Check loaded C extensions
        self.check_c_extensions()

        # 4. Force garbage collection and see what happens
        self.force_gc_analysis()

    def check_finalizers(self):
        """Check for objects with __del__ methods"""
        print("=== FINALIZER CHECK ===", file=sys.stderr)

        try:
            # Safer approach - just count types
            finalizer_types = {}
            obj_count = 0

            for obj in gc.get_objects():
                obj_count += 1
                if obj_count > 10000:  # Limit to prevent hanging here
                    break

                if hasattr(obj, '__del__'):
                    obj_type = type(obj).__name__
                    module = getattr(type(obj), '__module__', 'unknown')
                    key = f"{module}.{obj_type}"
                    finalizer_types[key] = finalizer_types.get(key, 0) + 1

            print(f"  Checked {obj_count} objects", file=sys.stderr)
            print(f"  Objects with finalizers by type:", file=sys.stderr)

            # Show top finalizer types
            for obj_type, count in sorted(finalizer_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"    {obj_type}: {count}", file=sys.stderr)

        except Exception as e:
            print(f"  Finalizer check failed: {e}", file=sys.stderr)

    def check_gc_state(self):
        """Check garbage collection state"""
        print("=== GC STATE CHECK ===", file=sys.stderr)

        print(f"  GC enabled: {gc.isenabled()}", file=sys.stderr)
        print(f"  GC counts: {gc.get_count()}", file=sys.stderr)
        print(f"  GC thresholds: {gc.get_threshold()}", file=sys.stderr)

        # Check for uncollectable garbage
        uncollectable = len(gc.garbage)
        print(f"  Uncollectable objects: {uncollectable}", file=sys.stderr)

        if uncollectable > 0:
            print("  Uncollectable object types:", file=sys.stderr)
            type_counts = {}
            for obj in gc.garbage[:20]:  # First 20
                obj_type = type(obj).__name__
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

            for obj_type, count in type_counts.items():
                print(f"    {obj_type}: {count}", file=sys.stderr)

    def check_c_extensions(self):
        """Check loaded C extensions"""
        print("=== C EXTENSION CHECK ===", file=sys.stderr)

        c_extensions = []
        for name, module in sys.modules.items():
            if hasattr(module, '__file__') and module.__file__:
                if module.__file__.endswith('.so') or module.__file__.endswith('.pyd'):
                    c_extensions.append((name, module.__file__))

        print(f"  Loaded C extensions: {len(c_extensions)}", file=sys.stderr)

        # Look for PyTorch related extensions
        torch_extensions = [ext for ext in c_extensions if 'torch' in ext[0].lower() or 'torch' in ext[1].lower()]
        if torch_extensions:
            print("  PyTorch C extensions:", file=sys.stderr)
            for name, path in torch_extensions:
                print(f"    {name}: {path}", file=sys.stderr)

    def force_gc_analysis(self):
        """Force GC and analyze what takes time"""
        print("=== FORCED GC ANALYSIS ===", file=sys.stderr)

        # Multiple GC passes to see if anything is stuck
        for i in range(3):
            start_time = time.time()
            try:
                collected = gc.collect()
                elapsed = time.time() - start_time
                print(f"  GC pass {i + 1}: collected {collected} objects in {elapsed:.4f}s", file=sys.stderr)

                if elapsed > 1.0:
                    print(f"    ^ GC pass {i + 1} took unusually long!", file=sys.stderr)

            except Exception as e:
                elapsed = time.time() - start_time
                print(f"  GC pass {i + 1} failed after {elapsed:.4f}s: {e}", file=sys.stderr)

        # Check if disabling GC helps identify the issue
        try:
            gc.disable()
            print("  GC disabled for testing", file=sys.stderr)
            time.sleep(1)
            gc.enable()
            print("  GC re-enabled", file=sys.stderr)
        except Exception as e:
            print(f"  GC disable/enable test failed: {e}", file=sys.stderr)


# Use this:
non_threading_debugger = NonThreadingHangDebugger()