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
import errno
import functools
import os
import re
import sys
import tempfile
import warnings
from os.path import abspath, dirname, join
from unittest import mock

import _pytest
import pytest

from transformers.testing_utils import (
    HfDoctestModule,
    HfDocTestParser,
    is_torch_available,
    patch_testing_methods_to_collect_info,
    patch_torch_compile_force_graph,
)
from transformers.utils import enable_tf32
from transformers.utils.network_logging import register_network_debug_plugin


_ci_fallback_cache_dir = None
# Directory holding one append-only file per process recording the repo/file id for every
# call retried through the read-only cache fallback. A directory (rather than an in-memory
# list) is used so the count survives `pytest-xdist`: tests run in worker subprocesses, but
# `pytest_terminal_summary` runs on the controller, which aggregates all the files. The path
# is shared with workers via an env var (they inherit it at spawn, like DISABLE_SAFETENSORS_CONVERSION).
_ci_fallback_events_dir = None


def _record_fallback_event(repo_id):
    """Append `repo_id` to this process's fallback-event file (xdist-safe, best-effort)."""
    if not _ci_fallback_events_dir:
        return
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    try:
        with open(os.path.join(_ci_fallback_events_dir, f"{worker}-{os.getpid()}.log"), "a", encoding="utf-8") as fh:
            fh.write(f"{repo_id}\n")
    except OSError:
        pass


def _collect_fallback_events():
    """Aggregate fallback events recorded by every process (controller + xdist workers)."""
    events = []
    if _ci_fallback_events_dir and os.path.isdir(_ci_fallback_events_dir):
        for name in sorted(os.listdir(_ci_fallback_events_dir)):
            try:
                with open(os.path.join(_ci_fallback_events_dir, name), encoding="utf-8") as fh:
                    events.extend(line.strip() for line in fh if line.strip())
            except OSError:
                pass
    return events


# Rust's `std::io::Error` renders OS-level failures as ``"<strerror> (os error <N>)"``
# where ``<N>`` is the raw platform errno. The trailing ``(os error N)`` fragment is
# emitted by Rust in English regardless of locale, so the numeric code -- not the
# human-readable ``<strerror>`` text -- is the stable signal to key off.
_OS_ERROR_CODE_RE = re.compile(r"os error (\d+)", re.IGNORECASE)


def _is_readonly_fs_error(e):
    """Return True if `e` signals a read-only filesystem (EROFS).

    The plain download path raises `OSError` with `errno == EROFS`. The Xet path
    (`hf_xet` Rust library) instead raises a bare `RuntimeError` with no `.errno` set,
    carrying a Rust-rendered message that ends with ``(os error <N>)``. We extract that
    numeric code and compare it to `errno.EROFS` rather than matching the locale-dependent
    strerror text, which is far more robust. The cause/context chain is followed so a
    wrapped error is still recognised.
    """
    # Follow the cause/context chain so a wrapped error is still recognised.
    while e is not None:
        # Standard path: an OSError carrying the EROFS errno.
        if isinstance(e, OSError) and e.errno == errno.EROFS:
            return True
        # hf_xet path: a bare RuntimeError with the raw OS errno rendered as "(os error N)".
        if isinstance(e, RuntimeError) and any(int(c) == errno.EROFS for c in _OS_ERROR_CODE_RE.findall(str(e))):
            return True
        e = e.__cause__ or e.__context__
    return False


def _with_tmpdir_cache_fallback(fn):
    """Decorator that retries `fn` with a writable tmp cache dir if it raises EROFS.

    In CI, the shared HF cache is read-only. Most models are pre-populated there and
    work fine. Only downloads of new or updated files fail with EROFS. On such failure,
    a session-scoped tmp dir is created once (via `tempfile.mkdtemp`, which is atomic
    and process-safe) and the call is retried with `cache_dir` set to the tmp dir.

    The retry also disables Xet (``HF_HUB_DISABLE_XET``) so the download takes the plain
    HTTP path, which writes into the redirected `cache_dir`. This is required because the
    Xet path (`hf_xet` Rust library) uses a process-wide session singleton that binds to
    the original read-only `HF_XET_CACHE` the first time it runs; redirecting `HF_XET_CACHE`
    on retry has no effect on that already-created session, so the write keeps failing.
    `HF_XET_CACHE` is still redirected too, as a belt-and-suspenders measure.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (OSError, RuntimeError) as e:
            if not _is_readonly_fs_error(e):
                raise
            global _ci_fallback_cache_dir
            if _ci_fallback_cache_dir is None:
                _ci_fallback_cache_dir = tempfile.mkdtemp(prefix="ci_fallback_tmpdir_cache_dir_")
            # `cached_files` names it `path_or_repo_id`; `hf_hub_download` names it `repo_id`.
            repo_id = kwargs.get("path_or_repo_id") or kwargs.get("repo_id") or (args[0] if args else "?")
            _record_fallback_event(str(repo_id))
            # Emit a marker as well; captured per-test by pytest, but the guaranteed-visible
            # count is reported in `pytest_terminal_summary` at the end of the run.
            print(
                f"[CI_CACHE_FALLBACK] read-only cache hit for {repo_id!r} ({type(e).__name__}); "
                "retrying via writable tmp cache_dir with Xet disabled",
                file=sys.stderr,
                flush=True,
            )
            import huggingface_hub.constants as hf_constants

            with (
                mock.patch.object(hf_constants, "HF_HUB_DISABLE_XET", True),
                mock.patch.dict(os.environ, {"HF_HUB_DISABLE_XET": "1"}),
                mock.patch.object(hf_constants, "HF_XET_CACHE", _ci_fallback_cache_dir),
                mock.patch.dict(os.environ, {"HF_XET_CACHE": _ci_fallback_cache_dir}),
            ):
                return fn(*args, **{**kwargs, "cache_dir": _ci_fallback_cache_dir})

    return wrapper


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
    "test_forward_signature",
    "test_model_get_set_embeddings",
    "test_model_main_input_name",
    "test_correct_missing_keys",
    "test_can_use_safetensors",
    "test_load_save_without_tied_weights",
    "test_tied_weights_keys",
    "test_model_weights_reload_no_missing_tied_weights",
    "test_can_load_ignoring_mismatched_shapes",
    "test_model_is_small",
    "ModelTest::test_pipeline_",  # None of the pipeline tests from PipelineTesterMixin (of which XxxModelTest inherits from) are running on device
    "ModelTester::test_pipeline_",
    "/repo_utils/",
    "/utils/",
    "/conftest_tests/",
}

# allow having multiple repository checkouts and not needing to remember to rerun
# `pip install -e '.[dev]'` when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(__file__), "src"))
sys.path.insert(1, git_repo_path)

# silence FutureWarning warnings in tests since often we can't act on them until
# they become normal warnings - i.e. the tests still need to test the current functionality
warnings.simplefilter(action="ignore", category=FutureWarning)


def pytest_configure(config):
    # Shared directory for the read-only cache fallback events. The controller creates it and
    # exports the path; xdist workers (spawned later) inherit the env var and write into the
    # same directory, so `pytest_terminal_summary` on the controller can aggregate all of them.
    global _ci_fallback_events_dir
    _ci_fallback_events_dir = os.environ.get("CI_FALLBACK_EVENTS_DIR")
    if _ci_fallback_events_dir is None:
        _ci_fallback_events_dir = tempfile.mkdtemp(prefix="ci_fallback_events_")
        os.environ["CI_FALLBACK_EVENTS_DIR"] = _ci_fallback_events_dir

    import transformers.utils.hub as _hub

    _hub.cached_files = _with_tmpdir_cache_fallback(_hub.cached_files)

    # `cached_files` is transformers' entry point, but libraries such as PEFT call
    # `huggingface_hub.hf_hub_download` directly (e.g. PeftConfig.from_pretrained fetching
    # adapter_config.json), bypassing it. Wrap that too and re-bind it wherever it has already
    # been imported via `from huggingface_hub import hf_hub_download`; modules imported later
    # (peft.config is loaded lazily during the test) pick up the wrapped version automatically.
    import huggingface_hub

    _original_hf_hub_download = huggingface_hub.hf_hub_download
    if not getattr(_original_hf_hub_download, "_ci_fallback_wrapped", False):
        _wrapped_hf_hub_download = _with_tmpdir_cache_fallback(_original_hf_hub_download)
        _wrapped_hf_hub_download._ci_fallback_wrapped = True
        huggingface_hub.hf_hub_download = _wrapped_hf_hub_download
        # Re-bind already-imported *third-party* consumers (e.g. peft). Skip transformers and
        # huggingface_hub themselves: transformers funnels downloads through the already-wrapped
        # `cached_files`, and huggingface_hub's own `snapshot_download` calls `hf_hub_download`
        # internally -- wrapping either inner call would redirect only the download to the tmp dir
        # while the caller still resolves the file against the original (read-only) cache_dir,
        # producing a spurious "missing file". Consumers imported later (peft.config is loaded
        # lazily during the test) pick up the wrapped function via the package attribute above.
        for _mod in list(sys.modules.values()):
            _name = getattr(_mod, "__name__", "") or ""
            if _name.split(".", 1)[0] in ("transformers", "huggingface_hub"):
                continue
            if getattr(_mod, "hf_hub_download", None) is _original_hf_hub_download:
                _mod.hf_hub_download = _wrapped_hf_hub_download

    config.addinivalue_line("markers", "is_pipeline_test: mark test to run only when pipelines are tested")
    config.addinivalue_line("markers", "is_staging_test: mark test to run only in the staging environment")
    config.addinivalue_line("markers", "accelerate_tests: mark test that require accelerate")
    config.addinivalue_line("markers", "not_device_test: mark the tests always running on cpu")
    config.addinivalue_line("markers", "torch_compile_test: mark test which tests torch compile functionality")
    config.addinivalue_line("markers", "torch_export_test: mark test which tests torch export functionality")
    config.addinivalue_line("markers", "flash_attn_test: mark test which tests flash attention functionality")
    config.addinivalue_line("markers", "flash_attn_3_test: mark test which tests flash attention 3 functionality")
    config.addinivalue_line("markers", "flash_attn_4_test: mark test which tests flash attention 4 functionality")
    config.addinivalue_line(
        "markers", "all_flash_attn_test: mark test which tests all mainline flash attentions' functionality"
    )
    config.addinivalue_line("markers", "training_ci: mark test for training CI validation")
    config.addinivalue_line("markers", "tensor_parallel_ci: mark test for tensor parallel CI validation")

    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "true"
    register_network_debug_plugin(config)


def pytest_collection_modifyitems(items):
    for item in items:
        if any(test_name in item.nodeid for test_name in NOT_DEVICE_TESTS):
            item.add_marker(pytest.mark.not_device_test)


def pytest_addoption(parser):
    from transformers.testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)


def pytest_runtest_logreport(report):
    if report.when == "call":
        outcome = "PASSED" if report.passed else "FAILED" if report.failed else "SKIPPED"
        print(f"{report.nodeid} [{outcome}] {report.duration:.2f}s")


def pytest_terminal_summary(terminalreporter):
    # Always report whether the read-only cache fallback fired, so CI logs give an
    # unambiguous signal that the fallback path was (or was not) exercised this run.
    # Events are aggregated across processes (xdist workers write to a shared directory).
    events = _collect_fallback_events()
    if events:
        from collections import Counter

        terminalreporter.write_sep("=", "CI READ-ONLY CACHE FALLBACK", yellow=True, bold=True)
        terminalreporter.write_line(f"Read-only cache fallback fired {len(events)} time(s):")
        for repo_id, count in sorted(Counter(events).items()):
            terminalreporter.write_line(f"  - {repo_id} (x{count})")
    else:
        terminalreporter.write_sep("=", "CI READ-ONLY CACHE FALLBACK: not triggered", bold=True)

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
    # # torch.backends.fp32_precision does not cascade to torch.backends.cudnn.conv.fp32_precision and torch.backends.cudnn.rnn.fp32_precision
    # TODO: Considering move this to `enable_tf32`, or report a bug to `torch`.
    import torch

    # In order to set `torch.backends.cudnn.conv.fp32_precision = "ieee"` below (new API), we still need to set this
    # (old API) because it defaults to `True` (and not changed automatically when we change `cudnn.conv.fp32_precision`)
    # and such inconsistency cause `torch` to complain `RuntimeError: PyTorch is checking whether allow_tf32 is enabled for cuDNN without a specific operator name,but the current flag(s) indica
    # te that cuDNN conv and cuDNN RNN have different TF32 flags.This combination indicates that you have used a mix of the legacy and new APIs
    #  to set the TF32 flags. We suggest only using the new API to set the TF32 flag(s).`.
    # TODO: report a bug to `torch`
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    # We set it to `False` for CI. See https://github.com/pytorch/pytorch/issues/157274#issuecomment-3090791615
    enable_tf32(False)

    # This is necessary to make several `test_batching_equivalence` pass (within the tolerance `1e-5`)
    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        torch.backends.cudnn.conv.fp32_precision = "ieee"

    # patch `torch.compile`: if `TORCH_COMPILE_FORCE_FULLGRAPH=1` (or values considered as true, e.g. yes, y, etc.),
    # the patched version will always run with `fullgraph=True`.
    patch_torch_compile_force_graph()


if os.environ.get("PATCH_TESTING_METHODS_TO_COLLECT_OUTPUTS", "").lower() in ("yes", "true", "on", "y", "1"):
    patch_testing_methods_to_collect_info()
