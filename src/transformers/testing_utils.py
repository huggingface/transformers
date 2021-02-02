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

import inspect
import logging
import os
import re
import shutil
import sys
import tempfile
import unittest
from distutils.util import strtobool
from io import StringIO
from pathlib import Path

from .file_utils import (
    is_datasets_available,
    is_faiss_available,
    is_flax_available,
    is_pandas_available,
    is_scatter_available,
    is_sentencepiece_available,
    is_soundfile_availble,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    is_torch_tpu_available,
)
from .integrations import is_optuna_available, is_ray_available


SMALL_MODEL_IDENTIFIER = "julien-c/bert-xsmall-dummy"
DUMMY_UNKWOWN_IDENTIFIER = "julien-c/dummy-unknown"
DUMMY_DIFF_TOKENIZER_IDENTIFIER = "julien-c/dummy-diff-tokenizer"
# Used to test Auto{Config, Model, Tokenizer} model_type detection.


def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:
        # KEY isn't set, default to `default`.
        _value = default
    else:
        # KEY is set, convert it to True or False.
        try:
            _value = strtobool(value)
        except ValueError:
            # More values are supported, but let's keep the message simple.
            raise ValueError("If set, {} must be yes or no.".format(key))
    return _value


def parse_int_from_env(key, default=None):
    try:
        value = os.environ[key]
    except KeyError:
        _value = default
    else:
        try:
            _value = int(value)
        except ValueError:
            raise ValueError("If set, {} must be a int.".format(key))
    return _value


_run_slow_tests = parse_flag_from_env("RUN_SLOW", default=False)
_run_pt_tf_cross_tests = parse_flag_from_env("RUN_PT_TF_CROSS_TESTS", default=False)
_run_custom_tokenizers = parse_flag_from_env("RUN_CUSTOM_TOKENIZERS", default=False)
_run_pipeline_tests = parse_flag_from_env("RUN_PIPELINE_TESTS", default=False)
_run_git_lfs_tests = parse_flag_from_env("RUN_GIT_LFS_TESTS", default=False)
_tf_gpu_memory_limit = parse_int_from_env("TF_GPU_MEMORY_LIMIT", default=None)


def is_pt_tf_cross_test(test_case):
    """
    Decorator marking a test as a test that control interactions between PyTorch and TensorFlow.

    PT+TF tests are skipped by default and we can run only them by setting RUN_PT_TF_CROSS_TESTS environment variable
    to a truthy value and selecting the is_pt_tf_cross_test pytest mark.

    """
    if not _run_pt_tf_cross_tests or not is_torch_available() or not is_tf_available():
        return unittest.skip("test is PT+TF test")(test_case)
    else:
        try:
            import pytest  # We don't need a hard dependency on pytest in the main library
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_pt_tf_cross_test()(test_case)


def is_pipeline_test(test_case):
    """
    Decorator marking a test as a pipeline test.

    Pipeline tests are skipped by default and we can run only them by setting RUN_PIPELINE_TESTS environment variable
    to a truthy value and selecting the is_pipeline_test pytest mark.

    """
    if not _run_pipeline_tests:
        return unittest.skip("test is pipeline test")(test_case)
    else:
        try:
            import pytest  # We don't need a hard dependency on pytest in the main library
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_pipeline_test()(test_case)


def slow(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable to a truthy value to run them.

    """
    if not _run_slow_tests:
        return unittest.skip("test is slow")(test_case)
    else:
        return test_case


def custom_tokenizers(test_case):
    """
    Decorator marking a test for a custom tokenizer.

    Custom tokenizers require additional dependencies, and are skipped by default. Set the RUN_CUSTOM_TOKENIZERS
    environment variable to a truthy value to run them.
    """
    if not _run_custom_tokenizers:
        return unittest.skip("test of custom tokenizers")(test_case)
    else:
        return test_case


def require_git_lfs(test_case):
    """
    Decorator marking a test that requires git-lfs.

    git-lfs requires additional dependencies, and tests are skipped by default. Set the RUN_GIT_LFS_TESTS environment
    variable to a truthy value to run them.
    """
    if not _run_git_lfs_tests:
        return unittest.skip("test of git lfs workflow")(test_case)
    else:
        return test_case


def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch.

    These tests are skipped when PyTorch isn't installed.

    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)
    else:
        return test_case


def require_torch_scatter(test_case):
    """
    Decorator marking a test that requires PyTorch scatter.

    These tests are skipped when PyTorch scatter isn't installed.

    """
    if not is_scatter_available():
        return unittest.skip("test requires PyTorch scatter")(test_case)
    else:
        return test_case


def require_tf(test_case):
    """
    Decorator marking a test that requires TensorFlow.

    These tests are skipped when TensorFlow isn't installed.

    """
    if not is_tf_available():
        return unittest.skip("test requires TensorFlow")(test_case)
    else:
        return test_case


def require_flax(test_case):
    """
    Decorator marking a test that requires JAX & Flax

    These tests are skipped when one / both are not installed

    """
    if not is_flax_available():
        test_case = unittest.skip("test requires JAX & Flax")(test_case)
    return test_case


def require_sentencepiece(test_case):
    """
    Decorator marking a test that requires SentencePiece.

    These tests are skipped when SentencePiece isn't installed.

    """
    if not is_sentencepiece_available():
        return unittest.skip("test requires SentencePiece")(test_case)
    else:
        return test_case


def require_tokenizers(test_case):
    """
    Decorator marking a test that requires 🤗 Tokenizers.

    These tests are skipped when 🤗 Tokenizers isn't installed.

    """
    if not is_tokenizers_available():
        return unittest.skip("test requires tokenizers")(test_case)
    else:
        return test_case


def require_pandas(test_case):
    """
    Decorator marking a test that requires pandas. These tests are skipped when pandas isn't installed.
    """
    if not is_pandas_available():
        return unittest.skip("test requires pandas")(test_case)
    else:
        return test_case


def require_scatter(test_case):
    """
    Decorator marking a test that requires PyTorch Scatter. These tests are skipped when PyTorch Scatter isn't
    installed.
    """
    if not is_scatter_available():
        return unittest.skip("test requires PyTorch Scatter")(test_case)
    else:
        return test_case


def require_torch_multi_gpu(test_case):
    """
    Decorator marking a test that requires a multi-GPU setup (in PyTorch).

    These tests are skipped on a machine without multiple GPUs.

    To run *only* the multi_gpu tests, assuming all test names contain multi_gpu: $ pytest -sv ./tests -k "multi_gpu"
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    if torch.cuda.device_count() < 2:
        return unittest.skip("test requires multiple GPUs")(test_case)
    else:
        return test_case


def require_torch_non_multi_gpu(test_case):
    """
    Decorator marking a test that requires 0 or 1 GPU setup (in PyTorch).
    """
    if not is_torch_available():
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    if torch.cuda.device_count() > 1:
        return unittest.skip("test requires 0 or 1 GPU")(test_case)
    else:
        return test_case


# this is a decorator identical to require_torch_non_multi_gpu, but is used as a quick band-aid to
# allow all of examples to be run multi-gpu CI and it reminds us that tests decorated with this one
# need to be ported and aren't so by design.
require_torch_non_multi_gpu_but_fix_me = require_torch_non_multi_gpu


def require_torch_tpu(test_case):
    """
    Decorator marking a test that requires a TPU (in PyTorch).
    """
    if not is_torch_tpu_available():
        return unittest.skip("test requires PyTorch TPU")
    else:
        return test_case


if is_torch_available():
    # Set env var CUDA_VISIBLE_DEVICES="" to force cpu-mode
    import torch

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    torch_device = None


def require_torch_gpu(test_case):
    """Decorator marking a test that requires CUDA and PyTorch. """
    if torch_device != "cuda":
        return unittest.skip("test requires CUDA")(test_case)
    else:
        return test_case


def require_datasets(test_case):
    """Decorator marking a test that requires datasets."""

    if not is_datasets_available():
        return unittest.skip("test requires `datasets`")(test_case)
    else:
        return test_case


def require_faiss(test_case):
    """Decorator marking a test that requires faiss."""
    if not is_faiss_available():
        return unittest.skip("test requires `faiss`")(test_case)
    else:
        return test_case


def require_optuna(test_case):
    """
    Decorator marking a test that requires optuna.

    These tests are skipped when optuna isn't installed.

    """
    if not is_optuna_available():
        return unittest.skip("test requires optuna")(test_case)
    else:
        return test_case


def require_ray(test_case):
    """
    Decorator marking a test that requires Ray/tune.

    These tests are skipped when Ray/tune isn't installed.

    """
    if not is_ray_available():
        return unittest.skip("test requires Ray/tune")(test_case)
    else:
        return test_case


def require_soundfile(test_case):
    """
    Decorator marking a test that requires soundfile

    These tests are skipped when soundfile isn't installed.

    """
    if not is_soundfile_availble():
        return unittest.skip("test requires soundfile")(test_case)
    else:
        return test_case


def get_gpu_count():
    """
    Return the number of available gpus (regardless of whether torch or tf is used)
    """
    if is_torch_available():
        import torch

        return torch.cuda.device_count()
    elif is_tf_available():
        import tensorflow as tf

        return len(tf.config.list_physical_devices("GPU"))
    else:
        return 0


def get_tests_dir(append_path=None):
    """
    Args:
        append_path: optional path to append to the tests dir path

    Return:
        The full path to the `tests` dir, so that the tests can be invoked from anywhere. Optionally `append_path` is
        joined after the `tests` dir the former is provided.

    """
    # this function caller's __file__
    caller__file__ = inspect.stack()[1][1]
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))
    if append_path:
        return os.path.join(tests_dir, append_path)
    else:
        return tests_dir


#
# Helper functions for dealing with testing text outputs
# The original code came from:
# https://github.com/fastai/fastai/blob/master/tests/utils/text.py

# When any function contains print() calls that get overwritten, like progress bars,
# a special care needs to be applied, since under pytest -s captured output (capsys
# or contextlib.redirect_stdout) contains any temporary printed strings, followed by
# \r's. This helper function ensures that the buffer will contain the same output
# with and without -s in pytest, by turning:
# foo bar\r tar mar\r final message
# into:
# final message
# it can handle a single string or a multiline buffer
def apply_print_resets(buf):
    return re.sub(r"^.*\r", "", buf, 0, re.M)


def assert_screenout(out, what):
    out_pr = apply_print_resets(out).lower()
    match_str = out_pr.find(what.lower())
    assert match_str != -1, f"expecting to find {what} in output: f{out_pr}"


class CaptureStd:
    """
    Context manager to capture:
        stdout, clean it up and make it available via obj.out stderr, and make it available via obj.err

        init arguments: - out - capture stdout: True/False, default True - err - capture stdout: True/False, default
        True

        Examples::

            with CaptureStdout() as cs:
                print("Secret message")
            print(f"captured: {cs.out}")

            import sys
            with CaptureStderr() as cs:
                print("Warning: ", file=sys.stderr)
            print(f"captured: {cs.err}")

            # to capture just one of the streams, but not the other
            with CaptureStd(err=False) as cs:
                print("Secret message")
            print(f"captured: {cs.out}")
            # but best use the stream-specific subclasses

    """

    def __init__(self, out=True, err=True):
        if out:
            self.out_buf = StringIO()
            self.out = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.out_buf = None
            self.out = "not capturing stdout"

        if err:
            self.err_buf = StringIO()
            self.err = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.err_buf = None
            self.err = "not capturing stderr"

    def __enter__(self):
        if self.out_buf:
            self.out_old = sys.stdout
            sys.stdout = self.out_buf

        if self.err_buf:
            self.err_old = sys.stderr
            sys.stderr = self.err_buf

        return self

    def __exit__(self, *exc):
        if self.out_buf:
            sys.stdout = self.out_old
            self.out = apply_print_resets(self.out_buf.getvalue())

        if self.err_buf:
            sys.stderr = self.err_old
            self.err = self.err_buf.getvalue()

    def __repr__(self):
        msg = ""
        if self.out_buf:
            msg += f"stdout: {self.out}\n"
        if self.err_buf:
            msg += f"stderr: {self.err}\n"
        return msg


# in tests it's the best to capture only the stream that's wanted, otherwise
# it's easy to miss things, so unless you need to capture both streams, use the
# subclasses below (less typing). Or alternatively, configure `CaptureStd` to
# disable the stream you don't need to test.


class CaptureStdout(CaptureStd):
    """ Same as CaptureStd but captures only stdout """

    def __init__(self):
        super().__init__(err=False)


class CaptureStderr(CaptureStd):
    """ Same as CaptureStd but captures only stderr """

    def __init__(self):
        super().__init__(out=False)


class CaptureLogger:
    """
    Context manager to capture `logging` streams

    Args:

    - logger: 'logging` logger object

    Results:
        The captured output is available via `self.out`

    Example::

        >>> from transformers import logging
        >>> from transformers.testing_utils import CaptureLogger

        >>> msg = "Testing 1, 2, 3"
        >>> logging.set_verbosity_info()
        >>> logger = logging.get_logger("transformers.models.bart.tokenization_bart")
        >>> with CaptureLogger(logger) as cl:
        ...     logger.info(msg)
        >>> assert cl.out, msg+"\n"
    """

    def __init__(self, logger):
        self.logger = logger
        self.io = StringIO()
        self.sh = logging.StreamHandler(self.io)
        self.out = ""

    def __enter__(self):
        self.logger.addHandler(self.sh)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.sh)
        self.out = self.io.getvalue()

    def __repr__(self):
        return f"captured: {self.out}\n"


class TestCasePlus(unittest.TestCase):
    """
    This class extends `unittest.TestCase` with additional features.

    Feature 1: A set of fully resolved important file and dir path accessors.

    In tests often we need to know where things are relative to the current test file, and it's not trivial since the
    test could be invoked from more than one directory or could reside in sub-directories with different depths. This
    class solves this problem by sorting out all the basic paths and provides easy accessors to them:

    * ``pathlib`` objects (all fully resolved):

       - ``test_file_path`` - the current test file path (=``__file__``)
       - ``test_file_dir`` - the directory containing the current test file
       - ``tests_dir`` - the directory of the ``tests`` test suite
       - ``examples_dir`` - the directory of the ``examples`` test suite
       - ``repo_root_dir`` - the directory of the repository
       - ``src_dir`` - the directory of ``src`` (i.e. where the ``transformers`` sub-dir resides)

    * stringified paths---same as above but these return paths as strings, rather than ``pathlib`` objects:

       - ``test_file_path_str``
       - ``test_file_dir_str``
       - ``tests_dir_str``
       - ``examples_dir_str``
       - ``repo_root_dir_str``
       - ``src_dir_str``

    Feature 2: Flexible auto-removable temporary dirs which are guaranteed to get removed at the end of test.

    1. Create a unique temporary dir:

    ::

        def test_whatever(self):
            tmp_dir = self.get_auto_remove_tmp_dir()

    ``tmp_dir`` will contain the path to the created temporary dir. It will be automatically removed at the end of the
    test.


    2. Create a temporary dir of my choice, ensure it's empty before the test starts and don't
    empty it after the test.

    ::

        def test_whatever(self):
            tmp_dir = self.get_auto_remove_tmp_dir("./xxx")

    This is useful for debug when you want to monitor a specific directory and want to make sure the previous tests
    didn't leave any data in there.

    3. You can override the first two options by directly overriding the ``before`` and ``after`` args, leading to the
       following behavior:

    ``before=True``: the temporary dir will always be cleared at the beginning of the test.

    ``before=False``: if the temporary dir already existed, any existing files will remain there.

    ``after=True``: the temporary dir will always be deleted at the end of the test.

    ``after=False``: the temporary dir will always be left intact at the end of the test.

    Note 1: In order to run the equivalent of ``rm -r`` safely, only subdirs of the project repository checkout are
    allowed if an explicit ``tmp_dir`` is used, so that by mistake no ``/tmp`` or similar important part of the
    filesystem will get nuked. i.e. please always pass paths that start with ``./``

    Note 2: Each test can register multiple temporary dirs and they all will get auto-removed, unless requested
    otherwise.

    Feature 3: Get a copy of the ``os.environ`` object that sets up ``PYTHONPATH`` specific to the current test suite.
    This is useful for invoking external programs from the test suite - e.g. distributed training.


    ::
        def test_whatever(self):
            env = self.get_env()

    """

    def setUp(self):
        # get_auto_remove_tmp_dir feature:
        self.teardown_tmp_dirs = []

        # figure out the resolved paths for repo_root, tests, examples, etc.
        self._test_file_path = inspect.getfile(self.__class__)
        path = Path(self._test_file_path).resolve()
        self._test_file_dir = path.parents[0]
        for up in [1, 2, 3]:
            tmp_dir = path.parents[up]
            if (tmp_dir / "src").is_dir() and (tmp_dir / "tests").is_dir():
                break
        if tmp_dir:
            self._repo_root_dir = tmp_dir
        else:
            raise ValueError(f"can't figure out the root of the repo from {self._test_file_path}")
        self._tests_dir = self._repo_root_dir / "tests"
        self._examples_dir = self._repo_root_dir / "examples"
        self._src_dir = self._repo_root_dir / "src"

    @property
    def test_file_path(self):
        return self._test_file_path

    @property
    def test_file_path_str(self):
        return str(self._test_file_path)

    @property
    def test_file_dir(self):
        return self._test_file_dir

    @property
    def test_file_dir_str(self):
        return str(self._test_file_dir)

    @property
    def tests_dir(self):
        return self._tests_dir

    @property
    def tests_dir_str(self):
        return str(self._tests_dir)

    @property
    def examples_dir(self):
        return self._examples_dir

    @property
    def examples_dir_str(self):
        return str(self._examples_dir)

    @property
    def repo_root_dir(self):
        return self._repo_root_dir

    @property
    def repo_root_dir_str(self):
        return str(self._repo_root_dir)

    @property
    def src_dir(self):
        return self._src_dir

    @property
    def src_dir_str(self):
        return str(self._src_dir)

    def get_env(self):
        """
        Return a copy of the ``os.environ`` object that sets up ``PYTHONPATH`` correctly, depending on the test suite
        it's invoked from. This is useful for invoking external programs from the test suite - e.g. distributed
        training.

        It always inserts ``./src`` first, then ``./tests`` or ``./examples`` depending on the test suite type and
        finally the preset ``PYTHONPATH`` if any (all full resolved paths).

        """
        env = os.environ.copy()
        paths = [self.src_dir_str]
        if "/examples" in self.test_file_dir_str:
            paths.append(self.examples_dir_str)
        else:
            paths.append(self.tests_dir_str)
        paths.append(env.get("PYTHONPATH", ""))

        env["PYTHONPATH"] = ":".join(paths)
        return env

    def get_auto_remove_tmp_dir(self, tmp_dir=None, before=None, after=None):
        """
        Args:
            tmp_dir (:obj:`string`, `optional`):
                if :obj:`None`:

                   - a unique temporary path will be created
                   - sets ``before=True`` if ``before`` is :obj:`None`
                   - sets ``after=True`` if ``after`` is :obj:`None`
                else:

                   - :obj:`tmp_dir` will be created
                   - sets ``before=True`` if ``before`` is :obj:`None`
                   - sets ``after=False`` if ``after`` is :obj:`None`
            before (:obj:`bool`, `optional`):
                If :obj:`True` and the :obj:`tmp_dir` already exists, make sure to empty it right away if :obj:`False`
                and the :obj:`tmp_dir` already exists, any existing files will remain there.
            after (:obj:`bool`, `optional`):
                If :obj:`True`, delete the :obj:`tmp_dir` at the end of the test if :obj:`False`, leave the
                :obj:`tmp_dir` and its contents intact at the end of the test.

        Returns:
            tmp_dir(:obj:`string`): either the same value as passed via `tmp_dir` or the path to the auto-selected tmp
            dir
        """
        if tmp_dir is not None:

            # defining the most likely desired behavior for when a custom path is provided.
            # this most likely indicates the debug mode where we want an easily locatable dir that:
            # 1. gets cleared out before the test (if it already exists)
            # 2. is left intact after the test
            if before is None:
                before = True
            if after is None:
                after = False

            # using provided path
            path = Path(tmp_dir).resolve()

            # to avoid nuking parts of the filesystem, only relative paths are allowed
            if not tmp_dir.startswith("./"):
                raise ValueError(
                    f"`tmp_dir` can only be a relative path, i.e. `./some/path`, but received `{tmp_dir}`"
                )

            # ensure the dir is empty to start with
            if before is True and path.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

            path.mkdir(parents=True, exist_ok=True)

        else:
            # defining the most likely desired behavior for when a unique tmp path is auto generated
            # (not a debug mode), here we require a unique tmp dir that:
            # 1. is empty before the test (it will be empty in this situation anyway)
            # 2. gets fully removed after the test
            if before is None:
                before = True
            if after is None:
                after = True

            # using unique tmp dir (always empty, regardless of `before`)
            tmp_dir = tempfile.mkdtemp()

        if after is True:
            # register for deletion
            self.teardown_tmp_dirs.append(tmp_dir)

        return tmp_dir

    def tearDown(self):

        # get_auto_remove_tmp_dir feature: remove registered temp dirs
        for path in self.teardown_tmp_dirs:
            shutil.rmtree(path, ignore_errors=True)
        self.teardown_tmp_dirs = []


def mockenv(**kwargs):
    """
    this is a convenience wrapper, that allows this:

    @mockenv(RUN_SLOW=True, USE_TF=False) def test_something(): run_slow = os.getenv("RUN_SLOW", False) use_tf =
    os.getenv("USE_TF", False)
    """
    return unittest.mock.patch.dict(os.environ, kwargs)


# --- pytest conf functions --- #

# to avoid multiple invocation from tests/conftest.py and examples/conftest.py - make sure it's called only once
pytest_opt_registered = {}


def pytest_addoption_shared(parser):
    """
    This function is to be called from `conftest.py` via `pytest_addoption` wrapper that has to be defined there.

    It allows loading both `conftest.py` files at once without causing a failure due to adding the same `pytest`
    option.

    """
    option = "--make-reports"
    if option not in pytest_opt_registered:
        parser.addoption(
            option,
            action="store",
            default=False,
            help="generate report files. The value of this option is used as a prefix to report names",
        )
        pytest_opt_registered[option] = 1


def pytest_terminal_summary_main(tr, id):
    """
    Generate multiple reports at the end of test suite run - each report goes into a dedicated file in the current
    directory. The report files are prefixed with the test suite name.

    This function emulates --duration and -rA pytest arguments.

    This function is to be called from `conftest.py` via `pytest_terminal_summary` wrapper that has to be defined
    there.

    Args:

    - tr: `terminalreporter` passed from `conftest.py`
    - id: unique id like `tests` or `examples` that will be incorporated into the final reports filenames - this is
      needed as some jobs have multiple runs of pytest, so we can't have them overwrite each other.

    NB: this functions taps into a private _pytest API and while unlikely, it could break should
    pytest do internal changes - also it calls default internal methods of terminalreporter which
    can be hijacked by various `pytest-` plugins and interfere.

    """
    from _pytest.config import create_terminal_writer

    if not len(id):
        id = "tests"

    config = tr.config
    orig_writer = config.get_terminal_writer()
    orig_tbstyle = config.option.tbstyle
    orig_reportchars = tr.reportchars

    dir = "reports"
    Path(dir).mkdir(parents=True, exist_ok=True)
    report_files = {
        k: f"{dir}/{id}_{k}.txt"
        for k in [
            "durations",
            "errors",
            "failures_long",
            "failures_short",
            "failures_line",
            "passes",
            "stats",
            "summary_short",
            "warnings",
        ]
    }

    # custom durations report
    # note: there is no need to call pytest --durations=XX to get this separate report
    # adapted from https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/runner.py#L66
    dlist = []
    for replist in tr.stats.values():
        for rep in replist:
            if hasattr(rep, "duration"):
                dlist.append(rep)
    if dlist:
        dlist.sort(key=lambda x: x.duration, reverse=True)
        with open(report_files["durations"], "w") as f:
            durations_min = 0.05  # sec
            f.write("slowest durations\n")
            for i, rep in enumerate(dlist):
                if rep.duration < durations_min:
                    f.write(f"{len(dlist)-i} durations < {durations_min} secs were omitted")
                    break
                f.write(f"{rep.duration:02.2f}s {rep.when:<8} {rep.nodeid}\n")

    def summary_failures_short(tr):
        # expecting that the reports were --tb=long (default) so we chop them off here to the last frame
        reports = tr.getreports("failed")
        if not reports:
            return
        tr.write_sep("=", "FAILURES SHORT STACK")
        for rep in reports:
            msg = tr._getfailureheadline(rep)
            tr.write_sep("_", msg, red=True, bold=True)
            # chop off the optional leading extra frames, leaving only the last one
            longrepr = re.sub(r".*_ _ _ (_ ){10,}_ _ ", "", rep.longreprtext, 0, re.M | re.S)
            tr._tw.line(longrepr)
            # note: not printing out any rep.sections to keep the report short

    # use ready-made report funcs, we are just hijacking the filehandle to log to a dedicated file each
    # adapted from https://github.com/pytest-dev/pytest/blob/897f151e/src/_pytest/terminal.py#L814
    # note: some pytest plugins may interfere by hijacking the default `terminalreporter` (e.g.
    # pytest-instafail does that)

    # report failures with line/short/long styles
    config.option.tbstyle = "auto"  # full tb
    with open(report_files["failures_long"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    # config.option.tbstyle = "short" # short tb
    with open(report_files["failures_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        summary_failures_short(tr)

    config.option.tbstyle = "line"  # one line per error
    with open(report_files["failures_line"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    with open(report_files["errors"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_errors()

    with open(report_files["warnings"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_warnings()  # normal warnings
        tr.summary_warnings()  # final warnings

    tr.reportchars = "wPpsxXEf"  # emulate -rA (used in summary_passes() and short_test_summary())
    with open(report_files["passes"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_passes()

    with open(report_files["summary_short"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.short_test_summary()

    with open(report_files["stats"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_stats()

    # restore:
    tr._tw = orig_writer
    tr.reportchars = orig_reportchars
    config.option.tbstyle = orig_tbstyle


# --- distributed testing functions --- #

# adapted from https://stackoverflow.com/a/59041913/9201239
import asyncio  # noqa


class _RunOutput:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


async def _read_stream(stream, callback):
    while True:
        line = await stream.readline()
        if line:
            callback(line)
        else:
            break


async def _stream_subprocess(cmd, env=None, stdin=None, timeout=None, quiet=False, echo=False) -> _RunOutput:
    if echo:
        print("\nRunning: ", " ".join(cmd))

    p = await asyncio.create_subprocess_exec(
        cmd[0],
        *cmd[1:],
        stdin=stdin,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    # note: there is a warning for a possible deadlock when using `wait` with huge amounts of data in the pipe
    # https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.asyncio.subprocess.Process.wait
    #
    # If it starts hanging, will need to switch to the following code. The problem is that no data
    # will be seen until it's done and if it hangs for example there will be no debug info.
    # out, err = await p.communicate()
    # return _RunOutput(p.returncode, out, err)

    out = []
    err = []

    def tee(line, sink, pipe, label=""):
        line = line.decode("utf-8").rstrip()
        sink.append(line)
        if not quiet:
            print(label, line, file=pipe)

    # XXX: the timeout doesn't seem to make any difference here
    await asyncio.wait(
        [
            _read_stream(p.stdout, lambda l: tee(l, out, sys.stdout, label="stdout:")),
            _read_stream(p.stderr, lambda l: tee(l, err, sys.stderr, label="stderr:")),
        ],
        timeout=timeout,
    )
    return _RunOutput(await p.wait(), out, err)


def execute_subprocess_async(cmd, env=None, stdin=None, timeout=180, quiet=False, echo=True) -> _RunOutput:

    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        _stream_subprocess(cmd, env=env, stdin=stdin, timeout=timeout, quiet=quiet, echo=echo)
    )

    cmd_str = " ".join(cmd)
    if result.returncode > 0:
        stderr = "\n".join(result.stderr)
        raise RuntimeError(
            f"'{cmd_str}' failed with returncode {result.returncode}\n\n"
            f"The combined stderr from workers follows:\n{stderr}"
        )

    # check that the subprocess actually did run and produced some output, should the test rely on
    # the remote side to do the testing
    if not result.stdout and not result.stderr:
        raise RuntimeError(f"'{cmd_str}' produced no output.")

    return result
