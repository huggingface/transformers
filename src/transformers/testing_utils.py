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

from .file_utils import _tf_available, _torch_available, _torch_tpu_available


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
_run_custom_tokenizers = parse_flag_from_env("RUN_CUSTOM_TOKENIZERS", default=False)
_tf_gpu_memory_limit = parse_int_from_env("TF_GPU_MEMORY_LIMIT", default=None)


def slow(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable
    to a truthy value to run them.

    """
    if not _run_slow_tests:
        test_case = unittest.skip("test is slow")(test_case)
    return test_case


def custom_tokenizers(test_case):
    """
    Decorator marking a test for a custom tokenizer.

    Custom tokenizers require additional dependencies, and are skipped
    by default. Set the RUN_CUSTOM_TOKENIZERS environment variable
    to a truthy value to run them.
    """
    if not _run_custom_tokenizers:
        test_case = unittest.skip("test of custom tokenizers")(test_case)
    return test_case


def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch.

    These tests are skipped when PyTorch isn't installed.

    """
    if not _torch_available:
        test_case = unittest.skip("test requires PyTorch")(test_case)
    return test_case


def require_tf(test_case):
    """
    Decorator marking a test that requires TensorFlow.

    These tests are skipped when TensorFlow isn't installed.

    """
    if not _tf_available:
        test_case = unittest.skip("test requires TensorFlow")(test_case)
    return test_case


def require_multigpu(test_case):
    """
    Decorator marking a test that requires a multi-GPU setup (in PyTorch).

    These tests are skipped on a machine without multiple GPUs.

    To run *only* the multigpu tests, assuming all test names contain multigpu:
    $ pytest -sv ./tests -k "multigpu"
    """
    if not _torch_available:
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    if torch.cuda.device_count() < 2:
        return unittest.skip("test requires multiple GPUs")(test_case)
    return test_case


def require_non_multigpu(test_case):
    """
    Decorator marking a test that requires 0 or 1 GPU setup (in PyTorch).
    """
    if not _torch_available:
        return unittest.skip("test requires PyTorch")(test_case)

    import torch

    if torch.cuda.device_count() > 1:
        return unittest.skip("test requires 0 or 1 GPU")(test_case)
    return test_case


def require_torch_tpu(test_case):
    """
    Decorator marking a test that requires a TPU (in PyTorch).
    """
    if not _torch_tpu_available:
        return unittest.skip("test requires PyTorch TPU")

    return test_case


if _torch_available:
    # Set the USE_CUDA environment variable to select a GPU.
    torch_device = "cuda" if parse_flag_from_env("USE_CUDA") else "cpu"
else:
    torch_device = None


def require_torch_and_cuda(test_case):
    """Decorator marking a test that requires CUDA and PyTorch). """
    if torch_device != "cuda":
        return unittest.skip("test requires CUDA")
    else:
        return test_case


def get_tests_dir():
    """
    returns the full path to the `tests` dir, so that the tests can be invoked from anywhere
    """
    # this function caller's __file__
    caller__file__ = inspect.stack()[1][1]
    return os.path.abspath(os.path.dirname(caller__file__))


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
    """Context manager to capture:
    stdout, clean it up and make it available via obj.out
    stderr, and make it available via obj.err

    init arguments:
    - out - capture stdout: True/False, default True
    - err - capture stdout: True/False, default True

    Examples:

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
    """Context manager to capture `logging` streams

    Args:
    - logger: 'logging` logger object

    Results:
        The captured output is available via `self.out`

    Example:

    >>> from transformers import logging
    >>> from transformers.testing_utils import CaptureLogger

    >>> msg = "Testing 1, 2, 3"
    >>> logging.set_verbosity_info()
    >>> logger = logging.get_logger("transformers.tokenization_bart")
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
    """This class extends `unittest.TestCase` with additional features.

    Feature 1: Flexible auto-removable temp dirs which are guaranteed to get
    removed at the end of test.

    In all the following scenarios the temp dir will be auto-removed at the end
    of test, unless `after=False`.

    # 1. create a unique temp dir, `tmp_dir` will contain the path to the created temp dir
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir()

    # 2. create a temp dir of my choice and delete it at the end - useful for debug when you want to
    # monitor a specific directory
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir(tmp_dir="./tmp/run/test")

    # 3. create a temp dir of my choice and do not delete it at the end - useful for when you want
    # to look at the temp results
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir(tmp_dir="./tmp/run/test", after=False)

    # 4. create a temp dir of my choice and ensure to delete it right away - useful for when you
    # disabled deletion in the previous test run and want to make sure the that tmp dir is empty
    # before the new test is run
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir(tmp_dir="./tmp/run/test", before=True)

    Note 1: In order to run the equivalent of `rm -r` safely, only subdirs of the
    project repository checkout are allowed if an explicit `tmp_dir` is used, so
    that by mistake no `/tmp` or similar important part of the filesystem will
    get nuked. i.e. please always pass paths that start with `./`

    Note 2: Each test can register multiple temp dirs and they all will get
    auto-removed, unless requested otherwise.

    """

    def setUp(self):
        self.teardown_tmp_dirs = []

    def get_auto_remove_tmp_dir(self, tmp_dir=None, after=True, before=False):
        """
        Args:
            tmp_dir (:obj:`string`, `optional`):
                use this path, if None a unique path will be assigned
            before (:obj:`bool`, `optional`, defaults to :obj:`False`):
                if `True` and tmp dir already exists make sure to empty it right away
            after (:obj:`bool`, `optional`, defaults to :obj:`True`):
                delete the tmp dir at the end of the test

        Returns:
            tmp_dir(:obj:`string`):
                either the same value as passed via `tmp_dir` or the path to the auto-created tmp dir
        """
        if tmp_dir is not None:
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
            # using unique tmp dir (always empty, regardless of `before`)
            tmp_dir = tempfile.mkdtemp()

        if after is True:
            # register for deletion
            self.teardown_tmp_dirs.append(tmp_dir)

        return tmp_dir

    def tearDown(self):
        # remove registered temp dirs
        for path in self.teardown_tmp_dirs:
            shutil.rmtree(path, ignore_errors=True)
        self.teardown_tmp_dirs = []


def mockenv(**kwargs):
    """this is a convenience wrapper, that allows this:

    @mockenv(USE_CUDA=True, USE_TF=False)
    def test_something():
        use_cuda = os.getenv("USE_CUDA", False)
        use_tf = os.getenv("USE_TF", False)
    """
    return unittest.mock.patch.dict(os.environ, kwargs)
