import os
import re
import sys
import unittest
from distutils.util import strtobool
from io import StringIO

from .file_utils import _tf_available, _torch_available, _torch_tpu_available


SMALL_MODEL_IDENTIFIER = "julien-c/bert-xsmall-dummy"
DUMMY_UNKWOWN_IDENTIFIER = "julien-c/dummy-unknown"
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
    """ Context manager to capture:
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
