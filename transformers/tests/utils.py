import os
import unittest

from distutils.util import strtobool


try:
    run_slow = os.environ["RUN_SLOW"]
except KeyError:
    # RUN_SLOW isn't set, default to skipping slow tests.
    _run_slow_tests = False
else:
    # RUN_SLOW is set, convert it to True or False.
    try:
        _run_slow_tests = strtobool(run_slow)
    except ValueError:
        # More values are supported, but let's keep the message simple.
        raise ValueError("If set, RUN_SLOW must be yes or no.")


def slow(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable
    to a truthy value to run them.

    """
    if not _run_slow_tests:
        test_case = unittest.skip("test is slow")(test_case)
    return test_case
