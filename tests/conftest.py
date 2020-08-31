# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

import sys
from os.path import abspath, dirname, join

import pytest


# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(dirname(__file__)), "src"))
sys.path.insert(1, git_repo_path)

# import local modules after fixing up sys.path
from transformers.testing_utils import logging_level_str_to_code, logging_levels_as_strings, set_verbosity_all  # NOQA


def pytest_addoption(parser):
    parser.addoption(
        "--log-level-all",
        type=str,
        default=False,
        choices=logging_levels_as_strings(),
        help="set global logger level before each test",
    )


@pytest.fixture(scope="session", autouse=True)
def run_this_before_each_test(request):
    # set the loglevel for all loggers (not just transformers) to the desired level
    # note: as --log-level-all=error override is intended to be run only for one test
    # under debugging, this practically will have no impact on the test suite at large
    loglevel = request.config.getoption("--log-level-all")
    if loglevel:
        set_verbosity_all(level=logging_level_str_to_code(loglevel))
