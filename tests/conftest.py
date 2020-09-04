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
if 1:  # flake be quiet
    from transformers.utils.logging import (
        logging_level_str_to_code,
        logging_levels_as_strings,
        set_global_logging_level,
    )


def pytest_addoption(parser):
    parser.addoption(
        "--loglevel",
        type=str,
        default=False,
        choices=logging_levels_as_strings(),
        help="set global logger level before each test",
    )


@pytest.fixture(scope="session", autouse=True)
def run_this_before_each_test(request):
    # set the loglevel for all loggers to the desired level
    loglevel = request.config.getoption("--loglevel")
    if loglevel:
        set_global_logging_level(level=logging_level_str_to_code(loglevel))
