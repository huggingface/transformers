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

import sys
import warnings
from os.path import abspath, dirname, join


# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(dirname(__file__)), "src"))
sys.path.insert(1, git_repo_path)

# silence FutureWarning warnings in tests since often we can't act on them until
# they become normal warnings - i.e. the tests still need to test the current functionality
warnings.simplefilter(action="ignore", category=FutureWarning)


def pytest_configure(config):
    config.addinivalue_line("markers", "is_pipeline_test: mark test to run only when pipeline are tested")
    config.addinivalue_line(
        "markers", "is_pt_tf_cross_test: mark test to run only when PT and TF interactions are tested"
    )
    config.addinivalue_line(
        "markers", "is_pt_flax_cross_test: mark test to run only when PT and FLAX interactions are tested"
    )
    config.addinivalue_line("markers", "is_staging_test: mark test to run only in the staging environment")


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
