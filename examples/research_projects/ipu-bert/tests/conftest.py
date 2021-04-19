# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import pytest


@pytest.fixture(autouse=True)
def cleanup():
    # Explicitly clean up to make sure we detach from the IPU and
    # free the graph before the next test starts.
    gc.collect()


# Below functions enable long tests to be skipped, unless a --long-test
# cli option is specified.
def pytest_addoption(parser):
    # Add cli option --long-test to run tests that take too long
    parser.addoption("--long-test", action="store_true", default=False,
                     help="Run long tests")


def pytest_configure(config):
    # Register marker longtest
    config.addinivalue_line(
        "markers",
        "skip_longtest_needs_dataset: long tests needing dataset to run")


def pytest_collection_modifyitems(config, items):
    # Add skip test marker based on --long-test option during execution
    if config.getoption("--long-test"):
        return
    marker = pytest.mark.skip(reason="This test takes too long to run. "
                              "Run manually with the --long-test option.")
    for item in items:
        if "skip_longtest_needs_dataset" in item.keywords:
            item.add_marker(marker)
