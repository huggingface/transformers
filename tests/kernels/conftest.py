# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import os
from unittest import mock

import pytest


@pytest.fixture(scope="session", autouse=True)
def cheap_finegrained_tuning():
    # Use the minimum Bayesian search budget for correctness tests. Set this before the
    # kernel bundle loads: its autotuners read the trial budget at import time.
    # Subprocesses inherit it too, including distributed workers.
    with mock.patch.dict(os.environ, {"FINEGRAINED_AUTOTUNE_TRIALS": "1"}):
        yield
