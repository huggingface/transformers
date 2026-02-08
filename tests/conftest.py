# Copyright 2024 HuggingFace Inc.
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

"""
Pytest configuration and fixtures for transformers tests.
"""

import pytest

from transformers import set_seed as _set_seed


@pytest.fixture(autouse=True, scope="function")
def set_seed():
    """
    Set a fixed seed before each test to improve determinism and reduce flakiness.
    Uses the same seed (42) as in PR #43794 and existing test patterns.
    """
    _set_seed(42)
