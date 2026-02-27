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

"""
Shared constants and base classes for distributed trainer tests.
"""

import os


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
DISTRIBUTED_DIR = os.path.dirname(__file__)
CONFIGS_DIR = os.path.join(DISTRIBUTED_DIR, "accelerate_configs")
SCRIPTS_DIR = os.path.join(DISTRIBUTED_DIR, "scripts")


class TestTrainerDistributedDDPCommon:
    """Common distributed trainer tests shared across frameworks.

    Empty for now — will hold launcher-agnostic tests in the future.
    """

    pass
