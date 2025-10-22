# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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

from .configuration import SafetyConfig


def validate_safety_config(config: SafetyConfig) -> bool:
    """
    Validate a safety configuration and return whether it's valid.

    Args:
        config (`SafetyConfig`): Configuration to validate.

    Returns:
        `bool`: True if configuration is valid, False otherwise.

    Example:
    ```python
    config = SafetyConfig(enabled=True, thresholds={"toxicity": 0.5})
    if validate_safety_config(config):
        print("Configuration is valid")
    ```
    """
    try:
        config.validate()
        return True
    except (ValueError, TypeError):
        return False
