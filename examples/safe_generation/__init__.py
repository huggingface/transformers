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

"""
Safe Generation Example Implementations

This module provides reference implementations of safety checkers for the transformers
safe generation feature. These are example implementations that users can use directly
or adapt for their specific needs.

The core transformers library provides only the infrastructure (SafetyChecker abstract base,
processors, configuration). Concrete implementations like BasicToxicityChecker are provided
here as examples to demonstrate how to implement custom safety checkers.

Example usage:
    from examples.safe_generation import BasicToxicityChecker
    from transformers import pipeline
    from transformers.generation.safety import SafetyConfig

    # Create a safety checker
    checker = BasicToxicityChecker(threshold=0.7)

    # Use with pipeline
    config = SafetyConfig.from_checker(checker)
    pipe = pipeline("text-generation", model="gpt2", safety_config=config)
"""

from .checkers import BasicToxicityChecker


__all__ = ["BasicToxicityChecker"]
