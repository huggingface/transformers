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

import unittest

from .test_pipelines_common import MonoInputPipelineCommonMixin


class Text2TextGenerationPipelineTests(MonoInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "text2text-generation"
    small_models = ["patrickvonplaten/t5-tiny-random"]  # Default model - Models tested without the @slow decorator
    large_models = []  # Models tested with the @slow decorator
    invalid_inputs = [4, "<mask>"]
    mandatory_keys = ["generated_text"]
