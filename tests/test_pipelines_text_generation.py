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

from transformers import pipeline

from .test_pipelines_common import MonoInputPipelineCommonMixin


class TextGenerationPipelineTests(MonoInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "text-generation"
    pipeline_running_kwargs = {"prefix": "This is "}
    small_models = ["sshleifer/tiny-ctrl"]  # Models tested without the @slow decorator
    large_models = []  # Models tested with the @slow decorator

    def test_simple_generation(self):
        nlp = pipeline(task="text-generation", model=self.small_models[0])
        # text-generation is non-deterministic by nature, we can't fully test the output

        outputs = nlp("This is a test")

        self.assertEqual(len(outputs), 1)
        self.assertEqual(list(outputs[0].keys()), ["generated_text"])
        self.assertEqual(type(outputs[0]["generated_text"]), str)

        outputs = nlp(["This is a test", "This is a second test"])
        self.assertEqual(len(outputs[0]), 1)
        self.assertEqual(list(outputs[0][0].keys()), ["generated_text"])
        self.assertEqual(type(outputs[0][0]["generated_text"]), str)
        self.assertEqual(list(outputs[1][0].keys()), ["generated_text"])
        self.assertEqual(type(outputs[1][0]["generated_text"]), str)
