# coding=utf-8
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

from transformers import is_flax_available
from transformers.testing_utils import DUMMY_UNKNOWN_IDENTIFIER, require_flax


if is_flax_available():
    from transformers import FlaxAutoModel


@require_flax
class FlaxAutoModelTest(unittest.TestCase):
    def test_repo_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, "bert-base is not a local folder and is not a valid model identifier"
        ):
            _ = FlaxAutoModel.from_pretrained("bert-base")

    def test_revision_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, r"aaaaaa is not a valid git identifier \(branch name, tag name or commit id\)"
        ):
            _ = FlaxAutoModel.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, revision="aaaaaa")

    def test_model_file_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError,
            "hf-internal-testing/config-no-model does not appear to have a file named flax_model.msgpack",
        ):
            _ = FlaxAutoModel.from_pretrained("hf-internal-testing/config-no-model")

    def test_model_from_pt_suggestion(self):
        with self.assertRaisesRegex(EnvironmentError, "Use `from_pt=True` to load this model"):
            _ = FlaxAutoModel.from_pretrained("hf-internal-testing/tiny-bert-pt-only")
