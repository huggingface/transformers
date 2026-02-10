# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the configuration of GlmMoeDsa."""

import unittest

from transformers import GlmMoeDsaConfig


class GlmMoeDsaConfigTest(unittest.TestCase):
    def test_default_mlp_layer_types(self):
        config = GlmMoeDsaConfig(num_hidden_layers=8)
        self.assertEqual(
            config.mlp_layer_types, ["dense", "dense", "dense", "sparse", "sparse", "sparse", "sparse", "sparse"]
        )


if __name__ == "__main__":
    unittest.main()
