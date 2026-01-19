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

import unittest

from transformers import AutoConfig, SnacConfig


class SnacConfigOfflineTest(unittest.TestCase):
    def test_auto_config_for_model(self):
        cfg = AutoConfig.for_model("snac")
        self.assertIsInstance(cfg, SnacConfig)
        self.assertEqual(cfg.model_type, "snac")
        self.assertEqual(cfg.__class__.__name__, "SnacConfig")
        self.assertIn("transformers.models.snac", cfg.__class__.__module__)
