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
"""Normalize producer metadata without silently changing module targets or quantization formats."""

import unittest

from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.utils.quantization_config import NVFP4Config


class QuantizationConfigReaderRegistryTest(unittest.TestCase):
    def test_a_registered_reader_reads_its_method(self):
        from unittest import mock

        from transformers.quantizers import auto, register_quantization_config_reader

        with mock.patch.dict(auto.AUTO_QUANTIZATION_CONFIG_READERS):
            register_quantization_config_reader("some-tool")(
                lambda config: NVFP4Config() if config.get("fp4") else None
            )
            self.assertIsInstance(
                AutoQuantizationConfig.from_dict({"quant_method": "some-tool", "fp4": True}), NVFP4Config
            )
            self.assertTrue(auto.AutoHfQuantizer.supports_quant_method({"quant_method": "some-tool"}))
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_quantization_config_reader("some-tool")(lambda config: None)
            # a reader returning None leaves the method to its own config class, here none
            with self.assertRaisesRegex(ValueError, "Unknown quantization type"):
                AutoQuantizationConfig.from_dict({"quant_method": "some-tool"})
