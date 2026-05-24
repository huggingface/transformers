# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Tests for Qwen3TTSProcessor."""

import unittest

from transformers import Qwen3TTSProcessor, is_torch_available
from transformers.testing_utils import require_torch, slow

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    pass


@require_torch
class Qwen3TTSProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Qwen3TTSProcessor
    model_id = "qwen3_tts_converted"

    @slow
    def test_can_load_processor_from_pretrained(self):
        processor = Qwen3TTSProcessor.from_pretrained(self.model_id)
        self.assertIsNotNone(processor.tokenizer)
        self.assertIsNotNone(processor.feature_extractor)
