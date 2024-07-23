# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import tempfile
import unittest

from transformers import AutoTokenizer, Qwen2AudioProcessor, WhisperFeatureExtractor
from transformers.testing_utils import require_torch, require_torchaudio


@require_torch
@require_torchaudio
class Qwen2AudioProcessorTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = "Qwen/Qwen2-Audio-7B"
        self.tmpdirname = tempfile.mkdtemp()

    def test_can_load_various_tokenizers(self):
        processor = Qwen2AudioProcessor.from_pretrained(self.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    def test_save_load_pretrained_default(self):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        processor = Qwen2AudioProcessor.from_pretrained(self.checkpoint)
        feature_extractor = processor.feature_extractor

        processor = Qwen2AudioProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        processor.save_pretrained(self.tmpdirname)
        processor = Qwen2AudioProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, WhisperFeatureExtractor)
