# Copyright 2025 HuggingFace Inc.
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


from transformers import VibeVoiceRealTimeProcessor
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    pass


@require_torch
class VibeVoiceProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = VibeVoiceRealTimeProcessor
    model_id = "bezzam/VibeVoice-0.5B"

    def test_voice_preset_batch(self):
        # test that voice preset batching works correctly
        text = ["Hello, nice to meet you.", "This is a test sentence."]
        processor = self.processor_class.from_pretrained(self.model_id)
        inputs = processor(text=text, return_tensors="pt", padding=True)
        voice_preset = inputs["voice_preset"]
        for key in voice_preset:
            self.assertEqual(voice_preset[key]["last_hidden_state"].shape[0], len(text))
            past_key_values = voice_preset[key]["past_key_values"]
            for cache_key in ["key_cache", "value_cache"]:
                for tensor in past_key_values[cache_key]:
                    self.assertEqual(tensor.shape[0], len(text))
