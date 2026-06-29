# Copyright 2026 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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

from transformers import FunAsrNanoProcessor


class FunAsrNanoProcessorTest(unittest.TestCase):
    def _make_processor(self):
        captured = {}
        processor = FunAsrNanoProcessor.__new__(FunAsrNanoProcessor)
        processor.default_transcription_prompt = "Transcribe the audio:"

        def apply_chat_template(conversations, **kwargs):
            captured["conversations"] = conversations
            captured["kwargs"] = kwargs
            return {"ok": True}

        processor.apply_chat_template = apply_chat_template
        return processor, captured

    def test_apply_transcription_request_single_path(self):
        processor, captured = self._make_processor()

        self.assertEqual(processor.apply_transcription_request(audio="audio.wav"), {"ok": True})

        self.assertEqual(
            captured["conversations"],
            [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe the audio:"},
                            {"type": "audio", "path": "audio.wav"},
                        ],
                    }
                ]
            ],
        )
        self.assertTrue(captured["kwargs"]["tokenize"])
        self.assertTrue(captured["kwargs"]["add_generation_prompt"])
        self.assertTrue(captured["kwargs"]["return_dict"])


if __name__ == "__main__":
    unittest.main()
