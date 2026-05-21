# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from transformers.processing_utils import ProcessorMixin


class DummyMultimodalProcessor(ProcessorMixin):
    pass


class ProcessorMixinTextReplacementTest(unittest.TestCase):
    def get_processor(self):
        processor = DummyMultimodalProcessor()
        processor.image_token = "<image>"
        processor.video_token = "<video>"
        return processor

    def test_get_text_with_replacements_preserves_missing_replacement_placeholders(self):
        processor = self.get_processor()

        text, replacement_offsets = processor.get_text_with_replacements(
            ["Look <image> then <video> then <image>."],
            images_replacements=["<image><image>"],
            videos_replacements=["<video><video>"],
        )

        self.assertEqual(text, ["Look <image><image> then <video><video> then <image>."])
        self.assertEqual(
            [offset["replacement"] for offset in replacement_offsets[0]],
            ["<image><image>", "<video><video>"],
        )

    def test_get_text_with_replacements_preserves_placeholder_when_no_modality_data_is_provided(self):
        processor = self.get_processor()

        text, replacement_offsets = processor.get_text_with_replacements(["Profile <image> without image data."])

        self.assertEqual(text, ["Profile <image> without image data."])
        self.assertEqual(replacement_offsets, [[]])
