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

from parameterized import parameterized

from transformers import NemotronParseProcessor
from transformers.testing_utils import require_torch, require_vision

from ...test_processing_common import ProcessorTesterMixin


@require_torch
@require_vision
class NemotronParseProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    model_id = "nvidia/NVIDIA-Nemotron-Parse-2.0"
    processor_class = NemotronParseProcessor
    text_input_name = "decoder_input_ids"

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        # the default 2048x1664 page is ~40 MB per image as float32
        return image_processor_class.from_pretrained(cls.model_id, size={"height": 64, "width": 64})

    @parameterized.expand([(1, "pt"), (2, "pt")])
    @unittest.skip(reason="The common test reads `input_ids`, the processor returns `decoder_input_ids`")
    def test_apply_chat_template_image(self, batch_size, return_tensors):
        pass

    @unittest.skip(reason="The common test reads `input_ids`, the processor returns `decoder_input_ids`")
    def test_apply_chat_template_assistant_mask(self):
        pass

    def test_prompt_is_decoder_input(self):
        processor = self.get_processor()
        prompt = "</s><s><predict_bbox><predict_classes><output_markdown>"
        inputs = processor(images=self.prepare_images_inputs(), text=prompt, return_tensors="pt")
        self.assertEqual(set(inputs), {"pixel_values", "decoder_input_ids", "decoder_attention_mask"})
        self.assertEqual(
            inputs["decoder_input_ids"].tolist(),
            [
                processor.tokenizer.convert_tokens_to_ids(
                    ["</s>", "<s>", "<predict_bbox>", "<predict_classes>", "<output_markdown>"]
                )
            ],
        )
