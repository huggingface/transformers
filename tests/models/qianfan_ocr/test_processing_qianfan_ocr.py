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
"""Testing suite for the QianfanOCR processor."""

import unittest

from transformers import QianfanOCRProcessor
from transformers.testing_utils import require_torch, require_vision, slow
from transformers.utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch


# QianfanOCRProcessor.__init__ reads special token attributes (start_image_token,
# end_image_token, context_image_token, etc.) directly from the tokenizer, which are
# only present on the InternVL-family tokenizer shipped with the real checkpoint.
# A generic Qwen2Tokenizer does not have these attributes, so the processor cannot be
# constructed from scratch without access to the pretrained tokenizer.  All tests
# therefore require network access and are marked @slow.
@slow
@require_vision
class QianfanOCRProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = QianfanOCRProcessor
    model_id = "/mnt/cfs_bj_mt/workspace/zhuangbairong/qfocr-trs-qa/Qianfan-OCR-hf-v3"
    # QianfanOCR has no video support; images and pixel values share the same tensor key
    videos_input_name = "pixel_values"

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token

    @unittest.skip("QianfanOCR does not support video processing")
    def test_video_processor_defaults(self):
        pass

    @unittest.skip("QianfanOCR does not support video processing")
    def test_process_interleaved_images_videos(self):
        pass

    @unittest.skip("QianfanOCR does not support video processing")
    def test_apply_chat_template_video_frame_sampling(self):
        pass

    @staticmethod
    def prepare_processor_dict():
        return {"image_seq_length": 2}

    @require_torch
    def test_get_num_vision_tokens(self):
        """Tests general functionality of the helper used internally in vLLM."""
        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertIn("num_image_tokens", output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertIn("num_image_patches", output)
        self.assertEqual(len(output["num_image_patches"]), 3)
