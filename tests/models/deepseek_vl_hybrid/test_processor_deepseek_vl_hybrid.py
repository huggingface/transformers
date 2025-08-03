# coding=utf-8
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
import tempfile
import unittest

from transformers import DeepseekVLHybridProcessor, LlamaTokenizer
from transformers.models.deepseek_vl.convert_deepseek_vl_weights_to_hf import CHAT_TEMPLATE
from transformers.testing_utils import get_tests_dir
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import DeepseekVLHybridImageProcessor


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


class DeepseekVLHybridProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = DeepseekVLHybridProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = DeepseekVLHybridImageProcessor()
        tokenizer = LlamaTokenizer(
            vocab_file=SAMPLE_VOCAB,
            extra_special_tokens={
                "pad_token": "<｜end▁of▁sentence｜>",
                "image_token": "<image_placeholder>",
            },
        )
        processor = self.processor_class(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=CHAT_TEMPLATE,
        )
        processor.save_pretrained(self.tmpdirname)

    def prepare_processor_dict(self):
        return {"chat_template": CHAT_TEMPLATE, "num_image_tokens": 576}
