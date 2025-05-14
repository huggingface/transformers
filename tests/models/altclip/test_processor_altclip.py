# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from transformers import AltCLIPProcessor, CLIPImageProcessor, XLMRobertaTokenizer, XLMRobertaTokenizerFast
from transformers.testing_utils import require_vision

from ...test_processing_common import ProcessorTesterMixin


@require_vision
class AltClipProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = AltCLIPProcessor

    @classmethod
    def setUpClass(cls):
        cls.model_id = "BAAI/AltCLIP"
        cls.tmpdirname = tempfile.mkdtemp()
        image_processor = CLIPImageProcessor()
        tokenizer = XLMRobertaTokenizer.from_pretrained(cls.model_id)

        processor = cls.processor_class(image_processor, tokenizer)

        processor.save_pretrained(cls.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return XLMRobertaTokenizer.from_pretrained(self.model_id, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        return XLMRobertaTokenizerFast.from_pretrained(self.model_id, **kwargs)

    def get_image_processor(self, **kwargs):
        return CLIPImageProcessor.from_pretrained(self.model_id, **kwargs)
