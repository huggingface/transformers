# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from transformers import CosmosProcessor, T5TokenizerFast
from transformers.testing_utils import (
    require_vision,
)
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import CosmosVideoProcessor


@require_vision
class CosmosProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = CosmosProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        video_processor = CosmosVideoProcessor()
        tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-11b")
        processor_kwargs = self.prepare_processor_dict()
        processor = CosmosProcessor(video_processor=video_processor, tokenizer=tokenizer, **processor_kwargs)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return CosmosProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_video_processor(self, **kwargs):
        return CosmosProcessor.from_pretrained(self.tmpdirname, **kwargs).video_processor
