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

import shutil
import tempfile
import unittest

from transformers import AutoProcessor, ParakeetProcessor
from transformers.testing_utils import require_torch, require_torchaudio

from ...test_processing_common import ProcessorTesterMixin


@require_torch
@require_torchaudio
class ParakeetProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = ParakeetProcessor
    text_input_name = "labels"

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        cls.checkpoint = "nvidia/parakeet-ctc-1.1b"
        processor = ParakeetProcessor.from_pretrained(cls.checkpoint)
        processor.save_pretrained(cls.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_feature_extractor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).feature_extractor

    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)
