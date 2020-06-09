# coding=utf-8
# Copyright 2020 Huggingface
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


import os
import unittest
from pathlib import Path
from shutil import copyfile

from transformers import MBartTokenizer
from transformers.tokenization_utils import BatchEncoding

from .test_tokenization_common import TokenizerTesterMixin
from .utils import slow


SAMPLE_SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/test_sentencepiece.model")

class MBartTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = MBartTokenizer

    def setUp(self):
        super().setUp()
        save_dir = Path(self.tmpdirname)
        if not (save_dir / "sentencepiece.bpe.model").exists():
            copyfile(SAMPLE_SP, save_dir / "sentencepiece.bpe.model")
        tokenizer = MBartTokenizer.from_pretrained(self.tmpdirname)
        tokenizer.save_pretrained(self.tmpdirname)
    
