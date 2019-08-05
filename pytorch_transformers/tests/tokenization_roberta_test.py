# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import unittest
import pytest
import six

from pytorch_transformers.tokenization_roberta import RobertaTokenizer


class RobertaTokenizationTest(unittest.TestCase):

    # @pytest.mark.slow
    def test_full_tokenizer(self):
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.assertListEqual(
            tokenizer.encode('Hello world!'),
            [0, 31414, 232, 328, 2]
        )
        if six.PY3:
            self.assertListEqual(
                tokenizer.encode('Hello world! cécé herlolip'),
                [0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]
            )



if __name__ == '__main__':
    unittest.main()
