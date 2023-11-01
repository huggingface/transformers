# coding=utf-8
# Copyright 2018 HuggingFace Inc..
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
"""
isort:skip_file
"""
import os
import pickle
import tempfile
import unittest
from unittest.mock import patch
import transformers
from transformers import AutoTokenizer, AutoConfig
from transformers.testing_utils import slow
from transformers.utils import cached_file

# mock cached_file function to remove "tokenizer_config_file" keys, ensure the entire call pipeline contains AutoConfig.from_pretrained
def mock_cached_file(*args, **kwargs):
    resolved_vocab_files = cached_file(*args, **kwargs)
    _ = resolved_vocab_files.pop("tokenizer_config_file", None)
    return resolved_vocab_files

#mock AutoConfig.from_pretrained to check the parameters is passed
original_from_pretrained = transformers.AutoConfig.from_pretrained
def mock_autoconfig_from_pretrained(*args, **kwargs):
    config = original_from_pretrained(*args, **kwargs)
    assert (kwargs['special_arg1'] == "value1")
    assert (kwargs['special_arg2'] == "value2")
    return config

class TokenizationUtilsBaseTest(unittest.TestCase):

    @slow
    @patch('transformers.utils.hub.cached_file', side_effect=mock_cached_file)
    @patch('transformers.AutoConfig.from_pretrained', side_effect=mock_autoconfig_from_pretrained)
    def test_from_pretrained_with_kwargs(self, mock_cached_file, mock_autoconfig):
        # additional kwargs
        additional_kwargs = {
            "special_arg1": "value1",
            "special_arg2": "value2"
        }
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            **additional_kwargs
        )
        self.assertIsNotNone(tokenizer)

if __name__ == '__main__':
    unittest.main()
