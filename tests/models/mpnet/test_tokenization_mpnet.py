# Copyright 2020 The HuggingFace Inc. team, Microsoft Corporation.
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

from transformers import AutoTokenizer
from transformers.models.mpnet.tokenization_mpnet import MPNetTokenizer
from transformers.testing_utils import require_tokenizers, slow

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class MPNetTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "microsoft/mpnet-base"
    tokenizer_class = MPNetTokenizer
    test_rust_tokenizer = False
    space_between_special_tokens = True

    # Integration test data - expected outputs for the default input string
    integration_expected_tokens = None  # Set to None to skip integration test for now
    integration_expected_token_ids = None
    integration_expected_decoded_text = '[UNK] is a test [UNK] [UNK] was born in 92000, and this is [UNK]. 生 [UNK] 的 真 [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] <s> hi <s> there [UNK] following string should be properly encoded : [UNK]. [UNK] ird and [UNK] ird [UNK] [UNK] how are you doing'
    integration_expected_text_from_tokens = '[UNK] is a test [UNK] [UNK] was born in 92000, and this is [UNK]. 生 [UNK] 的 真 [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] <s> hi <s> there [UNK] following string should be properly encoded : [UNK]. [UNK] ird and [UNK] ird [UNK] [UNK] how are you doing'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Load a pretrained tokenizer for testing
        from_pretrained_id = "microsoft/mpnet-base"
        
        tokenizer = AutoTokenizer.from_pretrained(from_pretrained_id)
        tokenizer.save_pretrained(cls.tmpdirname)

        # Build backend for tokenizer from vocab
        vocab = tokenizer.get_vocab()
        tokenizer_from_vocab = MPNetTokenizer(vocab=vocab)

        cls.tokenizers = [tokenizer, tokenizer_from_vocab]

    def get_input_output_texts(self, tokenizer):
        input_text = "UNwant\u00e9d,running"
        output_text = "unwanted, running"
        return input_text, output_text

    def get_tokenizers(self, **kwargs):
        return super().get_tokenizers(**kwargs)
