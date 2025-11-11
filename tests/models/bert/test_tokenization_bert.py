# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from transformers import AutoTokenizer, BertTokenizer
from transformers.models.bert.tokenization_bert import (
    BertTokenizer,
)
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin

@require_read_token
@require_tokenizers
class BertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = ["google-bert/bert-base-uncased"]
    tokenizer_class = BertTokenizer

    integration_expected_tokens = ['[UNK]', 'is', 'a', 'test', '[UNK]', '[UNK]', 'was', 'born', 'in', '92', '##00', '##0', ',', 'and', 'this', 'is', '[UNK]', '.', '生', '[UNK]', '的', '真', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '<', 's', '>', 'hi', '<', 's', '>', 'there', '[UNK]', 'following', 'string', 'should', 'be', 'properly', 'encoded', ':', '[UNK]', '.', '[UNK]', 'ir', '##d', 'and', '[UNK]', 'ir', '##d', '[UNK]', '[UNK]', 'how', 'are', 'you', 'doing']
    integration_expected_token_ids = [100, 2003, 1037, 3231, 100, 100, 2001, 2141, 1999, 6227, 8889, 2692, 1010, 1998, 2023, 2003, 100, 1012, 1910, 100, 1916, 1921, 100, 100, 100, 100, 100, 100, 100, 1026, 1055, 1028, 7632, 1026, 1055, 1028, 2045, 100, 2206, 5164, 2323, 2022, 7919, 12359, 1024, 100, 1012, 100, 20868, 2094, 1998, 100, 20868, 2094, 100, 100, 2129, 2024, 2017, 2725]
    expected_tokens_from_ids = ['[UNK]', 'is', 'a', 'test', '[UNK]', '[UNK]', 'was', 'born', 'in', '92', '##00', '##0', ',', 'and', 'this', 'is', '[UNK]', '.', '生', '[UNK]', '的', '真', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '<', 's', '>', 'hi', '<', 's', '>', 'there', '[UNK]', 'following', 'string', 'should', 'be', 'properly', 'encoded', ':', '[UNK]', '.', '[UNK]', 'ir', '##d', 'and', '[UNK]', 'ir', '##d', '[UNK]', '[UNK]', 'how', 'are', 'you', 'doing']
    integration_expected_decoded_text = '[UNK] is a test [UNK] [UNK] was born in 92000, and this is [UNK]. 生 [UNK] 的 真 [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] < s > hi < s > there [UNK] following string should be properly encoded : [UNK]. [UNK] ird and [UNK] ird [UNK] [UNK] how are you doing'

    def _run_integration_checks(self, tokenizer, tokenizer_type):
        try:
            return super()._run_integration_checks(tokenizer, tokenizer_type)
        except AssertionError as err:
            decoded = tokenizer.decode(
                self.integration_expected_token_ids, clean_up_tokenization_spaces=False
            )
            decoder_state = "unavailable"
            tokenizer_class = type(tokenizer).__name__
            tokenizer_module = type(tokenizer).__module__
            has_backend_tokenizer = hasattr(tokenizer, "backend_tokenizer")
            has_tokenizer_attr = hasattr(tokenizer, "_tokenizer")
            backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None) or getattr(tokenizer, "_tokenizer", None)
            
            # Additional debugging info
            tokenizer_mro = [cls.__name__ for cls in type(tokenizer).__mro__]
            all_attrs = [attr for attr in dir(tokenizer) if not attr.startswith("__")]
            
            if backend_tokenizer is not None and hasattr(backend_tokenizer, "decoder"):
                decoder_state = backend_tokenizer.decoder.__getstate__()
                if isinstance(decoder_state, bytes):
                    decoder_state = decoder_state.decode("utf-8", errors="replace")
            
            debug_info = (
                f"{err}\nBERT tokenizer debug info:\n"
                f"  tokenizer_class: {tokenizer_class}\n"
                f"  tokenizer_module: {tokenizer_module}\n"
                f"  tokenizer_mro: {tokenizer_mro}\n"
                f"  has_backend_tokenizer attr: {has_backend_tokenizer}\n"
                f"  has_tokenizer attr: {has_tokenizer_attr}\n"
                f"  backend_tokenizer value: {backend_tokenizer}\n"
                f"  decoder_state: {decoder_state}\n"
                f"  decoded: {decoded!r}\n"
                f"  tokenizer_type param: {tokenizer_type}"
            )
            raise AssertionError(debug_info) from err
