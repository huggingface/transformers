# Copyright 2020 The SqueezeBert authors and The HuggingFace Inc. team.
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

from functools import lru_cache

from transformers import SqueezeBertTokenizer, SqueezeBertTokenizerFast
from transformers.testing_utils import require_tokenizers, slow

from ...test_tokenization_common import use_cache_if_possible

# Avoid import `BertTokenizationTest` directly as it will run as `test_tokenization_squeezebert.py::BertTokenizationTest`
# together with `test_tokenization_bert.py::BertTokenizationTest`.
from ..bert import test_tokenization_bert


@require_tokenizers
class SqueezeBertTokenizationTest(test_tokenization_bert.BertTokenizationTest):
    tokenizer_class = SqueezeBertTokenizer
    rust_tokenizer_class = SqueezeBertTokenizerFast
    test_rust_tokenizer = True
    from_pretrained_id = "squeezebert/squeezebert-uncased"

    @classmethod
    @use_cache_if_possible
    @lru_cache(maxsize=64)
    def get_rust_tokenizer(cls, pretrained_name=None, **kwargs):
        pretrained_name = pretrained_name or cls.tmpdirname
        return SqueezeBertTokenizerFast.from_pretrained(pretrained_name, **kwargs)

    @slow
    def test_sequence_builders(self):
        tokenizer = SqueezeBertTokenizer.from_pretrained("squeezebert/squeezebert-mnli-headless")

        text = tokenizer.encode("sequence builders", add_special_tokens=False)
        text_2 = tokenizer.encode("multi-sequence build", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id] + text_2 + [
            tokenizer.sep_token_id
        ]
