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


from transformers import DistilBertTokenizer, DistilBertTokenizerFast
from transformers.testing_utils import require_tokenizers, slow

from ..bert import test_tokenization_bert


@require_tokenizers
class DistilBertTokenizationTest(test_tokenization_bert.BertTokenizationTest):
    tokenizer_class = DistilBertTokenizer
    rust_tokenizer_class = DistilBertTokenizerFast
    test_rust_tokenizer = True
    from_pretrained_id = "distilbert/distilbert-base-uncased"

    @slow
    def test_sequence_builders(self):
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        text = tokenizer.encode("sequence builders", add_special_tokens=False)
        text_2 = tokenizer.encode("multi-sequence build", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id] + text_2 + [
            tokenizer.sep_token_id
        ]
