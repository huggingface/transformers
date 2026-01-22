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


from transformers import AutoTokenizer
from transformers.models.distilbert.tokenization_distilbert import DistilBertTokenizer
from transformers.testing_utils import require_tokenizers

from ..bert import test_tokenization_bert


# TODO: Ita remove this test file?
@require_tokenizers
class DistilBertTokenizationTest(test_tokenization_bert.BertTokenizationTest):
    tokenizer_class = DistilBertTokenizer
    rust_tokenizer_class = DistilBertTokenizer
    test_rust_tokenizer = False
    from_pretrained_id = "distilbert/distilbert-base-uncased"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "distilbert/distilbert-base-uncased"

        tok_auto = AutoTokenizer.from_pretrained(from_pretrained_id)
        tok_auto.save_pretrained(cls.tmpdirname)

        cls.tokenizers = [tok_auto]
