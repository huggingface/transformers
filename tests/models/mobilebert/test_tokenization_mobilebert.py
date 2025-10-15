# Copyright 2022 Leon Derczynski. All rights reserved.
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
"""Testing suite for the MobileBERT tokenizer."""

import os
import unittest

from transformers import AutoTokenizer
from transformers.models.mobilebert.tokenization_mobilebert import MobileBertTokenizer

from ..bert import test_tokenization_bert

from transformers.testing_utils import require_tokenizers, slow

from ...test_tokenization_common import TokenizerTesterMixin, filter_non_english


@require_tokenizers
class MobileBERTTokenizationTest(test_tokenization_bert.BertTokenizationTest):
    from_pretrained_id = "google/mobilebert-uncased"
    tokenizer_class = MobileBertTokenizer
    rust_tokenizer_class = MobileBertTokenizer
    test_rust_tokenizer = False
    space_between_special_tokens = True
    from_pretrained_filter = filter_non_english
    pre_trained_model_path = "google/mobilebert-uncased"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "google/mobilebert-uncased"

        tok_auto = AutoTokenizer.from_pretrained(from_pretrained_id)
        tok_auto.save_pretrained(cls.tmpdirname)

        cls.tokenizers = [tok_auto]
