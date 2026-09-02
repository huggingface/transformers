# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import tempfile
import unittest

from transformers import Siglip2Tokenizer
from transformers.testing_utils import require_tokenizers


@require_tokenizers
class Siglip2TokenizerTest(unittest.TestCase):
    """
    Integration test for Siglip2Tokenizer:
    - verify hub loading,
    - default lowercasing behavior,
    - save/load roundtrip.
    """

    from_pretrained_id = "google/siglip2-base-patch16-224"

    def test_tokenizer(self):
        tokenizer = Siglip2Tokenizer.from_pretrained(self.from_pretrained_id)

        texts_uc = [
            "HELLO WORLD!",
            "Hello   World!!",
            "A Picture Of Zürich",
            "San Francisco",
            "MIXED-case: TeSt 123",
        ]
        texts_lc = [t.lower() for t in texts_uc]

        # default lowercasing (single + batch paths)
        for t_uc, t_lc in zip(texts_uc, texts_lc):
            with self.subTest(text=t_uc):
                enc_uc = tokenizer(t_uc, truncation=True)
                enc_lc = tokenizer(t_lc, truncation=True)
                self.assertListEqual(enc_uc["input_ids"], enc_lc["input_ids"])

        batch_uc = tokenizer(texts_uc, truncation=True)
        batch_lc = tokenizer(texts_lc, truncation=True)
        self.assertListEqual(batch_uc["input_ids"], batch_lc["input_ids"])

        # padding/truncation path (avoid relying on model_max_length)
        max_len = 64
        padded = tokenizer(texts_uc, padding="max_length", truncation=True, max_length=max_len)
        # ensure every sequence is padded/truncated to max_len
        for seq in padded["input_ids"]:
            self.assertEqual(len(seq), max_len)

        # save/load roundtrip preserves behavior
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save_pretrained(tmpdir)
            tokenizer_reloaded = Siglip2Tokenizer.from_pretrained(tmpdir)

            batch_uc_2 = tokenizer_reloaded(texts_uc, truncation=True)
            batch_lc_2 = tokenizer_reloaded(texts_lc, truncation=True)
            self.assertListEqual(batch_uc_2["input_ids"], batch_lc_2["input_ids"])
            self.assertListEqual(batch_uc["input_ids"], batch_uc_2["input_ids"])

            padded_2 = tokenizer_reloaded(texts_uc, padding="max_length", truncation=True, max_length=max_len)
            for seq in padded_2["input_ids"]:
                self.assertEqual(len(seq), max_len)
