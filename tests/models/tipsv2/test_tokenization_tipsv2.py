# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from transformers import AutoTokenizer, Tipsv2Tokenizer
from transformers.testing_utils import require_sentencepiece


def get_tipsv2_test_sentencepiece_model(tmp_dir):
    import os

    import sentencepiece as spm

    corpus_file = os.path.join(tmp_dir, "corpus.txt")
    with open(corpus_file, "w", encoding="utf-8") as fp:
        fp.write(
            "\n".join(
                [
                    "a cat on a mat",
                    "a dog in the fog",
                    "mixed case text for tipsv2 tokenizer",
                    "zuerich san francisco image text alignment",
                    "padding tokens should use id zero",
                ]
            )
        )

    model_prefix = os.path.join(tmp_dir, "tipsv2_test")
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=64,
        model_type="unigram",
        character_coverage=1.0,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=3,
        hard_vocab_limit=False,
        num_threads=1,
    )
    return f"{model_prefix}.model"


@require_sentencepiece
class Tipsv2TokenizerTest(unittest.TestCase):
    def test_tokenizer_defaults_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            vocab_file = get_tipsv2_test_sentencepiece_model(tmp_dir)
            tokenizer = Tipsv2Tokenizer(vocab_file)

            self.assertEqual(tokenizer.pad_token_id, 0)
            self.assertIsNone(tokenizer.bos_token_id)
            self.assertIsNone(tokenizer.eos_token_id)
            self.assertEqual(tokenizer.model_max_length, 64)

            text = "A Cat on a Mat"
            lower_text = text.lower()
            self.assertListEqual(tokenizer(text).input_ids, tokenizer(lower_text).input_ids)

            token_ids = tokenizer(text, add_special_tokens=False).input_ids
            self.assertListEqual(tokenizer.build_inputs_with_special_tokens(token_ids), token_ids)
            self.assertListEqual(tokenizer.get_special_tokens_mask(token_ids), [0] * len(token_ids))

            encoded = tokenizer([text, "A DOG in the Fog"], padding="max_length", truncation=True, max_length=64)
            for input_ids in encoded.input_ids:
                self.assertEqual(len(input_ids), 64)
                self.assertEqual(input_ids[-1], tokenizer.pad_token_id)

            tokenizer.save_pretrained(tmp_dir)
            reloaded = Tipsv2Tokenizer.from_pretrained(tmp_dir)
            auto_reloaded = AutoTokenizer.from_pretrained(tmp_dir)

            self.assertListEqual(reloaded(text).input_ids, tokenizer(text).input_ids)
            self.assertListEqual(auto_reloaded(text).input_ids, tokenizer(text).input_ids)
