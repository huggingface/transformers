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

import json
import os
import tempfile
import unittest

from transformers import Siglip2Tokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


def _write_test_tokenizer_json(path: str):
    """
    Deterministic tokenizer.json for unit tests that passes TokenizerTesterMixin.

    - WordPiece backend (fast tokenizer).
    - Whitespace pre-tokenizer.
    - WordPiece decoder so decode() is stable and space-separated.
    - Vocab includes enough tokens/pieces/punctuation for common tests,
      plus both cases for do_lower_case checks.
    """
    from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
    from tokenizers.models import WordPiece

    pad = "<pad>"
    eos = "(EOS)"
    bos = "(BOS)"
    unk = "<unk>"
    mask = "<mask>"

    vocab = {}
    idx = 0

    def add(tok: str):
        nonlocal idx
        if tok not in vocab:
            vocab[tok] = idx
            idx += 1

    # Special tokens first (stable ids)
    for t in [pad, eos, bos, unk, mask]:
        add(t)

    # Used by get_input_output_texts: "a b c ... t"
    for t in list("abcdefghijklmnopqrst"):
        add(t)

    # Common words used across TokenizerTesterMixin tests
    words = [
        "HELLO",
        "WORLD",
        "hello",
        "world",
        "This",
        "is",
        "another",
        "sentence",
        "to",
        "be",
        "encoded",
        ".",
        "Test",
        "this",
        "method",
        "With",
        "these",
        "inputs",
        "and",
        "some",
        "extra",
        "tokens",
        "here",
        "high-level",
        "end-to-end",
        "state-of-the-art",
        "user's",
        "system's",
        "example",
        "feature",
        "model",
        "pipeline",
        "dataset",
        "process",
        "result",
        "output",
        "value",
        "performance",
        "running",
        "working",
        "tested",
        "validated",
        "improved",
        "simple",
        "complex",
        "happy",
        "fast",
        "robust",
        "He",
        "She",
        "They",
        "it",
    ]
    for w in words:
        add(w)

    # Punctuation as standalone tokens
    for p in [",", "!", "?", "'", '"', "-", "(", ")", ":", ";"]:
        add(p)

    # WordPiece fallback pieces
    for c in "abcdefghijklmnopqrstuvwxyz":
        add(c)
        add(f"##{c}")
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        add(c)
        add(f"##{c}")
    for c in "0123456789":
        add(c)
        add(f"##{c}")

    # Accented char used in common sample_text
    add("é")
    add("##é")

    tok = Tokenizer(WordPiece(vocab=vocab, unk_token=unk, continuing_subword_prefix="##"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece(prefix="##")

    tok.add_special_tokens([pad, eos, bos, unk, mask])
    tok.save(path)


@require_tokenizers
class Siglip2TokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = Siglip2Tokenizer

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._tok_dir = cls._tmpdir.name

        tok_json_path = os.path.join(cls._tok_dir, "tokenizer.json")
        _write_test_tokenizer_json(tok_json_path)

        with open(os.path.join(cls._tok_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump({"tokenizer_class": "Siglip2Tokenizer", "do_lower_case": False}, f)

        with open(os.path.join(cls._tok_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pad_token": "<pad>",
                    "eos_token": "(EOS)",
                    "bos_token": "(BOS)",
                    "unk_token": "<unk>",
                    "mask_token": "<mask>",
                },
                f,
            )

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()
        super().tearDownClass()

    def get_tokenizer(self, **kwargs):
        return self.tokenizer_class.from_pretrained(self._tok_dir, **kwargs)

    def test_lowercasing_is_backend_normalizer(self):
        tok = self.get_tokenizer(do_lower_case=True)
        self.assertEqual(tok("HELLO WORLD")["input_ids"], tok("hello world")["input_ids"])

    def test_do_lower_case_flag(self):
        tok = self.get_tokenizer(do_lower_case=False)
        self.assertNotEqual(tok("HELLO WORLD")["input_ids"], tok("hello world")["input_ids"])
