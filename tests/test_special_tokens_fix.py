"""
Standalone test for the _set_model_specific_special_tokens fix.
Uses a locally-created BertTokenizer to avoid Hub downloads.
"""
import json
import os
import shutil
import tempfile
import unittest

from transformers import BertTokenizer

from .test_tokenization_common import TokenizerTesterMixin


def _create_local_bert_tokenizer(tmpdir):
    """Create a minimal BertTokenizer saved locally (no Hub access needed)."""
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    for c in "abcdefghijklmnopqrstuvwxyz":
        tokens.append(c)
    for w in ["the", "is", "a", "test", "hello", "world", "##s", "##ing", "##ed"]:
        tokens.append(w)

    with open(os.path.join(tmpdir, "vocab.txt"), "w") as f:
        for t in tokens:
            f.write(t + "\n")

    config = {
        "model_type": "bert",
        "tokenizer_class": "BertTokenizer",
        "do_lower_case": True,
    }
    with open(os.path.join(tmpdir, "tokenizer_config.json"), "w") as f:
        json.dump(config, f)

    tok = BertTokenizer(os.path.join(tmpdir, "vocab.txt"))
    tok.save_pretrained(tmpdir)
    return tmpdir


class TestSetModelSpecificSpecialTokens(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = BertTokenizer
    from_pretrained_id = []  # empty — no Hub downloads

    @classmethod
    def setUpClass(cls):
        cls.tokenizers_list = []
        fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        with open(os.path.join(fixtures_dir, "sample_text.txt"), encoding="utf-8") as f:
            cls._data = f.read().replace("\n\n", "\n").strip()

        cls.tmpdirname = tempfile.mkdtemp()
        _create_local_bert_tokenizer(cls.tmpdirname)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
