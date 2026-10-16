import json
import os
import tempfile
import unittest

from transformers import AutoTokenizer


class TokenizerPreTokenizerMismatchWarningTest(unittest.TestCase):
    def test_pretokenizer_mismatch_warning(self):
        # Dummy tokenizer.json whose pre_tokenizer is ByteLevel, while
        # LlamaTokenizerFast hardcodes a Metaspace pre-tokenizer.
        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True},
            "post_processor": None,
            "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True},
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": None,
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "byte_fallback": False,
                "vocab": {"a": 0, "b": 1, "ab": 2},
                "merges": ["a b"],
            },
        }

        tokenizer_config = {"tokenizer_class": "LlamaTokenizerFast"}

        with tempfile.TemporaryDirectory() as tmpdirname:
            with open(os.path.join(tmpdirname, "tokenizer.json"), "w", encoding="utf-8") as f:
                json.dump(tokenizer_json, f)
            with open(os.path.join(tmpdirname, "tokenizer_config.json"), "w", encoding="utf-8") as f:
                json.dump(tokenizer_config, f)

            # Loading this should warn: LlamaTokenizerFast builds a Metaspace
            # pre-tokenizer, but the JSON declares ByteLevel.
            with self.assertLogs("transformers.tokenization_utils_tokenizers", level="WARNING") as cm:
                AutoTokenizer.from_pretrained(tmpdirname)

            self.assertTrue(
                any("pre-tokenizer" in log and "ByteLevel" in log for log in cm.output),
                f"Expected pre-tokenizer mismatch warning, got: {cm.output}",
            )
