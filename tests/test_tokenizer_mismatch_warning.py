import json
import os
import tempfile
import unittest

from transformers import AutoTokenizer


class TokenizerMismatchWarningTest(unittest.TestCase):
    def test_decoder_mismatch_warning(self):
        # Create a dummy ByteLevel tokenizer.json
        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
            "post_processor": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": False,
                "use_regex": True,
            },
            "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
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

            # Loading this should emit a warning because LlamaTokenizer builds a Sequence decoder,
            # but the JSON has a ByteLevel decoder.
            with self.assertLogs("transformers.tokenization_utils_tokenizers", level="WARNING") as cm:
                AutoTokenizer.from_pretrained(tmpdirname)

            self.assertTrue(
                any(
                    "However, the `tokenizer.json` file found in this checkpoint contains a 'ByteLevel' pipeline." in log
                    for log in cm.output
                )
            )
