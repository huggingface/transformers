import json
import os
import tempfile
import unittest

from tokenizers import pre_tokenizers

from transformers import AutoTokenizer
from transformers.tokenization_utils_tokenizers import _pretokenizer_mismatch


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
                any("pre-tokenizer" in log and "behaves differently" in log for log in cm.output),
                f"Expected pre-tokenizer mismatch warning, got: {cm.output}",
            )


class TokenizerPreTokenizerBehavioralMismatchTest(unittest.TestCase):
    def test_equivalent_pipelines_do_not_warn(self):
        # JSON declares bare Metaspace; a class building Sequence([Metaspace])
        # is behaviorally identical: no warning (no false positive).
        json_pt = {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True}
        built = pre_tokenizers.Sequence([pre_tokenizers.Metaspace()])
        self.assertFalse(_pretokenizer_mismatch(json_pt, built))

    def test_same_type_different_params_warn(self):
        # Both are Sequence, but Digits individual_digits differs (the
        # Cohere/tiny-aya case from #49176): must warn (no false negative).
        json_pt = {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Digits", "individual_digits": False},
                {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
            ],
        }
        built = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), pre_tokenizers.ByteLevel()])
        self.assertTrue(_pretokenizer_mismatch(json_pt, built))

    def test_unbuildable_pretokenizer_never_warns(self):
        # Unknown types must skip the check, never crash loading.
        self.assertFalse(_pretokenizer_mismatch({"type": "Nope"}, pre_tokenizers.Metaspace()))
        self.assertFalse(_pretokenizer_mismatch(None, pre_tokenizers.Metaspace()))
