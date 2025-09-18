import os
import tempfile
import unittest

from transformers.models.blueberry.tokenization_blueberry import (
    BlueberryTokenizer,
    BlueberryTokenizerFast,
)


class BlueberryTokenizerTest(unittest.TestCase):
    def setUp(self):
        # Minimal fake vocab/merges for GPT2-style tokenizers
        self.tmpdir = tempfile.TemporaryDirectory()
        vocab_path = os.path.join(self.tmpdir.name, "vocab.json")
        merges_path = os.path.join(self.tmpdir.name, "merges.txt")
        with open(vocab_path, "w", encoding="utf-8") as vf:
            vf.write("{\n  \"<|endoftext|>\":0, \"hello\":1, \"world\":2\n}\n")
        with open(merges_path, "w", encoding="utf-8") as mf:
            mf.write("#version: 0.2\n\nh e\nl l\nlo </w>\n")
        self.vocab_file = vocab_path
        self.merges_file = merges_path

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_encode_decode_roundtrip(self):
        tok = BlueberryTokenizer(vocab_file=self.vocab_file, merges_file=self.merges_file)
        text = "hello world"
        ids = tok(text)["input_ids"]
        decoded = tok.decode(ids)
        self.assertIsInstance(ids, list)
        self.assertIsInstance(decoded, str)

    def test_harmony_chat_template_presence(self):
        tok = BlueberryTokenizer(vocab_file=self.vocab_file, merges_file=self.merges_file)
        self.assertTrue(hasattr(tok, "chat_template"))
        template = tok.chat_template
        self.assertIn("<|start|>", template)
        self.assertIn("<|assistant|>", template)


class BlueberryTokenizerFastTest(unittest.TestCase):
    def setUp(self):
        # Use slow to build then convert to fast via save/load tokenizer.json is out of scope for unit
        self.tmpdir = tempfile.TemporaryDirectory()
        vocab_path = os.path.join(self.tmpdir.name, "vocab.json")
        merges_path = os.path.join(self.tmpdir.name, "merges.txt")
        with open(vocab_path, "w", encoding="utf-8") as vf:
            vf.write("{\n  \"<|endoftext|>\":0, \"hello\":1, \"world\":2\n}\n")
        with open(merges_path, "w", encoding="utf-8") as mf:
            mf.write("#version: 0.2\n\nh e\nl l\nlo </w>\n")
        self.vocab_file = vocab_path
        self.merges_file = merges_path

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_fast_init_and_chat_template(self):
        tok = BlueberryTokenizerFast(vocab_file=self.vocab_file, merges_file=self.merges_file)
        self.assertTrue(hasattr(tok, "chat_template"))
        self.assertIn("<|start|>", tok.chat_template)


if __name__ == "__main__":
    unittest.main()

