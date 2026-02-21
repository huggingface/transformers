import json
import os
import tempfile
import unittest

from tests.test_tokenization_common import TokenizerTesterMixin
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from transformers.testing_utils import (
    require_tokenizers,
)


@require_tokenizers
class LlamaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = [
        "hf-internal-testing/llama-tokenizer",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Meta-Llama-3-8B",
    ]
    tokenizer_class = LlamaTokenizer
    from_pretrained_kwargs = {}

    # Integration test data - expected outputs for the default input string
    integration_expected_tokens = ["▁This", "▁is", "▁a", "▁test", "▁", "<0xF0>", "<0x9F>", "<0x98>", "<0x8A>", "<0x0A>", "I", "▁was", "▁born", "▁in", "▁", "9", "2", "0", "0", "0", ",", "▁and", "▁this", "▁is", "▁f", "als", "é", ".", "<0x0A>", "生", "活", "的", "真", "<0xE8>", "<0xB0>", "<0x9B>", "是", "<0x0A>", "Hi", "▁", "▁Hello", "<0x0A>", "Hi", "▁▁", "▁Hello", "<0x0A>", "<0x0A>", "▁", "<0x0A>", "▁▁", "<0x0A>", "▁Hello", "<0x0A>", "<s>", "<0x0A>", "hi", "<s>", "there", "<0x0A>", "The", "▁following", "▁string", "▁should", "▁be", "▁properly", "▁encoded", ":", "▁Hello", ".", "<0x0A>", "But", "▁", "ird", "▁and", "▁", "ป", "ี", "▁▁▁", "ird", "▁▁▁", "ด", "<0x0A>", "H", "ey", "▁how", "▁are", "▁you", "▁doing"]  # fmt: skip
    integration_expected_token_ids = [910, 338, 263, 1243, 29871, 243, 162, 155, 141, 13, 29902, 471, 6345, 297, 29871, 29929, 29906, 29900, 29900, 29900, 29892, 322, 445, 338, 285, 1338, 29948, 29889, 13, 30486, 31704, 30210, 30848, 235, 179, 158, 30392, 13, 18567, 29871, 15043, 13, 18567, 259, 15043, 13, 13, 29871, 13, 259, 13, 15043, 13, 1, 13, 2918, 1, 12711, 13, 1576, 1494, 1347, 881, 367, 6284, 18511, 29901, 15043, 29889, 13, 6246, 29871, 1823, 322, 29871, 31010, 30691, 1678, 1823, 1678, 30718, 13, 29950, 1032, 920, 526, 366, 2599]  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "hf-internal-testing/llama-tokenizer"

        tokenizer = LlamaTokenizer.from_pretrained(from_pretrained_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(cls.tmpdirname)

    def get_tokenizers(self, **kwargs):
        kwargs.setdefault("pad_token", "<PAD>")
        return super().get_tokenizers(**kwargs)

    def test_bytelevel_decoder_preserved_on_load(self):
        """
        Regression test for https://github.com/huggingface/transformers/issues/43975

        Some models (e.g. deepseek-ai/deepseek-coder-6.7b-instruct) declare
        ``tokenizer_class: LlamaTokenizerFast`` in their tokenizer_config.json but use
        a GPT-2-style ByteLevel decoder rather than the SentencePiece Metaspace decoder
        that ``LlamaTokenizer.__init__`` normally configures.  When loaded with v5, the
        custom ``__init__`` was replacing the correct ByteLevel decoder with a Metaspace
        one, causing all spaces to be silently dropped on decode.
        """
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder

        # Build a minimal ByteLevel BPE tokenizer (same style as deepseek-coder).
        tokenizer_backend = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer_backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer_backend.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=500,
            special_tokens=["<unk>", "<s>", "</s>"],
        )
        tokenizer_backend.train_from_iterator(
            [
                "simple test sentence",
                "hello world foo bar baz",
                "a b c d e f g h i j k l m n o p q r s t u v w x y z",
            ],
            trainer=trainer,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer_backend.save(os.path.join(tmp_dir, "tokenizer.json"))

            # Mimic the tokenizer_config.json that deepseek-coder ships with.
            tokenizer_config = {
                "tokenizer_class": "LlamaTokenizerFast",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "legacy": True,
            }
            with open(os.path.join(tmp_dir, "tokenizer_config.json"), "w") as f:
                json.dump(tokenizer_config, f)

            loaded = LlamaTokenizer.from_pretrained(tmp_dir)

            # The decoder must still be ByteLevel, not replaced by the Metaspace decoder.
            self.assertIsInstance(loaded._tokenizer.decoder, ByteLevelDecoder)

            # Spaces must be preserved in an encode → decode round-trip.
            test_input = "simple test sentence"
            ids = loaded.encode(test_input, add_special_tokens=False)
            decoded = loaded.decode(ids, skip_special_tokens=True)
            self.assertEqual(decoded, test_input)
