import unittest
import tempfile
from transformers import AutoTokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin
from transformers.models.blenderbot.tokenization_blenderbot import BlenderbotTokenizer


# Master input string of combined test cases
input_string = """This is a test
I was born in 92000, and this is falsé.
生活的真谛是
Hi  Hello
Hi   Hello

 
  
 Hello
<s>
hi<s>there
The following string should be properly encoded: Hello.
But ird and ปี   ird   ด
Hey how are you doing"""

# Inlined from expected_tokens_belnderbot.py
expected_tokens = ['ĠThis', 'Ġis', 'Ġa', 'Ġtest', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ9', '2', '000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġf', 'als', 'Ã©', '.', 'Ċ', 'ç', 'Ķ', 'Ł', 'æ', '´', '»', 'ç', 'ļ', 'Ħ', 'ç', 'ľ', 'Ł', 'è', '°', 'Ľ', 'æ', 'ĺ', '¯', 'Ċ', 'H', 'i', 'Ġ', 'ĠHello', 'Ċ', 'H', 'i', 'Ġ', 'Ġ', 'ĠHello', 'Ċ', 'Ċ', 'Ġ', 'Ċ', 'Ġ', 'Ġ', 'Ċ', 'ĠHello', 'Ċ', '<s>', 'Ġ', 'Ċ', 'hi', '<s>', 'Ġthere', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġenc', 'od', 'ed', ':', 'ĠHello', '.', 'Ċ', 'B', 'ut', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à', '¸', 'Ľ', 'à', '¸', 'µ', 'Ġ', 'Ġ', 'Ġ', 'ird', 'Ġ', 'Ġ', 'Ġ', 'à', '¸', 'Ķ', 'Ċ', 'H', 'ey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']
expected_token_ids = [678, 315, 265, 1689, 206, 48, 372, 3647, 302, 1207, 25, 1694, 19, 298, 381, 315, 284, 1095, 3952, 21, 206, 171, 250, 261, 170, 120, 127, 171, 256, 234, 171, 258, 261, 172, 116, 257, 170, 254, 115, 206, 47, 80, 228, 6950, 206, 47, 80, 228, 228, 6950, 206, 206, 228, 206, 228, 228, 206, 6950, 206, 1, 228, 206, 7417, 1, 505, 206, 2839, 3504, 7884, 636, 310, 3867, 2525, 621, 296, 33, 6950, 21, 206, 41, 329, 228, 1221, 298, 228, 164, 124, 257, 164, 124, 121, 228, 228, 228, 1221, 228, 228, 228, 164, 124, 250, 206, 47, 3110, 544, 366, 304, 929, 2]


@require_tokenizers
class BlenderbotTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = ["facebook/blenderbot-3B"]
    tokenizer_class = BlenderbotTokenizer
    rust_tokenizer_class = BlenderbotTokenizer
    test_rust_tokenizer = False
    test_sentencepiece = True
    test_sentencepiece_ignore_case = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from_pretrained_id = "facebook/blenderbot-3B"
        tok_auto = AutoTokenizer.from_pretrained(from_pretrained_id)
        tok_auto.pad_token = getattr(tok_auto, "pad_token", None) or getattr(tok_auto, "eos_token", None)
        tok_auto.save_pretrained(cls.tmpdirname)
        cls.tokenizers = [tok_auto]

    def test_integration_expected_tokens(self):
        for tok in self.tokenizers:
            self.assertEqual(tok.tokenize(input_string), expected_tokens)

    def test_integration_expected_token_ids(self):
        for tok in self.tokenizers:
            self.assertEqual(tok.encode(input_string), expected_token_ids)

    @unittest.skip
    def test_pretokenized_inputs(self, *args, **kwargs):
        # It's very difficult to mix/test pretokenization with byte-level tokenizers
        # The issue is that when you have a sequence with leading spaces, splitting it
        # with .split() loses the leading spaces, so the tokenization results differ
        pass

    def test_save_and_reload(self):
        for tok in self.tokenizers:
            with self.subTest(f"{tok.__class__.__name__}"):
                original_tokens = tok.tokenize(input_string)
                original_ids = tok.encode(input_string)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    tok.save_pretrained(tmp_dir)
                    reloaded_tok = tok.__class__.from_pretrained(tmp_dir)
                    reloaded_tokens = reloaded_tok.tokenize(input_string)
                    reloaded_ids = reloaded_tok.encode(input_string)
                    self.assertEqual(original_tokens, reloaded_tokens)
                    self.assertEqual(original_ids, reloaded_ids)

    def test_tokenization_for_chat(self):
        tok = self.tokenizers[0]
        test_chats = [
            [{"role": "system", "content": "You are a helpful chatbot."}, {"role": "user", "content": "Hello!"}],
            [
                {"role": "system", "content": "You are a helpful chatbot."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Nice to meet you."},
            ],
            [{"role": "assistant", "content": "Nice to meet you."}, {"role": "user", "content": "Hello!"}],
        ]
        tokenized_chats = [tok.apply_chat_template(test_chat) for test_chat in test_chats]
        expected_tokens = [
            [553, 366, 265, 4792, 3879, 73, 311, 21, 228, 228, 6950, 8, 2],
            [553, 366, 265, 4792, 3879, 73, 311, 21, 228, 228, 6950, 8, 228, 3490, 287, 2273, 304, 21, 2],
            [3490, 287, 2273, 304, 21, 228, 228, 6950, 8, 2],
        ]
        for tokenized_chat, expected_tokens in zip(tokenized_chats, expected_tokens):
            self.assertListEqual(tokenized_chat, expected_tokens)
