import unittest

from transformers.models.clip.tokenization_clip import CLIPTokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class CLIPTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "openai/clip-vit-base-patch32"
    tokenizer_class = CLIPTokenizer

    integration_expected_tokens = ['this</w>', 'is</w>', 'a</w>', 'test</w>', 'ðŁĺĬ</w>', 'i</w>', 'was</w>', 'born</w>', 'in</w>', '9</w>', '2</w>', '0</w>', '0</w>', '0</w>', ',</w>', 'and</w>', 'this</w>', 'is</w>', 'fal', 's', 'Ã©</w>', '.</w>', 'çĶŁ', 'æ', '´', '»', 'ç', 'ļ', 'Ħ', 'çľŁ', 'è', '°', 'Ľ', 'æĺ', '¯</w>', 'hi</w>', 'hello</w>', 'hi</w>', 'hello</w>', 'hello</w>', '<</w>', 's</w>', '></w>', 'hi</w>', '<</w>', 's</w>', '></w>', 'there</w>', 'the</w>', 'following</w>', 'string</w>', 'should</w>', 'be</w>', 'properly</w>', 'en', 'coded</w>', ':</w>', 'hello</w>', '.</w>', 'but</w>', 'ird</w>', 'and</w>', 'à¸', 'Ľ</w>', 'à¸µ</w>', 'ird</w>', 'à¸Ķ</w>', 'hey</w>', 'how</w>', 'are</w>', 'you</w>', 'doing</w>']  # fmt: skip
    integration_expected_token_ids = [589, 533, 320, 1628, 3020, 328, 739, 2683, 530, 280, 273, 271, 271, 271, 267, 537, 589, 533, 2778, 82, 4166, 269, 33375, 162, 112, 119, 163, 248, 226, 41570, 164, 108, 249, 42891, 363, 1883, 3306, 1883, 3306, 3306, 283, 338, 285, 1883, 283, 338, 285, 997, 518, 3473, 9696, 1535, 655, 12560, 524, 33703, 281, 3306, 269, 767, 2770, 537, 1777, 505, 20278, 2770, 38825, 2189, 829, 631, 592, 1960]  # fmt: skip
    expected_tokens_from_ids = ['this</w>', 'is</w>', 'a</w>', 'test</w>', 'ðŁĺĬ</w>', 'i</w>', 'was</w>', 'born</w>', 'in</w>', '9</w>', '2</w>', '0</w>', '0</w>', '0</w>', ',</w>', 'and</w>', 'this</w>', 'is</w>', 'fal', 's', 'Ã©</w>', '.</w>', 'çĶŁ', 'æ', '´', '»', 'ç', 'ļ', 'Ħ', 'çľŁ', 'è', '°', 'Ľ', 'æĺ', '¯</w>', 'hi</w>', 'hello</w>', 'hi</w>', 'hello</w>', 'hello</w>', '<</w>', 's</w>', '></w>', 'hi</w>', '<</w>', 's</w>', '></w>', 'there</w>', 'the</w>', 'following</w>', 'string</w>', 'should</w>', 'be</w>', 'properly</w>', 'en', 'coded</w>', ':</w>', 'hello</w>', '.</w>', 'but</w>', 'ird</w>', 'and</w>', 'à¸', 'Ľ</w>', 'à¸µ</w>', 'ird</w>', 'à¸Ķ</w>', 'hey</w>', 'how</w>', 'are</w>', 'you</w>', 'doing</w>']  # fmt: skip
    integration_expected_decoded_text = "this is a test 😊 i was born in 9 2 0 0 0 , and this is falsé . 生活的真谛是 hi hello hi hello hello < s > hi < s > there the following string should be properly encoded : hello . but ird and ป ี ird ด hey how are you doing"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from_pretrained_id = "openai/clip-vit-base-patch32"
        tokenizer = CLIPTokenizer.from_pretrained(from_pretrained_id)
        tokenizer.pad_token = getattr(tokenizer, "pad_token", None) or getattr(tokenizer, "eos_token", None)
        tokenizer.save_pretrained(cls.tmpdirname)

        vocab = ["l", "o", "w", "e", "r", "s", "t", "i", "d", "n", "lo", "l</w>", "w</w>", "r</w>", "t</w>", "low</w>", "er</w>", "lowest</w>", "newer</w>", "wider", "<unk>", "<|startoftext|>", "<|endoftext|>"]  # fmt: skip
        cls.vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges_raw = ["#version: 0.2", "l o", "lo w</w>", "e r</w>"]
        cls.special_tokens_map = {"unk_token": "<unk>"}

        cls.merges = []
        for line in merges_raw:
            line = line.strip()
            if line and not line.startswith("#"):
                cls.merges.append(tuple(line.split()))

        tokenizer_from_vocab = CLIPTokenizer(vocab=cls.vocab_tokens, merges=cls.merges)

        cls.tokenizers = [tokenizer, tokenizer_from_vocab]

    def test_padding_to_multiple_of(self):
        self.skipTest("Skipping padding to multiple of test bc vocab is too small.")
