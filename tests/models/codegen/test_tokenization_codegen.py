import json
import os
import tempfile
import time
import unittest

from tests.test_tokenization_common import TokenizerTesterMixin
from transformers.models.codegen.tokenization_codegen import CodeGenTokenizer
from transformers.testing_utils import (
    require_tokenizers,
)


@require_tokenizers
class CodeGenTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = ["Salesforce/codegen-350M-mono"]
    tokenizer_class = CodeGenTokenizer

    integration_expected_tokens = ['This', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ92', '000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁ', 'æ', '´', '»', 'çļĦ', 'çľ', 'Ł', 'è', '°', 'Ľ', 'æĺ¯', 'Ċ', 'Hi', '  ', 'Hello', 'Ċ', 'Hi', '   ', 'Hello', 'ĊĊ', 'Ġ', 'Ċ', '  ', 'Ċ', 'ĠHello', 'Ċ', '<', 's', '>', 'Ċ', 'hi', '<', 's', '>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸', 'Ľ', 'à¸', 'µ', '   ', 'ird', '   ', 'à¸', 'Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_token_ids = [1212, 318, 257, 1332, 30325, 232, 198, 40, 373, 4642, 287, 10190, 830, 11, 290, 428, 318, 27807, 2634, 13, 198, 37955, 162, 112, 119, 21410, 40367, 253, 164, 108, 249, 42468, 198, 17250, 50286, 15496, 198, 17250, 50285, 15496, 628, 220, 198, 50286, 198, 18435, 198, 27, 82, 29, 198, 5303, 27, 82, 29, 8117, 198, 464, 1708, 4731, 815, 307, 6105, 30240, 25, 18435, 13, 198, 1537, 220, 1447, 290, 220, 19567, 249, 19567, 113, 50285, 1447, 50285, 19567, 242, 198, 10814, 703, 389, 345, 1804]  # fmt: skip
    expected_tokens_from_ids = ['This', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ92', '000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁ', 'æ', '´', '»', 'çļĦ', 'çľ', 'Ł', 'è', '°', 'Ľ', 'æĺ¯', 'Ċ', 'Hi', '  ', 'Hello', 'Ċ', 'Hi', '   ', 'Hello', 'ĊĊ', 'Ġ', 'Ċ', '  ', 'Ċ', 'ĠHello', 'Ċ', '<', 's', '>', 'Ċ', 'hi', '<', 's', '>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸', 'Ľ', 'à¸', 'µ', '   ', 'ird', '   ', 'à¸', 'Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing"


class CodeGenTruncateTest(unittest.TestCase):
    def test_truncate_is_redos_safe_and_matches_literal_stop(self):
        # `truncate_before_pattern` is caller-controlled; entries are matched as literal strings
        # so previously catastrophic regex patterns like `(a+)+$` cannot trigger backtracking.
        # Pre-fix, a 28-char input already took ~18s; cap generously at 1s.
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "vocab.json"), "w") as f:
                json.dump({"a": 0, "X": 1}, f)
            with open(os.path.join(d, "merges.txt"), "w") as f:
                f.write("#version: 0.2\n")
            tok = CodeGenTokenizer(os.path.join(d, "vocab.json"), os.path.join(d, "merges.txt"))
            start = time.perf_counter()
            tok.truncate("a" * 200 + "X", ["(a+)+$"])
            self.assertLess(time.perf_counter() - start, 1.0)
            self.assertEqual(tok.truncate("hello<|endoftext|>more", ["<|endoftext|>"]), "hello")
