# Copyright 2020 The HuggingFace Team. All rights reserved.
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


import unittest

from transformers import AutoTokenizer, GPT2Tokenizer
from transformers.testing_utils import require_tiktoken, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class GPT2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = ["openai-community/gpt2"]
    tokenizer_class = GPT2Tokenizer
    from_pretrained_kwargs = {"add_prefix_space": False}

    integration_expected_tokens = ['This', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ92', '000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁ', 'æ', '´', '»', 'çļĦ', 'çľ', 'Ł', 'è', '°', 'Ľ', 'æĺ¯', 'Ċ', 'Hi', 'Ġ', 'ĠHello', 'Ċ', 'Hi', 'Ġ', 'Ġ', 'ĠHello', 'ĊĊ', 'Ġ', 'Ċ', 'Ġ', 'Ġ', 'Ċ', 'ĠHello', 'Ċ', '<', 's', '>', 'Ċ', 'hi', '<', 's', '>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸', 'Ľ', 'à¸', 'µ', 'Ġ', 'Ġ', 'Ġ', 'ird', 'Ġ', 'Ġ', 'Ġ', 'à¸', 'Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_token_ids = [1212, 318, 257, 1332, 30325, 232, 198, 40, 373, 4642, 287, 10190, 830, 11, 290, 428, 318, 27807, 2634, 13, 198, 37955, 162, 112, 119, 21410, 40367, 253, 164, 108, 249, 42468, 198, 17250, 220, 18435, 198, 17250, 220, 220, 18435, 628, 220, 198, 220, 220, 198, 18435, 198, 27, 82, 29, 198, 5303, 27, 82, 29, 8117, 198, 464, 1708, 4731, 815, 307, 6105, 30240, 25, 18435, 13, 198, 1537, 220, 1447, 290, 220, 19567, 249, 19567, 113, 220, 220, 220, 1447, 220, 220, 220, 19567, 242, 198, 10814, 703, 389, 345, 1804]  # fmt: skip
    expected_tokens_from_ids = ['This', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ92', '000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁ', 'æ', '´', '»', 'çļĦ', 'çľ', 'Ł', 'è', '°', 'Ľ', 'æĺ¯', 'Ċ', 'Hi', 'Ġ', 'ĠHello', 'Ċ', 'Hi', 'Ġ', 'Ġ', 'ĠHello', 'ĊĊ', 'Ġ', 'Ċ', 'Ġ', 'Ġ', 'Ċ', 'ĠHello', 'Ċ', '<', 's', '>', 'Ċ', 'hi', '<', 's', '>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸', 'Ľ', 'à¸', 'µ', 'Ġ', 'Ġ', 'Ġ', 'ird', 'Ġ', 'Ġ', 'Ġ', 'à¸', 'Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing"

    @unittest.skip
    def test_pretokenized_inputs(self, *args, **kwargs):
        # It's very difficult to mix/test pretokenization with byte-level
        # And get both GPT2 and Roberta to work at the same time (mostly an issue of adding a space before the string)
        pass

    @unittest.skip(reason="tokenizer has no padding token")
    def test_padding_different_model_input_name(self):
        pass

    def test_special_tokens_mask_input_pairs_and_bos_token(self):
        # TODO: change to self.get_tokenizers() when the fast version is implemented
        tokenizers = [self.get_tokenizer(do_lower_case=False, add_bos_token=True)]
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence_0 = "Encode this."
                sequence_1 = "This one too please."
                encoded_sequence = tokenizer.encode(sequence_0, add_special_tokens=False)
                encoded_sequence += tokenizer.encode(sequence_1, add_special_tokens=False)
                encoded_sequence_dict = tokenizer(
                    sequence_0,
                    sequence_1,
                    add_special_tokens=True,
                    return_special_tokens_mask=True,
                )
                encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
                special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
                self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))

                filtered_sequence = [
                    (x if not special_tokens_mask[i] else None) for i, x in enumerate(encoded_sequence_w_special)
                ]
                filtered_sequence = [x for x in filtered_sequence if x is not None]
                self.assertEqual(encoded_sequence, filtered_sequence)

    @require_tiktoken
    def test_tokenization_tiktoken(self):
        from tiktoken import encoding_name_for_model

        from transformers.integrations.tiktoken import convert_tiktoken_to_fast

        encoding = encoding_name_for_model("gpt2")
        convert_tiktoken_to_fast(encoding, self.tmpdirname)

        tiktoken_fast_tokenizer = GPT2Tokenizer.from_pretrained(self.tmpdirname)
        rust_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        sequence = "lower newer"
        self.assertEqual(
            rust_tokenizer.decode(rust_tokenizer.encode(sequence)),
            tiktoken_fast_tokenizer.decode(rust_tokenizer.encode(sequence)),
        )

    def test_added_tokens_unicode_roundtrip_with_bytelevel(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        new_tokens = ["Začnimo", "kuća", "međa"]
        tokenizer.add_tokens(new_tokens)

        for word in new_tokens:
            with self.subTest(word=word):
                ids = tokenizer.encode(word, add_special_tokens=False)
                decoded = tokenizer.decode(ids, skip_special_tokens=False)
                self.assertEqual(decoded, word)


@require_tokenizers
class OPTTokenizationTest(unittest.TestCase):
    def test_serialize_deserialize_fast_opt(self):
        # More context:
        # https://huggingface.co/wjmcat/opt-350m-paddle/discussions/1
        # https://huggingface.slack.com/archives/C01N44FJDHT/p1653511495183519
        # https://github.com/huggingface/transformers/pull/17088#discussion_r871246439

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        text = "A photo of a cat"

        tokens_ids = tokenizer.encode(
            text,
        )
        self.assertEqual(tokens_ids, [2, 250, 1345, 9, 10, 4758])
        tokenizer.save_pretrained("test_opt")

        tokenizer = AutoTokenizer.from_pretrained("./test_opt")
        tokens_ids = tokenizer.encode(
            text,
        )
        self.assertEqual(tokens_ids, [2, 250, 1345, 9, 10, 4758])

    def test_fast_slow_equivalence(self):
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        text = "A photo of a cat"

        tokens_ids = tokenizer.encode(
            text,
        )
        # Same as above
        self.assertEqual(tokens_ids, [2, 250, 1345, 9, 10, 4758])

    @unittest.skip(reason="This test is failing because of a bug in the fast tokenizer")
    def test_users_can_modify_bos(self):
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        tokenizer.bos_token = "bos"
        tokenizer.bos_token_id = tokenizer.get_vocab()["bos"]

        text = "A photo of a cat"
        tokens_ids = tokenizer.encode(
            text,
        )
        # We changed the bos token
        self.assertEqual(tokens_ids, [31957, 250, 1345, 9, 10, 4758])
        tokenizer.save_pretrained("./tok")
        tokenizer = AutoTokenizer.from_pretrained("./tok")
        self.assertTrue(tokenizer.is_fast)
        tokens_ids = tokenizer.encode(
            text,
        )
        self.assertEqual(tokens_ids, [31957, 250, 1345, 9, 10, 4758])
