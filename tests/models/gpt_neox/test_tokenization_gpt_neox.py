# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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

import json
import os
import tempfile
import unittest

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE

from transformers import GPTNeoXTokenizer, TokenizersBackend
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class GPTNeoXTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "EleutherAI/gpt-neox-20b"
    tokenizer_class = GPTNeoXTokenizer
    integration_expected_tokens = ['This', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ9', '2000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁ', 'æ´»', 'çļĦ', 'çľŁ', 'è°', 'Ľ', 'æĺ¯', 'Ċ', 'Hi', '  ', 'Hello', 'Ċ', 'Hi', '   ', 'Hello', 'ĊĊĠĊ', '  ', 'Ċ', 'ĠHello', 'Ċ', '<', 's', '>', 'Ċ', 'hi', '<', 's', '>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸', 'Ľ', 'à¸µ', '   ', 'ird', '   ', 'à¸Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_token_ids = [1552, 310, 247, 1071, 49042, 221, 187, 42, 369, 5686, 275, 898, 6914, 13, 285, 436, 310, 21649, 860, 15, 187, 20025, 46549, 5225, 48561, 33656, 238, 12105, 187, 12764, 50276, 12092, 187, 12764, 50275, 12092, 46603, 50276, 187, 24387, 187, 29, 84, 31, 187, 5801, 29, 84, 31, 9088, 187, 510, 1563, 2876, 943, 320, 6283, 16202, 27, 24387, 15, 187, 1989, 209, 1817, 285, 209, 2869, 238, 26863, 50275, 1817, 50275, 35071, 187, 8262, 849, 403, 368, 2509]  # fmt: skip
    expected_tokens_from_ids = ['This', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ9', '2000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁ', 'æ´»', 'çļĦ', 'çľŁ', 'è°', 'Ľ', 'æĺ¯', 'Ċ', 'Hi', '  ', 'Hello', 'Ċ', 'Hi', '   ', 'Hello', 'ĊĊĠĊ', '  ', 'Ċ', 'ĠHello', 'Ċ', '<', 's', '>', 'Ċ', 'hi', '<', 's', '>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸', 'Ľ', 'à¸µ', '   ', 'ird', '   ', 'à¸Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing"


@require_tokenizers
class GPTNeoXTokenizerConfigFlagsTest(unittest.TestCase):
    """
    GPTNeoXTokenizer rebuilds its backend post-processor from the add_bos_token/
    add_eos_token settings (see #47988). Flags explicitly declared in tokenizer_config.json
    replace the matching side of a post-processor serialized in tokenizer.json; sides the
    flags do not declare keep the serialized behavior. Hub checkpoints exist in either
    polarity: blab-jhu/test-32m-dec declares add_bos_token: true next to a [CLS]/[SEP]
    template, while allenai/OLMo-7B-hf declares add_eos_token: false next to a template that
    appends EOS. When no flags are declared at all, the serialized template applies as-is.
    """

    @staticmethod
    def _prepare_tokenizer_dir(
        tmpdirname, tokenizer_config, add_eos_post_processor=False, add_bos_post_processor=False
    ):
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        vocab = {char: i for i, char in enumerate(alphabet)}
        eos_id = len(vocab)
        vocab["<|endoftext|>"] = eos_id
        vocab["<|padding|>"] = len(vocab)

        backend = Tokenizer(
            BPE(
                vocab=vocab,
                merges=[],
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )
        backend.normalizer = normalizers.NFC()
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True)
        backend.decoder = decoders.ByteLevel()

        if add_eos_post_processor or add_bos_post_processor:
            prefix = "<|padding|>" if add_bos_post_processor else None
            suffix = "<|endoftext|>" if add_eos_post_processor else None
            single = "".join(
                part
                for part in (
                    f"{prefix} $A" if prefix else "$A",
                    f" {suffix}" if suffix else "",
                )
                if part
            )
            pair = "".join(
                part
                for part in (
                    f"{prefix} $A" if prefix else "$A",
                    f" {prefix} $B" if prefix else " $B",
                    f" {suffix}" if suffix else "",
                )
                if part
            )
            special_tokens = []
            if prefix:
                special_tokens.append((prefix, vocab[prefix]))
            if suffix:
                special_tokens.append((suffix, vocab[suffix]))
            backend.post_processor = processors.TemplateProcessing(
                single=single,
                pair=pair,
                special_tokens=special_tokens,
            )

        backend.save(os.path.join(tmpdirname, "tokenizer.json"))
        with open(os.path.join(tmpdirname, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
            json.dump(tokenizer_config, config_file)
        return vocab

    def test_config_add_eos_false_wins_over_stale_post_processor(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            vocab = self._prepare_tokenizer_dir(
                tmpdirname,
                {"add_bos_token": False, "add_eos_token": False},
                add_eos_post_processor=True,
            )
            tokenizer = GPTNeoXTokenizer.from_pretrained(tmpdirname)

        input_ids = tokenizer("hello world")["input_ids"]
        self.assertNotIn(vocab["<|endoftext|>"], input_ids[1:])
        self.assertFalse(tokenizer.add_eos_token)

    def test_declared_bos_applies_while_undeclared_eos_side_is_kept(self):
        # blab-jhu/test-32m-dec case: config declares add_bos_token: true next to a serialized
        # template that also appends a special token; the eos side stays serialized (no
        # add_eos_token is declared), so both the bos prefix and the eos suffix are applied.
        with tempfile.TemporaryDirectory() as tmpdirname:
            vocab = self._prepare_tokenizer_dir(
                tmpdirname,
                {"bos_token": "<|padding|>", "add_bos_token": True},
                add_bos_post_processor=True,
                add_eos_post_processor=True,
            )
            tokenizer = GPTNeoXTokenizer.from_pretrained(tmpdirname)

        input_ids = tokenizer("hello world")["input_ids"]
        self.assertEqual(input_ids[0], vocab["<|padding|>"])
        self.assertEqual(input_ids[-1], vocab["<|endoftext|>"])
        self.assertTrue(tokenizer.add_bos_token)

    def test_serialized_sides_apply_when_flags_absent(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            vocab = self._prepare_tokenizer_dir(
                tmpdirname,
                {},
                add_eos_post_processor=True,
            )
            tokenizer = GPTNeoXTokenizer.from_pretrained(tmpdirname)

        input_ids = tokenizer("hello world")["input_ids"]
        self.assertEqual(input_ids[-1], vocab["<|endoftext|>"])
        self.assertFalse(tokenizer.add_eos_token)

    def test_undeclared_sides_kept_on_tokenizer_object_path(self):
        # Classes without a custom __init__ load their backend through tokenizer_object instead
        # of rebuilding it; the serialized template's undeclared sides must be honored there too.
        class PlainBackend(TokenizersBackend):
            model = BPE
            vocab_files_names = {"tokenizer_file": "tokenizer.json"}

        with tempfile.TemporaryDirectory() as tmpdirname:
            vocab = self._prepare_tokenizer_dir(
                tmpdirname,
                {"bos_token": "<|padding|>", "add_bos_token": True},
                add_bos_post_processor=True,
                add_eos_post_processor=True,
            )
            tokenizer = PlainBackend.from_pretrained(tmpdirname)

        input_ids = tokenizer("hello world")["input_ids"]
        self.assertEqual(input_ids[0], vocab["<|padding|>"])
        self.assertEqual(input_ids[-1], vocab["<|endoftext|>"])
        self.assertTrue(tokenizer.add_bos_token)


if __name__ == "__main__":
    unittest.main()
