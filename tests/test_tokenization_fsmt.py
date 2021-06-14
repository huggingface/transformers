# coding=utf-8
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


import json
import os
import unittest

from transformers.file_utils import cached_property
from transformers.models.fsmt.tokenization_fsmt import VOCAB_FILES_NAMES, FSMTTokenizer
from transformers.testing_utils import slow

from .test_tokenization_common import TokenizerTesterMixin


# using a different tiny model than the one used for default params defined in init to ensure proper testing
FSMT_TINY2 = "stas/tiny-wmt19-en-ru"


class FSMTTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = FSMTTokenizer
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "w</w>",
            "r</w>",
            "t</w>",
            "lo",
            "low",
            "er</w>",
            "low</w>",
            "lowest</w>",
            "newer</w>",
            "wider</w>",
            "<unk>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["l o 123", "lo w 1456", "e r</w> 1789", ""]

        self.langs = ["en", "ru"]
        config = {
            "langs": self.langs,
            "src_vocab_size": 10,
            "tgt_vocab_size": 20,
        }

        self.src_vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["src_vocab_file"])
        self.tgt_vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["tgt_vocab_file"])
        config_file = os.path.join(self.tmpdirname, "tokenizer_config.json")
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.src_vocab_file, "w") as fp:
            fp.write(json.dumps(vocab_tokens))
        with open(self.tgt_vocab_file, "w") as fp:
            fp.write(json.dumps(vocab_tokens))
        with open(self.merges_file, "w") as fp:
            fp.write("\n".join(merges))
        with open(config_file, "w") as fp:
            fp.write(json.dumps(config))

    @cached_property
    def tokenizer_ru_en(self):
        return FSMTTokenizer.from_pretrained("facebook/wmt19-ru-en")

    @cached_property
    def tokenizer_en_ru(self):
        return FSMTTokenizer.from_pretrained("facebook/wmt19-en-ru")

    def test_online_tokenizer_config(self):
        """this just tests that the online tokenizer files get correctly fetched and
        loaded via its tokenizer_config.json and it's not slow so it's run by normal CI
        """
        tokenizer = FSMTTokenizer.from_pretrained(FSMT_TINY2)
        self.assertListEqual([tokenizer.src_lang, tokenizer.tgt_lang], ["en", "ru"])
        self.assertEqual(tokenizer.src_vocab_size, 21)
        self.assertEqual(tokenizer.tgt_vocab_size, 21)

    def test_full_tokenizer(self):
        """Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt"""
        tokenizer = FSMTTokenizer(self.langs, self.src_vocab_file, self.tgt_vocab_file, self.merges_file)

        text = "lower"
        bpe_tokens = ["low", "er</w>"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + ["<unk>"]
        input_bpe_tokens = [14, 15, 20]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_ru_en

        text = tokenizer.encode("sequence builders", add_special_tokens=False)
        text_2 = tokenizer.encode("multi-sequence build", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == text + [2]
        assert encoded_pair == text + [2] + text_2 + [2]

    @slow
    def test_match_encode_decode(self):
        tokenizer_enc = self.tokenizer_en_ru
        tokenizer_dec = self.tokenizer_ru_en

        targets = [
            [
                "Here's a little song I wrote. Don't worry, be happy.",
                [2470, 39, 11, 2349, 7222, 70, 5979, 7, 8450, 1050, 13160, 5, 26, 6445, 7, 2],
            ],
            ["This is it. No more. I'm done!", [132, 21, 37, 7, 1434, 86, 7, 70, 6476, 1305, 427, 2]],
        ]

        # if data needs to be recreated or added, run:
        # import torch
        # model = torch.hub.load("pytorch/fairseq", "transformer.wmt19.en-ru", checkpoint_file="model4.pt", tokenizer="moses", bpe="fastbpe")
        # for src_text, _ in targets: print(f"""[\n"{src_text}",\n {model.encode(src_text).tolist()}\n],""")

        for src_text, tgt_input_ids in targets:
            encoded_ids = tokenizer_enc.encode(src_text, return_tensors=None)
            self.assertListEqual(encoded_ids, tgt_input_ids)

            # and decode backward, using the reversed languages model
            decoded_text = tokenizer_dec.decode(encoded_ids, skip_special_tokens=True)
            self.assertEqual(decoded_text, src_text)

    @slow
    def test_tokenizer_lower(self):
        tokenizer = FSMTTokenizer.from_pretrained("facebook/wmt19-ru-en", do_lower_case=True)
        tokens = tokenizer.tokenize("USA is United States of America")
        expected = ["us", "a</w>", "is</w>", "un", "i", "ted</w>", "st", "ates</w>", "of</w>", "am", "er", "ica</w>"]
        self.assertListEqual(tokens, expected)

    @unittest.skip("FSMTConfig.__init__  requires non-optional args")
    def test_torch_encode_plus_sent_to_model(self):
        pass

    @unittest.skip("FSMTConfig.__init__  requires non-optional args")
    def test_np_encode_plus_sent_to_model(self):
        pass
