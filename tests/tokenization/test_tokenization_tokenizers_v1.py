# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import copy
import json
import os
import pickle
import tempfile
import unittest

import numpy as np

from transformers import AutoTokenizer
from transformers.testing_utils import require_tokenizers_v1
from transformers.utils import is_tokenizers_available, is_tokenizers_v1_available


VOCAB = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "hello": 4, "world": 5, "how": 6, "are": 7, "you": 8}


def _added_token(content, id, special=True):
    return {
        "id": id,
        "content": content,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": not special,
        "special": special,
    }


def _template(*pieces):
    return [
        {"SpecialToken": {"id": p, "type_id": t}} if p.startswith("[") else {"Sequence": {"id": p, "type_id": t}}
        for p, t in pieces
    ]


TOKENIZER_JSON = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [_added_token(token, id) for token, id in list(VOCAB.items())[:4]],
    "normalizer": {"type": "Lowercase"},
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": {
        "type": "TemplateProcessing",
        "single": _template(("[CLS]", 0), ("A", 0), ("[SEP]", 0)),
        "pair": _template(("[CLS]", 0), ("A", 0), ("[SEP]", 0), ("B", 1), ("[SEP]", 1)),
        "special_tokens": {t: {"id": t, "ids": [VOCAB[t]], "tokens": [t]} for t in ("[CLS]", "[SEP]")},
    },
    "decoder": None,
    "model": {"type": "WordLevel", "vocab": VOCAB, "unk_token": "[UNK]"},
}
TOKENIZER_CONFIG = {
    "unk_token": "[UNK]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "model_input_names": ["input_ids", "token_type_ids", "attention_mask"],
}


@require_tokenizers_v1
class TokenizersV1BackendTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        with open(os.path.join(self.tmpdir.name, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(TOKENIZER_JSON, f)
        with open(os.path.join(self.tmpdir.name, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(TOKENIZER_CONFIG, f)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_v0_backend_is_disabled(self):
        self.assertTrue(is_tokenizers_v1_available())
        self.assertFalse(is_tokenizers_available())
        self.assertEqual(type(self.tokenizer).__name__, "TokenizersV1Backend")

    def test_encode_single(self):
        encoding = self.tokenizer("Hello world")

        self.assertEqual(encoding["input_ids"], [1, 4, 5, 2])
        self.assertEqual(encoding["token_type_ids"], [0, 0, 0, 0])
        self.assertEqual(encoding["attention_mask"], [1, 1, 1, 1])

    def test_encode_without_special_tokens(self):
        self.assertEqual(self.tokenizer("hello world", add_special_tokens=False)["input_ids"], [4, 5])
        self.assertEqual(self.tokenizer.encode("hello world"), [1, 4, 5, 2])

    def test_encode_pair(self):
        encoding = self.tokenizer("hello", "how are you")

        self.assertEqual(encoding["input_ids"], [1, 4, 2, 6, 7, 8, 2])
        self.assertEqual(encoding["token_type_ids"], [0, 0, 0, 1, 1, 1, 1])

    def test_encode_batch_of_pairs(self):
        encoding = self.tokenizer(["hello", "world"], ["you", "how are"])

        self.assertEqual(encoding["input_ids"], [[1, 4, 2, 8, 2], [1, 5, 2, 6, 7, 2]])

    def test_padding_longest(self):
        encoding = self.tokenizer(["hello", "hello world how"], padding=True)

        self.assertEqual(encoding["input_ids"], [[1, 4, 2, 3, 3], [1, 4, 5, 6, 2]])
        self.assertEqual(encoding["attention_mask"], [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])

    def test_padding_left_to_max_length(self):
        encoding = self.tokenizer(["hello"], padding="max_length", max_length=6, padding_side="left")

        self.assertEqual(encoding["input_ids"], [[3, 3, 3, 1, 4, 2]])

    def test_pad_to_multiple_of(self):
        encoding = self.tokenizer("hello world how", padding=True, pad_to_multiple_of=4)

        self.assertEqual(encoding["input_ids"], [1, 4, 5, 6, 2, 3, 3, 3])

    def test_truncation(self):
        encoding = self.tokenizer("hello world how are you", truncation=True, max_length=4)

        self.assertEqual(encoding["input_ids"], [1, 4, 5, 2])

    def test_truncation_left(self):
        self.tokenizer.truncation_side = "left"

        encoding = self.tokenizer("hello world how are you", truncation=True, max_length=4)

        self.assertEqual(encoding["input_ids"], [1, 7, 8, 2])

    def test_truncation_only_second(self):
        encoding = self.tokenizer("hello world", "how are you", truncation="only_second", max_length=7)

        self.assertEqual(encoding["input_ids"], [1, 4, 5, 2, 6, 7, 2])

    def test_return_tensors(self):
        encoding = self.tokenizer(["hello", "hello world"], padding=True, return_tensors="np")

        np.testing.assert_array_equal(encoding["input_ids"], np.array([[1, 4, 2, 3], [1, 4, 5, 2]]))

    def test_return_length(self):
        self.assertEqual(self.tokenizer(["hello", "hello world"], return_length=True)["length"], [3, 4])

    def test_decode(self):
        self.assertEqual(self.tokenizer.decode([1, 4, 5, 2]), "[CLS] hello world [SEP]")
        self.assertEqual(self.tokenizer.decode([1, 4, 5, 2], skip_special_tokens=True), "hello world")
        self.assertEqual(self.tokenizer.batch_decode([[4], [5, 2]], skip_special_tokens=True), ["hello", "world"])

    def test_tokenize(self):
        self.assertEqual(self.tokenizer.tokenize("Hello world"), ["hello", "world"])
        self.assertEqual(self.tokenizer.tokenize("hello", "you"), ["hello", "you"])

    def test_vocab(self):
        self.assertEqual(self.tokenizer.get_vocab(), VOCAB)
        self.assertEqual(self.tokenizer.vocab_size, 9)
        self.assertEqual(len(self.tokenizer), 9)

    def test_convert_tokens_and_ids(self):
        self.assertEqual(self.tokenizer.convert_tokens_to_ids(["hello", "nope"]), [4, 0])
        self.assertEqual(self.tokenizer.convert_tokens_to_ids("world"), 5)
        self.assertEqual(self.tokenizer.convert_ids_to_tokens([1, 4, 2]), ["[CLS]", "hello", "[SEP]"])
        self.assertEqual(self.tokenizer.convert_ids_to_tokens([1, 4, 2], skip_special_tokens=True), ["hello"])
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(5), "world")
        self.assertEqual(self.tokenizer.convert_tokens_to_string(["hello", "world"]), "hello world")

    def test_special_tokens(self):
        self.assertEqual(self.tokenizer.pad_token_id, 3)
        self.assertEqual(self.tokenizer.cls_token_id, 1)
        self.assertEqual(sorted(self.tokenizer.all_special_ids), [0, 1, 2, 3])
        self.assertEqual(self.tokenizer.num_special_tokens_to_add(), 2)
        self.assertEqual(self.tokenizer.num_special_tokens_to_add(pair=True), 3)
        self.assertEqual(sorted(self.tokenizer.added_tokens_decoder), [0, 1, 2, 3])
        self.assertTrue(self.tokenizer.added_tokens_decoder[1].special)

    def test_add_tokens(self):
        added = self.tokenizer.add_tokens(["<new>", "hello"])

        self.assertEqual(added, 2)  # as in 0.23: "hello" is in the model vocab, but now also an added token
        self.assertEqual(len(self.tokenizer), 10)
        self.assertEqual(self.tokenizer.convert_tokens_to_ids("<new>"), 9)
        self.assertEqual(self.tokenizer("hello <new> world")["input_ids"], [1, 4, 9, 5, 2])
        self.assertEqual(self.tokenizer.get_added_vocab()["<new>"], 9)

    def test_add_special_tokens(self):
        self.tokenizer.add_special_tokens({"extra_special_tokens": ["<tool>"]})

        self.assertEqual(self.tokenizer("hello <tool>")["input_ids"], [1, 4, 9, 2])
        self.assertEqual(self.tokenizer.decode([4, 9], skip_special_tokens=True), "hello")

    def test_special_token_missing_from_vocab_is_added(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tmpdir.name, mask_token="[MASK]")

        self.assertEqual(tokenizer.mask_token_id, 9)
        self.assertEqual(tokenizer("hello [MASK]")["input_ids"], [1, 4, 9, 2])

    def test_save_and_reload(self):
        self.tokenizer.add_tokens(["<new>"])
        self.tokenizer.add_special_tokens({"extra_special_tokens": ["<tool>"]})

        with tempfile.TemporaryDirectory() as save_dir:
            self.tokenizer.save_pretrained(save_dir)
            reloaded = AutoTokenizer.from_pretrained(save_dir)

        text = "hello <new> world <tool>"
        self.assertEqual(reloaded(text)["input_ids"], self.tokenizer(text)["input_ids"])
        self.assertEqual(len(reloaded), len(self.tokenizer))
        self.assertEqual(reloaded.extra_special_tokens, ["<tool>"])

    def test_chat_template(self):
        self.tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"

        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}, {"role": "user", "content": "you"}]
        )

        self.assertEqual(ids["input_ids"], [4, 8])

    def test_pickle_and_deepcopy(self):
        for clone in (pickle.loads(pickle.dumps(self.tokenizer)), copy.deepcopy(self.tokenizer)):
            self.assertEqual(clone("hello world")["input_ids"], [1, 4, 5, 2])

    def test_unsupported_features_raise(self):
        for kwargs in (
            {"return_offsets_mapping": True},
            {"return_overflowing_tokens": True, "truncation": True, "max_length": 3},
            {"stride": 1, "truncation": True, "max_length": 3},
            {"return_special_tokens_mask": True},
        ):
            with self.subTest(**kwargs), self.assertRaises(NotImplementedError):
                self.tokenizer("hello world", **kwargs)

        with self.assertRaises(NotImplementedError):
            self.tokenizer([["hello", "world"]], is_split_into_words=True)
