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
"""
`TokenizersBackend` on tokenizers 0.x and 1.x. The `TokenizersBackendTest` cases run on both: on 0.x they pin the
expected values, on 1.x they check that `V0TokenizerFacade` reproduces them.
"""

import copy
import json
import os
import pickle
import tempfile
import unittest

import numpy as np

from transformers import AutoTokenizer, TokenizersBackend
from transformers.testing_utils import require_tokenizers, require_tokenizers_v1, require_torchvision


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


class _TokenizerJsonTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        with open(os.path.join(self.tmpdir.name, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(TOKENIZER_JSON, f)
        with open(os.path.join(self.tmpdir.name, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(TOKENIZER_CONFIG, f)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()


@require_tokenizers
class TokenizersBackendTest(_TokenizerJsonTestCase):
    def test_loads_tokenizers_backend(self):
        self.assertIsInstance(self.tokenizer, TokenizersBackend)

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

    def test_special_tokens_mask_marks_template_and_padding(self):
        encoding = self.tokenizer(["hello", "hello world"], padding=True, return_special_tokens_mask=True)

        self.assertEqual(encoding["special_tokens_mask"], [[1, 0, 1, 1], [1, 0, 0, 1]])

    def test_sequence_ids(self):
        self.assertEqual(self.tokenizer("hello", "you how").sequence_ids(), [None, 0, None, 1, 1, None])

    def test_overflowing_tokens_with_stride(self):
        encoding = self.tokenizer(
            "hello world how are you", truncation=True, max_length=5, stride=1, return_overflowing_tokens=True
        )

        self.assertEqual(encoding["input_ids"], [[1, 4, 5, 6, 2], [1, 6, 7, 8, 2]])
        self.assertEqual(encoding["overflow_to_sample_mapping"], [0, 0])

    def test_overflowing_tokens_left(self):
        self.tokenizer.truncation_side = "left"

        encoding = self.tokenizer(
            "hello world how are you", truncation=True, max_length=4, stride=1, return_overflowing_tokens=True
        )

        self.assertEqual(encoding["input_ids"], [[1, 7, 8, 2], [1, 6, 7, 2], [1, 5, 6, 2], [1, 4, 5, 2]])

    def test_overflowing_tokens_of_pair_second_sequence(self):
        encoding = self.tokenizer(
            ["hello", "you"],
            ["how are you world", "hello"],
            truncation="only_second",
            max_length=6,
            padding=True,
            return_overflowing_tokens=True,
        )

        self.assertEqual(encoding["input_ids"], [[1, 4, 2, 6, 7, 2], [1, 4, 2, 8, 5, 2], [1, 8, 2, 4, 2, 3]])
        self.assertEqual(encoding["overflow_to_sample_mapping"], [0, 0, 1])

    def test_split_into_words_keeps_two_words_a_single_sequence(self):
        encoding = self.tokenizer(["Hello", "world"], is_split_into_words=True)

        self.assertEqual(encoding["input_ids"], [1, 4, 5, 2])
        self.assertEqual(encoding["token_type_ids"], [0, 0, 0, 0])
        self.assertEqual(encoding.word_ids(), [None, 0, 1, None])

    def test_split_into_words_pairs_batch(self):
        encoding = self.tokenizer(
            [["hello"], ["how", "are", "you"]], [["world"], ["hello"]], is_split_into_words=True, padding=True
        )

        self.assertEqual(encoding["input_ids"], [[1, 4, 2, 5, 2, 3, 3], [1, 6, 7, 8, 2, 4, 2]])
        self.assertEqual(encoding.word_ids(1), [None, 0, 1, 2, None, 0, None])
        self.assertEqual(encoding.word_to_tokens(1, 2, sequence_index=0), (3, 4))

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
        self.assertEqual(self.tokenizer.vocab, VOCAB)
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

    def test_chat_template_sanitizes_control_tokens(self):
        self.tokenizer.chat_template = "[SEP]{% for m in messages %}{{ m['content'] }} {% endfor %}"

        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello [SEP]"}], sanitize_control_tokens=True
        )

        self.assertEqual(ids["input_ids"].count(self.tokenizer.sep_token_id), 1)

    def test_pickle_and_deepcopy(self):
        for clone in (pickle.loads(pickle.dumps(self.tokenizer)), copy.deepcopy(self.tokenizer)):
            self.assertEqual(clone("hello world")["input_ids"], [1, 4, 5, 2])

    @require_torchvision
    def test_processors_with_hardcoded_tokenizer_class_load_tokenizers_backend(self):
        from transformers import FuyuImageProcessorPil, FuyuProcessor, PixtralImageProcessor, PixtralProcessor

        for processor_class, image_processor in (
            (PixtralProcessor, PixtralImageProcessor()),
            (FuyuProcessor, FuyuImageProcessorPil()),
        ):
            with self.subTest(processor_class.__name__):
                self.tokenizer.add_tokens(["|SPEAKER|", "|NEWLINE|"], special_tokens=True)  # looked up by Fuyu
                self.tokenizer.save_pretrained(self.tmpdir.name)
                image_processor.save_pretrained(self.tmpdir.name)
                processor = processor_class.from_pretrained(self.tmpdir.name)
                self.assertIsInstance(processor.tokenizer, TokenizersBackend)


@require_tokenizers_v1
class TokenizersV1GapsTest(_TokenizerJsonTestCase):
    def test_offsets_raise(self):
        with self.assertRaises(NotImplementedError):
            self.tokenizer("hello world", return_offsets_mapping=True)
        with self.assertRaises(NotImplementedError):
            self.tokenizer("hello world").char_to_token(0)

    def test_word_ids_of_raw_text_raise(self):
        with self.assertRaises(NotImplementedError):
            self.tokenizer("hello world").word_ids()

    def test_training_raises(self):
        with self.assertRaises(NotImplementedError):
            self.tokenizer.train_new_from_iterator([["hello world"]], vocab_size=10)

    def test_is_not_fast_without_offsets(self):
        # `is_fast` gates the offset-based paths (token classification pipeline, whole word masking)
        self.assertFalse(self.tokenizer.is_fast)
