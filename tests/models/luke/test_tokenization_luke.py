# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from transformers import LukeTokenizer
from transformers.testing_utils import get_tests_dir, require_torch, slow

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/vocab.json")
SAMPLE_MERGE_FILE = get_tests_dir("fixtures/merges.txt")
SAMPLE_ENTITY_VOCAB = get_tests_dir("fixtures/test_entity_vocab.json")


class LukeTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "studio-ousia/luke-base"
    tokenizer_class = LukeTokenizer
    from_pretrained_kwargs = {"cls_token": "<s>"}

    integration_expected_tokens = ['This', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ92', '000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁ', 'æ', '´', '»', 'çļĦ', 'çľ', 'Ł', 'è', '°', 'Ľ', 'æĺ¯', 'Ċ', 'Hi', 'Ġ', 'ĠHello', 'Ċ', 'Hi', 'Ġ', 'Ġ', 'ĠHello', 'ĊĊ', 'Ġ', 'Ċ', 'Ġ', 'Ġ', 'Ċ', 'ĠHello', 'Ċ', '<s>', 'Ċ', 'hi', '<s>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸', 'Ľ', 'à¸', 'µ', 'Ġ', 'Ġ', 'Ġ', 'ird', 'Ġ', 'Ġ', 'Ġ', 'à¸', 'Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_token_ids = [713, 16, 10, 1296, 17841, 27969, 50118, 100, 21, 2421, 11, 8403, 151, 6, 8, 42, 16, 22461, 1140, 4, 50118, 48998, 37127, 20024, 2023, 44574, 49122, 4333, 36484, 7487, 3726, 48569, 50118, 30086, 1437, 20920, 50118, 30086, 1437, 1437, 20920, 50140, 1437, 50118, 1437, 1437, 50118, 20920, 50118, 0, 50118, 3592, 0, 8585, 50118, 133, 511, 6755, 197, 28, 5083, 45320, 35, 20920, 4, 50118, 1708, 1437, 8602, 8, 1437, 24107, 3726, 24107, 8906, 1437, 1437, 1437, 8602, 1437, 1437, 1437, 24107, 10674, 50118, 13368, 141, 32, 47, 608]  # fmt: skip
    expected_tokens_from_ids = ['This', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ92', '000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁ', 'æ', '´', '»', 'çļĦ', 'çľ', 'Ł', 'è', '°', 'Ľ', 'æĺ¯', 'Ċ', 'Hi', 'Ġ', 'ĠHello', 'Ċ', 'Hi', 'Ġ', 'Ġ', 'ĠHello', 'ĊĊ', 'Ġ', 'Ċ', 'Ġ', 'Ġ', 'Ċ', 'ĠHello', 'Ċ', '<s>', 'Ċ', 'hi', '<s>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸', 'Ľ', 'à¸', 'µ', 'Ġ', 'Ġ', 'Ġ', 'ird', 'Ġ', 'Ġ', 'Ġ', 'à¸', 'Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing"

    @slow
    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("studio-ousia/luke-large")

        text = tokenizer.encode("sequence builders", add_special_tokens=False)
        text_2 = tokenizer.encode("multi-sequence build", add_special_tokens=False)

        encoded_text_from_decode = tokenizer.encode(
            "sequence builders", add_special_tokens=True, add_prefix_space=False
        )
        encoded_pair_from_decode = tokenizer.encode(
            "sequence builders", "multi-sequence build", add_special_tokens=True, add_prefix_space=False
        )

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        self.assertEqual(encoded_sentence, encoded_text_from_decode)
        self.assertEqual(encoded_pair, encoded_pair_from_decode)

    def get_clean_sequence(self, tokenizer, max_length=20) -> tuple[str, list]:
        txt = "Beyonce lives in Los Angeles"
        ids = tokenizer.encode(txt, add_special_tokens=False)
        return txt, ids

    def test_padding_entity_inputs(self):
        tokenizer = self.get_tokenizer()

        sentence = "Japanese is an East Asian language spoken by about 128 million people, primarily in Japan."
        span = (15, 34)
        pad_id = tokenizer.entity_vocab["[PAD]"]
        mask_id = tokenizer.entity_vocab["[MASK]"]

        encoding = tokenizer([sentence, sentence], entity_spans=[[span], [span, span]], padding=True)
        self.assertEqual(encoding["entity_ids"], [[mask_id, pad_id], [mask_id, mask_id]])

        # test with a sentence with no entity
        encoding = tokenizer([sentence, sentence], entity_spans=[[], [span, span]], padding=True)
        self.assertEqual(encoding["entity_ids"], [[pad_id, pad_id], [mask_id, mask_id]])


@slow
@require_torch
class LukeTokenizerIntegrationTests(unittest.TestCase):
    tokenizer_class = LukeTokenizer
    from_pretrained_kwargs = {"cls_token": "<s>"}

    def setUp(self):
        super().setUp()

    def test_single_text_no_padding_or_truncation(self):
        tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base", return_token_type_ids=True)
        sentence = "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck."
        entities = ["Ana Ivanovic", "Thursday", "Dummy Entity"]
        spans = [(9, 21), (30, 38), (39, 42)]

        encoding = tokenizer(sentence, entities=entities, entity_spans=spans, return_token_type_ids=True)

        self.assertEqual(
            tokenizer.decode(encoding["input_ids"], spaces_between_special_tokens=False),
            "<s>Top seed Ana Ivanovic said on Thursday she could hardly believe her luck.</s>",
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][3:6], spaces_between_special_tokens=False), " Ana Ivanovic"
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][8:9], spaces_between_special_tokens=False), " Thursday"
        )
        self.assertEqual(tokenizer.decode(encoding["input_ids"][9:10], spaces_between_special_tokens=False), " she")

        self.assertEqual(
            encoding["entity_ids"],
            [
                tokenizer.entity_vocab["Ana Ivanovic"],
                tokenizer.entity_vocab["Thursday"],
                tokenizer.entity_vocab["[UNK]"],
            ],
        )
        self.assertEqual(encoding["entity_attention_mask"], [1, 1, 1])
        self.assertEqual(encoding["entity_token_type_ids"], [0, 0, 0])
        # fmt: off
        self.assertEqual(
            encoding["entity_position_ids"],
            [
                [3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ]
        )
        # fmt: on

    def test_single_text_only_entity_spans_no_padding_or_truncation(self):
        tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base", return_token_type_ids=True)
        sentence = "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck."
        spans = [(9, 21), (30, 38), (39, 42)]

        encoding = tokenizer(sentence, entity_spans=spans, return_token_type_ids=True)

        self.assertEqual(
            tokenizer.decode(encoding["input_ids"], spaces_between_special_tokens=False),
            "<s>Top seed Ana Ivanovic said on Thursday she could hardly believe her luck.</s>",
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][3:6], spaces_between_special_tokens=False), " Ana Ivanovic"
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][8:9], spaces_between_special_tokens=False), " Thursday"
        )
        self.assertEqual(tokenizer.decode(encoding["input_ids"][9:10], spaces_between_special_tokens=False), " she")

        mask_id = tokenizer.entity_vocab["[MASK]"]
        self.assertEqual(encoding["entity_ids"], [mask_id, mask_id, mask_id])
        self.assertEqual(encoding["entity_attention_mask"], [1, 1, 1])
        self.assertEqual(encoding["entity_token_type_ids"], [0, 0, 0])
        # fmt: off
        self.assertEqual(
            encoding["entity_position_ids"],
            [
                [3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, ],
                [9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, ]
            ]
        )
        # fmt: on

    def test_single_text_padding_pytorch_tensors(self):
        tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base", return_token_type_ids=True)
        sentence = "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck."
        entities = ["Ana Ivanovic", "Thursday", "Dummy Entity"]
        spans = [(9, 21), (30, 38), (39, 42)]

        encoding = tokenizer(
            sentence,
            entities=entities,
            entity_spans=spans,
            return_token_type_ids=True,
            padding="max_length",
            max_length=30,
            max_entity_length=16,
            return_tensors="pt",
        )

        # test words
        self.assertEqual(encoding["input_ids"].shape, (1, 30))
        self.assertEqual(encoding["attention_mask"].shape, (1, 30))
        self.assertEqual(encoding["token_type_ids"].shape, (1, 30))

        # test entities
        self.assertEqual(encoding["entity_ids"].shape, (1, 16))
        self.assertEqual(encoding["entity_attention_mask"].shape, (1, 16))
        self.assertEqual(encoding["entity_token_type_ids"].shape, (1, 16))
        self.assertEqual(encoding["entity_position_ids"].shape, (1, 16, tokenizer.max_mention_length))

    def test_text_pair_no_padding_or_truncation(self):
        tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base", return_token_type_ids=True)
        sentence = "Top seed Ana Ivanovic said on Thursday"
        sentence_pair = "She could hardly believe her luck."
        entities = ["Ana Ivanovic", "Thursday"]
        entities_pair = ["Dummy Entity"]
        spans = [(9, 21), (30, 38)]
        spans_pair = [(0, 3)]

        encoding = tokenizer(
            sentence,
            sentence_pair,
            entities=entities,
            entities_pair=entities_pair,
            entity_spans=spans,
            entity_spans_pair=spans_pair,
            return_token_type_ids=True,
        )

        self.assertEqual(
            tokenizer.decode(encoding["input_ids"], spaces_between_special_tokens=False),
            "<s>Top seed Ana Ivanovic said on Thursday</s></s>She could hardly believe her luck.</s>",
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][3:6], spaces_between_special_tokens=False), " Ana Ivanovic"
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][8:9], spaces_between_special_tokens=False), " Thursday"
        )
        self.assertEqual(tokenizer.decode(encoding["input_ids"][11:12], spaces_between_special_tokens=False), "She")

        self.assertEqual(
            encoding["entity_ids"],
            [
                tokenizer.entity_vocab["Ana Ivanovic"],
                tokenizer.entity_vocab["Thursday"],
                tokenizer.entity_vocab["[UNK]"],
            ],
        )
        self.assertEqual(encoding["entity_attention_mask"], [1, 1, 1])
        self.assertEqual(encoding["entity_token_type_ids"], [0, 0, 0])
        # fmt: off
        self.assertEqual(
            encoding["entity_position_ids"],
            [
                [3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ]
        )
        # fmt: on

    def test_text_pair_only_entity_spans_no_padding_or_truncation(self):
        tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base", return_token_type_ids=True)
        sentence = "Top seed Ana Ivanovic said on Thursday"
        sentence_pair = "She could hardly believe her luck."
        spans = [(9, 21), (30, 38)]
        spans_pair = [(0, 3)]

        encoding = tokenizer(
            sentence,
            sentence_pair,
            entity_spans=spans,
            entity_spans_pair=spans_pair,
            return_token_type_ids=True,
        )

        self.assertEqual(
            tokenizer.decode(encoding["input_ids"], spaces_between_special_tokens=False),
            "<s>Top seed Ana Ivanovic said on Thursday</s></s>She could hardly believe her luck.</s>",
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][3:6], spaces_between_special_tokens=False), " Ana Ivanovic"
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][8:9], spaces_between_special_tokens=False), " Thursday"
        )
        self.assertEqual(tokenizer.decode(encoding["input_ids"][11:12], spaces_between_special_tokens=False), "She")

        mask_id = tokenizer.entity_vocab["[MASK]"]
        self.assertEqual(encoding["entity_ids"], [mask_id, mask_id, mask_id])
        self.assertEqual(encoding["entity_attention_mask"], [1, 1, 1])
        self.assertEqual(encoding["entity_token_type_ids"], [0, 0, 0])
        # fmt: off
        self.assertEqual(
            encoding["entity_position_ids"],
            [
                [3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ]
        )
        # fmt: on

    def test_text_pair_padding_pytorch_tensors(self):
        tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base", return_token_type_ids=True)
        sentence = "Top seed Ana Ivanovic said on Thursday"
        sentence_pair = "She could hardly believe her luck."
        entities = ["Ana Ivanovic", "Thursday"]
        entities_pair = ["Dummy Entity"]
        spans = [(9, 21), (30, 38)]
        spans_pair = [(0, 3)]

        encoding = tokenizer(
            sentence,
            sentence_pair,
            entities=entities,
            entities_pair=entities_pair,
            entity_spans=spans,
            entity_spans_pair=spans_pair,
            return_token_type_ids=True,
            padding="max_length",
            max_length=30,
            max_entity_length=16,
            return_tensors="pt",
        )

        # test words
        self.assertEqual(encoding["input_ids"].shape, (1, 30))
        self.assertEqual(encoding["attention_mask"].shape, (1, 30))
        self.assertEqual(encoding["token_type_ids"].shape, (1, 30))

        # test entities
        self.assertEqual(encoding["entity_ids"].shape, (1, 16))
        self.assertEqual(encoding["entity_attention_mask"].shape, (1, 16))
        self.assertEqual(encoding["entity_token_type_ids"].shape, (1, 16))
        self.assertEqual(encoding["entity_position_ids"].shape, (1, 16, tokenizer.max_mention_length))

    def test_entity_classification_no_padding_or_truncation(self):
        tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base", task="entity_classification")
        sentence = (
            "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck as a fortuitous netcord helped"
            " the new world number one avoid a humiliating second- round exit at Wimbledon ."
        )
        span = (39, 42)

        encoding = tokenizer(sentence, entity_spans=[span], return_token_type_ids=True)

        # test words
        self.assertEqual(len(encoding["input_ids"]), 42)
        self.assertEqual(len(encoding["attention_mask"]), 42)
        self.assertEqual(len(encoding["token_type_ids"]), 42)
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"], spaces_between_special_tokens=False),
            "<s>Top seed Ana Ivanovic said on Thursday<ent> she<ent> could hardly believe her luck as a fortuitous"
            " netcord helped the new world number one avoid a humiliating second- round exit at Wimbledon.</s>",
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][9:12], spaces_between_special_tokens=False), "<ent> she<ent>"
        )

        # test entities
        self.assertEqual(encoding["entity_ids"], [2])
        self.assertEqual(encoding["entity_attention_mask"], [1])
        self.assertEqual(encoding["entity_token_type_ids"], [0])
        # fmt: off
        self.assertEqual(
            encoding["entity_position_ids"],
            [
                [9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            ]
        )
        # fmt: on

    def test_entity_classification_padding_pytorch_tensors(self):
        tokenizer = LukeTokenizer.from_pretrained(
            "studio-ousia/luke-base", task="entity_classification", return_token_type_ids=True
        )
        sentence = (
            "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck as a fortuitous netcord helped"
            " the new world number one avoid a humiliating second- round exit at Wimbledon ."
        )
        # entity information
        span = (39, 42)

        encoding = tokenizer(
            sentence, entity_spans=[span], return_token_type_ids=True, padding="max_length", return_tensors="pt"
        )

        # test words
        self.assertEqual(encoding["input_ids"].shape, (1, 512))
        self.assertEqual(encoding["attention_mask"].shape, (1, 512))
        self.assertEqual(encoding["token_type_ids"].shape, (1, 512))

        # test entities
        self.assertEqual(encoding["entity_ids"].shape, (1, 1))
        self.assertEqual(encoding["entity_attention_mask"].shape, (1, 1))
        self.assertEqual(encoding["entity_token_type_ids"].shape, (1, 1))
        self.assertEqual(
            encoding["entity_position_ids"].shape, (1, tokenizer.max_entity_length, tokenizer.max_mention_length)
        )

    def test_entity_pair_classification_no_padding_or_truncation(self):
        tokenizer = LukeTokenizer.from_pretrained(
            "studio-ousia/luke-base", task="entity_pair_classification", return_token_type_ids=True
        )
        sentence = "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck."
        # head and tail information
        spans = [(9, 21), (39, 42)]

        encoding = tokenizer(sentence, entity_spans=spans, return_token_type_ids=True)

        self.assertEqual(
            tokenizer.decode(encoding["input_ids"], spaces_between_special_tokens=False),
            "<s>Top seed<ent> Ana Ivanovic<ent> said on Thursday<ent2> she<ent2> could hardly believe her luck.</s>",
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][3:8], spaces_between_special_tokens=False),
            "<ent> Ana Ivanovic<ent>",
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][11:14], spaces_between_special_tokens=False), "<ent2> she<ent2>"
        )

        self.assertEqual(encoding["entity_ids"], [2, 3])
        self.assertEqual(encoding["entity_attention_mask"], [1, 1])
        self.assertEqual(encoding["entity_token_type_ids"], [0, 0])
        # fmt: off
        self.assertEqual(
            encoding["entity_position_ids"],
            [
                [3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [11, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ]
        )
        # fmt: on

    def test_entity_pair_classification_padding_pytorch_tensors(self):
        tokenizer = LukeTokenizer.from_pretrained(
            "studio-ousia/luke-base", task="entity_pair_classification", return_token_type_ids=True
        )
        sentence = "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck."
        # head and tail information
        spans = [(9, 21), (39, 42)]

        encoding = tokenizer(
            sentence,
            entity_spans=spans,
            return_token_type_ids=True,
            padding="max_length",
            max_length=30,
            return_tensors="pt",
        )

        # test words
        self.assertEqual(encoding["input_ids"].shape, (1, 30))
        self.assertEqual(encoding["attention_mask"].shape, (1, 30))
        self.assertEqual(encoding["token_type_ids"].shape, (1, 30))

        # test entities
        self.assertEqual(encoding["entity_ids"].shape, (1, 2))
        self.assertEqual(encoding["entity_attention_mask"].shape, (1, 2))
        self.assertEqual(encoding["entity_token_type_ids"].shape, (1, 2))
        self.assertEqual(
            encoding["entity_position_ids"].shape, (1, tokenizer.max_entity_length, tokenizer.max_mention_length)
        )

    def test_entity_span_classification_no_padding_or_truncation(self):
        tokenizer = LukeTokenizer.from_pretrained(
            "studio-ousia/luke-base", task="entity_span_classification", return_token_type_ids=True
        )
        sentence = "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck."
        spans = [(0, 8), (9, 21), (39, 42)]

        encoding = tokenizer(sentence, entity_spans=spans, return_token_type_ids=True)

        self.assertEqual(
            tokenizer.decode(encoding["input_ids"], spaces_between_special_tokens=False),
            "<s>Top seed Ana Ivanovic said on Thursday she could hardly believe her luck.</s>",
        )

        self.assertEqual(encoding["entity_ids"], [2, 2, 2])
        self.assertEqual(encoding["entity_attention_mask"], [1, 1, 1])
        self.assertEqual(encoding["entity_token_type_ids"], [0, 0, 0])
        # fmt: off
        self.assertEqual(
            encoding["entity_position_ids"],
            [
                [1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ]
        )
        # fmt: on
        self.assertEqual(encoding["entity_start_positions"], [1, 3, 9])
        self.assertEqual(encoding["entity_end_positions"], [2, 5, 9])

    def test_entity_span_classification_padding_pytorch_tensors(self):
        tokenizer = LukeTokenizer.from_pretrained(
            "studio-ousia/luke-base", task="entity_span_classification", return_token_type_ids=True
        )
        sentence = "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck."
        spans = [(0, 8), (9, 21), (39, 42)]

        encoding = tokenizer(
            sentence,
            entity_spans=spans,
            return_token_type_ids=True,
            padding="max_length",
            max_length=30,
            max_entity_length=16,
            return_tensors="pt",
        )

        # test words
        self.assertEqual(encoding["input_ids"].shape, (1, 30))
        self.assertEqual(encoding["attention_mask"].shape, (1, 30))
        self.assertEqual(encoding["token_type_ids"].shape, (1, 30))

        # test entities
        self.assertEqual(encoding["entity_ids"].shape, (1, 16))
        self.assertEqual(encoding["entity_attention_mask"].shape, (1, 16))
        self.assertEqual(encoding["entity_token_type_ids"].shape, (1, 16))
        self.assertEqual(encoding["entity_position_ids"].shape, (1, 16, tokenizer.max_mention_length))
        self.assertEqual(encoding["entity_start_positions"].shape, (1, 16))
        self.assertEqual(encoding["entity_end_positions"].shape, (1, 16))
