# coding=utf-8
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

from transformers import AddedToken, LukeTokenizer
from transformers.testing_utils import require_torch, slow

from .test_tokenization_common import TokenizerTesterMixin


class Luke(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = LukeTokenizer
    from_pretrained_kwargs = {"cls_token": "<s>"}

    def setUp(self):
        super().setUp()

        self.special_tokens_map = {"entity_token_1": "<ent>", "entity_token_2": "<ent2>"}

    def get_tokenizer(self, task=None, **kwargs):
        kwargs.update(self.special_tokens_map)
        return self.tokenizer_class.from_pretrained("studio-ousia/luke-base", task=task, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.tokenizer_class.from_pretrained("studio-ousia/luke-base")
        text = "lower newer"
        bpe_tokens = ["lower", "\u0120newer"]
        tokens = tokenizer.tokenize(text)  # , add_prefix_space=True)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [29668, 13964, 3]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def luke_dict_integration_testing(self):
        tokenizer = self.get_tokenizer()

        self.assertListEqual(tokenizer.encode("Hello world!", add_special_tokens=False), [0, 31414, 232, 328, 2])
        self.assertListEqual(
            tokenizer.encode("Hello world! cécé herlolip 418", add_special_tokens=False),
            [0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2],
        )

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

        assert encoded_sentence == encoded_text_from_decode
        assert encoded_pair == encoded_pair_from_decode

    def test_space_encoding(self):
        tokenizer = self.get_tokenizer()

        sequence = "Encode this sequence."
        space_encoding = tokenizer.byte_encoder[" ".encode("utf-8")[0]]

        # Testing encoder arguments
        encoded = tokenizer.encode(sequence, add_special_tokens=False, add_prefix_space=False)
        first_char = tokenizer.convert_ids_to_tokens(encoded[0])[0]
        self.assertNotEqual(first_char, space_encoding)

        encoded = tokenizer.encode(sequence, add_special_tokens=False, add_prefix_space=True)
        first_char = tokenizer.convert_ids_to_tokens(encoded[0])[0]
        self.assertEqual(first_char, space_encoding)

        tokenizer.add_special_tokens({"bos_token": "<s>"})
        encoded = tokenizer.encode(sequence, add_special_tokens=True)
        first_char = tokenizer.convert_ids_to_tokens(encoded[1])[0]
        self.assertNotEqual(first_char, space_encoding)

        # Testing spaces after special tokens
        mask = "<mask>"
        tokenizer.add_special_tokens(
            {"mask_token": AddedToken(mask, lstrip=True, rstrip=False)}
        )  # mask token has a left space
        mask_ind = tokenizer.convert_tokens_to_ids(mask)

        sequence = "Encode <mask> sequence"
        sequence_nospace = "Encode <mask>sequence"

        encoded = tokenizer.encode(sequence)
        mask_loc = encoded.index(mask_ind)
        first_char = tokenizer.convert_ids_to_tokens(encoded[mask_loc + 1])[0]
        self.assertEqual(first_char, space_encoding)

        encoded = tokenizer.encode(sequence_nospace)
        mask_loc = encoded.index(mask_ind)
        first_char = tokenizer.convert_ids_to_tokens(encoded[mask_loc + 1])[0]
        self.assertNotEqual(first_char, space_encoding)

    def test_pretokenized_inputs(self):
        pass

    def test_embeded_special_tokens(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest("{} ({})".format(tokenizer.__class__.__name__, pretrained_name)):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                sentence = "A, <mask> AllenNLP sentence."
                tokens_r = tokenizer_r.encode_plus(sentence, add_special_tokens=True, return_token_type_ids=True)
                tokens_p = tokenizer_p.encode_plus(sentence, add_special_tokens=True, return_token_type_ids=True)

                # token_type_ids should put 0 everywhere
                self.assertEqual(sum(tokens_r["token_type_ids"]), sum(tokens_p["token_type_ids"]))

                # token_type_ids should put 0 everywhere
                self.assertEqual(sum(tokens_r["token_type_ids"]), sum(tokens_p["token_type_ids"]))

                # attention_mask should put 1 everywhere, so sum over length should be 1
                self.assertEqual(
                    sum(tokens_p["attention_mask"]) / len(tokens_p["attention_mask"]),
                )

                tokens_p_str = tokenizer_p.convert_ids_to_tokens(tokens_p["input_ids"])

                # Rust correctly handles the space before the mask while python doesnt
                self.assertSequenceEqual(tokens_p["input_ids"], [0, 250, 6, 50264, 3823, 487, 21992, 3645, 4, 2])

                self.assertSequenceEqual(
                    tokens_p_str, ["<s>", "A", ",", "<mask>", "ĠAllen", "N", "LP", "Ġsentence", ".", "</s>"]
                )


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
        sentence = "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck as a fortuitous netcord helped the new world number one avoid a humiliating second- round exit at Wimbledon ."
        span = (39, 42)

        encoding = tokenizer(sentence, entity_spans=[span], return_token_type_ids=True)

        # test words
        self.assertEqual(len(encoding["input_ids"]), 42)
        self.assertEqual(len(encoding["attention_mask"]), 42)
        self.assertEqual(len(encoding["token_type_ids"]), 42)
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"], spaces_between_special_tokens=False),
            "<s>Top seed Ana Ivanovic said on Thursday<ent> she<ent> could hardly believe her luck as a fortuitous netcord helped the new world number one avoid a humiliating second- round exit at Wimbledon.</s>",
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
        sentence = "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck as a fortuitous netcord helped the new world number one avoid a humiliating second- round exit at Wimbledon ."
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
