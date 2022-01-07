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


import os
import unittest
from typing import Tuple

from transformers.models.mluke.tokenization_mluke import MLukeTokenizer
from transformers.testing_utils import require_torch, slow

from .test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/test_sentencepiece.model")
SAMPLE_ENTITY_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/test_entity_vocab.json")


class MLukeTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = MLukeTokenizer
    test_rust_tokenizer = False
    from_pretrained_kwargs = {"cls_token": "<s>"}

    def setUp(self):
        super().setUp()

        self.special_tokens_map = {"entity_token_1": "<ent>", "entity_token_2": "<ent2>"}

    def get_tokenizer(self, task=None, **kwargs):
        kwargs.update(self.special_tokens_map)
        kwargs.update({"task": task})
        tokenizer = MLukeTokenizer(vocab_file=SAMPLE_VOCAB, entity_vocab_file=SAMPLE_ENTITY_VOCAB, **kwargs)
        tokenizer.sanitize_special_tokens()
        return tokenizer

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()
        text = "lower newer"
        spm_tokens = ["▁l", "ow", "er", "▁new", "er"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, spm_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_spm_tokens = [149, 116, 40, 410, 40] + [3]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_spm_tokens)

    def mluke_dict_integration_testing(self):
        tokenizer = self.get_tokenizer()

        self.assertListEqual(tokenizer.encode("Hello world!", add_special_tokens=False), [35378, 8999, 38])
        self.assertListEqual(
            tokenizer.encode("Hello world! cécé herlolip 418", add_special_tokens=False),
            [35378, 8999, 38, 33273, 11676, 604, 365, 21392, 201, 1819],
        )

    def test_sequence_builders(self):
        tokenizer = self.tokenizer_class.from_pretrained("hf-internal-testing/tiny-random-mluke")

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

    def get_clean_sequence(self, tokenizer, max_length=20) -> Tuple[str, list]:
        txt = "Beyonce lives in Los Angeles"
        ids = tokenizer.encode(txt, add_special_tokens=False)
        return txt, ids

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

    def test_if_tokenize_single_text_raise_error_with_invalid_inputs(self):
        tokenizer = self.get_tokenizer()

        sentence = "ISO 639-3 uses the code fas for the dialects spoken across Iran and Afghanistan."
        entities = ["DUMMY"]
        spans = [(0, 9)]

        with self.assertRaises(ValueError):
            tokenizer(sentence, entities=tuple(entities), entity_spans=spans)

        with self.assertRaises(ValueError):
            tokenizer(sentence, entities=entities, entity_spans=tuple(spans))

        with self.assertRaises(ValueError):
            tokenizer(sentence, entities=[0], entity_spans=spans)

        with self.assertRaises(ValueError):
            tokenizer(sentence, entities=entities, entity_spans=[0])

        with self.assertRaises(ValueError):
            tokenizer(sentence, entities=entities, entity_spans=spans + [(0, 9)])

    def test_if_tokenize_entity_classification_raise_error_with_invalid_inputs(self):
        tokenizer = self.get_tokenizer(task="entity_classification")

        sentence = "Japanese is an East Asian language spoken by about 128 million people, primarily in Japan."
        span = (15, 34)

        with self.assertRaises(ValueError):
            tokenizer(sentence, entity_spans=[])

        with self.assertRaises(ValueError):
            tokenizer(sentence, entity_spans=[span, span])

        with self.assertRaises(ValueError):
            tokenizer(sentence, entity_spans=[0])

    def test_if_tokenize_entity_pair_classification_raise_error_with_invalid_inputs(self):
        tokenizer = self.get_tokenizer(task="entity_pair_classification")

        sentence = "Japanese is an East Asian language spoken by about 128 million people, primarily in Japan."
        # head and tail information

        with self.assertRaises(ValueError):
            tokenizer(sentence, entity_spans=[])

        with self.assertRaises(ValueError):
            tokenizer(sentence, entity_spans=[0, 0])

    def test_if_tokenize_entity_span_classification_raise_error_with_invalid_inputs(self):
        tokenizer = self.get_tokenizer(task="entity_span_classification")

        sentence = "Japanese is an East Asian language spoken by about 128 million people, primarily in Japan."

        with self.assertRaises(ValueError):
            tokenizer(sentence, entity_spans=[])

        with self.assertRaises(ValueError):
            tokenizer(sentence, entity_spans=[0, 0, 0])


@slow
@require_torch
class MLukeTokenizerIntegrationTests(unittest.TestCase):
    tokenizer_class = MLukeTokenizer
    from_pretrained_kwargs = {"cls_token": "<s>"}

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = MLukeTokenizer.from_pretrained("studio-ousia/mluke-base", return_token_type_ids=True)
        cls.entity_classification_tokenizer = MLukeTokenizer.from_pretrained(
            "studio-ousia/mluke-base", return_token_type_ids=True, task="entity_classification"
        )
        cls.entity_pair_tokenizer = MLukeTokenizer.from_pretrained(
            "studio-ousia/mluke-base", return_token_type_ids=True, task="entity_pair_classification"
        )

        cls.entity_span_tokenizer = MLukeTokenizer.from_pretrained(
            "studio-ousia/mluke-base", return_token_type_ids=True, task="entity_span_classification"
        )

    def test_single_text_no_padding_or_truncation(self):
        tokenizer = self.tokenizer
        sentence = "ISO 639-3 uses the code fas for the dialects spoken across Iran and アフガニスタン (Afghanistan)."
        entities = ["en:ISO 639-3", "DUMMY_ENTITY", "ja:アフガニスタン", "en:Afghanistan"]
        spans = [(0, 9), (59, 63), (68, 75), (77, 88)]

        encoding = tokenizer(sentence, entities=entities, entity_spans=spans, return_token_type_ids=True)

        self.assertEqual(
            tokenizer.decode(encoding["input_ids"], spaces_between_special_tokens=False),
            "<s> ISO 639-3 uses the code fas for the dialects spoken across Iran and アフガニスタン ( Afghanistan ).</s>",
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][1:5], spaces_between_special_tokens=False), "ISO 639-3"
        )
        self.assertEqual(tokenizer.decode(encoding["input_ids"][17], spaces_between_special_tokens=False), "Iran")
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][19:25], spaces_between_special_tokens=False), "アフガニスタン"
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][26], spaces_between_special_tokens=False), "Afghanistan"
        )

        self.assertEqual(
            encoding["entity_ids"],
            [
                tokenizer.entity_vocab["en:ISO 639-3"],
                tokenizer.entity_vocab["[UNK]"],
                tokenizer.entity_vocab["ja:アフガニスタン"],
                tokenizer.entity_vocab["en:Afghanistan"],
            ],
        )
        self.assertEqual(encoding["entity_attention_mask"], [1, 1, 1, 1])
        self.assertEqual(encoding["entity_token_type_ids"], [0, 0, 0, 0])
        # fmt: off
        self.assertEqual(
            encoding["entity_position_ids"],
            [
                [1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [17, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [19, 20, 21, 22, 23, 24, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [26, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            ]
        )
        # fmt: on

    def test_single_text_only_entity_spans_no_padding_or_truncation(self):
        tokenizer = self.tokenizer

        sentence = "ISO 639-3 uses the code fas for the dialects spoken across Iran and アフガニスタン (Afghanistan)."
        entities = ["en:ISO 639-3", "DUMMY_ENTITY", "ja:アフガニスタン", "en:Afghanistan"]
        spans = [(0, 9), (59, 63), (68, 75), (77, 88)]

        encoding = tokenizer(sentence, entities=entities, entity_spans=spans, return_token_type_ids=True)

        self.assertEqual(
            tokenizer.decode(encoding["input_ids"], spaces_between_special_tokens=False),
            "<s> ISO 639-3 uses the code fas for the dialects spoken across Iran and アフガニスタン ( Afghanistan ).</s>",
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][1:5], spaces_between_special_tokens=False), "ISO 639-3"
        )
        self.assertEqual(tokenizer.decode(encoding["input_ids"][17], spaces_between_special_tokens=False), "Iran")
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][20:25], spaces_between_special_tokens=False), "アフガニスタン"
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][26], spaces_between_special_tokens=False), "Afghanistan"
        )

        self.assertEqual(
            encoding["entity_ids"],
            [
                tokenizer.entity_vocab["en:ISO 639-3"],
                tokenizer.entity_vocab["[UNK]"],
                tokenizer.entity_vocab["ja:アフガニスタン"],
                tokenizer.entity_vocab["en:Afghanistan"],
            ],
        )
        self.assertEqual(encoding["entity_attention_mask"], [1, 1, 1, 1])
        self.assertEqual(encoding["entity_token_type_ids"], [0, 0, 0, 0])
        # fmt: off
        self.assertEqual(
            encoding["entity_position_ids"],
            [
                [1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [17, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [19, 20, 21, 22, 23, 24, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [26, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            ]
        )
        # fmt: on

    def test_single_text_padding_pytorch_tensors(self):
        tokenizer = self.tokenizer

        sentence = "ISO 639-3 uses the code fas for the dialects spoken across Iran and アフガニスタン (Afghanistan)."
        entities = ["en:ISO 639-3", "DUMMY_ENTITY", "ja:アフガニスタン", "en:Afghanistan"]
        spans = [(0, 9), (59, 63), (68, 75), (77, 88)]

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
        tokenizer = self.tokenizer

        sentence = "ISO 639-3 uses the code fas"
        sentence_pair = "for the dialects spoken across Iran and アフガニスタン (Afghanistan)."
        entities = ["en:ISO 639-3"]
        entities_pair = ["DUMMY_ENTITY", "ja:アフガニスタン", "en:Afghanistan"]
        spans = [(0, 9)]
        spans_pair = [(31, 35), (40, 47), (49, 60)]

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
            "<s> ISO 639-3 uses the code fas</s></s> for the dialects spoken across Iran and アフガニスタン ( Afghanistan ).</s>",
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][1:5], spaces_between_special_tokens=False), "ISO 639-3"
        )
        self.assertEqual(tokenizer.decode(encoding["input_ids"][19], spaces_between_special_tokens=False), "Iran")
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][21:27], spaces_between_special_tokens=False), "アフガニスタン"
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][28], spaces_between_special_tokens=False), "Afghanistan"
        )

        self.assertEqual(
            encoding["entity_ids"],
            [
                tokenizer.entity_vocab["en:ISO 639-3"],
                tokenizer.entity_vocab["[UNK]"],
                tokenizer.entity_vocab["ja:アフガニスタン"],
                tokenizer.entity_vocab["en:Afghanistan"],
            ],
        )
        self.assertEqual(encoding["entity_attention_mask"], [1, 1, 1, 1])
        self.assertEqual(encoding["entity_token_type_ids"], [0, 0, 0, 0])
        # fmt: off
        self.assertEqual(
            encoding["entity_position_ids"],
            [
                [1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [19, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [21, 22, 23, 24, 25, 26, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [28, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            ]
        )
        # fmt: on

    def test_text_pair_only_entity_spans_no_padding_or_truncation(self):
        tokenizer = self.tokenizer

        sentence = "ISO 639-3 uses the code fas"
        sentence_pair = "for the dialects spoken across Iran and アフガニスタン (Afghanistan)."
        entities = ["en:ISO 639-3"]
        entities_pair = ["DUMMY_ENTITY", "ja:アフガニスタン", "en:Afghanistan"]
        spans = [(0, 9)]
        spans_pair = [(31, 35), (40, 47), (49, 60)]

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
            "<s> ISO 639-3 uses the code fas</s></s> for the dialects spoken across Iran and アフガニスタン ( Afghanistan ).</s>",
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][1:5], spaces_between_special_tokens=False), "ISO 639-3"
        )
        self.assertEqual(tokenizer.decode(encoding["input_ids"][19], spaces_between_special_tokens=False), "Iran")
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][21:27], spaces_between_special_tokens=False), "アフガニスタン"
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][28], spaces_between_special_tokens=False), "Afghanistan"
        )

        self.assertEqual(
            encoding["entity_ids"],
            [
                tokenizer.entity_vocab["en:ISO 639-3"],
                tokenizer.entity_vocab["[UNK]"],
                tokenizer.entity_vocab["ja:アフガニスタン"],
                tokenizer.entity_vocab["en:Afghanistan"],
            ],
        )
        # fmt: off
        self.assertEqual(
            encoding["entity_position_ids"],
            [
                [1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [19, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [21, 22, 23, 24, 25, 26, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [28, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            ]
        )
        # fmt: on

    def test_text_pair_padding_pytorch_tensors(self):
        tokenizer = self.tokenizer

        sentence = "ISO 639-3 uses the code fas"
        sentence_pair = "for the dialects spoken across Iran and アフガニスタン (Afghanistan)."
        entities = ["en:ISO 639-3"]
        entities_pair = ["DUMMY_ENTITY", "ja:アフガニスタン", "en:Afghanistan"]
        spans = [(0, 9)]
        spans_pair = [(31, 35), (40, 47), (49, 60)]

        encoding = tokenizer(
            sentence,
            sentence_pair,
            entities=entities,
            entities_pair=entities_pair,
            entity_spans=spans,
            entity_spans_pair=spans_pair,
            return_token_type_ids=True,
            padding="max_length",
            max_length=40,
            max_entity_length=16,
            return_tensors="pt",
        )

        # test words
        self.assertEqual(encoding["input_ids"].shape, (1, 40))
        self.assertEqual(encoding["attention_mask"].shape, (1, 40))
        self.assertEqual(encoding["token_type_ids"].shape, (1, 40))

        # test entities
        self.assertEqual(encoding["entity_ids"].shape, (1, 16))
        self.assertEqual(encoding["entity_attention_mask"].shape, (1, 16))
        self.assertEqual(encoding["entity_token_type_ids"].shape, (1, 16))
        self.assertEqual(encoding["entity_position_ids"].shape, (1, 16, tokenizer.max_mention_length))

    def test_entity_classification_no_padding_or_truncation(self):
        tokenizer = self.entity_classification_tokenizer

        sentence = "Japanese is an East Asian language spoken by about 128 million people, primarily in Japan."
        span = (15, 34)

        encoding = tokenizer(sentence, entity_spans=[span], return_token_type_ids=True)

        # test words
        self.assertEqual(len(encoding["input_ids"]), 23)
        self.assertEqual(len(encoding["attention_mask"]), 23)
        self.assertEqual(len(encoding["token_type_ids"]), 23)
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"], spaces_between_special_tokens=False),
            "<s> Japanese is an<ent>East Asian language<ent>spoken by about 128 million people, primarily in Japan.</s>",
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][4:9], spaces_between_special_tokens=False),
            "<ent>East Asian language<ent>",
        )

        # test entities
        mask_id = tokenizer.entity_vocab["[MASK]"]
        self.assertEqual(encoding["entity_ids"], [mask_id])
        self.assertEqual(encoding["entity_attention_mask"], [1])
        self.assertEqual(encoding["entity_token_type_ids"], [0])
        # fmt: off
        self.assertEqual(
            encoding["entity_position_ids"],
            [[4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
        )
        # fmt: on

    def test_entity_classification_padding_pytorch_tensors(self):
        tokenizer = self.entity_classification_tokenizer

        sentence = "Japanese is an East Asian language spoken by about 128 million people, primarily in Japan."
        span = (15, 34)

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
        tokenizer = self.entity_pair_tokenizer

        sentence = "Japanese is an East Asian language spoken by about 128 million people, primarily in Japan."
        # head and tail information
        spans = [(0, 8), (84, 89)]

        encoding = tokenizer(sentence, entity_spans=spans, return_token_type_ids=True)

        self.assertEqual(
            tokenizer.decode(encoding["input_ids"], spaces_between_special_tokens=False),
            "<s><ent>Japanese<ent>is an East Asian language spoken by about 128 million people, primarily in<ent2>Japan<ent2>.</s>",
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][1:4], spaces_between_special_tokens=False),
            "<ent>Japanese<ent>",
        )
        self.assertEqual(
            tokenizer.decode(encoding["input_ids"][20:23], spaces_between_special_tokens=False), "<ent2>Japan<ent2>"
        )

        mask_id = tokenizer.entity_vocab["[MASK]"]
        mask2_id = tokenizer.entity_vocab["[MASK2]"]
        self.assertEqual(encoding["entity_ids"], [mask_id, mask2_id])
        self.assertEqual(encoding["entity_attention_mask"], [1, 1])
        self.assertEqual(encoding["entity_token_type_ids"], [0, 0])
        # fmt: off
        self.assertEqual(
            encoding["entity_position_ids"],
            [
                [1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [20, 21, 22, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            ]
        )
        # fmt: on

    def test_entity_pair_classification_padding_pytorch_tensors(self):
        tokenizer = self.entity_pair_tokenizer

        sentence = "Japanese is an East Asian language spoken by about 128 million people, primarily in Japan."
        # head and tail information
        spans = [(0, 8), (84, 89)]

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
        tokenizer = self.entity_span_tokenizer

        sentence = "Japanese is an East Asian language spoken by about 128 million people, primarily in Japan."
        spans = [(0, 8), (15, 34), (84, 89)]

        encoding = tokenizer(sentence, entity_spans=spans, return_token_type_ids=True)

        self.assertEqual(
            tokenizer.decode(encoding["input_ids"], spaces_between_special_tokens=False),
            "<s> Japanese is an East Asian language spoken by about 128 million people, primarily in Japan.</s>",
        )

        mask_id = tokenizer.entity_vocab["[MASK]"]
        self.assertEqual(encoding["entity_ids"], [mask_id, mask_id, mask_id])
        self.assertEqual(encoding["entity_attention_mask"], [1, 1, 1])
        self.assertEqual(encoding["entity_token_type_ids"], [0, 0, 0])
        # fmt: off
        self.assertEqual(
            encoding["entity_position_ids"],
            [
                [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [4, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [18, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
        )
        # fmt: on
        self.assertEqual(encoding["entity_start_positions"], [1, 4, 18])
        self.assertEqual(encoding["entity_end_positions"], [1, 6, 18])

    def test_entity_span_classification_padding_pytorch_tensors(self):
        tokenizer = self.entity_span_tokenizer

        sentence = "Japanese is an East Asian language spoken by about 128 million people, primarily in Japan."
        spans = [(0, 8), (15, 34), (84, 89)]

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
