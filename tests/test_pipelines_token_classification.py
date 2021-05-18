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

import numpy as np

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from transformers.pipelines import AggregationStrategy, Pipeline, TokenClassificationArgumentHandler
from transformers.testing_utils import nested_simplify, require_tf, require_torch, slow

from .test_pipelines_common import CustomInputPipelineCommonMixin


VALID_INPUTS = ["A simple string", ["list of strings", "A simple string that is quite a bit longer"]]


class TokenClassificationPipelineTests(CustomInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "ner"
    small_models = [
        "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"
    ]  # Default model - Models tested without the @slow decorator
    large_models = []  # Models tested with the @slow decorator

    def _test_pipeline(self, token_classifier: Pipeline):
        output_keys = {"entity", "word", "score", "start", "end", "index"}
        if token_classifier.aggregation_strategy != AggregationStrategy.NONE:
            output_keys = {"entity_group", "word", "score", "start", "end"}

        self.assertIsNotNone(token_classifier)

        mono_result = token_classifier(VALID_INPUTS[0])
        self.assertIsInstance(mono_result, list)
        self.assertIsInstance(mono_result[0], (dict, list))

        if isinstance(mono_result[0], list):
            mono_result = mono_result[0]

        for key in output_keys:
            self.assertIn(key, mono_result[0])

        multi_result = [token_classifier(input) for input in VALID_INPUTS]
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], (dict, list))

        if isinstance(multi_result[0], list):
            multi_result = multi_result[0]

        for result in multi_result:
            for key in output_keys:
                self.assertIn(key, result)

    @require_torch
    @slow
    def test_spanish_bert(self):
        # https://github.com/huggingface/transformers/pull/4987
        NER_MODEL = "mrm8488/bert-spanish-cased-finetuned-ner"
        model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(NER_MODEL, use_fast=True)
        sentence = """Consuelo Araújo Noguera, ministra de cultura del presidente Andrés Pastrana (1998.2002) fue asesinada por las Farc luego de haber permanecido secuestrada por algunos meses."""

        token_classifier = pipeline("ner", model=model, tokenizer=tokenizer)
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output[:3]),
            [
                {"entity": "B-PER", "score": 0.999, "word": "Cons", "start": 0, "end": 4, "index": 1},
                {"entity": "B-PER", "score": 0.803, "word": "##uelo", "start": 4, "end": 8, "index": 2},
                {"entity": "I-PER", "score": 0.999, "word": "Ara", "start": 9, "end": 12, "index": 3},
            ],
        )

        token_classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output[:3]),
            [
                {"entity_group": "PER", "score": 0.999, "word": "Cons", "start": 0, "end": 4},
                {"entity_group": "PER", "score": 0.966, "word": "##uelo Araújo Noguera", "start": 4, "end": 23},
                {"entity_group": "PER", "score": 1.0, "word": "Andrés Pastrana", "start": 60, "end": 75},
            ],
        )

        token_classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output[:3]),
            [
                {"entity_group": "PER", "score": 0.999, "word": "Consuelo Araújo Noguera", "start": 0, "end": 23},
                {"entity_group": "PER", "score": 1.0, "word": "Andrés Pastrana", "start": 60, "end": 75},
                {"entity_group": "ORG", "score": 0.999, "word": "Farc", "start": 110, "end": 114},
            ],
        )

        token_classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="max")
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output[:3]),
            [
                {"entity_group": "PER", "score": 0.999, "word": "Consuelo Araújo Noguera", "start": 0, "end": 23},
                {"entity_group": "PER", "score": 1.0, "word": "Andrés Pastrana", "start": 60, "end": 75},
                {"entity_group": "ORG", "score": 0.999, "word": "Farc", "start": 110, "end": 114},
            ],
        )

        token_classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="average")
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output[:3]),
            [
                {"entity_group": "PER", "score": 0.966, "word": "Consuelo Araújo Noguera", "start": 0, "end": 23},
                {"entity_group": "PER", "score": 1.0, "word": "Andrés Pastrana", "start": 60, "end": 75},
                {"entity_group": "ORG", "score": 0.542, "word": "Farc", "start": 110, "end": 114},
            ],
        )

    @require_torch
    @slow
    def test_dbmdz_english(self):
        # Other sentence
        NER_MODEL = "dbmdz/bert-large-cased-finetuned-conll03-english"
        model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(NER_MODEL, use_fast=True)
        sentence = """Enzo works at the the UN"""
        token_classifier = pipeline("ner", model=model, tokenizer=tokenizer)
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output),
            [
                {"entity": "I-PER", "score": 0.997, "word": "En", "start": 0, "end": 2, "index": 1},
                {"entity": "I-PER", "score": 0.996, "word": "##zo", "start": 2, "end": 4, "index": 2},
                {"entity": "I-ORG", "score": 0.999, "word": "UN", "start": 22, "end": 24, "index": 7},
            ],
        )

        token_classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output),
            [
                {"entity_group": "PER", "score": 0.996, "word": "Enzo", "start": 0, "end": 4},
                {"entity_group": "ORG", "score": 0.999, "word": "UN", "start": 22, "end": 24},
            ],
        )

        token_classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output[:3]),
            [
                {"entity_group": "PER", "score": 0.997, "word": "Enzo", "start": 0, "end": 4},
                {"entity_group": "ORG", "score": 0.999, "word": "UN", "start": 22, "end": 24},
            ],
        )

        token_classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="max")
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output[:3]),
            [
                {"entity_group": "PER", "score": 0.997, "word": "Enzo", "start": 0, "end": 4},
                {"entity_group": "ORG", "score": 0.999, "word": "UN", "start": 22, "end": 24},
            ],
        )

        token_classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="average")
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output),
            [
                {"entity_group": "PER", "score": 0.996, "word": "Enzo", "start": 0, "end": 4},
                {"entity_group": "ORG", "score": 0.999, "word": "UN", "start": 22, "end": 24},
            ],
        )

    @require_torch
    def test_aggregation_strategy(self):
        model_name = self.small_models[0]
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        token_classifier = pipeline(task="ner", model=model_name, tokenizer=tokenizer, framework="pt")
        # Just to understand scores indexes in this test
        self.assertEqual(
            token_classifier.model.config.id2label,
            {0: "O", 1: "B-MISC", 2: "I-MISC", 3: "B-PER", 4: "I-PER", 5: "B-ORG", 6: "I-ORG", 7: "B-LOC", 8: "I-LOC"},
        )
        example = [
            {
                # fmt : off
                "scores": np.array([0, 0, 0, 0, 0.9968166351318359, 0, 0, 0]),
                "index": 1,
                "is_subword": False,
                "word": "En",
                "start": 0,
                "end": 2,
            },
            {
                # fmt : off
                "scores": np.array([0, 0, 0, 0, 0.9957635998725891, 0, 0, 0]),
                "index": 2,
                "is_subword": True,
                "word": "##zo",
                "start": 2,
                "end": 4,
            },
            {
                # fmt: off
                "scores": np.array([0, 0, 0, 0, 0, 0.9986497163772583, 0, 0, ]),
                # fmt: on
                "index": 7,
                "word": "UN",
                "is_subword": False,
                "start": 11,
                "end": 13,
            },
        ]
        self.assertEqual(
            nested_simplify(token_classifier.aggregate(example, AggregationStrategy.NONE)),
            [
                {"end": 2, "entity": "I-PER", "score": 0.997, "start": 0, "word": "En", "index": 1},
                {"end": 4, "entity": "I-PER", "score": 0.996, "start": 2, "word": "##zo", "index": 2},
                {"end": 13, "entity": "B-ORG", "score": 0.999, "start": 11, "word": "UN", "index": 7},
            ],
        )
        self.assertEqual(
            nested_simplify(token_classifier.aggregate(example, AggregationStrategy.SIMPLE)),
            [
                {"entity_group": "PER", "score": 0.996, "word": "Enzo", "start": 0, "end": 4},
                {"entity_group": "ORG", "score": 0.999, "word": "UN", "start": 11, "end": 13},
            ],
        )
        self.assertEqual(
            nested_simplify(token_classifier.aggregate(example, AggregationStrategy.FIRST)),
            [
                {"entity_group": "PER", "score": 0.997, "word": "Enzo", "start": 0, "end": 4},
                {"entity_group": "ORG", "score": 0.999, "word": "UN", "start": 11, "end": 13},
            ],
        )
        self.assertEqual(
            nested_simplify(token_classifier.aggregate(example, AggregationStrategy.MAX)),
            [
                {"entity_group": "PER", "score": 0.997, "word": "Enzo", "start": 0, "end": 4},
                {"entity_group": "ORG", "score": 0.999, "word": "UN", "start": 11, "end": 13},
            ],
        )
        self.assertEqual(
            nested_simplify(token_classifier.aggregate(example, AggregationStrategy.AVERAGE)),
            [
                {"entity_group": "PER", "score": 0.996, "word": "Enzo", "start": 0, "end": 4},
                {"entity_group": "ORG", "score": 0.999, "word": "UN", "start": 11, "end": 13},
            ],
        )

    @require_torch
    def test_aggregation_strategy_example2(self):
        model_name = self.small_models[0]
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        token_classifier = pipeline(task="ner", model=model_name, tokenizer=tokenizer, framework="pt")
        # Just to understand scores indexes in this test
        self.assertEqual(
            token_classifier.model.config.id2label,
            {0: "O", 1: "B-MISC", 2: "I-MISC", 3: "B-PER", 4: "I-PER", 5: "B-ORG", 6: "I-ORG", 7: "B-LOC", 8: "I-LOC"},
        )
        example = [
            {
                # Necessary for AVERAGE
                "scores": np.array([0, 0.55, 0, 0.45, 0, 0, 0, 0, 0, 0]),
                "is_subword": False,
                "index": 1,
                "word": "Ra",
                "start": 0,
                "end": 2,
            },
            {
                "scores": np.array([0, 0, 0, 0.2, 0, 0, 0, 0.8, 0, 0]),
                "is_subword": True,
                "word": "##ma",
                "start": 2,
                "end": 4,
                "index": 2,
            },
            {
                # 4th score will have the higher average
                # 4th score is B-PER for this model
                # It's does not correspond to any of the subtokens.
                "scores": np.array([0, 0, 0, 0.4, 0, 0, 0.6, 0, 0, 0]),
                "is_subword": True,
                "word": "##zotti",
                "start": 11,
                "end": 13,
                "index": 3,
            },
        ]
        self.assertEqual(
            token_classifier.aggregate(example, AggregationStrategy.NONE),
            [
                {"end": 2, "entity": "B-MISC", "score": 0.55, "start": 0, "word": "Ra", "index": 1},
                {"end": 4, "entity": "B-LOC", "score": 0.8, "start": 2, "word": "##ma", "index": 2},
                {"end": 13, "entity": "I-ORG", "score": 0.6, "start": 11, "word": "##zotti", "index": 3},
            ],
        )

        self.assertEqual(
            token_classifier.aggregate(example, AggregationStrategy.FIRST),
            [{"entity_group": "MISC", "score": 0.55, "word": "Ramazotti", "start": 0, "end": 13}],
        )
        self.assertEqual(
            token_classifier.aggregate(example, AggregationStrategy.MAX),
            [{"entity_group": "LOC", "score": 0.8, "word": "Ramazotti", "start": 0, "end": 13}],
        )
        self.assertEqual(
            nested_simplify(token_classifier.aggregate(example, AggregationStrategy.AVERAGE)),
            [{"entity_group": "PER", "score": 0.35, "word": "Ramazotti", "start": 0, "end": 13}],
        )

    @require_torch
    def test_gather_pre_entities(self):

        model_name = self.small_models[0]
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        token_classifier = pipeline(task="ner", model=model_name, tokenizer=tokenizer, framework="pt")

        sentence = "Hello there"

        tokens = tokenizer(
            sentence,
            return_attention_mask=False,
            return_tensors="pt",
            truncation=True,
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
        offset_mapping = tokens.pop("offset_mapping").cpu().numpy()[0]
        special_tokens_mask = tokens.pop("special_tokens_mask").cpu().numpy()[0]
        input_ids = tokens["input_ids"].numpy()[0]
        # First element in [CLS]
        scores = np.array([[1, 0, 0], [0.1, 0.3, 0.6], [0.8, 0.1, 0.1]])

        pre_entities = token_classifier.gather_pre_entities(
            sentence, input_ids, scores, offset_mapping, special_tokens_mask
        )
        self.assertEqual(
            nested_simplify(pre_entities),
            [
                {"word": "Hello", "scores": [0.1, 0.3, 0.6], "start": 0, "end": 5, "is_subword": False, "index": 1},
                {
                    "word": "there",
                    "scores": [0.8, 0.1, 0.1],
                    "index": 2,
                    "start": 6,
                    "end": 11,
                    "is_subword": False,
                },
            ],
        )

    @require_tf
    def test_tf_only(self):
        model_name = "Narsil/small"  # This model only has a TensorFlow version
        # We test that if we don't specificy framework='tf', it gets detected automatically
        token_classifier = pipeline(task="ner", model=model_name)
        self._test_pipeline(token_classifier)

    @require_tf
    def test_tf_defaults(self):
        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            token_classifier = pipeline(task="ner", model=model_name, tokenizer=tokenizer, framework="tf")
        self._test_pipeline(token_classifier)

    @require_tf
    def test_tf_small_ignore_subwords_available_for_fast_tokenizers(self):
        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            token_classifier = pipeline(
                task="ner",
                model=model_name,
                tokenizer=tokenizer,
                framework="tf",
                aggregation_strategy=AggregationStrategy.FIRST,
            )
            self._test_pipeline(token_classifier)

        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            token_classifier = pipeline(
                task="ner",
                model=model_name,
                tokenizer=tokenizer,
                framework="tf",
                aggregation_strategy=AggregationStrategy.SIMPLE,
            )
            self._test_pipeline(token_classifier)

    @require_torch
    def test_pt_ignore_subwords_slow_tokenizer_raises(self):
        model_name = self.small_models[0]
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        with self.assertRaises(ValueError):
            pipeline(task="ner", model=model_name, tokenizer=tokenizer, aggregation_strategy=AggregationStrategy.FIRST)
        with self.assertRaises(ValueError):
            pipeline(
                task="ner", model=model_name, tokenizer=tokenizer, aggregation_strategy=AggregationStrategy.AVERAGE
            )
        with self.assertRaises(ValueError):
            pipeline(task="ner", model=model_name, tokenizer=tokenizer, aggregation_strategy=AggregationStrategy.MAX)

    @require_torch
    def test_pt_defaults_slow_tokenizer(self):
        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            token_classifier = pipeline(task="ner", model=model_name, tokenizer=tokenizer)
            self._test_pipeline(token_classifier)

    @require_torch
    def test_pt_defaults(self):
        for model_name in self.small_models:
            token_classifier = pipeline(task="ner", model=model_name)
            self._test_pipeline(token_classifier)

    @slow
    @require_torch
    def test_warnings(self):
        with self.assertWarns(UserWarning):
            token_classifier = pipeline(task="ner", model=self.small_models[0], grouped_entities=True)
        self.assertEqual(token_classifier.aggregation_strategy, AggregationStrategy.SIMPLE)
        with self.assertWarns(UserWarning):
            token_classifier = pipeline(
                task="ner", model=self.small_models[0], grouped_entities=True, ignore_subwords=True
            )
        self.assertEqual(token_classifier.aggregation_strategy, AggregationStrategy.FIRST)

    @slow
    @require_torch
    def test_simple(self):
        token_classifier = pipeline(task="ner", model="dslim/bert-base-NER", grouped_entities=True)
        sentence = "Hello Sarah Jessica Parker who Jessica lives in New York"
        sentence2 = "This is a simple test"
        output = token_classifier(sentence)

        output_ = nested_simplify(output)

        self.assertEqual(
            output_,
            [
                {
                    "entity_group": "PER",
                    "score": 0.996,
                    "word": "Sarah Jessica Parker",
                    "start": 6,
                    "end": 26,
                },
                {"entity_group": "PER", "score": 0.977, "word": "Jessica", "start": 31, "end": 38},
                {"entity_group": "LOC", "score": 0.999, "word": "New York", "start": 48, "end": 56},
            ],
        )

        output = token_classifier([sentence, sentence2])
        output_ = nested_simplify(output)

        self.assertEqual(
            output_,
            [
                [
                    {"entity_group": "PER", "score": 0.996, "word": "Sarah Jessica Parker", "start": 6, "end": 26},
                    {"entity_group": "PER", "score": 0.977, "word": "Jessica", "start": 31, "end": 38},
                    {"entity_group": "LOC", "score": 0.999, "word": "New York", "start": 48, "end": 56},
                ],
                [],
            ],
        )

    @require_torch
    def test_pt_small_ignore_subwords_available_for_fast_tokenizers(self):
        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            token_classifier = pipeline(
                task="ner", model=model_name, tokenizer=tokenizer, grouped_entities=True, ignore_subwords=True
            )
            self._test_pipeline(token_classifier)

        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            token_classifier = pipeline(
                task="ner", model=model_name, tokenizer=tokenizer, grouped_entities=True, ignore_subwords=False
            )
            self._test_pipeline(token_classifier)


class TokenClassificationArgumentHandlerTestCase(unittest.TestCase):
    def setUp(self):
        self.args_parser = TokenClassificationArgumentHandler()

    def test_simple(self):
        string = "This is a simple input"

        inputs, offset_mapping = self.args_parser(string)
        self.assertEqual(inputs, [string])
        self.assertEqual(offset_mapping, None)

        inputs, offset_mapping = self.args_parser([string, string])
        self.assertEqual(inputs, [string, string])
        self.assertEqual(offset_mapping, None)

        inputs, offset_mapping = self.args_parser(string, offset_mapping=[(0, 1), (1, 2)])
        self.assertEqual(inputs, [string])
        self.assertEqual(offset_mapping, [[(0, 1), (1, 2)]])

        inputs, offset_mapping = self.args_parser(
            [string, string], offset_mapping=[[(0, 1), (1, 2)], [(0, 2), (2, 3)]]
        )
        self.assertEqual(inputs, [string, string])
        self.assertEqual(offset_mapping, [[(0, 1), (1, 2)], [(0, 2), (2, 3)]])

    def test_errors(self):
        string = "This is a simple input"

        # 2 sentences, 1 offset_mapping, args
        with self.assertRaises(TypeError):
            self.args_parser(string, string, offset_mapping=[[(0, 1), (1, 2)]])

        # 2 sentences, 1 offset_mapping, args
        with self.assertRaises(TypeError):
            self.args_parser(string, string, offset_mapping=[(0, 1), (1, 2)])

        # 2 sentences, 1 offset_mapping, input_list
        with self.assertRaises(ValueError):
            self.args_parser([string, string], offset_mapping=[[(0, 1), (1, 2)]])

        # 2 sentences, 1 offset_mapping, input_list
        with self.assertRaises(ValueError):
            self.args_parser([string, string], offset_mapping=[(0, 1), (1, 2)])

        # 1 sentences, 2 offset_mapping
        with self.assertRaises(ValueError):
            self.args_parser(string, offset_mapping=[[(0, 1), (1, 2)], [(0, 2), (2, 3)]])

        # 0 sentences, 1 offset_mapping
        with self.assertRaises(TypeError):
            self.args_parser(offset_mapping=[[(0, 1), (1, 2)]])
