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

import numpy as np

from transformers import AutoTokenizer, pipeline
from transformers.pipelines import Pipeline, TokenClassificationArgumentHandler
from transformers.testing_utils import require_tf, require_torch, slow

from .test_pipelines_common import CustomInputPipelineCommonMixin


VALID_INPUTS = ["A simple string", ["list of strings", "A simple string that is quite a bit longer"]]


class TokenClassificationPipelineTests(CustomInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "ner"
    small_models = [
        "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"
    ]  # Default model - Models tested without the @slow decorator
    large_models = []  # Models tested with the @slow decorator

    def _test_pipeline(self, nlp: Pipeline):
        output_keys = {"entity", "word", "score", "start", "end"}
        if nlp.grouped_entities:
            output_keys = {"entity_group", "word", "score", "start", "end"}

        ungrouped_ner_inputs_filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "fixtures/ner_pipeline_inputs.json"
        )
        with open(ungrouped_ner_inputs_filepath) as ungrouped_ner_inputs_file:
            ungrouped_ner_inputs = json.load(ungrouped_ner_inputs_file)

        ungrouped_inputs_all_scores_filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "fixtures/ner_pipeline_inputs_all_scores.json"
        )
        with open(ungrouped_inputs_all_scores_filepath) as ungrouped_inputs_all_scores_file:
            ungrouped_inputs_all_scores = json.load(ungrouped_inputs_all_scores_file)

        expected_grouped_ner_results = [
            [
                {
                    "entity_group": "PER",
                    "score": 0.999369223912557,
                    "word": "Consuelo Araújo Noguera",
                    "start": 0,
                    "end": 22,
                },
                {
                    "entity_group": "PER",
                    "score": 0.9997771680355072,
                    "word": "Andrés Pastrana",
                    "start": 23,
                    "end": 37,
                },
                {"entity_group": "ORG", "score": 0.9989739060401917, "word": "Farc", "start": 39, "end": 43},
            ],
            [
                {"entity_group": "PER", "score": 0.9968166351318359, "word": "Enzo", "start": 0, "end": 4},
                {"entity_group": "ORG", "score": 0.9986497163772583, "word": "UN", "start": 11, "end": 13},
            ],
        ]

        expected_grouped_ner_results_w_subword = [
            [
                {"entity_group": "PER", "score": 0.9994944930076599, "word": "Cons", "start": 0, "end": 4},
                {
                    "entity_group": "PER",
                    "score": 0.9663328925768534,
                    "word": "##uelo Araújo Noguera",
                    "start": 4,
                    "end": 22,
                },
                {
                    "entity_group": "PER",
                    "score": 0.9997273534536362,
                    "word": "Andrés Pastrana",
                    "start": 23,
                    "end": 37,
                },
                {"entity_group": "ORG", "score": 0.8589080572128296, "word": "Farc", "start": 39, "end": 43},
            ],
            [
                {"entity_group": "PER", "score": 0.9962901175022125, "word": "Enzo", "start": 0, "end": 4},
                {"entity_group": "ORG", "score": 0.9986497163772583, "word": "UN", "start": 11, "end": 13},
            ],
        ]

        expected_aligned_results_filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "fixtures/ner_pipeline_aligned.json"
        )
        with open(expected_aligned_results_filepath) as expected_aligned_results_file:
            expected_aligned_results = json.load(expected_aligned_results_file)

        expected_aligned_results_w_subword_filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "fixtures/ner_pipeline_aligned_w_subwords.json"
        )
        with open(expected_aligned_results_w_subword_filepath) as expected_aligned_results_w_subword_file:
            expected_aligned_results_w_subword = json.load(expected_aligned_results_w_subword_file)

        self.assertIsNotNone(nlp)

        mono_result = nlp(VALID_INPUTS[0])
        self.assertIsInstance(mono_result, list)
        self.assertIsInstance(mono_result[0], (dict, list))

        if isinstance(mono_result[0], list):
            mono_result = mono_result[0]

        for key in output_keys:
            self.assertIn(key, mono_result[0])

        multi_result = [nlp(input) for input in VALID_INPUTS]
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], (dict, list))

        if isinstance(multi_result[0], list):
            multi_result = multi_result[0]

        for result in multi_result:
            for key in output_keys:
                self.assertIn(key, result)

        if nlp.grouped_entities:
            if nlp.ignore_subwords:
                for ungrouped_input, grouped_result in zip(ungrouped_ner_inputs, expected_grouped_ner_results):
                    self.assertEqual(nlp.group_entities(ungrouped_input), grouped_result)
            else:
                for ungrouped_input, grouped_result in zip(
                    ungrouped_ner_inputs, expected_grouped_ner_results_w_subword
                ):
                    self.assertEqual(nlp.group_entities(ungrouped_input), grouped_result)
        aligned_results = nlp.set_subwords_label(ungrouped_inputs_all_scores)
        if nlp.ignore_subwords:
            self.assertEqual(aligned_results, expected_aligned_results[nlp.aggregation_strategy.value])
        else:
            self.assertEqual(aligned_results, expected_aligned_results_w_subword[nlp.aggregation_strategy.value])

    @require_tf
    def test_tf_only(self):
        model_name = "Narsil/small"  # This model only has a TensorFlow version
        # We test that if we don't specificy framework='tf', it gets detected automatically
        nlp = pipeline(task="ner", model=model_name)
        self._test_pipeline(nlp)

    @require_tf
    def test_tf_defaults(self):
        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            nlp = pipeline(task="ner", model=model_name, tokenizer=tokenizer, framework="tf")
        self._test_pipeline(nlp)

    @require_tf
    def test_tf_small_ignore_subwords_available_for_fast_tokenizers(self):
        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            nlp = pipeline(
                task="ner",
                model=model_name,
                tokenizer=tokenizer,
                framework="tf",
                grouped_entities=True,
                ignore_subwords=True,
            )
            self._test_pipeline(nlp)

        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            nlp = pipeline(
                task="ner",
                model=model_name,
                tokenizer=tokenizer,
                framework="tf",
                grouped_entities=True,
                ignore_subwords=False,
            )
            self._test_pipeline(nlp)

        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            nlp = pipeline(task="ner", model=model_name, tokenizer=tokenizer, framework="tf")
        self._test_pipeline(nlp)

    @require_tf
    def test_tf_small_subwords_aggregation_available_for_fast_tokenizers(self):
        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            for strategy in ["first", "max", "average"]:
                nlp = pipeline(
                    task="ner",
                    model=model_name,
                    tokenizer=tokenizer,
                    framework="tf",
                    aggregation_strategy=strategy,
                    ignore_subwords=True,
                )
                self._test_pipeline(nlp)
                nlp = pipeline(
                    task="ner",
                    model=model_name,
                    tokenizer=tokenizer,
                    framework="tf",
                    aggregation_strategy=strategy,
                    ignore_subwords=False,
                )
                self._test_pipeline(nlp)

    @require_torch
    def test_pt_ignore_subwords_slow_tokenizer_raises(self):
        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

            with self.assertRaises(ValueError):
                pipeline(task="ner", model=model_name, tokenizer=tokenizer, ignore_subwords=True, use_fast=False)

    @require_torch
    def test_pt_defaults_slow_tokenizer(self):
        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            nlp = pipeline(task="ner", model=model_name, tokenizer=tokenizer)
            self._test_pipeline(nlp)

    @require_torch
    def test_pt_defaults(self):
        for model_name in self.small_models:
            nlp = pipeline(task="ner", model=model_name)
            self._test_pipeline(nlp)

    @slow
    @require_torch
    def test_simple(self):
        nlp = pipeline(task="ner", model="dslim/bert-base-NER", grouped_entities=True)
        sentence = "Hello Sarah Jessica Parker who Jessica lives in New York"
        sentence2 = "This is a simple test"
        output = nlp(sentence)

        def simplify(output):
            if isinstance(output, (list, tuple)):
                return [simplify(item) for item in output]
            elif isinstance(output, dict):
                return {simplify(k): simplify(v) for k, v in output.items()}
            elif isinstance(output, (str, int, np.int64)):
                return output
            elif isinstance(output, float):
                return round(output, 3)
            else:
                raise Exception(f"Cannot handle {type(output)}")

        output_ = simplify(output)

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

        output = nlp([sentence, sentence2])
        output_ = simplify(output)

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
            nlp = pipeline(
                task="ner", model=model_name, tokenizer=tokenizer, grouped_entities=True, ignore_subwords=True
            )
            self._test_pipeline(nlp)

        for model_name in self.small_models:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            nlp = pipeline(
                task="ner", model=model_name, tokenizer=tokenizer, grouped_entities=True, ignore_subwords=False
            )
            self._test_pipeline(nlp)


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
