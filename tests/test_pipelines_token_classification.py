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

import json
import os
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

        ungrouped_ner_inputs = [
            [
                {
                    "entity": "B-PER",
                    "index": 1,
                    "score": 0.9994944930076599,
                    "is_subword": False,
                    "word": "Cons",
                    "start": 0,
                    "end": 4,
                },
                {
                    "entity": "B-PER",
                    "index": 2,
                    "score": 0.8025449514389038,
                    "is_subword": True,
                    "word": "##uelo",
                    "start": 4,
                    "end": 8,
                },
                {
                    "entity": "I-PER",
                    "index": 3,
                    "score": 0.9993102550506592,
                    "is_subword": False,
                    "word": "Ara",
                    "start": 9,
                    "end": 11,
                },
                {
                    "entity": "I-PER",
                    "index": 4,
                    "score": 0.9993743896484375,
                    "is_subword": True,
                    "word": "##új",
                    "start": 11,
                    "end": 13,
                },
                {
                    "entity": "I-PER",
                    "index": 5,
                    "score": 0.9992871880531311,
                    "is_subword": True,
                    "word": "##o",
                    "start": 13,
                    "end": 14,
                },
                {
                    "entity": "I-PER",
                    "index": 6,
                    "score": 0.9993029236793518,
                    "is_subword": False,
                    "word": "No",
                    "start": 15,
                    "end": 17,
                },
                {
                    "entity": "I-PER",
                    "index": 7,
                    "score": 0.9981776475906372,
                    "is_subword": True,
                    "word": "##guera",
                    "start": 17,
                    "end": 22,
                },
                {
                    "entity": "B-PER",
                    "index": 15,
                    "score": 0.9998136162757874,
                    "is_subword": False,
                    "word": "Andrés",
                    "start": 23,
                    "end": 28,
                },
                {
                    "entity": "I-PER",
                    "index": 16,
                    "score": 0.999740719795227,
                    "is_subword": False,
                    "word": "Pas",
                    "start": 29,
                    "end": 32,
                },
                {
                    "entity": "I-PER",
                    "index": 17,
                    "score": 0.9997414350509644,
                    "is_subword": True,
                    "word": "##tran",
                    "start": 32,
                    "end": 36,
                },
                {
                    "entity": "I-PER",
                    "index": 18,
                    "score": 0.9996136426925659,
                    "is_subword": True,
                    "word": "##a",
                    "start": 36,
                    "end": 37,
                },
                {
                    "entity": "B-ORG",
                    "index": 28,
                    "score": 0.9989739060401917,
                    "is_subword": False,
                    "word": "Far",
                    "start": 39,
                    "end": 42,
                },
                {
                    "entity": "I-ORG",
                    "index": 29,
                    "score": 0.7188422083854675,
                    "is_subword": True,
                    "word": "##c",
                    "start": 42,
                    "end": 43,
                },
            ],
            [
                {
                    "entity": "I-PER",
                    "index": 1,
                    "score": 0.9968166351318359,
                    "is_subword": False,
                    "word": "En",
                    "start": 0,
                    "end": 2,
                },
                {
                    "entity": "I-PER",
                    "index": 2,
                    "score": 0.9957635998725891,
                    "is_subword": True,
                    "word": "##zo",
                    "start": 2,
                    "end": 4,
                },
                {
                    "entity": "I-ORG",
                    "index": 7,
                    "score": 0.9986497163772583,
                    "is_subword": False,
                    "word": "UN",
                    "start": 11,
                    "end": 13,
                },
            ],
        ]

        ungrouped_inputs_all_scores = [
            {
                "word": "T",
                "score": np.array(
                    [
                        9.29590046e-01,
                        1.67631579e-03,
                        1.13618866e-04,
                        5.24316654e-02,
                        2.45529664e-04,
                        1.76369259e-03,
                        7.47492450e-05,
                        1.37843387e-02,
                        3.20013991e-04,
                    ]
                ),
                "index": 1,
                "start": 0,
                "end": 1,
                "is_subword": False,
            },
            {
                "word": "##H",
                "score": np.array(
                    [
                        9.6568632e-01,
                        1.0918044e-03,
                        1.3044178e-04,
                        1.8581267e-02,
                        3.0361942e-04,
                        5.9505191e-04,
                        1.3611461e-04,
                        1.3115727e-02,
                        3.5968653e-04,
                    ]
                ),
                "index": 2,
                "start": 1,
                "end": 2,
                "is_subword": True,
            },
            {
                "word": "##e",
                "score": np.array(
                    [
                        5.6958544e-01,
                        6.7770253e-03,
                        3.0190460e-04,
                        1.6712399e-01,
                        8.6468906e-04,
                        3.3661814e-03,
                        2.7250437e-04,
                        2.5070670e-01,
                        1.0015352e-03,
                    ]
                ),
                "index": 3,
                "start": 2,
                "end": 3,
                "is_subword": True,
            },
            {
                "word": "quasi",
                "score": np.array(
                    [
                        3.6461815e-02,
                        4.7163994e-04,
                        7.7644864e-04,
                        9.0705883e-03,
                        3.4141801e-03,
                        5.1569089e-04,
                        4.5607667e-04,
                        6.2320358e-01,
                        3.2563004e-01,
                    ]
                ),
                "index": 4,
                "start": 4,
                "end": 9,
                "is_subword": False,
            },
            {
                "word": "##ame",
                "score": np.array(
                    [
                        0.15970601,
                        0.00174455,
                        0.00181823,
                        0.00997834,
                        0.00794088,
                        0.00160232,
                        0.00122593,
                        0.28658998,
                        0.5293938,
                    ]
                ),
                "index": 5,
                "start": 9,
                "end": 12,
                "is_subword": True,
            },
            {
                "word": "##rica",
                "score": np.array(
                    [
                        6.4172082e-02,
                        1.5784599e-03,
                        1.1795119e-03,
                        1.3267660e-02,
                        2.3705978e-03,
                        1.2073644e-03,
                        6.1625458e-04,
                        6.7841798e-01,
                        2.3719010e-01,
                    ]
                ),
                "index": 6,
                "start": 12,
                "end": 16,
                "is_subword": True,
            },
            {
                "word": "##n",
                "score": np.array(
                    [
                        2.8728947e-02,
                        3.4915505e-04,
                        8.2594587e-04,
                        3.7162553e-03,
                        4.2545195e-03,
                        6.7599339e-04,
                        5.5553199e-04,
                        6.6298299e-02,
                        8.9459532e-01,
                    ]
                ),
                "index": 7,
                "start": 16,
                "end": 17,
                "is_subword": True,
            },
            {
                "word": "Mark",
                "score": np.array(
                    [
                        3.5328791e-03,
                        9.8336750e-01,
                        8.3820109e-04,
                        7.2647040e-03,
                        4.9874303e-04,
                        5.9280143e-04,
                        2.9107687e-04,
                        2.3955773e-03,
                        1.2184654e-03,
                    ]
                ),
                "index": 8,
                "start": 18,
                "end": 22,
                "is_subword": False,
            },
            {
                "word": "Must",
                "score": np.array(
                    [
                        2.59605236e-04,
                        2.98619037e-04,
                        9.98437107e-01,
                        1.12211506e-04,
                        1.69078063e-04,
                        2.54273153e-04,
                        1.28210217e-04,
                        6.97476498e-05,
                        2.71162222e-04,
                    ]
                ),
                "index": 9,
                "start": 23,
                "end": 27,
                "is_subword": False,
            },
            {
                "word": "##erman",
                "score": np.array(
                    [
                        2.4742461e-04,
                        5.6770945e-04,
                        9.9810404e-01,
                        8.5848493e-05,
                        3.5093780e-04,
                        2.3541904e-04,
                        1.4951559e-04,
                        5.4812852e-05,
                        2.0431961e-04,
                    ]
                ),
                "index": 10,
                "start": 27,
                "end": 32,
                "is_subword": True,
            },
            {
                "word": "works",
                "score": np.array(
                    [
                        9.9952888e-01,
                        3.4134755e-05,
                        3.2518183e-05,
                        2.5094318e-05,
                        2.4778466e-04,
                        1.6179767e-05,
                        3.7871898e-05,
                        1.8624953e-05,
                        5.8835212e-05,
                    ]
                ),
                "index": 11,
                "start": 33,
                "end": 38,
                "is_subword": False,
            },
            {
                "word": "at",
                "score": np.array(
                    [
                        9.9974287e-01,
                        1.8711693e-05,
                        6.3416733e-06,
                        3.5407997e-05,
                        1.3391797e-04,
                        2.0818115e-05,
                        1.8278386e-05,
                        9.1749180e-06,
                        1.4530495e-05,
                    ]
                ),
                "index": 12,
                "start": 39,
                "end": 41,
                "is_subword": False,
            },
            {
                "word": "Made",
                "score": np.array(
                    [
                        1.4745704e-03,
                        3.5883293e-03,
                        7.5529744e-05,
                        2.7471399e-01,
                        2.6137877e-04,
                        7.1945870e-01,
                        5.7339250e-05,
                        3.3724433e-04,
                        3.2908196e-05,
                    ]
                ),
                "index": 13,
                "start": 42,
                "end": 46,
                "is_subword": False,
            },
            {
                "word": "##up",
                "score": np.array(
                    [
                        1.9155587e-03,
                        5.6133829e-03,
                        8.0013328e-05,
                        3.6844590e-01,
                        3.8169319e-04,
                        6.2301862e-01,
                        8.4445433e-05,
                        4.2535455e-04,
                        3.5037294e-05,
                    ]
                ),
                "index": 14,
                "start": 46,
                "end": 48,
                "is_subword": True,
            },
            {
                "word": "##pit",
                "score": np.array(
                    [
                        1.7833502e-03,
                        3.1480102e-03,
                        7.2892937e-05,
                        4.9196857e-01,
                        2.8379925e-04,
                        5.0234205e-01,
                        7.0629183e-05,
                        2.9715578e-04,
                        3.3520439e-05,
                    ]
                ),
                "index": 15,
                "start": 48,
                "end": 51,
                "is_subword": True,
            },
            {
                "word": "##y",
                "score": np.array(
                    [
                        8.12840555e-03,
                        1.23405308e-02,
                        3.25592933e-04,
                        3.28718781e-01,
                        1.23740966e-03,
                        6.48038268e-01,
                        3.58282792e-04,
                        7.40314135e-04,
                        1.12461144e-04,
                    ]
                ),
                "index": 16,
                "start": 51,
                "end": 52,
                "is_subword": True,
            },
        ]

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
            os.path.dirname(os.path.abspath(__file__)),
            "fixtures/ner_pipeline_aligned.json")
        with open(expected_aligned_results_filepath) as expected_aligned_results_file:
            expected_aligned_results = json.load(expected_aligned_results_file)

        expected_aligned_results_w_subword_filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "fixtures/ner_pipeline_aligned_w_subwords.json")
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
