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

from transformers import (
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    AutoModelForTokenClassification,
    AutoTokenizer,
    TokenClassificationPipeline,
    pipeline,
)
from transformers.pipelines import AggregationStrategy, TokenClassificationArgumentHandler
from transformers.pipelines.token_classification import \
    SlidingWindowTokenClassificationPipeline
from transformers.testing_utils import (
    is_pipeline_test,
    nested_simplify,
    require_tf,
    require_torch,
    require_torch_gpu,
    slow,
)

from .test_pipelines_common import ANY


VALID_INPUTS = ["A simple string", ["list of strings", "A simple string that is quite a bit longer"]]

# These 2 model types require different inputs than those of the usual text models.
_TO_SKIP = {"LayoutLMv2Config", "LayoutLMv3Config"}


@is_pipeline_test
class TokenClassificationPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
    tf_model_mapping = TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING

    if model_mapping is not None:
        model_mapping = {config: model for config, model in model_mapping.items() if config.__name__ not in _TO_SKIP}
    if tf_model_mapping is not None:
        tf_model_mapping = {
            config: model for config, model in tf_model_mapping.items() if config.__name__ not in _TO_SKIP
        }

    def get_test_pipeline(self, model, tokenizer, processor):
        token_classifier = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
        return token_classifier, ["A simple string", "A simple string that is quite a bit longer"]

    def run_pipeline_test(self, token_classifier, _):
        model = token_classifier.model
        tokenizer = token_classifier.tokenizer
        if not tokenizer.is_fast:
            return  # Slow tokenizers do not return offsets mappings, so this test will fail

        outputs = token_classifier("A simple string")
        self.assertIsInstance(outputs, list)
        n = len(outputs)
        self.assertEqual(
            nested_simplify(outputs),
            [
                {
                    "entity": ANY(str),
                    "score": ANY(float),
                    "start": ANY(int),
                    "end": ANY(int),
                    "index": ANY(int),
                    "word": ANY(str),
                }
                for i in range(n)
            ],
        )
        outputs = token_classifier(["list of strings", "A simple string that is quite a bit longer"])
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 2)
        n = len(outputs[0])
        m = len(outputs[1])

        self.assertEqual(
            nested_simplify(outputs),
            [
                [
                    {
                        "entity": ANY(str),
                        "score": ANY(float),
                        "start": ANY(int),
                        "end": ANY(int),
                        "index": ANY(int),
                        "word": ANY(str),
                    }
                    for i in range(n)
                ],
                [
                    {
                        "entity": ANY(str),
                        "score": ANY(float),
                        "start": ANY(int),
                        "end": ANY(int),
                        "index": ANY(int),
                        "word": ANY(str),
                    }
                    for i in range(m)
                ],
            ],
        )

        self.run_aggregation_strategy(model, tokenizer)

    def run_aggregation_strategy(self, model, tokenizer):
        token_classifier = TokenClassificationPipeline(model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        self.assertEqual(token_classifier._postprocess_params["aggregation_strategy"], AggregationStrategy.SIMPLE)
        outputs = token_classifier("A simple string")
        self.assertIsInstance(outputs, list)
        n = len(outputs)
        self.assertEqual(
            nested_simplify(outputs),
            [
                {
                    "entity_group": ANY(str),
                    "score": ANY(float),
                    "start": ANY(int),
                    "end": ANY(int),
                    "word": ANY(str),
                }
                for i in range(n)
            ],
        )

        token_classifier = TokenClassificationPipeline(model=model, tokenizer=tokenizer, aggregation_strategy="first")
        self.assertEqual(token_classifier._postprocess_params["aggregation_strategy"], AggregationStrategy.FIRST)
        outputs = token_classifier("A simple string")
        self.assertIsInstance(outputs, list)
        n = len(outputs)
        self.assertEqual(
            nested_simplify(outputs),
            [
                {
                    "entity_group": ANY(str),
                    "score": ANY(float),
                    "start": ANY(int),
                    "end": ANY(int),
                    "word": ANY(str),
                }
                for i in range(n)
            ],
        )

        token_classifier = TokenClassificationPipeline(model=model, tokenizer=tokenizer, aggregation_strategy="max")
        self.assertEqual(token_classifier._postprocess_params["aggregation_strategy"], AggregationStrategy.MAX)
        outputs = token_classifier("A simple string")
        self.assertIsInstance(outputs, list)
        n = len(outputs)
        self.assertEqual(
            nested_simplify(outputs),
            [
                {
                    "entity_group": ANY(str),
                    "score": ANY(float),
                    "start": ANY(int),
                    "end": ANY(int),
                    "word": ANY(str),
                }
                for i in range(n)
            ],
        )

        token_classifier = TokenClassificationPipeline(
            model=model, tokenizer=tokenizer, aggregation_strategy="average"
        )
        self.assertEqual(token_classifier._postprocess_params["aggregation_strategy"], AggregationStrategy.AVERAGE)
        outputs = token_classifier("A simple string")
        self.assertIsInstance(outputs, list)
        n = len(outputs)
        self.assertEqual(
            nested_simplify(outputs),
            [
                {
                    "entity_group": ANY(str),
                    "score": ANY(float),
                    "start": ANY(int),
                    "end": ANY(int),
                    "word": ANY(str),
                }
                for i in range(n)
            ],
        )

        with self.assertWarns(UserWarning):
            token_classifier = pipeline(task="ner", model=model, tokenizer=tokenizer, grouped_entities=True)
        self.assertEqual(token_classifier._postprocess_params["aggregation_strategy"], AggregationStrategy.SIMPLE)
        with self.assertWarns(UserWarning):
            token_classifier = pipeline(
                task="ner", model=model, tokenizer=tokenizer, grouped_entities=True, ignore_subwords=True
            )
        self.assertEqual(token_classifier._postprocess_params["aggregation_strategy"], AggregationStrategy.FIRST)

    @slow
    @require_torch
    def test_chunking(self):
        NER_MODEL = "elastic/distilbert-base-uncased-finetuned-conll03-english"
        model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(NER_MODEL, use_fast=True)
        tokenizer.model_max_length = 10
        stride = 5
        sentence = (
            "Hugging Face, Inc. is a French company that develops tools for building applications using machine learning. "
            "The company, based in New York City was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf."
        )

        token_classifier = TokenClassificationPipeline(
            model=model, tokenizer=tokenizer, aggregation_strategy="simple", stride=stride
        )
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output),
            [
                {"entity_group": "ORG", "score": 0.978, "word": "hugging face, inc.", "start": 0, "end": 18},
                {"entity_group": "MISC", "score": 0.999, "word": "french", "start": 24, "end": 30},
                {"entity_group": "LOC", "score": 0.997, "word": "new york city", "start": 131, "end": 144},
                {"entity_group": "MISC", "score": 0.999, "word": "french", "start": 168, "end": 174},
                {"entity_group": "PER", "score": 0.999, "word": "clement delangue", "start": 189, "end": 205},
                {"entity_group": "PER", "score": 0.999, "word": "julien chaumond", "start": 207, "end": 222},
                {"entity_group": "PER", "score": 0.999, "word": "thomas wolf", "start": 228, "end": 239},
            ],
        )

        token_classifier = TokenClassificationPipeline(
            model=model, tokenizer=tokenizer, aggregation_strategy="first", stride=stride
        )
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output),
            [
                {"entity_group": "ORG", "score": 0.978, "word": "hugging face, inc.", "start": 0, "end": 18},
                {"entity_group": "MISC", "score": 0.999, "word": "french", "start": 24, "end": 30},
                {"entity_group": "LOC", "score": 0.997, "word": "new york city", "start": 131, "end": 144},
                {"entity_group": "MISC", "score": 0.999, "word": "french", "start": 168, "end": 174},
                {"entity_group": "PER", "score": 0.999, "word": "clement delangue", "start": 189, "end": 205},
                {"entity_group": "PER", "score": 0.999, "word": "julien chaumond", "start": 207, "end": 222},
                {"entity_group": "PER", "score": 0.999, "word": "thomas wolf", "start": 228, "end": 239},
            ],
        )

        token_classifier = TokenClassificationPipeline(
            model=model, tokenizer=tokenizer, aggregation_strategy="max", stride=stride
        )
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output),
            [
                {"entity_group": "ORG", "score": 0.978, "word": "hugging face, inc.", "start": 0, "end": 18},
                {"entity_group": "MISC", "score": 0.999, "word": "french", "start": 24, "end": 30},
                {"entity_group": "LOC", "score": 0.997, "word": "new york city", "start": 131, "end": 144},
                {"entity_group": "MISC", "score": 0.999, "word": "french", "start": 168, "end": 174},
                {"entity_group": "PER", "score": 0.999, "word": "clement delangue", "start": 189, "end": 205},
                {"entity_group": "PER", "score": 0.999, "word": "julien chaumond", "start": 207, "end": 222},
                {"entity_group": "PER", "score": 0.999, "word": "thomas wolf", "start": 228, "end": 239},
            ],
        )

        token_classifier = TokenClassificationPipeline(
            model=model, tokenizer=tokenizer, aggregation_strategy="average", stride=stride
        )
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output),
            [
                {"entity_group": "ORG", "score": 0.978, "word": "hugging face, inc.", "start": 0, "end": 18},
                {"entity_group": "MISC", "score": 0.999, "word": "french", "start": 24, "end": 30},
                {"entity_group": "LOC", "score": 0.997, "word": "new york city", "start": 131, "end": 144},
                {"entity_group": "MISC", "score": 0.999, "word": "french", "start": 168, "end": 174},
                {"entity_group": "PER", "score": 0.999, "word": "clement delangue", "start": 189, "end": 205},
                {"entity_group": "PER", "score": 0.999, "word": "julien chaumond", "start": 207, "end": 222},
                {"entity_group": "PER", "score": 0.999, "word": "thomas wolf", "start": 228, "end": 239},
            ],
        )

    @require_torch
    def test_chunking_fast(self):
        # Note: We cannot run the test on "conflicts" on the chunking.
        # The problem is that the model is random, and thus the results do heavily
        # depend on the chunking, so we cannot expect "abcd" and "bcd" to find
        # the same entities. We defer to slow tests for this.
        pipe = pipeline(model="hf-internal-testing/tiny-bert-for-token-classification")
        sentence = "The company, based in New York City was founded in 2016 by French entrepreneurs"

        results = pipe(sentence, aggregation_strategy="first")
        # This is what this random model gives on the full sentence
        self.assertEqual(
            nested_simplify(results),
            [
                # This is 2 actual tokens
                {"end": 39, "entity_group": "MISC", "score": 0.115, "start": 31, "word": "city was"},
                {"end": 79, "entity_group": "MISC", "score": 0.115, "start": 66, "word": "entrepreneurs"},
            ],
        )

        # This will force the tokenizer to split after "city was".
        pipe.tokenizer.model_max_length = 12
        self.assertEqual(
            pipe.tokenizer.decode(pipe.tokenizer.encode(sentence, truncation=True)),
            "[CLS] the company, based in new york city was [SEP]",
        )

        stride = 4
        results = pipe(sentence, aggregation_strategy="first", stride=stride)
        self.assertEqual(
            nested_simplify(results),
            [
                {"end": 39, "entity_group": "MISC", "score": 0.115, "start": 31, "word": "city was"},
                # This is an extra entity found by this random model, but at least both original
                # entities are there
                {"end": 58, "entity_group": "MISC", "score": 0.115, "start": 56, "word": "by"},
                {"end": 79, "entity_group": "MISC", "score": 0.115, "start": 66, "word": "entrepreneurs"},
            ],
        )

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

    @require_torch_gpu
    @slow
    def test_gpu(self):
        sentence = "This is dummy sentence"
        ner = pipeline(
            "token-classification",
            device=0,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )

        output = ner(sentence)
        self.assertEqual(nested_simplify(output), [])

    @require_torch
    @slow
    def test_dbmdz_english(self):
        # Other sentence
        NER_MODEL = "dbmdz/bert-large-cased-finetuned-conll03-english"
        model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(NER_MODEL, use_fast=True)
        sentence = """Enzo works at the UN"""
        token_classifier = pipeline("ner", model=model, tokenizer=tokenizer)
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output),
            [
                {"entity": "I-PER", "score": 0.998, "word": "En", "start": 0, "end": 2, "index": 1},
                {"entity": "I-PER", "score": 0.997, "word": "##zo", "start": 2, "end": 4, "index": 2},
                {"entity": "I-ORG", "score": 0.999, "word": "UN", "start": 18, "end": 20, "index": 6},
            ],
        )

        token_classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output),
            [
                {"entity_group": "PER", "score": 0.997, "word": "Enzo", "start": 0, "end": 4},
                {"entity_group": "ORG", "score": 0.999, "word": "UN", "start": 18, "end": 20},
            ],
        )

        token_classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output[:3]),
            [
                {"entity_group": "PER", "score": 0.998, "word": "Enzo", "start": 0, "end": 4},
                {"entity_group": "ORG", "score": 0.999, "word": "UN", "start": 18, "end": 20},
            ],
        )

        token_classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="max")
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output[:3]),
            [
                {"entity_group": "PER", "score": 0.998, "word": "Enzo", "start": 0, "end": 4},
                {"entity_group": "ORG", "score": 0.999, "word": "UN", "start": 18, "end": 20},
            ],
        )

        token_classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="average")
        output = token_classifier(sentence)
        self.assertEqual(
            nested_simplify(output),
            [
                {"entity_group": "PER", "score": 0.997, "word": "Enzo", "start": 0, "end": 4},
                {"entity_group": "ORG", "score": 0.999, "word": "UN", "start": 18, "end": 20},
            ],
        )

    @require_torch
    @slow
    def test_aggregation_strategy_byte_level_tokenizer(self):
        sentence = "Groenlinks praat over Schiphol."
        ner = pipeline("ner", model="xlm-roberta-large-finetuned-conll02-dutch", aggregation_strategy="max")
        self.assertEqual(
            nested_simplify(ner(sentence)),
            [
                {"end": 10, "entity_group": "ORG", "score": 0.994, "start": 0, "word": "Groenlinks"},
                {"entity_group": "LOC", "score": 1.0, "word": "Schiphol.", "start": 22, "end": 31},
            ],
        )

    @require_torch
    def test_aggregation_strategy_no_b_i_prefix(self):
        model_name = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        token_classifier = pipeline(task="ner", model=model_name, tokenizer=tokenizer, framework="pt")
        # Just to understand scores indexes in this test
        token_classifier.model.config.id2label = {0: "O", 1: "MISC", 2: "PER", 3: "ORG", 4: "LOC"}
        example = [
            {
                # fmt : off
                "scores": np.array([0, 0, 0, 0, 0.9968166351318359]),
                "index": 1,
                "is_subword": False,
                "word": "En",
                "start": 0,
                "end": 2,
            },
            {
                # fmt : off
                "scores": np.array([0, 0, 0, 0, 0.9957635998725891]),
                "index": 2,
                "is_subword": True,
                "word": "##zo",
                "start": 2,
                "end": 4,
            },
            {
                # fmt: off
                "scores": np.array([0, 0, 0, 0.9986497163772583, 0]),
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
                {"end": 2, "entity": "LOC", "score": 0.997, "start": 0, "word": "En", "index": 1},
                {"end": 4, "entity": "LOC", "score": 0.996, "start": 2, "word": "##zo", "index": 2},
                {"end": 13, "entity": "ORG", "score": 0.999, "start": 11, "word": "UN", "index": 7},
            ],
        )
        self.assertEqual(
            nested_simplify(token_classifier.aggregate(example, AggregationStrategy.SIMPLE)),
            [
                {"entity_group": "LOC", "score": 0.996, "word": "Enzo", "start": 0, "end": 4},
                {"entity_group": "ORG", "score": 0.999, "word": "UN", "start": 11, "end": 13},
            ],
        )

    @require_torch
    def test_aggregation_strategy(self):
        model_name = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"
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
        model_name = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"
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
    @slow
    def test_aggregation_strategy_offsets_with_leading_space(self):
        sentence = "We're from New York"
        model_name = "brandon25/deberta-base-finetuned-ner"
        ner = pipeline("ner", model=model_name, ignore_labels=[], aggregation_strategy="max")
        self.assertEqual(
            nested_simplify(ner(sentence)),
            [
                {"entity_group": "O", "score": 1.0, "word": " We're from", "start": 0, "end": 10},
                {"entity_group": "LOC", "score": 1.0, "word": " New York", "start": 10, "end": 19},
            ],
        )

    @require_torch
    def test_gather_pre_entities(self):
        model_name = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"
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
            sentence,
            input_ids,
            scores,
            offset_mapping,
            special_tokens_mask,
            aggregation_strategy=AggregationStrategy.NONE,
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

    @require_torch
    def test_word_heuristic_leading_space(self):
        model_name = "hf-internal-testing/tiny-random-deberta-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        token_classifier = pipeline(task="ner", model=model_name, tokenizer=tokenizer, framework="pt")

        sentence = "I play the theremin"

        tokens = tokenizer(
            sentence,
            return_attention_mask=False,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_offsets_mapping=True,
        )
        offset_mapping = tokens.pop("offset_mapping").cpu().numpy()[0]
        special_tokens_mask = tokens.pop("special_tokens_mask").cpu().numpy()[0]
        input_ids = tokens["input_ids"].numpy()[0]
        scores = np.array([[1, 0] for _ in input_ids])  # values irrelevant for heuristic

        pre_entities = token_classifier.gather_pre_entities(
            sentence,
            input_ids,
            scores,
            offset_mapping,
            special_tokens_mask,
            aggregation_strategy=AggregationStrategy.FIRST,
        )

        # ensure expected tokenization and correct is_subword values
        self.assertEqual(
            [(entity["word"], entity["is_subword"]) for entity in pre_entities],
            [("▁I", False), ("▁play", False), ("▁the", False), ("▁there", False), ("min", True)],
        )

    @require_tf
    def test_tf_only(self):
        model_name = "hf-internal-testing/tiny-random-bert-tf-only"  # This model only has a TensorFlow version
        # We test that if we don't specificy framework='tf', it gets detected automatically
        token_classifier = pipeline(task="ner", model=model_name)
        self.assertEqual(token_classifier.framework, "tf")

    @require_tf
    def test_small_model_tf(self):
        model_name = "hf-internal-testing/tiny-bert-for-token-classification"
        token_classifier = pipeline(task="token-classification", model=model_name, framework="tf")
        outputs = token_classifier("This is a test !")
        self.assertEqual(
            nested_simplify(outputs),
            [
                {"entity": "I-MISC", "score": 0.115, "index": 1, "word": "this", "start": 0, "end": 4},
                {"entity": "I-MISC", "score": 0.115, "index": 2, "word": "is", "start": 5, "end": 7},
            ],
        )

    @require_torch
    def test_no_offset_tokenizer(self):
        model_name = "hf-internal-testing/tiny-bert-for-token-classification"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        token_classifier = pipeline(task="token-classification", model=model_name, tokenizer=tokenizer, framework="pt")
        outputs = token_classifier("This is a test !")
        self.assertEqual(
            nested_simplify(outputs),
            [
                {"entity": "I-MISC", "score": 0.115, "index": 1, "word": "this", "start": None, "end": None},
                {"entity": "I-MISC", "score": 0.115, "index": 2, "word": "is", "start": None, "end": None},
            ],
        )

    @require_torch
    def test_small_model_pt(self):
        model_name = "hf-internal-testing/tiny-bert-for-token-classification"
        token_classifier = pipeline(task="token-classification", model=model_name, framework="pt")
        outputs = token_classifier("This is a test !")
        self.assertEqual(
            nested_simplify(outputs),
            [
                {"entity": "I-MISC", "score": 0.115, "index": 1, "word": "this", "start": 0, "end": 4},
                {"entity": "I-MISC", "score": 0.115, "index": 2, "word": "is", "start": 5, "end": 7},
            ],
        )

        token_classifier = pipeline(
            task="token-classification", model=model_name, framework="pt", ignore_labels=["O", "I-MISC"]
        )
        outputs = token_classifier("This is a test !")
        self.assertEqual(
            nested_simplify(outputs),
            [],
        )

        token_classifier = pipeline(task="token-classification", model=model_name, framework="pt")
        # Overload offset_mapping
        outputs = token_classifier(
            "This is a test !", offset_mapping=[(0, 0), (0, 1), (0, 2), (0, 0), (0, 0), (0, 0), (0, 0)]
        )
        self.assertEqual(
            nested_simplify(outputs),
            [
                {"entity": "I-MISC", "score": 0.115, "index": 1, "word": "this", "start": 0, "end": 1},
                {"entity": "I-MISC", "score": 0.115, "index": 2, "word": "is", "start": 0, "end": 2},
            ],
        )

        # Batch size does not affect outputs (attention_mask are required)
        sentences = ["This is a test !", "Another test this is with longer sentence"]
        outputs = token_classifier(sentences)
        outputs_batched = token_classifier(sentences, batch_size=2)
        # Batching does not make a difference in predictions
        self.assertEqual(nested_simplify(outputs_batched), nested_simplify(outputs))
        self.assertEqual(
            nested_simplify(outputs_batched),
            [
                [
                    {"entity": "I-MISC", "score": 0.115, "index": 1, "word": "this", "start": 0, "end": 4},
                    {"entity": "I-MISC", "score": 0.115, "index": 2, "word": "is", "start": 5, "end": 7},
                ],
                [],
            ],
        )

    @require_torch
    def test_pt_ignore_subwords_slow_tokenizer_raises(self):
        model_name = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        with self.assertRaises(ValueError):
            pipeline(task="ner", model=model_name, tokenizer=tokenizer, aggregation_strategy=AggregationStrategy.FIRST)
        with self.assertRaises(ValueError):
            pipeline(
                task="ner", model=model_name, tokenizer=tokenizer, aggregation_strategy=AggregationStrategy.AVERAGE
            )
        with self.assertRaises(ValueError):
            pipeline(task="ner", model=model_name, tokenizer=tokenizer, aggregation_strategy=AggregationStrategy.MAX)

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

    @slow
    @require_torch
    def test_sliding_window(self):
        input_text = "Parliamentary elections were held in Estonia on 5 March 2023 to elect all 101 members of the Riigikogu. The Estonian Centre Party led by Jüri Ratas formed a government after the 2019 Estonian parliamentary election, with Ratas serving as prime minister. His government was brought down in January 2021 after a corruption investigation; Kaja Kallas of the Estonian Reform Party formed a coalition government with the Centre Party afterwards, although it collapsed in June 2022. Kallas then formed a government with Isamaa and the Social Democratic Party and remained in the position of prime minister. In January 2023, the National Electoral Committee announced that nine political parties and ten individual candidates had registered to take part in the 2023 parliamentary election.\n\nDuring the campaign period, issues discussed most extensively regarded national defence and security, due to the 2022 Russian invasion of Ukraine, and the economy. Individuals from contesting political parties also took part in organised debates in January and February 2023. Voting at foreign embassies took place from 18 to 23 February, while voters had the option to vote during the pre-election period from 27 February to 4 March. The election resulted in the victory of the Reform Party, which won 37 seats in total, while the Conservative People's Party of Estonia (EKRE) placed second with 17 seats. The Centre Party won 16 seats, a loss of 10 seats, while Estonia 200 won 14 seats, gaining representation in Riigikogu. These were the first elections where more than half of votes were cast electronically over the internet.\nThe previous parliamentary election, which was held in March 2019, saw the loss of the absolute majority held by Jüri Ratas's first cabinet in Riigikogu, the unicameral parliament of Estonia. Ratas's Centre Party, Isamaa, and Social Democratic Party (SDE) all suffered a setback in favour of the Reform Party, led by Kallas, and the EKRE. Kersti Kaljulaid, the president of Estonia, gave a mandate to Kallas to form a government after the election. The Reform Party negotiated with the Centre Party, Isamaa, and SDE but ultimately failed to form a government. After the vote in April 2019, Ratas received the mandate and successfully formed a government with Isamaa and EKRE. Jüri Ratas's second cabinet was sworn on 29 April 2019.\n\nIn January 2021, the Centre Party-led government collapsed after a corruption investigation in which the Centre Party was accused of requesting a financial support of up to €1 million within a year in return of the €39 million loan to Hillar Teder's real estate development in Tallinn In response, Ratas resigned as prime minister of Estonia, while Kallas was invited to form a government. She struck a deal with the Centre Party, with Kallas now serving as prime minister. In June 2022 however, the coalition government between the Centre and Reform Party collapsed due to the Centre Party's opposition against a law regarding education. This occurred during the 2022 Russian invasion of Ukraine; the Centre Party was seen as being close to Russians in Estonia. In response, Kallas opened negotiations with Isamaa and SDE, successfully forming Kaja Kallas's second cabinet on 15 July 2022.\n"
        token_classifier = SlidingWindowTokenClassificationPipeline(model=AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER"), tokenizer=AutoTokenizer.from_pretrained("dslim/bert-base-NER"), aggregation_strategy="FIRST")
        output_ = nested_simplify(token_classifier(input_text))

        expected_output = [
            {'entity_group': 'LOC', 'score': 1.0, 'word': 'Estonia', 'start': 37, 'end': 44},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'Riigikogu', 'start': 93, 'end': 102},
            {'entity_group': 'ORG', 'score': 0.998, 'word': 'Estonian Centre Party', 'start': 108, 'end': 129},
            {'entity_group': 'PER', 'score': 1.0, 'word': 'Jüri Ratas', 'start': 137, 'end': 147},
            {'entity_group': 'MISC', 'score': 1.0, 'word': 'Estonian', 'start': 183, 'end': 191},
            {'entity_group': 'PER', 'score': 0.999, 'word': 'Ratas', 'start': 221, 'end': 226},
            {'entity_group': 'PER', 'score': 1.0, 'word': 'Kaja Kallas', 'start': 336, 'end': 347},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'Estonian Reform Party', 'start': 355, 'end': 376},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'Centre Party', 'start': 416, 'end': 428},
            {'entity_group': 'PER', 'score': 0.999, 'word': 'Kallas', 'start': 477, 'end': 483},
            {'entity_group': 'PER', 'score': 0.989, 'word': 'Isamaa', 'start': 514, 'end': 520},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'Social Democratic Party', 'start': 529, 'end': 552},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'National Electoral Committee', 'start': 622, 'end': 650},
            {'entity_group': 'MISC', 'score': 1.0, 'word': 'Russian', 'start': 902, 'end': 909},
            {'entity_group': 'LOC', 'score': 1.0, 'word': 'Ukraine', 'start': 922, 'end': 929},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'Reform Party', 'start': 1263, 'end': 1275},
            {'entity_group': 'ORG', 'score': 0.876, 'word': "Conservative People ' s Party of Estonia", 'start': 1316, 'end': 1354},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'EKRE', 'start': 1356, 'end': 1360},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'Centre Party', 'start': 1395, 'end': 1407},
            {'entity_group': 'LOC', 'score': 1.0, 'word': 'Estonia', 'start': 1448, 'end': 1455},
            {'entity_group': 'ORG', 'score': 0.992, 'word': 'Riigikogu', 'start': 1500, 'end': 1509},
            {'entity_group': 'PER', 'score': 0.999, 'word': 'Jüri Ratas', 'start': 1729, 'end': 1739},
            {'entity_group': 'ORG', 'score': 0.987, 'word': 'Riigikogu', 'start': 1759, 'end': 1768},
            {'entity_group': 'LOC', 'score': 1.0, 'word': 'Estonia', 'start': 1799, 'end': 1806},
            {'entity_group': 'PER', 'score': 0.988, 'word': 'Ratas', 'start': 1808, 'end': 1813},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'Centre Party', 'start': 1816, 'end': 1828},
            {'entity_group': 'ORG', 'score': 0.995, 'word': 'Isamaa', 'start': 1830, 'end': 1836},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'Social Democratic Party', 'start': 1842, 'end': 1865},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'SDE', 'start': 1867, 'end': 1870},
            {'entity_group': 'ORG', 'score': 0.998, 'word': 'Reform Party', 'start': 1912, 'end': 1924},
            {'entity_group': 'PER', 'score': 0.798, 'word': 'Kallas', 'start': 1933, 'end': 1939},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'EKRE', 'start': 1949, 'end': 1953},
            {'entity_group': 'PER', 'score': 0.721, 'word': 'Kersti Kaljulaid', 'start': 1955, 'end': 1971},
            {'entity_group': 'LOC', 'score': 0.991, 'word': 'Estonia', 'start': 1990, 'end': 1997},
            {'entity_group': 'PER', 'score': 0.868, 'word': 'Kallas', 'start': 2017, 'end': 2023},
            {'entity_group': 'ORG', 'score': 0.998, 'word': 'Reform Party', 'start': 2069, 'end': 2081},
            {'entity_group': 'ORG', 'score': 0.998, 'word': 'Centre Party', 'start': 2102, 'end': 2114},
            {'entity_group': 'ORG', 'score': 0.987, 'word': 'Isamaa', 'start': 2116, 'end': 2122},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'SDE', 'start': 2128, 'end': 2131},
            {'entity_group': 'ORG', 'score': 0.67, 'word': 'Ratas', 'start': 2206, 'end': 2211},
            {'entity_group': 'ORG', 'score': 0.953, 'word': 'Isamaa', 'start': 2275, 'end': 2281},
            {'entity_group': 'ORG', 'score': 0.998, 'word': 'EKRE', 'start': 2286, 'end': 2290},
            {'entity_group': 'PER', 'score': 0.569, 'word': 'Jüri Ratas', 'start': 2292, 'end': 2302},
            {'entity_group': 'ORG', 'score': 0.996, 'word': 'Centre Party', 'start': 2370, 'end': 2382},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'Centre Party', 'start': 2454, 'end': 2466},
            {'entity_group': 'PER', 'score': 0.999, 'word': 'Hillar Teder', 'start': 2584, 'end': 2596},
            {'entity_group': 'LOC', 'score': 0.999, 'word': 'Tallinn', 'start': 2626, 'end': 2633},
            {'entity_group': 'PER', 'score': 0.997, 'word': 'Ratas', 'start': 2647, 'end': 2652},
            {'entity_group': 'LOC', 'score': 1.0, 'word': 'Estonia', 'start': 2683, 'end': 2690},
            {'entity_group': 'PER', 'score': 0.999, 'word': 'Kallas', 'start': 2698, 'end': 2704},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'Centre Party', 'start': 2766, 'end': 2778},
            {'entity_group': 'PER', 'score': 0.999, 'word': 'Kallas', 'start': 2785, 'end': 2791},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'Centre', 'start': 2882, 'end': 2888},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'Reform Party', 'start': 2893, 'end': 2905},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'Centre Party', 'start': 2927, 'end': 2939},
            {'entity_group': 'MISC', 'score': 1.0, 'word': 'Russian', 'start': 3018, 'end': 3025},
            {'entity_group': 'LOC', 'score': 1.0, 'word': 'Ukraine', 'start': 3038, 'end': 3045},
            {'entity_group': 'ORG', 'score': 0.994, 'word': 'Centre Party', 'start': 3051, 'end': 3063},
            {'entity_group': 'ORG', 'score': 0.716, 'word': 'Russians', 'start': 3091, 'end': 3099},
            {'entity_group': 'LOC', 'score': 1.0, 'word': 'Estonia', 'start': 3103, 'end': 3110},
            {'entity_group': 'PER', 'score': 0.771, 'word': 'Kallas', 'start': 3125, 'end': 3131},
            {'entity_group': 'ORG', 'score': 0.944, 'word': 'Isamaa', 'start': 3157, 'end': 3163},
            {'entity_group': 'ORG', 'score': 0.998, 'word': 'SDE', 'start': 3168, 'end': 3171},
            {'entity_group': 'PER', 'score': 0.816, 'word': 'Kaja', 'start': 3194, 'end': 3198},
            {'entity_group': 'PER', 'score': 0.74, 'word': 'Kallas', 'start': 3199, 'end': 3205}
        ]
        self.assertEqual(
            output_,
            expected_output
        )

        input_text = "The media landscape in the United States is dominated by a handful of major news companies that are known for their influential coverage of national and international news. Some of the most well-known news companies in the US include CNN, Fox News, MSNBC, The New York Times, The Washington Post, and The Wall Street Journal. CNN, or the Cable News Network, is a global news company that was founded in 1980 by Ted Turner. It is known for its 24-hour coverage of breaking news and its analysis of politics, business, and world events. CNN has correspondents in over 200 countries and reaches millions of viewers around the world. Fox News, on the other hand, is a conservative news company that was founded in 1996 by Rupert Murdoch. It is known for its opinionated coverage of politics, business, and culture, and has been criticized for its biased reporting. Despite this criticism, Fox News is one of the most-watched cable news networks in the US.MSNBC, or the Microsoft National Broadcasting Company, is a liberal news company that was founded in 1996 by Microsoft and NBC. It is known for its left-leaning political coverage and its analysis of social issues such as race, gender, and LGBT rights. MSNBC is the second-most watched cable news network in the US, after Fox News.The New York Times is one of the most influential newspapers in the US, and is known for its in-depth reporting on national and international news. It was founded in 1851 and has won more Pulitzer Prizes than any other newspaper. The New York Times is often considered to be a \"paper of record\" for its authoritative reporting on politics, business, and culture.The Washington Post is another major newspaper in the US that is known for its investigative reporting and political coverage. It was founded in 1877 and has won 69 Pulitzer Prizes. The Washington Post is also known for its coverage of national security issues, and its reporting on the Watergate scandal in the 1970s led to the resignation of President Richard Nixon.The Wall Street Journal is a business-focused newspaper that was founded in 1889. It is known for its coverage of finance, economics, and business news, and is widely read by business leaders and investors around the world. The Wall Street Journal has won 37 Pulitzer Prizes, and is owned by News Corp, which is also the parent company of Fox News.These major news companies in the US have a significant impact on the national and international discourse, and are often cited as sources of information and analysis by policymakers, journalists, and academics. While each company has its own distinctive voice and perspective, they all play a vital role in shaping the public's understanding of the world around them.However, these news companies have also faced criticism for their coverage, including accusations of bias, sensationalism, and the propagation of fake news. In recent years, the rise of social media and alternative news sources has challenged the dominance of traditional news companies, and has raised questions about the role of news media in the modern world. Despite these challenges, however, the major news companies in the US remain an important part of the media landscape, and will continue to shape public discourse for years to come."
        output_ = nested_simplify(token_classifier(input_text))
        expected_output = [
            {'entity_group': 'LOC', 'score': 0.999, 'word': 'United States', 'start': 27, 'end': 40},
            {'entity_group': 'LOC', 'score': 0.999, 'word': 'US', 'start': 223, 'end': 225},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'CNN', 'start': 234, 'end': 237},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'Fox News', 'start': 239, 'end': 247},
            {'entity_group': 'ORG', 'score': 0.998, 'word': 'MSNBC', 'start': 249, 'end': 254},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'The New York Times', 'start': 256, 'end': 274},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'The Washington Post', 'start': 276, 'end': 295},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'The Wall Street Journal', 'start': 301, 'end': 324},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'CNN', 'start': 326, 'end': 329},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'Cable News Network', 'start': 338, 'end': 356},
            {'entity_group': 'PER', 'score': 0.988, 'word': 'Ted Turner', 'start': 411, 'end': 421},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'CNN', 'start': 535, 'end': 538},
            {'entity_group': 'ORG', 'score': 0.998, 'word': 'Fox News', 'start': 630, 'end': 638},
            {'entity_group': 'PER', 'score': 0.994, 'word': 'Rupert Murdoch', 'start': 718, 'end': 732},
            {'entity_group': 'ORG', 'score': 0.998, 'word': 'Fox News', 'start': 885, 'end': 893},
            {'entity_group': 'LOC', 'score': 0.998, 'word': 'US', 'start': 948, 'end': 950},
            {'entity_group': 'ORG', 'score': 0.996, 'word': 'MSNBC', 'start': 951, 'end': 956},
            {'entity_group': 'ORG', 'score': 0.999, 'word': 'Microsoft National Broadcasting Company', 'start': 965, 'end': 1004},
            {'entity_group': 'ORG', 'score': 0.997, 'word': 'Microsoft', 'start': 1060, 'end': 1069},
            {'entity_group': 'ORG', 'score': 0.998, 'word': 'NBC', 'start': 1074, 'end': 1077},
            {'entity_group': 'MISC', 'score': 0.863, 'word': 'LGBT', 'start': 1191, 'end': 1195},
            {'entity_group': 'ORG', 'score': 0.972, 'word': 'MSNBC', 'start': 1204, 'end': 1209},
            {'entity_group': 'LOC', 'score': 0.998, 'word': 'US', 'start': 1263, 'end': 1265},
            {'entity_group': 'ORG', 'score': 0.996, 'word': 'Fox News', 'start': 1273, 'end': 1281},
            {'entity_group': 'ORG', 'score': 0.968, 'word': 'The New York Times', 'start': 1282, 'end': 1300},
            {'entity_group': 'LOC', 'score': 0.999, 'word': 'US', 'start': 1350, 'end': 1352},
            {'entity_group': 'MISC', 'score': 0.997, 'word': 'Pulitzer Prizes', 'start': 1470, 'end': 1485},
            {'entity_group': 'ORG', 'score': 0.908, 'word': 'The New York Times', 'start': 1512, 'end': 1530},
            {'entity_group': 'ORG', 'score': 0.996, 'word': 'The Washington Post', 'start': 1644, 'end': 1663},
            {'entity_group': 'LOC', 'score': 0.999, 'word': 'US', 'start': 1698, 'end': 1700},
            {'entity_group': 'MISC', 'score': 0.991, 'word': 'Pulitzer Prizes', 'start': 1809, 'end': 1824},
            {'entity_group': 'ORG', 'score': 0.959, 'word': 'The Washington Post', 'start': 1826, 'end': 1845},
            {'entity_group': 'MISC', 'score': 0.604, 'word': 'Watergate', 'start': 1931, 'end': 1940},
            {'entity_group': 'PER', 'score': 0.758, 'word': 'Richard Nixon', 'start': 1998, 'end': 2011},
            {'entity_group': 'ORG', 'score': 0.963, 'word': 'Wall Street Journal', 'start': 2016, 'end': 2035},
            {'entity_group': 'ORG', 'score': 0.981, 'word': 'Wall Street Journal', 'start': 2240, 'end': 2259},
            {'entity_group': 'MISC', 'score': 0.846, 'word': 'Pulitzer Prizes', 'start': 2271, 'end': 2286},
            {'entity_group': 'ORG', 'score': 0.996, 'word': 'News Corp', 'start': 2304, 'end': 2313},
            {'entity_group': 'ORG', 'score': 0.812, 'word': 'Fox News', 'start': 2351, 'end': 2359},
            {'entity_group': 'LOC', 'score': 0.997, 'word': 'US', 'start': 2394, 'end': 2396},
            {'entity_group': 'LOC', 'score': 0.999, 'word': 'US', 'start': 3158, 'end': 3160}
        ]
        self.assertEqual(
            output_,
            expected_output
        )


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
