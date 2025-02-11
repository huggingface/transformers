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
from transformers.testing_utils import (
    is_pipeline_test,
    is_torch_available,
    nested_simplify,
    require_tf,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from .test_pipelines_common import ANY


if is_torch_available():
    import torch


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

    def get_test_pipeline(
        self,
        model,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        processor=None,
        torch_dtype="float32",
    ):
        token_classifier = TokenClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            torch_dtype=torch_dtype,
        )
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

    @require_torch_accelerator
    @slow
    def test_accelerator(self):
        sentence = "This is dummy sentence"
        ner = pipeline(
            "token-classification",
            device=torch_device,
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
        ner = pipeline("ner", model="FacebookAI/xlm-roberta-large-finetuned-conll02-dutch", aggregation_strategy="max")
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
                "scores": np.array([0, 0, 0, 0, 0.9968166351318359]),  # fmt : skip
                "index": 1,
                "is_subword": False,
                "word": "En",
                "start": 0,
                "end": 2,
            },
            {
                "scores": np.array([0, 0, 0, 0, 0.9957635998725891]),  # fmt : skip
                "index": 2,
                "is_subword": True,
                "word": "##zo",
                "start": 2,
                "end": 4,
            },
            {
                "scores": np.array([0, 0, 0, 0.9986497163772583, 0]),  # fmt : skip
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
                "scores": np.array([0, 0, 0, 0, 0.9968166351318359, 0, 0, 0]),  # fmt : skip
                "index": 1,
                "is_subword": False,
                "word": "En",
                "start": 0,
                "end": 2,
            },
            {
                "scores": np.array([0, 0, 0, 0, 0.9957635998725891, 0, 0, 0]),  # fmt : skip
                "index": 2,
                "is_subword": True,
                "word": "##zo",
                "start": 2,
                "end": 4,
            },
            {
                "scores": np.array([0, 0, 0, 0, 0, 0.9986497163772583, 0, 0]),  # fmt : skip
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
    def test_small_model_pt_fp16(self):
        model_name = "hf-internal-testing/tiny-bert-for-token-classification"
        token_classifier = pipeline(
            task="token-classification", model=model_name, framework="pt", torch_dtype=torch.float16
        )
        outputs = token_classifier("This is a test !")
        self.assertEqual(
            nested_simplify(outputs),
            [
                {"entity": "I-MISC", "score": 0.115, "index": 1, "word": "this", "start": 0, "end": 4},
                {"entity": "I-MISC", "score": 0.115, "index": 2, "word": "is", "start": 5, "end": 7},
            ],
        )

    @require_torch
    def test_small_model_pt_bf16(self):
        model_name = "hf-internal-testing/tiny-bert-for-token-classification"
        token_classifier = pipeline(
            task="token-classification", model=model_name, framework="pt", torch_dtype=torch.bfloat16
        )
        outputs = token_classifier("This is a test !")
        self.assertEqual(
            nested_simplify(outputs),
            [
                {"entity": "I-MISC", "score": 0.115, "index": 1, "word": "this", "start": 0, "end": 4},
                {"entity": "I-MISC", "score": 0.115, "index": 2, "word": "is", "start": 5, "end": 7},
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
