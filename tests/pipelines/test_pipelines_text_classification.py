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

from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    TextClassificationPipeline,
    pipeline,
)
from transformers.testing_utils import is_pipeline_test, nested_simplify, require_tf, require_torch, slow

from .test_pipelines_common import ANY, PipelineTestCaseMeta


@is_pipeline_test
class TextClassificationPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    tf_model_mapping = TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

    @require_torch
    def test_small_model_pt(self):
        text_classifier = pipeline(
            task="text-classification", model="hf-internal-testing/tiny-random-distilbert", framework="pt"
        )

        outputs = text_classifier("This is great !")
        self.assertEqual(nested_simplify(outputs), [{"label": "LABEL_0", "score": 0.504}])

        outputs = text_classifier("This is great !", top_k=2)
        self.assertEqual(
            nested_simplify(outputs), [{"label": "LABEL_0", "score": 0.504}, {"label": "LABEL_1", "score": 0.496}]
        )

        outputs = text_classifier(["This is great !", "This is bad"], top_k=2)
        self.assertEqual(
            nested_simplify(outputs),
            [
                [{"label": "LABEL_0", "score": 0.504}, {"label": "LABEL_1", "score": 0.496}],
                [{"label": "LABEL_0", "score": 0.504}, {"label": "LABEL_1", "score": 0.496}],
            ],
        )

        outputs = text_classifier("This is great !", top_k=1)
        self.assertEqual(nested_simplify(outputs), [{"label": "LABEL_0", "score": 0.504}])

        # Legacy behavior
        outputs = text_classifier("This is great !", return_all_scores=False)
        self.assertEqual(nested_simplify(outputs), [{"label": "LABEL_0", "score": 0.504}])

        outputs = text_classifier("This is great !", return_all_scores=True)
        self.assertEqual(
            nested_simplify(outputs), [[{"label": "LABEL_0", "score": 0.504}, {"label": "LABEL_1", "score": 0.496}]]
        )

        outputs = text_classifier(["This is great !", "Something else"], return_all_scores=True)
        self.assertEqual(
            nested_simplify(outputs),
            [
                [{"label": "LABEL_0", "score": 0.504}, {"label": "LABEL_1", "score": 0.496}],
                [{"label": "LABEL_0", "score": 0.504}, {"label": "LABEL_1", "score": 0.496}],
            ],
        )

        outputs = text_classifier(["This is great !", "Something else"], return_all_scores=False)
        self.assertEqual(
            nested_simplify(outputs),
            [
                {"label": "LABEL_0", "score": 0.504},
                {"label": "LABEL_0", "score": 0.504},
            ],
        )

    @require_torch
    def test_accepts_torch_device(self):
        import torch

        text_classifier = pipeline(
            task="text-classification",
            model="hf-internal-testing/tiny-random-distilbert",
            framework="pt",
            device=torch.device("cpu"),
        )

        outputs = text_classifier("This is great !")
        self.assertEqual(nested_simplify(outputs), [{"label": "LABEL_0", "score": 0.504}])

    @require_tf
    def test_small_model_tf(self):
        text_classifier = pipeline(
            task="text-classification", model="hf-internal-testing/tiny-random-distilbert", framework="tf"
        )

        outputs = text_classifier("This is great !")
        self.assertEqual(nested_simplify(outputs), [{"label": "LABEL_0", "score": 0.504}])

    @slow
    @require_torch
    def test_pt_bert(self):
        text_classifier = pipeline("text-classification")

        outputs = text_classifier("This is great !")
        self.assertEqual(nested_simplify(outputs), [{"label": "POSITIVE", "score": 1.0}])
        outputs = text_classifier("This is bad !")
        self.assertEqual(nested_simplify(outputs), [{"label": "NEGATIVE", "score": 1.0}])
        outputs = text_classifier("Birds are a type of animal")
        self.assertEqual(nested_simplify(outputs), [{"label": "POSITIVE", "score": 0.988}])

    @slow
    @require_tf
    def test_tf_bert(self):
        text_classifier = pipeline("text-classification", framework="tf")

        outputs = text_classifier("This is great !")
        self.assertEqual(nested_simplify(outputs), [{"label": "POSITIVE", "score": 1.0}])
        outputs = text_classifier("This is bad !")
        self.assertEqual(nested_simplify(outputs), [{"label": "NEGATIVE", "score": 1.0}])
        outputs = text_classifier("Birds are a type of animal")
        self.assertEqual(nested_simplify(outputs), [{"label": "POSITIVE", "score": 0.988}])

    def get_test_pipeline(self, model, tokenizer, feature_extractor):
        text_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)
        return text_classifier, ["HuggingFace is in", "This is another test"]

    def run_pipeline_test(self, text_classifier, _):
        model = text_classifier.model
        # Small inputs because BartTokenizer tiny has maximum position embeddings = 22
        valid_inputs = "HuggingFace is in"
        outputs = text_classifier(valid_inputs)

        self.assertEqual(nested_simplify(outputs), [{"label": ANY(str), "score": ANY(float)}])
        self.assertTrue(outputs[0]["label"] in model.config.id2label.values())

        valid_inputs = ["HuggingFace is in ", "Paris is in France"]
        outputs = text_classifier(valid_inputs)
        self.assertEqual(
            nested_simplify(outputs),
            [{"label": ANY(str), "score": ANY(float)}, {"label": ANY(str), "score": ANY(float)}],
        )
        self.assertTrue(outputs[0]["label"] in model.config.id2label.values())
        self.assertTrue(outputs[1]["label"] in model.config.id2label.values())

        # Forcing to get all results with `top_k=None`
        # This is NOT the legacy format
        outputs = text_classifier(valid_inputs, top_k=None)
        N = len(model.config.id2label.values())
        self.assertEqual(
            nested_simplify(outputs),
            [[{"label": ANY(str), "score": ANY(float)}] * N, [{"label": ANY(str), "score": ANY(float)}] * N],
        )

        valid_inputs = {"text": "HuggingFace is in ", "text_pair": "Paris is in France"}
        outputs = text_classifier(valid_inputs)
        self.assertEqual(
            nested_simplify(outputs),
            {"label": ANY(str), "score": ANY(float)},
        )
        self.assertTrue(outputs["label"] in model.config.id2label.values())

        # This might be used a text pair, but tokenizer + pipe interaction
        # makes it hard to understand that it's not using the pair properly
        # https://github.com/huggingface/transformers/issues/17305
        # We disabled this usage instead as it was outputting wrong outputs.
        invalid_input = [["HuggingFace is in ", "Paris is in France"]]
        with self.assertRaises(ValueError):
            text_classifier(invalid_input)

        # This used to be valid for doing text pairs
        # We're keeping it working because of backward compatibility
        outputs = text_classifier([[["HuggingFace is in ", "Paris is in France"]]])
        self.assertEqual(
            nested_simplify(outputs),
            [{"label": ANY(str), "score": ANY(float)}],
        )
        self.assertTrue(outputs[0]["label"] in model.config.id2label.values())
