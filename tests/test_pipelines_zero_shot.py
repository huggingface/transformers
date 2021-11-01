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
    Pipeline,
    ZeroShotClassificationPipeline,
    pipeline,
)
from transformers.testing_utils import is_pipeline_test, nested_simplify, require_tf, require_torch, slow

from .test_pipelines_common import ANY, PipelineTestCaseMeta


@is_pipeline_test
class ZeroShotClassificationPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    tf_model_mapping = TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

    def get_test_pipeline(self, model, tokenizer, feature_extractor):
        classifier = ZeroShotClassificationPipeline(
            model=model, tokenizer=tokenizer, candidate_labels=["polics", "health"]
        )
        return classifier, ["Who are you voting for in 2020?", "My stomach hurts."]

    def run_pipeline_test(self, classifier, _):
        outputs = classifier("Who are you voting for in 2020?", candidate_labels="politics")
        self.assertEqual(outputs, {"sequence": ANY(str), "labels": [ANY(str)], "scores": [ANY(float)]})

        # No kwarg
        outputs = classifier("Who are you voting for in 2020?", ["politics"])
        self.assertEqual(outputs, {"sequence": ANY(str), "labels": [ANY(str)], "scores": [ANY(float)]})

        outputs = classifier("Who are you voting for in 2020?", candidate_labels=["politics"])
        self.assertEqual(outputs, {"sequence": ANY(str), "labels": [ANY(str)], "scores": [ANY(float)]})

        outputs = classifier("Who are you voting for in 2020?", candidate_labels="politics, public health")
        self.assertEqual(
            outputs, {"sequence": ANY(str), "labels": [ANY(str), ANY(str)], "scores": [ANY(float), ANY(float)]}
        )
        self.assertAlmostEqual(sum(nested_simplify(outputs["scores"])), 1.0)

        outputs = classifier("Who are you voting for in 2020?", candidate_labels=["politics", "public health"])
        self.assertEqual(
            outputs, {"sequence": ANY(str), "labels": [ANY(str), ANY(str)], "scores": [ANY(float), ANY(float)]}
        )
        self.assertAlmostEqual(sum(nested_simplify(outputs["scores"])), 1.0)

        outputs = classifier(
            "Who are you voting for in 2020?", candidate_labels="politics", hypothesis_template="This text is about {}"
        )
        self.assertEqual(outputs, {"sequence": ANY(str), "labels": [ANY(str)], "scores": [ANY(float)]})

        # https://github.com/huggingface/transformers/issues/13846
        outputs = classifier(["I am happy"], ["positive", "negative"])
        self.assertEqual(
            outputs,
            [
                {"sequence": ANY(str), "labels": [ANY(str), ANY(str)], "scores": [ANY(float), ANY(float)]}
                for i in range(1)
            ],
        )
        outputs = classifier(["I am happy", "I am sad"], ["positive", "negative"])
        self.assertEqual(
            outputs,
            [
                {"sequence": ANY(str), "labels": [ANY(str), ANY(str)], "scores": [ANY(float), ANY(float)]}
                for i in range(2)
            ],
        )

        with self.assertRaises(ValueError):
            classifier("", candidate_labels="politics")

        with self.assertRaises(TypeError):
            classifier(None, candidate_labels="politics")

        with self.assertRaises(ValueError):
            classifier("Who are you voting for in 2020?", candidate_labels="")

        with self.assertRaises(TypeError):
            classifier("Who are you voting for in 2020?", candidate_labels=None)

        with self.assertRaises(ValueError):
            classifier(
                "Who are you voting for in 2020?",
                candidate_labels="politics",
                hypothesis_template="Not formatting template",
            )

        with self.assertRaises(AttributeError):
            classifier(
                "Who are you voting for in 2020?",
                candidate_labels="politics",
                hypothesis_template=None,
            )

        self.run_entailment_id(classifier)

    def run_entailment_id(self, zero_shot_classifier: Pipeline):
        config = zero_shot_classifier.model.config
        original_label2id = config.label2id
        original_entailment = zero_shot_classifier.entailment_id

        config.label2id = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
        self.assertEqual(zero_shot_classifier.entailment_id, -1)

        config.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
        self.assertEqual(zero_shot_classifier.entailment_id, 0)

        config.label2id = {"ENTAIL": 0, "NON-ENTAIL": 1}
        self.assertEqual(zero_shot_classifier.entailment_id, 0)

        config.label2id = {"ENTAIL": 2, "NEUTRAL": 1, "CONTR": 0}
        self.assertEqual(zero_shot_classifier.entailment_id, 2)

        zero_shot_classifier.model.config.label2id = original_label2id
        self.assertEqual(original_entailment, zero_shot_classifier.entailment_id)

    @require_torch
    def test_truncation(self):
        zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="sshleifer/tiny-distilbert-base-cased-distilled-squad",
            framework="pt",
        )
        # There was a regression in 4.10 for this
        # Adding a test so we don't make the mistake again.
        # https://github.com/huggingface/transformers/issues/13381#issuecomment-912343499
        zero_shot_classifier(
            "Who are you voting for in 2020?" * 100, candidate_labels=["politics", "public health", "science"]
        )

    @require_torch
    def test_small_model_pt(self):
        zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="sshleifer/tiny-distilbert-base-cased-distilled-squad",
            framework="pt",
        )
        outputs = zero_shot_classifier(
            "Who are you voting for in 2020?", candidate_labels=["politics", "public health", "science"]
        )

        self.assertEqual(
            nested_simplify(outputs),
            {
                "sequence": "Who are you voting for in 2020?",
                "labels": ["science", "public health", "politics"],
                "scores": [0.333, 0.333, 0.333],
            },
        )

    @require_tf
    def test_small_model_tf(self):
        zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="sshleifer/tiny-distilbert-base-cased-distilled-squad",
            framework="tf",
        )
        outputs = zero_shot_classifier(
            "Who are you voting for in 2020?", candidate_labels=["politics", "public health", "science"]
        )

        self.assertEqual(
            nested_simplify(outputs),
            {
                "sequence": "Who are you voting for in 2020?",
                "labels": ["science", "public health", "politics"],
                "scores": [0.333, 0.333, 0.333],
            },
        )

    @slow
    @require_torch
    def test_large_model_pt(self):
        zero_shot_classifier = pipeline("zero-shot-classification", model="roberta-large-mnli", framework="pt")
        outputs = zero_shot_classifier(
            "Who are you voting for in 2020?", candidate_labels=["politics", "public health", "science"]
        )

        self.assertEqual(
            nested_simplify(outputs),
            {
                "sequence": "Who are you voting for in 2020?",
                "labels": ["politics", "public health", "science"],
                "scores": [0.976, 0.015, 0.009],
            },
        )
        outputs = zero_shot_classifier(
            "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
            candidate_labels=["machine learning", "statistics", "translation", "vision"],
            multi_label=True,
        )
        self.assertEqual(
            nested_simplify(outputs),
            {
                "sequence": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
                "labels": ["translation", "machine learning", "vision", "statistics"],
                "scores": [0.817, 0.713, 0.018, 0.018],
            },
        )

    @slow
    @require_tf
    def test_large_model_tf(self):
        zero_shot_classifier = pipeline("zero-shot-classification", model="roberta-large-mnli", framework="tf")
        outputs = zero_shot_classifier(
            "Who are you voting for in 2020?", candidate_labels=["politics", "public health", "science"]
        )

        self.assertEqual(
            nested_simplify(outputs),
            {
                "sequence": "Who are you voting for in 2020?",
                "labels": ["politics", "public health", "science"],
                "scores": [0.976, 0.015, 0.009],
            },
        )
        outputs = zero_shot_classifier(
            "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
            candidate_labels=["machine learning", "statistics", "translation", "vision"],
            multi_label=True,
        )
        self.assertEqual(
            nested_simplify(outputs),
            {
                "sequence": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
                "labels": ["translation", "machine learning", "vision", "statistics"],
                "scores": [0.817, 0.713, 0.018, 0.018],
            },
        )
