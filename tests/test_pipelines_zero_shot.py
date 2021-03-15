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
from copy import deepcopy

from transformers.pipelines import Pipeline

from .test_pipelines_common import CustomInputPipelineCommonMixin


class ZeroShotClassificationPipelineTests(CustomInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "zero-shot-classification"
    small_models = [
        "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"
    ]  # Models tested without the @slow decorator
    large_models = ["roberta-large-mnli"]  # Models tested with the @slow decorator
    valid_inputs = [
        {"sequences": "Who are you voting for in 2020?", "candidate_labels": "politics"},
        {"sequences": "Who are you voting for in 2020?", "candidate_labels": ["politics"]},
        {"sequences": "Who are you voting for in 2020?", "candidate_labels": "politics, public health"},
        {"sequences": "Who are you voting for in 2020?", "candidate_labels": ["politics", "public health"]},
        {"sequences": ["Who are you voting for in 2020?"], "candidate_labels": "politics"},
        {
            "sequences": "Who are you voting for in 2020?",
            "candidate_labels": "politics",
            "hypothesis_template": "This text is about {}",
        },
    ]

    def _test_scores_sum_to_one(self, result):
        sum = 0.0
        for score in result["scores"]:
            sum += score
        self.assertAlmostEqual(sum, 1.0, places=5)

    def _test_entailment_id(self, nlp: Pipeline):
        config = nlp.model.config
        original_config = deepcopy(config)

        config.label2id = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}
        self.assertEqual(nlp.entailment_id, -1)

        config.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
        self.assertEqual(nlp.entailment_id, 0)

        config.label2id = {"ENTAIL": 0, "NON-ENTAIL": 1}
        self.assertEqual(nlp.entailment_id, 0)

        config.label2id = {"ENTAIL": 2, "NEUTRAL": 1, "CONTR": 0}
        self.assertEqual(nlp.entailment_id, 2)

        nlp.model.config = original_config

    def _test_pipeline(self, nlp: Pipeline):
        output_keys = {"sequence", "labels", "scores"}
        valid_mono_inputs = [
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": "politics"},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": ["politics"]},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": "politics, public health"},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": ["politics", "public health"]},
            {"sequences": ["Who are you voting for in 2020?"], "candidate_labels": "politics"},
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": "politics",
                "hypothesis_template": "This text is about {}",
            },
        ]
        valid_multi_input = {
            "sequences": ["Who are you voting for in 2020?", "What is the capital of Spain?"],
            "candidate_labels": "politics",
        }
        invalid_inputs = [
            {"sequences": None, "candidate_labels": "politics"},
            {"sequences": "", "candidate_labels": "politics"},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": None},
            {"sequences": "Who are you voting for in 2020?", "candidate_labels": ""},
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": "politics",
                "hypothesis_template": None,
            },
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": "politics",
                "hypothesis_template": "",
            },
            {
                "sequences": "Who are you voting for in 2020?",
                "candidate_labels": "politics",
                "hypothesis_template": "Template without formatting syntax.",
            },
        ]
        self.assertIsNotNone(nlp)

        self._test_entailment_id(nlp)

        for mono_input in valid_mono_inputs:
            mono_result = nlp(**mono_input)
            self.assertIsInstance(mono_result, dict)
            if len(mono_result["labels"]) > 1:
                self._test_scores_sum_to_one(mono_result)

            for key in output_keys:
                self.assertIn(key, mono_result)

        multi_result = nlp(**valid_multi_input)
        self.assertIsInstance(multi_result, list)
        self.assertIsInstance(multi_result[0], dict)
        self.assertEqual(len(multi_result), len(valid_multi_input["sequences"]))

        for result in multi_result:
            for key in output_keys:
                self.assertIn(key, result)

            if len(result["labels"]) > 1:
                self._test_scores_sum_to_one(result)

        for bad_input in invalid_inputs:
            self.assertRaises(Exception, nlp, **bad_input)

        if nlp.model.name_or_path in self.large_models:
            # We also check the outputs for the large models
            inputs = [
                {
                    "sequences": "Who are you voting for in 2020?",
                    "candidate_labels": ["politics", "public health", "science"],
                },
                {
                    "sequences": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
                    "candidate_labels": ["machine learning", "statistics", "translation", "vision"],
                    "multi_label": True,
                },
            ]

            expected_outputs = [
                {
                    "sequence": "Who are you voting for in 2020?",
                    "labels": ["politics", "public health", "science"],
                    "scores": [0.975, 0.015, 0.008],
                },
                {
                    "sequence": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
                    "labels": ["translation", "machine learning", "vision", "statistics"],
                    "scores": [0.817, 0.712, 0.018, 0.017],
                },
            ]

            for input, expected_output in zip(inputs, expected_outputs):
                output = nlp(**input)
                for key in output:
                    if key == "scores":
                        for output_score, expected_score in zip(output[key], expected_output[key]):
                            self.assertAlmostEqual(output_score, expected_score, places=2)
                    else:
                        self.assertEqual(output[key], expected_output[key])
