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

from transformers import MODEL_MAPPING, TF_MODEL_MAPPING, FeatureExtractionPipeline, pipeline
from transformers.testing_utils import is_pipeline_test, nested_simplify

from .test_pipelines_common import PipelineTestCaseMeta


@is_pipeline_test
class FeatureExtractionPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_MAPPING
    tf_model_mapping = TF_MODEL_MAPPING

    def test_small_model(self):
        feature_extractor = pipeline(task="feature-extraction", model="sshleifer/tiny-distilbert-base-cased")
        outputs = feature_extractor("This is a test")
        self.assertEqual(
            nested_simplify(outputs), [[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0]]]
        )

    def run_pipeline_test(self, model, tokenizer):
        feature_extractor = FeatureExtractionPipeline(model=model, tokenizer=tokenizer)

        outputs = feature_extractor("This is a test")
        # Output shape is NxTxE where
        # N = number of sequences passed
        # T = number of tokens in sequence (depends on the tokenizer)
        # E = number of embedding dimensions
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(outputs[0], list)
        self.assertIsInstance(outputs[0][0], list)
        self.assertIsInstance(outputs[0][0][0], float)
        self.assertTrue(all(isinstance(el, float) for row in outputs for col in row for el in col))

        outputs = feature_extractor(["This is a test", "Another test"])
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 2)
        self.assertIsInstance(outputs[0], list)
        self.assertIsInstance(outputs[0][0], list)
        self.assertIsInstance(outputs[0][0][0], float)
        self.assertTrue(all(isinstance(el, float) for row in outputs for col in row for el in col))
