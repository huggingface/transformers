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

from transformers import MODEL_MAPPING, TF_MODEL_MAPPING, FeatureExtractionPipeline, LxmertConfig, pipeline
from transformers.testing_utils import is_pipeline_test, nested_simplify, require_torch  # require_tf

from .test_pipelines_common import PipelineTestCaseMeta


@is_pipeline_test
class FeatureExtractionPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_MAPPING
    tf_model_mapping = TF_MODEL_MAPPING

    @require_torch
    def test_small_model_pt(self):
        feature_extractor = pipeline(
            task="feature-extraction", model="sshleifer/tiny-distilbert-base-cased", framework="pt"
        )
        outputs = feature_extractor("This is a test")
        self.assertEqual(
            nested_simplify(outputs), [[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0]]]
        )

    # XXX: This is flaky !
    # @require_tf
    # def test_small_model_tf(self):
    #     feature_extractor = pipeline(
    #         task="feature-extraction", model="sshleifer/tiny-distilbert-base-cased", framework="tf"
    #     )
    #     outputs = feature_extractor("This is a test")
    #     print(nested_simplify(outputs))
    #     # [[[1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0]]]
    #     # [[[1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0]]]
    #     # [[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]]
    #     self.assertEqual(
    #         nested_simplify(outputs), [[[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0]]]
    #     )
    def get_shape(self, input_, shape=None):
        if shape is None:
            shape = []
        if isinstance(input_, list):
            subshapes = [self.get_shape(in_, shape) for in_ in input_]
            if all(s == 0 for s in subshapes):
                shape.append(len(input_))
            else:
                subshape = subshapes[0]
                shape = [len(input_), *subshape]
        elif isinstance(input_, float):
            return 0
        else:
            raise Exception("We expect lists of floats, nothing else")
        return shape

    def run_pipeline_test(self, model, tokenizer):
        if isinstance(model.config, LxmertConfig):
            # This is an bimodal model, we need to find a more consistent way
            # to switch on those models.
            return

        feature_extractor = FeatureExtractionPipeline(model=model, tokenizer=tokenizer)
        if feature_extractor.model.config.is_encoder_decoder:
            # encoder_decoder models are trickier for this pipeline.
            # Do we want encoder + decoder inputs to get some featues?
            # Do we want encoder only features ?
            # For now ignore those.
            return

        outputs = feature_extractor("This is a test")
        # Output shape is NxTxE where
        # N = number of sequences passed
        # T = number of tokens in sequence (depends on the tokenizer)
        # E = number of embedding dimensions

        shape = self.get_shape(outputs)
        self.assertEqual(shape[0], 1)
        # self.assertIsInstance(outputs[0], list)
        # self.assertIsInstance(outputs[0][0], list)
        # self.assertIsInstance(outputs[0][0][0], float)
        # self.assertTrue(all(isinstance(el, float) for row in outputs for col in row for el in col))

        outputs = feature_extractor(["This is a test", "Another test"])
        shape = self.get_shape(outputs)
        self.assertEqual(shape[0], 2)
        # self.assertIsInstance(outputs, list)
        # self.assertEqual(len(outputs), 2)
        # self.assertIsInstance(outputs[0], list)
        # self.assertIsInstance(outputs[0][0], list)
        # self.assertIsInstance(outputs[0][0][0], float)
        # self.assertTrue(all(isinstance(el, float) for row in outputs for col in row for el in col))
