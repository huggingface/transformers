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

from transformers import MODEL_MAPPING, TF_MODEL_MAPPING, CLIPConfig, FeatureExtractionPipeline, LxmertConfig, pipeline
from transformers.testing_utils import is_pipeline_test, nested_simplify, require_tf, require_torch

from .test_pipelines_common import PipelineTestCaseMeta


@is_pipeline_test
class FeatureExtractionPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_MAPPING
    tf_model_mapping = TF_MODEL_MAPPING

    @require_torch
    def test_small_model_pt(self):
        feature_extractor = pipeline(
            task="feature-extraction", model="hf-internal-testing/tiny-random-distilbert", framework="pt"
        )
        outputs = feature_extractor("This is a test")
        self.assertEqual(
            nested_simplify(outputs),
            [[[-0.454, 0.966, 0.619, 0.262, 0.669, -0.661, -0.066, -0.513, -0.768, -0.177, 1.771, -0.665, -0.649, 0.219, 0.236, -0.375, 1.155, -1.07, 0.208, -0.799, 1.065, -1.223, 0.554, 1.274, 0.458, 2.292, -0.481, -0.928, -2.469, -1.692, 0.182, 1.06], [-0.187, -1.277, 0.849, -0.439, -0.967, -1.347, 1.063, 0.469, 1.086, -1.253, 0.349, 0.057, 1.031, -1.903, -0.432, -1.377, 0.379, 0.733, -1.043, 1.307, 0.865, 0.229, 1.373, 1.671, -0.285, 0.599, -1.418, -1.179, -0.369, 1.039, -0.705, 1.082], [-1.735, 1.102, 0.398, -0.245, 1.452, 0.46, -1.734, -0.746, 1.831, 0.562, 1.464, -0.342, -0.619, -0.455, 0.127, -1.209, -0.686, -0.395, -0.316, 2.467, -0.379, 0.328, 0.639, 0.4, -1.097, -0.096, 0.397, -0.806, -1.621, 1.127, -0.345, 0.074], [0.296, -0.638, 1.938, -0.151, -1.19, 1.445, 1.318, 0.711, -0.125, 0.127, -2.179, 0.481, -1.019, 1.178, 0.318, 1.858, -1.646, 0.185, -0.072, -0.979, 0.82, -1.374, 0.836, -1.019, 0.043, -0.156, -0.095, 0.641, -0.195, -0.076, -1.554, 0.275], [-0.266, 0.971, 0.745, -0.37, 1.42, -0.5, -0.53, 0.061, 1.311, -0.1, 1.796, 0.53, -0.739, -0.325, 0.28, -1.72, 0.382, -1.118, 0.442, 1.84, -2.497, 1.003, -0.788, -0.224, -0.604, -1.259, -0.475, 1.18, -1.356, 0.695, 0.201, 0.016], [-0.618, -1.495, -0.67, -0.106, -1.265, -0.51, -1.752, 1.018, 0.674, 0.181, 0.297, 0.479, -0.185, 0.081, -2.44, -0.239, 1.081, -1.38, 0.679, 0.878, 1.336, -1.347, 0.969, -0.847, 0.293, 0.476, 1.647, -0.641, 0.66, 1.236, 0.761, 0.751]]])  # fmt: skip

    @require_tf
    def test_small_model_tf(self):
        feature_extractor = pipeline(
            task="feature-extraction", model="hf-internal-testing/tiny-random-distilbert", framework="tf"
        )
        outputs = feature_extractor("This is a test")
        self.assertEqual(
            nested_simplify(outputs),
            [[[-0.454, 0.966, 0.619, 0.262, 0.669, -0.661, -0.066, -0.513, -0.768, -0.177, 1.771, -0.665, -0.649, 0.219, 0.236, -0.375, 1.155, -1.07, 0.208, -0.799, 1.065, -1.223, 0.554, 1.274, 0.458, 2.292, -0.481, -0.928, -2.469, -1.692, 0.182, 1.06], [-0.187, -1.277, 0.849, -0.439, -0.967, -1.347, 1.063, 0.469, 1.086, -1.253, 0.349, 0.057, 1.031, -1.903, -0.432, -1.377, 0.379, 0.733, -1.043, 1.307, 0.865, 0.229, 1.373, 1.671, -0.285, 0.599, -1.418, -1.179, -0.369, 1.039, -0.705, 1.082], [-1.735, 1.102, 0.398, -0.245, 1.452, 0.46, -1.734, -0.746, 1.831, 0.562, 1.464, -0.342, -0.619, -0.455, 0.127, -1.209, -0.686, -0.395, -0.316, 2.467, -0.379, 0.328, 0.639, 0.4, -1.097, -0.096, 0.397, -0.806, -1.621, 1.127, -0.345, 0.074], [0.296, -0.638, 1.938, -0.151, -1.19, 1.445, 1.318, 0.711, -0.125, 0.127, -2.179, 0.481, -1.019, 1.178, 0.318, 1.858, -1.646, 0.185, -0.072, -0.979, 0.82, -1.374, 0.836, -1.019, 0.043, -0.156, -0.095, 0.641, -0.195, -0.076, -1.554, 0.275], [-0.266, 0.971, 0.745, -0.37, 1.42, -0.5, -0.53, 0.061, 1.311, -0.1, 1.796, 0.53, -0.739, -0.325, 0.28, -1.72, 0.382, -1.118, 0.442, 1.84, -2.497, 1.003, -0.788, -0.224, -0.604, -1.259, -0.475, 1.18, -1.356, 0.695, 0.201, 0.016], [-0.618, -1.495, -0.67, -0.106, -1.265, -0.51, -1.752, 1.018, 0.674, 0.181, 0.297, 0.479, -0.185, 0.081, -2.44, -0.239, 1.081, -1.38, 0.679, 0.878, 1.336, -1.347, 0.969, -0.847, 0.293, 0.476, 1.647, -0.641, 0.66, 1.236, 0.761, 0.751]]])  # fmt: skip

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
            raise ValueError("We expect lists of floats, nothing else")
        return shape

    def run_pipeline_test(self, model, tokenizer, feature_extractor):
        if tokenizer is None:
            self.skipTest("No tokenizer")
            return

        elif isinstance(model.config, (LxmertConfig, CLIPConfig)):
            self.skipTest(
                "This is an Lxmert bimodal model, we need to find a more consistent way to switch on those models."
            )
            return
        elif model.config.is_encoder_decoder:
            self.skipTest(
                """encoder_decoder models are trickier for this pipeline.
                Do we want encoder + decoder inputs to get some featues?
                Do we want encoder only features ?
                For now ignore those.
                """
            )

            return

        feature_extractor = FeatureExtractionPipeline(
            model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
        )

        outputs = feature_extractor("This is a test")

        shape = self.get_shape(outputs)
        self.assertEqual(shape[0], 1)

        outputs = feature_extractor(["This is a test", "Another test"])
        shape = self.get_shape(outputs)
        self.assertEqual(shape[0], 2)
