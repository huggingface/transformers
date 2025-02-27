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
    FEATURE_EXTRACTOR_MAPPING,
    IMAGE_PROCESSOR_MAPPING,
    MODEL_MAPPING,
    TF_MODEL_MAPPING,
    FeatureExtractionPipeline,
    LxmertConfig,
    is_tf_available,
    is_torch_available,
    pipeline,
)
from transformers.testing_utils import is_pipeline_test, nested_simplify, require_tf, require_torch


if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf


@is_pipeline_test
class FeatureExtractionPipelineTests(unittest.TestCase):
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
            [[[2.287, 1.234, 0.042, 1.53, 1.306, 0.879, -0.526, -1.71, -1.276, 0.756, -0.775, -1.048, -0.25, -0.595, -0.137, -0.598, 2.022, -0.812, 0.284, -0.488, -0.391, -0.403, -0.525, -0.061, -0.228, 1.086, 0.378, -0.14, 0.599, -0.087, -2.259, -0.098], [1.676, 0.232, -1.508, -0.145, 1.798, -1.388, 1.331, -0.37, -0.939, 0.043, 0.06, -0.414, -1.408, 0.24, 0.622, -0.55, -0.569, 1.873, -0.706, 1.924, -0.254, 1.927, -0.423, 0.152, -0.952, 0.509, -0.496, -0.968, 0.093, -1.049, -0.65, 0.312], [0.207, -0.775, -1.822, 0.321, -0.71, -0.201, 0.3, 1.146, -0.233, -0.753, -0.305, 1.309, -1.47, -0.21, 1.802, -1.555, -1.175, 1.323, -0.303, 0.722, -0.076, 0.103, -1.406, 1.931, 0.091, 0.237, 1.172, 1.607, 0.253, -0.9, -1.068, 0.438], [0.615, 1.077, 0.171, -0.175, 1.3, 0.901, -0.653, -0.138, 0.341, -0.654, -0.184, -0.441, -0.424, 0.356, -0.075, 0.26, -1.023, 0.814, 0.524, -0.904, -0.204, -0.623, 1.234, -1.03, 2.594, 0.56, 1.831, -0.199, -1.508, -0.492, -1.687, -2.165], [0.129, 0.008, -1.279, -0.412, -0.004, 1.663, 0.196, 0.104, 0.123, 0.119, 0.635, 1.757, 2.334, -0.799, -1.626, -1.26, 0.595, -0.316, -1.399, 0.232, 0.264, 1.386, -1.171, -0.256, -0.256, -1.944, 1.168, -0.368, -0.714, -0.51, 0.454, 1.148], [-0.32, 0.29, -1.309, -0.177, 0.453, 0.636, -0.024, 0.509, 0.931, -1.754, -1.575, 0.786, 0.046, -1.165, -1.416, 1.373, 1.293, -0.285, -1.541, -1.186, -0.106, -0.994, 2.001, 0.972, -0.02, 1.654, -0.236, 0.643, 1.02, 0.572, -0.914, -0.154], [0.7, -0.937, 0.441, 0.25, 0.78, -0.022, 0.282, -0.095, 1.558, -0.336, 1.706, 0.884, 1.28, 0.198, -0.796, 1.218, -1.769, 1.197, -0.342, -0.177, -0.645, 1.364, 0.008, -0.597, -0.484, -2.772, -0.696, -0.632, -0.34, -1.527, -0.562, 0.862], [2.504, 0.831, -1.271, -0.033, 0.298, -0.735, 1.339, 1.74, 0.233, -1.424, -0.819, -0.761, 0.291, 0.853, -0.092, -0.885, 0.164, 1.025, 0.907, 0.749, -1.515, -0.545, -1.365, 0.271, 0.034, -2.005, 0.031, 0.244, 0.621, 0.176, 0.336, -1.196], [-0.711, 0.591, -1.001, -0.946, 0.784, -1.66, 1.545, 0.799, -0.857, 1.148, 0.213, -0.285, 0.464, -0.139, 0.79, -1.663, -1.121, 0.575, -0.178, -0.508, 1.565, -0.242, -0.346, 1.024, -1.135, -0.158, -2.101, 0.275, 2.009, -0.425, 0.716, 0.981], [0.912, -1.186, -0.846, -0.421, -1.315, -0.827, 0.309, 0.533, 1.029, -2.343, 1.513, -1.238, 1.487, -0.849, 0.896, -0.927, -0.459, 0.159, 0.177, 0.873, 0.935, 1.433, -0.485, 0.737, 1.327, -0.338, 1.608, -0.47, -0.445, -1.118, -0.213, -0.446], [-0.434, -1.362, -1.098, -1.068, 1.507, 0.003, 0.413, -0.395, 0.897, -0.237, 1.405, -0.344, 1.693, 0.677, 0.097, -0.257, -0.602, 1.026, -1.229, 0.855, -0.713, 1.014, 0.443, 0.238, 0.425, -2.184, 1.933, -1.157, -1.132, -0.597, -0.785, 0.967], [0.58, -0.971, 0.789, -0.468, -0.576, 1.779, 1.747, 1.715, -1.939, 0.125, 0.656, -0.042, -1.024, -1.767, 0.107, -0.408, -0.866, -1.774, 1.248, 0.939, -0.033, 1.523, 1.168, -0.744, 0.209, -0.168, -0.316, 0.207, -0.432, 0.047, -0.646, -0.664], [-0.185, -0.613, -1.695, 1.602, -0.32, -0.277, 0.967, 0.728, -0.965, -0.234, 1.069, -0.63, -1.631, 0.711, 0.426, 1.298, -0.191, -0.467, -0.771, 0.971, -0.118, -1.577, -2.064, -0.055, -0.59, 0.642, -0.997, 1.251, 0.538, 1.367, 0.106, 1.704]]])  # fmt: skip

    @require_tf
    def test_small_model_tf(self):
        feature_extractor = pipeline(
            task="feature-extraction", model="hf-internal-testing/tiny-random-distilbert", framework="tf"
        )
        outputs = feature_extractor("This is a test")
        self.assertEqual(
            nested_simplify(outputs),
            [[[2.287, 1.234, 0.042, 1.53, 1.306, 0.879, -0.526, -1.71, -1.276, 0.756, -0.775, -1.048, -0.25, -0.595, -0.137, -0.598, 2.022, -0.812, 0.284, -0.488, -0.391, -0.403, -0.525, -0.061, -0.228, 1.086, 0.378, -0.14, 0.599, -0.087, -2.259, -0.098], [1.676, 0.232, -1.508, -0.145, 1.798, -1.388, 1.331, -0.37, -0.939, 0.043, 0.06, -0.414, -1.408, 0.24, 0.622, -0.55, -0.569, 1.873, -0.706, 1.924, -0.254, 1.927, -0.423, 0.152, -0.952, 0.509, -0.496, -0.968, 0.093, -1.049, -0.65, 0.312], [0.207, -0.775, -1.822, 0.321, -0.71, -0.201, 0.3, 1.146, -0.233, -0.753, -0.305, 1.309, -1.47, -0.21, 1.802, -1.555, -1.175, 1.323, -0.303, 0.722, -0.076, 0.103, -1.406, 1.931, 0.091, 0.237, 1.172, 1.607, 0.253, -0.9, -1.068, 0.438], [0.615, 1.077, 0.171, -0.175, 1.3, 0.901, -0.653, -0.138, 0.341, -0.654, -0.184, -0.441, -0.424, 0.356, -0.075, 0.26, -1.023, 0.814, 0.524, -0.904, -0.204, -0.623, 1.234, -1.03, 2.594, 0.56, 1.831, -0.199, -1.508, -0.492, -1.687, -2.165], [0.129, 0.008, -1.279, -0.412, -0.004, 1.663, 0.196, 0.104, 0.123, 0.119, 0.635, 1.757, 2.334, -0.799, -1.626, -1.26, 0.595, -0.316, -1.399, 0.232, 0.264, 1.386, -1.171, -0.256, -0.256, -1.944, 1.168, -0.368, -0.714, -0.51, 0.454, 1.148], [-0.32, 0.29, -1.309, -0.177, 0.453, 0.636, -0.024, 0.509, 0.931, -1.754, -1.575, 0.786, 0.046, -1.165, -1.416, 1.373, 1.293, -0.285, -1.541, -1.186, -0.106, -0.994, 2.001, 0.972, -0.02, 1.654, -0.236, 0.643, 1.02, 0.572, -0.914, -0.154], [0.7, -0.937, 0.441, 0.25, 0.78, -0.022, 0.282, -0.095, 1.558, -0.336, 1.706, 0.884, 1.28, 0.198, -0.796, 1.218, -1.769, 1.197, -0.342, -0.177, -0.645, 1.364, 0.008, -0.597, -0.484, -2.772, -0.696, -0.632, -0.34, -1.527, -0.562, 0.862], [2.504, 0.831, -1.271, -0.033, 0.298, -0.735, 1.339, 1.74, 0.233, -1.424, -0.819, -0.761, 0.291, 0.853, -0.092, -0.885, 0.164, 1.025, 0.907, 0.749, -1.515, -0.545, -1.365, 0.271, 0.034, -2.005, 0.031, 0.244, 0.621, 0.176, 0.336, -1.196], [-0.711, 0.591, -1.001, -0.946, 0.784, -1.66, 1.545, 0.799, -0.857, 1.148, 0.213, -0.285, 0.464, -0.139, 0.79, -1.663, -1.121, 0.575, -0.178, -0.508, 1.565, -0.242, -0.346, 1.024, -1.135, -0.158, -2.101, 0.275, 2.009, -0.425, 0.716, 0.981], [0.912, -1.186, -0.846, -0.421, -1.315, -0.827, 0.309, 0.533, 1.029, -2.343, 1.513, -1.238, 1.487, -0.849, 0.896, -0.927, -0.459, 0.159, 0.177, 0.873, 0.935, 1.433, -0.485, 0.737, 1.327, -0.338, 1.608, -0.47, -0.445, -1.118, -0.213, -0.446], [-0.434, -1.362, -1.098, -1.068, 1.507, 0.003, 0.413, -0.395, 0.897, -0.237, 1.405, -0.344, 1.693, 0.677, 0.097, -0.257, -0.602, 1.026, -1.229, 0.855, -0.713, 1.014, 0.443, 0.238, 0.425, -2.184, 1.933, -1.157, -1.132, -0.597, -0.785, 0.967], [0.58, -0.971, 0.789, -0.468, -0.576, 1.779, 1.747, 1.715, -1.939, 0.125, 0.656, -0.042, -1.024, -1.767, 0.107, -0.408, -0.866, -1.774, 1.248, 0.939, -0.033, 1.523, 1.168, -0.744, 0.209, -0.168, -0.316, 0.207, -0.432, 0.047, -0.646, -0.664], [-0.185, -0.613, -1.695, 1.602, -0.32, -0.277, 0.967, 0.728, -0.965, -0.234, 1.069, -0.63, -1.631, 0.711, 0.426, 1.298, -0.191, -0.467, -0.771, 0.971, -0.118, -1.577, -2.064, -0.055, -0.59, 0.642, -0.997, 1.251, 0.538, 1.367, 0.106, 1.704]]])  # fmt: skip

    @require_torch
    def test_tokenization_small_model_pt(self):
        feature_extractor = pipeline(
            task="feature-extraction", model="hf-internal-testing/tiny-random-distilbert", framework="pt"
        )
        # test with empty parameters
        outputs = feature_extractor("This is a test")
        self.assertEqual(
            nested_simplify(outputs),
            [[[2.287, 1.234, 0.042, 1.53, 1.306, 0.879, -0.526, -1.71, -1.276, 0.756, -0.775, -1.048, -0.25, -0.595, -0.137, -0.598, 2.022, -0.812, 0.284, -0.488, -0.391, -0.403, -0.525, -0.061, -0.228, 1.086, 0.378, -0.14, 0.599, -0.087, -2.259, -0.098], [1.676, 0.232, -1.508, -0.145, 1.798, -1.388, 1.331, -0.37, -0.939, 0.043, 0.06, -0.414, -1.408, 0.24, 0.622, -0.55, -0.569, 1.873, -0.706, 1.924, -0.254, 1.927, -0.423, 0.152, -0.952, 0.509, -0.496, -0.968, 0.093, -1.049, -0.65, 0.312], [0.207, -0.775, -1.822, 0.321, -0.71, -0.201, 0.3, 1.146, -0.233, -0.753, -0.305, 1.309, -1.47, -0.21, 1.802, -1.555, -1.175, 1.323, -0.303, 0.722, -0.076, 0.103, -1.406, 1.931, 0.091, 0.237, 1.172, 1.607, 0.253, -0.9, -1.068, 0.438], [0.615, 1.077, 0.171, -0.175, 1.3, 0.901, -0.653, -0.138, 0.341, -0.654, -0.184, -0.441, -0.424, 0.356, -0.075, 0.26, -1.023, 0.814, 0.524, -0.904, -0.204, -0.623, 1.234, -1.03, 2.594, 0.56, 1.831, -0.199, -1.508, -0.492, -1.687, -2.165], [0.129, 0.008, -1.279, -0.412, -0.004, 1.663, 0.196, 0.104, 0.123, 0.119, 0.635, 1.757, 2.334, -0.799, -1.626, -1.26, 0.595, -0.316, -1.399, 0.232, 0.264, 1.386, -1.171, -0.256, -0.256, -1.944, 1.168, -0.368, -0.714, -0.51, 0.454, 1.148], [-0.32, 0.29, -1.309, -0.177, 0.453, 0.636, -0.024, 0.509, 0.931, -1.754, -1.575, 0.786, 0.046, -1.165, -1.416, 1.373, 1.293, -0.285, -1.541, -1.186, -0.106, -0.994, 2.001, 0.972, -0.02, 1.654, -0.236, 0.643, 1.02, 0.572, -0.914, -0.154], [0.7, -0.937, 0.441, 0.25, 0.78, -0.022, 0.282, -0.095, 1.558, -0.336, 1.706, 0.884, 1.28, 0.198, -0.796, 1.218, -1.769, 1.197, -0.342, -0.177, -0.645, 1.364, 0.008, -0.597, -0.484, -2.772, -0.696, -0.632, -0.34, -1.527, -0.562, 0.862], [2.504, 0.831, -1.271, -0.033, 0.298, -0.735, 1.339, 1.74, 0.233, -1.424, -0.819, -0.761, 0.291, 0.853, -0.092, -0.885, 0.164, 1.025, 0.907, 0.749, -1.515, -0.545, -1.365, 0.271, 0.034, -2.005, 0.031, 0.244, 0.621, 0.176, 0.336, -1.196], [-0.711, 0.591, -1.001, -0.946, 0.784, -1.66, 1.545, 0.799, -0.857, 1.148, 0.213, -0.285, 0.464, -0.139, 0.79, -1.663, -1.121, 0.575, -0.178, -0.508, 1.565, -0.242, -0.346, 1.024, -1.135, -0.158, -2.101, 0.275, 2.009, -0.425, 0.716, 0.981], [0.912, -1.186, -0.846, -0.421, -1.315, -0.827, 0.309, 0.533, 1.029, -2.343, 1.513, -1.238, 1.487, -0.849, 0.896, -0.927, -0.459, 0.159, 0.177, 0.873, 0.935, 1.433, -0.485, 0.737, 1.327, -0.338, 1.608, -0.47, -0.445, -1.118, -0.213, -0.446], [-0.434, -1.362, -1.098, -1.068, 1.507, 0.003, 0.413, -0.395, 0.897, -0.237, 1.405, -0.344, 1.693, 0.677, 0.097, -0.257, -0.602, 1.026, -1.229, 0.855, -0.713, 1.014, 0.443, 0.238, 0.425, -2.184, 1.933, -1.157, -1.132, -0.597, -0.785, 0.967], [0.58, -0.971, 0.789, -0.468, -0.576, 1.779, 1.747, 1.715, -1.939, 0.125, 0.656, -0.042, -1.024, -1.767, 0.107, -0.408, -0.866, -1.774, 1.248, 0.939, -0.033, 1.523, 1.168, -0.744, 0.209, -0.168, -0.316, 0.207, -0.432, 0.047, -0.646, -0.664], [-0.185, -0.613, -1.695, 1.602, -0.32, -0.277, 0.967, 0.728, -0.965, -0.234, 1.069, -0.63, -1.631, 0.711, 0.426, 1.298, -0.191, -0.467, -0.771, 0.971, -0.118, -1.577, -2.064, -0.055, -0.59, 0.642, -0.997, 1.251, 0.538, 1.367, 0.106, 1.704]]])  # fmt: skip

        # test with various tokenizer parameters
        tokenize_kwargs = {"max_length": 3}
        outputs = feature_extractor("This is a test", tokenize_kwargs=tokenize_kwargs)
        self.assertEqual(np.squeeze(outputs).shape, (3, 32))

        tokenize_kwargs = {"truncation": True, "padding": True, "max_length": 4}
        outputs = feature_extractor(
            ["This is a test", "This", "This is", "This is a", "This is a test test test test"],
            tokenize_kwargs=tokenize_kwargs,
        )
        self.assertEqual(np.squeeze(outputs).shape, (5, 4, 32))

        tokenize_kwargs = {"padding": True, "max_length": 4}
        outputs = feature_extractor(
            ["This is a test", "This", "This is", "This is a", "This is a test test test test"],
            truncation=True,
            tokenize_kwargs=tokenize_kwargs,
        )
        self.assertEqual(np.squeeze(outputs).shape, (5, 4, 32))

        # raise value error if truncation parameter given for two places
        tokenize_kwargs = {"truncation": True}
        with self.assertRaises(ValueError):
            _ = feature_extractor(
                ["This is a test", "This", "This is", "This is a", "This is a test test test test"],
                truncation=True,
                tokenize_kwargs=tokenize_kwargs,
            )

    @require_tf
    def test_tokenization_small_model_tf(self):
        feature_extractor = pipeline(
            task="feature-extraction", model="hf-internal-testing/tiny-random-distilbert", framework="tf"
        )
        # test with empty parameters
        outputs = feature_extractor("This is a test")
        self.assertEqual(
            nested_simplify(outputs),
            [[[2.287, 1.234, 0.042, 1.53, 1.306, 0.879, -0.526, -1.71, -1.276, 0.756, -0.775, -1.048, -0.25, -0.595, -0.137, -0.598, 2.022, -0.812, 0.284, -0.488, -0.391, -0.403, -0.525, -0.061, -0.228, 1.086, 0.378, -0.14, 0.599, -0.087, -2.259, -0.098], [1.676, 0.232, -1.508, -0.145, 1.798, -1.388, 1.331, -0.37, -0.939, 0.043, 0.06, -0.414, -1.408, 0.24, 0.622, -0.55, -0.569, 1.873, -0.706, 1.924, -0.254, 1.927, -0.423, 0.152, -0.952, 0.509, -0.496, -0.968, 0.093, -1.049, -0.65, 0.312], [0.207, -0.775, -1.822, 0.321, -0.71, -0.201, 0.3, 1.146, -0.233, -0.753, -0.305, 1.309, -1.47, -0.21, 1.802, -1.555, -1.175, 1.323, -0.303, 0.722, -0.076, 0.103, -1.406, 1.931, 0.091, 0.237, 1.172, 1.607, 0.253, -0.9, -1.068, 0.438], [0.615, 1.077, 0.171, -0.175, 1.3, 0.901, -0.653, -0.138, 0.341, -0.654, -0.184, -0.441, -0.424, 0.356, -0.075, 0.26, -1.023, 0.814, 0.524, -0.904, -0.204, -0.623, 1.234, -1.03, 2.594, 0.56, 1.831, -0.199, -1.508, -0.492, -1.687, -2.165], [0.129, 0.008, -1.279, -0.412, -0.004, 1.663, 0.196, 0.104, 0.123, 0.119, 0.635, 1.757, 2.334, -0.799, -1.626, -1.26, 0.595, -0.316, -1.399, 0.232, 0.264, 1.386, -1.171, -0.256, -0.256, -1.944, 1.168, -0.368, -0.714, -0.51, 0.454, 1.148], [-0.32, 0.29, -1.309, -0.177, 0.453, 0.636, -0.024, 0.509, 0.931, -1.754, -1.575, 0.786, 0.046, -1.165, -1.416, 1.373, 1.293, -0.285, -1.541, -1.186, -0.106, -0.994, 2.001, 0.972, -0.02, 1.654, -0.236, 0.643, 1.02, 0.572, -0.914, -0.154], [0.7, -0.937, 0.441, 0.25, 0.78, -0.022, 0.282, -0.095, 1.558, -0.336, 1.706, 0.884, 1.28, 0.198, -0.796, 1.218, -1.769, 1.197, -0.342, -0.177, -0.645, 1.364, 0.008, -0.597, -0.484, -2.772, -0.696, -0.632, -0.34, -1.527, -0.562, 0.862], [2.504, 0.831, -1.271, -0.033, 0.298, -0.735, 1.339, 1.74, 0.233, -1.424, -0.819, -0.761, 0.291, 0.853, -0.092, -0.885, 0.164, 1.025, 0.907, 0.749, -1.515, -0.545, -1.365, 0.271, 0.034, -2.005, 0.031, 0.244, 0.621, 0.176, 0.336, -1.196], [-0.711, 0.591, -1.001, -0.946, 0.784, -1.66, 1.545, 0.799, -0.857, 1.148, 0.213, -0.285, 0.464, -0.139, 0.79, -1.663, -1.121, 0.575, -0.178, -0.508, 1.565, -0.242, -0.346, 1.024, -1.135, -0.158, -2.101, 0.275, 2.009, -0.425, 0.716, 0.981], [0.912, -1.186, -0.846, -0.421, -1.315, -0.827, 0.309, 0.533, 1.029, -2.343, 1.513, -1.238, 1.487, -0.849, 0.896, -0.927, -0.459, 0.159, 0.177, 0.873, 0.935, 1.433, -0.485, 0.737, 1.327, -0.338, 1.608, -0.47, -0.445, -1.118, -0.213, -0.446], [-0.434, -1.362, -1.098, -1.068, 1.507, 0.003, 0.413, -0.395, 0.897, -0.237, 1.405, -0.344, 1.693, 0.677, 0.097, -0.257, -0.602, 1.026, -1.229, 0.855, -0.713, 1.014, 0.443, 0.238, 0.425, -2.184, 1.933, -1.157, -1.132, -0.597, -0.785, 0.967], [0.58, -0.971, 0.789, -0.468, -0.576, 1.779, 1.747, 1.715, -1.939, 0.125, 0.656, -0.042, -1.024, -1.767, 0.107, -0.408, -0.866, -1.774, 1.248, 0.939, -0.033, 1.523, 1.168, -0.744, 0.209, -0.168, -0.316, 0.207, -0.432, 0.047, -0.646, -0.664], [-0.185, -0.613, -1.695, 1.602, -0.32, -0.277, 0.967, 0.728, -0.965, -0.234, 1.069, -0.63, -1.631, 0.711, 0.426, 1.298, -0.191, -0.467, -0.771, 0.971, -0.118, -1.577, -2.064, -0.055, -0.59, 0.642, -0.997, 1.251, 0.538, 1.367, 0.106, 1.704]]])  # fmt: skip

        # test with various tokenizer parameters
        tokenize_kwargs = {"max_length": 3}
        outputs = feature_extractor("This is a test", tokenize_kwargs=tokenize_kwargs)
        self.assertEqual(np.squeeze(outputs).shape, (3, 32))

        tokenize_kwargs = {"truncation": True, "padding": True, "max_length": 4}
        outputs = feature_extractor(
            ["This is a test", "This", "This is", "This is a", "This is a test test test test"],
            tokenize_kwargs=tokenize_kwargs,
        )
        self.assertEqual(np.squeeze(outputs).shape, (5, 4, 32))

        tokenize_kwargs = {"padding": True, "max_length": 4}
        outputs = feature_extractor(
            ["This is a test", "This", "This is", "This is a", "This is a test test test test"],
            truncation=True,
            tokenize_kwargs=tokenize_kwargs,
        )
        self.assertEqual(np.squeeze(outputs).shape, (5, 4, 32))

        # raise value error if truncation parameter given for two places
        tokenize_kwargs = {"truncation": True}
        with self.assertRaises(ValueError):
            _ = feature_extractor(
                ["This is a test", "This", "This is", "This is a", "This is a test test test test"],
                truncation=True,
                tokenize_kwargs=tokenize_kwargs,
            )

    @require_torch
    def test_return_tensors_pt(self):
        feature_extractor = pipeline(
            task="feature-extraction", model="hf-internal-testing/tiny-random-distilbert", framework="pt"
        )
        outputs = feature_extractor("This is a test", return_tensors=True)
        self.assertTrue(torch.is_tensor(outputs))

    @require_tf
    def test_return_tensors_tf(self):
        feature_extractor = pipeline(
            task="feature-extraction", model="hf-internal-testing/tiny-random-distilbert", framework="tf"
        )
        outputs = feature_extractor("This is a test", return_tensors=True)
        self.assertTrue(tf.is_tensor(outputs))

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
            raise TypeError("We expect lists of floats, nothing else")
        return shape

    def get_test_pipeline(
        self,
        model,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        processor=None,
        torch_dtype="float32",
    ):
        if tokenizer is None:
            self.skipTest(reason="No tokenizer")
        elif (
            type(model.config) in FEATURE_EXTRACTOR_MAPPING
            or isinstance(model.config, LxmertConfig)
            or type(model.config) in IMAGE_PROCESSOR_MAPPING
        ):
            self.skipTest(
                reason="This is a bimodal model, we need to find a more consistent way to switch on those models."
            )
        elif model.config.is_encoder_decoder:
            self.skipTest(
                """encoder_decoder models are trickier for this pipeline.
                Do we want encoder + decoder inputs to get some featues?
                Do we want encoder only features ?
                For now ignore those.
                """
            )
        feature_extractor_pipeline = FeatureExtractionPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            torch_dtype=torch_dtype,
        )
        return feature_extractor_pipeline, ["This is a test", "This is another test"]

    def run_pipeline_test(self, feature_extractor, examples):
        outputs = feature_extractor("This is a test")

        shape = self.get_shape(outputs)
        self.assertEqual(shape[0], 1)

        # If we send too small input
        # there's a bug within FunnelModel (output with shape [1, 4, 2, 1] doesn't match the broadcast shape [1, 4, 2, 2])
        outputs = feature_extractor(["This is a test", "Another longer test"])
        shape = self.get_shape(outputs)
        self.assertEqual(shape[0], 2)

        outputs = feature_extractor("This is a test" * 100, truncation=True)
        shape = self.get_shape(outputs)
        self.assertEqual(shape[0], 1)
