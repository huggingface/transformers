# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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

from transformers import is_tf_available
from transformers.testing_utils import require_tf, slow


if is_tf_available():
    import tensorflow as tf
    import numpy as np
    from transformers import TFFlaubertModel


@require_tf
class TFFlaubertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_output_embeds_base_model(self):
        model = TFFlaubertModel.from_pretrained("jplu/tf-flaubert-small-cased")

        input_ids = tf.convert_to_tensor(
            [[0, 158, 735, 2592, 1424, 6727, 82, 1]], dtype=tf.int32,
        )  # "J'aime flaubert !"

        output = model(input_ids)[0]
        expected_shape = tf.TensorShape((1, 8, 512))
        self.assertEqual(output.shape, expected_shape)
        # compare the actual values for a slice.
        expected_slice = tf.convert_to_tensor(
            [
                [
                    [-1.8768773, -1.566555, 0.27072418],
                    [-1.6920038, -0.5873505, 1.9329599],
                    [-2.9563985, -1.6993835, 1.7972052],
                ]
            ],
            dtype=tf.float32,
        )

        self.assertTrue(np.allclose(output[:, :3, :3].numpy(), expected_slice.numpy(), atol=1e-4))
