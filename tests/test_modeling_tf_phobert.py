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
    import numpy as np
    import tensorflow as tf

    from transformers import TFPhobertModel


@require_tf
class TFPhobertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_output_embeds_base_model(self):
        model = TFPhobertModel.from_pretrained("vinai/phobert-base")

        input_ids = tf.convert_to_tensor(
            [[0, 218, 8, 649, 212, 956, 2413, 5, 2]],
            dtype=tf.int32,
        )  # Tôi là sinh_viên trường đại_học Công_nghệ .

        output = model(input_ids)["last_hidden_state"]
        expected_shape = tf.TensorShape((1, 9, 768))
        self.assertEqual(output.shape, expected_shape)

        # compare the actual values for a slice.
        expected_slice = tf.convert_to_tensor(
            [[[-0.0254, 0.0235, 0.1027], [0.0606, -0.1811, -0.0418], [-0.1561, -0.1127, 0.2687]]],
            dtype=tf.float32,
        )

        self.assertTrue(np.allclose(output[:, :3, :3].numpy(), expected_slice.numpy(), atol=1e-4))
