# coding=utf-8
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

from transformers import is_tf_available
from transformers.testing_utils import require_sentencepiece, require_tf, require_tokenizers, slow


if is_tf_available():
    import numpy as np
    import tensorflow as tf

    from transformers import TFAutoModel


@require_tf
@require_sentencepiece
@require_tokenizers
class TFBortIntegrationTest(unittest.TestCase):
    @slow
    def test_output_embeds_base_model(self):
        model = TFAutoModel.from_pretrained("amazon/bort")

        input_ids = tf.convert_to_tensor(
            [[0, 18077, 4082, 7804, 8606, 6195, 2457, 3321, 11, 10489, 16, 269, 2579, 328, 2]],
            dtype=tf.int32,
        )  # Schloß Nymphenburg in Munich is really nice!

        output = model(input_ids)["last_hidden_state"]
        expected_shape = tf.TensorShape((1, 15, 1024))
        self.assertEqual(output.shape, expected_shape)
        # compare the actual values for a slice.
        expected_slice = tf.convert_to_tensor(
            [[[-0.0349, 0.0436, -1.8654], [-0.6964, 0.0835, -1.7393], [-0.9819, 0.2956, -0.2868]]],
            dtype=tf.float32,
        )

        self.assertTrue(np.allclose(output[:, :3, :3].numpy(), expected_slice.numpy(), atol=1e-4))
