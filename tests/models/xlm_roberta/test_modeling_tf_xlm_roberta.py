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

from __future__ import annotations

import unittest

from transformers import is_tf_available
from transformers.testing_utils import require_sentencepiece, require_tf, require_tokenizers, slow


if is_tf_available():
    import numpy as np
    import tensorflow as tf

    from transformers import TFXLMRobertaModel


@require_tf
@require_sentencepiece
@require_tokenizers
class TFFlaubertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_output_embeds_base_model(self):
        model = TFXLMRobertaModel.from_pretrained("jplu/tf-xlm-roberta-base")

        features = {
            "input_ids": tf.convert_to_tensor([[0, 2646, 10269, 83, 99942, 2]], dtype=tf.int32),  # "My dog is cute"
            "attention_mask": tf.convert_to_tensor([[1, 1, 1, 1, 1, 1]], dtype=tf.int32),
        }

        output = model(features)["last_hidden_state"]
        expected_shape = tf.TensorShape((1, 6, 768))
        self.assertEqual(output.shape, expected_shape)
        # compare the actual values for a slice.
        expected_slice = tf.convert_to_tensor(
            [
                [
                    [0.0681762, 0.10894451, 0.06772504],
                    [-0.06423668, 0.02366615, 0.04329344],
                    [-0.06057295, 0.09974135, -0.00070584],
                ]
            ],
            dtype=tf.float32,
        )

        self.assertTrue(np.allclose(output[:, :3, :3].numpy(), expected_slice.numpy(), atol=1e-4))
