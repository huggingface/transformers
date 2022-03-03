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

from transformers import is_flax_available
from transformers.testing_utils import require_flax, require_sentencepiece, require_tokenizers, slow


if is_flax_available():
    import jax.numpy as jnp
    from transformers import FlaxXLMRobertaModel


@require_sentencepiece
@require_tokenizers
@require_flax
class FlaxXLMRobertaModelIntegrationTest(unittest.TestCase):
    @slow
    def test_flax_xlm_roberta_base(self):
        model = FlaxXLMRobertaModel.from_pretrained("xlm-roberta-base", from_pt=True)
        input_ids = jnp.array([[0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2]])
        # The dog is cute and lives in the garden house

        expected_output_shape = (1, 12, 768)  # batch_size, sequence_length, embedding_vector_dim
        expected_output_values_last_dim = jnp.array(
            [[-0.0101, 0.1218, -0.0803, 0.0801, 0.1327, 0.0776, -0.1215, 0.2383, 0.3338, 0.3106, 0.0300, 0.0252]]
        )

        output = model(input_ids)["last_hidden_state"]
        self.assertEqual(output.shape, expected_output_shape)
        # compare the actual values for a slice of last dim
        self.assertTrue(jnp.allclose(output[:, :, -1], expected_output_values_last_dim, atol=1e-3))

    @slow
    def test_flax_xlm_roberta_large(self):
        model = FlaxXLMRobertaModel.from_pretrained("xlm-roberta-large", from_pt=True)
        input_ids = jnp.array([[0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2]])
        # The dog is cute and lives in the garden house

        expected_output_shape = (1, 12, 1024)  # batch_size, sequence_length, embedding_vector_dim
        expected_output_values_last_dim = jnp.array(
            [[-0.0699, -0.0318, 0.0705, -0.1241, 0.0999, -0.0520, 0.1004, -0.1838, -0.4704, 0.1437, 0.0821, 0.0126]]
        )

        output = model(input_ids)["last_hidden_state"]
        self.assertEqual(output.shape, expected_output_shape)
        # compare the actual values for a slice of last dim
        self.assertTrue(jnp.allclose(output[:, :, -1], expected_output_values_last_dim, atol=1e-3))
