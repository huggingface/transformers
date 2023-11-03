# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from transformers import AutoTokenizer, is_flax_available
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
        model = FlaxXLMRobertaModel.from_pretrained("xlm-roberta-base")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        text = "The dog is cute and lives in the garden house"
        input_ids = jnp.array([tokenizer.encode(text)])

        expected_output_shape = (1, 12, 768)  # batch_size, sequence_length, embedding_vector_dim
        expected_output_values_last_dim = jnp.array(
            [[-0.0101, 0.1218, -0.0803, 0.0801, 0.1327, 0.0776, -0.1215, 0.2383, 0.3338, 0.3106, 0.0300, 0.0252]]
        )

        output = model(input_ids)["last_hidden_state"]
        self.assertEqual(output.shape, expected_output_shape)
        # compare the actual values for a slice of last dim
        self.assertTrue(jnp.allclose(output[:, :, -1], expected_output_values_last_dim, atol=1e-3))
