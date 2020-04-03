# coding=utf-8 # Copyright 2020 Huggingface
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

# trax imports - to be deleted later
from trax import math as trax_math
from trax.shapes import ShapeDtype as trax_ShapeDtype
import jax
from trax.layers.research.efficient_attention_v2 import (
    LSHSelfAttention as TraxLSHSelfAttention,
)
from transformers import ReformerAttention, ReformerConfig


from transformers import is_torch_available  # noqa: F401
from .utils import require_torch, torch_device  # noqa: F401


if is_torch_available():
    import torch  # noqa: F401
#    from transformers.modeling_reformer import ()


class TraxUtils(object):
    """ class that will help for testing in the beginning
        should be deleted step-by-step

        README (HOW-TO-INSTALL TRAX):
        1) git clone https://github.com/patrickvonplaten/trax.git

           - I had to do one tiny change to make the imports work,
             see: https://github.com/patrickvonplaten/trax/commit/6c23e88afe7f1c57b0c38eeaa4d450e5f912590c)
        2) link your PYTHON_PATH to ~/trax/trax
        3) pip install all the missing packages HINT: the package gin is installed

           - HINT: the package gin is installed with pip install gin-config==0.1.4
                   and not pip install gin.
           - The other packages can just be installed with pip install <package> form
             error message "<package> missing"
    """

    def __init__(self, shape):
        self._shape = shape

    def convert_to_jax_array(self, np_array):
        return jax.numpy.asarray(np_array)

    def get_input_signature(self, shape=None):
        with trax_math.use_backend("jax"):
            if shape is None:
                shape = self._shape
            input_signature = trax_ShapeDtype(shape)
        return input_signature

    def get_layer(
        self,
        config,
        use_reference_code=True,
        mode="train",
        path_to_save_weights="/home/patrick/hugging_face/experiments/reformer/intermediate_weights",
        **kwargs
    ):

        with trax_math.use_backend("jax"):
            hidden_size_per_head = config.hidden_size // config.num_attention_heads
            layer = TraxLSHSelfAttention(
                n_heads=config.num_attention_heads,
                d_qk=hidden_size_per_head,
                d_v=hidden_size_per_head,
                chunk_len=config.chunk_length,
                n_chunks_before=config.num_chunks_before,
                n_chunks_after=config.num_chunks_after,
                n_hashes=config.num_hashes,
                n_buckets=config.num_buckets,
                attention_dropout=config.attention_probs_dropout_prob,
                output_dropout=config.hidden_dropout_prob,
                hash_seed=config.seed,
                causal=False,
                use_reference_code=use_reference_code,
                mode=mode,
                path_to_save_weights=path_to_save_weights
            )

        return layer

    def forward_layer(
        self,
        np_input_data,
        layer,
        input_signature=None,
        random_number_generator=None,
    ):
        with trax_math.use_backend("jax"):
            input_data = self.convert_to_jax_array(np_input_data)

            if input_signature is None:
                input_signature = self.get_input_signature()

            weights, state = layer.init(input_signature)

            if random_number_generator is None:
                random_number_generator = layer.new_rngs(1)[0]

            output = layer(
                input_data, weights=weights, state=state, rng=random_number_generator
            )

        return output, weights, state


@require_torch
class ReformerIntegrationTests(unittest.TestCase):

    def _set_weights_in_torch(self, weights, torch_layer, hidden_size_per_head):
        # set torch weights for 1-to-1 comparison
        with torch.no_grad():
            np_query_key = np.asarray(weights[0])
            np_value = np.asarray(weights[1])
            np_dense = np.asarray(weights[2])

            torch_layer.self_attention.query_key.weight = torch.nn.Parameter(torch.tensor(np_query_key).transpose(1, 2).contiguous().view(-1, hidden_size_per_head))

            torch_layer.self_attention.value.weight = torch.nn.Parameter(torch.tensor(np_value).transpose(1, 2).contiguous().view(-1, hidden_size_per_head))

            torch_layer.output.dense.weight = torch.nn.Parameter(torch.tensor(np_dense).view(-1, hidden_size_per_head).contiguous().transpose(0, 1))

    def test_lsh_hashing(self):
        config = ReformerConfig()

        hidden_size_per_head = config.hidden_size // config.num_attention_heads

        shape = (3, 7, hidden_size_per_head)  # Batch x SeqLen x ModelDimPerHead
        np_input = np.random.rand(*shape)

        trax_utils = TraxUtils(shape)
        trax_layer = trax_utils.get_layer(config)
        trax_output, trax_weights, trax_state = trax_utils.forward_layer(np_input, layer=trax_layer)
        trax_torch_output = torch.tensor(np.asarray(trax_output))

        hf_input = torch.tensor(np_input, dtype=torch.float)
        hf_layer = ReformerAttention(config)
        self._set_weights_in_torch(trax_weights, hf_layer, hidden_size_per_head)
        hf_output = hf_layer(hf_input)[0]

        self.assertTrue(torch.allclose(hf_output, trax_torch_output, atol=1e-6))
