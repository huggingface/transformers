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
from transformers import LSHSelfAttention


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

    def __init__(self, shape=(3, 32, 8)):
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
        shape=None,
        num_attention_heads=None,
        hidden_size=None,
        query_key_chunk_len=None,
        num_chunks_before=None,
        num_chunks_after=None,
        num_hashes=None,
        num_buckets=None,
        use_reference_code=True,
        attention_dropout=0.0,
        mode="train",
        seed=None,
        path_to_save_weights=None,
        **kwargs
    ):

        with trax_math.use_backend("jax"):
            if shape is None:
                shape = self._shape
            layer = TraxLSHSelfAttention(
                n_heads=num_attention_heads,
                d_qk=hidden_size,
                d_v=hidden_size,
                causal=False,
                chunk_len=query_key_chunk_len,
                n_chunks_before=num_chunks_before,
                n_chunks_after=num_chunks_after,
                n_hashes=num_hashes,
                n_buckets=num_buckets,
                use_reference_code=use_reference_code,
                attention_dropout=attention_dropout,
                mode=mode,
                hash_seed=seed,
                path_to_save_weights=path_to_save_weights
            )

        return layer

    def forward_layer(
        self,
        np_input_data,
        layer=None,
        input_signature=None,
        random_number_generator=None,
    ):
        with trax_math.use_backend("jax"):
            input_data = self.convert_to_jax_array(np_input_data)

            if layer is None:
                layer = self.get_layer()

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

    hidden_size = 32
    seq_len = 7

    def _get_random_input(self, shape):
        return np.random.rand(*shape)

    def _get_trax_utils(self, shape):
        return TraxUtils(shape)

    def _create_config(
        self,
        input_size=8,
        num_attention_heads=2,
        num_hashes=2,
        num_buckets=4,
        num_chunks_before=1,
        num_chunks_after=0,
        seed=0,
        path_to_save_weights="/home/patrick/hugging_face/experiments/reformer/intermediate_weights"
    ):
        return {
            "input_size": self.hidden_size,
            "num_attention_heads": num_attention_heads,
            "hidden_size": self.hidden_size,
            "hf_hidden_size": num_attention_heads * self.hidden_size,
            "num_hashes": num_hashes,
            "num_buckets": num_buckets,
            "query_key_chunk_len": self.seq_len,
            "num_chunks_before": num_chunks_before,
            "num_chunks_after": num_chunks_after,
            "seed": seed,
            "path_to_save_weights": path_to_save_weights
        }

    def test_lsh_hashing(self):
        shape = (1, self.seq_len, self.hidden_size)  # Batch x SeqLen x ModelDim

        config_dict = self._create_config()

        np_input = self._get_random_input(shape)

        trax_utils = self._get_trax_utils(shape)

        trax_layer = trax_utils.get_layer(**config_dict)
        trax_output, trax_weights, trax_state = trax_utils.forward_layer(
            np_input, layer=trax_layer
        )  # noqa: F841
        trax_torch_output = torch.tensor(np.asarray(trax_output))

        hf_layer = LSHSelfAttention(config_dict)

        # set torch weights for 1-to-1 comparison
        with torch.no_grad():
            np_query_key = np.asarray(trax_weights[0])
            np_value = np.asarray(trax_weights[1])
            np_dense = np.asarray(trax_weights[2])

            hf_layer.query_key.weight = torch.nn.Parameter(torch.tensor(np_query_key).transpose(1, 2).contiguous().view(-1, self.hidden_size))
            hf_layer.value.weight = torch.nn.Parameter(torch.tensor(np_value).transpose(1, 2).contiguous().view(-1, self.hidden_size))
            hf_layer.dense.weight = torch.nn.Parameter(torch.tensor(np_dense).view(-1, self.hidden_size).contiguous().transpose(0, 1))

        hf_input = torch.tensor(np_input, dtype=torch.float)
        hf_output = hf_layer(hf_input)

        assert torch.allclose(hf_output, trax_torch_output, atol=1e-6)

        pass
