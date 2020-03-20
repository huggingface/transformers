# coding=utf-8
# Copyright 2020 Huggingface
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
        from trax import math
        from trax.math import numpy as trax_np
        from trax.shapes import ShapeDtype
        import jax
        from trax.layers.research.efficient_attention_v2 import LSHSelfAttention

        self.math = math
        self.np_trax = trax_np
        self.ShapeDtype = ShapeDtype
        self.jax = jax
        self.LSHSelfAttention = LSHSelfAttention
        self._shape = shape

    def convert_to_jax_array(self, np_array):
        return self.jax.numpy.asarray(np_array)

    def get_input_signature(self, shape=None):
        with self.math.use_backend("jax"):
            if shape is None:
                shape = self._shape
            input_signature = self.ShapeDtype(shape)
        return input_signature

    def get_lsh_self_attention_layer(
        self,
        shape=None,
        n_heads=5,
        d_qk=7,
        d_v=17,
        causal=True,
        chunk_len=8,
        n_chunks_before=1,
        n_chunks_after=0,
        n_hashes=2,
        n_buckets=4,
        use_reference_code=True,
        attention_dropout=0.0,
        mode="train",
    ):

        with self.math.use_backend("jax"):
            if shape is None:
                shape = self._shape
            layer = self.LSHSelfAttention(
                n_heads=n_heads,
                d_qk=d_qk,
                d_v=d_v,
                causal=causal,
                chunk_len=chunk_len,
                n_chunks_before=n_chunks_before,
                n_chunks_after=n_chunks_after,
                n_hashes=n_hashes,
                n_buckets=n_buckets,
                use_reference_code=use_reference_code,
                attention_dropout=attention_dropout,
                mode=mode,
            )

        return layer

    def forward_layer(
        self,
        np_input_data,
        layer=None,
        input_signature=None,
        random_number_generator=None,
    ):
        with self.math.use_backend("jax"):
            input_data = self.convert_to_jax_array(np_input_data)

            if layer is None:
                layer = self.get_lsh_self_attention_layer()

            if input_signature is None:
                input_signature = self.get_input_signature()

            import ipdb
            ipdb.set_trace()

            weights, state = layer.init(input_signature)

            if random_number_generator is None:
                random_number_generator = layer.new_rngs(1)[0]

            output = layer(
                input_data, weights=weights, state=state, rng=random_number_generator
            )

        return output


@require_torch
class ReformerIntegrationTests(unittest.TestCase):
    def _get_random_input(self, shape):
        return np.random.rand(*shape)

    def _get_trax_utils(self, shape):
        return TraxUtils(shape)

    def test_lsh_hashing(self):
        shape = (3, 32, 8)

        np_input = self._get_random_input(shape)
        trax_utils = self._get_trax_utils(shape)

        lsh_trax_output = trax_utils.forward_layer(np_input)  # noqa: F841

        pass
