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
from transformers.testing_utils import require_tf


if is_tf_available():
    from transformers.activations_tf import get_tf_activation


@require_tf
class TestTFActivations(unittest.TestCase):
    def test_get_activation(self):
        get_tf_activation("swish")
        get_tf_activation("silu")
        get_tf_activation("gelu")
        get_tf_activation("relu")
        get_tf_activation("tanh")
        get_tf_activation("gelu_new")
        get_tf_activation("gelu_fast")
        get_tf_activation("mish")
        with self.assertRaises(KeyError):
            get_tf_activation("bogus")
        with self.assertRaises(KeyError):
            get_tf_activation(None)
