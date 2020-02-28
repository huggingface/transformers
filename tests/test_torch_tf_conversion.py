# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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

from transformers import is_torch_available
from .utils import require_torch, torch_device
import sys

import tensorflow as tf
import torch
import numpy as np

import unittest

sys.path.insert(1, '../src')

from transformers.modeling_utils import top_k_top_p_filtering  # noqa: E402
from transformers.modeling_tf_utils import tf_top_k_top_p_filtering  # noqa: E402


@require_torch
class TorchTFConversionTest(unittest.TestCase):

    def test_top_k_top_p_filtering(self):
        np_logits = get_random_numpy_array((5, 100))
        pt_logits = to_pt(np_logits)
        tf_logits = to_tf(np_logits)

        # check with all arguments used
        top_k, top_p, min_tokens_to_keep = 10, 0.5, 6

        filtered_logits_pt = top_k_top_p_filtering(pt_logits, top_k=top_k, top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)
        filtered_logits_tf = tf_top_k_top_p_filtering(tf_logits, top_k=top_k, top_p=top_p, min_tokens_to_keep=min_tokens_to_keep)

        self.assertTrue(np.allclose(filtered_logits_pt, filtered_logits_tf))


# TODO: think about dtype
def get_random_numpy_array(shape, upper_limit=1, lower_limit=-1):
    return (upper_limit - lower_limit) * np.random.random(shape) + lower_limit


# TODO: think about dtype and device
def to_pt(np_tensor):
    return torch.from_numpy(np_tensor)


# TODO: think about dtype
def to_tf(np_tensor):
    return tf.convert_to_tensor(np_tensor)
