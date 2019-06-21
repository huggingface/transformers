# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Utilities for PyTorch XLNet model.
"""

from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

special_symbols = {
    "<unk>"  : 0,
    "<s>"    : 1,
    "</s>"   : 2,
    "<cls>"  : 3,
    "<sep>"  : 4,
    "<pad>"  : 5,
    "<mask>" : 6,
    "<eod>"  : 7,
    "<eop>"  : 8,
}

VOCAB_SIZE = 32000
UNK_ID = special_symbols["<unk>"]
CLS_ID = special_symbols["<cls>"]
SEP_ID = special_symbols["<sep>"]
MASK_ID = special_symbols["<mask>"]
EOD_ID = special_symbols["<eod>"]


def permutation_mask(inputs, targets, is_masked, perm_size, seq_len):
    """
    Sample a permutation of the factorization order, and create an
    attention mask accordingly.
    Args:
        inputs: int64 Tensor in shape [seq_len], input ids.
        targets: int64 Tensor in shape [seq_len], target ids.
        is_masked: bool Tensor in shape [seq_len]. True means being selected
            for partial prediction.
        perm_size: the length of longest permutation. Could be set to be reuse_len.
            Should not be larger than reuse_len or there will be data leaks.
        seq_len: int, sequence length.
    """

    # Generate permutation indices
    index = np.arange(10)
    index = np.transpose(np.reshape(index, [-1, perm_size]))
    index = np.random.shuffle(index)
    index = np.reshape(np.transpose(index), [-1])

    # `perm_mask` and `target_mask`
    # non-functional tokens
    non_func_tokens = tf.logical_not(tf.logical_or(
        tf.equal(inputs, SEP_ID),
        tf.equal(inputs, CLS_ID)))

    non_mask_tokens = tf.logical_and(tf.logical_not(is_masked), non_func_tokens)
    masked_or_func_tokens = tf.logical_not(non_mask_tokens)

    # Set the permutation indices of non-masked (& non-funcional) tokens to the
    # smallest index (-1):
    # (1) they can be seen by all other positions
    # (2) they cannot see masked positions, so there won"t be information leak
    smallest_index = -tf.ones([seq_len], dtype=tf.int64)
    rev_index = tf.where(non_mask_tokens, smallest_index, index)

    # Create `target_mask`: non-funcional and maksed tokens
    # 1: use mask as input and have loss
    # 0: use token (or [SEP], [CLS]) as input and do not have loss
    target_tokens = tf.logical_and(masked_or_func_tokens, non_func_tokens)
    target_mask = tf.cast(target_tokens, tf.float32)

    # Create `perm_mask`
    # `target_tokens` cannot see themselves
    self_rev_index = tf.where(target_tokens, rev_index, rev_index + 1)

    # 1: cannot attend if i <= j and j is not non-masked (masked_or_func_tokens)
    # 0: can attend if i > j or j is non-masked
    perm_mask = tf.logical_and(
        self_rev_index[:, None] <= rev_index[None, :],
        masked_or_func_tokens)
    perm_mask = tf.cast(perm_mask, tf.float32)

    # new target: [next token] for LM and [curr token] (self) for PLM
    new_targets = tf.concat([inputs[0: 1], targets[: -1]],
                            axis=0)

    # construct inputs_k
    inputs_k = inputs

    # construct inputs_q
    inputs_q = target_mask

    return perm_mask, new_targets, target_mask, inputs_k, inputs_q

