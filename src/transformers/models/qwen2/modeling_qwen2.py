# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch Qwen2 model."""


import torch
import torch.utils.checkpoint

from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...utils import (
    is_flash_attn_2_available,
    logging,
)


if is_flash_attn_2_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)


def qwen2_flash_attention_forward(
    config, query, key, value, attention_mask, target_dtype=torch.float16, training=False, layer_idx=0, **kwargs
):
    if attention_mask is not None:
        seq_len = attention_mask.shape[1]
        query = query[:, :, :seq_len]
        value = value[:, :, :seq_len]
    else:
        seq_len = query.shape[1]

    dropout_rate = config.attention_dropout if training else 0.0

    sliding_window = None
    if (
        config.use_sliding_window
        and getattr(config, "sliding_window", None) is not None
        and layer_idx >= config.max_window_layers
    ):
        sliding_window = config.sliding_window

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        seq_len,
        config=config,
        dropout=dropout_rate,
        layer_idx=layer_idx,
        sliding_window=sliding_window,
        **kwargs,
    )

    return attn_output, None

ALL_ATTENTION_FUNCTIONS["qwen2_flash_attention2"] = qwen2_flash_attention_forward

# TODO mask creation is also different for Qwen2
# MAKE SURE THAT config._attn_implementation is automatically qwen2_flash_attention2
