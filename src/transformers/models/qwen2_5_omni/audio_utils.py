# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""Pure audio utility functions for Qwen2.5-Omni.

All functions are standalone (no model weights) and compute data-dependent
tensors from ``feature_lens`` + config scalars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_cu_seqlens(chunk_lengths: torch.Tensor) -> torch.Tensor:
    """Compute cumulative sequence lengths from chunk lengths after conv1.

    Args:
        chunk_lengths: ``(num_chunks,)``

    Returns:
        ``cu_seqlens``: ``(num_chunks + 1,)`` int32
    """
    after_conv1, _ = _get_feat_extract_output_lengths(chunk_lengths)
    return F.pad(after_conv1.cumsum(0), (1, 0), value=0).to(torch.int32)


def chunk_and_pad_features(
    input_features: torch.Tensor, feature_lens: torch.Tensor, n_window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk audio features into fixed-size windows and pad to equal length.

    Uses ``.tolist()`` for the variable-length split — not traceable by ``torch.export``.

    Args:
        input_features: ``(channels, total_frames)``
        feature_lens: ``(batch_size,)``
        n_window: half-window size from audio config.

    Returns:
        ``padded_feature``: ``(num_chunks, channels, max_chunk_len)``
        ``chunk_lengths``: ``(num_chunks,)`` long
    """
    chunk_num = torch.ceil(feature_lens / (n_window * 2)).long()
    chunk_lengths = torch.full((chunk_num.sum(),), n_window * 2, dtype=torch.long, device=feature_lens.device)
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (n_window * 2)
    chunk_lengths = torch.where(chunk_lengths == 0, n_window * 2, chunk_lengths)
    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)

    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    return padded_feature, chunk_lengths


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute output lengths after each CNN layer (conv1 then conv2)."""
    after_conv1 = (input_lengths - 1) // 2 + 1
    after_conv2 = (after_conv1 - 2) // 2 + 1
    return after_conv1, after_conv2


def get_valid_indices(chunk_lengths: torch.Tensor) -> torch.Tensor:
    """Compute flat indices of valid (non-padding) positions after conv1.

    Args:
        chunk_lengths: ``(num_chunks,)``

    Returns:
        ``valid_indices``: ``(total_valid_tokens,)`` long
    """
    after_conv1, _ = _get_feat_extract_output_lengths(chunk_lengths)
    max_len = after_conv1.max().item()
    mask = torch.arange(max_len, device=chunk_lengths.device) < after_conv1.unsqueeze(1)
    return mask.flatten().nonzero().squeeze(-1)


def get_pool_indices(feature_lens: torch.Tensor) -> torch.Tensor:
    """Compute indices for post-encoder average pooling on ragged hidden states.

    Args:
        feature_lens: ``(batch_size,)``

    Returns:
        ``pool_indices``: ``(total_pooled_tokens,)`` long
    """
    _, after_conv2 = _get_feat_extract_output_lengths(feature_lens)
    offsets = F.pad(after_conv2[:-1].cumsum(0), (1, 0), value=0)
    return torch.cat(
        [
            torch.arange(0, length - 1, 2, device=feature_lens.device) + offset
            for offset, length in zip(offsets, after_conv2)
        ]
    )
