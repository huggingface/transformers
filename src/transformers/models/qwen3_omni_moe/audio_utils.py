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
"""Pure audio utility functions for Qwen3-Omni-MoE.

All functions are standalone (no model weights) and compute data-dependent
tensors from ``feature_lens`` + config scalars.
"""

import torch

from ..qwen2_5_omni.audio_utils import get_chunk_lengths  # noqa: F401 — re-export


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Compute output lengths after the 3-layer Conv2D stack."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def get_valid_indices(chunk_lengths: torch.Tensor) -> torch.Tensor:
    """Compute flat indices of valid (non-padding) positions after CNN downsampling.

    Uses the Qwen3-Omni-MoE 3-layer Conv2D formula.

    Args:
        chunk_lengths: ``(num_chunks,)``

    Returns:
        ``valid_indices``: ``(total_valid_tokens,)`` long
    """
    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    max_len_after_cnn = feature_lens_after_cnn.max().item()
    mask = torch.arange(max_len_after_cnn, device=chunk_lengths.device) < feature_lens_after_cnn.unsqueeze(1)
    return mask.flatten().nonzero().squeeze(-1)


def get_cu_seqlens(
    chunk_lengths: torch.Tensor, feature_lens: torch.Tensor, n_window_infer: int, n_window: int
) -> torch.Tensor:
    """Compute cumulative sequence lengths for windowed audio attention.

    Args:
        chunk_lengths: ``(num_chunks,)``
        feature_lens: ``(batch_size,)``
        n_window_infer: inference window size from config.
        n_window: half-window size from config.

    Returns:
        ``cu_seqlens``: ``(num_windows + 1,)`` int32
    """
    aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    max_len_after_cnn = feature_lens_after_cnn.max().item()
    cu_chunk_lens = [0]
    n_window_ratio = n_window_infer // (n_window * 2)
    window_aftercnn = max_len_after_cnn * n_window_ratio
    for cnn_len in aftercnn_lens:
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        remainder = cnn_len % window_aftercnn
        if remainder != 0:
            cu_chunk_lens += [remainder]
    return torch.tensor(cu_chunk_lens, device=feature_lens.device).cumsum(-1, dtype=torch.int32)
