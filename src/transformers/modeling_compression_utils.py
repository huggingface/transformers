# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
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
import torch
import torch.nn.functional as F

from .utils.generic import is_flash_attention_requested


def build_selection_mask(
    topk_indices: torch.Tensor,
    kv_length: int,
) -> torch.Tensor:
    """Convert selected indices into a boolean mask, ignoring unavailable entries."""
    # -1 is invalid as per convention
    topk_valid = topk_indices.ge(0) & topk_indices.lt(kv_length)

    # Clamp only so scatter has a legal index
    safe_indices = topk_indices.clamp(0, kv_length - 1)
    selected_counts = torch.zeros(
        *topk_indices.shape[:-1],
        kv_length,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    # On empty compression edge cases
    if kv_length > 0:
        selected_counts.scatter_add_(-1, safe_indices.long(), topk_valid.int())

    # Final mask 0 == False (not visible), 1 == True (visible)
    return selected_counts.ne(0)


def build_attention_mask_from_topk(
    config,
    topk_indices: torch.Tensor,
    query_states: torch.Tensor,
    kv_length: int,
) -> torch.Tensor | None:
    """
    Convert topk_indices into the mask expected by the active backend.
    SDPA and Eager return a full 4D mask while FA(4) can consume the indices directly (no mask).

    NOTE: Causality and padding is accounted for by previous modules, e.g. an indexer.
    """
    if is_flash_attention_requested(config):
        return None

    mask = build_selection_mask(topk_indices, kv_length).unsqueeze(1)

    # SDPA
    if config._attn_implementation == "sdpa":
        return mask

    # Eager
    min_dtype = torch.finfo(query_states.dtype).min
    # we need 0s where the tokens should be taken into account, and -inf otherwise (mask is already of boolean type)
    mask = torch.where(mask, torch.full((), 0.0, device=query_states.device, dtype=query_states.dtype), min_dtype)
    return mask


def prepare_compression_groups(
    states: torch.Tensor,
    attention_mask: torch.Tensor,
    group_size: int = 1,
    overlap_groups: int = 0,
    tail: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Combine retained history with incoming valid tokens. The attention mask contains
    cumulative valid-token counts, allowing us to group tokens independently of padding.

    Example with group_size=4:
        Tokens:         [A, B, C, D, E, F]
        Groups:         [ABCD, EF00]
        Tail:           [E, F, 0, 0]

    On the subsequent update:
        Previous tail:  [E, F, 0, 0]
        Incoming:       [G, H, I]
        Groups:         [EFGH]
        New tail:       [I, 0, 0, 0]

    With overlap_groups=1 (e.g. DSV4):
        Groups:         [0000ABCD, ABCDEF00]
        Tail:           [A, B, C, D, E, F, 0, 0]

    Output shapes:
        - Groups `(batch_size, ceil(seq_length / group_size), (1 + overlap_groups) * group_size, dim)`
        - Tails `(batch_size, (1 + overlap_groups) * group_size, dim)`

    NOTE: We keep incomplete groups into uniform shapes across the batch (unused slots == 0).
          They will be filtered out later on in `select_compression_indices` by default.
    """
    batch_size, seq_length, dim = states.shape
    device = states.device

    counts = attention_mask[:, -(seq_length + 1) :]
    history_length = (1 + overlap_groups) * group_size

    past_groups = counts[:, :1] // group_size
    new_groups = counts[:, -1:] // group_size - past_groups
    num_candidates = (seq_length + group_size - 1) // group_size

    if tail is None:
        window = states.new_zeros(batch_size, history_length + num_candidates * group_size, dim)
    else:
        window = F.pad(tail, (0, 0, 0, num_candidates * group_size))

    # Append valid tokens without overwriting retained history
    slots = (counts[:, 1:] - 1 - (past_groups - overlap_groups) * group_size).clamp_min(0)
    valid = counts.diff(dim=1) > 0

    window.scatter_add_(
        1,
        slots[..., None].expand_as(states),
        states.masked_fill(~valid[..., None], 0),
    )

    # Ordinary and overlapping compression differ only in window width
    groups = window.unfold(1, history_length, group_size).transpose(-1, -2)

    batch_indices = torch.arange(batch_size, device=device)
    tail = groups[batch_indices, new_groups.squeeze(1)]

    return groups[:, :-1], tail


def select_compression_indices(
    scores: torch.Tensor,
    attention_mask: torch.Tensor,
    budget: int,
    group_size: int = 1,
    include_tail: bool = False,
    return_groups: bool = False,
    allow_partial_groups: bool = False,
    deterministic: bool = False,
) -> torch.Tensor:
    """
    Select causal groups, excluding incomplete groups by default.

    By default, the budget counts tokens and selected groups are expanded into physical KV indices.

    With
        - `include_tail=True`, up to `group_size-1` additional tokens are selected outside the token budget when `return_groups=False`.
        - `return_groups=True`, the budget counts compressed entries instead (e.g. DSV4).
        - `allow_partial_groups=True`, an incomplete group competes within the budget using its supplied score.
        - `deterministic=True`, ties are broken towards earlier groups.

    Example with group_size=2 and budget=2:
        Tokens:            [PAD, A, B, PAD, C, D, E]
        Cumulative mask:   [0, 0, 1, 2, 2, 3, 4, 5]
        Complete groups:   [AB, CD]
        Selected group:    1 [CD]
        Valid-token ranks in mask: [2, 3]
        Physical indices:  [4, 5]
        Tail indices:      [6]
        Final selection:   [4, 5, 6]

    Finally, we pad unavailable token selections with -1 to keep uniform shapes.
    """
    if include_tail and allow_partial_groups:
        raise ValueError("`include_tail` and `allow_partial_groups` are mutually exclusive.")

    counts = attention_mask[:, -(scores.shape[-2] + 1) :]
    query_counts = counts[:, 1:].masked_fill(counts.diff(dim=1) == 0, 0).unsqueeze(-1)
    complete_groups = query_counts // group_size
    eligible_groups = (query_counts + group_size - 1) // group_size if allow_partial_groups else complete_groups

    # Select only eligible causal groups
    invalid = torch.arange(scores.shape[-1], device=scores.device) >= eligible_groups
    scores = scores.masked_fill(invalid, float("-inf"))
    k = min(budget if return_groups else budget // group_size, scores.shape[-1])

    if deterministic:
        top_scores, groups = scores.sort(dim=-1, descending=True, stable=True)
        top_scores, groups = top_scores[..., :k], groups[..., :k]
    else:
        top_scores, groups = scores.topk(k, dim=-1)

    groups = groups.masked_fill(top_scores == float("-inf"), -1)

    if return_groups:
        return groups

    # Expand groups into valid-token ranks, preserving -1 for invalid selections
    offsets = torch.arange(group_size, device=scores.device)
    indices = (groups[..., None] * group_size + offsets).flatten(-2).clamp_min(-1)

    if allow_partial_groups:
        indices = indices.masked_fill(indices >= query_counts, -1)

    if indices.shape[-1] < budget:
        indices = F.pad(indices, (0, budget - indices.shape[-1]), value=-1)

    # Optionally append the unfinished group outside the budget
    if include_tail:
        tail = complete_groups * group_size + offsets[:-1]
        indices = torch.cat((indices, tail.masked_fill(tail >= query_counts, -1)), dim=-1)

    # Convert ranks into KV indices
    return (
        torch.searchsorted(attention_mask, indices.flatten(1), right=True, out_int32=True).reshape_as(indices)
        - 1  # boundary
        + attention_mask[0, 0]  # negative prefix for static masks, zero otherwise
    )
