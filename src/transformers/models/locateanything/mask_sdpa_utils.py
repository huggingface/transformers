# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch


def find_prefix_seq_length_by_pe(pe: torch.Tensor) -> torch.Tensor:
    """
    Find the sequence length where position encoding drops (indicating prefix boundary).
    Args:
        pe: Position encoding tensor of shape [Batch size, Sequence length ]
            Contains position indices for each token in the sequence.
    Returns:
        torch.Tensor: A tensor of shape [B] containing:
            - The index where position encoding drops for each sequence
            - -1 if no drop occurs in the sequence
    """
    batch_size, seq_len = pe.shape
    prev = pe[:, :-1]
    curr = pe[:, 1:]
    drop_mask = curr < prev  #  [batch_size, seq_len-1]

    seq_len = torch.full((batch_size,), -1, dtype=torch.long)

    for b in range(batch_size):
        drop_pos = torch.nonzero(drop_mask[b], as_tuple=False)
        if drop_pos.numel() > 0:
            i = drop_pos[0].item() + 1  # Take first drop position (+1 because we compared shifted sequences)
            seq_len[b] = i

    return seq_len


def update_causal_mask_with_pad_non_visible_2d(
    input_ids: torch.Tensor,
    attn_mask_2d: torch.Tensor,
    text_mask_token_id: int,
    block_size: int = 4,
    causal_attn: bool = False,
) -> torch.Tensor:
    """
    Updates a 2D attention mask for hole sequence through input_ids and text_mask_token_id

    Args:
        input_ids: Input token IDs (unused in current implementation)
        attn_mask_2d: 2D attention mask matrix of shape [seq_len, seq_len] where:
            - 0.0 indicates allowed attention
            - -inf indicates masked attention
        text_mask_token_id: ID representing masked tokens
        block_size: Size of the diffusion window
        causal_attn: If True, maintains strict causal masking throughout

    Returns:
        Modified attention mask with updated visibility patterns
    """
    seq_len = input_ids.shape[0]
    device = input_ids.device

    # Identify masked tokens and their preceding positions
    input_mask = input_ids.eq(text_mask_token_id)
    input_before_mask = torch.zeros_like(input_mask)
    input_before_mask[:-1] = input_mask[1:]
    mask_cols = input_mask | input_before_mask
    non_mask = ~mask_cols

    rows = torch.arange(seq_len, device=device)[:, None]
    cols = torch.arange(seq_len, device=device)

    indices = torch.arange(seq_len, device=device)
    prev_non_mask = (indices * non_mask).cummax(dim=0).values

    max_value = torch.iinfo(indices.dtype).max
    mask_indices = torch.where(non_mask, indices, torch.full_like(indices, max_value))
    reversed_mask_indices = torch.flip(mask_indices, dims=[0])
    reversed_cummin = reversed_mask_indices.cummin(dim=0).values
    next_non_mask = torch.flip(reversed_cummin, dims=[0])

    infra_mask = (cols > prev_non_mask) & (rows >= next_non_mask[None, :]) & mask_cols[None, :]
    attn_mask_2d.masked_fill_(infra_mask, -float("inf"))

    if not causal_attn:
        visible_mask = (rows > prev_non_mask[None, :]) & (rows < cols) & mask_cols[None, :]
        attn_mask_2d.masked_fill_(visible_mask, 0.0)

    return attn_mask_2d


def update_causal_mask_for_one_gen_window_2d(
    input_ids: torch.Tensor,
    attn_mask_2d: torch.Tensor,
    block_size: int = 4,
    use_cache: bool = True,
    causal_attn: bool = False,
) -> torch.Tensor:
    """
    Updates a 2D attention mask for a diffusion window in transformer inference.

    Args:
        input_ids: Input token IDs (unused in current implementation)
        attn_mask_2d: 2D attention mask matrix of shape [seq_len, seq_len] where:
            - 0.0 indicates allowed attention
            - -inf indicates masked attention
        block_size: Size of the diffusion window
        use_cache: Whether key-value cache is being used
        causal_attn: If True, maintains strict causal masking throughout

    Returns:
        Modified attention mask with updated visibility patterns
    """

    if not causal_attn:
        # Make the diffusion window (last block_size tokens) fully visible to itself
        # This allows bidirectional attention within the diffusion window
        attn_mask_2d[-block_size:, -block_size:] = 0.0
    if use_cache:
        # Mask the last token from previous round to prevent recomputation and maintain generation consistency.
        attn_mask_2d[-block_size:, -block_size - 1] = -float("inf")

    return attn_mask_2d


def create_block_diff_mask_by_pe_4d(
    block_size: int, x0_len_list: torch.Tensor, position_ids: torch.Tensor, causal_attn: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a 4D attention mask for block-difference attention patterns.

    The mask consists of three regions:
    1. Causal block (top-left): Standard causal attention for `x0` tokens.
    2. Mutual block (bottom-right): Non-causal attention within the same block for non-`x0` tokens.
    3. Prefix block (bottom-left): Non-`x0` tokens can attend to a prefix of `x0` tokens.

    Args:
        block_size (int): Size of processing blocks for non-`x0` tokens.
        x0_len_list (torch.Tensor): Tensor of shape [B] containing lengths of `x0` segments per batch.
        position_ids (torch.Tensor): Tensor of shape [B, seq_len] containing position IDs.
        causal_attn (bool, optional): If True, enforces causal masking in mutual blocks. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - A float mask of shape [batch_size, 1, seq_len, seq_len] with `-inf` for masked positions (non visiable).
            - A boolean mask of shape [batch_size, 1, seq_len, seq_len] indicating allowed attention positions.
    """
    batch_size, seq_len = position_ids.shape
    device = position_ids.device

    # Create position indices [batch_size, seq_len, seq_len]
    q_idx = torch.arange(seq_len, device=device).view(1, seq_len, 1)  # [1, seq_len, 1]
    kv_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len)  # [1, 1, seq_len]

    # Broadcast to [B, seq_len, seq_len]
    x0_len = x0_len_list.view(batch_size, 1, 1)  # [batch_size, 1, 1]
    x0_flag_q = q_idx < x0_len  # [batch_size, seq_len, seq_len]
    x0_flag_kv = kv_idx < x0_len

    # Block indices calculation [batch_size, seq_len, seq_len]
    q_block_idx = (q_idx - x0_len) // block_size
    kv_block_idx = (kv_idx - x0_len) // block_size

    # causal block (top-left)
    block_causal = x0_flag_q & x0_flag_kv & (q_idx >= kv_idx)

    mutual_condition = (q_idx >= kv_idx) if causal_attn else torch.ones_like(q_idx, dtype=torch.bool)
    block_mutual = ~x0_flag_q & ~x0_flag_kv & (q_block_idx == kv_block_idx) & mutual_condition

    q_blk = torch.div(q_idx - x0_len, block_size, rounding_mode="floor")
    q_blk_start = (x0_len_list.view(batch_size, 1) + q_blk[:, :, 0] * block_size).clamp(min=0, max=seq_len - 1)
    prefix_len = position_ids.gather(1, q_blk_start)
    prefix_len = prefix_len.unsqueeze(2)
    block_prefix = (~x0_flag_q & x0_flag_kv) & (kv_idx < prefix_len)

    final_mask = block_causal | block_mutual | block_prefix
    customized_mask = torch.full_like(final_mask, float("-inf"), dtype=torch.bfloat16)
    customized_mask.masked_fill_(final_mask, 0.0)

    return customized_mask.unsqueeze(1).to(device=device), final_mask.unsqueeze(1).to(device=device)


def find_pred_pos_from_input_ids(
    input_ids: torch.LongTensor = None,
    text_mask_token_id: int | None = None,
) -> torch.Tensor:
    """Compute the relative prediction positions for masked tokens in a sequence.

    For non-masked positions, the output is 0. For masked positions, the value increments
    by 1 for each consecutive mask token, indicating how many steps ahead the prediction is.

    Args:
        input_ids (torch.LongTensor): Input token IDs of shape [batch_size, seq_len].
        text_mask_token_id (int, optional): Token ID representing masked positions. Defaults to 151666.

    Returns:
        torch.Tensor: A tensor of shape [batch_size, seq_len] where:
            - 0 indicates a non-masked token.
            - n > 0 indicates the nth consecutive masked token (e.g., 1 = first mask, 2 = second mask, etc.).
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    is_mask = input_ids == text_mask_token_id

    base_mask = torch.zeros((batch_size, seq_len), dtype=torch.int8, device=device)

    for b in range(batch_size):
        for ix in range(1, seq_len):
            if is_mask[b][ix]:
                # Increment counter if current token is masked
                base_mask[b][ix] = base_mask[b][ix - 1] + 1

    return base_mask
