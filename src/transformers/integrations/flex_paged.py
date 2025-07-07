import torch

from ..generation.continuous_batching import PagedAttentionCache
from ..utils import is_flash_attn_2_available
from torch.nn.attention.flex_attention import BlockMask, _identity, create_block_mask

create_block_mask = torch.compile(create_block_mask)

# self.block_mask = create_block_mask(
#     lambda b, h, q, kv: q >= kv, batch_size, 1, 64 * 1024, 64 * 1024, BLOCK_SIZE=page_size
# )

# PREFILL:
#         # generate block mask. The same block mask is used for all layers.
# new_block_mask = slice_block_mask(self.block_mask, batch_idx, prompt_len, prompt_len)
# converted_block_mask = self.paged_attention.convert_logical_block_mask(
#     new_block_mask, torch.tensor([batch_idx], device="cuda")
# )
# converted_score_mod = self.paged_attention.get_score_mod(_identity)
#
# DECODE
#         mask = self.get_decode_mask(batch_idx, input_pos)
# converted_block_mask = self.paged_attention.convert_logical_block_mask(mask, batch_idx)
# converted_score_mod = self.paged_attention.get_score_mod(_identity)


def convert_logical_block_mask(
    block_mask: BlockMask,
    page_size: int,
    n_pages: int,
    page_table: torch.Tensor,
    physical_to_logical: torch.Tensor,
    batch_idx: Optional[torch.Tensor] = None,
) -> BlockMask:
    B, H, ROWS, MAX_BLOCKS_IN_COL = block_mask.kv_indices.shape

    if block_mask.BLOCK_SIZE[1] != page_size:
        raise RuntimeError(
            f"Expect block_mask has the same column block size as page_size"
            f" but got size={block_mask.BLOCK_SIZE[1]} and size={page_size}"
        )

    device = block_mask.kv_num_blocks.device
    if batch_idx is None:
        batch_idx = torch.arange(B, device=device)

    batched_page_table = page_table[batch_idx]
    new_kv_num_blocks = block_mask.kv_num_blocks.clone()
    new_kv_indices = torch.zeros((B, H, ROWS, n_pages), dtype=torch.int32, device=device)
    new_kv_indices[:, :, :, :MAX_BLOCKS_IN_COL] = (
        torch.gather(batched_page_table, 1, block_mask.kv_indices.view(B, -1).to(torch.int64))
        .view(block_mask.kv_indices.shape)
        .to(torch.int32)
    )

    new_full_kv_indices, new_full_kv_num_blocks = None, None
    if block_mask.full_kv_num_blocks is not None:
        assert block_mask.full_kv_indices is not None
        new_full_kv_num_blocks = block_mask.full_kv_num_blocks.clone()
        new_full_kv_indices = torch.zeros((B, H, ROWS, n_pages), dtype=torch.int32, device=device)
        new_full_kv_indices[:, :, :, :MAX_BLOCKS_IN_COL] = (
            torch.gather(
                batched_page_table,
                1,
                block_mask.full_kv_indices.view(B, -1).to(torch.int64),
            )
            .view(block_mask.full_kv_indices.shape)
            .to(torch.int32)
        )

    new_mask_mod = get_mask_mod(block_mask.mask_mod, page_size, physical_to_logical)
    seq_lengths = (block_mask.seq_lengths[0], n_pages * page_size)

    return BlockMask.from_kv_blocks(
        new_kv_num_blocks,
        new_kv_indices,
        new_full_kv_num_blocks,
        new_full_kv_indices,
        block_mask.BLOCK_SIZE,
        new_mask_mod,
        seq_lengths=seq_lengths,
    )


def get_mask_mod(
    mask_mod: Optional[_mask_mod_signature],
    page_size: int,
    physical_to_logical: torch.Tensor,
) -> _mask_mod_signature:
    if mask_mod is None:
        mask_mod = noop_mask

    def new_mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        physical_kv_idx: torch.Tensor,
    ):
        physical_kv_block = physical_kv_idx // page_size
        physical_kv_offset = physical_kv_idx % page_size
        logical_block_idx = physical_to_logical[b, physical_kv_block]
        logical_kv_idx = logical_block_idx * page_size + physical_kv_offset
        return torch.where(logical_block_idx >= 0, mask_mod(b, h, q_idx, logical_kv_idx), False)

    return new_mask_mod


def get_score_mod(
    score_mod: Optional[_score_mod_signature],
    page_size: int,
    physical_to_logical: torch.Tensor,
) -> _score_mod_signature:
    if score_mod is None:
        score_mod = _identity

    def new_score_mod(
        score: torch.Tensor,
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        physical_kv_idx: torch.Tensor,
    ):
        physical_kv_block = physical_kv_idx // page_size
        physical_kv_offset = physical_kv_idx % page_size
        logical_block_idx = physical_to_logical[b, physical_kv_block]
        logical_kv_idx = logical_block_idx * page_size + physical_kv_offset
        return torch.where(
            logical_block_idx >= 0,
            score_mod(score, b, h, q_idx, logical_kv_idx),
            float("-inf"),
        )

    return new_score_mod


def get_decode_mask(
    batch_idx: torch.Tensor,
    input_pos: torch.Tensor,
    block_mask: BlockMask,
) -> BlockMask:
    """
    Constructs a BlockMask for decoding using only the relevant input position.

    Args:
        batch_idx (Tensor): shape [B]
        input_pos (Tensor): shape [B]
        block_mask (BlockMask): full block mask to extract from
    """
    (B,) = batch_idx.shape
    BLOCK_SIZE = block_mask.BLOCK_SIZE
    input_block_idx = input_pos // BLOCK_SIZE[0]  # [B]

    kv_num_blocks = block_mask.kv_num_blocks[batch_idx, :, input_block_idx].view(B, 1, 1)
    kv_indices = block_mask.kv_indices[batch_idx, :, input_block_idx].view(B, 1, 1, -1)

    full_kv_num_blocks, full_kv_indices = None, None
    if block_mask.full_kv_num_blocks is not None:
        full_kv_num_blocks = block_mask.full_kv_num_blocks[batch_idx, :, input_block_idx].view(B, 1, 1)
        full_kv_indices = block_mask.full_kv_indices[batch_idx, :, input_block_idx].view(B, 1, 1, -1)

    seq_lengths = (1, block_mask.seq_lengths[1])

    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=BLOCK_SIZE,
        mask_mod=gen_offset(input_pos),
        seq_lengths=seq_lengths,
    )


def flex_paged_forward(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor = None,
    cache: PagedAttentionCache = None,
    cumulative_seqlens_q=None,
    cumulative_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    block_tables=None,
    is_prefill=None,
    **kwargs,
) -> torch.Tensor:
    k, v = cache.update(k, v, module.layer_idx, cumulative_seqlens_k=cumulative_seqlens_k, **kwargs)

    attn_output = flex_attention(
        q,
        k,
        v,
        score_mod=score_mod,
        blcok_mask=mask_mod,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None
