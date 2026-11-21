# [scratch, gap4 zssample] Fused sampler: two kernels over the bf16 logits give the sampled token, its exact logprob,
# and the speculative ratio test of a drafted token. Used by the CB model runner when CB_FUSED_SAMPLE=1.
#
# Sampling is hierarchical and exact in distribution: pass 1 reduces each block of BLOCK tokens to its max and sum of
# exponentials (no randomness, one read of the logits), pass 2 picks a block with probability proportional to its
# mass (exponential race over the blocks), then a token in that block with probability proportional to its own mass
# (exponential race over the block's tokens, the only per-token random numbers): P(token) = P(block) * P(token | block)
# = softmax(logits / T)[token].
#
# A drafted token (point-mass proposal) is accepted with probability p(draft); if rejected, the emitted token is a
# sample of p without the draft (the residual distribution): pass 1 leaves the draft out of its block sums and pass 2
# out of its race, and adds it back only to the normalizer, so the residual mass is never a difference of sums.
import torch
import triton
import triton.language as tl


@triton.jit
def _uniform(seed, c0, c1, c2, c3):
    """Uniform in the open interval (0, 1) from Philox4x32-10 counters (c0, c1, c2, c3), resolved to 2^-33 near 0."""
    r, _, _, _ = tl.philox(seed, c0, c1, c2, c3)
    return tl.minimum((r.to(tl.float32) + 0.5) * 2.3283064365386963e-10, 0.99999994)


@triton.jit
def _exponential(seed, c0, c1, c2, c3):
    """Exp(1) variates, never 0 (a zero would make its token win the race whatever its probability) and resolved down
    to 1e-10 (the tail where unlikely tokens win)."""
    v = _uniform(seed, c0, c1, c2, c3)
    # E = -log(1 - v): the series near 0, where 1 - v would lose the low bits of v
    series = v * (1.0 + v * (0.5 + v * 0.3333333333333333))
    return tl.where(v < 0.0078125, series, -tl.log(1.0 - v))


@triton.jit
def _block_partials_kernel(
    logits_ptr, stride_row, temp_ptr, draft_ptr, part_max_ptr, part_sum_ptr, V, NB,
    HAS_TEMP: tl.constexpr, HAS_DRAFT: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offs = block * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(logits_ptr + row.to(tl.int64) * stride_row + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
    if HAS_TEMP:
        x = x / tl.load(temp_ptr + row)
    if HAS_DRAFT:
        x = tl.where(offs == tl.load(draft_ptr + row), float("-inf"), x)
    m = tl.max(x, 0)
    s = tl.sum(tl.exp(x - m), 0)
    s = tl.where(m == float("-inf"), 0.0, s)
    tl.store(part_max_ptr + row * NB + block, m)
    tl.store(part_sum_ptr + row * NB + block, s)


@triton.jit
def _select_kernel(
    logits_ptr, stride_row, temp_ptr, draft_ptr, part_max_ptr, part_sum_ptr, seed, counter_ptr,
    token_ptr, logprob_ptr, logz_ptr, draft_logprob_ptr, accepted_ptr, V, NB,
    HAS_TEMP: tl.constexpr, HAS_DRAFT: tl.constexpr, BLOCK: tl.constexpr, NB_P2: tl.constexpr,
):
    row = tl.program_id(0)
    row_ptr = logits_ptr + row.to(tl.int64) * stride_row
    step = tl.load(counter_ptr).to(tl.uint32)
    temp = 1.0
    if HAS_TEMP:
        temp = tl.load(temp_ptr + row)

    # The row's mass per block (relative to the largest block max), without the draft
    blocks = tl.arange(0, NB_P2)
    part_max = tl.load(part_max_ptr + row * NB + blocks, mask=blocks < NB, other=float("-inf"))
    part_sum = tl.load(part_sum_ptr + row * NB + blocks, mask=blocks < NB, other=0.0)
    m = tl.max(part_max, 0)
    mass = tl.where(part_sum > 0, part_sum * tl.exp(part_max - m), 0.0)
    total = tl.sum(mass, 0)

    # Log of the normalizer over every token, the draft added back
    draft = -1
    if HAS_DRAFT:
        draft = tl.load(draft_ptr + row)
        has_draft = (draft >= 0) & (draft < V)
        x_draft = tl.load(row_ptr + tl.where(has_draft, draft, 0)).to(tl.float32)
        if HAS_TEMP:
            x_draft = x_draft / temp
        x_draft = tl.where(has_draft, x_draft, float("-inf"))
        m_all = tl.maximum(m, x_draft)
        logz = m_all + tl.log(total * tl.exp(m - m_all) + tl.exp(x_draft - m_all))
    else:
        logz = m + tl.log(total)

    # Exponential race over the blocks (stream 0), then over the chosen block's tokens (stream 1)
    block_key = tl.where(mass > 0, tl.log(mass) - tl.log(_exponential(seed, blocks, row, 0, step)), float("-inf"))
    chosen = tl.argmax(block_key, 0)
    offs = chosen * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(row_ptr + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
    if HAS_TEMP:
        x = x / temp
    if HAS_DRAFT:
        x = tl.where(offs == draft, float("-inf"), x)
    score = x - tl.log(_exponential(seed, offs, row, 1, step))
    j = tl.argmax(score, 0)
    token = tl.minimum(chosen * BLOCK + j, V - 1).to(tl.int64)
    logprob = tl.max(tl.where(tl.arange(0, BLOCK) == j, x, float("-inf")), 0) - logz

    if HAS_DRAFT:
        # Ratio test of the draft (stream 2): accepted with probability p(draft), else the residual sample above
        draft_logprob = x_draft - logz
        accepted = has_draft & (_uniform(seed, 0, row, 2, step) < tl.exp(draft_logprob))
        token = tl.where(accepted, draft.to(tl.int64), token)
        logprob = tl.where(accepted, draft_logprob, logprob)
        tl.store(draft_logprob_ptr + row, draft_logprob)
        tl.store(accepted_ptr + row, accepted)
    tl.store(token_ptr + row, token)
    tl.store(logprob_ptr + row, logprob)
    tl.store(logz_ptr + row, logz)


class FusedSampler:
    """Samples one token per row of `logits` ([rows, vocab], bf16 or fp32, vocab contiguous) from softmax(logits / T)
    and returns `(tokens, logprobs, logz, draft_logprobs, accepted)`: the token, log softmax(logits / T)[token], the
    log of the row's normalizer over every token. With `draft` ([rows], -1 where none), a row's draft is accepted with
    probability p(draft) and emitted, else the token is a sample of p without the draft; `draft_logprobs` and
    `accepted` are then filled. `temperature` is per row ([rows] fp32). Random numbers come from Philox keyed by `seed`
    and a device-side step counter advanced after every call (CUDA-graph safe; every tensor-parallel rank holds the
    same seed and the same gathered logits, so every rank samples the same tokens)."""

    BLOCK = 2048
    PARTIALS_WARPS = 4
    SELECT_WARPS = 4

    def __init__(self, device: torch.device, seed: int):
        self.seed = seed & ((1 << 63) - 1)
        self.counter = torch.zeros(1, dtype=torch.int64, device=device)

    def __call__(self, logits: torch.Tensor, temperature: torch.Tensor | None = None, draft: torch.Tensor | None = None):
        rows, vocab = logits.shape
        assert logits.stride(1) == 1 and vocab < 2**31
        block = self.BLOCK
        num_blocks = triton.cdiv(vocab, block)
        device = logits.device
        part_max = torch.empty(rows, num_blocks, dtype=torch.float32, device=device)
        part_sum = torch.empty_like(part_max)
        tokens = torch.empty(rows, dtype=torch.int64, device=device)
        logprobs = torch.empty(rows, dtype=torch.float32, device=device)
        logz = torch.empty_like(logprobs)
        draft_logprobs = torch.empty_like(logprobs) if draft is not None else None
        accepted = torch.empty(rows, dtype=torch.bool, device=device) if draft is not None else None
        if rows == 0:
            return tokens, logprobs, logz, draft_logprobs, accepted
        temp = temperature if temperature is not None else logprobs  # unused pointers when the flags are off
        draft_arg = draft if draft is not None else tokens
        flags = {"HAS_TEMP": temperature is not None, "HAS_DRAFT": draft is not None, "BLOCK": block}
        _block_partials_kernel[(rows, num_blocks)](
            logits, logits.stride(0), temp, draft_arg, part_max, part_sum, vocab, num_blocks, **flags,
            num_warps=self.PARTIALS_WARPS,
        )
        _select_kernel[(rows,)](
            logits, logits.stride(0), temp, draft_arg, part_max, part_sum, self.seed, self.counter, tokens, logprobs,
            logz, draft_logprobs if draft is not None else logz, accepted if draft is not None else tokens, vocab,
            num_blocks, **flags, NB_P2=triton.next_power_of_2(num_blocks), num_warps=self.SELECT_WARPS,
        )
        self.counter.add_(1)
        return tokens, logprobs, logz, draft_logprobs, accepted
