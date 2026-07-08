import inspect

import torch

from ..masking_utils import find_packed_sequence_indices
from ..modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids
from ..utils import is_torch_npu_available, is_torch_xpu_available, logging
from ..utils.import_utils import is_torch_greater_or_equal


logger = logging.get_logger(__name__)


_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)
_is_torch_greater_or_equal_than_2_10 = is_torch_greater_or_equal("2.10", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()
_is_torch_npu_available = is_torch_npu_available()

# `varlen_attn` runs true variable-length attention from `cu_seqlens`, so packed (multi-document) rows can skip
# the O(seq_len^2) block-diagonal mask. Guarded import: unavailable below torch 2.10.
try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:
    varlen_attn = None

_is_torch_varlen_attn_available = _is_torch_greater_or_equal_than_2_10 and varlen_attn is not None

# `varlen_attn`'s signature varies across torch>=2.10 builds; inspect the optional kwargs it supports once.
# `window_size` expresses sliding windows (a build with only `is_causal` cannot, so it refuses those layers),
# `enable_gqa` avoids manually repeating K/V, and `scale` passes a custom softmax scale through.
_varlen_attn_parameters = set(inspect.signature(varlen_attn).parameters) if _is_torch_varlen_attn_available else set()
_varlen_attn_supports_window_size = "window_size" in _varlen_attn_parameters
_varlen_attn_supports_enable_gqa = "enable_gqa" in _varlen_attn_parameters
_varlen_attn_supports_scale = "scale" in _varlen_attn_parameters
_varlen_attn_supports_is_causal = "is_causal" in _varlen_attn_parameters

# head_dim limit of the flash-attention kernel backing `varlen_attn`.
_VARLEN_MAX_HEAD_DIM = 256


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def use_gqa_in_sdpa(attention_mask: torch.Tensor | None, key: torch.Tensor, value: torch.Tensor) -> bool:
    # GQA can only be used under the following conditions
    # 1.cuda or Ascend NPU
    #   - torch version >= 2.5
    #   - attention_mask is None (otherwise it will fall back to the math kernel)
    #   - key head_dim == value head_dim <= 256 (otherwise it will fall back to the math kernel)
    # 2.xpu
    #   - torch version >= 2.8
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None and key.shape[-1] == value.shape[-1] <= 256


def _is_packed_row(position_ids: torch.Tensor) -> bool:
    """Whether `position_ids` packs more than one document per row, using the same detection as
    `masking_utils` so the two files agree on which rows are packed."""
    pid = position_ids if position_ids.dim() > 1 else position_ids[None]
    return find_packed_sequence_indices(pid) is not None


def _packed_block_diagonal_mask(
    position_ids: torch.Tensor, seq_len: int, sliding_window: int | None = None
) -> torch.Tensor:
    """
    Build a boolean `(batch, 1, seq_len, seq_len)` block-diagonal causal mask from `position_ids` (which reset to
    0 at each document boundary). Fallback for packed rows ineligible for the `varlen_attn` fast path, so stock
    SDPA still cannot attend across documents.
    """
    pid = position_ids if position_ids.dim() > 1 else position_ids[None]
    document_ids = (pid == 0).cumsum(-1)
    same_document = document_ids[:, :, None] == document_ids[:, None, :]
    token_idx = torch.arange(seq_len, device=pid.device)
    causal = token_idx[:, None] >= token_idx[None, :]
    mask = same_document & causal[None]
    if sliding_window is not None:
        mask = mask & (token_idx[:, None] - token_idx[None, :] < sliding_window)
    return mask[:, None]


def _sdpa_varlen_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position_ids: torch.Tensor,
    scaling: float | None,
    sliding_window: int | None,
) -> tuple[torch.Tensor, None]:
    """
    Run a packed row through `varlen_attn`, deriving `cu_seqlens` from `position_ids` via
    `prepare_fa_kwargs_from_position_ids`. Skips materializing the O(seq_len^2) mask and the compute on its
    masked-out cross-document blocks.
    """
    batch_size, num_heads_q, seq_len, head_dim = query.shape
    num_heads_kv = key.shape[1]

    varlen_kwargs = {}
    if num_heads_q != num_heads_kv:
        if _varlen_attn_supports_enable_gqa:
            varlen_kwargs["enable_gqa"] = True
        else:
            key = repeat_kv(key, num_heads_q // num_heads_kv)
            value = repeat_kv(value, num_heads_q // num_heads_kv)

    if _varlen_attn_supports_scale:
        varlen_kwargs["scale"] = scaling
    if _varlen_attn_supports_window_size:
        # (left, right): (-1, 0) = causal; (W-1, 0) = causal sliding window of W.
        varlen_kwargs["window_size"] = (sliding_window - 1, 0) if sliding_window else (-1, 0)
    elif _varlen_attn_supports_is_causal:
        # The caller already refuses sliding-window layers on an `is_causal`-only build.
        varlen_kwargs["is_causal"] = True

    (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(position_ids)

    q = query.transpose(1, 2).reshape(batch_size * seq_len, num_heads_q, head_dim)
    k = key.transpose(1, 2).reshape(batch_size * seq_len, key.shape[1], head_dim)
    v = value.transpose(1, 2).reshape(batch_size * seq_len, value.shape[1], head_dim)

    attn_output = varlen_attn(
        q,
        k,
        v,
        cu_seq_lens_q.to(torch.int32),
        cu_seq_lens_k.to(torch.int32),
        int(max_length_q),
        int(max_length_k),
        **varlen_kwargs,
    )
    if isinstance(attn_output, tuple):
        attn_output = attn_output[0]

    # match sdpa_attention_forward's return contract: (attn_output [batch, seq, heads, head_dim], None)
    return attn_output.reshape(batch_size, seq_len, num_heads_q, head_dim), None


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True`."
            " Please set your attention to `eager` if you want any of these features."
        )

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)

    # Mask creation may skip materialization for a packed row and pass `attention_mask=None` (see
    # `allow_torch_varlen_skip` in `masking_utils.py`). We re-detect packing from `position_ids` and own document
    # isolation for such a row: either the `varlen_attn` fast path, or a locally-rebuilt block-diagonal mask
    # before stock SDPA. A packed row must never reach SDPA with `attn_mask=None, is_causal=True`, or it would
    # attend across document boundaries.
    position_ids = kwargs.get("position_ids")
    sliding_window = kwargs.get("sliding_window") or getattr(module, "sliding_window", None)
    packed_row = is_causal and attention_mask is None and position_ids is not None and _is_packed_row(position_ids)

    if packed_row:
        head_dim = query.shape[-1]
        standard_scale = _varlen_attn_supports_scale or scaling is None or abs(scaling - head_dim**-0.5) < 1e-9
        use_varlen = (
            _is_torch_varlen_attn_available
            and (_varlen_attn_supports_window_size or _varlen_attn_supports_is_causal)
            and dropout == 0.0
            and head_dim <= _VARLEN_MAX_HEAD_DIM
            and query.is_cuda
            and standard_scale
            and (not sliding_window or _varlen_attn_supports_window_size)
        )
        if use_varlen:
            return _sdpa_varlen_attention_forward(
                query, key, value, position_ids, scaling=scaling, sliding_window=sliding_window
            )
        # Ineligible for the fast path (head_dim too large, dropout, non-CUDA tensors, or an inexpressible
        # sliding window): rebuild the isolation the skipped mask no longer provides.
        attention_mask = _packed_block_diagonal_mask(position_ids, query.shape[2], sliding_window)

    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
        if not use_gqa_in_sdpa(attention_mask, key, value):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    q_length = query.shape[2]
    kv_length = key.shape[2]

    # SDPA's Flash Attention (and cuDNN) kernels rely on the `is_causal` flag. However, there are certain conditions:
    # - Not in decoding phase (otherwise we want full attention on the single query token)
    # - Attention mask is not to be provided (even if it is a causal pattern)
    # - Internally, we marked this as compatible with causal, i.e. it is a decoder attention type
    #
    # Quirks on the conditionals:
    # - We avoid inline passing this to the SDPA function directly to support both torch.compile's dynamic shapes and
    #   full graph options. Otherwise, dynamic shapes are prevented from compiling.
    # - It is important to check first for the shape, otherwise compile will fail with
    #   `argument 'is_causal' must be bool, not SymBool`.
    is_causal = q_length > 1 and attention_mask is None and is_causal

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # When `is_causal = False` and the `attention_mask` is not of boolean type, the Ascend NPU's SDPA interface cannot utilize the FlashAttentionScore operator，
    # and falls back to small-operator concatenation. To invoke the FlashAttentionScore, the attention_mask must be converted to boolean type.
    # This adaptation ensures the `attention_mask` meets the requirement for using FlashAttentionScore.
    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # Convert to boolean type, making sdpa to force call FlashAttentionScore to improve performance.
            attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)

    # This scenario can only happen during prefill with an empty StaticCache. Technically, since sdpa's `is_causal` mask alignment
    # is upper-left, `is_causal=True` is enough to correctly compute the attention. However, sdpa will only dispatch to
    # flash kernel if and only if q_length == kv_length, therefore it is more efficient to slice here and remove the masked tokens
    # rather than to use the other available kernels for such a case.
    # Note that we never compile prefill, and even if the user is doing it on its own, prefill and decode are 2 separate graphs
    # anyway, so altering the shapes is fine here
    if is_causal and attention_mask is None and q_length > 1 and kv_length > q_length:
        key = key[:, :, :q_length, :]
        value = value[:, :, :q_length, :]

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
