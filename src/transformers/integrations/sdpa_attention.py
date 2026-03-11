import torch
import torch.nn.functional as F

from ..utils import is_torch_npu_available, is_torch_xpu_available, logging
from ..utils.import_utils import is_torch_greater_or_equal


logger = logging.get_logger(__name__)


_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)
_is_torch_greater_or_equal_than_2_11 = is_torch_greater_or_equal("2.11", accept_dev=True)
_is_torch_greater_or_equal_than_2_12 = is_torch_greater_or_equal("2.12", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()
_is_torch_npu_available = is_torch_npu_available()


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


def use_gqa_in_sdpa(attention_mask: torch.Tensor | None, key: torch.Tensor) -> bool:
    # GQA can only be used under the following conditions
    # 1.cuda or Ascend NPU
    #   - torch version >= 2.5
    #   - attention_mask is None (otherwise it will fall back to the math kernel)
    # 2.xpu
    #   - torch version >= 2.8
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None


def _apply_mps_fixes(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int | None]:
    """Apply workarounds for known MPS SDPA bugs in PyTorch.

    Returns the (possibly modified) query, key, value, attention_mask, and the original
    value head dim if padding was applied (None otherwise).

    Fixes two upstream PyTorch issues:

    1. pytorch/pytorch#176767 (fixed in PyTorch 2.12): scaled_dot_product_attention returns
       incorrect results and corrupted output when value head dim differs from query head dim
       on MPS. Workaround: pad value to match query head dim, then slice output after SDPA.

    2. pytorch/pytorch#174861 (fixed in PyTorch 2.11): silent correctness issue in bidirectional
       attention on MPS with non-float32 dtypes, no mask or bool mask, non-causal mode, and
       specific head dim / sequence length conditions. Workaround: provide a zeros attention
       mask to force the sdpa_general_mps code path instead of the broken sdpa_vector_2pass_mps.
    """
    original_v_head_dim = None

    # Fix 1: pad value tensor when v head dim != q head dim (pytorch/pytorch#176767)
    if not _is_torch_greater_or_equal_than_2_12:
        q_head_dim = query.shape[-1]
        v_head_dim = value.shape[-1]
        if v_head_dim != q_head_dim:
            original_v_head_dim = v_head_dim
            value = F.pad(value, (0, q_head_dim - v_head_dim))

    # Fix 2: force non-bool mask for bidirectional attention (pytorch/pytorch#174861)
    # Only applies to non-causal attention (is_causal will be handled separately).
    if _is_torch_greater_or_equal_than_2_8 and not _is_torch_greater_or_equal_than_2_11:
        if not is_causal and query.dtype != torch.float32:
            if attention_mask is None or attention_mask.dtype == torch.bool:
                attention_mask = torch.zeros(
                    (query.shape[0], 1, query.shape[2], key.shape[2]),
                    dtype=query.dtype,
                    device=query.device,
                )

    return query, key, value, attention_mask, original_v_head_dim


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
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)

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
    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal

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

    original_v_head_dim = None
    if query.device.type == "mps":
        query, key, value, attention_mask, original_v_head_dim = _apply_mps_fixes(
            query, key, value, attention_mask, is_causal
        )

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

    if original_v_head_dim is not None:
        attn_output = attn_output[..., :original_v_head_dim]

    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
