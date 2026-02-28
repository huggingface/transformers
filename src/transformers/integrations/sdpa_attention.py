import torch

from ..utils import is_torch_npu_available, is_torch_xpu_available, logging
from ..utils.import_utils import is_torch_greater_or_equal


logger = logging.get_logger(__name__)


_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)
_is_torch_greater_or_equal_than_2_11 = is_torch_greater_or_equal("2.11", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()
_is_torch_npu_available = is_torch_npu_available()
# PyTorch MPS has a silent correctness bug in sdpa_vector_2pass_mps (pytorch/pytorch#174861)
# affecting versions 2.8.0 through 2.10.x. The bug produces wrong results under specific conditions.
# Fixed upstream in PyTorch 2.11.0 (pytorch/pytorch#174945).
_mps_sdpa_bug_affected = _is_torch_greater_or_equal_than_2_8 and not _is_torch_greater_or_equal_than_2_11
# Head dimensions that trigger the buggy sdpa_vector_2pass_mps kernel
_MPS_SDPA_BUG_HEAD_DIMS = {64, 96, 128}


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


def _needs_mps_sdpa_workaround(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor | None,
    is_causal: bool,
) -> bool:
    """
    Check if the current SDPA call hits the PyTorch MPS correctness bug (pytorch/pytorch#174861).

    The bug is in ``sdpa_vector_2pass_mps`` which is selected when:
    - dtype is not float32
    - mask is None or boolean (i.e. not a float attention mask)
    - non-causal attention
    - query_len <= 8 and query_len <= key_len
    - head_dim in {64, 96, 128}
    - key_len >= 1024, or (num_key_heads < num_query_heads and key_len >= 4096)

    Returns True if the workaround (forcing a float mask) is needed.
    """
    if not _mps_sdpa_bug_affected:
        return False
    if query.device.type != "mps":
        return False
    if query.dtype == torch.float32:
        return False
    if is_causal:
        return False
    # Mask must be None or boolean for the bug to trigger
    if attention_mask is not None and attention_mask.dtype != torch.bool:
        return False

    query_len = query.shape[2]
    key_len = key.shape[2]
    head_dim = query.shape[3]

    if query_len > 8:
        return False
    if query_len > key_len:
        return False
    if head_dim not in _MPS_SDPA_BUG_HEAD_DIMS:
        return False

    num_query_heads = query.shape[1]
    num_key_heads = key.shape[1]

    if key_len >= 1024:
        return True
    if num_key_heads < num_query_heads and key_len >= 4096:
        return True

    return False


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

    # When `is_causal = False` and the `attention_mask` is not of boolean type, the Ascend NPU's SDPA interface cannot utilize the FlashAttentionScore operator,
    # and falls back to small-operator concatenation. To invoke the FlashAttentionScore, the attention_mask must be converted to boolean type.
    # This adaptation ensures the `attention_mask` meets the requirement for using FlashAttentionScore.
    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # Convert to boolean type, making sdpa to force call FlashAttentionScore to improve performance.
            attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)

    # Workaround for PyTorch MPS silent correctness bug in sdpa_vector_2pass_mps
    # (pytorch/pytorch#174861). Forcing a float attention mask routes through sdpa_general_mps
    # which does not have the bug. See huggingface/transformers#44247.
    if _needs_mps_sdpa_workaround(query, key, attention_mask, is_causal):
        logger.warning_once(
            "Applying MPS SDPA workaround for PyTorch < 2.11 correctness bug "
            "(pytorch/pytorch#174861). Forcing float attention mask to avoid "
            "sdpa_vector_2pass_mps. Upgrade to PyTorch >= 2.11 to remove this workaround."
        )
        # Create a zero-filled float mask with the same dtype as query to force
        # PyTorch to use sdpa_general_mps instead of the buggy sdpa_vector_2pass_mps.
        # A zero-filled additive mask is semantically equivalent to no mask.
        batch_size = query.shape[0]
        query_len = query.shape[2]
        key_len = key.shape[2]
        attention_mask = torch.zeros(
            (batch_size, 1, query_len, key_len), dtype=query.dtype, device=query.device
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
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
