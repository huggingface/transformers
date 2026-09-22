# Copyright 2025 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
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
import importlib
import inspect
import os
from collections.abc import Callable
from functools import partial
from typing import TypedDict

import torch
import torch.nn.functional as F

from .utils import (
    is_flash_attn_2_available,
    is_flash_attn_3_available,
    is_flash_attn_4_available,
    is_rocm_platform,
    is_torch_cuda_available,
    is_torch_mlu_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xpu_available,
    logging,
)
from .utils.generic import split_attention_implementation
from .utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING, is_tracing


logger = logging.get_logger(__name__)


# TODO Deprecate when all models have the attention interface
def flash_attn_supports_top_left_mask():
    if is_flash_attn_2_available() or is_flash_attn_3_available() or is_flash_attn_4_available():
        return False

    from .integrations.npu_flash_attention import is_npu_fa2_top_left_aligned_causal_mask

    return is_npu_fa2_top_left_aligned_causal_mask()


# TODO Deprecate when all models have the attention interface
def is_flash_attn_available():
    return (
        is_flash_attn_4_available()
        or is_flash_attn_3_available()
        or is_flash_attn_2_available()
        or is_torch_npu_available()
        or is_torch_xpu_available()
    )


# Mapping from flash attention implementations to their kernel fallback repositories.

FLASH_ATTN_KERNEL_FALLBACK = {
    "flash_attention_2": "kernels-community/flash-attn2",
    "flash_attention_3": (
        "kernels-community/aiter-flash-attn" if is_rocm_platform() else "kernels-community/vllm-flash-attn3"
    ),
    "flash_attention_4": "kernels-community/flash-attn4",
}


# Meta information on each mainline FA compatibility:
#   1. The import structure and availability
#   2. Device support (with custom ones that use other workarounds, e.g. kernels)
#   3. Supported major cuda devices, e.g. Hopper, Blackwell. Mostly found in the newest FA versions
FLASH_ATTENTION_COMPATIBILITY_MATRIX = {
    2: {
        "flash_attn_version": 2,
        "general_availability_check": is_flash_attn_2_available,
        "pkg_availability_check": lambda *args, **kwargs: (
            importlib.util.find_spec("flash_attn") is not None
            and "flash-attn" in [pkg.replace("_", "-") for pkg in PACKAGE_DISTRIBUTION_MAPPING.get("flash_attn", [])]
        ),
        "supported_devices": (
            (is_torch_cuda_available, "cuda"),
            (is_torch_mlu_available, "mlu"),
            (is_torch_musa_available, "musa"),
            (is_torch_npu_available, "npu"),
            (is_torch_xpu_available, "xpu"),
        ),
        "custom_supported_devices": (
            (is_torch_npu_available, "Detect using FlashAttention2 on Ascend NPU."),
            (
                is_torch_xpu_available,
                f"Detect using FlashAttention2 (via kernel `{FLASH_ATTN_KERNEL_FALLBACK['flash_attention_2']}`) on XPU.",
            ),
        ),
    },
    3: {
        "flash_attn_version": 3,
        "general_availability_check": is_flash_attn_3_available,
        "pkg_availability_check": lambda *args, **kwargs: (
            importlib.util.find_spec("flash_attn_interface") is not None
            and "flash-attn-3"
            in [pkg.replace("_", "-") for pkg in PACKAGE_DISTRIBUTION_MAPPING.get("flash_attn_interface", [])]
        ),
        "supported_devices": ((is_torch_cuda_available, "cuda"),),
        "cuda_min_major_version": 8,  # Ampere
    },
    4: {
        "flash_attn_version": 4,
        "general_availability_check": is_flash_attn_4_available,
        "pkg_availability_check": lambda *args, **kwargs: (
            importlib.util.find_spec("flash_attn") is not None
            and "flash-attn-4" in [pkg.replace("_", "-") for pkg in PACKAGE_DISTRIBUTION_MAPPING.get("flash_attn", [])]
        ),
        "supported_devices": ((is_torch_cuda_available, "cuda"),),
        "cuda_min_major_version": 9,  # Hopper
    },
}


# `globals()` is not compatible with dynamo, hence we have do define them in global scope ourselves
_loaded_implementation = None
_flash_fn = None
_flash_varlen_fn = None
_flash_with_kvcache_fn = None
_pad_fn = None
_unpad_fn = None

# function that processes kwargs, generalized to handle any supported kwarg within the function
_process_flash_kwargs_fn = None
# exceptions where hf API doesn't match the original flash attention API
_hf_api_to_flash_mapping = {
    "dropout": "dropout_p",
    "sliding_window": "window_size",
}
# alternative names within the different flash attention APIs, e.g. for attention sinks
_flash_api_alternative_names = {"s_aux": "learnable_sink", "block_table": "page_table"}


def _lazy_imports(
    implementation: str | None, attention_wrapper: Callable | None = None, allow_all_kernels: bool = False
) -> tuple[Callable, Callable, Callable]:
    """
    Lazy loads the respective flash attention implementations.

    Return:
        flash_attn_func: The base flash attention function.
        flash_attn_varlen_func: The flash attention function supporting variable sequence lengths, e.g. for padding-free
            training.
        flash_attn_with_kvcache: The flash attention function supporting block tables, for inference with paged cache
    """
    is_fa2 = is_flash_attn_2_available()
    is_fa3 = is_flash_attn_3_available()
    is_fa4 = is_flash_attn_4_available()
    fa_fallback_version = 0 if implementation is not None else max(2 * int(is_fa2), 3 * int(is_fa3), 4 * int(is_fa4))

    is_paged, implementation = split_attention_implementation(implementation)

    # Try the flash attention package first
    if (implementation == "flash_attention_2" and is_fa2) or fa_fallback_version == 2:
        from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
    elif is_torch_npu_available():
        from .integrations.npu_flash_attention import npu_flash_attn_func as flash_attn_func
        from .integrations.npu_flash_attention import npu_flash_attn_varlen_func as flash_attn_varlen_func
        from .integrations.npu_flash_attention import npu_flash_attn_with_kvcache as flash_attn_with_kvcache
    elif implementation == "flash_attention_3" or fa_fallback_version == 3:
        from flash_attn_interface import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
    elif implementation == "flash_attention_4" or fa_fallback_version == 4:
        from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

        flash_attn_with_kvcache = None  # not supported yet

    # Otherwise, use the `kernels` package as a fallback
    else:
        from .integrations.hub_kernels import load_and_register_attn_kernel

        # Map standard attention names to hub kernel repos
        kernel_repo = FLASH_ATTN_KERNEL_FALLBACK.get(implementation, implementation)
        # We want to explicitly register the name with `paged|` if found
        kernel_implementation = f"paged|{implementation}" if is_paged else kernel_repo
        kernel = load_and_register_attn_kernel(kernel_implementation, attention_wrapper, allow_all_kernels)

        flash_attn_func = getattr(kernel, "flash_attn_func", None)
        flash_attn_varlen_func = getattr(kernel, "flash_attn_varlen_func", None)
        flash_attn_with_kvcache = getattr(kernel, "flash_attn_with_kvcache", None)
        # Some kernels ship their own attention entry point rather than a varlen function, already
        # registered into ``ALL_ATTENTION_FUNCTIONS``, so preloading them here is a no-op.
        if flash_attn_varlen_func is None and (
            hasattr(kernel, "sparse_atten_func") or hasattr(kernel, "flash_attn_forward")
        ):
            return flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
        if flash_attn_varlen_func is None:
            raise ValueError(
                f"Could not find the currently requested flash attention implementation at `{implementation}`."
                "Make sure that you request a valid kernel from the hub, e.g. `kernels-community/flash-attn2`."
            )
        if flash_attn_func is None:
            logger.warning(
                f"The loaded flash attention implementation at `{implementation}` only supports varlen, i.e. "
                "it can only be used with continuous batching and does not support the full functionality for "
                "the base transformers generation methods."
            )
        if flash_attn_with_kvcache is None:
            logger.warning(
                f"The loaded flash attention implementation at `{implementation}` does not support block tables, so"
                " the full performances of continuous batching will not be achieved, only the varlen path will be "
                "used."
            )

    return flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache


def _lazy_define_process_function(flash_function):
    """
    Depending on the version and kernel some features are not supported. Due to limitations in
    `torch.compile`, we opt to statically type which (optional) kwarg parameters are supported
    within `_process_flash_attention_kwargs`.

    NOTE: While all supported kwargs are marked as `True`, everything else is marked as `False`.
          This might be confusing for kwargs that we use in any case, e.g. `is_causal`.
    """

    flash_parameters = inspect.signature(flash_function).parameters
    process_parameters = inspect.signature(_process_flash_attention_kwargs).parameters

    supports_mapping = {}
    for param in process_parameters:
        fa_param = _hf_api_to_flash_mapping.get(param, param)
        supports_mapping[fa_param] = fa_param in flash_parameters

        if (fa_alternative_name := _flash_api_alternative_names.get(param, param)) != fa_param:
            supports_mapping[fa_alternative_name] = fa_alternative_name in flash_parameters

    return partial(_process_flash_attention_kwargs, supports_mapping=supports_mapping)


def lazy_import_flash_attention(
    implementation: str | None, attention_wrapper: Callable | None = None, allow_all_kernels: bool = False
) -> tuple[tuple[Callable, Callable, Callable], Callable]:
    """
    Lazily import flash attention and return the respective functions + flags.

    NOTE: For fullgraph, this needs to be called before compile, while no fullgraph can
    work without preloading. See `load_and_register_attn_kernel` in `integrations.hub_kernels`.
    """
    global _loaded_implementation
    if implementation is None and _loaded_implementation is None:
        raise ValueError("Could not find any flash attn implementation based on your environment.")

    global _flash_fn, _flash_varlen_fn, _flash_with_kvcache_fn, _process_flash_kwargs_fn
    if implementation is not None and _loaded_implementation != implementation:
        _loaded_implementation = implementation

        # This is the point where we actually import the flash attention function
        _flash_fn, _flash_varlen_fn, _flash_with_kvcache_fn = _lazy_imports(
            implementation, attention_wrapper, allow_all_kernels=allow_all_kernels
        )

        # Some kernels, like minimax_m3_vl's block spare kernel, have no varlen function. In this case, the varlen path
        # can never be used, so no need to build a processing function for it, just return a dict builder.
        if _flash_varlen_fn is not None:
            _process_flash_kwargs_fn = _lazy_define_process_function(_flash_varlen_fn)
        else:
            _process_flash_kwargs_fn = dict

    return (_flash_fn, _flash_varlen_fn, _flash_with_kvcache_fn), _process_flash_kwargs_fn


def _prepare_unpad_state(
    state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """
    Given a state, with can be queries or keys, computes the variables needed to unpad it and run it through flash
    attention.
    """
    unpadded_indices = attention_mask.nonzero()  # returns a tensor with shape (num_nonzero, 2) of [row, col] indices
    unpadded_indices[:, 0] *= state.shape[1]  # converts the row index to a flattned global index
    unpadded_indices = unpadded_indices.sum(dim=-1)

    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = seqlens.max().item()  # using .item() here is required to prevent a performance regression (#46693)
    return unpadded_indices, cu_seqlens, max_seqlen


def prepare_fa_kwargs_from_attn_mask(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], tuple[int, int]]:
    """
    Prepares the variables needed to use flash attention with padded states. This returns the unpadding indices, the
    cumulative sequence lengths, and the maximum sequence length for the query and key states.
    The query_states and key_states are expected to already be flattened to shape [num__tokens, num_heads, head_dim] and
    the attention mask is a boolean tensor of shape [batch_size, num_kv_tokens].
    """
    batch_size, query_length = query_states.shape[:2]
    indices_k, cu_seqlens_k, max_seqlen_k = _prepare_unpad_state(key_states, attention_mask)
    # For the queries, the kwargs are trivial if there is only one query token (decoding)
    if query_length == 1:
        max_seqlen_q = 1
        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_states.device)
        indices_q = cu_seqlens_q[:-1]
    # Otherwise, we perform the same operation as for the keys
    else:
        indices_q, cu_seqlens_q, max_seqlen_q = _prepare_unpad_state(query_states, attention_mask[:, -query_length:])
    return (indices_q, indices_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)


def prepare_fa_kwargs_from_position_ids(
    position_ids: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[int, int]]:
    """This function returns all the necessary kwargs to call `flash_attn_varlen_func` extracted from position_ids. The
    attention mask is a boolean tensor of shape [batch_size, num_kv_tokens]."""
    position_ids = position_ids.reshape(-1)
    # Packed sequences all restart from the same first position id, but it is not always 0
    # (RoBERTa-like models start at padding_idx + 1)
    indices_q = (position_ids == position_ids.min()).nonzero().view(-1)
    cu_seq_lens_q = torch.cat(
        (
            indices_q.to(dtype=torch.int32, device=position_ids.device),
            torch.tensor(position_ids.size(), dtype=torch.int32, device=position_ids.device),
        )
    )
    # https://github.com/Dao-AILab/flash-attention/blob/2dd8078adc1d9b74e315ee99718c0dea0de8eeb6/flash_attn/flash_attn_interface.py#L1423-L1424
    # We should use cu_seq_lens instead of position_ids to get the max length since position_ids is not always
    # increasing for some models (e.g. qwen2-vl).
    max_seqlen_q = cu_seq_lens_q.diff().max()
    return (cu_seq_lens_q, cu_seq_lens_q), (max_seqlen_q, max_seqlen_q)


def _is_packed_sequence(position_ids: torch.Tensor | None, batch_size: int) -> bool:
    """
    Check the position ids whether packed sequences are indicated or not
        1. Position ids exist
        2. Flattened sequences only are supported
        3. Compile-friendly `not (torch.diff(position_ids, dim=-1) >= 0).all()`, i.e. we have multiple increasing sequences
    """
    if position_ids is None:
        return False

    increasing_position_sequences = (
        torch.arange(position_ids.shape[1], device=position_ids.device) + position_ids.min()
    )
    return batch_size == 1 and (increasing_position_sequences - position_ids).abs().sum().bool()


def cast_to_flash_compatible_dtype(
    module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """If the query is in float32, converts the query, key and value to a dtype compatible with flash attention."""
    # Early exit if the query is not in float32
    if query.dtype != torch.float32:
        return query, key, value

    # Otherwise, look for the right dtype to cast to
    device_type = query.device.type
    if torch.is_autocast_enabled(device_type):
        target_dtype = torch.get_autocast_dtype(device_type)
    # Handle the case where the model is quantized
    elif hasattr(module.config, "_is_quantized"):
        target_dtype = module.config.dtype
    else:
        target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    logger.warning_once(f"Casting fp32 inputs back to {target_dtype} for flash-attn compatibility.")
    return query.to(target_dtype), key.to(target_dtype), value.to(target_dtype)


class FlashAttentionKwargs(TypedDict, total=False):
    """
    Keyword arguments for Flash Attention with Compile.

    Attributes:
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_seqlen_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_seqlen_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    cu_seq_lens_q: torch.LongTensor | None
    cu_seq_lens_k: torch.LongTensor | None
    max_seqlen_q: int | None
    max_seqlen_k: int | None


def _process_flash_attention_kwargs(
    query_length: int,
    key_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    use_top_left_mask: bool = False,
    softcap: float | None = None,
    deterministic: bool | None = None,
    s_aux: torch.Tensor | None = None,
    max_seqlen_q: int | torch.IntTensor | None = None,
    max_seqlen_k: int | torch.IntTensor | None = None,
    block_table: torch.Tensor | None = None,
    supports_mapping: dict[str, bool] | None = None,
    **kwargs,
):
    """
    Returns a set of kwargs that are passed down to the according flash attention function based on
    requested features and whether it is supported - depends on the version and kernel implementation
    which is dynamically configured at `lazy_import_flash_attention`. The (un)supported features can be
    inspected in `supports_mapping`, see `_lazy_define_process_function` for more details.

    Args:
        query_length (`int`):
            Length of the query states
        key_length (`int`):
            Length of the key states
        is_causal (`bool`):
            Whether we perform causal (decoder) attention or full attention.
        dropout (`float`):
            Attention dropout.
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to `1 / sqrt(head_dim)`.
        sliding_window (`int`, *optional*):
            The size of the sliding window, i.e. we look at a max of `sliding_window` tokens back.
        use_top_left_mask (`bool`):
            Deprecated behavior of older versions of flash attention requiring different masking.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
        s_aux (`torch.Tensor`, *optional*):
            Attention sink auxiliary that adds a `bias` to the attention calculation via an additional head.
        max_seqlen_q (`Union[int, torch.IntTensor]`, *optional*):
            The maximum sequence length in the query tensor during a varlen forward.
        max_seqlen_k (`Union[int, torch.IntTensor]`, *optional*):
            The maximum sequence length in the key/value tensor during a varlen forward.
    Return:
        flash_kwargs (`dict`):
            A dict of kwargs that are requested and supported.
    """
    flash_kwargs = {
        "causal": is_causal and not (use_top_left_mask and query_length == 1),
        "softmax_scale": softmax_scale,
    }

    if supports_mapping["dropout_p"]:
        flash_kwargs["dropout_p"] = dropout

    if supports_mapping["window_size"] and sliding_window is not None and key_length > sliding_window:
        # The flash attention API sets inclusive boundaries, i.e. (4, 0) would take 4 tokens to the left
        # and the current token for a total size of 5. However, we usually define our window sizes by
        # their total window size (when causal). Encoder models as of now seldom use SWA and when they
        # do, they must align with this symmetric logic, i.e. for a total of `2*sliding_window + 1`.
        flash_kwargs["window_size"] = (sliding_window - 1, sliding_window - 1)

    if supports_mapping["deterministic"]:
        flash_kwargs["deterministic"] = (
            deterministic if deterministic is not None else os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
        )

    if supports_mapping["softcap"] and softcap is not None:
        flash_kwargs["softcap"] = softcap

    if s_aux is not None:
        if supports_mapping["s_aux"]:
            flash_kwargs["s_aux"] = s_aux
        elif supports_mapping["learnable_sink"]:
            flash_kwargs["learnable_sink"] = s_aux

    # The block table is named `block_table` in Tri Dao's kernels and `page_table` in vLLM's FA3 kernel
    if block_table is not None:
        if supports_mapping["block_table"]:
            flash_kwargs["block_table"] = block_table
        elif supports_mapping["page_table"]:
            flash_kwargs["page_table"] = block_table

    # There is a limitation of the flash attention API, as the function `flash_attn_varlen_func`
    # may require `max_seqlen_q`, `max_seqlen_k` to be passed as `int` and not `torch.Tensor`.
    #
    # You can either set
    #   - Env: `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`
    #   - Before compiling: `torch._dynamo.config.capture_scalar_outputs = True`
    # to allow torch compile to handle scalar outputs in those cases.
    same_max_seqlen = max_seqlen_q is max_seqlen_k  # to avoid 2x device syncs
    if supports_mapping["max_seqlen_q"] and max_seqlen_q is not None:
        if not isinstance(max_seqlen_q, int) and is_tracing(max_seqlen_q):
            max_seqlen_q = max_seqlen_q.item()
        flash_kwargs["max_seqlen_q"] = max_seqlen_q

    if supports_mapping["max_seqlen_k"] and max_seqlen_k is not None:
        if same_max_seqlen and flash_kwargs["max_seqlen_q"] is not None:
            max_seqlen_k = flash_kwargs["max_seqlen_q"]
        elif not isinstance(max_seqlen_k, int) and is_tracing(max_seqlen_k):
            max_seqlen_k = max_seqlen_k.item()
        flash_kwargs["max_seqlen_k"] = max_seqlen_k

    return flash_kwargs


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    query_length: int,
    is_causal: bool,
    position_ids: torch.Tensor | None = None,
    cu_seq_lens_q: torch.LongTensor | None = None,
    cu_seq_lens_k: torch.LongTensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    k_cache: torch.Tensor | None = None,
    v_cache: torch.Tensor | None = None,
    cache_seqlens: torch.LongTensor | None = None,
    block_table: torch.Tensor | None = None,
    attn_implementation: str | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    (Optional) kwargs are described further in `_process_flash_attention_kwargs` and `FlashAttentionKwargs`.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`, *optional*):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        attn_implementation (`str`, *optional*):
            The attention implementation to use. If None, will default to the one based on the environment.
        block_table (`torch.Tensor`, *optional*):
            The block table to use if this is a call to flash_kv_fn, which updates the cache in-place.
    """
    (flash_fn, flash_varlen_fn, flash_kv_fn), process_flash_kwargs_fn = lazy_import_flash_attention(attn_implementation)
    batch_size, key_length = key_states.shape[:2]

    # Extract the flash attention kwargs that have been requested (and are supported by the implementation)
    extract_flash_kwargs = partial(
        process_flash_kwargs_fn, query_length=query_length, key_length=key_length, is_causal=is_causal, **kwargs
    )

    # We use `flash_varlen_fn` to prevent cross-sequence attention and allow padding free approaches under two cases:
    # Case 1. If position ids is provided and the position_ids indicates packed sequences, see `_is_packed_sequence`.
    # Case 2. Some models pass directly pre-computed `cu_seqlens` so we don't need to infer it from position ids.
    #         It is safe to use `flash_varlen_fn` knowing we already have all necessary the kwargs.

    is_fa_with_varlen_kwargs = None not in (cu_seq_lens_q, cu_seq_lens_k, max_seqlen_q, max_seqlen_k)
    is_fa_with_block_table = None not in (k_cache, v_cache, cache_seqlens, block_table)

    # If there is no padding and the sequence are not packed, we can just run flash and return
    if attention_mask is None and not (is_fa_with_varlen_kwargs or is_fa_with_block_table):
        # This check is more compute heavy, so it is separate from the rest. Also, it's a user's responsibility to take
        # care of flattening `position_ids` if that's needed by the model. See #39121 for more information.
        if not _is_packed_sequence(position_ids, batch_size):
            out = flash_fn(query_states, key_states, value_states, **extract_flash_kwargs())
            return out[0] if isinstance(out, tuple) else out

    # Flattens the batch dimension, which does not exist in varlen or with block table
    query_states, key_states, value_states = [
        x.view(-1, *x.shape[2:]) for x in (query_states, key_states, value_states)
    ]
    # Block table has a singleton dimension to align with the cache though
    if is_fa_with_block_table:
        query_states, key_states, value_states = [x.unsqueeze(1) for x in (query_states, key_states, value_states)]

    # If they have not been provided, compute the sequence-defining attributes
    if attention_mask is not None:
        (indices_q, indices_k), (cu_seq_lens_q, cu_seq_lens_k), (max_seqlen_q, max_seqlen_k) = (
            prepare_fa_kwargs_from_attn_mask(attention_mask, query_length, key_length)
        )
        query_states = query_states[indices_q]  # unpadding
        key_states, value_states = key_states[indices_k], value_states[indices_k]
    elif not (is_fa_with_varlen_kwargs or is_fa_with_block_table):
        (cu_seq_lens_q, cu_seq_lens_k), (max_seqlen_q, max_seqlen_k) = prepare_fa_kwargs_from_position_ids(position_ids)

    flash_kwargs = extract_flash_kwargs(max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k, block_table=block_table)

    # Compute the right seq_lens objects and call flash
    if is_fa_with_block_table:
        flash_kwargs["cache_seqlens"] = cache_seqlens
        out = flash_kv_fn(query_states, k_cache, v_cache, key_states, value_states, **flash_kwargs)

    else:
        flash_kwargs["cu_seqlens_q"] = cu_seq_lens_q
        flash_kwargs["cu_seqlens_k"] = cu_seq_lens_k.clone()  # type: ignore | not cloning crashes on MPS and on CUDA
        out = flash_varlen_fn(query_states, key_states, value_states, **flash_kwargs)

    out = out[0] if isinstance(out, tuple) else out

    # If there was an attention mask, restore the padding
    if attention_mask is not None:
        padded_out = torch.zeros((batch_size * query_length, out.shape[1:]), device=out.device, dtype=out.dtype)
        padded_out[indices_q] = out
        return padded_out.view(batch_size, query_length, *out.shape[1:])

    return out.view(batch_size, -1, *out.shape[1:])
