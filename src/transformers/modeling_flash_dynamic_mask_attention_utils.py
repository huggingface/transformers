# Copyright 2025 Jingze Shi and the HuggingFace Inc. team. All rights reserved.
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
import inspect
import os
from functools import partial
from typing import Optional

import torch

from .utils import (
    is_flash_dmattn_available,
    logging,
)


logger = logging.get_logger(__name__)


# `globals()` is not compatible with dynamo, hence we have do define them in global scope ourselves
_fdma_fn = None

# function that processes kwargs, generalized to handle any supported kwarg within the function
_process_flash_kwargs_fn = None
# exceptions where hf API doesn't match the original FDMA API
_hf_api_to_flash_mapping = {
    "dropout": None,
    "sliding_window": None,
}


def _lazy_imports(implementation: Optional[str]):
    """
    Lazy loads the respective flash dynamic mask attention implementations.

    Return:
        flash_dmattn_func: The base flash dynamic mask attention function.
    """
    is_fdma = is_flash_dmattn_available()

    if (implementation == "flash_dynamic_mask_attention" and is_fdma) or (implementation is None and is_fdma):
        from flash_dmattn import flash_dmattn_func

    return flash_dmattn_func


def _lazy_define_process_function(flash_function):
    """
    Depending on the version and kernel some features are not supported. Due to limitations in
    `torch.compile`, we opt to statically type which (optional) kwarg parameters are supported
    within `_process_flash_dynamic_mask_attention_kwargs`.

    NOTE: While all supported kwargs are marked as `True`, everything else is marked as `False`.
          This might be confusing for kwargs that we use in any case, e.g. `is_causal`.
    """

    flash_parameters = inspect.signature(flash_function).parameters
    process_parameters = inspect.signature(_process_flash_dynamic_mask_attention_kwargs).parameters

    supports_mapping = {}
    for param in process_parameters:
        fdma_param = _hf_api_to_flash_mapping.get(param, param)
        supports_mapping[fdma_param] = fdma_param in flash_parameters

    return partial(_process_flash_dynamic_mask_attention_kwargs, supports_mapping=supports_mapping)


def lazy_import_flash_dynamic_mask_attention(implementation: Optional[str], force_import: Optional[bool] = False):
    """
    Lazily import flash dmattn and return the respective functions + flags.

    NOTE: For fullgraph, this needs to be called before compile, while no fullgraph can
    work without preloading. See `load_and_register_kernel` in `integrations.hub_kernels`.
    """
    global _fdma_fn
    if force_import or any(k is None for k in [_fdma_fn]):
        _fdma_fn = _lazy_imports(implementation)

    global _process_flash_kwargs_fn
    if force_import or _process_flash_kwargs_fn is None:
        _process_flash_kwargs_fn = _lazy_define_process_function(_fdma_fn)

    return (_fdma_fn), _process_flash_kwargs_fn


def fdma_peft_integration_check(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: Optional[torch.Tensor],
    target_dtype: Optional[torch.dtype] = None,
):
    """
    PEFT usually casts the layer norms in float32 for training stability reasons
    therefore the input hidden states gets silently casted in float32. Hence, we need
    cast them back in float16 / bfloat16 just to be sure everything works as expected.
    This might slowdown training & inference so it is recommended to not cast the LayerNorms!
    """
    if target_dtype and q.dtype == torch.float32:
        logger.warning_once(f"Casting fp32 inputs back to {target_dtype} for flash-dmattn compatibility.")
        q, k, v = q.to(target_dtype), k.to(target_dtype), v.to(target_dtype)
        if bias is not None:
            bias = bias.to(target_dtype)
    return q, k, v, bias


def _process_flash_dynamic_mask_attention_kwargs(
    query_length: int,
    key_length: int,
    is_causal: bool,
    softmax_scale: Optional[float] = None,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    s_aux: Optional[torch.Tensor] = None,
    supports_mapping: Optional[dict[str, bool]] = None,
    **kwargs,
):
    """
    Returns a set of kwargs that are passed down to the according flash dynamic mask attention function based on
    requested features and whether it is supported - depends on the version and kernel implementation
    which is dynamically configured at `lazy_import_flash_dynamic_mask_attention`. The (un)supported features can be
    inspected in `supports_mapping`, see `_lazy_define_process_function` for more details.

    Args:
        query_length (`int`):
            Length of the query states
        key_length (`int`):
            Length of the key states
        is_causal (`bool`):
            Whether we perform causal (decoder) attention or full attention.
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to `1 / sqrt(head_dim)`.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_dmattn>=1.0.0 is enabled.
        s_aux (`torch.Tensor`, *optional*):
            Attention sink auxiliary that adds a `bias` to the attention calculation via an additional head.
    Return:
        flash_kwargs (`dict`):
            A dict of kwargs that are requested and supported.
    """
    flash_kwargs = {
        "is_causal": is_causal and query_length != 1,
        "softmax_scale": softmax_scale,
    }

    if supports_mapping["deterministic"]:
        flash_kwargs["deterministic"] = (
            deterministic if deterministic is not None else os.getenv("FLASH_DMATTN_DETERMINISTIC", "0") == "1"
        )

    if supports_mapping["softcap"] and softcap is not None:
        flash_kwargs["softcap"] = softcap

    # Only within kernel implementation atm
    if supports_mapping["s_aux"] and s_aux is not None:
        flash_kwargs["s_aux"] = s_aux

    return flash_kwargs


def _flash_dynamic_mask_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    attention_bias: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool,
    softmax_scale: Optional[float] = None,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    target_dtype: Optional[torch.dtype] = None,
    implementation: Optional[str] = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Dynamic Mask Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    (Optional) kwargs are described further in `_process_flash_dynamic_mask_attention_kwargs` and `FlashDynamicMaskAttentionKwargs`.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Dynamic Mask Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Dynamic Mask Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Dynamic Mask Attention API
        attention_mask (`torch.Tensor`, *optional*):
            The padding mask - corresponds to a tensor of size `(batch_size, num_heads, query_len, key_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        attention_bias (`torch.Tensor`, *optional*):
            The attention bias tensor of size `(batch_size, num_heads, query_len, key_len)` to add to attention scores.
        implementation (`str`, *optional*):
            The attention implementation to use. If None, will default to the one based on the environment.
    """

    (fdma_fn), process_flash_kwargs_fn = lazy_import_flash_dynamic_mask_attention(implementation)

    # PEFT possibly silently casts tensors to fp32, this potentially reconverts to correct dtype or is a no op
    query_states, key_states, value_states, attention_bias = fdma_peft_integration_check(
        query_states, key_states, value_states, attention_bias, target_dtype
    )

    # Extract the flash dynamic mask attention kwargs that have been requested (and are supported by the implementation)
    flash_kwargs = process_flash_kwargs_fn(
        query_length=query_length,
        key_length=key_states.size(1),
        is_causal=is_causal,
        softmax_scale=softmax_scale,
        softcap=softcap,
        deterministic=deterministic,
        **kwargs,
    )

    out = fdma_fn(
        query_states,
        key_states,
        value_states,
        attention_mask,
        attention_bias,
        **flash_kwargs,
    )
    if isinstance(out, tuple):
        out = out[0]

    return out
