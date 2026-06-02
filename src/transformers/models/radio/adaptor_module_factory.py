# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from torch import nn

from .adaptor_attn import AttnFDHead
from .adaptor_mlp import MLP, MLP2
from .enable_spectral_reparam import disable_spectral_reparam, enable_spectral_reparam


MLP_SUMMARY_FACTORY = {
    "v1": MLP,
    "v2": MLP2,
}

MLP_FD_FACTORY = {
    "v1": MLP,
    "v2": MLP2,
    "attn": AttnFDHead,
}


def strip_prefix(state: dict[str, torch.Tensor], prefix: str):
    state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    return state


def get_mlp_info_from_state(
    version: str, state: dict[str, torch.Tensor], prefix: str = "", spectral_weights: bool = False
):
    state = strip_prefix(state, prefix)

    weight_suffix = "weight" if not spectral_weights else "parametrizations.weight.original"

    if version == "v1":
        hidden_dim, input_dim = state[f"fc1.{weight_suffix}"].shape
        output_dim = state[f"fc2.{weight_suffix}"].shape[0]

        for num_inner in range(1000):
            k = f"inner.{num_inner}.0.weight"
            if k not in state:
                break
    elif version == "v2":
        hidden_dim, input_dim = state[f"fc1.{weight_suffix}"].shape
        output_dim = state[f"final.2.{weight_suffix}"].shape[0]

        for num_inner in range(1000):
            k = f"blocks.{num_inner}.0.weight"
            if k not in state:
                break
    elif version == "attn":
        hidden_dim, input_dim = state[f"mlp.fc1.{weight_suffix}"].shape
        output_dim = state[f"mlp.final.2.{weight_suffix}"].shape[0]
        num_inner = 0
    else:
        raise ValueError(f"Unsupported MLP version: {version}")

    return input_dim, hidden_dim, output_dim, num_inner


def create_mlp_from_config(
    version: str, input_dim: int, hidden_dim: int, output_dim: int, num_inner: int, is_summary: bool = True, **kwargs
):
    factory = MLP_SUMMARY_FACTORY if is_summary else MLP_FD_FACTORY

    ret: nn.Module = factory[version](input_dim, hidden_dim, output_dim, num_inner, from_config=True, **kwargs)

    return ret


def create_mlp_from_state(
    version: str,
    state: dict[str, torch.Tensor],
    prefix: str = "",
    spectral_weights: bool = False,
    is_summary: bool = True,
    **kwargs,
):
    state = strip_prefix(state, prefix)

    input_dim, hidden_dim, output_dim, num_inner = get_mlp_info_from_state(
        version, state, spectral_weights=spectral_weights
    )

    ret: nn.Module = create_mlp_from_config(
        version, input_dim, hidden_dim, output_dim, num_inner, is_summary=is_summary, **kwargs
    )
    if spectral_weights:
        enable_spectral_reparam(ret, init_norm_to_current=False, state_dict_guidance=state)

    ret.load_state_dict(state)

    if spectral_weights:
        disable_spectral_reparam(ret)

    return ret
