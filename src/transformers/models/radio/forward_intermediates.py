# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from collections.abc import Callable

import torch
from torch import nn

from .feature_normalizer import IntermediateFeatureNormalizerBase, NullIntermediateFeatureNormalizer


def _take_indices(
    num_blocks: int,
    n: int | list[int] | tuple[int] | None,
) -> tuple[set[int], int]:
    if isinstance(n, int):
        assert n >= 0
        take_indices = {x for x in range(num_blocks - n, num_blocks)}
    else:
        take_indices = {num_blocks + idx if idx < 0 else idx for idx in n}
    return take_indices, max(take_indices)


def forward_intermediates(
    model: nn.Module,
    patch_extractor: Callable[[torch.Tensor], torch.Tensor],
    norm: nn.Module,
    num_summary_tokens: int,
    num_cls_tokens: int,
    x: torch.Tensor,
    indices: int | list[int] | tuple[int] | None = None,
    return_prefix_tokens: bool = False,
    stop_early: bool = False,
    output_fmt: str = "NCHW",
    intermediates_only: bool = False,
    aggregation: str | None = "sparse",
    inter_feature_normalizer: IntermediateFeatureNormalizerBase | None = None,
    norm_alpha_scheme="post-alpha",
    block_kwargs: dict = None,
) -> list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
    """Forward features that returns intermediates.

    The Dense layer aggregation method is inspired from the paper: "Dense Connector for MLLMs"
    by Yao, Huanjin et al. (2024). arXiv preprint arXiv:2405.13800}

    Args:
        x: Input image tensor
        indices: Take last n blocks if int, select matching indices if sequence
        return_prefix_tokens: Return both prefix and spatial intermediate tokens
        norm: Apply norm layer to all intermediates
        stop_early: Stop iterating over blocks when last desired intermediate hit
        output_fmt: Shape of intermediate feature outputs
        intermediates_only: Only return intermediate features
        aggregation: intermediate layer aggregation method (sparse or dense)
        norm_alpha_scheme: apply alpha before ("pre-alpha") or after accumulation ("post-alpha")
    Returns:
    """
    assert output_fmt in ("NCHW", "NLC"), "Output format must be one of NCHW or NLC."
    assert aggregation in ("sparse", "dense"), "Aggregation must be one of sparse or dense."
    reshape = output_fmt == "NCHW"
    intermediates = []

    block_kwargs = block_kwargs or dict()

    blocks = model.blocks

    take_indices, max_index = _take_indices(len(blocks), indices)
    take_indices = sorted(take_indices)
    # forward pass
    B, _, height, width = x.shape

    x = patch_extractor(x)

    if stop_early:
        blocks = blocks[: max_index + 1]

    if inter_feature_normalizer is None or norm_alpha_scheme == "none":
        inter_feature_normalizer = NullIntermediateFeatureNormalizer.get_instance(x.dtype, x.device)

    assert norm_alpha_scheme in ("none", "pre-alpha", "post-alpha"), f"Unsupported alpha scheme: {norm_alpha_scheme}"
    post_alpha_scheme = norm_alpha_scheme == "post-alpha"

    accumulator = 0
    alpha_sum = 0
    num_accumulated = 0

    take_off = 0

    for i, blk in enumerate(blocks):
        x = blk(x, **block_kwargs)
        if aggregation == "dense":
            # Arbitrarily use the rotation matrix from the final layer in the dense group
            y, alpha = inter_feature_normalizer(x, i, rot_index=take_indices[take_off], skip=num_summary_tokens)
            if post_alpha_scheme:
                accumulator = accumulator + y
                alpha_sum = alpha_sum + alpha
            else:
                accumulator = accumulator + (alpha * y)
                alpha_sum += 1
            num_accumulated += 1
        if i == take_indices[take_off]:
            if aggregation == "dense":
                alpha = alpha_sum / num_accumulated
                x_ = alpha * accumulator / num_accumulated
                num_accumulated = 0
                accumulator = 0
                alpha_sum = 0
            else:
                y, alpha = inter_feature_normalizer(x, i, skip=num_summary_tokens)
                x_ = alpha * y
            # normalize intermediates with final norm layer if enabled
            intermediates.append(norm(x_))
            take_off = min(take_off + 1, len(take_indices) - 1)

    # process intermediates

    # split prefix (e.g. class, distill) and spatial feature tokens
    prefix_tokens = [y[:, :num_cls_tokens] for y in intermediates]
    intermediates = [y[:, num_summary_tokens:] for y in intermediates]

    if reshape:
        # reshape to BCHW output format
        H = height // model.patch_size
        W = width // model.patch_size
        intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]
    if not torch.jit.is_scripting() and return_prefix_tokens:
        # return_prefix not support in torchscript due to poor type handling
        intermediates = list(zip(prefix_tokens, intermediates))
    if intermediates_only:
        return intermediates
    x = norm(x)
    return x, intermediates
