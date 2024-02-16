# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles
#
# Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan,
# Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed,
# Judy Hoffman, Jitendra Malik, Yanghao Li, Christoph Feichtenhofer.
#
# Paper: https://arxiv.org/abs/2306.00989/
#
# References:
# slowfast: https://github.com/facebookresearch/SlowFast
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

import math
from typing import List, Tuple, Optional, Type, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .convert_hiera_to_pytorch import convert_state_dict

def pretrained_model(checkpoints: Dict[str, str], default: str = None) -> Callable:
    """ Loads a Hiera model from a pretrained source (if pretrained=True). Use "checkpoint" to specify the checkpoint. """

    def inner(model_func: Callable) -> Callable:
        def model_def(pretrained: bool = False, checkpoint: str = default, strict: bool = True, **kwdargs) -> nn.Module:
            if pretrained:
                if checkpoints is None:
                    raise RuntimeError("This model currently doesn't have pretrained weights available.")
                elif checkpoint is None:
                    raise RuntimeError("No checkpoint specified.")
                elif checkpoint not in checkpoints:
                    raise RuntimeError(f"Invalid checkpoint specified ({checkpoint}). Options are: {list(checkpoints.keys())}.")

                state_dict = torch.hub.load_state_dict_from_url(checkpoints[checkpoint], map_location="cpu")
                state_dict["model_state"] = convert_state_dict(state_dict["model_state"],{})
                if "head.projection.weight" in state_dict["model_state"]:
                    # Set the number of classes equal to the state_dict only if the user doesn't want to overwrite it
                    if "num_classes" not in kwdargs:
                        kwdargs["num_classes"] = state_dict["model_state"]["head.projection.weight"].shape[0]
                    # If the user specified a different number of classes, remove the projection weights or else we'll error out
                    elif kwdargs["num_classes"] != state_dict["model_state"]["head.projection.weight"].shape[0]:
                        del state_dict["model_state"]["head.projection.weight"]
                        del state_dict["model_state"]["head.projection.bias"]

            model = model_func(**kwdargs)
            if pretrained:
                # Disable being strict when trying to load a encoder-decoder model into an encoder-only model
                if "decoder_position_embeddings" in state_dict["model_state"] and not hasattr(model, "decoder_position_embeddings"):
                    strict = False

                model.load_state_dict(state_dict["model_state"], strict=strict)
            
            return model

        return model_def
    
    return inner



def conv_nd(n: int) -> Type[nn.Module]:
    """
    Returns a conv with nd (e.g., Conv2d for n=2). Work up to n=3.
    If you wanted a 4d Hiera, you could probably just implement this for n=4. (no promises)
    """
    return [nn.Identity, nn.Conv1d, nn.Conv2d, nn.Conv3d][n]


def do_pool(x: torch.Tensor, stride: int) -> torch.Tensor:
    # Refer to `Unroll` to see how this performs a maxpool-Nd
    return x.view(x.shape[0], stride, -1, x.shape[-1]).max(dim=1).values


def get_resized_mask(target_size: torch.Size, mask: torch.Tensor) -> torch.Tensor:
    # target_size: [(T), (H), W]
    # (spatial) mask: [B, C, (t), (h), w]
    if mask is None:
        return mask

    assert len(mask.shape[2:]) == len(target_size)
    if mask.shape[2:] != target_size:
        return F.interpolate(mask.float(), size=target_size)
    return mask


def do_masked_conv(
    x: torch.Tensor, conv: nn.Module, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Zero-out the masked regions of the input before conv.
    Prevents leakage of masked regions when using overlapping kernels.
    """
    if conv is None:
        return x
    if mask is None:
        return conv(x)

    mask = get_resized_mask(target_size=x.shape[2:], mask=mask)
    return conv(x * mask.bool())


def undo_windowing(
    x: torch.Tensor, shape: List[int], mu_shape: List[int]
) -> torch.Tensor:
    """
    Restore spatial organization by undoing windowed organization of mask units.

    Args:
        x: organized by mask units windows, e.g. in 2d [B, #MUy*#MUx, MUy, MUx, C]
        shape: current spatial shape, if it were not organized into mask unit
            windows, e.g. in 2d [B, #MUy*MUy, #MUx*MUx, C].
        mu_shape: current mask unit shape, e.g. in 2d [MUy, MUx]
    Returns:
        x: e.g. in 2d, [B, #MUy*MUy, #MUx*MUx, C]
    """
    D = len(shape)
    B, C = x.shape[0], x.shape[-1]
    # [B, #MUy*#MUx, MUy, MUx, C] -> [B, #MUy, #MUx, MUy, MUx, C]
    num_MUs = [s // mu for s, mu in zip(shape, mu_shape)]
    x = x.view(B, *num_MUs, *mu_shape, C)

    # [B, #MUy, #MUx, MUy, MUx, C] -> [B, #MUy*MUy, #MUx*MUx, C]
    permute = (
        [0]
        + sum(
            [list(p) for p in zip(range(1, 1 + D), range(1 + D, 1 + 2 * D))],
            [],
        )
        + [len(x.shape) - 1]
    )
    x = x.permute(permute).reshape(B, *shape, C)

    return x



class Unroll(nn.Module):
    """
    Reorders the tokens such that patches are contiguous in memory.
    E.g., given [B, (H, W), C] and stride of (Sy, Sx), this will re-order the tokens as
                           [B, (Sy, Sx, H // Sy, W // Sx), C]

    This allows operations like Max2d to be computed as x.view(B, Sx*Sy, -1, C).max(dim=1).
    Not only is this faster, but it also makes it easy to support inputs of arbitrary
    dimensions in addition to patch-wise sparsity.

    Performing this operation multiple times in sequence puts entire windows as contiguous
    in memory. For instance, if you applied the stride (2, 2) 3 times, entire windows of
    size 8x8 would be contiguous in memory, allowing operations like mask unit attention
    computed easily and efficiently, while also allowing max to be applied sequentially.

    Note: This means that intermediate values of the model are not in HxW order, so they
    need to be re-rolled if you want to use the intermediate values as a HxW feature map.
    The last block of the network is fine though, since by then the strides are all consumed.
    """

    def __init__(
        self,
        input_size: Tuple[int, ...],
        patch_stride: Tuple[int, ...],
        unroll_schedule: List[Tuple[int, ...]],
    ):
        super().__init__()
        self.size = [i // s for i, s in zip(input_size, patch_stride)]
        self.schedule = unroll_schedule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Flattened patch embeddings [B, N, C]
        Output: Patch embeddings [B, N, C] permuted such that [B, 4, N//4, C].max(1) etc. performs MaxPoolNd
        """
        B, _, C = x.shape

        cur_size = self.size
        x = x.view(*([B] + cur_size + [C]))

        for strides in self.schedule:
            # Move patches with the given strides to the batch dimension

            # Create a view of the tensor with the patch stride as separate dims
            # For example in 2d: [B, H // Sy, Sy, W // Sx, Sx, C]
            cur_size = [i // s for i, s in zip(cur_size, strides)]
            new_shape = [B] + sum([[i, s] for i, s in zip(cur_size, strides)], []) + [C]
            x = x.view(new_shape)

            # Move the patch stride into the batch dimension
            # For example in 2d: [B, Sy, Sx, H // Sy, W // Sx, C]
            L = len(new_shape)
            permute = (
                [0] + list(range(2, L - 1, 2)) + list(range(1, L - 1, 2)) + [L - 1]
            )
            x = x.permute(permute)

            # Now finally flatten the relevant dims into the batch dimension
            x = x.flatten(0, len(strides))
            B *= math.prod(strides)

        x = x.reshape(-1, math.prod(self.size), C)
        return x


class Reroll(nn.Module):
    """
    Undos the "unroll" operation so that you can use intermediate features.
    """

    def __init__(
        self,
        input_size: Tuple[int, ...],
        patch_stride: Tuple[int, ...],
        unroll_schedule: List[Tuple[int, ...]],
        stage_ends: List[int],
        q_pool: int,
    ):
        super().__init__()
        self.size = [i // s for i, s in zip(input_size, patch_stride)]

        # The first stage has to reverse everything
        # The next stage has to reverse all but the first unroll, etc.
        self.schedule = {}
        size = self.size
        for i in range(stage_ends[-1] + 1):
            self.schedule[i] = unroll_schedule, size
            # schedule unchanged if no pooling at a stage end
            if i in stage_ends[:q_pool]:
                if len(unroll_schedule) > 0:
                    size = [n // s for n, s in zip(size, unroll_schedule[0])]
                unroll_schedule = unroll_schedule[1:]

    def forward(
        self, x: torch.Tensor, block_idx: int, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Roll the given tensor back up to spatial order assuming it's from the given block.

        If no mask is provided:
            - Returns [B, H, W, C] for 2d, [B, T, H, W, C] for 3d, etc.
        If a mask is provided:
            - Returns [B, #MUs, MUy, MUx, C] for 2d, etc.
        """
        schedule, size = self.schedule[block_idx]
        B, N, C = x.shape

        D = len(size)
        cur_mu_shape = [1] * D

        for strides in schedule:
            # Extract the current patch from N
            x = x.view(B, *strides, N // math.prod(strides), *cur_mu_shape, C)

            # Move that patch into the current MU
            # Example in 2d: [B, Sy, Sx, N//(Sy*Sx), MUy, MUx, C] -> [B, N//(Sy*Sx), Sy, MUy, Sx, MUx, C]
            L = len(x.shape)
            permute = (
                [0, 1 + D]
                + sum(
                    [list(p) for p in zip(range(1, 1 + D), range(1 + D + 1, L - 1))],
                    [],
                )
                + [L - 1]
            )
            x = x.permute(permute)

            # Reshape to [B, N//(Sy*Sx), *MU, C]
            for i in range(D):
                cur_mu_shape[i] *= strides[i]
            x = x.reshape(B, -1, *cur_mu_shape, C)
            N = x.shape[1]

        # Current shape (e.g., 2d: [B, #MUy*#MUx, MUy, MUx, C])
        x = x.view(B, N, *cur_mu_shape, C)

        # If masked, return [B, #MUs, MUy, MUx, C]
        if mask is not None:
            return x

        # If not masked, we can return [B, H, W, C]
        x = undo_windowing(x, size, cur_mu_shape)

        return x