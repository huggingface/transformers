import math
import sys

import torch
from einops import rearrange
from timm.models import VisionTransformer
from torch import nn

from .extra_models import DinoWrapper


DEFAULT_NUM_WINDOWED = 5
DEFAULT_NUM_GLOBAL = 4


class VitDetArgs:
    def __init__(
        self,
        window_size: int,
        num_summary_tokens: int,
        num_windowed: int = None,
        num_global: int = None,
    ):
        self.window_size = window_size
        self.num_summary_tokens = num_summary_tokens
        self.num_windowed = num_windowed
        self.num_global = num_global


def apply_vitdet_arch(model: VisionTransformer | DinoWrapper, args: VitDetArgs):
    if isinstance(model, VisionTransformer):
        patch_embed = getattr(model, "patch_generator", model.patch_embed)

        return ViTDetHook(patch_embed, model.blocks, args)
    elif isinstance(model, DinoWrapper):
        inner = model.inner

        patch_embed = getattr(inner, "patch_generator", inner.patch_embed)
        return ViTDetHook(patch_embed, inner.blocks, args)
    else:
        print("Warning: Unable to apply VitDet aug!", file=sys.stderr)


class ViTDetHook:
    def __init__(
        self,
        embedder: nn.Module,
        blocks: nn.Sequential,
        args: VitDetArgs,
    ):
        self.blocks = blocks
        self.num_summary_tokens = args.num_summary_tokens
        self.window_size = args.window_size

        self._input_resolution = None
        self._num_windows = None
        self._cls_patch = None
        self._order_cache = dict()

        embedder.register_forward_pre_hook(self._enter_model)

        # This will decide if we window-fy the patches
        # and enable vit-det for this iteration, and if so,
        # rearrange the patches for efficient mode switching
        blocks.register_forward_pre_hook(self._enter_blocks)

        is_global = True
        if args.num_windowed is not None:
            period = args.num_windowed + 1
        else:
            num_global = args.num_global or DEFAULT_NUM_GLOBAL
            period = max(len(blocks) // num_global, 1)

        for i, layer in enumerate(blocks[:-1]):
            ctr = i % period
            if ctr == 0:
                layer.register_forward_pre_hook(self._to_windows)
                is_global = False
            elif ctr == period - 1:
                layer.register_forward_pre_hook(self._to_global)
                is_global = True

        # Always ensure the final layer is a global layer
        if not is_global:
            blocks[-1].register_forward_pre_hook(self._to_global)

        blocks.register_forward_hook(self._exit_model)

    def _enter_model(self, _, input: list[torch.Tensor]):
        self._input_resolution = input[0].shape[-2:]

    def _enter_blocks(self, _, input: list[torch.Tensor]):
        # print(f'{get_rank()} - ViTDet Window Size: {self._window_size}', file=sys.stderr)

        patches = input[0]
        patches = self._rearrange_patches(patches)

        return (patches,) + input[1:]

    def _to_windows(self, _, input: list[torch.Tensor]):
        patches = input[0]

        if self.num_summary_tokens:
            self._cls_patch = patches[:, : self.num_summary_tokens]
            patches = patches[:, self.num_summary_tokens :]

        patches = rearrange(
            patches,
            "b (p t) c -> (b p) t c",
            p=self._num_windows,
            t=self.window_size**2,
        )

        return (patches,) + input[1:]

    def _to_global(self, _, input: list[torch.Tensor]):
        patches = input[0]

        patches = rearrange(
            patches,
            "(b p) t c -> b (p t) c",
            p=self._num_windows,
            t=self.window_size**2,
            b=patches.shape[0] // self._num_windows,
        )

        if self.num_summary_tokens:
            patches = torch.cat(
                [
                    self._cls_patch,
                    patches,
                ],
                dim=1,
            )

        return (patches,) + input[1:]

    def _exit_model(self, _, inputs: list[torch.Tensor], patches: torch.Tensor):
        # Return patches to their original order
        patch_order = self._order_cache[self._input_resolution][0]
        patch_order = patch_order.reshape(1, -1, 1).expand_as(patches)

        ret_patches = torch.empty_like(patches)
        ret_patches = torch.scatter(
            ret_patches,
            dim=1,
            index=patch_order,
            src=patches,
        )

        return ret_patches

    def _rearrange_patches(self, patches: torch.Tensor):
        # We rearrange the patches so that we can efficiently
        # switch between windowed and global mode by just
        # reshaping the tensor

        patch_order, self._num_windows = self._order_cache.get(self._input_resolution, (None, None))
        if patch_order is None:
            num_feat_patches = patches.shape[1] - self.num_summary_tokens
            num_pixels = self._input_resolution[0] * self._input_resolution[1]

            patch_size = int(round(math.sqrt(num_pixels / num_feat_patches)))
            rows = self._input_resolution[-2] // patch_size
            cols = self._input_resolution[-1] // patch_size

            w_rows = rows // self.window_size
            w_cols = cols // self.window_size

            patch_order = torch.arange(0, num_feat_patches, device=patches.device)

            patch_order = rearrange(
                patch_order,
                "(wy py wx px) -> (wy wx py px)",
                wy=w_rows,
                wx=w_cols,
                py=self.window_size,
                px=self.window_size,
            )

            if self.num_summary_tokens:
                patch_order = torch.cat(
                    [
                        torch.arange(self.num_summary_tokens, dtype=patch_order.dtype, device=patch_order.device),
                        patch_order + self.num_summary_tokens,
                    ]
                )

            self._num_windows = w_rows * w_cols
            self._order_cache[self._input_resolution] = (
                patch_order,
                self._num_windows,
            )

        patch_order = patch_order.reshape(1, -1, 1).expand_as(patches)
        patches = torch.gather(patches, dim=1, index=patch_order)
        return patches
