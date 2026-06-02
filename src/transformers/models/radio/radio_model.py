# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from collections.abc import Callable, Iterable
from typing import NamedTuple

import torch
from timm.models import VisionTransformer, create_model
from torch import nn

from . import eradio_model
from .adaptor_base import AdaptorBase, AdaptorInput, RadioOutput
from .enable_cpe_support import enable_cpe
from .feature_normalizer import FeatureNormalizer, IntermediateFeatureNormalizer
from .input_conditioner import InputConditioner


class Resolution(NamedTuple):
    height: int
    width: int


class RADIOModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        input_conditioner: InputConditioner,
        patch_size: int,
        max_resolution: int,
        preferred_resolution: Resolution,
        summary_idxs: torch.Tensor | None = None,
        window_size: int = None,
        adaptors: dict[str, AdaptorBase] = None,
        feature_normalizer: FeatureNormalizer | None = None,
        inter_feature_normalizer: IntermediateFeatureNormalizer | None = None,
    ):
        super().__init__()

        self.model = model
        self.input_conditioner = input_conditioner
        if summary_idxs is not None:
            self.register_buffer("summary_idxs", summary_idxs)
        else:
            self.summary_idxs = None

        self._preferred_resolution = preferred_resolution
        self._patch_size = patch_size
        self._max_resolution = max_resolution
        self._window_size = window_size

        adaptors = adaptors or dict()
        self.adaptors = nn.ModuleDict(adaptors)

        if feature_normalizer is None:
            feature_normalizer = nn.Identity()
        self.feature_normalizer = feature_normalizer
        self.inter_feature_normalizer = inter_feature_normalizer

    @property
    def num_summary_tokens(self) -> int:
        if hasattr(self.model, "num_summary_tokens"):
            return self.model.num_summary_tokens

        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            return patch_gen.num_skip
        elif getattr(self.model, "global_pool", None) == "avg":
            return 0
        return 1

    @property
    def num_cls_tokens(self) -> int:
        if hasattr(self.model, "num_cls_tokens"):
            return self.model.num_cls_tokens

        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            return patch_gen.num_cls_tokens
        elif getattr(self.model, "global_pool", None) == "avg":
            return 0
        return 1

    @property
    def patch_size(self) -> int:
        if self._patch_size is not None:
            return self._patch_size
        if hasattr(self.model, "patch_size"):
            return self.model.patch_size
        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            return patch_gen.patch_size
        return None

    @property
    def max_resolution(self) -> int:
        return self._max_resolution

    @property
    def preferred_resolution(self) -> Resolution:
        return self._preferred_resolution

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def min_resolution_step(self) -> int:
        res = self.patch_size
        if self.window_size is not None:
            res *= self.window_size
        return res

    @property
    def blocks(self) -> Iterable[nn.Module]:
        blocks = getattr(self.model, "blocks", None)
        if blocks is not None:
            return blocks
        return None

    @property
    def embed_dim(self) -> int:
        return self.model.embed_dim

    @property
    def summary_dim(self) -> int:
        embed_dim = self.embed_dim
        if self.summary_idxs is not None:
            embed_dim *= self.summary_idxs.shape[0]
        return embed_dim

    def make_preprocessor_external(self) -> Callable[[torch.Tensor], torch.Tensor]:
        ret = self.input_conditioner
        self.input_conditioner = nn.Identity()
        return ret

    def get_nearest_supported_resolution(self, height: int, width: int) -> Resolution:
        height = int(round(height / self.min_resolution_step) * self.min_resolution_step)
        width = int(round(width / self.min_resolution_step) * self.min_resolution_step)

        height = max(height, self.min_resolution_step)
        width = max(width, self.min_resolution_step)

        return Resolution(height=height, width=width)

    def switch_to_deploy(self):
        fn = getattr(self.model, "switch_to_deploy", None)
        if fn is not None:
            fn()

    def cpe_video_mode(self, t: int):
        """
        Context Manager.

        Puts the patch generator into video mode, with the specified number of temporal frames.
        In video mode, the expectation is that the input buffer is of shape `(B*T, C, H, W)`.
        Video mode means that the same position viewport will be used for every frame in the temporal sequence, while keeping
        distinct viewports for each video in the batch.

        Usage:
        with radio_model.cpe_video_mode(t=t):
            y = radio_model(x)
        """
        return self.model.cpe_video_mode(t)

    def forward(self, x: torch.Tensor, feature_fmt: str = "NLC") -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process for model.
        Args:
            x: Input tensor. Unless `make_preprocessor_external` has been called, then the dynamic range of `x` is expected to be `[0, 1]`,
                             otherwise `x` is expected to be mean centered with unit standard deviation.
            feature_format: ['NLC', 'NCHW'] - The output format for the features.
        """
        res_step = self.min_resolution_step
        if res_step is not None and (x.shape[-2] % res_step != 0 or x.shape[-1] % res_step != 0):
            raise ValueError(
                "The input resolution must be a multiple of `self.min_resolution_step`. "
                "`self.get_nearest_supported_resolution(<height>, <width>) is provided as a convenience API. "
                f"Input: {x.shape[-2:]}, Nearest: {self.get_nearest_supported_resolution(*x.shape[-2:])}"
            )

        x = self.input_conditioner(x)
        y = self.model.forward_features(x)
        ret = self._extract_final(x, y, feature_fmt=feature_fmt)
        return ret

    def _extract_final(self, x: torch.Tensor, y: torch.Tensor, feature_fmt: str = "NLC"):
        if isinstance(self.model, VisionTransformer):
            patch_gen = getattr(self.model, "patch_generator", None)
            if patch_gen is not None:
                all_summary = y[:, : patch_gen.num_cls_tokens]
                if self.summary_idxs is not None:
                    bb_summary = all_summary[:, self.summary_idxs]
                else:
                    bb_summary = all_summary
                all_feat = y[:, patch_gen.num_skip :]
            elif self.model.global_pool == "avg":
                all_summary = y[:, self.model.num_prefix_tokens :].mean(dim=1)
                bb_summary = all_summary
                all_feat = y
            else:
                all_summary = y[:, 0]
                bb_summary = all_summary
                all_feat = y[:, 1:]
        elif isinstance(self.model, eradio_model.ERADIO):
            _, f = y
            all_feat = f.flatten(2).transpose(1, 2)
            all_summary = all_feat.mean(dim=1)
            bb_summary = all_summary
        elif isinstance(y, (list, tuple)):
            all_summary, all_feat = y
            bb_summary = all_summary
        else:
            all_summary = y[:, : self.num_cls_tokens]
            if self.summary_idxs is not None and all_summary.shape[1] > 1:
                if all_summary.shape[1] == 1:
                    # Create dummy duplicates
                    all_summary = all_summary.expand(-1, 128, -1)
                bb_summary = all_summary[:, self.summary_idxs]
            else:
                bb_summary = all_summary
            all_feat = y[:, self.num_summary_tokens :]

        all_feat = self.feature_normalizer(all_feat)

        if feature_fmt == "NCHW":
            fmt_feat = all_feat.reshape(
                all_feat.shape[0], x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size, all_feat.shape[2]
            ).permute(0, 3, 1, 2)
        elif feature_fmt == "NLC":
            fmt_feat = all_feat
        else:
            raise ValueError(f'Unsupported feature_fmt: {feature_fmt}. Must be one of ["NLC", "NCHW"]')

        ret = RadioOutput(bb_summary.flatten(1), fmt_feat)

        if self.adaptors:
            ret = dict(backbone=ret)
            for name, adaptor in self.adaptors.items():
                if all_summary.ndim == 3:
                    if all_summary.shape[1] == 1:
                        summary = all_summary[:, 0]
                    else:
                        summary = all_summary[:, adaptor.head_idx]
                else:
                    summary = all_summary
                ada_input = AdaptorInput(
                    images=x,
                    summary=summary.float(),
                    features=all_feat,
                    feature_fmt=feature_fmt,
                    patch_size=self.patch_size,
                )
                v = adaptor(ada_input).to(torch.float32)
                ret[name] = v

        return ret

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: int | list[int] | tuple[int] | None = None,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = "NCHW",
        intermediates_only: bool = False,
        aggregation: str | None = "sparse",
        norm_alpha_scheme: str | None = "post-alpha",
    ) -> list[RadioOutput]:
        """Forward features that returns intermediates.
        Args:
            x: Input image tensor
            indices: Take last n blocks if int, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            norm: Apply norm layer to all intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs. Options: NCHW, NLC
            intermediates_only: Only return intermediate features
            aggregation: intermediate layer aggregation method (sparse or dense).
                Dense accumulation is done by averaging the features in each group.
            norm_alpha_scheme: apply alpha before ("pre-alpha") or after accumulation ("post-alpha"), or don't normalize ("none")
                Only affects dense aggregation
        Returns:
            List of RadioOutput objects.
        """
        x = self.input_conditioner(x)
        intermediates = self.model.forward_intermediates(
            x,
            indices=indices,
            return_prefix_tokens=return_prefix_tokens,
            norm=norm,
            stop_early=stop_early,
            output_fmt=output_fmt,
            intermediates_only=intermediates_only,
            aggregation=aggregation,
            inter_feature_normalizer=self.inter_feature_normalizer,
            norm_alpha_scheme=norm_alpha_scheme,
        )

        if not intermediates_only:
            final, intermediates = intermediates

        def prepare_summary(summ: torch.Tensor | None):
            if summ is None:
                return summ
            if self.summary_idxs is not None and summ.shape[1] > 1:
                summ = summ[:, self.summary_idxs]
            return summ.flatten(1)

        if return_prefix_tokens:
            radio_outputs = [RadioOutput(prepare_summary(summary), features) for summary, features in intermediates]
        else:
            radio_outputs = intermediates

        if intermediates_only:
            return radio_outputs
        else:
            final = self._extract_final(x, final, feature_fmt=output_fmt)
            return final, radio_outputs


def create_model_from_args(args) -> nn.Module:
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    # Skip weight initialization unless it's explicitly requested.
    weight_init = args.model_kwargs.pop("weight_init", "skip")

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        weight_init=weight_init,
        **args.model_kwargs,
    )

    if hasattr(model, "norm") and not getattr(args, "model_norm", False):
        model.norm = nn.Identity()

    model.head = nn.Identity()

    if args.cpe_max_size is not None:
        uq_teachers = set(t["name"] for t in args.teachers)
        enable_cpe(
            model,
            args.cpe_max_size,
            num_cls_tokens=len(uq_teachers) if args.cls_token_per_teacher else 1,
            register_multiple=getattr(args, "register_multiple", None),
            num_registers=getattr(args, "cpe_num_registers", None),
        )

    return model
