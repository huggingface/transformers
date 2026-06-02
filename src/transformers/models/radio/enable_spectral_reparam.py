# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import os
from logging import getLogger

import torch
from timm.models.vision_transformer import Attention
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import _SpectralNorm


_EPS = 1e-5


class _SNReweight(_SpectralNorm):
    def __init__(
        self,
        weight: torch.Tensor,
        *args,
        init_norm_to_current: bool = False,
        alpha: float = 0.05,
        version: int = 2,
        **kwargs,
    ):
        super().__init__(weight, *args, **kwargs)

        self.alpha = alpha
        self.version = version
        self.register_buffer("_sn_version", torch.tensor(version))

        if init_norm_to_current:
            # This will set the numerator to match the denominator, which should preserve the original values
            init_scale = self._get_sigma(weight, n_power_iterations=20).item()
        else:
            init_scale = 1.0

        if version == 1:
            init_value = init_scale
        elif version == 2:
            t = init_scale - alpha
            if t < _EPS:
                getLogger("spectral_reparam").warn(
                    f"The initialized spectral norm {init_scale} is too small to be represented. Setting to {_EPS} instead."
                )
                t = _EPS

            init_value = math.log(math.exp(t) - 1)
        else:
            raise ValueError(f"Unsupported version: {version}")

        # Make 2D so that weight decay gets applied
        self.scale = nn.Parameter(torch.tensor([[init_value]], dtype=torch.float32, device=weight.device))

    # Re-implementing this because we need to make division by sigma safe
    def _get_sigma(self, weight: torch.Tensor, n_power_iterations: int = None) -> torch.Tensor:
        if not n_power_iterations:
            n_power_iterations = self.n_power_iterations
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            sigma = weight.norm()
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, n_power_iterations)
            # See above on why we need to clone
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            # The proper way of computing this should be through F.bilinear, but
            # it seems to have some efficiency issues:
            # https://github.com/pytorch/pytorch/issues/58093
            sigma = torch.dot(u, torch.mv(weight_mat, v))

        return sigma + self.eps

    def forward(self, weight: torch.Tensor, *args, **kwargs):
        dtype = weight.dtype
        sigma = self._get_sigma(weight, *args, **kwargs)

        if self.version == 1:
            scale = self.scale
        elif self.version == 2:
            scale = F.softplus(self.scale) + self.alpha
        else:
            raise ValueError(f"Unsupported version: {self.version}")

        scale = scale.float() / sigma.float()

        y = weight * scale

        if dtype in (torch.float16, torch.bfloat16):
            y = y.to(dtype)
        return y

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version_key = f"{prefix}_sn_version"
        if version_key not in state_dict:
            self.version = 1
            state_dict[version_key] = torch.tensor(1)
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class _ChunkedSNReweight(nn.Module):
    def __init__(self, weight: torch.Tensor, num_chunks: int, *args, init_norm_to_current: bool = False, **kwargs):
        super().__init__()

        self.num_chunks = num_chunks
        parts = weight.split(weight.shape[0] // num_chunks, dim=0)

        self.parts = nn.ModuleList(
            [_SNReweight(p, *args, init_norm_to_current=init_norm_to_current, **kwargs) for p in parts]
        )

    def forward(self, weight: torch.Tensor, *args, **kwargs):
        parts = weight.split(weight.shape[0] // self.num_chunks, dim=0)

        parts = [fn(p) for fn, p in zip(self.parts, parts)]

        return torch.cat(parts, dim=0)


class _AttnSNReweight(_ChunkedSNReweight):
    def __init__(
        self, weight: torch.Tensor, *args, init_norm_to_current: bool = False, renorm_values: bool = False, **kwargs
    ):
        super().__init__(weight, 3, *args, init_norm_to_current=init_norm_to_current, **kwargs)

        if not renorm_values:
            self.parts[2] = nn.Identity()


def enable_spectral_reparam(
    model: nn.Module | list[nn.Module],
    n_power_iterations: int = 1,
    eps: float = 1e-6,
    init_norm_to_current: bool = False,
    renorm_values: bool = True,
    renorm_mlp: bool = True,
    state_dict_guidance: dict[str, torch.Tensor] | None = None,
):
    if isinstance(model, (list, tuple)):
        for i, sub in enumerate(model):
            sub_sd = state_dict_guidance[i] if isinstance(state_dict_guidance, (list, tuple)) else state_dict_guidance
            enable_spectral_reparam(
                sub,
                n_power_iterations=n_power_iterations,
                eps=eps,
                init_norm_to_current=init_norm_to_current,
                renorm_values=renorm_values,
                renorm_mlp=renorm_mlp,
                state_dict_guidance=sub_sd,
            )
        return

    print("Enabling spectral reparametrization")
    args = dict(n_power_iterations=n_power_iterations, dim=0, eps=eps, init_norm_to_current=init_norm_to_current)
    visited_prefixes = set()

    def is_guidance_parametrized(name: str):
        if state_dict_guidance is None:
            return True

        p_name = f"{name}.parametrizations"
        is_prm = any(k for k in state_dict_guidance if k.startswith(p_name) and k.endswith("_sn_version"))
        return is_prm

    def parametrize_linear(linear: nn.Linear):
        parametrize.register_parametrization(linear, "weight", _SNReweight(linear.weight, **args))

    for name, mod in model.named_modules():
        pref = ".".join(name.split(".")[:-1])
        if pref in visited_prefixes:
            continue

        if isinstance(mod, Attention) or name.endswith(".attn"):
            if is_guidance_parametrized(f"{name}.qkv"):
                parametrize.register_parametrization(
                    mod.qkv,
                    "weight",
                    _AttnSNReweight(mod.qkv.weight, renorm_values=renorm_values, **args),
                )
            if hasattr(mod, "proj") and is_guidance_parametrized(f"{name}.proj"):
                parametrize_linear(mod.proj)
            visited_prefixes.add(name)
        elif name.endswith("mlp") and renorm_mlp and hasattr(mod, "w12"):
            if is_guidance_parametrized(f"{name}.w12"):
                parametrize.register_parametrization(
                    mod.w12,
                    "weight",
                    _ChunkedSNReweight(mod.w12.weight, num_chunks=2, **args),
                )
            if is_guidance_parametrized(f"{name}.w3"):
                parametrize_linear(mod.w3)
            visited_prefixes.add(name)
        elif isinstance(mod, nn.Linear) and "patch_generator" not in name and is_guidance_parametrized(name):
            parametrize_linear(mod)


def configure_spectral_reparam_from_args(
    model: nn.Module, args, state_dict_guidance: dict[str, torch.Tensor] | None = None
):
    spectral_reparam = getattr(args, "spectral_reparam", False)
    if isinstance(spectral_reparam, bool) and spectral_reparam:
        enable_spectral_reparam(model, init_norm_to_current=True, state_dict_guidance=state_dict_guidance)
    elif isinstance(spectral_reparam, dict):
        enable_spectral_reparam(
            model,
            n_power_iterations=spectral_reparam.get("n_power_iterations", 1),
            eps=spectral_reparam.get("eps", 1e-12),
            init_norm_to_current=True,
            state_dict_guidance=state_dict_guidance,
        )


def disable_spectral_reparam(model: nn.Module):
    print("Disabling spectral reparametrization")
    for name, mod in model.named_modules():
        if parametrize.is_parametrized(mod):
            parametrize.remove_parametrizations(mod, "weight")
            pass


if __name__ == "__main__":
    import argparse

    from . import radio_model as create_model

    parser = argparse.ArgumentParser(description="Remove parametrization from state dict")
    parser.add_argument("--checkpoint", type=str, required=True, help="The checkpoint to load")
    parser.add_argument("--output", type=str, default="", help="Where to store the checkpoint")
    parser.add_argument("--release", default=False, action="store_true", help="Prune extraneous checkpoint fields")
    parser.add_argument("--strict", default=False, action="store_true", help="Strictly load the state dict")

    args = parser.parse_args()

    if not args.output:
        chk_dir, chk_name = os.path.split(args.checkpoint)
        args.output = os.path.join(chk_dir, f"clean_{chk_name}")
        print(f'Set output to "{args.output}"')

    chk = torch.load(args.checkpoint, map_location="cpu", mmap=True)

    model = create_model.create_model_from_args(chk["args"])

    key = "base_model."
    mod_state = dict()
    extra_state = dict()
    for k, v in chk["state_dict"].items():
        if k.startswith(key):
            mod_state[k[len(key) :]] = v
        else:
            extra_state[k] = v

    chk_load_info = model.load_state_dict(mod_state, strict=args.strict)
    if chk_load_info.unexpected_keys or chk_load_info.missing_keys:
        print(chk_load_info)

    if chk["args"].spectral_reparam:
        disable_spectral_reparam(model)

    if hasattr(chk["args"], "dtype"):
        model.to(dtype=chk["args"].dtype)

    mod_state = model.state_dict()
    final_state = dict()
    final_state.update({f"{key}{k}": v for k, v in mod_state.items()})
    final_state.update(extra_state)

    chk["state_dict"] = final_state
    chk["args"].spectral_reparam = False

    if args.release:
        chk = {
            "arch": chk["arch"],
            "epoch": chk["epoch"],
            "state_dict": chk["state_dict"],
            "args": chk["args"],
        }

    torch.save(chk, args.output)
    pass
