# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import functools
import importlib.metadata
import os
import re
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from types import ModuleType
from typing import TYPE_CHECKING

from packaging import version as pkg_version

from ..conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from ..core_model_loading import WeightRenaming
from ..monkey_patching import register_patch_mapping
from ..utils import ENV_VARS_TRUE_VALUES, logging
from ..utils.import_utils import is_kernels_available, is_torch_available
from .flash_attention import flash_attention_forward


if TYPE_CHECKING:
    from ..configuration_utils import PretrainedConfig
    from ..modeling_utils import PreTrainedModel
    from ..utils.kernel_config import KernelConfig

if is_torch_available():
    import torch
    import torch.nn as nn


logger = logging.get_logger(__name__)

try:
    from kernels import (
        Device,
        LayerRepository,
        LocalLayerRepository,
        Mode,
        register_kernel_mapping,
        replace_kernel_forward_from_hub,
    )
    from kernels import (
        get_kernel as get_kernel_hub,
    )
    from kernels import (
        use_kernel_forward_from_hub as _kernels_use_kernel_forward_from_hub,
    )

    # Try to import FuncRepository, fallback if not available
    try:
        from kernels import FuncRepository
    except ImportError:
        FuncRepository = None

    # Try to import use_kernel_func_from_hub, fallback if not available
    try:
        from kernels import use_kernel_func_from_hub as _kernels_use_kernel_func_from_hub

        _has_use_kernel_func_from_hub = True
    except ImportError:
        _has_use_kernel_func_from_hub = False

    _TRANSFORMERS_USE_HUB_KERNELS = os.environ.get("USE_HUB_KERNELS", "YES").upper()
    _kernels_available = True
    _kernels_enabled = _TRANSFORMERS_USE_HUB_KERNELS in ENV_VARS_TRUE_VALUES

    def use_kernel_forward_from_hub(layer_name: str):
        if _kernels_enabled:
            return _kernels_use_kernel_forward_from_hub(layer_name)
        else:
            logger.warning_once(
                f"kernels hub usage is disabled through the environment USE_HUB_KERNELS={_TRANSFORMERS_USE_HUB_KERNELS}"
            )
            return lambda cls: cls

    def use_kernel_func_from_hub(func_name: str):
        if _kernels_enabled and _has_use_kernel_func_from_hub:
            return _kernels_use_kernel_func_from_hub(func_name)
        else:
            if not _has_use_kernel_func_from_hub:
                logger.warning_once(
                    "use_kernel_func_from_hub is not available in the installed kernels version. "
                    "Please upgrade kernels to use this feature."
                )
            else:
                logger.warning_once(
                    f"kernels hub usage is disabled through the environment USE_HUB_KERNELS={_TRANSFORMERS_USE_HUB_KERNELS}"
                )
            return lambda func: func

    _KERNEL_MAPPING: dict[str, dict[Device | str, LayerRepository | dict[Mode, LayerRepository]]] = {
        "MultiScaleDeformableAttention": {
            "cuda": LayerRepository(
                repo_id="kernels-community/deformable-detr",
                layer_name="MultiScaleDeformableAttention",
            )
        },
        "Llama4TextMoe": {
            "cuda": LayerRepository(
                repo_id="kernels-community/moe",
                layer_name="Llama4TextMoe",
            )
        },
        "RMSNorm": {
            "cuda": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/liger_kernels",
                    layer_name="LigerRMSNorm",
                    # revision="pure-layer-test",
                ),
            },
            "rocm": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/liger_kernels",
                    layer_name="LigerRMSNorm",
                )
            },
            "xpu": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/rmsnorm",
                    layer_name="RMSNorm",
                )
            },
            "mps": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/mlx_rmsnorm",
                    layer_name="RMSNorm",
                )
            },
            "npu": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/liger_kernels",
                    layer_name="LigerRMSNorm",
                )
            },
        },
        "MLP": {
            "cuda": LayerRepository(
                repo_id="medmekk/triton-llama-mlp",
                layer_name="TritonLlamaMLP",
            )
        },
        "MegaBlocksMoeMLP": {
            "cuda": {
                Mode.TRAINING: LayerRepository(
                    repo_id="kernels-community/megablocks",
                    layer_name="MegaBlocksMoeMLP",
                ),
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/megablocks",
                    layer_name="MegaBlocksMoeMLP",
                ),
            },
            "rocm": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="ahadnagy/megablocks",
                    layer_name="MegaBlocksMoeMLP",
                )
            },
            "xpu": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/megablocks",
                    layer_name="MegaBlocksMoeMLP",
                )
            },
            "cpu": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/megablocks",
                    layer_name="CPUMegaBlocksMoeMLP",
                )
            },
        },
        "FastGELU": {
            "cuda": {
                Mode.INFERENCE | Mode.TORCH_COMPILE: LayerRepository(
                    repo_id="kernels-community/activation",
                    layer_name="FastGELU",
                    version=1,
                )
            }
        },
        "QuickGELU": {
            "cuda": {
                Mode.INFERENCE | Mode.TORCH_COMPILE: LayerRepository(
                    repo_id="kernels-community/activation",
                    layer_name="QuickGELU",
                    version=1,
                )
            }
        },
        "NewGELU": {
            "cuda": {
                Mode.INFERENCE | Mode.TORCH_COMPILE: LayerRepository(
                    repo_id="kernels-community/activation",
                    layer_name="NewGELU",
                    version=1,
                )
            }
        },
        "SiLU": {
            "cuda": {
                Mode.INFERENCE | Mode.TORCH_COMPILE: LayerRepository(
                    repo_id="kernels-community/activation", layer_name="Silu", version=1
                )
            }
        },
        "GeLU": {
            "cuda": {
                Mode.INFERENCE | Mode.TORCH_COMPILE: LayerRepository(
                    repo_id="kernels-community/activation", layer_name="Gelu", version=1
                )
            }
        },
        "GeluTanh": {
            "cuda": {
                Mode.INFERENCE | Mode.TORCH_COMPILE: LayerRepository(
                    repo_id="kernels-community/activation", layer_name="GeluTanh", version=1
                )
            }
        },
    }

    # Add function kernel mappings if FuncRepository is available
    if FuncRepository is not None:
        _KERNEL_MAPPING["rotary_pos_emb"] = {
            "xpu": {
                Mode.INFERENCE: FuncRepository(
                    repo_id="kernels-community/rotary", func_name="apply_rotary_transformers"
                )
            },
            "cuda": {
                Mode.TRAINING: FuncRepository(
                    repo_id="kernels-community/rotary", func_name="apply_rotary_transformers"
                ),
                Mode.INFERENCE: FuncRepository(
                    repo_id="kernels-community/rotary", func_name="apply_rotary_transformers"
                ),
            },
        }

    def has_key(d, key):
        return key in d or any(isinstance(v, dict) and has_key(v, key) for v in d.values())

    def register_kernel_mapping_transformers(mapping=None):
        if mapping is None:
            mapping = _KERNEL_MAPPING
        if has_key(mapping, "xpu") and not is_kernels_available(MIN_VERSION="0.10.2"):
            raise ImportError(
                "kernels uses an incompatible version. Please install the latest version with `pip install -U kernels`."
            )
        register_kernel_mapping(mapping)


except ImportError:
    _kernels_available = False
    _kernels_enabled = False

    # Stub to make decorators int transformers work when `kernels`
    # is not installed.
    def use_kernel_forward_from_hub(*args, **kwargs):
        def decorator(cls):
            return cls

        return decorator

    def use_kernel_func_from_hub(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    class LayerRepository:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("LayerRepository requires `kernels` to be installed. Run `pip install kernels`.")

    class LocalLayerRepository:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("LocalLayerRepository requires `kernels` to be installed. Run `pip install kernels`.")

    def replace_kernel_forward_from_hub(*args, **kwargs):
        raise RuntimeError(
            "replace_kernel_forward_from_hub requires `kernels` to be installed. Run `pip install kernels`."
        )

    def register_kernel_mapping(*args, **kwargs):
        raise RuntimeError("register_kernel_mapping requires `kernels` to be installed. Run `pip install kernels`.")

    def register_kernel_mapping_transformers(*args, **kwargs):
        raise RuntimeError(
            "register_kernel_mapping_transformers requires `kernels` to be installed. Run `pip install kernels`."
        )


_HUB_KERNEL_MAPPING: dict[str, dict[str, str]] = {
    "causal-conv1d": {"repo_id": "kernels-community/causal-conv1d", "version": 1},
    "mamba-ssm": {"repo_id": "kernels-community/mamba-ssm", "version": 1},
    "falcon_mamba-ssm": {"repo_id": "kernels-community/mamba-ssm", "version": 1},
    "finegrained-fp8": {"repo_id": "kernels-community/finegrained-fp8", "version": 2},
    "deep-gemm": {"repo_id": "kernels-community/deep-gemm", "version": 2},
    "sonic-moe": {"repo_id": "kernels-community/sonic-moe", "revision": "ep-support"},
}

_KERNEL_MODULE_MAPPING: dict[str, ModuleType | None] = {}


def is_kernel(attn_implementation: str | None) -> bool:
    """Check whether `attn_implementation` matches a kernel pattern from the hub."""
    return (
        attn_implementation is not None
        and re.search(r"^[^/:]+/[^/:]+(?:@[^/:]+)?(?::[^/:]+)?$", attn_implementation) is not None
    )


def load_and_register_attn_kernel(
    attn_implementation: str, attention_wrapper: Callable | None = None, allow_all_kernels: bool = False
) -> ModuleType | None:
    """
    Load and register the kernel associated to `attn_implementation`.

    Args:
        attn_implementation: A string, usually a kernel repo like "kernels-community/flash-mla".
        attn_wrapper: a callable for the wrapper around the attention implementation. In `transformers` we
            have a wrapper around the `flash_attn_var_len` call, and the same goes for `sdpa` and `eager`.
            They just prepare the arguments properly. This is mostly used for continious batching, where we
            want the `paged` wrapper, which calls the paged cache.
        allow_all_kernels (`bool`, optional):
            Whether to load kernels from unverified hub repos, if it is a custom kernel outside of the `kernels-community`
            hub repository.
    """
    from ..masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from ..modeling_utils import ALL_ATTENTION_FUNCTIONS

    actual_attn_name = attn_implementation.split("|")[1] if "|" in attn_implementation else attn_implementation
    if not is_kernel(actual_attn_name):
        return None
    if not _kernels_available:
        raise ImportError(
            "`kernels` is either not installed or uses an incompatible version. "
            "Please install the latest version with `pip install -U kernels`."
        )

    # Extract repo_id and kernel_name from the string
    if ":" in actual_attn_name:
        repo_id, kernel_name = actual_attn_name.split(":")
        kernel_name = kernel_name.strip()
    else:
        repo_id = actual_attn_name
        kernel_name = None
    repo_id = repo_id.strip()
    # extract the rev after the @ if it exists
    repo_id, _, rev = repo_id.partition("@")
    repo_id = repo_id.strip()
    rev = rev.strip() if rev else None

    # Load the kernel from hub
    try:
        kernel = get_kernel(repo_id, revision=rev, allow_all_kernels=allow_all_kernels)
    except Exception as e:
        raise ValueError(f"An error occurred while trying to load from '{repo_id}': {e}.")
    # correctly wrap the kernel
    if hasattr(kernel, "flash_attn_varlen_func"):
        if attention_wrapper is None:
            attention_wrapper = flash_attention_forward
        kernel_function = attention_wrapper
    elif kernel_name is not None:
        kernel_function = getattr(kernel, kernel_name)

    # Register the kernel as a valid attention
    ALL_ATTENTION_FUNCTIONS.register(attn_implementation, kernel_function)
    ALL_MASK_ATTENTION_FUNCTIONS.register(attn_implementation, ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"])

    return kernel


def lazy_load_kernel(kernel_name: str, mapping: dict[str, ModuleType | None] = _KERNEL_MODULE_MAPPING):
    if kernel_name in mapping and isinstance(mapping[kernel_name], ModuleType):
        return mapping[kernel_name]
    if kernel_name not in _HUB_KERNEL_MAPPING:
        logger.warning_once(f"Kernel {kernel_name} not found in _HUB_KERNEL_MAPPING")
        mapping[kernel_name] = None
        return None
    if _kernels_available:
        try:
            repo_id = _HUB_KERNEL_MAPPING[kernel_name]["repo_id"]
            revision = _HUB_KERNEL_MAPPING[kernel_name].get("revision", None)
            version = _HUB_KERNEL_MAPPING[kernel_name].get("version", None)
            kernel = get_kernel(repo_id, revision=revision, version=version, allow_all_kernels=ALLOW_ALL_KERNELS)
            mapping[kernel_name] = kernel
        except FileNotFoundError as e:
            mapping[kernel_name] = None
            logger.warning_once(f"Failed to load kernel {kernel_name}: {e}")
        except AssertionError:
            # Happens when torch is built without an accelerator backend; fall back to slow path.
            mapping[kernel_name] = None

    else:
        # Try to import is_{kernel_name}_available from ..utils
        import importlib

        new_kernel_name = kernel_name.replace("-", "_")
        func_name = f"is_{new_kernel_name}_available"

        try:
            utils_mod = importlib.import_module("..utils.import_utils", __package__)
            is_kernel_available = getattr(utils_mod, func_name, None)
        except Exception:
            is_kernel_available = None

        if callable(is_kernel_available) and is_kernel_available():
            # Try to import the module "{kernel_name}" from parent package level
            try:
                module = importlib.import_module(f"{new_kernel_name}")
                mapping[kernel_name] = module
                return module
            except Exception:
                mapping[kernel_name] = None
        else:
            mapping[kernel_name] = None

    return mapping[kernel_name]


def get_kernel(
    kernel_name: str,
    revision: str | None = None,
    version: int | str | None = None,
    allow_all_kernels: bool = False,
) -> ModuleType:
    from .. import __version__

    if not _kernels_available:
        raise ImportError(
            "`kernels` is either not installed or uses an incompatible version. Please install the latest version "
            "with `pip install -U kernels`."
        )

    repo_parent = kernel_name.split("/")[0]
    # all `kernels-community` repos are trusted by default!
    if repo_parent != "kernels-community" and not allow_all_kernels:
        raise ValueError(
            "You need to specify `allow_all_kernels=True` to use kernels outside of the `kernels-community` repository"
        )

    user_agent = {"framework": "transformers", "version": __version__, "repo_id": kernel_name}
    kernels_version = importlib.metadata.version("kernels")
    if pkg_version.parse(kernels_version) >= pkg_version.parse("0.10.4"):
        return get_kernel_hub(kernel_name, revision=revision, version=version, user_agent=user_agent)
    else:
        return get_kernel_hub(kernel_name, revision=revision, version=version)


def use_kernelized_func(module_names: list[Callable] | Callable):
    """
    This decorator attaches the target function within the module as a plain attribute (not as a submodule).
    Keep in mind that this registration is only meant for `kernelize` to recognize its target modules (i.e.
    function exchanged for a weightless `nn.Module` with the same forward) to then exchange to the kernel
    variation (in-place) if the conditions are met.

    We cache each of these function-based registrations: After proper registration and exchange it is removed
    from the module's `_modules` dict as it does not really act as `nn.Module` but a base function.
    """
    if isinstance(module_names, Callable):
        module_names = [module_names]

    def decorator(cls):
        orig_init = cls.__init__

        def new_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)

            # Register new function as non-submodule within the modules dict
            hidden_kernels = self.__dict__.setdefault("_hidden_kernels", {})
            for fn in module_names:
                name = (
                    getattr(fn, "__name__", None)
                    or getattr(fn, "kernel_layer_name", None)
                    or getattr(fn, "func_name", None)
                )
                if name is None:
                    raise ValueError(f"Could not infer kernel function name for {fn!r}")

                # Do not register as submodule! Hide it behind a dict to be removed later after registering it
                hidden_kernels[name] = fn

        cls.__init__ = new_init
        return cls

    return decorator


# Whether to allow hub kernels coming from untrusted repos, i.e. repos outside `kernels-community`
ALLOW_ALL_KERNELS = False


@contextmanager
def allow_all_hub_kernels():
    """
    Context manager used to adjust the value of the global `ALLOW_HUB_KERNELS`. This is needed, as this argument
    cannot be forwarded directly to the `__init__` of the models, where we set the attention implementation.
    """
    global ALLOW_ALL_KERNELS

    try:
        ALLOW_ALL_KERNELS = True

        yield
    finally:
        # Set back the original
        ALLOW_ALL_KERNELS = False


class FusedModuleBase(nn.Module):
    def __init__(
        self,
        modules_to_fuse: list["nn.Module"],
        fused_module_names: list[str] | None = None,
    ):
        """
        Args:
            modules_to_fuse: The source modules to fuse together.
            fused_module_names: The names under which each source module is registered as a
                child of this container (i.e. `self.<name>`). When `None`, the
                `kernel_layer_name` attribute of each source module is used. Pass this
                explicitly when the source modules do not carry `@use_kernel_forward_from_hub`.
        """
        super().__init__()
        if len(modules_to_fuse) == 0:
            raise ValueError("At least one module must be provided for fusion.")

        if fused_module_names is not None:
            if len(fused_module_names) != len(modules_to_fuse):
                raise ValueError("Length of fused_module_names and modules_to_fuse must match.")
            for module, name in zip(modules_to_fuse, fused_module_names):
                self.add_module(name, module)
            self._fused_module_names = list(fused_module_names)
        else:
            for module in modules_to_fuse:
                attr_name = getattr(module, "kernel_layer_name", None)
                if attr_name is None:
                    raise ValueError(
                        f"Module {module} does not have a 'kernel_layer_name' attribute. "
                        f"Either decorate it with @use_kernel_forward_from_hub or provide "
                        f"explicit names via the inline pattern format: "
                        f'(("<name>", "<glob_path>"), ...).'
                    )
                self.add_module(attr_name, module)
            self._fused_module_names = [m.kernel_layer_name for m in modules_to_fuse]

        # `kernelize` validates the kernel's forward signature against the class being replaced.
        # Since the fused container sits at the position of the first module in the chain, the
        # kernel's forward must match that module's signature. We patch the class-level forward
        # here (via `functools.wraps`) so the signature is correct when `kernelize` inspects it.
        # The body raises because this forward is always replaced by the kernel before any call.
        @functools.wraps(type(modules_to_fuse[0]).forward)
        def forward(self, *args, **kwargs):
            raise NotImplementedError("FusedModule is a placeholder and should not be called directly.")

        self.__class__.forward = forward

    def __repr__(self):
        names = ", ".join(self._fused_module_names)
        return f"{self.__class__.__name__}(fused=({names}))"


@functools.cache
def make_fused_module_class(source_layer_names: tuple[str, ...], kernel_layer_name: str) -> type:
    """
    Dynamically create and cache a `FusedModuleBase` subclass for a given fusion combination.

    Args:
        source_layer_names (`tuple[str, ...]`):
            Ordered tuple of `kernel_layer_name` values of the modules being fused
            (e.g. `("RMSNorm", "MLP")`). Used as the cache key — the same combination
            always returns the same class object.
        kernel_layer_name (`str`):
            The name assigned to the fused class, used by `kernelize` to look up the
            kernel in the mapping (e.g. `"RMSNormMLP"`).

    Returns:
        A subclass of `FusedModuleBase` with `kernel_layer_name` set as a class attribute.
    """
    return type(
        f"Fused_{'_'.join(source_layer_names)}",
        (FusedModuleBase,),
        {"kernel_layer_name": kernel_layer_name},
    )


def _get_or_create_submodule(
    root: "nn.Module",
    mod_path: str,
    original_module: "nn.Module | None",
    future_weight: "torch.Tensor",
) -> "nn.Module":
    if not mod_path:
        return root
    parent, _, mod_name = mod_path.rpartition(".")
    parent_mod = root.get_submodule(parent) if parent else root
    if mod_name not in parent_mod._modules:
        parent_mod.add_module(mod_name, _create_typed_module(original_module, future_weight))
    return parent_mod._modules[mod_name]


def _apply_weight_conversions(module: "nn.Module", conversion_mapping: list) -> None:
    """Apply WeightConverter transforms to a module via state-dict walk. Handles glob patterns."""
    from ..core_model_loading import dot_natural_key, rename_source_key

    state_dict = module.state_dict()
    pattern_to_converter = {k: c for c in conversion_mapping for k in c.source_patterns}
    param_name_to_transform: dict = {}

    for param_name in sorted(state_dict, key=dot_natural_key):
        renamed_key, source_pattern = rename_source_key(param_name, [], conversion_mapping)
        if source_pattern is None:
            continue
        transform = param_name_to_transform.setdefault(renamed_key, deepcopy(pattern_to_converter[source_pattern]))
        transform.add_tensor(renamed_key, param_name, source_pattern, state_dict[param_name])

    for layer_name, transform in param_name_to_transform.items():
        source_names = list(transform.layer_targets.get(layer_name, []))
        if not source_names:
            continue

        original_mod = None
        for sname in source_names:
            smod_path = sname.rpartition(".")[0]
            try:
                original_mod = module.get_submodule(smod_path) if smod_path else module
                break
            except AttributeError:
                pass

        result = transform.convert(layer_name)

        for sname in source_names:
            mod_path, _, attr = sname.rpartition(".")
            try:
                source_mod = module.get_submodule(mod_path) if mod_path else module
            except AttributeError:
                continue
            source_mod._parameters.pop(attr, None)
            source_mod._buffers.pop(attr, None)

        for target_key, meta_val in result.items():
            meta_tensor = meta_val[0] if isinstance(meta_val, list) else meta_val
            t_mod_path, _, t_attr = target_key.rpartition(".")
            target_mod = _get_or_create_submodule(module, t_mod_path, original_mod, meta_tensor)
            if not isinstance(meta_tensor, nn.Parameter):
                meta_tensor = nn.Parameter(meta_tensor, requires_grad=meta_tensor.is_floating_point())
            target_mod.register_parameter(t_attr, meta_tensor)

        for source_name in source_names:
            mod_path = source_name.rpartition(".")[0]
            if not mod_path:
                continue
            try:
                source_mod = module.get_submodule(mod_path)
            except AttributeError:
                continue
            if (
                all(p is None for p in source_mod._parameters.values())
                and all(b is None for b in source_mod._buffers.values())
                and not source_mod._modules
            ):
                parent_path, _, mod_name = mod_path.rpartition(".")
                parent_mod = module.get_submodule(parent_path) if parent_path else module
                parent_mod._modules.pop(mod_name, None)


def _strip_converter_prefix(converter, prefix: str):
    """Return a new WeightConverter with `prefix.` stripped from all source and target patterns."""
    from ..core_model_loading import WeightConverter

    prefix_dot = prefix + "."
    new_src = [p.removeprefix(prefix_dot) for p in converter._original_source_patterns]
    new_tgt = [p.removeprefix(prefix_dot) for p in converter._original_target_patterns]
    return WeightConverter(new_src, new_tgt, operations=converter.operations)


def _apply_converter_to_module(module: "nn.Module", converter) -> None:
    """
    Apply a single WeightConverter to any nn.Module.

    For literal (non-glob) patterns: locates source submodules directly by path, runs the
    converter operations to infer the output shape, then creates the target submodule with
    the proper typed class (nn.Linear, nn.Embedding, …) and prunes empty source submodules.
    For glob patterns (containing *): falls back to _apply_weight_conversions which does a
    full state-dict walk and registers the result parameter explicitly.
    """
    src_patterns = converter._original_source_patterns

    # Glob patterns require a state-dict walk to expand wildcards.
    if any("*" in p for p in src_patterns):
        _apply_weight_conversions(module, [converter])
        return

    from copy import deepcopy

    from ..core_model_loading import dot_natural_key, rename_source_key

    # Collect source tensors directly from the module — bail if any are missing.
    src_tensors: dict[str, torch.Tensor] = {}
    for pattern in src_patterns:
        mod_path, _, attr = pattern.rpartition(".")
        try:
            src_mod = module.get_submodule(mod_path) if mod_path else module
        except AttributeError:
            return
        t = src_mod._parameters.get(attr)
        if t is None:
            t = src_mod._buffers.get(attr)
        if t is None:
            return
        src_tensors[pattern] = t

    # Feed the source tensors into a converter copy and run the operation to get the
    # result meta tensor (shape inference is free on meta device).
    conv = deepcopy(converter)
    for param_name in sorted(src_tensors, key=dot_natural_key):
        renamed_key, source_pattern = rename_source_key(param_name, [], [conv])
        if source_pattern is not None:
            conv.add_tensor(renamed_key, param_name, source_pattern, src_tensors[param_name])

    target_keys = list(conv.layer_targets)
    if not target_keys:
        return

    # Collect results from all target keys; a converter may produce more than one output tensor.
    result: dict = {}
    for target_key in target_keys:
        result.update(conv.convert(target_key))

    # Infer the typed module class from the first source submodule.
    first_src_path = src_patterns[0].rpartition(".")[0]
    try:
        first_src_mod = module.get_submodule(first_src_path) if first_src_path else module
    except AttributeError:
        first_src_mod = None

    # Remove source parameters before creating the target so shape queries stay clean.
    for pattern in src_patterns:
        mod_path, _, attr = pattern.rpartition(".")
        try:
            src_mod = module.get_submodule(mod_path) if mod_path else module
        except AttributeError:
            continue
        src_mod._parameters.pop(attr, None)
        src_mod._buffers.pop(attr, None)

    # Create the target submodule with the correct shape inferred from the result.
    # The module is initialised with the right in/out dimensions — no parameter replacement needed.
    for result_key, meta_val in result.items():
        meta_tensor = meta_val[0] if isinstance(meta_val, list) else meta_val
        t_mod_path, _, _ = result_key.rpartition(".")

        if t_mod_path:
            parent_path, _, mod_name = t_mod_path.rpartition(".")
            parent_mod = module.get_submodule(parent_path) if parent_path else module
            if mod_name not in parent_mod._modules:
                parent_mod.add_module(mod_name, _create_typed_module(first_src_mod, meta_tensor))

    # Prune source submodules that are now completely empty.
    for pattern in src_patterns:
        mod_path = pattern.rpartition(".")[0]
        if not mod_path:
            continue
        try:
            src_mod = module.get_submodule(mod_path)
        except AttributeError:
            continue
        # _parameters and _buffers may hold None-valued entries (e.g. bias=False in nn.Linear
        # registers _parameters['bias'] = None), so check all-None rather than emptiness.
        # _modules entries are always non-None module objects, so use emptiness there.
        if (
            all(p is None for p in src_mod._parameters.values())
            and all(b is None for b in src_mod._buffers.values())
            and not src_mod._modules
        ):
            parent_path, _, mod_name = mod_path.rpartition(".")
            parent_mod = module.get_submodule(parent_path) if parent_path else module
            parent_mod._modules.pop(mod_name, None)


def _create_typed_module(source_mod: "nn.Module | None", weight: "torch.Tensor") -> "nn.Module":
    """Instantiate a properly-typed nn.Module whose shape matches weight, inferred from source_mod."""
    cls_ = type(source_mod) if source_mod is not None else nn.Module
    if issubclass(cls_, nn.Linear):
        out_features, in_features = weight.shape
        return cls_(in_features=in_features, out_features=out_features, bias=source_mod.bias is not None)
    if issubclass(cls_, nn.Embedding):
        num_embeddings, embedding_dim = weight.shape
        return cls_(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=source_mod.padding_idx,
            max_norm=source_mod.max_norm,
            norm_type=source_mod.norm_type,
            scale_grad_by_freq=source_mod.scale_grad_by_freq,
            sparse=source_mod.sparse,
        )
    raise ValueError(f"Cannot create target module: unsupported source class {cls_.__name__!r}")


def _make_converted_child_class(child_cls: type, stripped_converters: list) -> type:
    """Return a subclass of child_cls whose __init__ applies weight conversions after construction."""
    original_init = child_cls.__init__
    _converters = list(stripped_converters)

    def converted_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for converter in _converters:
            _apply_converter_to_module(self, converter)

    converted_cls = type(f"Converted{child_cls.__name__}", (child_cls,), {"__init__": converted_init})
    converted_cls.__qualname__ = f"Converted{child_cls.__qualname__}"
    return converted_cls


def make_fused_parent_class(
    parent_cls: type,
    child_names: list[str],
    source_names: list[str],
    kernel_layer_name: str,
) -> type:
    original_init = parent_cls.__init__
    _child_names = list(child_names)
    _source_names = list(source_names)

    fused_module_cls = make_fused_module_class(tuple(_source_names), kernel_layer_name)

    def fused_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        modules_to_fuse = [getattr(self, name) for name in _child_names]
        fused = fused_module_cls(
            modules_to_fuse,
            fused_module_names=list(_source_names),
        )
        setattr(self, _child_names[0], fused)
        for name in _child_names[1:]:
            setattr(self, name, nn.Identity())

    fused_cls = type(f"Fused{parent_cls.__name__}", (parent_cls,), {"__init__": fused_init})
    fused_cls.__qualname__ = f"Fused{parent_cls.__qualname__}"
    return fused_cls


def infer_kernel_fusion_transforms(
    patterns_with_names: list[tuple[str, str]],
) -> list[WeightRenaming]:
    """
    Auto-infer WeightRenaming transforms for the path changes caused by FusedModuleBase wrapping.

    For each fusion (source_name, glob_path) pair:
        - The first module's weights get source_name inserted between the module path and the weight suffix.
        - Each subsequent module's weights are relocated from their original path to under the first module.

    For example, given
        [
            ("RMSNorm", "model.layers.*.post_attention_layernorm"),
            ("MLP", "model.layers.*.mlp")
        ]
    the inferred transforms would be:
        - model.layers.0.post_attention_layernorm.weight -> model.layers.0.post_attention_layernorm.RMSNorm.weight
        - model.layers.0.mlp.gate_proj.weight -> model.layers.0.post_attention_layernorm.MLP.gate_proj.weight
    """

    def segment_to_regex(seg: str) -> str:
        """
        Convert a glob segment to a regex segment.
        The wildcard "*" matches any non-empty sequence of characters that does not include a dot.
        Otherwise, the segment is escaped.
        """
        return r"[^.]+" if seg == "*" else re.escape(seg)

    _, first_glob = patterns_with_names[0]
    parent_path, first_child = first_glob.rsplit(".", 1)

    # Convert parent_path to a regex pattern, capturing it as a group for reuse in the target pattern.
    parent_regex = r"\.".join(segment_to_regex(s) for s in parent_path.split("."))

    transforms: list[WeightRenaming] = []

    # We must not match the moved child modules under the first child, so we add a negative lookahead.
    child_regexes = "|".join(rf"{name}\." for name, _ in patterns_with_names)
    moved_child_lookahead = f"(?!{child_regexes})"

    for i, (source_name, glob) in enumerate(patterns_with_names):
        child = glob.rsplit(".", 1)[1]
        child_re = segment_to_regex(child)
        guard = moved_child_lookahead if i == 0 else ""
        transforms.append(
            WeightRenaming(
                source_patterns=rf"({parent_regex}\.){child_re}\.{guard}",
                # \1 refers to the captured parent path
                target_patterns=rf"\1{first_child}.{source_name}\.",
            )
        )

    return transforms


def _first_str_leaf(obj) -> str | None:
    """Recursively extract the first string leaf from a potentially nested dict (device → mode → str)."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            result = _first_str_leaf(v)
            if result is not None:
                return result
    return None


def _try_load_kernel_class(repo_str: str, use_local: bool = False) -> type | None:
    if ":" not in repo_str:
        return None
    repo_id, _, layer_name = repo_str.rpartition(":")
    if not repo_id or not layer_name:
        return None
    try:
        if use_local:
            from pathlib import Path

            package_name = repo_id.rstrip("/").split("/")[-1]
            repo = LocalLayerRepository(
                repo_path=Path(repo_id),
                package_name=package_name,
                layer_name=layer_name,
            )
        else:
            repo = LayerRepository(repo_id=repo_id, layer_name=layer_name)
        return repo.load()
    except Exception:
        return None


def register_kernel_fusions(
    cls: "type[PreTrainedModel]",
    config: "PretrainedConfig",
    kernel_config: "KernelConfig",
) -> None:
    """
    Pre-register hub kernel n-to-1 fusions (tuple keys in KernelConfig) before model instantiation.

    For each inline tuple key `(("Name", "glob.path"), ...)` in `kernel_config.kernel_mapping`:

        1. Meta-instantiate the model to discover parent classes containing all target children.

        2. Register a monkey patch mapping each parent class to a fused subclass whose `__init__`
           wraps the target children in a `FusedModuleBase` with the resolved `kernel_layer_name`.

        3. Load or infer conversion mapping to handle the weight transformations caused by the kernel fusion.

        4. Replace the tuple key with the scalar kernel_layer_name in kernel_config so the
           downstream pipeline (sanitize, create_compatible_mapping, kernelize) is unchanged.
    """

    if not hasattr(cls, "config_class") or not hasattr(cls.config_class, "model_type"):
        raise ValueError(f"Model {cls.__name__} has no config_class or model_type.")
    model_type = cls.config_class.model_type

    new_mapping: dict = {}
    for layer_name, hub_repo in kernel_config.kernel_mapping.items():
        if not isinstance(layer_name, tuple) or not all(
            isinstance(item, tuple) and len(item) == 2 for item in layer_name
        ):
            new_mapping[layer_name] = hub_repo
            continue

        source_names = [item[0] for item in layer_name]
        glob_patterns = [item[1] for item in layer_name]
        child_names = [p.rsplit(".", 1)[1] for p in glob_patterns]

        # Resolve kernel_layer_name from "org/repo:ClassName" or a device/mode-nested dict.
        # All leaf strings in the nested dict point to the same kernel class, so any will do.
        repo_str = _first_str_leaf(hub_repo)
        if repo_str is None:
            raise ValueError(f"Cannot resolve a repo string from hub_repo={hub_repo!r}")
        kernel_layer_name = repo_str.split(":")[1] if ":" in repo_str else repo_str.split("/")[-1]

        # 1. Load the kernel class to get any conversion_mapping it declares.
        kernel_cls = _try_load_kernel_class(repo_str, use_local=kernel_config.use_local_kernel)
        kernel_conversion_mapping = (
            list(kernel_cls.conversion_mapping)
            if kernel_cls is not None and hasattr(kernel_cls, "conversion_mapping")
            else []
        )

        # 2. Meta-device scan — find parent classes that contain all target children.
        with torch.device("meta"):
            meta_model = cls(config)

        seen: set[type] = set()
        seen_children: set[type] = set()
        patch_mapping: dict[str, type] = {}
        for module in meta_model.modules():
            module_cls = type(module)
            if module_cls in seen:
                continue
            if not all(hasattr(module, name) for name in child_names):
                continue
            seen.add(module_cls)

            if kernel_conversion_mapping:
                for source_name, child_name in zip(source_names, child_names):
                    child_cls = type(getattr(module, child_name))
                    if child_cls in seen_children:
                        continue
                    seen_children.add(child_cls)
                    prefix_dot = source_name + "."
                    relevant = [
                        c
                        for c in kernel_conversion_mapping
                        if any(p.startswith(prefix_dot) for p in c._original_source_patterns)
                    ]
                    if relevant:
                        stripped = [_strip_converter_prefix(c, source_name) for c in relevant]
                        patch_mapping[child_cls.__name__] = _make_converted_child_class(child_cls, stripped)

            fused_parent_cls = make_fused_parent_class(module_cls, child_names, source_names, kernel_layer_name)
            patch_mapping[module_cls.__name__] = fused_parent_cls

        if not patch_mapping:
            logger.warning(
                f"No parent modules found containing children {child_names} for kernel fusion "
                f"'{kernel_layer_name}'. Skipping tuple key."
            )
            new_mapping[layer_name] = hub_repo
            continue

        # 3. Register class-level monkey patches.
        register_patch_mapping(patch_mapping, overwrite=True)

        # 4. Infer weight renaming mapping for the loader.
        transforms = infer_kernel_fusion_transforms(list(zip(source_names, glob_patterns)))
        if kernel_conversion_mapping:
            transforms = transforms + kernel_conversion_mapping

        # 5. Load existing conversion mapping for this model type.
        existing = get_checkpoint_conversion_mapping(model_type)
        if existing is not None:
            transforms = existing + transforms

        # 6. Register the combined mappings.
        register_checkpoint_conversion_mapping(model_type, transforms, overwrite=True)

        new_mapping[kernel_layer_name] = hub_repo

    kernel_config.kernel_mapping = new_mapping


__all__ = [
    "FusedModuleBase",
    "LayerRepository",
    "get_kernel",
    "make_fused_module_class",
    "register_kernel_fusions",
    "lazy_load_kernel",
    "register_kernel_mapping",
    "register_kernel_mapping_transformers",
    "replace_kernel_forward_from_hub",
    "use_kernel_forward_from_hub",
    "use_kernel_func_from_hub",
    "use_kernelized_func",
]  # type: ignore
