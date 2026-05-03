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
    from .configuration_utils import PretrainedConfig
    from .modeling_utils import PreTrainedModel
    from .utils.kernel_config import KernelConfig

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
    "finegrained-fp8": {"repo_id": "kernels-community/finegrained-fp8", "version": 1},
    "deep-gemm": {"repo_id": "kernels-community/deep-gemm", "version": 1},
    "sonic-moe": {"repo_id": "kernels-community/sonic-moe", "version": 1},
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
            kernel = get_kernel(repo_id, revision=revision, version=version)
            mapping[kernel_name] = kernel
        except FileNotFoundError:
            mapping[kernel_name] = None
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


# Model class → {kernel_layer_name → [glob patterns]}
# Populated via `register_fusion_patterns` for models that cannot be modified directly.
_FUSION_PATTERNS_REGISTRY: dict[type, dict[str, list[str]]] = {}


def register_fusion_patterns(
    model_class_or_instance,
    patterns: dict[str, list[str]],
) -> None:
    """
    Register kernel fusion patterns for a model class without modifying it directly.

    This is an alternative to setting `_kernel_fusion_patterns` as a class attribute,
    useful when the model class is frozen or comes from an external library.

    Args:
        model_class_or_instance:
            The model class (or an instance of it) for which patterns are being registered.
        patterns (`dict[str, list[str]]`):
            Mapping from `kernel_layer_name` to a list of glob-style module paths,
            identical in format to `_kernel_fusion_patterns`.
    """
    if not isinstance(model_class_or_instance, type):
        model_class_or_instance = type(model_class_or_instance)
    _FUSION_PATTERNS_REGISTRY[model_class_or_instance] = patterns


class FusedModuleBase(nn.Module):
    def __init__(
        self,
        modules_to_fuse: list["nn.Module"],
        source_names: list[str],
        fused_module_names: list[str] | None = None,
    ):
        """
        Args:
            modules_to_fuse: The source modules to fuse together.
            source_names: The attribute names under which each module lives in its parent
                (used to restore them on `unfuse_modules`).
            fused_module_names: The names under which each source module is registered as a
                child of this container (i.e. `self.<name>`). When `None`, the
                `kernel_layer_name` attribute of each source module is used. Pass this
                explicitly when the source modules do not carry `@use_kernel_forward_from_hub`.
        """
        super().__init__()
        if len(modules_to_fuse) == 0:
            raise ValueError("At least one module must be provided for fusion.")
        if len(modules_to_fuse) != len(source_names):
            raise ValueError("Length of modules_to_fuse and source_names must match.")

        self._source_names = source_names

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


def make_fused_parent_class(
    parent_cls: type,
    child_names: list[str],
    source_names: list[str],
    kernel_layer_name: str,
) -> tuple[type, type]:
    original_init = parent_cls.__init__
    _child_names = list(child_names)
    _source_names = list(source_names)
    _kernel_layer_name = kernel_layer_name

    fused_module_cls = make_fused_module_class(tuple(_source_names), _kernel_layer_name)

    def fused_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        modules_to_fuse = [getattr(self, name) for name in _child_names]
        fused = fused_module_cls(modules_to_fuse, _child_names, fused_module_names=list(_source_names))
        setattr(self, _child_names[0], fused)
        for name in _child_names[1:]:
            setattr(self, name, nn.Identity())

    fused_cls = type(f"Fused{parent_cls.__name__}", (parent_cls,), {"__init__": fused_init})
    fused_cls.__qualname__ = f"Fused{parent_cls.__qualname__}"
    return fused_cls, fused_module_cls


def infer_kernel_fusion_transforms(
    patterns_with_names: list[tuple[str, str]],
) -> list[WeightRenaming]:
    """
    Auto-infer WeightRenaming transforms for the path changes caused by FusedModuleBase wrapping.

    For each fusion (source_name, glob_path) pair:
    - The first module's weights get source_name inserted between the module path and the weight suffix.
    - Each subsequent module's weights are relocated from their original path to under the first module.

    For example, given [("RMSNorm", "model.layers.*.post_attention_layernorm"), ("MLP", "model.layers.*.mlp")]:
    - model.layers.0.post_attention_layernorm.weight  -> model.layers.0.post_attention_layernorm.RMSNorm.weight
    - model.layers.0.mlp.gate_proj.weight             -> model.layers.0.post_attention_layernorm.MLP.gate_proj.weight
    """

    def _seg_to_regex(seg: str) -> str:
        return r"[^.]+" if seg == "*" else re.escape(seg)

    def _glob_to_regex(glob: str) -> str:
        return r"\.".join(_seg_to_regex(s) for s in glob.split("."))

    first_source_name, first_glob = patterns_with_names[0]
    *parent_parts, first_child = first_glob.split(".")
    parent_regex = _glob_to_regex(".".join(parent_parts))
    first_child_regex = _seg_to_regex(first_child)

    transforms: list[WeightRenaming] = []

    # Module 0: insert source_name between the module path and the weight suffix.
    # Capture the full module prefix so \1 can be used in the target for variable layer indices.
    transforms.append(
        WeightRenaming(
            source_patterns=rf"({parent_regex}\.{first_child_regex}\.)",
            target_patterns=rf"\1{first_source_name}.",
        )
    )

    # Modules i > 0: relocate weights from their original path to under the first module.
    for source_name, glob in patterns_with_names[1:]:
        *_, child = glob.split(".")
        child_regex = _seg_to_regex(child)
        transforms.append(
            WeightRenaming(
                source_patterns=rf"({parent_regex}\.){child_regex}\.",
                target_patterns=rf"\1{first_child}.{source_name}.",
            )
        )

    return transforms


def _try_load_kernel_class(repo_str: str, use_local: bool = False) -> type | None:
    """
    Load a kernel class from a repo string like "org/repo:ClassName" or "/abs/path:ClassName".

    Returns the loaded class, or None if the string is malformed or loading fails.
    The kernels library caches the result, so repeated calls are free.
    """
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

        2. Register a monkey patch mapping each parent class to a fused subclass whose __init__
           wraps the target children in a FusedModuleBase with the resolved kernel_layer_name.

        3. Load or infer conversion mapping to handle the weight transformations caused by the kernel fusion.

        4. Replace the tuple key with the scalar kernel_layer_name in kernel_config so the
           downstream pipeline (sanitize, create_compatible_mapping, kernelize) is unchanged.

    Scalar keys are passed through untouched.
    """

    if not hasattr(cls, "config_class") or not hasattr(cls.config_class, "model_type"):
        raise ValueError(f"Model {cls.__name__} has no config_class or model_type.")
    model_type = cls.config_class.model_type

    new_mapping: dict = {}
    for layer_name, hub_repo in kernel_config.kernel_mapping.items():
        # Pass scalar keys through unchanged.
        if not isinstance(layer_name, tuple) or not all(
            isinstance(item, tuple) and len(item) == 2 for item in layer_name
        ):
            new_mapping[layer_name] = hub_repo
            continue

        source_names = [item[0] for item in layer_name]
        glob_patterns = [item[1] for item in layer_name]
        child_names = [p.rsplit(".", 1)[1] for p in glob_patterns]

        # Resolve kernel_layer_name from "org/repo:ClassName" or device-nested dict.
        repo_str = hub_repo
        if isinstance(repo_str, dict):
            repo_str = next(iter(repo_str.values()))
        if isinstance(repo_str, dict):
            repo_str = next(iter(repo_str.values()))
        kernel_layer_name = repo_str.split(":")[1] if ":" in repo_str else repo_str.split("/")[-1]

        # 1. Meta-device scan — find parent classes that contain all target children.
        with torch.device("meta"):
            meta_model = cls(config)

        seen: set[type] = set()
        patch_mapping: dict[str, type] = {}
        for module in meta_model.modules():
            module_cls = type(module)
            if module_cls in seen:
                continue
            if not all(hasattr(module, name) for name in child_names):
                continue
            seen.add(module_cls)
            fused_parent_cls, fused_cls = make_fused_parent_class(module_cls, child_names, source_names, kernel_layer_name)
            patch_mapping[module_cls.__name__] = fused_parent_cls

        if not patch_mapping:
            logger.warning(
                f"No parent modules found containing children {child_names} for kernel fusion "
                f"'{kernel_layer_name}'. Skipping tuple key."
            )
            new_mapping[layer_name] = hub_repo
            continue

        # 2. Register class-level monkey patches.
        register_patch_mapping(patch_mapping, overwrite=True)

        # 3. Register weight conversion mapping.
        # Always start with inferred transforms for the basic path renaming caused by fusion
        # (e.g. layernorm.weight → layernorm.RMSNorm.weight, mlp.* → layernorm.MLP.*).
        # Then append any kernel-provided custom transforms (e.g. weight concatenation).
        # The kernels library caches the result, so the later kernelize() call is a free cache hit.
        transforms = infer_kernel_fusion_transforms(list(zip(source_names, glob_patterns)))
        kernel_cls = _try_load_kernel_class(repo_str, use_local=kernel_config.use_local_kernel)
        if kernel_cls is not None and hasattr(kernel_cls, "conversion_mapping"):
            transforms = transforms + list(kernel_cls.conversion_mapping)

        existing = get_checkpoint_conversion_mapping(model_type)
        if existing is not None:
            transforms = existing + transforms

        register_checkpoint_conversion_mapping(model_type, transforms, overwrite=True)

        # 4. Resolve tuple key -> scalar key for the downstream pipeline.
        new_mapping[kernel_layer_name] = hub_repo

    kernel_config.kernel_mapping = new_mapping


__all__ = [
    "FusedModuleBase",
    "LayerRepository",
    "LocalLayerRepository",
    "get_kernel",
    "register_kernel_fusions",
    "lazy_load_kernel",
    "make_fused_module_class",
    "register_fusion_patterns",
    "register_kernel_mapping",
    "register_kernel_mapping_transformers",
    "replace_kernel_forward_from_hub",
    "use_kernel_forward_from_hub",
    "use_kernel_func_from_hub",
    "use_kernelized_func",
]  # type: ignore
