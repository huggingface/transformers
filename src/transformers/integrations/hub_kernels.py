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

from packaging import version as pkg_version

from ..utils import ENV_VARS_TRUE_VALUES, logging
from ..utils.import_utils import is_kernels_available, is_torch_available
from .flash_attention import flash_attention_forward


if is_torch_available():
    import torch.nn as nn


logger = logging.get_logger(__name__)

try:
    from kernels import (
        Device,
        LayerRepository,
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


def fuse_modules(
    model: "nn.Module",
    module_names_to_fuse: list[str],
    kernel_layer_name: str,
    source_layer_names: list[str] | None = None,
) -> None:
    """
    Fuse a sequence of submodules into a single `FusedModuleBase` subclass in-place.

    For every parent module whose immediate children match all entries in
    `module_names_to_fuse`, the function:

    - replaces the first module with a `FusedModuleBase` subclass instance that holds
      all source modules as named children,
    - replaces the remaining modules with `nn.Identity()` pass-throughs.

    The fused container's `forward` signature is patched to match the first source
    module's `forward`, satisfying the `kernelize` signature check.

    Args:
        model (`nn.Module`):
            The model to modify in-place.
        module_names_to_fuse (`list[str]`):
            Glob-style paths of the modules to fuse, e.g.
            `["model.layers.*.post_attention_layernorm", "model.layers.*.mlp"]`.
            Integer indices are replaced with `*` so the same pattern applies to
            every repeated block.
        kernel_layer_name (`str`):
            The `kernel_layer_name` assigned to the fused class, used by `kernelize`
            to look up the kernel in the mapping (e.g. `"RMSNormMLP"`).
        source_layer_names (`list[str]`, *optional*):
            Explicit names for the child modules inside the fused container
            (e.g. `["RMSNorm", "MLP"]`). When `None`, the `kernel_layer_name`
            attribute of each source module is used.
    """
    pattern = re.compile(r"\d+")
    for module_name, module in model.named_modules():
        generic_children = {
            re.sub(pattern, "*", f"{module_name}.{n}" if module_name else n): (n, child)
            for n, child in module.named_children()
        }
        if not all(p in generic_children for p in module_names_to_fuse):
            continue

        child_names = [generic_children[p][0] for p in module_names_to_fuse]
        modules_to_fuse = [generic_children[p][1] for p in module_names_to_fuse]

        if source_layer_names is not None:
            resolved_names = tuple(source_layer_names)
        else:
            resolved_names = tuple(getattr(m, "kernel_layer_name") for m in modules_to_fuse)
        FusedClass = make_fused_module_class(resolved_names, kernel_layer_name)
        fused_instance = FusedClass(modules_to_fuse, child_names, fused_module_names=list(resolved_names))

        module.add_module(child_names[0], fused_instance)
        for child_name in child_names[1:]:
            module.add_module(child_name, nn.Identity())


def unfuse_modules(model: "nn.Module") -> None:
    """
    Revert a previous `fuse_modules` call in-place, restoring the original modules.

    For each `FusedModuleBase` instance found in the model tree, the function:

    - restores the original first module at the fused container's position,
    - restores the remaining original modules at their original positions
      (replacing the `nn.Identity()` pass-throughs).

    Args:
        model (`nn.Module`): The model to restore in-place.
    """
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if not isinstance(child, FusedModuleBase):
                continue
            orig_modules = [getattr(child, layer_name) for layer_name in child._fused_module_names]
            parent.add_module(name, orig_modules[0])
            for sibling_name, orig_module in zip(child._source_names[1:], orig_modules[1:]):
                parent.add_module(sibling_name, orig_module)


__all__ = [
    "FusedModuleBase",
    "LayerRepository",
    "fuse_modules",
    "get_kernel",
    "lazy_load_kernel",
    "make_fused_module_class",
    "register_fusion_patterns",
    "register_kernel_mapping",
    "register_kernel_mapping_transformers",
    "replace_kernel_forward_from_hub",
    "unfuse_modules",
    "use_kernel_forward_from_hub",
    "use_kernel_func_from_hub",
    "use_kernelized_func",
]  # type: ignore
