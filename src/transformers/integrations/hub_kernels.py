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
import importlib.metadata
import os
import re
import sys
from collections.abc import Callable
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING

from packaging import version as pkg_version

from ..conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
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
    "finegrained-fp8": {"repo_id": "kernels-community/finegrained-fp8", "version": 1},
    "deep-gemm": {"repo_id": "kernels-community/deep-gemm", "version": 1},
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
            kernel = get_kernel(repo_id, revision=revision, version=version)
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


def make_kernel_init_parent_class(
    parent_cls: type,
    child_names: list[str],
    kernel_cls: type,
) -> type:
    original_init = parent_cls.__init__
    _child_names = list(child_names)
    _kernel_cls = kernel_cls

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        children = [getattr(self, name) for name in _child_names]
        kernel_instance = _kernel_cls(*children)
        setattr(self, _child_names[0], kernel_instance)
        for name in _child_names[1:]:
            setattr(self, name, nn.Identity())

    patched_cls = type(f"Fused{parent_cls.__name__}", (parent_cls,), {"__init__": patched_init})
    patched_cls.__qualname__ = f"Fused{parent_cls.__qualname__}"
    return patched_cls


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


def _find_layout_class(kernel_cls: type) -> type | None:
    """
    Look for a companion layout class named ``{kernel_cls.__name__}Layout`` in the
    same module as kernel_cls.
    """
    module = sys.modules.get(kernel_cls.__module__)
    if module is None:
        return None
    return getattr(module, f"{kernel_cls.__name__}Layout", None)


def register_kernel_replacements(
    cls: "type[PreTrainedModel]",
    kernel_config: "KernelConfig",
) -> None:
    if not hasattr(cls, "config_class") or not hasattr(cls.config_class, "model_type"):
        raise ValueError(f"Model {cls.__name__} has no config_class or model_type.")
    model_type = cls.config_class.model_type

    patch_mapping: dict[str, type] = {}
    new_mapping: dict = {}

    for layer_name, hub_repo in kernel_config.kernel_mapping.items():
        # Fusion case: handled by register_kernel_fusions, leave it alone.
        if isinstance(layer_name, tuple):
            new_mapping[layer_name] = hub_repo
            continue

        if isinstance(hub_repo, dict):
            if len(hub_repo.values()) != 1:
                raise ValueError(
                    f"Expected exactly one kernel repo regardless of device/mode specificity, got {hub_repo}"
                )
            repo_str = next(iter(hub_repo.values()))
        else:
            repo_str = hub_repo

        kernel_cls = _try_load_kernel_class(repo_str, use_local=kernel_config.use_local_kernel)

        if kernel_cls is None:
            new_mapping[layer_name] = hub_repo
            continue

        # Look for a companion layout class named "{kernel_cls.__name__}Layout".
        # If found, it handles __init__ (weight layout); kernel_cls handles forward.
        layout_cls = _find_layout_class(kernel_cls)
        if layout_cls is None:
            # No layout class: stateless kernel, leave for kernels.kernelize.
            new_mapping[layer_name] = hub_repo
            continue

        layout_cls.kernel_layer_name = kernel_cls.__name__
        patch_mapping[layer_name] = layout_cls

        # Keep the original repo string so kernelize can replace the layout's forward.
        new_mapping[kernel_cls.__name__] = hub_repo

        if hasattr(layout_cls, "conversion_mapping"):
            existing = get_checkpoint_conversion_mapping(model_type)
            transforms = list(layout_cls.conversion_mapping)
            if existing is not None:
                transforms = existing + transforms
            register_checkpoint_conversion_mapping(model_type, transforms, overwrite=True)

    if patch_mapping:
        register_patch_mapping(patch_mapping, overwrite=True)

    kernel_config.kernel_mapping = new_mapping


def register_kernel_fusions(
    cls: "type[PreTrainedModel]",
    config: "PretrainedConfig",
    kernel_config: "KernelConfig",
) -> None:
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

        glob_patterns = [item[1] for item in layer_name]
        child_names = [p.rsplit(".", 1)[1] for p in glob_patterns]

        # 1. Resolve the kernel class — accept a direct class reference or a repo string.
        if isinstance(hub_repo, dict):
            if len(hub_repo.values()) != 1:
                raise ValueError(
                    f"Expected exactly one kernel repo regardless of device/mode specificity, got {hub_repo}"
                )
            repo_str = next(iter(hub_repo.values()))
        else:
            repo_str = hub_repo

        kernel_cls = _try_load_kernel_class(repo_str, use_local=kernel_config.use_local_kernel)

        if kernel_cls is None:
            raise ValueError(f"Could not load kernel class from hub_repo={hub_repo!r}")

        layout_cls = _find_layout_class(kernel_cls)
        if layout_cls is None:
            raise ValueError(
                f"Fused kernel {kernel_cls.__name__!r} requires a companion layout class "
                f"named '{kernel_cls.__name__}Layout' in the same module. "
                f"Define it with __init__(self, {', '.join(child_names)}) to set up the "
                f"fused parameter layout."
            )

        layout_cls.kernel_layer_name = kernel_cls.__name__

        # 2. Meta-device scan — find parent classes that own all target children.
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
            patch_mapping[module_cls.__name__] = make_kernel_init_parent_class(module_cls, child_names, layout_cls)

        if not patch_mapping:
            logger.warning(
                f"No parent modules found containing children {child_names} for kernel fusion. "
                f"Skipping tuple key {layer_name!r}."
            )
            new_mapping[layer_name] = hub_repo
            continue

        # 3. Register class-level monkey patches.
        register_patch_mapping(patch_mapping, overwrite=True)

        # 4. Register the layout's conversion_mapping if it declares one.
        if hasattr(layout_cls, "conversion_mapping"):
            existing = get_checkpoint_conversion_mapping(model_type)
            transforms = list(layout_cls.conversion_mapping)
            if existing is not None:
                transforms = existing + transforms
            register_checkpoint_conversion_mapping(model_type, transforms, overwrite=True)

        # 5. Keep the original repo string: kernelize loads the forward-only kernel_cls
        #    and replaces the layout instances' forward via kernel_layer_name.
        new_mapping[kernel_cls.__name__] = hub_repo

    kernel_config.kernel_mapping = new_mapping


__all__ = [
    "LayerRepository",
    "get_kernel",
    "register_kernel_fusions",
    "register_kernel_replacements",
    "lazy_load_kernel",
    "register_kernel_mapping",
    "register_kernel_mapping_transformers",
    "replace_kernel_forward_from_hub",
    "use_kernel_forward_from_hub",
    "use_kernel_func_from_hub",
    "use_kernelized_func",
]  # type: ignore
