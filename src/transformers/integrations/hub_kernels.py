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
import os
import re
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from ..conversion_mapping import get_checkpoint_conversion_mapping, register_checkpoint_conversion_mapping
from ..monkey_patching import register_patch_mapping
from ..utils import ENV_VARS_TRUE_VALUES, logging
from ..utils.generic import is_flash_attention_requested
from ..utils.import_utils import KERNELS_MAX_VERSION, KERNELS_MIN_VERSION, is_kernels_available, is_torch_available
from .flash_attention import flash_attention_forward


if TYPE_CHECKING:
    from ..configuration_utils import PretrainedConfig
    from ..modeling_utils import PreTrainedModel
    from ..utils.kernel_config import KernelConfig

if is_torch_available():
    import torch
    import torch.nn as nn


logger = logging.get_logger(__name__)


_MISSING_KERNELS_MESSAGE = (
    "`kernels` is either not installed or uses an incompatible version. Please install a compatible version "
    f"({KERNELS_MIN_VERSION} <= version < {KERNELS_MAX_VERSION}), e.g. `pip install kernels=={KERNELS_MIN_VERSION}`"
)


if is_kernels_available():
    from kernels import (
        Device,
        FuncRepository,
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
    from kernels import use_kernel_func_from_hub as _kernels_use_kernel_func_from_hub

    _TRANSFORMERS_USE_HUB_KERNELS = os.environ.get("USE_HUB_KERNELS", "YES").upper()
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
        if _kernels_enabled:
            return _kernels_use_kernel_func_from_hub(func_name)
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
                version=1,
            )
        },
        # NOTE: No longer maintained
        # "Llama4TextMoe": {
        #    "cuda": LayerRepository(
        #        repo_id="kernels-community/moe",
        #        layer_name="Llama4TextMoe",
        #        version=1,
        #    )
        # },
        "SwiGLUMLP": {
            "cuda": {
                Mode.INFERENCE | Mode.TORCH_COMPILE: LayerRepository(
                    repo_id="kernels-community/liger-kernels",
                    layer_name="LigerSwiGLUMLP",
                    version=2,
                ),
                Mode.TRAINING | Mode.TORCH_COMPILE: LayerRepository(
                    repo_id="kernels-community/liger-kernels",
                    layer_name="LigerTiledSwiGLUMLP",
                    version=2,
                ),
            },
        },
        "GeGLUMLP": {
            "cuda": {
                Mode.INFERENCE | Mode.TORCH_COMPILE: LayerRepository(
                    repo_id="kernels-community/liger-kernels",
                    layer_name="LigerGEGLUMLP",
                    version=2,
                ),
                Mode.TRAINING | Mode.TORCH_COMPILE: LayerRepository(
                    repo_id="kernels-community/liger-kernels",
                    layer_name="LigerTiledGEGLUMLP",
                    version=2,
                ),
            },
        },
        "Linear": {
            "cuda": {
                Mode.TRAINING | Mode.TORCH_COMPILE: LayerRepository(
                    repo_id="kernels-community/liger-kernels",
                    layer_name="LigerLinear",
                    version=2,
                ),
            },
        },
        "RMSNorm": {
            # NOTE: Not torch.compile friendly for unknown reasons
            "cuda": {
                Mode.TRAINING: LayerRepository(
                    repo_id="kernels-community/liger-kernels",
                    layer_name="LigerRMSNorm",
                    version=2,
                ),
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/liger-kernels",
                    layer_name="LigerRMSNorm",
                    version=2,
                ),
            },
            "rocm": {
                Mode.TRAINING: LayerRepository(
                    repo_id="kernels-community/liger-kernels",
                    layer_name="LigerRMSNorm",
                    version=2,
                ),
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/liger-kernels",
                    layer_name="LigerRMSNorm",
                    version=2,
                ),
            },
            "xpu": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/rmsnorm",
                    layer_name="RMSNorm",
                    version=1,
                )
            },
            "mps": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/mlx_rmsnorm",
                    layer_name="RMSNorm",
                    version=1,
                )
            },
            "npu": {
                Mode.TRAINING: LayerRepository(
                    repo_id="kernels-community/liger-kernels",
                    layer_name="LigerRMSNorm",
                    version=2,
                ),
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/liger-kernels",
                    layer_name="LigerRMSNorm",
                    version=2,
                ),
            },
        },
        "MegaBlocksMoeMLP": {
            "cuda": {
                Mode.TRAINING: LayerRepository(
                    repo_id="kernels-community/megablocks",
                    layer_name="MegaBlocksMoeMLP",
                    version=1,
                ),
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/megablocks",
                    layer_name="MegaBlocksMoeMLP",
                    version=1,
                ),
            },
            "rocm": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="ahadnagy/megablocks",
                    layer_name="MegaBlocksMoeMLP",
                    version=1,
                )
            },
            "xpu": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/megablocks",
                    layer_name="MegaBlocksMoeMLP",
                    version=1,
                )
            },
            "cpu": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/megablocks",
                    layer_name="CPUMegaBlocksMoeMLP",
                    version=1,
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

    # Add function kernel mappings
    _FUNCTION_KERNEL_MAPPING = {
        "rotary_pos_emb": {
            "xpu": {
                Mode.INFERENCE: FuncRepository(
                    repo_id="kernels-community/rotary", func_name="apply_rotary_transformers", version=1
                )
            },
            "cuda": FuncRepository(
                repo_id="kernels-community/rotary", func_name="apply_rotary_transformers", version=1
            ),
        },
        "ForCausalLMLoss": {
            "cuda": {
                Mode.TRAINING | Mode.TORCH_COMPILE: FuncRepository(
                    repo_id="kernels-community/liger-kernels", func_name="LigerForCausalLMLoss", version=2
                ),
            },
        },
    }
    _KERNEL_MAPPING = _KERNEL_MAPPING | _FUNCTION_KERNEL_MAPPING

    def has_key(d, key):
        return key in d or any(isinstance(v, dict) and has_key(v, key) for v in d.values())

    def register_kernel_mapping_transformers(mapping=None):
        if mapping is None:
            mapping = _KERNEL_MAPPING
        register_kernel_mapping(mapping)

else:
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

    class FuncRepository:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("FuncRepository requires `kernels` to be installed. Run `pip install kernels`.")

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
    if not is_kernels_available():
        raise ImportError(_MISSING_KERNELS_MESSAGE)

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

    # create revision xor version
    rev = rev.strip() if rev else None
    version = None
    if rev is None:
        # FA4 is still in beta -> redirect to v0 else default to v1
        is_fa4 = is_flash_attention_requested(requested_attention_implementation=repo_id, version=4)
        version = 0 if is_fa4 else 1

    # Load the kernel from hub
    try:
        kernel = get_kernel(repo_id, revision=rev, version=version, allow_all_kernels=allow_all_kernels)
    except ValueError:
        raise
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
    if is_kernels_available():
        try:
            repo_id = _HUB_KERNEL_MAPPING[kernel_name]["repo_id"]
            revision = _HUB_KERNEL_MAPPING[kernel_name].get("revision", None)
            version = _HUB_KERNEL_MAPPING[kernel_name].get("version", None)
            # Default version as it's mandatory
            if version is None and revision is None:
                version = 1

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

    if not is_kernels_available():
        raise ImportError(_MISSING_KERNELS_MESSAGE)

    user_agent = {"framework": "transformers", "version": __version__, "repo_id": kernel_name}
    return get_kernel_hub(
        kernel_name, revision=revision, version=version, user_agent=user_agent, trust_remote_code=allow_all_kernels
    )


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


def make_parent_class_for_kernel_fusion(
    parent_cls: type,
    child_names: list[str],
    kernel_cls: type,
) -> type:
    """
    Create a new class that inherits from `parent_cls` and fuses the child modules specified in `child_names
    with the provided `kernel_cls`.
    The first child in `child_names` will be replaced with the `kernel_cls`, and the rest will be replaced with
    `nn.Identity()` to keep the same interface.
    """
    original_init = parent_cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        children = [getattr(self, name) for name in child_names]
        kernel_instance = kernel_cls(*children)
        setattr(self, child_names[0], kernel_instance)
        for name in child_names[1:]:
            setattr(self, name, nn.Identity())

    patched_cls = type(f"Fused{parent_cls.__name__}", (parent_cls,), {"__init__": patched_init})
    patched_cls.__qualname__ = f"Fused{parent_cls.__qualname__}"
    return patched_cls


def register_kernel_replacements_and_fusions(
    cls: "type[PreTrainedModel]",
    config: "PretrainedConfig",
    kernel_config: "KernelConfig",
) -> None:
    if not hasattr(cls, "config_class") or not hasattr(cls.config_class, "model_type"):
        raise ValueError(f"Model {cls.__name__} has no config_class or model_type.")
    model_type = cls.config_class.model_type

    patch_mapping: dict[str, type] = {}
    new_mapping: dict = {}

    # We might need to instantiate the model on meta device.
    # We do it lazily, only if we encounter a fused kernel.
    meta_model = None

    for layer_name, hub_repo in kernel_config.kernel_mapping.items():
        if isinstance(hub_repo, (str, tuple)):
            hub_repo = {None: hub_repo}

        if isinstance(hub_repo, dict):
            if len(hub_repo.values()) != 1:
                raise ValueError(
                    f"Expected exactly one kernel repo regardless of device/mode specificity, got {hub_repo}"
                )
        else:
            raise ValueError(f"Invalid hub repo {hub_repo!r} for layer {layer_name!r}")

        hub_repo = next(iter(hub_repo.values()))

        # Infer metadata (revision/version/trust_remote_code)
        if isinstance(hub_repo, tuple):
            repo_str, metadata = hub_repo

            revision = metadata.get("revision", None)
            version = metadata.get("version", None)
            trust_remote_code = metadata.get("trust_remote_code", False) or ALLOW_ALL_KERNELS
            metadata = {"version": version} if version is not None else {"revision": revision}
            metadata |= {"trust_remote_code": trust_remote_code}

            final_repo = (repo_str, metadata)
        else:
            repo_str = hub_repo
            metadata = {"version": 1, "trust_remote_code": ALLOW_ALL_KERNELS}
            final_repo = (repo_str, metadata)

        repo_id, _, layer_name_in_repo = repo_str.partition(":")
        if not repo_id or not layer_name_in_repo:
            raise ValueError(f"Invalid kernel repo string {repo_str!r} for layer {layer_name!r}")

        if kernel_config.use_local_kernel:
            package_name = repo_id.rstrip("/").split("/")[-1]
            repo = LocalLayerRepository(
                repo_path=Path(repo_id),
                package_name=package_name,
                layer_name=layer_name_in_repo,
            )
        else:
            repo = LayerRepository(
                repo_id=repo_id,
                layer_name=layer_name_in_repo,
                **metadata,
            )

        kernel_cls = repo.load()

        if kernel_cls is None:
            raise ValueError(f"Could not load kernel class from hub_repo={hub_repo!r}")

        kernel_mod = sys.modules.get(kernel_cls.__module__)
        layout_cls = getattr(kernel_mod, f"{kernel_cls.__name__}Layout", None) if kernel_mod else None

        # Case 1: no fusion.
        if isinstance(layer_name, str):
            # No layout class: stateless kernel, leave for kernels.kernelize.
            if layout_cls is None:
                new_mapping[layer_name] = final_repo
                continue

            # Register the layout class as a monkey patch for the parent module containing the target layer.
            layout_cls.kernel_layer_name = kernel_cls.__name__
            patch_mapping[layer_name] = layout_cls

            # Keep the original repo string so kernelize can replace the layout's forward.
            new_mapping[kernel_cls.__name__] = final_repo

        # Case 2: fusion.
        elif isinstance(layer_name, tuple):
            if layout_cls is None:
                raise ValueError(
                    f"Fused kernel {kernel_cls.__name__!r} requires a companion layout class "
                    f"named '{kernel_cls.__name__}Layout' in the same module."
                )

            layout_cls.kernel_layer_name = kernel_cls.__name__

            glob_patterns = [item[1] for item in layer_name]
            parent_patterns = [p.rsplit(".", 1)[0] for p in glob_patterns]

            if len(set(parent_patterns)) != 1:
                raise ValueError(
                    f"All patterns for a fused kernel must share the same parent module, got {glob_patterns}"
                )

            parent_pattern = parent_patterns[0].replace("*", r"\w+")
            child_names = [p.rsplit(".", 1)[1] for p in glob_patterns]

            if meta_model is None:
                with torch.device("meta"):
                    meta_model = cls(config)

            matched_any = False
            for name, module in meta_model.named_modules():
                if not re.fullmatch(parent_pattern, name):
                    continue
                if not all(hasattr(module, child) for child in child_names):
                    raise ValueError(
                        f"Module {name!r} does not have the expected child modules {child_names} required for "
                        f"the fused kernel {kernel_cls.__name__!r}"
                    )
                matched_any = True
                module_cls = type(module)
                patch_mapping[module_cls.__name__] = make_parent_class_for_kernel_fusion(
                    module_cls, child_names, layout_cls
                )

            if not matched_any:
                raise ValueError(
                    f"No module matched pattern {parent_pattern!r} for fused kernel {kernel_cls.__name__!r}. "
                    f"Provide the full dotted path from the model root."
                )

        register_patch_mapping(patch_mapping, overwrite=True)

        if hasattr(layout_cls, "conversion_mapping"):
            existing = get_checkpoint_conversion_mapping(model_type)
            transforms = list(layout_cls.conversion_mapping)
            if existing is not None:
                transforms = existing + transforms
            register_checkpoint_conversion_mapping(model_type, transforms, overwrite=True)

        new_mapping[kernel_cls.__name__] = final_repo

    kernel_config.kernel_mapping = new_mapping


__all__ = [
    "LayerRepository",
    "get_kernel",
    "lazy_load_kernel",
    "register_kernel_mapping",
    "register_kernel_mapping_transformers",
    "register_kernel_replacements_and_fusions",
    "replace_kernel_forward_from_hub",
    "use_kernel_forward_from_hub",
    "use_kernel_func_from_hub",
    "use_kernelized_func",
]  # type: ignore
