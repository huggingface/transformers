# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
import collections
import copy
import functools
import importlib.metadata
import inspect
import json
import os
import re
import sys
import warnings
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import partial, wraps
from itertools import cycle
from threading import Thread
from typing import Optional, TypeVar, get_type_hints
from zipfile import is_zipfile

import torch
from huggingface_hub import create_repo, is_offline_mode, split_torch_state_dict_into_shards
from packaging import version
from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file
from torch import Tensor, nn
from torch.distributions import constraints
from torch.utils.checkpoint import checkpoint

from . import initialization as init
from .configuration_utils import PreTrainedConfig
from .conversion_mapping import get_model_conversion_mapping
from .core_model_loading import (
    WeightConverter,
    WeightRenaming,
    convert_and_load_state_dict_in_model,
    revert_weight_conversion,
)
from .distributed import DistributedConfig
from .dynamic_module_utils import custom_object_save
from .generation import CompileConfig, GenerationConfig
from .integrations import PeftAdapterMixin, deepspeed_config, is_deepspeed_zero3_enabled, is_fsdp_enabled
from .integrations.accelerate import (
    _get_device_map,
    accelerate_disk_offload,
    accelerate_dispatch,
    check_and_set_device_map,
    expand_device_map,
    get_device,
    load_offloaded_parameter,
)
from .integrations.deepspeed import _load_state_dict_into_zero3_model
from .integrations.eager_paged import eager_paged_attention_forward
from .integrations.flash_attention import flash_attention_forward
from .integrations.flash_paged import paged_attention_forward
from .integrations.flex_attention import flex_attention_forward
from .integrations.hub_kernels import is_kernel
from .integrations.peft import maybe_load_adapters
from .integrations.sdpa_attention import sdpa_attention_forward
from .integrations.sdpa_paged import sdpa_attention_paged_forward
from .integrations.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    _get_parameter_tp_plan,
    distribute_model,
    gather_state_dict_for_save,
    initialize_tensor_parallelism,
    shard_and_distribute_module,
    verify_tp_plan,
)
from .loss.loss_utils import LOSS_MAPPING
from .modeling_flash_attention_utils import lazy_import_flash_attention, lazy_import_paged_flash_attention
from .modeling_rope_utils import ROPE_INIT_FUNCTIONS
from .pytorch_utils import id_tensor_storage
from .quantizers import HfQuantizer
from .quantizers.auto import get_hf_quantizer
from .quantizers.quantizers_utils import get_module_from_name
from .safetensors_conversion import auto_conversion
from .utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    DUMMY_INPUTS,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    ContextManagers,
    KernelConfig,
    PushToHubMixin,
    cached_file,
    check_torch_load_is_safe,
    copy_func,
    has_file,
    is_accelerate_available,
    is_bitsandbytes_available,
    is_env_variable_true,
    is_flash_attn_2_available,
    is_flash_attn_3_available,
    is_kernels_available,
    is_torch_flex_attn_available,
    is_torch_mlu_available,
    is_torch_npu_available,
    is_torch_xpu_available,
    logging,
)
from .utils.generic import GeneralInterface, is_flash_attention_requested
from .utils.hub import DownloadKwargs, create_and_tag_model_card, get_checkpoint_shard_files
from .utils.import_utils import (
    is_huggingface_hub_greater_or_equal,
    is_sagemaker_mp_enabled,
    is_tracing,
)
from .utils.loading_report import LoadStateDictInfo, log_state_dict_report
from .utils.output_capturing import _CAN_RECORD_REGISTRY, OutputRecorder
from .utils.quantization_config import QuantizationMethod


if is_accelerate_available():
    from accelerate.hooks import add_hook_to_module
    from accelerate.utils import extract_model_from_parallel


_torch_distributed_available = torch.distributed.is_available()

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


logger = logging.get_logger(__name__)

XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0").upper()
XLA_DOWNCAST_BF16 = os.environ.get("XLA_DOWNCAST_BF16", "0").upper()
SpecificPreTrainedModelType = TypeVar("SpecificPreTrainedModelType", bound="PreTrainedModel")
_is_quantized = False
_is_ds_init_called = False

# Mapping from flash attention implementations to their kernel fallback repositories
FLASH_ATTN_KERNEL_FALLBACK = {
    "flash_attention_2": "kernels-community/flash-attn2",
    "flash_attention_3": "kernels-community/vllm-flash-attn3",
}


@dataclass(frozen=True)
class LoadStateDictConfig:
    """
    Config for loading weights. This allows bundling arguments that are just
    passed around.
    """

    pretrained_model_name_or_path: str | None = None
    download_kwargs: DownloadKwargs | None = field(default_factory=DownloadKwargs)
    use_safetensors: bool | None = None
    ignore_mismatched_sizes: bool = False
    sharded_metadata: dict | None = None
    device_map: dict | None = None
    disk_offload_folder: str | None = None
    offload_buffers: bool = False
    dtype: torch.dtype | None = None
    dtype_plan: dict = field(default_factory=dict)
    hf_quantizer: HfQuantizer | None = None
    device_mesh: Optional["torch.distributed.device_mesh.DeviceMesh"] = None
    weights_only: bool = True
    weight_mapping: list[WeightConverter | WeightRenaming] | None = None

    @property
    def is_quantized(self) -> bool:
        return self.hf_quantizer is not None


def is_local_dist_rank_0():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and int(os.environ.get("LOCAL_RANK", "-1")) == 0
    )


@contextmanager
def set_quantized_state():
    global _is_quantized
    _is_quantized = True
    try:
        yield
    finally:
        _is_quantized = False


# Skip recursive calls to deepspeed.zero.Init to avoid pinning errors.
# This issue occurs with ZeRO stage 3 when using NVMe offloading.
# For more details, refer to issue #34429.
@contextmanager
def set_zero3_state():
    global _is_ds_init_called
    _is_ds_init_called = True
    try:
        yield
    finally:
        _is_ds_init_called = False


@contextmanager
def local_torch_dtype(dtype: torch.dtype, model_class_name: str | None = None):
    """
    Locally change the torch default dtype to `dtype`, and restore the old one upon exiting the context.
    If `model_class_name` is provided, it's used to provide a more helpful error message if `dtype` is not valid.
    """
    # Just a more helping error before we set `torch.set_default_dtype` later on which would crash in this case
    if not dtype.is_floating_point:
        if model_class_name is not None:
            error_message = (
                f"{model_class_name} cannot be instantiated under `dtype={dtype}` as it's not a floating-point dtype"
            )
        else:
            error_message = f"Cannot set `{dtype}` as torch's default as it's not a floating-point dtype"
        raise ValueError(error_message)

    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(original_dtype)


def get_torch_context_manager_or_global_device():
    """
    Test if a device context manager is currently in use, or if it is not the case, check if the default device
    is not "cpu". This is used to infer the correct device to load the model on, in case `device_map` is not provided.
    """
    device_in_context = torch.tensor([]).device
    default_device = torch.get_default_device()
    # This case means no context manager was used -> we still check if the default that was potentially set is not cpu
    if device_in_context == default_device:
        if default_device != torch.device("cpu"):
            return default_device
        return None
    return device_in_context


def get_state_dict_dtype(state_dict):
    """
    Returns the first found floating dtype in `state_dict` if there is one, otherwise returns the first dtype.
    """
    for t in state_dict.values():
        if t.is_floating_point():
            return t.dtype

    # if no floating dtype was found return whatever the first dtype is
    if len(state_dict) == 0:
        return torch.float32
    return next(iter(state_dict.values())).dtype


str_to_torch_dtype = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "U16": torch.uint16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "U32": torch.uint32,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "U64": torch.uint64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def load_state_dict(
    checkpoint_file: str | os.PathLike, map_location: str | torch.device = "cpu", weights_only: bool = True
) -> dict[str, torch.Tensor]:
    """
    Reads a `safetensor` or a `.bin` checkpoint file. We load the checkpoint on "cpu" by default.
    """
    # Use safetensors if possible
    if checkpoint_file.endswith(".safetensors"):
        with safe_open(checkpoint_file, framework="pt") as f:
            state_dict = {}
            for k in f.keys():
                if map_location == "meta":
                    _slice = f.get_slice(k)
                    k_dtype = _slice.get_dtype()
                    if k_dtype in str_to_torch_dtype:
                        dtype = str_to_torch_dtype[k_dtype]
                    else:
                        raise ValueError(f"Cannot load safetensors of unknown dtype {k_dtype}")
                    state_dict[k] = torch.empty(size=_slice.get_shape(), dtype=dtype, device="meta")
                else:
                    state_dict[k] = f.get_tensor(k).to(map_location)
            return state_dict

    # Fallback to torch.load (if weights_only was explicitly False, do not check safety as this is known to be unsafe)
    if weights_only:
        check_torch_load_is_safe()
    extra_args = {}
    # mmap can only be used with files serialized with zipfile-based format.
    if isinstance(checkpoint_file, str) and map_location != "meta" and is_zipfile(checkpoint_file):
        extra_args = {"mmap": True}

    return torch.load(checkpoint_file, map_location=map_location, weights_only=weights_only, **extra_args)


def _end_ptr(tensor: torch.Tensor) -> int:
    # extract the end of the pointer if the tensor is a slice of a bigger tensor
    if tensor.nelement():
        stop = tensor.view(-1)[-1].data_ptr() + tensor.element_size()
    else:
        stop = tensor.data_ptr()
    return stop


def _get_tied_weight_keys(module: nn.Module) -> list[str]:
    tied_weight_keys: list[str] = []
    for name, submodule in module.named_modules():
        tied = getattr(submodule, "_tied_weights_keys", {}) or {}
        tied_weight_keys.extend([f"{name}.{k}" if name else k for k in tied.keys()])
    return tied_weight_keys


def _find_disjoint(tensors: list[set[str]], state_dict: dict[str, torch.Tensor]) -> tuple[list[set[str]], list[str]]:
    filtered_tensors = []
    for shared in tensors:
        if len(shared) < 2:
            filtered_tensors.append(shared)
            continue

        areas = []
        for name in shared:
            tensor = state_dict[name]
            areas.append((tensor.data_ptr(), _end_ptr(tensor), name))
        areas.sort()

        _, last_stop, last_name = areas[0]
        filtered_tensors.append({last_name})
        for start, stop, name in areas[1:]:
            if start >= last_stop:
                filtered_tensors.append({name})
            else:
                filtered_tensors[-1].add(name)
            last_stop = stop
    disjoint_tensors = []
    shared_tensors = []
    for tensors in filtered_tensors:
        if len(tensors) == 1:
            disjoint_tensors.append(tensors.pop())
        else:
            shared_tensors.append(tensors)
    return shared_tensors, disjoint_tensors


def _find_identical(tensors: list[set[str]], state_dict: dict[str, torch.Tensor]) -> tuple[list[set[str]], set[str]]:
    shared_tensors = []
    identical = []
    for shared in tensors:
        if len(shared) < 2:
            continue

        areas = collections.defaultdict(set)
        for name in shared:
            tensor = state_dict[name]
            area = (tensor.device, tensor.data_ptr(), _end_ptr(tensor))
            areas[area].add(name)
        if len(areas) == 1:
            identical.append(shared)
        else:
            shared_tensors.append(shared)
    return shared_tensors, identical


def remove_tied_weights_from_state_dict(
    state_dict: dict[str, torch.Tensor], model: "PreTrainedModel"
) -> dict[str, torch.Tensor]:
    """
    Remove all tied weights from the given `state_dict`, making sure to keep only the main weight that `model`
    will expect when reloading (even if we know tie weights symmetrically, it's better to keep the intended one).
    This is because `safetensors` does not allow tensor aliasing - so we're going to remove aliases before saving.
    """
    # To avoid any potential mistakes and mismatches between config and actual tied weights, here we check the pointers
    # of the Tensors themselves -> we are guaranteed to find all the actual tied weights
    ptrs = collections.defaultdict(list)
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            # Sometimes in the state_dict we have non-tensor objects.
            # e.g. in bitsandbytes we have some `str` objects in the state_dict
            # In the non-tensor case, fall back to the pointer of the object itself
            ptrs[id(tensor)].append(name)

        elif tensor.device.type == "meta":
            # In offloaded cases, there may be meta tensors in the state_dict.
            # For these cases, key by the pointer of the original tensor object
            # (state_dict tensors are detached and therefore no longer shared)
            tensor = model.get_parameter(name)
            ptrs[id(tensor)].append(name)

        else:
            ptrs[id_tensor_storage(tensor)].append(name)

    shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

    # Recursively descend to find tied weight keys
    all_potential_tied_weights_keys = set(_get_tied_weight_keys(model))
    error_names = []
    to_delete_names = set()
    # Removing the keys which are declared as known duplicates on load. This allows to make sure the name which is
    # kept is consistent
    if all_potential_tied_weights_keys is not None:
        for names in shared_ptrs.values():
            found = 0
            for name in sorted(names):
                matches_pattern = any(re.search(pat, name) for pat in all_potential_tied_weights_keys)
                if matches_pattern and name in state_dict:
                    found += 1
                    if found < len(names):
                        to_delete_names.add(name)
    # We are entering a place where the weights and the transformers configuration do NOT match.
    shared_names, disjoint_names = _find_disjoint(shared_ptrs.values(), state_dict)
    # Those are actually tensor sharing but disjoint from each other, we can safely clone them
    # Reloaded won't have the same property, but it shouldn't matter in any meaningful way.
    for name in disjoint_names:
        state_dict[name] = state_dict[name].clone()

    # When not all duplicates have been cleaned, still remove those keys, but put a clear warning.
    # If the link between tensors was done at runtime then `from_pretrained` will not get
    # the key back leading to random tensor. A proper warning will be shown
    # during reload (if applicable), but since the file is not necessarily compatible with
    # the config, better show a proper warning.
    shared_names, identical_names = _find_identical(shared_names, state_dict)
    # delete tensors that have identical storage
    for inames in identical_names:
        known = inames.intersection(to_delete_names)
        for name in known:
            del state_dict[name]
        unknown = inames.difference(to_delete_names)
        if len(unknown) > 1:
            error_names.append(unknown)

    if shared_names:
        error_names.extend(shared_names)

    if len(error_names) > 0:
        raise RuntimeError(
            f"The weights trying to be saved contained shared tensors {error_names} which are not properly defined. "
            f"We found all the potential target tied weights keys to be: {all_potential_tied_weights_keys}.\n"
            "This can also just mean that the module's tied weight keys are wrong vs the actual tied weights in the model.",
        )

    return state_dict


def _load_parameter_into_model(model: "PreTrainedModel", param_name: str, tensor: torch.Tensor):
    """Cast a single parameter or buffer `param_name` into the `model`, with value `tensor`."""
    parent, param_type = get_module_from_name(model, param_name)
    if param_type in parent._parameters and not isinstance(tensor, nn.Parameter):
        tensor = nn.Parameter(tensor, requires_grad=tensor.is_floating_point())
    # We need to use setattr here, as we set non-persistent buffers as well with this function (`load_state_dict`
    # does not allow to do it)
    setattr(parent, param_type, tensor)


def _add_variant(weights_name: str, variant: str | None = None) -> str:
    if variant is not None:
        path, name = weights_name.rsplit(".", 1)
        weights_name = f"{path}.{variant}.{name}"
    return weights_name


def _get_resolved_checkpoint_files(
    pretrained_model_name_or_path: str | os.PathLike | None,
    variant: str | None,
    gguf_file: str | None,
    use_safetensors: bool | None,
    user_agent: dict | None,
    is_remote_code: bool,  # Because we can't determine this inside this function, we need it to be passed in
    transformers_explicit_filename: str | None = None,
    download_kwargs: DownloadKwargs | None = None,
) -> tuple[list[str] | None, dict | None]:
    """Get all the checkpoint filenames based on `pretrained_model_name_or_path`, and optional metadata if the
    checkpoints are sharded.
    This function will download the data if necessary.
    """
    download_kwargs = download_kwargs or DownloadKwargs()
    cache_dir = download_kwargs.get("cache_dir")
    force_download = download_kwargs.get("force_download", False)
    proxies = download_kwargs.get("proxies")
    local_files_only = download_kwargs.get("local_files_only", False)
    token = download_kwargs.get("token")
    revision = download_kwargs.get("revision") or "main"
    subfolder = download_kwargs.get("subfolder", "")
    commit_hash = download_kwargs.get("commit_hash")
    if transformers_explicit_filename is not None:
        if not transformers_explicit_filename.endswith(".safetensors") and not transformers_explicit_filename.endswith(
            ".safetensors.index.json"
        ):
            if transformers_explicit_filename != "adapter_model.bin":
                raise ValueError(
                    "The transformers file in the config seems to be incorrect: it is neither a safetensors file "
                    "(*.safetensors) nor a safetensors index file (*.safetensors.index.json): "
                    f"{transformers_explicit_filename}"
                )

    is_sharded = False

    if pretrained_model_name_or_path is not None and gguf_file is None:
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        # If the file is a local folder (but not in the HF_HOME cache, even if it's technically local)
        if is_local:
            if transformers_explicit_filename is not None:
                # If the filename is explicitly defined, load this by default.
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, transformers_explicit_filename)
                is_sharded = transformers_explicit_filename.endswith(".safetensors.index.json")
            elif use_safetensors is not False and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
            ):
                # Load from a safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant)
                )
            elif use_safetensors is not False and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
            ):
                # Load from a sharded safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                )
                is_sharded = True
            elif not use_safetensors and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant))
            ):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant)
                )
            elif not use_safetensors and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
            ):
                # Load from a sharded PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
                )
                is_sharded = True
            elif use_safetensors:
                raise OSError(
                    f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)} found in directory"
                    f" {pretrained_model_name_or_path}."
                )
            else:
                raise OSError(
                    f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)}, or {_add_variant(WEIGHTS_NAME, variant)},"
                    f" found in directory {pretrained_model_name_or_path}."
                )
        elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
            archive_file = pretrained_model_name_or_path
            is_local = True
        else:
            # set correct filename
            if transformers_explicit_filename is not None:
                filename = transformers_explicit_filename
                is_sharded = transformers_explicit_filename.endswith(".safetensors.index.json")
            elif use_safetensors is not False:
                filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
            else:
                filename = _add_variant(WEIGHTS_NAME, variant)

            # Prepare set of kwargs for hub functions
            has_file_kwargs = {
                "revision": revision,
                "proxies": proxies,
                "token": token,
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
            }
            cached_file_kwargs = {
                "force_download": force_download,
                "user_agent": user_agent,
                "subfolder": subfolder,
                "_raise_exceptions_for_gated_repo": False,
                "_raise_exceptions_for_missing_entries": False,
                "_commit_hash": commit_hash,
                **has_file_kwargs,
            }
            can_auto_convert = (
                not is_offline_mode()  # for obvious reasons
                # If we are in a CI environment or in a pytest run, we prevent the conversion
                and not is_env_variable_true("DISABLE_SAFETENSORS_CONVERSION")
                and not is_remote_code  # converter bot does not work on remote code
                and subfolder == ""  # converter bot does not work on subfolders
            )

            try:
                # Load from URL or cache if already cached
                # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                # result when internet is up, the repo and revision exist, but the file does not.
                resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                # Try safetensors files first if not already found
                if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True
                    elif use_safetensors:
                        if revision == "main" and can_auto_convert:
                            resolved_archive_file, revision, is_sharded = auto_conversion(
                                pretrained_model_name_or_path, **cached_file_kwargs
                            )
                        cached_file_kwargs["revision"] = revision
                        if resolved_archive_file is None:
                            raise OSError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} "
                                "and thus cannot be loaded with `safetensors`. Please do not set `use_safetensors=True`."
                            )
                    else:
                        # This repo has no safetensors file of any kind, we switch to PyTorch.
                        filename = _add_variant(WEIGHTS_NAME, variant)
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, filename, **cached_file_kwargs
                        )

                # Then try `.bin` files
                if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True

                # If we have a match, but it's `.bin` format, try to launch safetensors conversion for next time
                if resolved_archive_file is not None:
                    safe_weights_name = SAFE_WEIGHTS_INDEX_NAME if is_sharded else SAFE_WEIGHTS_NAME
                    if (
                        filename in [WEIGHTS_NAME, WEIGHTS_INDEX_NAME]
                        and not has_file(pretrained_model_name_or_path, safe_weights_name, **has_file_kwargs)
                        and can_auto_convert
                    ):
                        Thread(
                            target=auto_conversion,
                            args=(pretrained_model_name_or_path,),
                            kwargs={"ignore_errors_during_conversion": False, **cached_file_kwargs},
                            name="Thread-auto_conversion",
                        ).start()

                # If no match, raise appropriare errors
                else:
                    # Otherwise, no PyTorch file was found
                    if variant is not None and has_file(
                        pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs
                    ):
                        raise OSError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant"
                            f" {variant}. Use `variant=None` to load this model from those weights."
                        )
                    else:
                        raise OSError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_NAME, variant)}."
                        )

            except OSError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                # to the original exception.
                raise
            except Exception as e:
                # For any other exception, we throw a generic error.
                raise OSError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)}."
                ) from e

        if is_local:
            logger.info(f"loading weights file {archive_file}")
            resolved_archive_file = archive_file
        else:
            logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")

    elif gguf_file:
        # Case 1: the GGUF file is present locally
        if os.path.isfile(gguf_file):
            resolved_archive_file = gguf_file
        # Case 2: The GGUF path is a location on the Hub
        # Load from URL or cache if already cached
        else:
            cached_file_kwargs = {
                "cache_dir": cache_dir,
                "force_download": force_download,
                "proxies": proxies,
                "local_files_only": local_files_only,
                "token": token,
                "user_agent": user_agent,
                "revision": revision,
                "subfolder": subfolder,
                "_raise_exceptions_for_gated_repo": False,
                "_raise_exceptions_for_missing_entries": False,
                "_commit_hash": commit_hash,
            }

            resolved_archive_file = cached_file(pretrained_model_name_or_path, gguf_file, **cached_file_kwargs)

    # We now download and resolve all checkpoint files if the checkpoint is sharded
    sharded_metadata = None
    if is_sharded:
        checkpoint_files, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            user_agent=user_agent,
            revision=revision,
            subfolder=subfolder,
            _commit_hash=commit_hash,
        )
    else:
        checkpoint_files = [resolved_archive_file] if pretrained_model_name_or_path is not None else None

    return checkpoint_files, sharded_metadata


def _get_dtype(
    dtype: str | torch.dtype | dict | None,
    checkpoint_files: list[str] | None,
    config: PreTrainedConfig,
    sharded_metadata: dict | None,
    state_dict: dict | None,
    weights_only: bool,
    hf_quantizer: HfQuantizer | None = None,
) -> tuple[PreTrainedConfig, torch.dtype]:
    """Find the correct `dtype` to use based on provided arguments. Also update the `config` based on the
    inferred dtype. We do the following:
    1. If dtype is "auto", we try to read the config, else auto-detect dtype from the loaded state_dict, by checking
    its first weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
    2. Else, use the dtype provided as a dict or str
    """
    is_sharded = sharded_metadata is not None

    if dtype is not None:
        if isinstance(dtype, str):
            if dtype == "auto":
                if hasattr(config, "dtype") and config.dtype is not None:
                    dtype = config.dtype
                    logger.info(f"Will use dtype={dtype} as defined in model's config object")
                else:
                    if is_sharded and "dtype" in sharded_metadata:
                        dtype = sharded_metadata["dtype"]
                    elif state_dict is not None:
                        dtype = get_state_dict_dtype(state_dict)
                    else:
                        state_dict = load_state_dict(
                            checkpoint_files[0], map_location="meta", weights_only=weights_only
                        )
                        dtype = get_state_dict_dtype(state_dict)
                    logger.info(
                        "Since the `dtype` attribute can't be found in model's config object, "
                        "will use dtype={dtype} as derived from model's weights"
                    )
            elif hasattr(torch, dtype):
                dtype = getattr(torch, dtype)
            else:
                raise ValueError(
                    "`dtype` provided as a `str` can only be `'auto'`, or a string representation of a valid `torch.dtype`"
                )

            # cast it to a proper `torch.dtype` object
            dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        elif not isinstance(dtype, (dict, torch.dtype)):
            raise ValueError(
                f"`dtype` can be one of: `torch.dtype`, `'auto'`, a string of a valid `torch.dtype` or a `dict` with valid `dtype` "
                f"for each sub-config in composite configs, but received {dtype}"
            )
    else:
        # set torch.get_default_dtype() (usually fp32) as the default dtype if `None` is provided
        dtype = torch.get_default_dtype()

    if hf_quantizer is not None:
        hf_quantizer.update_dtype(dtype)

    # Get the main dtype
    if isinstance(dtype, dict):
        main_dtype = dtype.get("", torch.get_default_dtype())
        main_dtype = getattr(torch, main_dtype) if isinstance(main_dtype, str) else main_dtype

        logger.warning_once(
            "Using different dtypes per module is deprecated and will be removed in future versions "
            "Setting different dtypes per backbone model might cause device errors downstream, therefore "
            f"setting the dtype={main_dtype} for all modules."
        )

    else:
        main_dtype = dtype

    # Set it on the config and subconfigs
    config.dtype = main_dtype
    for sub_config_key in config.sub_configs:
        if (sub_config := getattr(config, sub_config_key)) is not None:
            sub_config.dtype = main_dtype

    return config, main_dtype


class PipelineParallel(Enum):
    inputs = 0
    outputs = 1


class ModuleUtilsMixin:
    """
    A few utilities for `torch.nn.Modules`, to be used as a mixin.
    """

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return next(param.device for param in self.parameters())

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return next(param.dtype for param in self.parameters() if param.is_floating_point())

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min

        return encoder_extended_attention_mask

    @staticmethod
    def create_extended_attention_mask_for_decoder(input_shape, attention_mask):
        device = attention_mask.device
        batch_size, seq_length = input_shape
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        # in case past_key_values are used we need to add a prefix ones mask to the causal mask
        causal_mask = causal_mask.to(attention_mask.dtype)

        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat(
                [
                    torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                    causal_mask,
                ],
                axis=-1,
            )

        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        return extended_attention_mask

    def get_extended_attention_mask(
        self,
        attention_mask: Tensor,
        input_shape: tuple[int, ...],
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if getattr(self.config, "is_decoder", None):
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]

        is_loaded_in_4bit = getattr(self, "is_loaded_in_4bit", False)
        if is_loaded_in_4bit:
            import bitsandbytes as bnb

        total_params = 0
        for name, param in self.named_parameters():
            if exclude_embeddings and name in embedding_param_names:
                continue
            if param.requires_grad or not only_trainable:
                # For 4bit models, we need to multiply the number of parameters by 2 as half of the parameters are
                # used for the 4bit quantization (uint8 tensors are stored)
                if is_loaded_in_4bit and isinstance(param, bnb.nn.Params4bit):
                    if hasattr(param, "element_size"):
                        num_bytes = param.element_size()
                    elif hasattr(param, "quant_storage"):
                        num_bytes = param.quant_storage.itemsize
                    else:
                        num_bytes = 1
                    total_params += param.numel() * 2 * num_bytes
                else:
                    total_params += param.numel()

        return total_params


class EmbeddingAccessMixin:
    """
    Base utilities to regroup getters and setters for embeddings.
    Introduces the `input_layer_embed` attribute, which indicates
    where the input embeddings come from and where they
    should be set.
    """

    _input_embed_layer = "embed_tokens"  # default layer that holds input embeddings.

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the model's input embeddings.

        Returns:
            `nn.Module`: A torch module mapping vocabulary to hidden states.
        """

        name = getattr(self, "_input_embed_layer", "embed_tokens")

        # 1) Direct attribute (most NLP models).
        if (default_embedding := getattr(self, name, None)) is not None:
            return default_embedding
        # 2) Nested embeddings (e.g., self.embeddings.patch_embedding for vision/audio models).
        if hasattr(self, "embeddings") and hasattr(self.embeddings, name):
            return getattr(self.embeddings, name)
        # 3) Encoder/decoder wrappers (e.g., `self.model.embed_tokens` or similar overrides).
        if hasattr(self, "model") and hasattr(self.model, name):
            return getattr(self.model, name)

        if hasattr(self, "base_model"):
            base_model = self.base_model
            if base_model is not None and base_model is not self:
                return base_model.get_input_embeddings()

        raise NotImplementedError(
            f"`get_input_embeddings` not auto‑handled for {self.__class__.__name__}; please override in the subclass."
        )

    def set_input_embeddings(self, value: nn.Module):
        """Fallback setter that handles **~70%** of models in the code-base.

        Order of attempts:
        1. `self.<_input_embed_layer>` (direct attribute)
        2. `self.embeddings.<_input_embed_layer>` (nested embeddings for vision/audio models)
        3. `self.model.<_input_embed_layer>` (encoder/decoder models)
        4. delegate to the *base model* if one exists
        5. otherwise raise `NotImplementedError` so subclasses still can (and
            should) override for exotic layouts.
        """

        name = getattr(self, "_input_embed_layer", "embed_tokens")
        # 1) Direct attribute (most NLP models)
        if hasattr(self, name):
            setattr(self, name, value)
        # 2) Nested embeddings (e.g., self.embeddings.patch_embedding for vision models)
        elif hasattr(self, "embeddings") and hasattr(self.embeddings, name):
            setattr(self.embeddings, name, value)
        # 3) encoder/decoder and VLMs like `Gemma3nForConditionalGeneration`
        elif hasattr(self, "model") and hasattr(self.model, name):
            setattr(self.model, name, value)
        # 4) recurse once into the registered *base* model (e.g. for encoder/decoder)
        elif hasattr(self, "base_model") and self.base_model is not self:
            self.base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError(
                f"`set_input_embeddings` not auto‑handled for {self.__class__.__name__}; please override in the subclass."
            )

    def get_output_embeddings(self):
        if not hasattr(self, "lm_head"):
            return None
        try:
            # Speech / vision backbones raise here, so we return None.
            # Legit use of get_input_embs?
            self.get_input_embeddings()
        except NotImplementedError:
            return None
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the model's output embedding, defaulting to setting new_embeddings to lm_head.
        """
        if getattr(self, "lm_head"):
            self.lm_head = new_embeddings


class PreTrainedModel(nn.Module, EmbeddingAccessMixin, ModuleUtilsMixin, PushToHubMixin, PeftAdapterMixin):
    r"""
    Base class for all models.

    [`PreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models as well as a few methods common to all models to:

        - resize the input embeddings

    Class attributes (overridden by derived classes):

        - **config_class** ([`PreTrainedConfig`]) -- A subclass of [`PreTrainedConfig`] to use as configuration class
          for this model architecture.
        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
        - **can_record_outputs** (dict):
    """

    # General model properties
    config_class: type[PreTrainedConfig] | None = None
    _auto_class = None
    base_model_prefix: str = ""
    _is_stateful: bool = False
    model_tags: list[str] | None = None

    # Input-related properties
    main_input_name: str = "input_ids"
    # Attributes used mainly in multimodal LLMs, though all models contain a valid field for these
    # Possible values are: text, image, video, audio and time
    input_modalities: str | list[str] = "text"

    # Device-map related properties
    _no_split_modules: set[str] | list[str] | None = None
    _skip_keys_device_placement: str | list[str] | None = None

    # Specific dtype upcasting
    # `_keep_in_fp32_modules` will upcast to fp32 only if the requested dtype is fp16
    # `_keep_in_fp32_modules_strict` will upcast to fp32 independently if the requested dtype is fp16 or bf16
    _keep_in_fp32_modules: set[str] | list[str] | None = None
    _keep_in_fp32_modules_strict: set[str] | list[str] | None = None

    # Loading-specific properties
    # A dictionary `{"target": "source"}` of checkpoint keys that are potentially tied to one another
    _tied_weights_keys: dict[str, str] = None
    # Used for BC support in VLMs, not meant to be used by new models
    _checkpoint_conversion_mapping: dict[str, str] = {}
    # A list of `re` patterns describing keys to ignore if they are missing from checkpoints to avoid warnings
    _keys_to_ignore_on_load_missing: list[str] | None = None
    # A list of `re` patterns describing keys to ignore if they are unexpected in the checkpoints to avoid warnings
    _keys_to_ignore_on_load_unexpected: list[str] | None = None
    # A list of keys to ignore when saving the model
    _keys_to_ignore_on_save: list[str] | None = None

    # Attention interfaces support properties
    _supports_sdpa: bool = False
    _supports_flash_attn: bool = False
    _supports_flex_attn: bool = False
    # Model's compatible flash kernels (e.g., "kernels-community/flash-mla") defaulting to the first in the list
    _compatible_flash_implementations: list[str] | None = None

    # Tensor-parallelism-related properties
    # A tensor parallel plan of the form `{"model.layer.mlp.param": "colwise"}` to be applied to the model when TP is enabled.
    # For top-level models, this attribute is currently defined in respective model code. For base models, this attribute comes
    # from `config.base_model_tp_plan` during `post_init`.
    _tp_plan: dict[str, str] = None
    # Tensor parallel degree to which model is sharded to
    _tp_size = None
    # A pipeline parallel plan specifying the layers which may not be present on all ranks when PP is enabled. For top-level
    # models, this attribute is currently defined in respective model code. For base models, it comes from
    # `config.base_model_pp_plan` during `post_init`.
    _pp_plan: dict[str, PipelineParallel] | None = None

    # Advanced functionalities support
    supports_gradient_checkpointing: bool = False
    _can_compile_fullgraph: bool = False
    # This flag signal that the model can be used as an efficient backend in TGI and vLLM
    # In practice, it means that they support attention (mask) interface functions, fully pass the kwargs
    # through all modules up to the Attention layer, can slice logits with Tensor, and have a default TP plan
    _supports_attention_backend: bool = False
    # A mapping describing what outputs can be captured by `capture_outputs` decorator during the forward pass
    _can_record_outputs: dict | None = None

    @property
    @torch.compiler.allow_in_graph
    def can_record_outputs(self) -> dict[str, OutputRecorder]:
        """
         Maps output names (e.g., "attentions", "hidden_states")
         to either:
             - A module class (e.g., `LlamaDecoderLayer`), using default index conventions:
                 * index=0 for "hidden_states"
                 * index=1 for "attentions"
             - Or an `OutputRecorder(...)` with `target_class`, optional `index`, and `layer_name`.

         Examples:
             These two are equivalent:

         ```python
             _can_record_outputs = {
                 "attentions": LlamaAttention,
                 "hidden_states": LlamaDecoderLayer
             }

             _can_record_outputs = {
                 "attentions": OutputRecorder(LlamaAttention, index=1),
                 "hidden_states": OutputRecorder(LlamaDecoderLayer, index=0)
             }
        ```

         This means you can record outputs from the same class, by specifying a layer name. Before
         collecting outputs, we check that they come from this layer.

         If you have cross attention that come from `LlamaAttention` and self attention that also
         come from `LlamaAttention` but from `self_attn` you can do this:

         ```python
         class LlamaModel(PreTrainedModel):
             _can_record_outputs = {
                 "attentions": OutputRecorder(LlamaAttention, index=1, layer-name="self_attn"),
                 "cross_attentions": OutputRecorder(LlamaAttention, index=1, layer_name="cross_attn")
             }

        ```
        """
        return self._can_record_outputs or {}

    @property
    def dummy_inputs(self) -> dict[str, torch.Tensor]:
        """
        `dict[str, torch.Tensor]`: Dummy inputs to do a forward pass in the network.
        """
        return {"input_ids": torch.tensor(DUMMY_INPUTS)}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # For BC we keep the original `config_class` definition in case
        # there is a `config_class` attribute (e.g. remote code models),
        # otherwise we derive it from the annotated `config` attribute.

        # defined in this particular subclass
        child_annotation = cls.__dict__.get("__annotations__", {}).get("config", None)
        child_attribute = cls.__dict__.get("config_class", None)

        # defined in the class (this subclass or any parent class)
        full_annotation = get_type_hints(cls).get("config", None)
        full_attribute = cls.config_class

        # priority (child class_config -> child annotation -> global class_config -> global annotation)
        if child_attribute is not None:
            cls.config_class = child_attribute
        elif child_annotation is not None:
            cls.config_class = child_annotation
        elif full_attribute is not None:
            cls.config_class = full_attribute
        elif full_annotation is not None:
            cls.config_class = full_annotation

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise TypeError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config
        self.name_or_path = config.name_or_path

        # Check the attention implementation is supported, or set it if not yet set (on the internal attr, to avoid
        # setting it recursively)
        self.config._attn_implementation_internal = self._check_and_adjust_attn_implementation(
            self.config._attn_implementation, is_init_check=True
        )
        # Check the experts implementation is supported, or set it if not yet set (on the internal attr, to avoid
        # setting it recursively)
        self.config._experts_implementation_internal = self._check_and_adjust_experts_implementation(
            self.config._experts_implementation
        )
        if self.can_generate():
            self.generation_config = GenerationConfig.from_model_config(config)

        # for initialization of the loss
        loss_type = self.__class__.__name__
        if loss_type not in LOSS_MAPPING:
            loss_groups = f"({'|'.join(LOSS_MAPPING)})"
            loss_type = re.findall(loss_groups, self.__class__.__name__)
            if len(loss_type) > 0:
                loss_type = loss_type[0]
            else:
                loss_type = None
        self.loss_type = loss_type

        _CAN_RECORD_REGISTRY[str(self.__class__)] = self._can_record_outputs  # added for executorch support only

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        It is also used to obtain all correct static properties (parallelism plans, tied_weights_keys, _keep_in_fp32_modules, etc)
        correctly in the case of composite models (that is, the top level model should know about those properties from its children).
        """
        # Attach the different parallel plans and tied weight keys to the top-most model, so that everything is
        # easily available
        self._tp_plan, self._ep_plan, self._pp_plan = {}, {}, {}
        # If current model is a base model, attach `base_model_tp_plan` and `base_model_pp_plan` from config
        if self.base_model is self:
            self._pp_plan = self.config.base_model_pp_plan.copy() if self.config.base_model_pp_plan is not None else {}
            self._tp_plan = self.config.base_model_tp_plan.copy() if self.config.base_model_tp_plan is not None else {}
            self._ep_plan = self.config.base_model_ep_plan.copy() if self.config.base_model_ep_plan is not None else {}
        # Current submodel should register its tied weights
        self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=False)
        # Current submodel should register its `_keep_in_fp32_modules`
        self._keep_in_fp32_modules = set(self._keep_in_fp32_modules or [])
        self._keep_in_fp32_modules_strict = set(self._keep_in_fp32_modules_strict or [])
        # Current submodel must register its `_no_split_modules` as well
        self._no_split_modules = set(self._no_split_modules or [])

        # Iterate over children only: as the final model is created, this is enough to gather the properties from all submodels.
        # This works because the way the `__init__` and `post_init` are called on all submodules is depth-first in the graph
        for name, module in self.named_children():
            # Parallel plans
            if plan := getattr(module, "_ep_plan", None):
                self._ep_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})
            if plan := getattr(module, "_tp_plan", None):
                self._tp_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})
            if plan := getattr(module, "_pp_plan", None):
                self._pp_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})
            # Always attach the keys of the children (if the children's config says to NOT tie, then it's empty)
            if tied_keys := getattr(module, "all_tied_weights_keys", None):
                self.all_tied_weights_keys.update({f"{name}.{k}": f"{name}.{v}" for k, v in tied_keys.copy().items()})
            # Record keep_in_fp_32 modules from the children as well
            if keep_fp32 := getattr(module, "_keep_in_fp32_modules", None):
                self._keep_in_fp32_modules.update(keep_fp32)
            if keep_fp32_strict := getattr(module, "_keep_in_fp32_modules_strict", None):
                self._keep_in_fp32_modules_strict.update(keep_fp32_strict)
            # Record `_no_split_modules` from the children
            if no_split := getattr(module, "_no_split_modules", None):
                self._no_split_modules.update(no_split)

        # Maybe initialize the weights and tie the keys
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()

    @property
    def tp_plan(self) -> dict[str, str]:
        """
        The full tp plan for the model's modules
        """
        if hasattr(self.config, "distributed_config") and self.config.distributed_config.enable_expert_parallel:
            return self._ep_plan
        return self._tp_plan

    @property
    def pp_plan(self) -> dict[str, tuple[str, str]]:
        return self._pp_plan

    @tp_plan.setter
    def tp_plan(self, plan: dict[str, str] | None):
        if plan is None:
            self._tp_plan = {}
            return
        if not isinstance(plan, dict):
            raise ValueError("Can only set a dictionary as `tp_plan`")

        # Ensure the styles are all valid
        for layer_pattern, parallel_style in plan.items():
            if parallel_style not in ALL_PARALLEL_STYLES:
                raise ValueError(
                    f"Unsupported tensor parallel style '{parallel_style}' for layer '{layer_pattern}'. "
                    f"Supported styles are {list(ALL_PARALLEL_STYLES.keys())}"
                )

        # Validate that the layer patterns match existing model structure. We check this by getting all parameter
        # names and seeing if any match the patterns
        model_param_names = [name for name, _ in self.named_parameters()]
        for layer_pattern in plan.keys():
            # Convert pattern to regex (replace * with .*)
            regex_pattern = layer_pattern.replace("*", r"\d+")
            pattern_matched = False
            for param_name in model_param_names:
                if re.match(regex_pattern, param_name):
                    pattern_matched = True
                    break
            if not pattern_matched:
                warnings.warn(
                    f"Layer pattern '{layer_pattern}' does not match any parameters in the model. This rule may not "
                    "be applied during tensor parallelization, or may lead to dimension mismatches"
                )

        # Set the plan
        self._tp_plan = plan

    @pp_plan.setter
    def pp_plan(self, plan: dict[str, tuple[str, str]]):
        self._pp_plan = plan

    def dequantize(self, dtype=None):
        """
        Potentially dequantize the model in case it has been quantized by a quantization method that support
        dequantization.
        """
        hf_quantizer = getattr(self, "hf_quantizer", None)

        if hf_quantizer is None:
            raise ValueError("You need to first quantize your model in order to dequantize it")

        return hf_quantizer.dequantize(self, dtype=dtype)

    def _backward_compatibility_gradient_checkpointing(self):
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()
            # Remove the attribute now that is has been consumed, so it's no saved in the config.
            delattr(self.config, "gradient_checkpointing")

    def add_model_tags(self, tags: list[str] | str) -> None:
        r"""
        Add custom tags into the model that gets pushed to the Hugging Face Hub. Will
        not overwrite existing tags in the model.

        Args:
            tags (`Union[list[str], str]`):
                The desired tags to inject in the model

        Examples:

        ```python
        from transformers import AutoModel

        model = AutoModel.from_pretrained("google-bert/bert-base-cased")

        model.add_model_tags(["custom", "custom-bert"])

        # Push the model to your namespace with the name "my-custom-bert".
        model.push_to_hub("my-custom-bert")
        ```
        """
        if isinstance(tags, str):
            tags = [tags]

        if self.model_tags is None:
            self.model_tags = []

        for tag in tags:
            if tag not in self.model_tags:
                self.model_tags.append(tag)

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            dtype (`torch.dtype`, *optional*):
                Override the default `dtype` and load the model under this dtype.
        """
        # For BC on the old `torch_dtype`
        dtype = kwargs.pop("dtype", config.dtype)
        if (torch_dtype := kwargs.pop("torch_dtype", None)) is not None:
            logger.warning_once("`torch_dtype` is deprecated! Use `dtype` instead!")
            # if both kwargs are provided, use `dtype`
            dtype = dtype if dtype != config.dtype else torch_dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        # If passing `attn_implementation` as kwargs, respect it (it will be applied recursively on subconfigs)
        if "attn_implementation" in kwargs:
            config._attn_implementation = kwargs.pop("attn_implementation")

        # If passing `experts_implementation` as kwargs, respect it (it will be applied recursively on subconfigs)
        if "experts_implementation" in kwargs:
            config._experts_implementation = kwargs.pop("experts_implementation")

        init_contexts = []
        if dtype is not None:
            init_contexts.append(local_torch_dtype(dtype, cls.__name__))

        if is_deepspeed_zero3_enabled() and not _is_quantized and not _is_ds_init_called:
            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            # this immediately partitions the model across all gpus, to avoid the overhead in time
            # and memory copying it on CPU or each GPU first
            import deepspeed

            init_contexts.extend([deepspeed.zero.Init(config_dict_or_path=deepspeed_config()), set_zero3_state()])

        # Instantiate the model
        with ContextManagers(init_contexts):
            model = cls(config, **kwargs)

        return model

    @property
    def base_model(self) -> nn.Module:
        """
        `torch.nn.Module`: The main body of the model.
        """
        return getattr(self, self.base_model_prefix, self)

    @classmethod
    def can_generate(cls) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()` from the `GenerationMixin`.

        Under the hood, on classes where this function returns True, some generation-specific changes are triggered:
        for instance, the model instance will have a populated `generation_config` attribute.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Directly inherits `GenerationMixin` -> can generate
        if "GenerationMixin" in str(cls.__bases__):
            return True
        # The class inherits from a class that can generate (recursive check) -> can generate
        for base in cls.__bases__:
            if not hasattr(base, "can_generate"):
                continue
            if "PreTrainedModel" not in str(base) and base.can_generate():
                return True
        # Detects whether `prepare_inputs_for_generation` has been overwritten in the model. Prior to v4.45, this
        # was how we detected whether a model could generate.
        if hasattr(cls, "prepare_inputs_for_generation"):  # implicit: doesn't inherit `GenerationMixin`
            logger.warning(
                f"{cls.__name__} has generative capabilities, as `prepare_inputs_for_generation` is explicitly "
                "defined. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, "
                "`PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability "
                "to call `generate` and other related functions."
                "\n  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the "
                "model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes"
                "\n  - If you are the owner of the model architecture code, please modify your model class such that "
                "it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception)."
                "\n  - If you are not the owner of the model architecture class, please contact the model code owner "
                "to update it."
            )
        # Otherwise, can't generate
        return False

    def _flash_attn_2_can_dispatch(self, is_init_check: bool = False) -> bool:
        """
        Check the availability of Flash Attention 2 for a given model.

        Args:
            is_init_check (`bool`, *optional*):
                Whether this check is performed early, i.e. at __init__ time, or later when the model and its weights are
                fully instantiated. This is needed as we also check the devices of the weights, which are only available
                later after __init__. This allows to raise proper exceptions early before instantiating the full models
                if we know that the model does not support the requested attention.
        """
        dtype = self.config.dtype

        # check `supports_flash_attn_2` for BC with custom code. TODO: remove after a few releases
        if not (self._supports_flash_attn or getattr(self, "_supports_flash_attn_2", False)):
            raise ValueError(
                f"{self.__class__.__name__} does not support Flash Attention 2.0 yet. Please request to add support where"
                f" the model is hosted, on its model hub page: https://huggingface.co/{self.config._name_or_path}/discussions/new"
                " or in the Transformers GitHub repo: https://github.com/huggingface/transformers/issues/new"
            )

        if not is_flash_attn_2_available():
            preface = "FlashAttention2 has been toggled on, but it cannot be used due to the following error:"
            install_message = "Please refer to the documentation of https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2 to install Flash Attention 2."

            # package `flash-attn` can not be installed on Ascend NPU, following validation logics can be ignored.
            if is_torch_npu_available():
                logger.info("Detect using FlashAttention2 on Ascend NPU.")
                return True

            if is_torch_xpu_available():
                logger.info(
                    f"Detect using FlashAttention2 (via kernel `{FLASH_ATTN_KERNEL_FALLBACK['flash_attention_2']}`) on XPU."
                )
                return True

            if importlib.util.find_spec("flash_attn") is None:
                raise ImportError(f"{preface} the package flash_attn seems to be not installed. {install_message}")
            else:
                # Check FA2 installed version compatibility
                flash_attention_version = version.parse(importlib.metadata.version("flash_attn"))
                if torch.version.cuda:
                    if flash_attention_version < version.parse("2.1.0"):
                        raise ImportError(
                            f"{preface} you need flash_attn package version to be greater or equal than 2.1.0. Detected version {flash_attention_version}. {install_message}"
                        )
                    elif not torch.cuda.is_available():
                        raise ValueError(
                            f"{preface} Flash Attention 2 is not available on CPU. Please make sure torch can access a CUDA device."
                        )
                    else:
                        raise ImportError(f"{preface} Flash Attention 2 is not available. {install_message}")
                elif torch.version.hip:
                    if flash_attention_version < version.parse("2.0.4"):
                        raise ImportError(
                            f"{preface} you need flash_attn package version to be greater or equal than 2.0.4. Detected version {flash_attention_version}. {install_message}"
                        )
                    else:
                        raise ImportError(f"{preface} Flash Attention 2 is not available. {install_message}")

        if dtype is None:
            logger.warning_once(
                "You are attempting to use Flash Attention 2 without specifying a torch dtype. This might lead to unexpected behaviour"
            )
        elif dtype is not None and dtype not in [torch.float16, torch.bfloat16]:
            logger.warning_once(
                "Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes, but"
                f" the current dype in {self.__class__.__name__} is {dtype}. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator,"
                ' or load the model with the `dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", dtype=torch.float16)`'
            )

        # With the early check, the parameters are not yet initialized correctly
        if not is_init_check:
            param_devices = list({param.device for param in self.parameters()})
            if len(param_devices) == 1 and param_devices[0].type == "cpu":
                if torch.cuda.is_available():
                    logger.warning_once(
                        "You are attempting to use Flash Attention 2 with a model not initialized on GPU. Make sure to move the model to GPU"
                        " after initializing it on CPU with `model.to('cuda')`."
                    )
                elif is_torch_mlu_available():
                    logger.warning_once(
                        "You are attempting to use Flash Attention 2 with a model not initialized on MLU. Make sure to move the model to MLU"
                        " after initializing it on CPU with `model.to('mlu')`."
                    )
                else:
                    raise ValueError(
                        "You are attempting to use Flash Attention 2 with a model not initialized on GPU and with no GPU available. "
                        "This is not supported yet. Please make sure to have access to a GPU and either initialise the model on a GPU by passing a device_map "
                        "or initialising the model on CPU and then moving it to GPU."
                    )

        # If no error raise by this point, we can return `True`
        return True

    def _flash_attn_3_can_dispatch(self, is_init_check: bool = False) -> bool:
        """
        Check the availability of Flash Attention 3 for a given model.

        Args:
            is_init_check (`bool`, *optional*):
                Whether this check is performed early, i.e. at __init__ time, or later when the model and its weights are
                fully instantiated. This is needed as we also check the devices of the weights, which are only available
                later after __init__. This allows to raise proper exceptions early before instantiating the full models
                if we know that the model does not support the requested attention.
        """
        dtype = self.config.dtype

        if not self._supports_flash_attn:
            raise ValueError(
                f"{self.__class__.__name__} does not support Flash Attention 3 yet. Please request to add support where"
                f" the model is hosted, on its model hub page: https://huggingface.co/{self.config._name_or_path}/discussions/new"
                " or in the Transformers GitHub repo: https://github.com/huggingface/transformers/issues/new"
            )

        if not is_flash_attn_3_available():
            preface = "FlashAttention3 has been toggled on, but it cannot be used due to the following error:"

            if importlib.util.find_spec("flash_attn_3") is None:
                raise ImportError(f"{preface} the package flash_attn_3 seems to be not installed.")

            if torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability()
                if major < 9:
                    raise ValueError(
                        f"{preface} Flash Attention 3 requires compute capability >= 9.0, but found {torch.cuda.get_device_capability()} with compute capability {major}.0."
                    )
                else:
                    raise ImportError(f"{preface} Flash Attention 3 is not available.")
            else:
                raise ValueError(
                    f"{preface} Flash Attention 3 is not available on CPU. Please make sure torch can access a CUDA device."
                )

        if dtype is None:
            logger.warning_once(
                "You are attempting to use Flash Attention 3 without specifying a torch dtype. This might lead to unexpected behaviour"
            )
        elif dtype is not None and dtype not in [torch.float16, torch.bfloat16]:
            logger.warning_once(
                "Flash Attention 3 only supports torch.float16 and torch.bfloat16 dtypes, but"
                f" the current dype in {self.__class__.__name__} is {dtype}. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator,"
                ' or load the model with the `dtype` argument. Example: `model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B", attn_implementation="flash_attention_3", dtype=torch.float16)`'
            )

        if getattr(self.config, "alibi", False) or getattr(self.config, "use_alibi", False):
            raise ValueError("Model is configured to use ALiBi, which is not supported by Flash Attention 3.")

        # Check for attention dropout, which is incompatible with FA3
        if hasattr(self.config, "attention_dropout") and self.config.attention_dropout > 0:
            raise ValueError(
                f"Model has attention_dropout={self.config.attention_dropout}, which is not supported by Flash Attention 3."
            )

        # With the early check, the parameters are not yet initialized correctly
        if not is_init_check:
            param_devices = list({param.device for param in self.parameters()})
            if len(param_devices) == 1 and param_devices[0].type == "cpu":
                if torch.cuda.is_available():
                    logger.warning_once(
                        "You are attempting to use Flash Attention 3 with a model not initialized on GPU. Make sure to move the model to GPU"
                        " after initializing it on CPU with `model.to('cuda')`."
                    )
                else:
                    raise ValueError(
                        "You are attempting to use Flash Attention 3 with a model not initialized on GPU and with no GPU available. "
                        "This is not supported yet. Please make sure to have access to a GPU and either initialise the model on a GPU by passing a device_map "
                        "or initialising the model on CPU and then moving it to GPU."
                    )

        return True

    def _sdpa_can_dispatch(self, is_init_check: bool = False) -> bool:
        """
        Check the availability of SDPA for a given model.

        Args:
            is_init_check (`bool`, *optional*):
                Whether this check is performed early, i.e. at __init__ time, or later when the model and its weights are
                fully instantiated. This is needed as we also check the devices of the weights, which are only available
                later after __init__. This allows to raise proper exceptions early before instantiating the full models
                if we know that the model does not support the requested attention.
        """
        if not self._supports_sdpa:
            raise ValueError(
                f"{self.__class__.__name__} does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention yet."
                " Please request the support for this architecture: https://github.com/huggingface/transformers/issues/28005. If you believe"
                ' this error is a bug, please open an issue in Transformers GitHub repository and load your model with the argument `attn_implementation="eager"` meanwhile. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="eager")`'
            )

        if (
            torch.version.hip is not None
            and torch.cuda.device_count() > 1
            and version.parse(torch.__version__) < version.parse("2.4.1")
        ):
            logger.warning_once(
                "Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends."
            )
            torch.backends.cuda.enable_flash_sdp(False)

        return True

    def _grouped_mm_can_dispatch(self) -> bool:
        """
        Check the availability of Grouped MM for a given model.
        """

        if not self._can_set_experts_implementation():
            raise ValueError(f"{self.__class__.__name__} does not support setting experts implementation.")

        # If no error raised by this point, we can return `True`
        return True

    def _flex_attn_can_dispatch(self, is_init_check: bool = False) -> bool:
        """
        Check the availability of Flex Attention for a given model.

        Args:
            is_init_check (`bool`, *optional*):
                Whether this check is performed early, i.e. at __init__ time, or later when the model and its weights are
                fully instantiated. This is needed as we also check the devices of the weights, which are only available
                later after __init__. This allows to raise proper exceptions early before instantiating the full models
                if we know that the model does not support the requested attention.
        """
        if not self._supports_flex_attn:
            raise ValueError(
                f"{self.__class__.__name__} does not support an attention implementation through torch's flex_attention."
                " Please request the support for this architecture: https://github.com/huggingface/transformers/issues/34809."
                " If you believe this error is a bug, please open an issue in Transformers GitHub repository"
                ' and load your model with the argument `attn_implementation="eager"` meanwhile.'
                ' Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="eager")`'
            )
        if not is_torch_flex_attn_available():
            raise ImportError(
                "PyTorch Flex Attention requirements in Transformers are not met. Please install torch>=2.5.0."
            )

        # If no error raise by this point, we can return `True`
        return True

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: str | None, is_init_check: bool = False
    ) -> str:
        """
        Check that the `attn_implementation` exists and is supported by the models, and try to get the kernel from hub if
        it matches hf kernels pattern.

        Args:
            attn_implementation (`str` or `None`):
                The attention implementation to check for existence/validity.
            is_init_check (`bool`, *optional*):
                Whether this check is performed early, i.e. at __init__ time, or later when the model and its weights are
                fully instantiated. This is needed as we also check the devices of the weights, which are only available
                later after __init__. This allows to raise proper exceptions early before instantiating the full models
                if we know that the model does not support the requested attention.

        Returns:
            `str`: The final attention implementation to use, including potential fallbacks from sdpa to eager, or from
            None to sdpa (to potentially eager).
        """
        # Auto-correct model's default flash implementation if specified
        if attn_implementation is not None:
            is_paged = attn_implementation.startswith("paged|")
            base_implementation = attn_implementation.removeprefix("paged|")

            compatible_flash_implementations = getattr(self, "_compatible_flash_implementations", None)
            if (
                compatible_flash_implementations
                and is_flash_attention_requested(requested_attention_implementation=base_implementation)
                and base_implementation not in compatible_flash_implementations
            ):
                default_flash_implementation = (
                    f"paged|{compatible_flash_implementations[0]}" if is_paged else compatible_flash_implementations[0]
                )

                logger.warning_once(
                    f"This model is compatible with the following flash attention implementations: `{compatible_flash_implementations}`. "
                    f"Automatically falling back to `{default_flash_implementation}` instead of `{attn_implementation}`."
                )
                attn_implementation = default_flash_implementation

        applicable_attn_implementation = attn_implementation

        is_paged = attn_implementation is not None and attn_implementation.startswith("paged|")

        # If FA not installed, do not fail but use kernels instead
        requested_original_flash_attn = attn_implementation is not None and (
            attn_implementation.removeprefix("paged|") == "flash_attention_2"
            or attn_implementation.removeprefix("paged|") == "flash_attention_3"
        )
        if (
            requested_original_flash_attn
            and self._supports_flash_attn
            and not (is_flash_attn_2_available() or is_flash_attn_3_available())
            and is_kernels_available()
            and not is_torch_npu_available()
        ):
            applicable_attn_implementation = FLASH_ATTN_KERNEL_FALLBACK[attn_implementation.removeprefix("paged|")]

            if is_torch_xpu_available() and attn_implementation.removeprefix("paged|") == "flash_attention_2":
                # On XPU, kernels library is the native implementation
                # Disabling this flag to avoid giving wrong fallbacks on errors and warnings
                requested_original_flash_attn = False

            if is_paged:
                applicable_attn_implementation = f"paged|{applicable_attn_implementation}"

        if is_kernel(applicable_attn_implementation):
            try:
                # preload flash attention here to allow compile with fullgraph
                if is_paged:
                    lazy_import_paged_flash_attention(applicable_attn_implementation)
                else:
                    lazy_import_flash_attention(applicable_attn_implementation)

                # log that we used kernel fallback if successful
                if requested_original_flash_attn:
                    logger.warning_once(
                        f"You do not have `flash_attn` installed, using `{applicable_attn_implementation}` "
                        "from the `kernels` library instead!"
                    )
            except Exception as e:
                # raise the proper exception for requested flash attention
                if requested_original_flash_attn:
                    if attn_implementation.endswith("2"):
                        self._flash_attn_2_can_dispatch()
                    else:
                        self._flash_attn_3_can_dispatch()

                # error properly out if a kernel was specifically requested
                raise e
        else:
            applicable_attn_implementation = self.get_correct_attn_implementation(
                applicable_attn_implementation, is_init_check
            )

            # preload flash attention here to allow compile with fullgraph
            if is_flash_attention_requested(requested_attention_implementation=applicable_attn_implementation):
                lazy_import_flash_attention(applicable_attn_implementation)

        return applicable_attn_implementation

    def _check_and_adjust_experts_implementation(self, experts_implementation: str | None) -> str:
        """
        Check that the `experts_implementation` exists and is supported by the models.

        Args:
            experts_implementation (`str` or `None`):
                The experts implementation to check for existence/validity.
        Returns:
            `str`: The final experts implementation to use.
        """
        applicable_experts_implementation = self.get_correct_experts_implementation(experts_implementation)
        return applicable_experts_implementation

    def get_correct_attn_implementation(self, requested_attention: str | None, is_init_check: bool = False) -> str:
        applicable_attention = "sdpa" if requested_attention is None else requested_attention
        if applicable_attention not in ["eager"] + ALL_ATTENTION_FUNCTIONS.valid_keys():
            message = (
                f'Specified `attn_implementation="{applicable_attention}"` is not supported. The only possible arguments are '
                '`attn_implementation="eager"`, `"paged|eager"`'
            )
            # check `supports_flash_attn_2` for BC with custom code. TODO: remove after a few releases
            if self._supports_flash_attn or getattr(self, "_supports_flash_attn_2", False):
                message += ', `"attn_implementation=flash_attention_3"`, `"attn_implementation=flash_attention_2"`, `"attn_implementation=paged|flash_attention_2"`'
            if self._supports_sdpa:
                message += ', `"attn_implementation=sdpa"`, `"attn_implementation=paged|sdpa"`'
            if self._supports_flex_attn:
                message += ', `"attn_implementation=flex_attention"`'
            raise ValueError(message + ".")

        # Perform relevant checks
        if "flash_attention_2" in applicable_attention:
            self._flash_attn_2_can_dispatch(is_init_check)
        elif "flash_attention_3" in applicable_attention:
            self._flash_attn_3_can_dispatch(is_init_check)
        elif "flex_attention" in applicable_attention:
            self._flex_attn_can_dispatch(is_init_check)
        elif "sdpa" in applicable_attention:
            # Sdpa is the default, so we try it and fallback to eager otherwise when not possible
            try:
                self._sdpa_can_dispatch(is_init_check)
            except (ValueError, ImportError) as e:
                if requested_attention is not None and "sdpa" in requested_attention:
                    raise e
                applicable_attention = "eager"

        return applicable_attention

    def get_correct_experts_implementation(self, requested_experts: str | None) -> str:
        applicable_experts = "grouped_mm" if requested_experts is None else requested_experts
        if applicable_experts not in ["eager", "grouped_mm", "batched_mm"]:
            message = (
                f'Specified `experts_implementation="{applicable_experts}"` is not supported. The only possible arguments are '
                '`experts_implementation="eager"`, `"experts_implementation=grouped_mm"` and `"experts_implementation=batched_mm"`.'
            )
            raise ValueError(message)

        # Perform relevant checks
        if applicable_experts == "grouped_mm":
            try:
                self._grouped_mm_can_dispatch()
            except (ValueError, ImportError) as e:
                if requested_experts == "grouped_mm":
                    raise e
                applicable_experts = "eager"

        return applicable_experts

    @classmethod
    def _can_set_attn_implementation(cls) -> bool:
        """Detect whether the class supports setting its attention implementation dynamically. It is an ugly check based on
        opening the file, but avoids maintaining yet another property flag.
        """
        class_module = sys.modules[cls.__module__]
        # This can happen for a custom model in a jupyter notebook or repl for example - simply do not allow to set it then
        if not hasattr(class_module, "__file__"):
            return False
        class_file = class_module.__file__
        with open(class_file, "r", encoding="utf-8") as f:
            code = f.read()
        # heuristic -> if we find those patterns, the model uses the correct interface
        if re.search(r"class \w+Attention\(nn.Module\)", code):
            return "eager_attention_forward" in code and "ALL_ATTENTION_FUNCTIONS.get_interface(" in code
        else:
            # If no attention layer, assume `True`. Most probably a multimodal model or inherits from existing models
            return True

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        """Detect whether the class supports setting its experts implementation dynamically. It is an ugly check based on
        opening the file, but avoids maintaining yet another property flag.
        """
        class_module = sys.modules[cls.__module__]
        # This can happen for a custom model in a jupyter notebook or repl for example - simply do not allow to set it then
        if not hasattr(class_module, "__file__"):
            return False
        class_file = class_module.__file__
        with open(class_file, "r", encoding="utf-8") as f:
            code = f.read()
        # heuristic -> if we the use_experts_implementation decorator is used, then we can set it
        return "@use_experts_implementation" in code

    def set_attn_implementation(self, attn_implementation: str | dict):
        """
        Set the requested `attn_implementation` for this model.

        Args:
            attn_implementation (`str` or `dict`):
                The attention implementation to set for this model. It can be either a `str`, in which case it will be
                dispatched to all submodels if relevant, or a `dict` where keys are the sub_configs name, in which case each
                submodel will dispatch the corresponding value.
        """
        requested_implementation = (
            attn_implementation
            if not isinstance(attn_implementation, dict)
            else attn_implementation.get("", self.config._attn_implementation)
        )

        if requested_implementation != self.config._attn_implementation:
            # In this case, raise
            if not self._can_set_attn_implementation():
                logger.warning(
                    f"{self.__class__.__name__} does not support setting its attention implementation dynamically, because it "
                    "does not follow the functional approach based on AttentionInterface "
                    "(see https://huggingface.co/docs/transformers/en/attention_interface)"
                )
            else:
                requested_implementation = self._check_and_adjust_attn_implementation(
                    requested_implementation, is_init_check=False
                )
                # Apply the change (on the internal attr, to avoid setting it recursively)
                self.config._attn_implementation_internal = requested_implementation

        # Apply it to all submodels as well
        for submodule in self.modules():
            # We found a submodel (which is not self) with a different config (otherwise, it may be the same "actual model",
            # e.g. ForCausalLM has a Model inside, but no need to check it again)
            if (
                submodule is not self
                and isinstance(submodule, PreTrainedModel)
                and submodule.config.__class__ != self.config.__class__
                # If it was already changed, no need to do it again
                and not hasattr(submodule.config, "_attn_was_changed")
            ):
                # In this case, warn and skip
                if not submodule._can_set_attn_implementation():
                    logger.warning(
                        f"{submodule.__class__.__name__} does not support setting its attention implementation dynamically, because it "
                        "does not follow the functional approach based on AttentionInterface "
                        "(see https://huggingface.co/docs/transformers/en/attention_interface)"
                    )
                # Set the attn on the submodule
                else:
                    sub_implementation = requested_implementation
                    if isinstance(attn_implementation, dict):
                        for subconfig_key in self.config.sub_configs:
                            # We need to check for exact object match here, with `is`
                            if getattr(self.config, subconfig_key) is submodule.config:
                                sub_implementation = attn_implementation.get(
                                    subconfig_key, submodule.config._attn_implementation
                                )
                                break
                    # Check the module can use correctly, otherwise we raise an error if requested attention can't be set for submodule
                    sub_implementation = submodule.get_correct_attn_implementation(sub_implementation)
                    submodule.config._attn_implementation_internal = sub_implementation

                # Still add it as "changed" even if it was skipped, as we would otherwise try to set it in the dark afterwards
                # We need to set it on the config itself, to differentiate 2 subconfigs of the same __class__ potentially
                submodule.config._attn_was_changed = True

        # We need this as some old and badly designed models use subconfigs without declaring the corresponding modules as PreTrainedModel
        for subconfig_key in self.config.sub_configs:
            if (subconfig := getattr(self.config, subconfig_key)) is not None:
                sub_implementation = (
                    requested_implementation
                    if not isinstance(attn_implementation, dict)
                    else attn_implementation.get(subconfig_key, subconfig._attn_implementation)
                )
                # This means we did not perform any check above for this particular subconfig -> set it in the dark if it is registered
                if (
                    not hasattr(subconfig, "_attn_was_changed")
                    # If it's already the same, then no need to enter here and raise warnings
                    and sub_implementation != subconfig._attn_implementation
                ):
                    if sub_implementation not in ["eager"] + ALL_ATTENTION_FUNCTIONS.valid_keys():
                        raise ValueError(
                            f'Specified `attn_implementation="{sub_implementation}"` is not supported for {subconfig_key}. '
                            'The only possible arguments are "eager" (manual attention implementation)'
                            f"or one of the following: {list(ALL_ATTENTION_FUNCTIONS.valid_keys())}"
                        )
                    subconfig._attn_implementation_internal = sub_implementation
                    logger.warning(
                        f"We set the attention implementation for the sub-config `{subconfig_key}` to `{sub_implementation}` "
                        "without finding the associated sub-model. For this reason we could not check if the model supports it. "
                        "You may encounter undefined behavior."
                    )
                # Unset the attribute in this case, to avoid issues in the future
                else:
                    if hasattr(subconfig, "_attn_was_changed"):
                        del subconfig._attn_was_changed

    def set_experts_implementation(self, experts_implementation: str | dict):
        """
        Set the requested `experts_implementation` for this model.

        Args:
            experts_implementation (`str` or `dict`):
                The experts implementation to set for this model. It can be either a `str`, in which case it will be
                dispatched to all submodels if relevant, or a `dict` where keys are the sub_configs name, in which case each
                submodel will dispatch the corresponding value.
        """
        requested_implementation = (
            experts_implementation
            if not isinstance(experts_implementation, dict)
            else experts_implementation.get("", self.config._experts_implementation)
        )

        if requested_implementation != self.config._experts_implementation:
            requested_implementation = self._check_and_adjust_experts_implementation(requested_implementation)
            # Apply the change (on the internal attr, to avoid setting it recursively)
            self.config._experts_implementation_internal = requested_implementation

        # Apply it to all submodels as well
        for submodule in self.modules():
            # We found a submodel (which is not self) with a different config (otherwise, it may be the same "actual model",
            # e.g. ForCausalLM has a Model inside, but no need to check it again)
            if (
                submodule is not self
                and isinstance(submodule, PreTrainedModel)
                and submodule.config.__class__ != self.config.__class__
            ):
                # Set the experts on the submodule
                sub_implementation = requested_implementation
                if isinstance(experts_implementation, dict):
                    for subconfig_key in self.config.sub_configs:
                        # We need to check for exact object match here, with `is`
                        if getattr(self.config, subconfig_key) is submodule.config:
                            sub_implementation = experts_implementation.get(
                                subconfig_key, submodule.config._experts_implementation
                            )
                            break
                # Check the module can use correctly, otherwise we raise an error if requested experts can't be set for submodule
                sub_implementation = submodule.get_correct_experts_implementation(sub_implementation)
                submodule.config._experts_implementation_internal = sub_implementation

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        hooks = []
        seen_modules = set()
        found_embeddings = False

        for module in self.modules():
            if not (isinstance(module, PreTrainedModel) and hasattr(module, "get_input_embeddings")):
                continue

            try:
                input_embeddings = module.get_input_embeddings()
            except NotImplementedError:
                continue

            if input_embeddings is None or not hasattr(input_embeddings, "register_forward_hook"):
                continue

            embedding_id = id(input_embeddings)
            if embedding_id in seen_modules:
                continue

            seen_modules.add(embedding_id)
            hooks.append(input_embeddings.register_forward_hook(make_inputs_require_grads))
            found_embeddings = True

        self._require_grads_hooks = hooks
        if hooks:
            # for BC
            self._require_grads_hook = hooks[0]
        if not found_embeddings:
            logger.warning_once(
                f"{self.__class__.__name__} does not expose input embeddings. Gradients cannot flow back to the token "
                "embeddings when using adapters or gradient checkpointing. Override `get_input_embeddings` to fully "
                "support those features, or set `_input_embed_layer` to the attribute name that holds the embeddings."
            )

    def disable_input_require_grads(self):
        """
        Removes the `_require_grads_hook`.
        """
        hooks = getattr(self, "_require_grads_hooks", None)
        if not hooks:
            return

        for hook in hooks:
            hook.remove()

        self._require_grads_hooks = []
        if hasattr(self, "_require_grads_hook"):
            del self._require_grads_hook

    def get_encoder(self, modality: str | None = None):
        """
        Best-effort lookup of the *encoder* module. If provided with `modality` argument,
        it looks for a modality-specific encoder in multimodal models (e.g. "image_encoder")
        By default the function returns model's text encoder if any, and otherwise returns `self`.

        Possible `modality` values are "image", "video" and "audio".
        """
        # NOTE: new models need to use existing names for layers if possible, so this list doesn't grow infinitely
        if modality in ["image", "video"]:
            possible_module_names = ["vision_tower", "visual", "vision_model", "vision_encoder", "image_tower"]
        elif modality == "audio":
            possible_module_names = ["audio_tower", "audio_encoder", "speech_encoder"]
        elif modality is None:
            possible_module_names = ["text_encoder", "encoder"]
        else:
            raise ValueError(f'Unnrecognized modality, has to be "image", "video" or "audio" but found {modality}')

        for name in possible_module_names:
            if hasattr(self, name):
                return getattr(self, name)

        if self.base_model is not self and hasattr(self.base_model, "get_encoder"):
            base_encoder = self.base_model.get_encoder(modality=modality)
            # Base model will always have attr `get_encoder` if inherited from `PreTrainedModel`
            # But it doesn't mean that the model has an encoder module, and we need to return `self`
            if base_encoder != self.base_model:
                return base_encoder

        # If this is a base transformer model (no encoder/model attributes), return self
        return self

    def set_encoder(self, encoder, modality: str | None = None):
        """
        Symmetric setter. Mirrors the lookup logic used in `get_encoder`.
        """

        # NOTE: new models need to use existing names for layers if possible, so this list doesn't grow infinitely
        if modality in ["image", "video"]:
            possible_module_names = ["vision_tower", "visual", "vision_model", "vision_encoder", "image_tower"]
        if modality == "audio":
            possible_module_names = ["audio_tower", "audio_encoder"]
        elif modality is None:
            possible_module_names = ["text_encoder", "encoder"]
        else:
            raise ValueError(f'Unnrecognized modality, has to be "image", "video" or "audio" but found {modality}')

        for name in possible_module_names:
            if hasattr(self, name):
                setattr(self, name, encoder)
                return

        if self.base_model is not self:
            if hasattr(self.base_model, "set_encoder"):
                self.base_model.set_encoder(encoder, modality=modality)
            else:
                self.model = encoder

    def get_decoder(self):
        """
        Best-effort lookup of the *decoder* module.

        Order of attempts (covers ~85 % of current usages):

        1. `self.decoder/self.language_model/self.text_model`
        2. `self.base_model`                  (many wrappers store the decoder here)
        3. `self.base_model.get_decoder()`    (nested wrappers)
        4. fallback: raise for the few exotic models that need a bespoke rule
        """
        possible_module_names = ["language_model", "text_model", "decoder", "text_decoder"]
        for name in possible_module_names:
            if hasattr(self, name):
                return getattr(self, name)

        if self.base_model is not self and hasattr(self.base_model, "get_decoder"):
            return self.base_model.get_decoder()

        # If this is a base transformer model (no decoder/model attributes), return self
        # This handles cases like MistralModel which is itself the decoder
        return self

    def set_decoder(self, decoder):
        """
        Symmetric setter. Mirrors the lookup logic used in `get_decoder`.
        """

        possible_module_names = ["language_model", "text_model", "decoder"]
        for name in possible_module_names:
            if hasattr(self, name):
                setattr(self, name, decoder)
                return

        if self.base_model is not self:
            if hasattr(self.base_model, "set_decoder"):
                self.base_model.set_decoder(decoder)
            else:
                self.model = decoder

    @torch.no_grad()
    def _init_weights(self, module):
        """
        Initialize the weights. This is quite general on purpose, in the spirit of what we usually do. For more complex
        initialization scheme, it should be overridden by the derived `PreTrainedModel` class. In case a model adds an explicit
        `nn.Parameter`, this method should also be overridden in order to initialize it correctly.
        """
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range or 0.02
        elif hasattr(self.config, "init_std"):
            std = self.config.init_std
        elif hasattr(self.config, "initializer_factor"):
            std = self.config.initializer_factor
        else:
            # 0.02 is the standard default value across the library
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            if getattr(module, "weight", None) is not None:
                init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            # Here we need the check explicitly, as we slice the weight in the `zeros_` call, so it looses the flag
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.MultiheadAttention):
            # This uses torch's original init
            module._reset_parameters()
        # We cannot use `isinstance` on the RMSNorms or LayerNorms, as they usually are custom modules which change names
        # between modelings (because they are prefixed with the model name)
        elif (
            isinstance(module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            or "LayerNorm" in module.__class__.__name__
            or "RMSNorm" in module.__class__.__name__
        ):
            # Norms can exist without weights (in which case they are None from torch primitives)
            if getattr(module, "weight", None) is not None:
                init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)
            # And the potential buffers for the BatchNorms
            if getattr(module, "running_mean", None) is not None:
                init.zeros_(module.running_mean)
                init.ones_(module.running_var)
                init.zeros_(module.num_batches_tracked)
        # This matches all the usual RotaryEmbeddings modules
        elif "RotaryEmbedding" in module.__class__.__name__ and hasattr(module, "original_inv_freq"):
            rope_fn = (
                ROPE_INIT_FUNCTIONS[module.rope_type]
                if module.rope_type != "default"
                else module.compute_default_rope_parameters
            )
            buffer_value, _ = rope_fn(module.config)
            init.copy_(module.inv_freq, buffer_value)
            init.copy_(module.original_inv_freq, buffer_value)

    def _initialize_weights(self, module, is_remote_code: bool = False):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(module, "_is_hf_initialized", False):
            return

        # This check is for remote code that does NOT use either `torch.init` or `transformers.initialization` in `_init_weights`
        # which allow to check the flag directly on param. As they don't and write the params in-place, params would be reinitialized
        # otherwise
        if (
            is_remote_code
            and all(getattr(param, "_is_hf_initialized", False) for param in module.parameters(recurse=False))
            and all(
                getattr(buffer, "_is_hf_initialized", False)
                for buffer in module.buffers(recurse=False)
                if buffer is not None
            )
        ):
            module._is_hf_initialized = True
            return

        self._init_weights(module)
        module._is_hf_initialized = True

    @torch.no_grad()
    @init.guard_torch_init_functions()
    def initialize_weights(self):
        """
        This is equivalent to calling `self.apply(self._initialize_weights)`, but correctly handles composite models.
        This function dynamically dispatches the correct `init_weights` function to the modules as we advance in the
        module graph along the recursion. It can handle an arbitrary number of sub-models. Without it, every composite
        model would have to recurse a second time on all sub-models explicitly in the outer-most `_init_weights`, which
        is extremely error prone and inefficient.
        """
        if not hasattr(torch.nn.Module, "smart_apply"):
            # This function is equivalent to `torch.nn.Module.apply`, except that it dynamically adjust the function
            # to apply as we go down the graph
            def smart_apply(self, fn, is_remote_code):
                for module in self.children():
                    # We found a sub-model: recursively dispatch its own init function now!
                    if isinstance(module, PreTrainedModel):
                        module.smart_apply(module._initialize_weights, is_remote_code)
                    else:
                        module.smart_apply(fn, is_remote_code)
                fn(self, is_remote_code)
                return self

            torch.nn.Module.smart_apply = smart_apply

        # Let the magic happen with this simple call
        self.smart_apply(self._initialize_weights, self.is_remote_code())

    def get_expanded_tied_weights_keys(self, all_submodels: bool = False) -> dict:
        r"""
        Return the expanded tied weight keys (in case they contain modules or regex patterns) for only the current
        model, or recursively for all submodels if `all_submodels=True` (i.e. it will re-check the config values for all
        submodels).

        For almost all models, we only require to tie the embeddings, so the model has an internal property
        `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`. In this case, the mapping is already
        "expanded", i.e. it already contains full parameters, and this function will simply return a copy of the property.
        For more complex patterns, e.g. for `DFineForObjectDetection`, we have the following attribute
        ```
        _tied_weights_keys = {
            r"bbox_embed.(?![0])\d+": "bbox_embed.0",
            r"class_embed.(?![0])\d+": "class_embed.0",
            "model.decoder.class_embed": "class_embed",
            "model.decoder.bbox_embed": "bbox_embed",
        }
        ```
        In this case, the function looks up all the model's parameters and buffers, and matches all the params,
        returning the following:
        ```
        {
            'bbox_embed.1.layers.0.bias': 'bbox_embed.0.layers.0.bias',
            'bbox_embed.1.layers.0.weight': 'bbox_embed.0.layers.0.weight',
            'bbox_embed.1.layers.1.bias': 'bbox_embed.0.layers.1.bias',
            'bbox_embed.1.layers.1.weight': 'bbox_embed.0.layers.1.weight',
            'bbox_embed.1.layers.2.bias': 'bbox_embed.0.layers.2.bias',
            'bbox_embed.1.layers.2.weight': 'bbox_embed.0.layers.2.weight',
            'bbox_embed.2.layers.0.bias': 'bbox_embed.0.layers.0.bias',
            'bbox_embed.2.layers.0.weight': 'bbox_embed.0.layers.0.weight',
            ...
            'class_embed.1.bias': 'class_embed.0.bias',
            'class_embed.1.weight': 'class_embed.0.weight',
            'class_embed.2.bias': 'class_embed.0.bias',
            'class_embed.2.weight': 'class_embed.0.weight',
            ...
            'model.decoder.class_embed.0.bias': 'class_embed.0.bias',
            'model.decoder.class_embed.0.weight': 'class_embed.0.weight',
            'model.decoder.class_embed.1.bias': 'class_embed.0.bias',
            'model.decoder.class_embed.1.weight': 'class_embed.0.weight',
            ...
            'model.decoder.bbox_embed.0.layers.0.bias': 'bbox_embed.0.layers.0.bias',
            'model.decoder.bbox_embed.0.layers.0.weight': 'bbox_embed.0.layers.0.weight',
            'model.decoder.bbox_embed.0.layers.1.bias': 'bbox_embed.0.layers.1.bias',
            'model.decoder.bbox_embed.0.layers.1.weight': 'bbox_embed.0.layers.1.weight',
            ...
        }
        ```
        i.e. all the parameters matching the regex and modules patterns in `_tied_weights_keys`
        """
        if all_submodels:
            expanded_tied_weights = {}
            for prefix, submodule in self.named_modules(remove_duplicate=False):
                if isinstance(submodule, PreTrainedModel):
                    # Will dynamically check the config if it has changed
                    submodel_tied_weights = submodule.get_expanded_tied_weights_keys(all_submodels=False)
                    if prefix != "":
                        submodel_tied_weights = {
                            f"{prefix}.{k}": f"{prefix}.{v}" for k, v in submodel_tied_weights.items()
                        }
                    expanded_tied_weights.update(submodel_tied_weights)
            return expanded_tied_weights

        tied_mapping = self._tied_weights_keys
        # If the config does not specify any tying, return empty dict
        # NOTE: not all modules have `tie_word_embeddings` attr, for example vision-only
        # modules do not have any word embeddings!
        tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)
        if not tie_word_embeddings:
            return {}
        # If None, return empty dict
        elif tied_mapping is None:
            return {}
        # Short-cut for the most common cases: if the tied weights mapping only contains already expanded params,
        # return it directly (the regex matches names containing only letters, numbers, dots, and underscores to make
        # sure it does not contain a regex pattern, and finishing by "bias" or "weight" to make sure it's not a module)
        common_case_regex = re.compile(r"^[A-Za-z0-9_\.]+(weight)|(bias)$")
        if all(common_case_regex.match(k) for k in tied_mapping.keys() | tied_mapping.values()):
            return tied_mapping.copy()

        # We need to expand the regex patterns or the modules into proper parameters
        expanded_tied_weights = {}
        all_param_names = {k for k, _ in self.named_parameters(remove_duplicate=False)} | {
            k for k, _ in self.named_buffers(remove_duplicate=False)
        }
        for target_name, source_name in tied_mapping.items():
            target_name = "^" + target_name
            source_name = "^" + source_name

            source_params = sorted(filter(lambda x: re.search(source_name, x), all_param_names))
            target_params = sorted(filter(lambda x: re.search(target_name, x), all_param_names))
            if (
                not len(source_params) > 0
                or not len(target_params) > 0
                or len(target_params) % len(source_params) != 0
            ):
                raise ValueError(
                    f"There is an issue with your definition of `tie_weights_keys` for {source_name}:{target_name}. "
                    f"We found {source_params} to tie into {target_params}"
                )
            # we cycle source as it should be dispatch in many target if regex
            for target_n, source_n in zip(target_params, cycle(source_params)):
                # If the source is already registered as a target, use the original corresponding source. This should never
                # happen in general, but some models such as `d_fine` have complicated regex patterns, so it end up being
                # the case for simplicity of the regexes. Fix it silently here
                if source_n in expanded_tied_weights.keys():
                    # Use original source instead of having keys both as source and targets
                    expanded_tied_weights[target_n] = expanded_tied_weights[source_n]
                # Usual case, everything is already correct
                else:
                    expanded_tied_weights[target_n] = source_n

        return expanded_tied_weights

    def tie_weights(self, missing_keys: set[str] | None = None, recompute_mapping: bool = True):
        """
        Tie the model weights. If `recompute_mapping=False` (default when called internally), it will rely on the
        `model.all_tied_weights_keys` attribute, containing the `{target: source}` mapping for the tied params.
        If `recompute_mapping=True`, it will re-check all internal submodels and their config to determine the params
        that need to be tied. This is the default when `model.tie_weights()` is called on its own, outside of
        `__init__`, and `from_pretrained`, in case the config values were changed somewhere.

        Note that during `from_pretrained`, tying is *symmetric*: if the mapping says "tie target -> source" but
        `source` is missing in the checkpoint while `target` exists, we *swap* source and target so we can still
        tie everything to the parameter that actually exists.
        """
        # In this case, the keys stored in `all_tied_weights_keys` are already correct
        if not recompute_mapping:
            tied_keys = self.all_tied_weights_keys
        else:
            tied_keys = self.get_expanded_tied_weights_keys(all_submodels=True)

        tied_keys = list(tied_keys.items())
        for i, (target_param_name, source_param_name) in enumerate(tied_keys):
            # This is `from_pretrained` -> let's check symmetrically in case the source key is not present
            if missing_keys is not None:
                remove_from_missing = True
                source_is_there = source_param_name not in missing_keys
                target_is_there = target_param_name not in missing_keys
                # Both are already present -> it means the config is wrong and do not reflect the actual
                # checkpoint -> let's raise a warning and NOT tie them
                if source_is_there and target_is_there:
                    logger.warning(
                        f"The tied weights mapping and config for this model specifies to tie {source_param_name} to "
                        f"{target_param_name}, but both are present in the checkpoints, so we will NOT tie them. "
                        "You should update the config with `tie_word_embeddings=False` to silence this warning"
                    )
                    # Remove from internal attribute to correctly reflect actual tied weights
                    self.all_tied_weights_keys.pop(target_param_name)
                    # Skip to next iteration
                    continue
                # We're missing the source but we have the target -> we swap them, tying the parameter that exists
                elif not source_is_there and target_is_there:
                    target_param_name, source_param_name = source_param_name, target_param_name
                # Both are missing -> check other keys in case more than 2 keys are tied to the same weight
                elif not source_is_there and not target_is_there:
                    for target_backup, source_backup in tied_keys[i + 1 :]:
                        # In case of more than 2 keys tied to the same weight, they are guaranteed to all have
                        # the same source thanks to `get_expanded_tied_weights_keys` so this check is enough
                        if source_backup == source_param_name:
                            target_backup_is_there = target_backup not in missing_keys
                            # If the target is present, we found the correct weight to tie into (we know the source is missing)
                            # Note here that we do not tie the missing source right now as well, as it will be done anyway when
                            # the pair (target_backup, source_backup) becomes the main pair (target_param_name, source_param_name)
                            if target_backup_is_there:
                                source_param_name = target_backup
                                break
                    # If we did not break from the loop, it was impossible to find a source key -> let's raise
                    else:
                        # TODO Cyril: here ideally we want to raise instead of warning, but will break our CI as we have
                        # tests loading model from empty dicts to perform init checks - since we don't raise, add a flag
                        # to NOT remove from missing keys as it's actually still missing
                        remove_from_missing = False
                        logger.warning(
                            f"This checkpoint seem corrupted. The tied weights mapping for this model specifies to tie "
                            f"{source_param_name} to {target_param_name}, but both are absent from the checkpoint, "
                            "and we could not find another related tied weight for those keys"
                        )

            # Perform the actual tying
            source_param = self.get_parameter_or_buffer(source_param_name)
            if "." in target_param_name:
                parent_name, name = target_param_name.rsplit(".", 1)
                parent = self.get_submodule(parent_name)
            else:
                name = target_param_name
                parent = self
            # Tie the weights
            setattr(parent, name, source_param)
            self._adjust_bias(parent, source_param)
            # Remove from missing if necessary
            if missing_keys is not None and remove_from_missing:
                missing_keys.discard(target_param_name)

    def _adjust_bias(self, output_embeddings, input_embeddings):
        if getattr(output_embeddings, "bias", None) is not None and hasattr(output_embeddings, "weight"):
            weight_shape = output_embeddings.weight.shape
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (0, weight_shape[0] - output_embeddings.bias.shape[0]),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value.If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
            mean_resizing (`bool`):
                Whether to initialize the added embeddings from a multivariate normal distribution that has old embeddings' mean and
                covariance or to initialize them with a normal distribution that has a mean of zero and std equals `config.initializer_range`.

                Setting `mean_resizing` to `True` is useful when increasing the size of the embeddings of causal language models,
                where the generated tokens' probabilities won't be affected by the added embeddings because initializing the new embeddings with the
                old embeddings' mean will reduce the kl-divergence between the next token probability before and after adding the new embeddings.
                Refer to this article for more information: https://nlp.stanford.edu/~johnhew/vocab-expansion.html

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Since we are basically reusing the same old embeddings with new weight values, gathering is required
        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            with deepspeed.zero.GatheredParameters(model_embeds.weight, modifier_rank=None):
                vocab_size = model_embeds.weight.shape[0]
        else:
            vocab_size = model_embeds.weight.shape[0]

        # Update base model and current model config.
        self.config.get_text_config().vocab_size = vocab_size
        self.vocab_size = vocab_size

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens, pad_to_multiple_of, mean_resizing
        )
        if hasattr(old_embeddings, "_hf_hook"):
            hook = old_embeddings._hf_hook
            add_hook_to_module(new_embeddings, hook)
        old_embeddings_requires_grad = old_embeddings.weight.requires_grad
        new_embeddings.requires_grad_(old_embeddings_requires_grad)
        self.set_input_embeddings(new_embeddings)
        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None

        # Update new_num_tokens with the actual size of new_embeddings
        if pad_to_multiple_of is not None:
            if is_deepspeed_zero3_enabled() and not is_quantized:
                import deepspeed

                with deepspeed.zero.GatheredParameters(new_embeddings.weight, modifier_rank=None):
                    new_num_tokens = new_embeddings.weight.shape[0]
            else:
                new_num_tokens = new_embeddings.weight.shape[0]

        # if word embeddings are not tied, make sure that lm head is resized as well
        if self.get_output_embeddings() is not None:
            old_lm_head = self.get_output_embeddings()
            if isinstance(old_lm_head, torch.nn.Embedding):
                new_lm_head = self._get_resized_embeddings(old_lm_head, new_num_tokens, mean_resizing=mean_resizing)
            else:
                new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens, mean_resizing=mean_resizing)
            if hasattr(old_lm_head, "_hf_hook"):
                hook = old_lm_head._hf_hook
                add_hook_to_module(new_lm_head, hook)
            old_lm_head_requires_grad = old_lm_head.weight.requires_grad
            new_lm_head.requires_grad_(old_lm_head_requires_grad)
            self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()

    def _get_resized_embeddings(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
            mean_resizing (`bool`):
                Whether to initialize the added embeddings from a multivariate normal distribution that has old embeddings' mean and
                covariance or to initialize them with a normal distribution that has a mean of zero and std equals `config.initializer_range`.

                Setting `mean_resizing` to `True` is useful when increasing the size of the embeddings of causal language models,
                where the generated tokens' probabilities will not be affected by the added embeddings because initializing the new embeddings with the
                old embeddings' mean will reduce the kl-divergence between the next token probability before and after adding the new embeddings.
                Refer to this article for more information: https://nlp.stanford.edu/~johnhew/vocab-expansion.html


        Return:
            `torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            `new_num_tokens` is `None`
        """

        if pad_to_multiple_of is not None:
            if not isinstance(pad_to_multiple_of, int):
                raise ValueError(
                    f"Asking to pad the embedding matrix to a multiple of `{pad_to_multiple_of}`, which is not and integer. Please make sure to pass an integer"
                )
            if new_num_tokens is None:
                new_num_tokens = old_embeddings.weight.shape[0]
            new_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        else:
            logger.info(
                "You are resizing the embedding layer without providing a `pad_to_multiple_of` parameter. This means that the new embedding"
                f" dimension will be {new_num_tokens}. This might induce some performance reduction as *Tensor Cores* will not be available."
                " For more details about this, or help on choosing the correct value for resizing, refer to this guide:"
                " https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc"
            )

        if new_num_tokens is None:
            return old_embeddings

        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
                old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens and not is_deepspeed_zero3_enabled():
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embedding_dim,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype,
        )

        if new_num_tokens > old_num_tokens and not mean_resizing:
            # initialize new embeddings (in particular added tokens) with a mean of 0 and std equals `config.initializer_range`.
            self._init_weights(new_embeddings)

        elif new_num_tokens > old_num_tokens and mean_resizing:
            # initialize new embeddings  (in particular added tokens). The new embeddings will be initialized
            # from a multivariate normal distribution that has old embeddings' mean and covariance.
            # as described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html
            logger.warning_once(
                "The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. "
                "As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. "
                "To disable this, use `mean_resizing=False`"
            )

            added_num_tokens = new_num_tokens - old_num_tokens
            if is_deepspeed_zero3_enabled() and not is_quantized:
                import deepspeed

                with deepspeed.zero.GatheredParameters([old_embeddings.weight], modifier_rank=None):
                    self._init_added_embeddings_weights_with_mean(
                        old_embeddings, new_embeddings, old_num_tokens, added_num_tokens
                    )
            else:
                self._init_added_embeddings_weights_with_mean(
                    old_embeddings, new_embeddings, old_num_tokens, added_num_tokens
                )

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)

        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            params = [old_embeddings.weight, new_embeddings.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        else:
            new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        # Replace weights in old_embeddings and return to maintain the same embedding type.
        # This ensures correct functionality when a Custom Embedding class is passed as input.
        # The input and output embedding types remain consistent. (c.f. https://github.com/huggingface/transformers/pull/31979)
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            params = [old_embeddings.weight, new_embeddings.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                old_embeddings.weight = new_embeddings.weight
                old_embeddings.num_embeddings = new_embeddings.weight.data.shape[0]

                # If the new number of tokens is smaller than the original `padding_idx`, the `padding_idx`
                # will be set to `None` in the resized embeddings.
                if old_embeddings.padding_idx is not None and (new_num_tokens - 1) < old_embeddings.padding_idx:
                    old_embeddings.padding_idx = None
        else:
            old_embeddings.weight.data = new_embeddings.weight.data
            old_embeddings.num_embeddings = new_embeddings.weight.data.shape[0]
            if old_embeddings.padding_idx is not None and (new_num_tokens - 1) < old_embeddings.padding_idx:
                old_embeddings.padding_idx = None

        return old_embeddings

    def _get_resized_lm_head(
        self,
        old_lm_head: nn.Linear,
        new_num_tokens: int | None = None,
        transposed: bool = False,
        mean_resizing: bool = True,
    ) -> nn.Linear:
        """
        Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_lm_head (`torch.nn.Linear`):
                Old lm head liner layer to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Linear` module of the model without doing anything. transposed (`bool`, *optional*, defaults
                to `False`): Whether `old_lm_head` is transposed or not. If True `old_lm_head.size()` is `lm_head_dim,
                vocab_size` else `vocab_size, lm_head_dim`.
            mean_resizing (`bool`):
                Whether to initialize the added embeddings from a multivariate normal distribution that has old embeddings' mean and
                covariance or to initialize them with a normal distribution that has a mean of zero and std equals `config.initializer_range`.

                Setting `mean_resizing` to `True` is useful when increasing the size of the embeddings of causal language models,
                where the generated tokens' probabilities will not be affected by the added embeddings because initializing the new embeddings with the
                old embeddings' mean will reduce the kl-divergence between the next token probability before and after adding the new embeddings.
                Refer to this article for more information: https://nlp.stanford.edu/~johnhew/vocab-expansion.html

        Return:
            `torch.nn.Linear`: Pointer to the resized Linear Module or the old Linear Module if `new_num_tokens` is
            `None`
        """

        if new_num_tokens is None:
            return old_lm_head

        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_lm_head.weight, modifier_rank=None):
                old_num_tokens, old_lm_head_dim = (
                    old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
                )
        else:
            old_num_tokens, old_lm_head_dim = (
                old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
            )

        if old_num_tokens == new_num_tokens and not is_deepspeed_zero3_enabled():
            return old_lm_head

        if not isinstance(old_lm_head, nn.Linear):
            raise TypeError(
                f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Linear}. You"
                " should either use a different resize function or make sure that `old_lm_head` are an instance of"
                f" {nn.Linear}."
            )

        # Build new lm head
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = old_lm_head.bias is not None

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_lm_head = nn.Linear(
            *new_lm_head_shape,
            bias=has_new_lm_head_bias,
            device=old_lm_head.weight.device,
            dtype=old_lm_head.weight.dtype,
        )

        if new_num_tokens > old_num_tokens and not mean_resizing:
            # initialize new embeddings (in particular added tokens) with a mean of 0 and std equals `config.initializer_range`.
            self._init_weights(new_lm_head)

        elif new_num_tokens > old_num_tokens and mean_resizing:
            # initialize new lm_head weights (in particular added tokens). The new lm_head weights
            # will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance.
            # as described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html
            logger.warning_once(
                "The new lm_head weights will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. "
                "As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. "
                "To disable this, use `mean_resizing=False`"
            )

            added_num_tokens = new_num_tokens - old_num_tokens
            if is_deepspeed_zero3_enabled() and not is_quantized:
                import deepspeed

                params = [old_lm_head.weight]
                if has_new_lm_head_bias:
                    params += [old_lm_head.bias]
                with deepspeed.zero.GatheredParameters(params, modifier_rank=None):
                    self._init_added_lm_head_weights_with_mean(
                        old_lm_head, new_lm_head, old_lm_head_dim, old_num_tokens, added_num_tokens, transposed
                    )
                    if has_new_lm_head_bias:
                        self._init_added_lm_head_bias_with_mean(old_lm_head, new_lm_head, added_num_tokens)

            else:
                self._init_added_lm_head_weights_with_mean(
                    old_lm_head, new_lm_head, old_lm_head_dim, old_num_tokens, added_num_tokens, transposed
                )
                if has_new_lm_head_bias:
                    self._init_added_lm_head_bias_with_mean(old_lm_head, new_lm_head, added_num_tokens)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            params = [old_lm_head.weight, old_lm_head.bias, new_lm_head.weight, new_lm_head.bias]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                self._copy_lm_head_original_to_resized(
                    new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
                )
        else:
            self._copy_lm_head_original_to_resized(
                new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
            )

        return new_lm_head

    def _init_added_embeddings_weights_with_mean(
        self, old_embeddings, new_embeddings, old_num_tokens, added_num_tokens
    ):
        old_embeddings_weight = old_embeddings.weight.data.to(torch.float32)
        mean_embeddings = torch.mean(old_embeddings_weight, axis=0)
        old_centered_embeddings = old_embeddings_weight - mean_embeddings
        covariance = old_centered_embeddings.T @ old_centered_embeddings / old_num_tokens

        # Check if the covariance is positive definite.
        epsilon = 1e-9
        is_covariance_psd = constraints.positive_definite.check(epsilon * covariance).all()
        if is_covariance_psd:
            # If covariances is positive definite, a distribution can be created. and we can sample new weights from it.
            distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                mean_embeddings, covariance_matrix=epsilon * covariance
            )
            new_embeddings.weight.data[-1 * added_num_tokens :, :] = distribution.sample(
                sample_shape=(added_num_tokens,)
            ).to(old_embeddings.weight.dtype)
        else:
            # Otherwise, just initialize with the mean. because distribution will not be created.
            new_embeddings.weight.data[-1 * added_num_tokens :, :] = (
                mean_embeddings[None, :].repeat(added_num_tokens, 1).to(old_embeddings.weight.dtype)
            )

    def _init_added_lm_head_weights_with_mean(
        self,
        old_lm_head,
        new_lm_head,
        old_lm_head_dim,
        old_num_tokens,
        added_num_tokens,
        transposed: bool = False,
    ):
        if transposed:
            # Transpose to the desired shape for the function.
            new_lm_head.weight.data = new_lm_head.weight.data.T
            old_lm_head.weight.data = old_lm_head.weight.data.T

        # The same initialization logic as Embeddings.
        self._init_added_embeddings_weights_with_mean(old_lm_head, new_lm_head, old_num_tokens, added_num_tokens)

        if transposed:
            # Transpose again to the correct shape.
            new_lm_head.weight.data = new_lm_head.weight.data.T
            old_lm_head.weight.data = old_lm_head.weight.data.T

    def _init_added_lm_head_bias_with_mean(self, old_lm_head, new_lm_head, added_num_tokens):
        bias_mean = torch.mean(old_lm_head.bias.data, axis=0, dtype=torch.float32)
        bias_std = torch.std(old_lm_head.bias.data, axis=0).to(torch.float32)
        new_lm_head.bias.data[-1 * added_num_tokens :].normal_(mean=bias_mean, std=1e-9 * bias_std)

    def _copy_lm_head_original_to_resized(
        self, new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
    ):
        # Copy old lm head weights to new lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        raise NotImplementedError(
            f"`resize_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    def get_position_embeddings(self) -> nn.Embedding | tuple[nn.Embedding]:
        raise NotImplementedError(
            f"`get_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    def init_weights(self):
        """
        Initialize and tie the weights if needed. If using a custom `PreTrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        # If we are initializing on meta device, there is no point in trying to run inits
        if get_torch_context_manager_or_global_device() != torch.device("meta"):
            # Initialize weights
            self.initialize_weights()
        # Tie weights needs to be called here, but it can use the pre-computed `all_tied_weights_keys`
        self.tie_weights(recompute_mapping=False)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.

        We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
        the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

        Args:
            gradient_checkpointing_kwargs (dict, *optional*):
                Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}

        gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)

        # For old GC format (transformers < 4.35.0) for models that live on the Hub
        # we will fall back to the overwritten `_set_gradient_checkpointing` method
        _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters

        if not _is_using_old_format:
            self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
        else:
            self.apply(partial(self._set_gradient_checkpointing, value=True))
            logger.warning(
                "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
            )

        needs_embedding_grads = self.main_input_name == "input_ids"
        # we use that also to detect whether or not we have to raise if embeddings are missing (the submodel might not have embeddings at all)
        enable_input_grads = needs_embedding_grads or getattr(self, "_hf_peft_config_loaded", False)
        if enable_input_grads:
            # When using PEFT + gradient checkpointing + Trainer we need to make sure the input has requires_grad=True
            # we do it also on PEFT: https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334
            # When training with PEFT, only LoRA layers will have requires grad set to True, but the output of frozen layers need to propagate
            # the gradients to make sure the gradient flows.
            self.enable_input_require_grads()

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint):
        is_gradient_checkpointing_set = False

        # Apply it on the top-level module in case the top-level modules supports it
        # for example, LongT5Stack inherits from `PreTrainedModel`.
        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.
        """
        if self.supports_gradient_checkpointing:
            # For old GC format (transformers < 4.35.0) for models that live on the Hub
            # we will fall back to the overwritten `_set_gradient_checkpointing` method
            _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters
            if not _is_using_old_format:
                self._set_gradient_checkpointing(enable=False)
            else:
                logger.warning(
                    "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                    "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
                )
                self.apply(partial(self._set_gradient_checkpointing, value=False))

        if getattr(self, "_hf_peft_config_loaded", False):
            self.disable_input_require_grads()

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.
        """
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules())

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        is_main_process: bool = True,
        state_dict: dict | None = None,
        push_to_hub: bool = False,
        max_shard_size: int | str = "50GB",
        variant: str | None = None,
        token: str | bool | None = None,
        save_peft_format: bool = True,
        save_original_format: bool = True,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~PreTrainedModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            state_dict (nested dictionary of `torch.Tensor`):
                The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only
                save parts of the model or if special precautions need to be taken when recovering the state dictionary
                of a model (like when using model parallelism).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            max_shard_size (`int` or `str`, *optional*, defaults to `"50GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).

                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>

            variant (`str`, *optional*):
                If specified, weights are saved in the format model.<variant>.safetensors.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `hf auth login` (stored in `~/.huggingface`).
            save_peft_format (`bool`, *optional*, defaults to `True`):
                For backward compatibility with PEFT library, in case adapter weights are attached to the model, all
                keys of the state dict of adapters needs to be prepended with `base_model.model`. Advanced users can
                disable this behaviours by setting `save_peft_format` to `False`.
            save_original_format (`bool`, *optional*, defaults to `True`):
                For backward compatibility with the previous versions of `transformers` you can save the checkpoint with
                its reverse mapping. The reverse mapping needs to exists even if the model was loaded from a None legacy
                checkpoint.
            kwargs (`dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if token is not None:
            kwargs["token"] = token

        _hf_peft_config_loaded = getattr(self, "_hf_peft_config_loaded", False)

        hf_quantizer = getattr(self, "hf_quantizer", None)
        quantization_serializable = (
            hf_quantizer is not None and isinstance(hf_quantizer, HfQuantizer) and hf_quantizer.is_serializable()
        )

        if hf_quantizer is not None and not _hf_peft_config_loaded and not quantization_serializable:
            raise ValueError(
                f"The model is quantized with {hf_quantizer.quantization_config.quant_method} and is not serializable - check out the warnings from"
                " the logger on the traceback to understand the reason why the quantized model is not serializable."
            )

        # we need to check against tp_size, not tp_plan, as tp_plan is substituted to the class one
        if self._tp_size is not None and not is_huggingface_hub_greater_or_equal("0.31.4"):
            raise ImportError(
                "Saving a model with tensor parallelism requires `huggingface_hub` version 0.31.4 or higher."
            )

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            create_pr = kwargs.pop("create_pr", False)
            repo_id = create_repo(repo_id, exist_ok=True, **kwargs).repo_id
            files_timestamps = self._get_files_timestamps(save_directory)

        metadata = {}
        if hf_quantizer is not None:
            state_dict, metadata = hf_quantizer.get_state_dict_and_metadata(self)
        metadata["format"] = "pt"

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)
        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = model_to_save.dtype
        model_to_save.config.dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        # When using FSDP2, unwrapping is a noop, so the model name doesn't change back to the original model name
        model_to_save.config.architectures = [model_to_save.__class__.__name__.removeprefix("FSDP")]

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)

        # Save the config
        if is_main_process:
            if not _hf_peft_config_loaded:
                model_to_save.config.save_pretrained(save_directory)
            if self.can_generate():
                model_to_save.generation_config.save_pretrained(save_directory)

            if _hf_peft_config_loaded:
                logger.info(
                    "Detected adapters on the model, saving the model in the PEFT format, only adapter weights will be saved."
                )
                state_dict = model_to_save.get_adapter_state_dict(state_dict=state_dict)

                if save_peft_format:
                    logger.info(
                        "To match the expected format of the PEFT library, all keys of the state dict of adapters will be prepended with `base_model.model`."
                    )
                    peft_state_dict = {}
                    for key, value in state_dict.items():
                        peft_state_dict[f"base_model.model.{key}"] = value
                    state_dict = peft_state_dict

                active_adapter = self.active_adapters()

                if len(active_adapter) > 1:
                    raise ValueError(
                        "Multiple active adapters detected, saving multiple active adapters is not supported yet. You can save adapters separately one by one "
                        "by iteratively calling `model.set_adapter(adapter_name)` then `model.save_pretrained(...)`"
                    )
                active_adapter = active_adapter[0]

                current_peft_config = self.peft_config[active_adapter]
                current_peft_config.save_pretrained(save_directory)

        # Get the model state_dict
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        # if any model parameters are offloaded, we need to know it for later
        is_offloaded = False
        if (
            hasattr(self, "hf_device_map")
            and len(set(self.hf_device_map.values())) > 1
            and ("cpu" in self.hf_device_map.values() or "disk" in self.hf_device_map.values())
        ):
            is_offloaded = True
            warnings.warn(
                "Attempting to save a model with offloaded modules. Ensure that unallocated cpu memory "
                "exceeds the `shard_size` (50GB default)"
            )

        # Translate state_dict from smp to hf if saving with smp >= 1.10
        if IS_SAGEMAKER_MP_POST_1_10:
            for smp_to_hf, _ in smp.state.module_manager.translate_functions:
                state_dict = smp_to_hf(state_dict)

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            for ignore_key in self._keys_to_ignore_on_save:
                if ignore_key in state_dict:
                    del state_dict[ignore_key]

        # If model was sharded with TP, gather full tensors for saving
        if self._tp_size is not None:
            state_dict = gather_state_dict_for_save(state_dict, self._tp_plan, self._device_mesh, self._tp_size)

        # Remove tied weights as safetensors do not handle them
        state_dict = remove_tied_weights_from_state_dict(state_dict, model_to_save)

        # Revert all renaming and/or weight operations
        if save_original_format and not _hf_peft_config_loaded:
            state_dict = revert_weight_conversion(model_to_save, state_dict)

        # Shard the model if it is too big.
        if not _hf_peft_config_loaded:
            weights_name = SAFE_WEIGHTS_NAME
            weights_name = _add_variant(weights_name, variant)
        else:
            weights_name = ADAPTER_SAFE_WEIGHTS_NAME

        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
        )
        # Save index if sharded
        index = None
        if state_dict_split.is_sharded:
            index = {
                "metadata": {"total_parameters": self.num_parameters(), **state_dict_split.metadata},
                "weight_map": state_dict_split.tensor_to_filename,
            }

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")

            # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
            filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in state_dict_split.filename_to_tensors
                and is_main_process
                and reg.fullmatch(filename_no_suffix) is not None
            ):
                os.remove(full_filename)

        # Save the model
        for shard_file, tensor_names in logging.tqdm(
            state_dict_split.filename_to_tensors.items(), desc="Writing model shards"
        ):
            filename = os.path.join(save_directory, shard_file)
            shard_state_dict = {}
            for tensor_name in tensor_names:
                # Get the tensor, and remove it from state_dict to avoid keeping the ref
                tensor = state_dict.pop(tensor_name)

                # If the param was offloaded, we need to load it back from disk to resave it. It's a strange pattern,
                # but it would otherwise not be contained in the saved shard if we were to simply move the file
                # or something
                if is_offloaded and tensor.device.type == "meta":
                    tensor = load_offloaded_parameter(model_to_save, tensor_name)

                # only do contiguous after it's permuted correctly in case of TP
                shard_state_dict[tensor_name] = tensor.contiguous()

            # TODO: it would be very nice to do the writing concurrently, but safetensors never releases the GIL,
            # so it's not possible for now....
            # Write the shard to disk
            safe_save_file(shard_state_dict, filename, metadata=metadata)
            # Cleanup the data before next loop (important with offloading, so we don't blowup cpu RAM)
            del shard_state_dict

        if index is None:
            path_to_weights = os.path.join(save_directory, weights_name)
            logger.info(f"Model weights saved in {path_to_weights}")
        else:
            save_index_file = SAFE_WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(state_dict_split.filename_to_tensors)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

        if push_to_hub:
            # Eventually create an empty model card
            model_card = create_and_tag_model_card(repo_id, self.model_tags, token=token)

            # Update model card if needed:
            model_card.save(os.path.join(save_directory, "README.md"))

            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
            )

    @wraps(PushToHubMixin.push_to_hub)
    def push_to_hub(self, *args, **kwargs):
        tags = self.model_tags if self.model_tags is not None else []

        tags_kwargs = kwargs.get("tags", [])
        if isinstance(tags_kwargs, str):
            tags_kwargs = [tags_kwargs]

        for tag in tags_kwargs:
            if tag not in tags:
                tags.append(tag)

        if tags:
            kwargs["tags"] = tags
        return super().push_to_hub(*args, **kwargs)

    def get_memory_footprint(self, return_buffers=True):
        r"""
        Get the memory footprint of a model. This will return the memory footprint of the current model in bytes.
        Useful to benchmark the memory footprint of the current model and design some tests. Solution inspired from the
        PyTorch discussions: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2

        Arguments:
            return_buffers (`bool`, *optional*, defaults to `True`):
                Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers
                are tensors that do not require gradients and not registered as parameters. E.g. mean and std in batch
                norm layers. Please see: https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2
        """
        mem = sum(param.nelement() * param.element_size() for param in self.parameters())
        if return_buffers:
            mem_bufs = sum(buf.nelement() * buf.element_size() for buf in self.buffers())
            mem = mem + mem_bufs
        return mem

    @wraps(torch.nn.Module.cuda)
    def cuda(self, *args, **kwargs):
        if getattr(self, "quantization_method", None) == QuantizationMethod.HQQ:
            from hqq.core.quantize import HQQLinear

            # Since HQQLinear stores some tensors in the 'meta' attribute,
            # it's necessary to manually call the `cuda` method on HQQLinear layers.
            super().cuda(*args, **kwargs)
            for module in self.modules():
                if isinstance(module, HQQLinear):
                    if len(args) > 0:
                        device = args[0]
                    else:
                        device = kwargs.get("device", "cuda")
                    module.cuda(device)
            return self

        # Checks if the model has been loaded in 4-bit or 8-bit with BNB
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            if getattr(self, "is_loaded_in_8bit", False):
                raise ValueError(
                    "Calling `cuda()` is not supported for `8-bit` quantized models. "
                    " Please use the model as it is, since the model has already been set to the correct devices."
                )
        return super().cuda(*args, **kwargs)

    @wraps(torch.nn.Module.to)
    def to(self, *args, **kwargs):
        # For BNB/GPTQ models, we prevent users from casting the model to another dtype to restrict unwanted behaviours.
        # the correct API should be to load the model with the desired dtype directly through `from_pretrained`.
        dtype_present_in_args = "dtype" in kwargs

        if not dtype_present_in_args:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype_present_in_args = True
                    break

        if getattr(self, "quantization_method", None) == QuantizationMethod.HQQ:
            from hqq.core.quantize import HQQLinear

            # Since HQQLinear stores some tensors in the 'meta' attribute, we must
            # explicitly move the parameters to the target device for each HQQLinear layer after `to`.
            super().to(*args, **kwargs)
            for module in self.modules():
                if isinstance(module, HQQLinear):
                    if "device" in kwargs:
                        device = kwargs["device"]
                    else:
                        device = args[0]
                    if "dtype" in kwargs:
                        dtype = kwargs["dtype"]
                    elif dtype_present_in_args:
                        dtype = arg
                    else:
                        dtype = None
                    # Due to the current messy implementation of HQQLinear, updating `compute_dtype`
                    # followed by calling the `cuda` method achieves the intended behavior of `to`,
                    # even when the target device is CPU.
                    if dtype is not None:
                        module.compute_dtype = dtype
                    module.cuda(device)
            return self

        if dtype_present_in_args and getattr(self, "quantization_method", None) == QuantizationMethod.QUARK:
            raise ValueError("Casting a Quark quantized model to a new `dtype` is not supported.")

        # Checks if the model has been loaded in 4-bit or 8-bit with BNB
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            if dtype_present_in_args:
                raise ValueError(
                    "You cannot cast a bitsandbytes model in a new `dtype`. Make sure to load the model using `from_pretrained` using the"
                    " desired `dtype` by passing the correct `dtype` argument."
                )

            if getattr(self, "is_loaded_in_8bit", False) and not is_bitsandbytes_available("0.48"):
                raise ValueError(
                    "You need to install `pip install bitsandbytes>=0.48.0` if you want to move a 8-bit model across devices using to()."
                )
        elif getattr(self, "quantization_method", None) == QuantizationMethod.GPTQ:
            if dtype_present_in_args:
                raise ValueError(
                    "You cannot cast a GPTQ model in a new `dtype`. Make sure to load the model using `from_pretrained` using the desired"
                    " `dtype` by passing the correct `dtype` argument."
                )
        return super().to(*args, **kwargs)

    def half(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.half()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            return super().half(*args)

    def float(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.float()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            return super().float(*args)

    @classmethod
    def get_init_context(cls, dtype: torch.dtype, is_quantized: bool, _is_ds_init_called: bool):
        # Need to instantiate with correct dtype
        init_contexts = [local_torch_dtype(dtype, cls.__name__), init.no_tie_weights()]
        if is_deepspeed_zero3_enabled():
            import deepspeed

            # We cannot initialize the model on meta device with deepspeed when not quantized
            if not is_quantized and not _is_ds_init_called:
                logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
                init_contexts.extend(
                    [
                        init.no_init_weights(),
                        deepspeed.zero.Init(config_dict_or_path=deepspeed_config()),
                        set_zero3_state(),
                    ]
                )
            elif is_quantized:
                init_contexts.extend([torch.device("meta"), set_quantized_state()])
        else:
            init_contexts.append(torch.device("meta"))

        return init_contexts

    def _get_dtype_plan(self, dtype: torch.dtype) -> dict:
        """Create the dtype_plan describing modules/parameters that should use the `keep_in_fp32` flag."""
        dtype_plan = {}

        # The _keep_in_fp32_modules flag is only used to avoid bf16 -> fp16 casting precision issues. It was introduced
        # in case of force loading a model that should stay in bf16 in fp16
        # See https://github.com/huggingface/transformers/issues/20287 for details.
        if self._keep_in_fp32_modules is not None and dtype == torch.float16:
            dtype_plan.update(dict.fromkeys(self._keep_in_fp32_modules, torch.float32))

        # The _keep_in_fp32_modules_strict was introduced to always force upcast to fp32, for both fp16 and bf16
        if self._keep_in_fp32_modules_strict is not None and dtype in (torch.float16, torch.bfloat16):
            dtype_plan.update(dict.fromkeys(self._keep_in_fp32_modules_strict, torch.float32))

        return dtype_plan

    def set_use_kernels(self, use_kernels, kernel_config: KernelConfig | None = None):
        """
        Set whether or not to use the `kernels` library to kernelize some layers of the model.
        Args:
            use_kernels (`bool`):
                Whether or not to use the `kernels` library to kernelize some layers of the model.
            kernel_config (`KernelConfig`, *optional*):
                The kernel configuration to use to kernelize the model. If `None`, the default kernel mapping will be used.
        """
        if use_kernels:
            if not is_kernels_available():
                raise ValueError(
                    "`use_kernels=True` requires kernels>=0.9.0. Please install the latest version with `pip install -U kernels`"
                )
            from kernels import use_kernel_mapping

            from .integrations.hub_kernels import register_kernel_mapping_transformers

            register_kernel_mapping_transformers()

            if kernel_config is not None and isinstance(kernel_config, KernelConfig):
                # This will make sure the mapping is valid, and the layers are registered in the model
                kernel_config.sanitize_kernel_mapping(self)

                # This will create a compatible mapping for the model with the kernels library
                kernel_config.create_compatible_mapping(self)

                # This is a context manager to override the default kernel mapping
                # We are calling kernelize inside this context manager using the use_kernels setter
                # Param inherit_mapping should be False to avoid still loading kernel from remote
                inherit_mapping = not kernel_config.use_local_kernel
                with use_kernel_mapping(kernel_config.kernel_mapping, inherit_mapping=inherit_mapping):
                    self.use_kernels = True
            # We use the default kernel mapping in .integrations.hub_kernels
            else:
                self.use_kernels = True
        else:
            self.use_kernels = False

    @classmethod
    def from_pretrained(
        cls: type[SpecificPreTrainedModelType],
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        config: PreTrainedConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        **kwargs,
    ) -> SpecificPreTrainedModelType:
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments `config` and `state_dict`).
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PreTrainedConfig, str, os.PathLike]`, *optional*):
                Can be either:

                    - an instance of a class derived from [`PreTrainedConfig`],
                    - a string or path valid as input to [`~PreTrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            state_dict (`dict[str, torch.Tensor]`, *optional*):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
                [`~PreTrainedModel.from_pretrained`] is not a simpler option.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `hf auth login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>
            attn_implementation (`str`, *optional*):
                The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)), or `"flash_attention_3"` (using [Dao-AILab/flash-attention/hopper](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

                Accept HF kernel references in the form:
                  <namespace>/<repo_name>[@<revision>][:<kernel_name>]

                - <namespace> and <repo_name> are any non-"/" and non-":" sequences.
                - "@<revision>" is optional (branch, tag, or commit-ish), e.g. "@main", "@v1.2.0", "@abc123".
                - ":<kernel_name>" is optional and selects a function inside the kernel repo.
                - Both options can appear together and in this order only: @revision first, then :kernel_name.
                - We intentionally allow a leading "<wrapper>|" prefix (e.g., "flash|...") because the code
                  strips it before loading; '|' is not excluded in the character classes here.

                Examples that match:
                  "org/model"
                  "org/model@main"
                  "org/model:custom_kernel"
                  "org/model@v1.2.3:custom_kernel"
            experts_implementation (`str`, *optional*):
                The experts implementation to use in the model (if relevant). Can be any of:

                - `"eager"` (sequential implementation of the experts matrix multiplications).
                - `"batched_mm"` (using [`torch.bmm`](https://pytorch.org/docs/stable/generated/torch.bmm.html)).
                - `"grouped_mm"` (using [`torch.nn.functional.grouped_mm`](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.grouped_mm.html)).

                By default, if the model supports it, `"grouped_mm"` will be used. The default is otherwise the manual `"eager"` implementation.

            > Parameters for big model inference

            dtype (`str` or `torch.dtype`, *optional*, defaults to `"auto"`):
                Override the default `torch_dtype` and load the model under a specific `dtype`. The different options
                are:

                1. `torch.float16` or `torch.bfloat16` or `torch.float`: load in a specified
                  `dtype`, ignoring the model's `config.dtype` if one exists. If not specified
                  - the model will get loaded in `torch.float` (fp32).

                2. `"auto"` - A `dtype` or `torch_dtype` entry in the `config.json` file of the model will be
                  attempted to be used. If this entry isn't found then next check the `dtype` of the first weight in
                  the checkpoint that's of a floating point type and use that as `dtype`. This will load the model
                  using the `dtype` it was saved in at the end of the training. It can't be used as an indicator of how
                  the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.

                3. A string that is a valid `torch.dtype`. E.g. "float32" loads the model in `torch.float32`, "float16" loads in `torch.float16` etc.

                <Tip>

                For some models the `dtype` they were trained in is unknown - you may try to check the model's paper or
                reach out to the authors and ask them to add this information to the model's card and to insert the
                `dtype` or `torch_dtype` entry in `config.json` on the hub.

                </Tip>

            device_map (`str` or `dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
                like `1`) on which the model will be allocated, the device map will map the entire model to this
                device. Passing `device_map = 0` means put the whole model on GPU 0.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier to maximum memory if using `device_map`. Will default to the maximum memory available for each
                GPU and the available CPU RAM if unset.
            tp_plan (`Optional[Union[dict, str]]`, *optional*):
                A torch tensor parallel plan, see [here](https://pytorch.org/tutorials/intermediate/TP_tutorial.html). Use `tp_plan="auto"` to
                use the predefined plan based on the model. If it's a dict, then it should match between module names and desired layout.
                Note that if you use it, you should launch your script accordingly with `torchrun [args] script.py`. This will be much
                faster than using a `device_map`, but has limitations.
            tp_size (`str`, *optional*):
                A torch tensor parallel degree. If not provided would default to world size.
            device_mesh (`torch.distributed.DeviceMesh`, *optional*):
                A torch device mesh. If not provided would default to world size. Used only for tensor parallel for now.
                If provided, it has to contain dimension named `"tp"` in case it's > 1 dimensional, this dimension will be used for tensor parallelism
            offload_folder (`str` or `os.PathLike`, *optional*):
                If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
            offload_buffers (`bool`, *optional*):
                Whether or not to offload the buffers with the model parameters.
            quantization_config (`Union[QuantizationConfigMixin,Dict]`, *optional*):
                A dictionary of configuration parameters or a QuantizationConfigMixin object for quantization (e.g
                bitsandbytes, gptq).
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                Whether or not to use `safetensors` checkpoints. Defaults to `None`. If not specified and `safetensors`
                is not installed, it will be set to `False`.
            weights_only (`bool`, *optional*, defaults to `True`):
                Indicates whether unpickler should be restricted to loading only tensors, primitive types,
                dictionaries and any types added via torch.serialization.add_safe_globals().
                When set to False, we can load wrapper tensor subclass weights.
            key_mapping (`dict[str, str], *optional*):
                A potential mapping of the weight names if using a model on the Hub which is compatible to a Transformers
                architecture, but was not converted accordingly.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PreTrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>

        Examples:

        ```python
        >>> from transformers import BertConfig, BertModel

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model = BertModel.from_pretrained("./test/saved_model/")
        >>> # Update configuration during loading.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", output_attentions=True)
        >>> assert model.config.output_attentions == True
        ```
        """
        state_dict = kwargs.pop("state_dict", None)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        dtype = kwargs.pop("dtype", None)
        torch_dtype = kwargs.pop("torch_dtype", None)  # kept for BC
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_buffers = kwargs.pop("offload_buffers", False)
        quantization_config = kwargs.pop("quantization_config", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        variant = kwargs.pop("variant", None)
        adapter_kwargs = (kwargs.pop("adapter_kwargs", {}) or {}).copy()
        adapter_name = kwargs.pop("adapter_name", "default")
        generation_config = kwargs.pop("generation_config", None)
        gguf_file = kwargs.pop("gguf_file", None)
        tp_plan = kwargs.pop("tp_plan", None)
        tp_size = kwargs.pop("tp_size", None)
        distributed_config: DistributedConfig = kwargs.pop("distributed_config", None)
        device_mesh = kwargs.pop("device_mesh", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        use_kernels = kwargs.pop("use_kernels", False)
        kernel_config = kwargs.pop("kernel_config", None)
        key_mapping = kwargs.pop("key_mapping", None)

        if distributed_config is not None and tp_plan is None:
            tp_plan = "auto"

        # Not used anymore -- remove them from the kwargs
        for name in ["mirror", "_fast_init", "low_cpu_mem_usage", "from_tf", "from_flax", "offload_state_dict"]:
            _ = kwargs.pop(name, None)

        # For BC on torch_dtype argument
        if torch_dtype is not None:
            dtype = dtype if dtype is not None else torch_dtype
        if dtype is None:
            dtype = "auto"

        if is_offline_mode() and not local_files_only:
            local_files_only = True

        download_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "local_files_only": local_files_only,
            "token": token,
            "revision": revision,
            "subfolder": subfolder,
        }
        download_kwargs_with_commit = {**download_kwargs, "commit_hash": commit_hash}

        if state_dict is not None and (pretrained_model_name_or_path is not None or gguf_file is not None):
            raise ValueError(
                "`state_dict` cannot be passed together with a model name or a `gguf_file`. Use one of the two loading strategies."
            )

        if device_map == "auto" and int(os.environ.get("WORLD_SIZE", "0")):
            logger.info(
                "You've set device_map=`auto` while triggering a distributed run with torchrun. This might lead to unexpected behavior. "
                "If your plan is to load the model on each device, you should set device_map={"
                ": PartialState().process_index} where PartialState comes from accelerate library"
            )

        if tp_plan is not None or tp_size is not None:  # TP warnings, and setup
            device_map, device_mesh, tp_size = initialize_tensor_parallelism(
                tp_plan, tp_size=tp_size, device_mesh=device_mesh, device_map=device_map
            )

        if gguf_file is not None and not is_accelerate_available():
            raise ValueError("accelerate is required when loading a GGUF file `pip install accelerate`.")

        if adapter_kwargs is None:
            adapter_kwargs = {}

        _adapter_model_path, pretrained_model_name_or_path, adapter_kwargs = maybe_load_adapters(
            pretrained_model_name_or_path,
            download_kwargs_with_commit,
            **adapter_kwargs,
        )
        device_map = check_and_set_device_map(device_map)  # warn, error and fix the device map

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        # Load config if we don't provide a configuration
        if not isinstance(config, PreTrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                return_unused_kwargs=True,
                gguf_file=gguf_file,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **download_kwargs,
                **kwargs,
            )
            if "gguf_file" in model_kwargs:
                model_kwargs.pop("gguf_file")
            commit_hash = model_kwargs.pop("_commit_hash", commit_hash)
        else:
            config = copy.deepcopy(config)
            model_kwargs = kwargs
            commit_hash = getattr(config, "_commit_hash", commit_hash)

        download_kwargs_with_commit["commit_hash"] = commit_hash

        # Because some composite configs call super().__init__ before instantiating the sub-configs, we need this call
        # to correctly redispatch recursively if the kwarg is provided
        if "attn_implementation" in kwargs:
            config._attn_implementation = kwargs.pop("attn_implementation")

        if "experts_implementation" in kwargs:
            config._experts_implementation = kwargs.pop("experts_implementation")

        hf_quantizer, config, device_map = get_hf_quantizer(
            config, quantization_config, device_map, weights_only, user_agent
        )

        if gguf_file:
            if hf_quantizer is not None:
                raise ValueError(
                    "You cannot combine Quantization and loading a model from a GGUF file, try again by making sure you did not passed a `quantization_config` or that you did not load a quantized model from the Hub."
                )
            if device_map is not None and (
                (isinstance(device_map, dict) and "disk" in device_map.values()) or "disk" in device_map
            ):
                raise RuntimeError(
                    "One or more modules is configured to be mapped to disk. Disk offload is not supported for models "
                    "loaded from GGUF files."
                )

        if kernel_config is not None and not use_kernels:
            logger.warning_once(
                "A kernel_config was provided but use_kernels is False; setting use_kernels=True automatically. To suppress this warning, explicitly set use_kernels to True."
            )
            use_kernels = True

        checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            variant=variant,
            gguf_file=gguf_file,
            use_safetensors=use_safetensors,
            download_kwargs=download_kwargs_with_commit,
            user_agent=user_agent,
            is_remote_code=cls.is_remote_code(),
            transformers_explicit_filename=getattr(config, "transformers_weights", None),
        )

        is_quantized = hf_quantizer is not None

        if gguf_file:
            from .modeling_gguf_pytorch_utils import load_gguf_checkpoint

            # we need a dummy model to get the state_dict - for this reason, we keep the state_dict as if it was
            # passed directly as a kwarg from now on
            with torch.device("meta"):
                dummy_model = cls(config)
            state_dict = load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)[
                "tensors"
            ]

        # Find the correct dtype based on current state
        config, dtype = _get_dtype(
            dtype, checkpoint_files, config, sharded_metadata, state_dict, weights_only, hf_quantizer
        )

        config.name_or_path = pretrained_model_name_or_path
        model_init_context = cls.get_init_context(dtype, is_quantized, _is_ds_init_called)
        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        with ContextManagers(model_init_context):
            # Let's make sure we don't run the init function of buffer modules
            model = cls(config, *model_args, **model_kwargs)

            if hf_quantizer is not None:  # replace module with quantized modules (does not touch weights)
                hf_quantizer.preprocess_model(
                    model=model,
                    dtype=dtype,
                    device_map=device_map,
                    checkpoint_files=checkpoint_files,
                    use_kernels=use_kernels,
                )

        # Create the dtype_plan to potentially use the `keep_in_fp32` flags (this needs to be called on the already
        # instantiated model, as the flags can be modified by instances sometimes)
        dtype_plan = model._get_dtype_plan(dtype)

        # Obtain the weight conversion mapping for this model if any are registered
        weight_conversions = get_model_conversion_mapping(model, key_mapping, hf_quantizer)

        if _torch_distributed_available and device_mesh is not None:  # add hooks to nn.Modules: no weights
            model = distribute_model(model, tp_plan, distributed_config, device_mesh, tp_size)

        # Prepare the full device map
        if device_map is not None:
            device_map = _get_device_map(model, device_map, max_memory, hf_quantizer)

        # Finalize model weight initialization
        load_config = LoadStateDictConfig(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            sharded_metadata=sharded_metadata,
            device_map=device_map,
            disk_offload_folder=offload_folder,
            offload_buffers=offload_buffers,
            dtype=dtype,
            dtype_plan=dtype_plan,
            hf_quantizer=hf_quantizer,
            device_mesh=device_mesh,
            weights_only=weights_only,
            weight_mapping=weight_conversions,
            use_safetensors=use_safetensors,
            download_kwargs=download_kwargs,
        )
        loading_info, disk_offload_index = cls._load_pretrained_model(model, state_dict, checkpoint_files, load_config)
        loading_info = cls._finalize_model_loading(model, load_config, loading_info)
        model.eval()  # Set model in evaluation mode to deactivate Dropout modules by default
        model.set_use_kernels(use_kernels, kernel_config)

        # If it is a model with generation capabilities, attempt to load generation files (generation config,
        # custom generate function)
        if model.can_generate() and hasattr(model, "adjust_generation_fn") and not gguf_file:
            model.adjust_generation_fn(
                generation_config,
                from_auto_class,
                from_pipeline,
                pretrained_model_name_or_path,
                **download_kwargs,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )

        # If the device_map has more than 1 device: dispatch model with hooks on all devices
        if device_map is not None and len(set(device_map.values())) > 1:
            accelerate_dispatch(model, hf_quantizer, device_map, offload_folder, disk_offload_index, offload_buffers)

        if hf_quantizer is not None:
            model.hf_quantizer = hf_quantizer
            hf_quantizer.postprocess_model(
                model
            )  # usually a no-op but sometimes needed, e.g to remove the quant config when dequantizing

        if _adapter_model_path is not None:
            if token is not None:
                adapter_kwargs["token"] = token
            loading_info = model.load_adapter(
                _adapter_model_path,
                adapter_name=adapter_name,
                load_config=load_config,
                adapter_kwargs=adapter_kwargs,
            )

        if output_loading_info:
            return model, loading_info.to_dict()
        return model

    @staticmethod
    def _load_pretrained_model(
        model: "PreTrainedModel",
        state_dict: dict | None,
        checkpoint_files: list[str] | None,
        load_config: LoadStateDictConfig,
    ) -> tuple[LoadStateDictInfo, dict]:
        """Perform the actual loading of some checkpoints into a `model`, by reading them from disk and dispatching them accordingly."""
        is_quantized = load_config.is_quantized
        is_hqq_or_quark = is_quantized and load_config.hf_quantizer.quantization_config.quant_method in {
            QuantizationMethod.HQQ,
            QuantizationMethod.QUARK,
        }

        # Model's definition arriving here is final (TP hooks added, quantized layers replaces)
        expected_keys = list(model.state_dict().keys())

        if logger.level >= logging.WARNING:
            verify_tp_plan(expected_keys, getattr(model, "_tp_plan", None))

        # This offload index if for params explicitly on the "disk" in the device_map
        disk_offload_index = None
        # Prepare parameters offloading if needed
        if load_config.device_map is not None and "disk" in load_config.device_map.values():
            disk_offload_index = accelerate_disk_offload(
                model,
                load_config.disk_offload_folder,
                checkpoint_files,
                load_config.device_map,
                load_config.sharded_metadata,
                load_config.dtype,
                load_config.weight_mapping,
            )

        # Warmup cuda to load the weights much faster on devices
        if load_config.device_map is not None and not is_hqq_or_quark:
            expanded_device_map = expand_device_map(load_config.device_map, expected_keys)
            caching_allocator_warmup(model, expanded_device_map, load_config.hf_quantizer)

        error_msgs = []

        if is_deepspeed_zero3_enabled() and not is_quantized:
            if state_dict is None:
                merged_state_dict = {}
                for ckpt_file in checkpoint_files:
                    merged_state_dict.update(
                        load_state_dict(ckpt_file, map_location="cpu", weights_only=load_config.weights_only)
                    )
                state_dict = merged_state_dict
            error_msgs, missing_keys = _load_state_dict_into_zero3_model(model, state_dict, load_config)
            # This is not true but for now we assume only best-case scenario with deepspeed, i.e. perfectly matching checkpoints
            loading_info = LoadStateDictInfo(
                missing_keys=missing_keys,
                error_msgs=error_msgs,
                unexpected_keys=set(),
                mismatched_keys=set(),
                conversion_errors={},
            )
        else:
            all_pointer = set()
            if state_dict is not None:
                merged_state_dict = state_dict
            elif checkpoint_files is not None and checkpoint_files[0].endswith(".safetensors") and state_dict is None:
                merged_state_dict = {}
                for file in checkpoint_files:
                    file_pointer = safe_open(file, framework="pt", device="cpu")
                    all_pointer.add(file_pointer)
                    for k in file_pointer.keys():
                        merged_state_dict[k] = file_pointer.get_slice(k)  # don't materialize yet
            # Checkpoints are .bin
            elif checkpoint_files is not None:
                merged_state_dict = {}
                for ckpt_file in checkpoint_files:
                    merged_state_dict.update(load_state_dict(ckpt_file))
            else:
                raise ValueError("Neither a state dict nor checkpoint files were found.")

            loading_info, disk_offload_index = convert_and_load_state_dict_in_model(
                model=model,
                state_dict=merged_state_dict,
                load_config=load_config,
                tp_plan=model._tp_plan,
                disk_offload_index=disk_offload_index,
            )

            # finally close all opened file pointers
            for k in all_pointer:
                k.__exit__(None, None, None)

        return loading_info, disk_offload_index

    @staticmethod
    def _finalize_model_loading(
        model, load_config: LoadStateDictConfig, loading_info: LoadStateDictInfo
    ) -> LoadStateDictInfo:
        """Perform all post processing operations after having loaded some checkpoints into a model, such as moving
        missing keys from meta device to their expected device, reinitializing missing weights according to proper
        distributions, tying the weights and logging the loading report."""
        try:
            # Marks tied weights as `_is_hf_initialized` to avoid initializing them (it's very important for efficiency)
            model.mark_tied_weights_as_initialized(loading_info)

            # Move missing (and potentially mismatched) keys and non-persistent buffers back to their expected device from
            # meta device (because they were not moved when loading the weights as they were not in the loaded state dict)
            model._move_missing_keys_from_meta_to_device(
                loading_info.missing_and_mismatched(),
                load_config.device_map,
                load_config.device_mesh,
                load_config.hf_quantizer,
            )

            # Correctly initialize the missing (and potentially mismatched) keys (all parameters without the `_is_hf_initialized` flag)
            model._initialize_missing_keys(load_config.is_quantized)

            # Tie the weights
            model.tie_weights(missing_keys=loading_info.missing_keys, recompute_mapping=False)

            # Adjust missing and unexpected keys
            model._adjust_missing_and_unexpected_keys(loading_info)
        finally:
            log_state_dict_report(
                model=model,
                pretrained_model_name_or_path=load_config.pretrained_model_name_or_path,
                ignore_mismatched_sizes=load_config.ignore_mismatched_sizes,
                loading_info=loading_info,
                logger=logger,
            )

        return loading_info

    def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
        module_keys = {".".join(key.split(".")[:-1]) for key in names}

        # torch.nn.ParameterList is a special case where two parameter keywords
        # are appended to the module name, *e.g.* bert.special_embeddings.0
        module_keys = module_keys.union(
            {".".join(key.split(".")[:-2]) for key in names if len(key) > 0 and key[-1].isdigit()}
        )

        retrieved_modules = []
        # retrieve all modules that has at least one missing weight name
        for name, module in self.named_modules():
            if remove_prefix:
                _prefix = f"{self.base_model_prefix}."
                name = name.removeprefix(_prefix)
            elif add_prefix:
                name = ".".join([self.base_model_prefix, name]) if len(name) > 0 else self.base_model_prefix

            if name in module_keys:
                retrieved_modules.append(module)

        return retrieved_modules

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoModel"):
        """
        Register this class with a given auto class. This should only be used for custom models as the ones in the
        library are already mapped with an auto class.



        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoModel"`):
                The auto class to register this new model with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    def warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask):
        """
        Shows a one-time warning if the input_ids appear to contain padding and no attention mask was given.
        """

        # Skip the check during tracing.
        if is_tracing(input_ids):
            return

        if (attention_mask is not None) or (self.config.pad_token_id is None):
            return

        # Check only the first and last input IDs to reduce overhead.
        if self.config.pad_token_id in input_ids[:, [-1, 0]]:
            warn_string = (
                "We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See "
                "https://huggingface.co/docs/transformers/troubleshooting"
                "#incorrect-output-when-padding-tokens-arent-masked."
            )

            # If the pad token is equal to either BOS, EOS, or SEP, we do not know whether the user should use an
            # attention_mask or not. In this case, we should still show a warning because this is a rare case.
            # NOTE: `sep_token_id` is not used in all models and it can be absent in the config
            sep_token_id = getattr(self.config, "sep_token_id", None)
            if (
                (self.config.bos_token_id is not None and self.config.bos_token_id == self.config.pad_token_id)
                or (self.config.eos_token_id is not None and self.config.eos_token_id == self.config.pad_token_id)
                or (sep_token_id is not None and sep_token_id == self.config.pad_token_id)
            ):
                warn_string += (
                    f"\nYou may ignore this warning if your `pad_token_id` ({self.config.pad_token_id}) is identical "
                    f"to the `bos_token_id` ({self.config.bos_token_id}), `eos_token_id` ({self.config.eos_token_id}), "
                    f"or the `sep_token_id` ({sep_token_id}), and your input is not padded."
                )

            logger.warning_once(warn_string)

    @property
    def supports_tp_plan(self):
        """
        Returns whether the model has a tensor parallelism plan.
        """
        if self._tp_plan is not None:
            return True
        # Check if base model has a TP plan
        if getattr(self.base_model, "_tp_plan", None) is not None:
            return True
        if self.config.base_model_tp_plan is not None:
            return True
        return False

    @property
    def tp_size(self):
        """
        Returns the model's tensor parallelism degree.
        """
        # if None, the model didn't undergo tensor parallel sharding
        return self._tp_size

    @property
    def supports_pp_plan(self):
        if self._pp_plan is not None:
            return True
        # Check if base model has PP plan
        if getattr(self.base_model, "_pp_plan", None) is not None:
            return True
        return False

    @property
    def loss_function(self):
        if hasattr(self, "_loss_function"):
            return self._loss_function

        loss_type = getattr(self, "loss_type", None)

        if loss_type is None or loss_type not in LOSS_MAPPING:
            logger.warning_once(
                f"`loss_type={loss_type}` was set in the config but it is unrecognized. "
                f"Using the default loss: `ForCausalLMLoss`."
            )
            loss_type = "ForCausalLM"
        return LOSS_MAPPING[loss_type]

    @loss_function.setter
    def loss_function(self, value):
        self._loss_function = value

    def kernelize(self, mode=None):
        if not is_kernels_available():
            raise ValueError(
                "Kernels are not available. To use kernels, please install kernels using `pip install kernels`"
            )
        from kernels import Device, Mode, kernelize

        mode = Mode.INFERENCE if not self.training else Mode.TRAINING if mode is None else mode
        kernelize(self, device=Device(type=self.device.type), mode=mode)
        self._use_kernels = True

    @property
    def use_kernels(self) -> bool:
        return getattr(self, "_use_kernels", False)

    @use_kernels.setter
    def use_kernels(self, value: bool) -> None:
        # Avoid re-kernelizing if already enabled
        if bool(value) and getattr(self, "_use_kernels", False):
            return

        if value:
            self.kernelize()
        else:
            if getattr(self, "_use_kernels", False):
                logger.warning_once(
                    "Disabling kernels at runtime is a no-op as there is no 'unkernelize' routine; keeping current kernels active."
                )
            self._use_kernels = False

    def get_compiled_call(self, compile_config: CompileConfig | None) -> Callable:
        """Return a `torch.compile`'d version of `self.__call__`. This is useful to dynamically choose between
        non-compiled/compiled `forward` during inference, especially to switch between prefill (where we don't
        want to use compiled version to avoid recomputing the graph with new shapes) and iterative decoding
        (where we want the speed-ups of compiled version with static shapes)."""
        # Only reset it if not present or different from previous config
        if "llama4" in self.config.model_type:  # TODO try to enable for FULL COMPILE HYBRID CACHE SUPPORT
            return self.__call__
        compile_config = compile_config or CompileConfig()
        default_config = getattr(self.generation_config, "compile_config", None) or CompileConfig()
        if (
            not hasattr(self, "_compiled_call")
            or getattr(self, "_last_compile_config", default_config) != compile_config
        ):
            self._last_compile_config = compile_config
            self._compiled_call = torch.compile(self.__call__, **compile_config.to_dict())
        return self._compiled_call

    @classmethod
    def is_backend_compatible(cls):
        return cls._supports_attention_backend

    def _move_missing_keys_from_meta_to_device(
        self,
        missing_keys: list[str],
        device_map: dict | None,
        device_mesh: "torch.distributed.device_mesh.DeviceMesh | None",
        hf_quantizer: HfQuantizer | None,
    ) -> None:
        """Move the missing keys (keys that are part of the model parameters, but were NOT found in the loaded state dicts)
        back from meta device to their device according to the `device_map` if any, else cpu. Takes care of sharding those
        missing parameters if `device_mesh` is provided, i.e. we are using TP.
        All non-persistent buffers are also moved back to the correct device (they are not part of the state_dict, but are
        not missing either).
        """
        is_quantized = hf_quantizer is not None
        # This is the only case where we do not initialize the model on meta device, so we don't have to do anything here
        if is_deepspeed_zero3_enabled() and not is_quantized:
            return

        # In this case we need to move everything back
        if is_fsdp_enabled() and not is_local_dist_rank_0() and not is_quantized:
            for key, param in self.named_parameters():
                value = torch.empty_like(param, device="cpu")
                _load_parameter_into_model(self, key, value)
            for key, buffer in self.named_buffers():
                value = torch.empty_like(buffer, device="cpu")
                _load_parameter_into_model(self, key, value)
            return

        # The tied weight keys are in the "missing" usually, but they should not be moved (they will be tied anyway)
        # This is especially important because if they are moved, they will lose the `_is_hf_initialized` flag, and they
        # will be re-initialized for nothing (which can be quite long)
        for key in missing_keys - self.all_tied_weights_keys.keys():
            param = self.get_parameter_or_buffer(key)
            param_device = get_device(device_map, key, valid_torch_device=True)
            value = torch.empty_like(param, device=param_device)
            # For TP, we may need to shard the param
            if device_mesh is not None:
                shard_and_distribute_module(
                    self, value, param, key, None, False, device_mesh.get_local_rank(), device_mesh
                )
            # Otherwise, just move it to device
            else:
                _load_parameter_into_model(self, key, value)
        # We need to move back non-persistent buffers as well, as they are not part of loaded weights anyway
        for key, buffer in self.named_non_persistent_buffers():
            buffer_device = get_device(device_map, key, valid_torch_device=True)
            value = torch.empty_like(buffer, device=buffer_device)
            _load_parameter_into_model(self, key, value)

    def _initialize_missing_keys(self, is_quantized: bool) -> None:
        """
        Initialize the missing keys (keys that are part of the model parameters, but were NOT found in the loaded state dicts), according to
        `_initialize_weights`. Indeed, since the corresponding weights are missing from the state dict, they will not be replaced and need to
        be initialized correctly (i.e. weight initialization distribution).

        Params that are not missing have the `is_hf_initialized` flag.
        """
        # This will only initialize submodules that are not marked as initialized by the line above.
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            # keep_vars=True as we need the original tensors, so that the "_is_hf_initialized" is present on them
            not_initialized_parameters = list(
                {v for v in self.state_dict(keep_vars=True).values() if not getattr(v, "_is_hf_initialized", False)}
            )
            with deepspeed.zero.GatheredParameters(not_initialized_parameters, modifier_rank=0):
                self.initialize_weights()
        else:
            self.initialize_weights()

    def _adjust_missing_and_unexpected_keys(self, loading_info: LoadStateDictInfo) -> None:
        """Adjust the `missing_keys` and `unexpected_keys` based on current model's exception rules, to avoid
        raising unneeded warnings/errors. This is performed in-place.
        """
        # Old checkpoints may have keys for rotary_emb.inv_freq for each layer, however we moved this buffer to the main model
        # (so the buffer name has changed). Remove them in such a case. This is another exception that was not added to
        # `_keys_to_ignore_on_load_unexpected` as it touches many models -> we add it manually to the existing patterns
        has_inv_freq_buffers = any(buffer.endswith("rotary_emb.inv_freq") for buffer, _ in self.named_buffers())
        additional_unexpected_patterns = [r"rotary_emb\.inv_freq"] if has_inv_freq_buffers else []

        missing_patterns = self._keys_to_ignore_on_load_missing or []
        unexpected_patterns = (self._keys_to_ignore_on_load_unexpected or []) + additional_unexpected_patterns
        ignore_missing_regex, ignore_unexpected_regex = None, None
        if len(missing_patterns) > 0:
            ignore_missing_regex = re.compile("|".join(rf"({pattern})" for pattern in missing_patterns))
        if len(unexpected_patterns) > 0:
            ignore_unexpected_regex = re.compile("|".join(rf"({pattern})" for pattern in unexpected_patterns))

        # Clean-up missing keys
        if ignore_missing_regex is not None:
            loading_info.missing_keys = {
                key for key in loading_info.missing_keys if ignore_missing_regex.search(key) is None
            }

        # Clean-up unexpected keys
        if ignore_unexpected_regex is not None:
            loading_info.unexpected_keys = {
                key for key in loading_info.unexpected_keys if ignore_unexpected_regex.search(key) is None
            }

    def mark_tied_weights_as_initialized(self, loading_info):
        """Adds the `_is_hf_initialized` flag on parameters that will be tied, in order to avoid initializing them
        later as they will be tied (overwritten) anyway.
        This is very important as most embeddings are tied, and they are huge params (vocabularies are often 256k), so
        running inits on them is very costly."""
        for tied_param in self.all_tied_weights_keys.keys():
            param = self.get_parameter(tied_param)
            param._is_hf_initialized = True

        # Some remote code models define module tying (not parameter tying) in their __init__. When modules themselves are shared,
        # weights inside both modules appear in the `state_dict` but only one will appear in the safetensors checkpoints
        # as they are inherently tied because the 2 modules are the same object. In this case, once we load a parameter
        # inside one of the 2 modules, the other will also automatically be loaded and will have the `_is_hf_initialized`
        # flag (because we call `setattr` with the loaded param on the module, which is the same object), but its counterpart
        # will still appear as a missing key as we never get it out of the set (because it appears in the state_dict as well).
        # So we remove it now - otherwise it's considered missing and will be wrongly reinitialized
        # Note: this is never an issue in main Transformers, as we never do module-tying, only parameter-tying, and we know
        # which params are supposed to be tied to which other params
        if self.is_remote_code():
            # Remove those that are already initialized, but appear as missing due to module tying
            loading_info.missing_keys = {
                key
                for key in loading_info.missing_keys
                if not getattr(self.get_parameter_or_buffer(key), "_is_hf_initialized", False)
            }

    def get_parameter_or_buffer(self, target: str):
        """
        Return the parameter or buffer given by `target` if it exists, otherwise throw an error. This combines
        `get_parameter()` and `get_buffer()` in a single handy function. If the target is an `_extra_state` attribute,
        it will return the extra state provided by the module. Note that it only work if `target` is a leaf of the model.
        """
        try:
            return self.get_parameter(target)
        except AttributeError:
            pass
        try:
            return self.get_buffer(target)
        except AttributeError:
            pass
        module, param_name = get_module_from_name(self, target)
        if (
            param_name == "_extra_state"
            and getattr(module.__class__, "get_extra_state", torch.nn.Module.get_extra_state)
            is not torch.nn.Module.get_extra_state
        ):
            return module.get_extra_state()

        raise AttributeError(f"`{target}` is neither a parameter, buffer, nor extra state.")

    def named_non_persistent_buffers(
        self, recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Similar to `named_buffers`, but only yield non-persistent ones. It is handy as it's not perfectly straightforward
        to know if they are persistent or not"""
        for name, tensor in self.named_buffers(recurse=recurse, remove_duplicate=remove_duplicate):
            # We have to grab the parent here, as the attribute `_non_persistent_buffers_set` is on the immediate
            # parent only
            parent, buf_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = self.get_submodule(parent)
            if buf_name in parent._non_persistent_buffers_set:
                yield name, tensor

    def train(self, mode: bool = True):
        out = super().train(mode)
        if self.use_kernels:
            self.kernelize()
        return out

    def eval(self):
        return self.train(False)

    @classmethod
    def is_remote_code(cls) -> bool:
        return cls._auto_class is not None


PreTrainedModel.push_to_hub = copy_func(PreTrainedModel.push_to_hub)
if PreTrainedModel.push_to_hub.__doc__ is not None:
    PreTrainedModel.push_to_hub.__doc__ = PreTrainedModel.push_to_hub.__doc__.format(
        object="model", object_class="AutoModel", object_files="model file"
    )


def unwrap_model(model: nn.Module, recursive: bool = False) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
        recursive (`bool`, *optional*, defaults to `False`):
            Whether to recursively extract all cases of `module.module` from `model` as well as unwrap child sublayers
            recursively, not just the top-level distributed containers.
    """
    # Use accelerate implementation if available (should always be the case when using torch)
    # This is for pytorch, as we also have to handle things like dynamo
    if is_accelerate_available():
        kwargs = {}
        if recursive:
            kwargs["recursive"] = recursive
        return extract_model_from_parallel(model, **kwargs)
    else:
        # since there could be multiple levels of wrapping, unwrap recursively
        if hasattr(model, "module"):
            return unwrap_model(model.module)
        else:
            return model


def is_accelerator_device(device: str | int | torch.device) -> bool:
    """Check if the device is an accelerator. We need to function, as device_map can be "disk" as well, which is not
    a proper `torch.device`.
    """
    if device == "disk":
        return False
    else:
        return torch.device(device).type not in ["meta", "cpu"]


def get_total_byte_count(
    model: PreTrainedModel, accelerator_device_map: dict, hf_quantizer: HfQuantizer | None = None
):
    """
    This utility function calculates the total bytes count needed to load the model on each device.
    This is useful for caching_allocator_warmup as we want to know how much cache we need to pre-allocate.
    """

    total_byte_count = defaultdict(lambda: 0)
    tied_param_names = model.all_tied_weights_keys.keys()
    tp_plan = model._tp_plan if torch.distributed.is_available() and torch.distributed.is_initialized() else []

    for param_name, device in accelerator_device_map.items():
        # Skip if the parameter has already been accounted for (tied weights)
        if param_name in tied_param_names:
            continue

        param = model.get_parameter_or_buffer(param_name)

        if hf_quantizer is not None:
            dtype_size = hf_quantizer.param_element_size(model, param_name, param)
        else:
            dtype_size = param.element_size()

        param_byte_count = param.numel() * dtype_size

        if len(tp_plan) > 0:
            is_part_of_plan = _get_parameter_tp_plan(param_name, tp_plan, is_weight=True) is not None
            param_byte_count //= torch.distributed.get_world_size() if is_part_of_plan else 1

        total_byte_count[device] += param_byte_count
    return total_byte_count


def caching_allocator_warmup(model: PreTrainedModel, expanded_device_map: dict, hf_quantizer: HfQuantizer | None):
    """This function warm-ups the caching allocator based on the size of the model tensors that will reside on each
    device. It allows to have one large call to Malloc, instead of recursively calling it later when loading
    the model, which is actually the loading speed bottleneck.
    Calling this function allows to cut the model loading time by a very large margin.

    A few facts related to loading speed (taking into account the use of this function):
    - When loading a model the first time, it is usually slower than the subsequent times, because the OS is very likely
    to cache the different state dicts (if enough resources/RAM are available)
    - Trying to force the OS to cache the files in advance (by e.g. accessing a small portion of them) is really hard,
    and not a good idea in general as this is low level OS optimizations that depend on resource usage anyway
    - As of 18/03/2025, loading a Llama 70B model with TP takes ~1 min without file cache, and ~13s with full file cache.
    The baseline, i.e. only loading the tensor shards on device and adjusting dtype (i.e. copying them) is ~5s with full cache.
    These numbers are reported for TP on 4 H100 GPUs.
    - It is useless to pre-allocate more than the model size in this function (i.e. using an `allocation_factor` > 1) as
    cudaMalloc is not a bottleneck at all anymore
    - Loading speed bottleneck is now almost only tensor copy (i.e. changing the dtype) and moving the tensors to the devices.
    However, we cannot really improve on those aspects obviously, as the data needs to be moved/copied in the end.
    """
    # Remove disk, cpu and meta devices, and cast to proper torch.device
    accelerator_device_map = {
        param: torch.device(device) for param, device in expanded_device_map.items() if is_accelerator_device(device)
    }
    if not accelerator_device_map:
        return

    total_byte_count = get_total_byte_count(model, accelerator_device_map, hf_quantizer)

    # This will kick off the caching allocator to avoid having to Malloc afterwards
    for device, byte_count in total_byte_count.items():
        if device.type in ["cuda", "xpu"]:
            accelerator_module = getattr(torch, device.type)
            index = device.index if device.index is not None else accelerator_module.current_device()
            free_device_memory, total_device_memory = accelerator_module.mem_get_info(index)
            unused_memory = accelerator_module.memory_reserved(index) - accelerator_module.memory_allocated(index)
            # If we have reserved but unused memory, we can lower the allocation we want to make, but only if it's still
            # higher than the unused memory. This is because otherwise torch will use that unused memory when performing
            # our own allocation, thus not allocating any new memory from the GPU. For example if byte_count=6 GiB,
            # unused_memory=4 GiB, then we cannot allocate only 2 GiB as this would *likely* (may not be exact, due to
            # fragmentation issues) simply use the pool of 4 GiB unused memory that is available. In those cases, it's better
            # to allocate more than the technically only 2 GiB required
            if byte_count - unused_memory > unused_memory:
                byte_count = byte_count - unused_memory
            # Minimum amount that will trigger new gpu allocation, even if it's technically "too much" compared to what we need
            elif byte_count - unused_memory > 1.5 * 1024**3:
                # Nothing we can do here, the memory will need to fill itself as we load params, but we cannot reallocate
                # from gpu until the unused memory is not filled
                if unused_memory + 1 > free_device_memory:
                    byte_count = 0
                # We allocate the minimum amount that will force new gpu allocation, even if it's technically "too much"
                else:
                    byte_count = unused_memory + 1
            # If we only need to reallocate less than 1.5 GiB of what is already allocated, then don't allocate more
            else:
                byte_count = 0
            # Allow up to (max device memory - 1.2 GiB) in resource-constrained hardware configurations. Trying to reserve more
            # than that amount might sometimes lead to unnecessary cuda/xpu OOM, if the last parameter to be loaded on the device is large,
            # and the remaining reserved memory portion is smaller than the param size -> torch will then try to fully re-allocate all
            # the param size, instead of using the remaining reserved part, and allocating only the difference, which can lead
            # to OOM. See https://github.com/huggingface/transformers/issues/37436#issuecomment-2808982161 for more details.
            # Note that we use an absolute value instead of device proportion here, as a 8GiB device could still allocate too much
            # if using e.g. 90% of device size, while a 140GiB device would allocate too little
            byte_count = min(byte_count, total_device_memory - 1.2 * 1024**3)
        # We divide by 2 here as we allocate in fp16
        _ = torch.empty(int(byte_count // 2), dtype=torch.float16, device=device, requires_grad=False)


class AttentionInterface(GeneralInterface):
    """
    Dict-like object keeping track of allowed attention functions. You can easily add a new attention function
    with a call to `register()`. If a model needs to locally overwrite an existing attention function, say `sdpa`,
    it needs to declare a new instance of this class inside the `modeling_<model>.py`, and declare it on that instance.
    """

    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {
        "flash_attention_3": flash_attention_forward,
        "flash_attention_2": flash_attention_forward,
        "flex_attention": flex_attention_forward,
        "sdpa": sdpa_attention_forward,
        "paged|flash_attention_3": paged_attention_forward,
        "paged|flash_attention_2": paged_attention_forward,
        "paged|sdpa": sdpa_attention_paged_forward,
        "paged|eager": eager_paged_attention_forward,
    }

    def get_interface(self, attn_implementation: str, default: Callable) -> Callable:
        """Return the requested `attn_implementation`. Also strictly check its validity, and raise if invalid."""
        if attn_implementation is None:
            logger.warning_once(
                "You tried to access the `AttentionInterface` with a `config._attn_implementation` set to `None`. This "
                "is expected if you use an Attention Module as a standalone Module. If this is not the case, something went "
                "wrong with the dispatch of `config._attn_implementation`"
            )
        elif attn_implementation != "eager" and attn_implementation not in self:
            raise KeyError(
                f"`{attn_implementation}` is not a valid attention implementation registered in the `AttentionInterface`"
            )
        return super().get(attn_implementation, default)


# Global AttentionInterface shared by all models which do not need to overwrite any of the existing ones
ALL_ATTENTION_FUNCTIONS: AttentionInterface = AttentionInterface()


class PreTrainedAudioTokenizerBase(PreTrainedModel):
    """
    Class that additionally defines the behavior of any `audio_tokenizer` to be added.
    Characteristic for any of them:
        1. Encode raw audio into discrete audio codebooks (with x channels)
        2. Decode from discrete audio codebooks back to raw audio
    It is possible that they can decode in different ways given a different representation
    but they are forced to support 2. nonetheless, e.g. see `DAC`.
    """

    @abstractmethod
    def encode(self, input_values: torch.Tensor, *args, **kwargs):
        """
        Encode raw audio retrieved from a respective `FeatureExtractor` into discrete audio codebooks (with x channels)
        """

    @abstractmethod
    def decode(self, audio_codes: torch.Tensor, *args, **kwargs):
        """Decode from discrete audio codebooks back to raw audio"""
