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
"""
Some of the functions here are derived from the `accelerate` library, with some tweaks for better performances
and simplicity/ease of use.
"""

import copy
import inspect
import os
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, Union

from ..utils import (
    is_accelerate_available,
    is_torch_available,
    is_torch_xpu_available,
    logging,
)
from ..utils.quantization_config import QuantizationMethod
from .deepspeed import is_deepspeed_zero3_enabled
from .fsdp import is_fsdp_enabled


if is_torch_available():
    import torch
    import torch.nn as nn

if is_accelerate_available():
    from accelerate import dispatch_model
    from accelerate.utils import get_max_memory
    from accelerate.utils.modeling import clean_device_map, get_max_layer_size, get_module_size_with_ties

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..quantizers import HfQuantizer


logger = logging.get_logger(__name__)


@contextmanager
def init_empty_weights(include_buffers: bool = False):
    """
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_empty_weights

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].
    Make sure to overwrite the default device_map param for [`load_checkpoint_and_dispatch`], otherwise dispatch is not
    called.

    </Tip>
    """
    with init_on_device(torch.device("meta"), include_buffers=include_buffers) as f:
        yield f


@contextmanager
def init_on_device(device: "torch.device", include_buffers: bool = False):
    """
    A context manager under which models are initialized with all parameters on the specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_on_device

    with init_on_device(device=torch.device("cuda")):
        tst = nn.Linear(100, 100)  # on `cuda` device
    ```
    """
    if include_buffers:
        with device:
            yield
        return

    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch:
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def check_and_set_device_map(device_map: "torch.device | int | str | dict | None") -> dict | str | None:
    from ..modeling_utils import get_torch_context_manager_or_global_device

    # Potentially detect context manager or global device, and use it (only if no device_map was provided)
    if device_map is None and not is_deepspeed_zero3_enabled():
        device_in_context = get_torch_context_manager_or_global_device()
        if device_in_context == torch.device("meta"):
            raise RuntimeError(
                "You are using `from_pretrained` with a meta device context manager or `torch.set_default_device('meta')`.\n"
                "This is an anti-pattern as `from_pretrained` wants to load existing weights.\nIf you want to initialize an "
                "empty model on the meta device, use the context manager or global device with `from_config`, or `ModelClass(config)`"
            )
        device_map = device_in_context

    # change device_map into a map if we passed an int, a str or a torch.device
    if isinstance(device_map, torch.device):
        device_map = {"": device_map}
    elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
        try:
            device_map = {"": torch.device(device_map)}
        except RuntimeError:
            raise ValueError(
                "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
            )
    elif isinstance(device_map, int):
        if device_map < 0:
            raise ValueError(
                "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
            )
        else:
            device_map = {"": device_map}

    if device_map is not None:
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed Zero-3 is not compatible with passing a `device_map`.")
        if not is_accelerate_available():
            raise ValueError(
                "Using a `device_map`, `tp_plan`, `torch.device` context manager or setting `torch.set_default_device(device)` "
                "requires `accelerate`. You can install it with `pip install accelerate`"
            )
    return device_map


def compute_module_sizes(
    model: "PreTrainedModel", hf_quantizer: "HfQuantizer | None" = None, buffers_only: bool = False
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Compute the size of each submodule of a given model (in bytes).
    Returns a tuple of 2 dicts, the fist one containing a mapping of all the modules and the corresponding size
    in bytes, and the 2nd one containing a mapping from all leaf modules (modules containing parameters, the end of
    the model graph) and the corresponding sizes.
    """
    all_module_sizes = defaultdict(int)
    leaves_module_sizes = defaultdict(int)

    if buffers_only:
        iterator = model.named_buffers()
    else:
        # We need parameters + buffers here, as state_dict does not count non-persistent buffers which are taking space
        def all_tensors():
            yield from model.named_parameters()
            yield from model.named_buffers()

        iterator = all_tensors()

    tied_keys = getattr(model, "all_tied_weights_keys", {}).keys()
    for name, param in iterator:
        # Do not count tied keys (the model is usually not tied yet here, so they will appear in the iterator)
        # If the model is already tied, then they simply do not appear in the iterator anyway (remove_duplicates=True by default)
        if name in tied_keys:
            continue
        if hf_quantizer is not None:
            dtype_size = hf_quantizer.param_element_size(model, name)
        else:
            dtype_size = param.element_size()
        size = param.numel() * dtype_size
        name_parts = name.split(".")
        for idx in range(len(name_parts)):
            all_module_sizes[".".join(name_parts[:idx])] += size
        if "." in name:
            leaves_module_sizes[name.rsplit(".", 1)[0]] += size

    return all_module_sizes, leaves_module_sizes


def compute_module_total_buffer_size(model: nn.Module, hf_quantizer: "HfQuantizer | None" = None):
    """
    Compute the total size of buffers in each submodule of a given model.
    """
    module_sizes, _ = compute_module_sizes(model, hf_quantizer, buffers_only=True)
    return module_sizes.get("", 0)


def get_balanced_memory(
    model: "PreTrainedModel",
    max_memory: dict[int | str, int | str] | None = None,
    no_split_module_classes: list[str] | None = None,
    hf_quantizer: "HfQuantizer | None" = None,
    low_zero: bool = False,
):
    """
    Compute a `max_memory` dictionary for [`infer_auto_device_map`] that will balance the use of each available GPU.

    <Tip>

    All computation is done analyzing sizes and dtypes of the model parameters. As a result, the model can be on the
    meta device (as it would if initialized within the `init_empty_weights` context manager).

    </Tip>

    Args:
        model (`PreTrainedModel`):
            The model to analyze.
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available if unset.
            Example: `max_memory={0: "1GB"}`.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        hf_quantizer (`HfQuantizer`, *optional*):
            A quantizer for the model.
        low_zero (`bool`, *optional*):
            Minimizes the number of weights on GPU 0, which is convenient when it's used for other operations (like the
            Transformers generate function).
    """
    # Get default / clean up max_memory
    user_not_set_max_memory = max_memory is None
    max_memory = get_max_memory(max_memory)
    # Check the number of accelerators available
    accelerator_max_memory = copy.deepcopy(max_memory)
    _, _ = accelerator_max_memory.pop("cpu", None), accelerator_max_memory.pop("disk", None)
    num_devices = len([d for d in accelerator_max_memory if accelerator_max_memory[d] > 0])

    if num_devices == 0:
        return max_memory

    if num_devices == 1:
        # We cannot do low_zero on just one GPU, but we will still reserve some memory for the buffer
        low_zero = False
        # If user just asked us to handle memory usage, we should avoid OOM
        if user_not_set_max_memory:
            for key in max_memory.keys():
                if isinstance(key, int):
                    max_memory[key] *= 0.9  # 90% is a good compromise
                    logger.info(
                        f"We will use 90% of the memory on device {key} for storing the model, and 10% for the buffer to avoid OOM. "
                        "You can set `max_memory` in to a higher value to use more memory (at your own risk)."
                    )
                    break  # only one device

    module_sizes, leave_modules_sizes = compute_module_sizes(model, hf_quantizer)
    per_gpu = module_sizes[""] // (num_devices - 1 if low_zero else num_devices)

    # We can't just set the memory to model_size // num_devices as it will end being too small: each GPU will get
    # slightly less layers and some layers will end up offload at the end. So this function computes a buffer size to
    # add which is the biggest of:
    # - the size of no split block (if applicable)
    # - the mean of the layer sizes
    if no_split_module_classes is None:
        no_split_module_classes = []
    elif not isinstance(no_split_module_classes, (list, tuple)):
        no_split_module_classes = [no_split_module_classes]

    # Identify the size of the no_split_block modules
    buffer = 0
    if len(no_split_module_classes) > 0:
        no_split_children = {}
        for name, size in module_sizes.items():
            if name == "":
                continue
            submodule = model.get_submodule(name)
            class_name = submodule.__class__.__name__
            if class_name in no_split_module_classes and class_name not in no_split_children:
                no_split_children[class_name] = size

            if set(no_split_children.keys()) == set(no_split_module_classes):
                break
        buffer = max(no_split_children.values()) if len(no_split_children) > 0 else 0

    mean_leaves = int(sum(leave_modules_sizes.values()) / max(len(leave_modules_sizes), 1))
    buffer = int(1.25 * max(buffer, mean_leaves))
    per_gpu += buffer

    # Sorted list of GPUs id (we may have some gpu ids not included in the our max_memory list - let's ignore them)
    gpus_idx_list = sorted(
        device_id for device_id, device_mem in max_memory.items() if isinstance(device_id, int) and device_mem > 0
    )
    # The last device is left with max_memory just in case the buffer is not enough.
    for idx in gpus_idx_list[:-1]:
        max_memory[idx] = min(max_memory[0] if low_zero and idx == 0 else per_gpu, max_memory[idx])

    if low_zero:
        min_zero = max(0, module_sizes[""] - sum([max_memory[i] for i in range(1, num_devices)]))
        max_memory[0] = min(min_zero, max_memory[0])

    return max_memory


def _get_device_map(
    model: "PreTrainedModel",
    device_map: dict | str | None,
    max_memory: dict | None,
    hf_quantizer: "HfQuantizer | None",
) -> dict:
    """Compute the final `device_map` to use if we passed a value in ['auto', 'balanced', 'balanced_low_0', 'sequential'].
    Otherwise, we check for any device inconsistencies in the device_map.
    """
    if isinstance(device_map, str):
        no_split_modules = model._get_no_split_modules(device_map)

        if device_map != "sequential":
            inferred_max_memory = get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_modules,
                hf_quantizer=hf_quantizer,
                low_zero=(device_map == "balanced_low_0"),
            )
        else:
            inferred_max_memory = get_max_memory(max_memory)
        if hf_quantizer is not None:
            inferred_max_memory = hf_quantizer.adjust_max_memory(inferred_max_memory)

        # `inferred_max_memory` contains non-reserved memory. There may be *unused* reserved memory in the GPU,
        # which we can use to allocate parameters.
        for device_name in inferred_max_memory:
            if isinstance(device_name, int):  # it's a GPU device
                if is_torch_xpu_available():
                    unused_memory = torch.xpu.memory_reserved(device_name) - torch.xpu.memory_allocated(device_name)
                else:
                    unused_memory = torch.cuda.memory_reserved(device_name) - torch.cuda.memory_allocated(device_name)
                inferred_max_memory[device_name] += unused_memory
            # respect the `max_memory` passed by the user
            if max_memory is not None and device_name in max_memory:
                inferred_max_memory[device_name] = min(inferred_max_memory[device_name], max_memory[device_name])

        device_map = infer_auto_device_map(
            model,
            max_memory=inferred_max_memory,
            no_split_module_classes=no_split_modules,
            hf_quantizer=hf_quantizer,
        )

        if hf_quantizer is not None:
            hf_quantizer.validate_environment(device_map=device_map)

    return device_map


def accelerate_dispatch(model, hf_quantizer, device_map, offload_folder, offload_index, offload_buffers):
    device_map_kwargs = {
        "device_map": device_map,
        "offload_dir": offload_folder,
        "offload_index": offload_index,
        "offload_buffers": offload_buffers,
    }
    if "skip_keys" in inspect.signature(dispatch_model).parameters:
        device_map_kwargs["skip_keys"] = model._skip_keys_device_placement
    # For HQQ method we force-set the hooks for single GPU envs
    if (
        "force_hooks" in inspect.signature(dispatch_model).parameters
        and hf_quantizer is not None
        and hf_quantizer.quantization_config.quant_method == QuantizationMethod.HQQ
    ):
        device_map_kwargs["force_hooks"] = True
    if (
        hf_quantizer is not None
        and hf_quantizer.quantization_config.quant_method == QuantizationMethod.FBGEMM_FP8
        and isinstance(device_map, dict)
        and ("cpu" in device_map.values() or "disk" in device_map.values())
    ):
        device_map_kwargs["offload_buffers"] = True

    if not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
        dispatch_model(model, **device_map_kwargs)


def get_disk_only_shard_files(device_map, weight_map):
    """
    Returns the list of shard files containing only weights offloaded to disk.
    """
    files_content = defaultdict(list)
    for weight_name, filename in weight_map.items():
        while len(weight_name) > 0 and weight_name not in device_map:
            weight_name = ".".join(weight_name.split(".")[:-1])
        files_content[filename].append(device_map[weight_name])

    return [fname for fname, devices in files_content.items() if set(devices) == {"disk"}]


def expand_device_map(device_map, param_names):
    """
    Expand a device map to return the correspondence parameter name to device.
    """
    new_device_map = {}
    for module, device in device_map.items():
        new_device_map.update(
            {p: device for p in param_names if p == module or p.startswith(f"{module}.") or module == ""}
        )
    return new_device_map


def accelerate_disk_offload(
    disk_offload_folder,
    checkpoint_files,
    device_map,
    checkpoint_keys,
    sharded_metadata,
    dtype,
):
    disk_only_shard_files = []
    if disk_offload_folder is not None:
        os.makedirs(disk_offload_folder, exist_ok=True)
    is_offloaded_safetensors = checkpoint_files is not None and checkpoint_files[0].endswith(".safetensors")
    if disk_offload_folder is None and not is_offloaded_safetensors:
        raise ValueError(
            "The current `device_map` had weights offloaded to the disk. Please provide an `offload_folder`"
            " for them. Alternatively, make sure you have `safetensors` installed if the model you are using"
            " offers the weights in this format."
        )
    if is_offloaded_safetensors:
        param_device_map = expand_device_map(device_map, checkpoint_keys)
        str_dtype = str(dtype).replace("torch.", "") if dtype is not None else "float32"
        if sharded_metadata is None:
            weight_map = dict.fromkeys(checkpoint_keys, checkpoint_files[0])
        else:
            folder = os.path.sep.join(checkpoint_files[0].split(os.path.sep)[:-1])
            weight_map = {k: os.path.join(folder, v) for k, v in weight_map.items()}
            # Find potential checkpoints containing only offloaded weights
            disk_only_shard_files = get_disk_only_shard_files(device_map, weight_map)
        disk_offload_index = {
            name: {
                "safetensors_file": file,
                "weight_name": name,
                "dtype": str_dtype,
            }
            for name, file in weight_map.items()
            if param_device_map[name] == "disk"
        }
    else:
        disk_offload_index = {}
    return disk_offload_index, disk_only_shard_files, is_offloaded_safetensors


def _init_infer_auto_device_map(
    model: nn.Module,
    max_memory: Optional[dict[Union[int, str], Union[int, str]]] = None,
    no_split_module_classes: Optional[list[str]] = None,
    tied_parameters: Optional[list[list[str]]] = None,
    hf_quantizer: "HfQuantizer | None" = None,
) -> tuple[
    list[Union[int, str]],
    dict[Union[int, str], Union[int, str]],
    list[Union[int, str]],
    list[int],
    dict[str, int],
    list[list[str]],
    list[str],
    list[tuple[str, nn.Module]],
]:
    """
    Initialize variables required for computing the device map for model allocation.
    """
    max_memory = get_max_memory(max_memory)
    if no_split_module_classes is None:
        no_split_module_classes = []
    elif not isinstance(no_split_module_classes, (list, tuple)):
        no_split_module_classes = [no_split_module_classes]

    devices = list(max_memory.keys())
    if "disk" not in devices:
        devices.append("disk")
    gpus = [device for device in devices if device not in ["cpu", "disk"]]

    # Devices that need to keep space for a potential offloaded layer.
    if "mps" in gpus:
        main_devices = ["mps"]
    elif len(gpus) > 0:
        main_devices = [gpus[0], "cpu"]
    else:
        main_devices = ["cpu"]

    module_sizes, _ = compute_module_sizes(model, hf_quantizer)

    if tied_parameters is None:
        if len(model.all_tied_weights_keys) > 0:
            # create a list of list of tied params based on unique tied groups
            groups = set(model.all_tied_weights_keys.values())
            tied_parameters = [
                sorted([k for k, v in model.all_tied_weights_keys.items() if v == target] + [target])
                for target in groups
            ]
        else:
            tied_parameters = [[]]

    # Direct submodules and parameters
    modules_to_treat = (
        list(model.named_parameters(recurse=False))
        + list(model.named_children())
        + list(model.named_buffers(recurse=False))
    )

    return (
        devices,
        max_memory,
        main_devices,
        gpus,
        module_sizes,
        tied_parameters,
        no_split_module_classes,
        modules_to_treat,
    )


def infer_auto_device_map(
    model: nn.Module,
    max_memory: Optional[dict[Union[int, str], Union[int, str]]] = None,
    no_split_module_classes: Optional[list[str]] = None,
    verbose: bool = False,
    clean_result: bool = True,
    offload_buffers: bool = False,
    tied_parameters: Optional[list[list[str]]] = None,
    hf_quantizer: "HfQuantizer | None" = None,
):
    """
    Compute a device map for a given model giving priority to GPUs, then offload on CPU and finally offload to disk,
    such that:
    - we don't exceed the memory available of any of the GPU.
    - if offload to the CPU is needed, there is always room left on GPU 0 to put back the layer offloaded on CPU that
      has the largest size.
    - if offload to the CPU is needed,we don't exceed the RAM available on the CPU.
    - if offload to the disk is needed, there is always room left on the CPU to put back the layer offloaded on disk
      that has the largest size.

    <Tip>

    All computation is done analyzing sizes and dtypes of the model parameters. As a result, the model can be on the
    meta device (as it would if initialized within the `init_empty_weights` context manager).

    </Tip>

    Args:
        model (`torch.nn.Module`):
            The model to analyze.
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available if unset.
            Example: `max_memory={0: "1GB"}`.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        verbose (`bool`, *optional*, defaults to `False`):
            Whether or not to provide debugging statements as the function builds the device_map.
        clean_result (`bool`, *optional*, defaults to `True`):
            Clean the resulting device_map by grouping all submodules that go on the same device together.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            In the layers that are offloaded on the CPU or the hard drive, whether or not to offload the buffers as
            well as the parameters.
    """

    # Initialize the variables
    (
        devices,
        max_memory,
        main_devices,
        gpus,
        module_sizes,
        tied_parameters,
        no_split_module_classes,
        modules_to_treat,
    ) = _init_infer_auto_device_map(model, max_memory, no_split_module_classes, tied_parameters, hf_quantizer)

    device_map = OrderedDict()
    current_device = 0
    device_memory_used = dict.fromkeys(devices, 0)
    device_buffer_sizes = {}
    device_minimum_assignment_memory = {}

    # Initialize maximum largest layer, to know which space to keep in memory
    max_layer_size, max_layer_names = get_max_layer_size(modules_to_treat, module_sizes, no_split_module_classes)

    # Ready ? This is going to be a bit messy.
    while len(modules_to_treat) > 0:
        name, module = modules_to_treat.pop(0)
        if verbose:
            print(f"\nTreating module {name}.")
        # Max size in the remaining layers may have changed since we took one, so we maybe update it.
        max_layer_names = [n for n in max_layer_names if n != name and not n.startswith(name + ".")]
        if len(max_layer_names) == 0:
            max_layer_size, max_layer_names = get_max_layer_size(
                [(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)],
                module_sizes,
                no_split_module_classes,
            )
        # Assess size needed
        module_size = module_sizes[name]

        # We keep relevant tied parameters only: one of the tied parameters in the group is inside the current module
        # and the other is not.
        # Note: If we are currently processing the name `compute.weight`, an other parameter named
        # e.g. `compute.weight_submodule.parameter`
        # needs to be considered outside the current module, hence the check with additional dots.
        tied_param_groups = [
            tied_group
            for tied_group in tied_parameters
            if any(name + "." in k + "." for k in tied_group) and not all(name + "." in k + "." for k in tied_group)
        ]

        if verbose and len(tied_param_groups) > 0:
            print(f"  Found the relevant tied param groups {tied_param_groups}")

        # Then we keep track of all the parameters that are tied to the current module, but not in the current module
        tied_params = sum(
            [[p for p in tied_group if name + "." not in p + "."] for tied_group in tied_param_groups], []
        )

        if verbose and len(tied_params) > 0:
            print(f"  So those parameters need to be taken into account {tied_params}")

        device = devices[current_device]
        current_max_size = max_memory[device] if device != "disk" else None
        current_memory_reserved = 0
        # Reduce max size available by the largest layer.
        if devices[current_device] in main_devices:
            current_max_size = current_max_size - max_layer_size
            current_memory_reserved = max_layer_size

        module_size_with_ties, tied_module_names, tied_modules = get_module_size_with_ties(
            tied_params, module_size, module_sizes, modules_to_treat
        )

        # The module and its tied modules fit on the current device.
        if current_max_size is None or device_memory_used[device] + module_size_with_ties <= current_max_size:
            if verbose:
                output = f"Putting {name}"

                if tied_module_names:
                    output += f" and {tied_module_names}"
                else:
                    output += f" (size={module_size})"

                if current_max_size is not None:
                    output += f" (available={current_max_size - device_memory_used[device]})"

                output += f" on {device}."
                print(output)

            device_memory_used[device] += module_size_with_ties

            # Assign the primary module to the device.
            device_map[name] = device

            # Assign tied modules if any.
            for tied_module_name in tied_module_names:
                if tied_module_name in [m[0] for m in modules_to_treat]:
                    # Find the index of the tied module in the list
                    tied_module_index = next(i for i, (n, _) in enumerate(modules_to_treat) if n == tied_module_name)
                    # Remove the tied module from the list to prevent reprocessing
                    modules_to_treat.pop(tied_module_index)

                # Assign the tied module to the device
                device_map[tied_module_name] = device

            # Buffer Handling
            if not offload_buffers and isinstance(module, nn.Module):
                # Compute the total buffer size for the module
                current_buffer_size = compute_module_total_buffer_size(module, hf_quantizer)
                # Update the buffer size on the device
                device_buffer_sizes[device] = device_buffer_sizes.get(device, 0) + current_buffer_size

            continue

        # The current module itself fits, so we try to split the tied modules.
        if len(tied_params) > 0 and device_memory_used[device] + module_size <= current_max_size:
            # can we split one of the tied modules to make it smaller or do we need to go on the next device?
            if verbose:
                print(
                    f"Not enough space on {devices[current_device]} to put {name} and {tied_module_names} (space "
                    f"available {current_max_size - device_memory_used[device]}, needed size {module_size_with_ties})."
                )
            split_happened = False
            for tied_module_name, tied_module in zip(tied_module_names, tied_modules):
                tied_module_children = list(tied_module.named_children())
                if len(tied_module_children) == 0 or tied_module.__class__.__name__ in no_split_module_classes:
                    # can't break this one.
                    continue

                if verbose:
                    print(f"Splitting {tied_module_name}.")
                tied_module_children = list(tied_module.named_parameters(recurse=False)) + tied_module_children
                tied_module_children = [(f"{tied_module_name}.{n}", v) for n, v in tied_module_children]
                tied_module_index = [i for i, (n, _) in enumerate(modules_to_treat) if n == tied_module_name][0]

                modules_to_treat = (
                    [(name, module)]
                    + modules_to_treat[:tied_module_index]
                    + tied_module_children
                    + modules_to_treat[tied_module_index + 1 :]
                )
                # Update the max layer size.
                max_layer_size, max_layer_names = get_max_layer_size(
                    [(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)],
                    module_sizes,
                    no_split_module_classes,
                )
                split_happened = True
                break

            if split_happened:
                continue

            # If the tied module is not split, we go to the next device
            if verbose:
                print("None of the tied module can be split, going to the next device.")

        # The current module itself doesn't fit, so we have to split it or go to the next device.
        if device_memory_used[device] + module_size >= current_max_size:
            # Split or not split?
            modules_children = (
                []
                if isinstance(module, nn.Parameter) or isinstance(module, torch.Tensor)
                else list(module.named_children())
            )
            if verbose:
                print(
                    f"Not enough space on {devices[current_device]} to put {name} (space available "
                    f"{current_max_size - device_memory_used[device]}, module size {module_size})."
                )
            if len(modules_children) == 0 or module.__class__.__name__ in no_split_module_classes:
                # -> no split, we go to the next device
                if verbose:
                    print("This module cannot be split, going to the next device.")

            else:
                # -> split, we replace the module studied by its children + parameters
                if verbose:
                    print(f"Splitting {name}.")
                modules_children = list(module.named_parameters(recurse=False)) + modules_children
                modules_to_treat = [(f"{name}.{n}", v) for n, v in modules_children] + modules_to_treat
                # Update the max layer size.
                max_layer_size, max_layer_names = get_max_layer_size(
                    [(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)],
                    module_sizes,
                    no_split_module_classes,
                )
                continue

        if device_memory_used[device] == 0:
            device_minimum_assignment_memory[device] = module_size_with_ties + current_memory_reserved

        #  Neither the current module nor any tied modules can be split, so we move to the next device.
        device_memory_used[device] = device_memory_used[device] + current_memory_reserved
        current_device += 1
        modules_to_treat = [(name, module)] + modules_to_treat

    device_memory_used = {device: mem for device, mem in device_memory_used.items() if mem > 0}

    if clean_result:
        device_map = clean_device_map(device_map)

    non_gpu_buffer_size = device_buffer_sizes.get("cpu", 0) + device_buffer_sizes.get("disk", 0)
    if non_gpu_buffer_size > 0 and not offload_buffers:
        is_buffer_fit_any_gpu = False
        for gpu_device, gpu_max_memory in max_memory.items():
            if gpu_device == "cpu" or gpu_device == "disk":
                continue

            if not is_buffer_fit_any_gpu:
                gpu_memory_used = device_memory_used.get(gpu_device, 0)

                if gpu_max_memory >= non_gpu_buffer_size + gpu_memory_used:
                    is_buffer_fit_any_gpu = True

        if len(gpus) > 0 and not is_buffer_fit_any_gpu:
            logger.warning(
                f"Current model requires {non_gpu_buffer_size} bytes of buffer for offloaded layers, which seems does "
                f"not fit any GPU's remaining memory. If you are experiencing a OOM later, please consider using "
                f"offload_buffers=True."
            )

    if device_minimum_assignment_memory:
        devices_info = "\n".join(
            f"  - {device}: {mem} bytes required" for device, mem in device_minimum_assignment_memory.items()
        )
        logger.info(
            f"Based on the current allocation process, no modules could be assigned to the following devices due to "
            f"insufficient memory:\n"
            f"{devices_info}\n"
            f"These minimum requirements are specific to this allocation attempt and may vary. Consider increasing "
            f"the available memory for these devices to at least the specified minimum, or adjusting the model config."
        )

    check_tied_parameters_on_same_device(tied_parameters, device_map)
    return device_map


def _get_param_device(param, device_map):
    if param in device_map:
        return device_map[param]
    parent_param = ".".join(param.split(".")[:-1])
    if parent_param == param:
        raise ValueError(f"The `device_map` does not contain the module {param}.")
    else:
        return _get_param_device(parent_param, device_map)


def check_tied_parameters_on_same_device(tied_params, device_map):
    """
    Check if tied parameters are on the same device

    Args:
        tied_params (`List[List[str]]`):
            A list of lists of parameter names being all tied together.

        device_map (`Dict[str, Union[int, str, torch.device]]`):
            A map that specifies where each submodule should go.

    """
    for tie_param in tied_params:
        tie_param_devices = {}
        for param in tie_param:
            tie_param_devices[param] = _get_param_device(param, device_map)
        if len(set(tie_param_devices.values())) > 1:
            logger.warning(
                f"Tied parameters are on different devices: {tie_param_devices}. "
                "Please modify your custom device map or set `device_map='auto'`. "
            )
