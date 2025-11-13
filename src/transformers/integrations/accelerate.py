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
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING

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
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import check_tied_parameters_on_same_device, get_max_memory

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


def find_tied_parameters(model: "nn.Module", **kwargs):
    """
    Find the tied parameters in a given model.

    <Tip warning={true}>

    The signature accepts keyword arguments, but they are for the recursive part of this function and you should ignore
    them.

    </Tip>

    Args:
        model (`torch.nn.Module`): The model to inspect.

    Returns:
        list[list[str]]: A list of lists of parameter names being all tied together.

    Example:

    ```py
    >>> from collections import OrderedDict
    >>> import torch.nn as nn

    >>> model = nn.Sequential(OrderedDict([("linear1", nn.Linear(4, 4)), ("linear2", nn.Linear(4, 4))]))
    >>> model.linear2.weight = model.linear1.weight
    >>> find_tied_parameters(model)
    [['linear1.weight', 'linear2.weight']]
    ```
    """

    # get ALL model parameters and their names
    all_named_parameters = dict(model.named_parameters(remove_duplicate=False))

    # get ONLY unique named parameters,
    # if parameter is tied and have multiple names, it will be included only once
    no_duplicate_named_parameters = dict(model.named_parameters(remove_duplicate=True))

    # the difference of the two sets will give us the tied parameters
    tied_param_names = set(all_named_parameters.keys()) - set(no_duplicate_named_parameters.keys())

    # 'tied_param_names' contains the names of parameters that are tied in the model, but we do not know
    # which names refer to the same parameter. To identify this, we need to group them together.
    tied_param_groups = {}
    for tied_param_name in tied_param_names:
        tied_param = all_named_parameters[tied_param_name]
        for param_name, param in no_duplicate_named_parameters.items():
            # compare if parameters are the same, if so, group their names together
            if param is tied_param:
                if param_name not in tied_param_groups:
                    tied_param_groups[param_name] = []
                tied_param_groups[param_name].append(tied_param_name)

    return [sorted([weight] + list(set(tied))) for weight, tied in tied_param_groups.items()]


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
    model: "PreTrainedModel", hf_quantizer: "HfQuantizer | None"
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Compute the size of each submodule of a given model (in bytes).
    Returns a tuple of 2 dicts, the fist one containing a mapping of all the modules and the corresponding size
    in bytes, and the 2nd one containing a mapping from all leaf modules (modules containing parameters, the end of
    the model graph) and the corresponding sizes.
    """
    all_module_sizes = defaultdict(int)
    leaves_module_sizes = defaultdict(int)
    for name, param in model.state_dict().items():
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
    dtype: torch.dtype | None,
) -> dict:
    """Compute the final `device_map` to use if we passed a value in ['auto', 'balanced', 'balanced_low_0', 'sequential'].
    Otherwise, we check for any device inconsistencies in the device_map.
    """
    if isinstance(device_map, str):
        special_dtypes = {}
        if hf_quantizer is not None:
            special_dtypes = hf_quantizer.get_special_dtypes_update(model, dtype)

        target_dtype = dtype
        if hf_quantizer is not None:
            target_dtype = hf_quantizer.adjust_target_dtype(target_dtype)

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
            dtype=target_dtype,
            no_split_module_classes=no_split_modules,
            special_dtypes=special_dtypes,
        )

        if hf_quantizer is not None:
            hf_quantizer.validate_environment(device_map=device_map)

    elif device_map is not None:
        tied_params = find_tied_parameters(model)
        # check if we don't have tied param in different devices
        check_tied_parameters_on_same_device(tied_params, device_map)

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
