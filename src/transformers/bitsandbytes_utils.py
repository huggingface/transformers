from contextlib import contextmanager
from typing import Optional, Union

from .utils import is_bitsandbytes_available


if is_bitsandbytes_available():
    import torch
    import torch.nn as nn

    import bitsandbytes as bnb


def set_module_8bit_tensor_to_device(
    module: nn.Module, tensor_name: str, device: Union[int, str, torch.device], value: Optional[torch.Tensor] = None
):
    """
    Args:
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function). The
    function is adapted from `set_module_tensor_to_device` function from accelerate that is adapted to support the
    class `Int8Params` from `bitsandbytes`.
        module (`torch.nn.Module`): The module in which the tensor we want to move lives. param_name (`str`): The full
        name of the parameter/buffer. device (`int`, `str` or `torch.device`): The device on which to set the tensor.
        value (`torch.Tensor`, *optional*): The value of the tensor (useful when going from the meta device to any
            other device).
    """
    # Recurse if needed
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    if is_buffer:
        has_fp16_weights = None
    else:
        has_fp16_weights = getattr(module._parameters[tensor_name], "has_fp16_weights", None)

    if has_fp16_weights is not None:
        param = module._parameters[tensor_name]
        if param.device.type != "cuda":
            with torch.no_grad():
                if value is None:
                    new_value = old_value.to(device)
                elif isinstance(value, torch.Tensor):
                    new_value = value.to("cpu")
                else:
                    new_value = torch.tensor(value, device="cpu")
                new_value = bnb.nn.Int8Params(new_value, requires_grad=False, has_fp16_weights=has_fp16_weights).to(
                    device
                )
                module._parameters[tensor_name] = new_value
    else:
        with torch.no_grad():
            if value is None:
                new_value = old_value.to(device)
            elif isinstance(value, torch.Tensor):
                new_value = value.to(device)
            else:
                new_value = torch.tensor(value, device=device)
        if is_buffer:
            module._buffers[tensor_name] = new_value
        else:
            new_value = nn.Parameter(new_value, requires_grad=old_value.requires_grad)
            module._parameters[tensor_name] = new_value


@contextmanager
def init_empty_weights_8bit(include_buffers: bool = False):
    """
    Args:
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.
        include_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to also put all buffers on the meta device while initializing.
    Example:
    ```pyton
    import torch.nn as nn
    from accelerate import init_empty_weights
    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].

    </Tip>
    """
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            has_fp16_weights = getattr(param_cls, "has_fp16_weights", None)
            if has_fp16_weights is not None:
                module._parameters[name] = param_cls(
                    module._parameters[name].to(torch.device("meta")), has_fp16_weights=has_fp16_weights
                )
            else:
                module._parameters[name] = param_cls(module._parameters[name].to(torch.device("meta")))

    def register_empty_buffer(module, name, buffer):
        old_register_buffer(module, name, buffer)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(torch.device("meta"))

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer


def replace_8bit_linear(model):
    import bitsandbytes as bnb

    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_8bit_linear(module)

        if isinstance(module, nn.Linear) and n != "lm_head":
            with init_empty_weights_8bit():
                model._modules[n] = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    has_fp16_weights=False,
                    threshold=6.0,
                )
    return model
