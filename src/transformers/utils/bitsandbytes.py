from transformers.utils import is_accelerate_available, is_bitsandbytes_available


if is_bitsandbytes_available():
    import torch
    import torch.nn as nn

    import bitsandbytes as bnb

if is_accelerate_available():
    from accelerate import init_empty_weights


def set_module_8bit_tensor_to_device(module, tensor_name, device, value=None):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function). The
    function is adapted from `set_module_tensor_to_device` function from accelerate that is adapted to support the
    class `Int8Params` from `bitsandbytes`.

    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        tensor_name (`str`): The full
            name of the parameter/buffer.
        device (`int`, `str` or `torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`, *optional*):
            The value of the tensor (useful when going from the meta device to any other device).
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


def replace_8bit_linear(model):

    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_8bit_linear(module)

        if isinstance(module, nn.Linear) and n != "lm_head":
            with init_empty_weights():
                model._modules[n] = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    has_fp16_weights=False,
                    threshold=6.0,
                )
    return model
