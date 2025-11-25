import inspect
from collections import defaultdict
from inspect import signature

from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import get_module_from_name
from ..utils import (
    get_available_devices,
    is_accelerate_available,
    is_bitsandbytes_available,
    is_torch_available,
    logging,
)


if is_bitsandbytes_available():
    import bitsandbytes as bnb

if is_torch_available():
    import torch
    import torch.nn as nn

    from ..pytorch_utils import Conv1D

if is_accelerate_available():
    import accelerate
    from accelerate import init_empty_weights
    from accelerate.hooks import add_hook_to_module, remove_hook_from_module

logger = logging.get_logger(__name__)


class Bnb4bitQuantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        model: torch.nn.Module | None = None,
        missing_keys=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        we need to store some parameters to create the quantized weight. For example, bnb requires 6 values that are stored in the checkpoint to recover the quantized weight. So we store them in a dict that it stored in hf_quantizer for now as we can't save it in the op since we create an op per tensor.
        """
        target_key, value = tuple(input_dict.items())[0]
        value = value[0] if isinstance(value, list) else value

        full_name = target_key
        # update param name to get the weights instead of the quantized stats
        target_key = self.hf_quantizer.get_param_name(target_key)
        module, _ = get_module_from_name(model, target_key)

        if not self.hf_quantizer.pre_quantized:
            # Support models using `Conv1D` in place of `nn.Linear` (e.g. openai-community/gpt2) by transposing the weight matrix prior to quantization.
            # Since weights are saved in the correct "orientation", we skip transposing when loading.
            if issubclass(module.source_cls, Conv1D):
                value = value.T
            old_value = model.get_parameter_or_buffer(target_key)
            new_value = bnb.nn.Params4bit(value, requires_grad=False, **old_value.__dict__).to(value.device)
            # remove missing keys that were create when initializing Params4bit
            for key in new_value.quant_state.as_dict(packed=True).keys():
                missing_keys.discard(f"{full_name}.{key}")
            module._is_hf_initialized = True
            return {target_key: new_value}
        else:
            module_name = target_key.rsplit(".", 1)[0]
            # Save the states for later quantization when they are all gathered
            if not hasattr(self.hf_quantizer, "param_quant_stats"):
                self.hf_quantizer.param_quant_stats = defaultdict(dict)
            self.hf_quantizer.param_quant_stats[module_name].update({full_name: value})
            missing_keys.discard(full_name)
            # We are ready for quantization in this case (note, the +1 is for the weight itself)
            if len(self.hf_quantizer.param_quant_stats[module_name]) == len(self.hf_quantizer.bnb_keys) + 1:
                weight = self.hf_quantizer.param_quant_stats[module_name].pop(f"{module_name}.weight")
                new_value = bnb.nn.Params4bit.from_prequantized(
                    data=weight,
                    quantized_stats=self.hf_quantizer.param_quant_stats[module_name],
                    requires_grad=False,
                    device=value.device,
                    module=module,
                )
                module._is_hf_initialized = True
                del self.hf_quantizer.param_quant_stats[module_name]
                return {target_key: new_value}
            return {}


class Bnb8bitQuantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        model: torch.nn.Module | None = None,
        missing_keys=None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        target_key, value = tuple(input_dict.items())[0]
        value = value[0] if isinstance(value, list) else value

        module, tensor_name = get_module_from_name(model, target_key)
        module_name = target_key.rsplit(".", 1)[0]

        if not self.hf_quantizer.pre_quantized:
            # Support models using `Conv1D` in place of `nn.Linear` (e.g. openai-community/gpt2) by transposing the weight matrix prior to quantization.
            # Since weights are saved in the correct "orientation", we skip transposing when loading.
            if issubclass(module.source_cls, Conv1D):
                value = value.T
            value_device = value.device
            kwargs = getattr(module, tensor_name).__dict__
            kwargs.pop("SCB", None)
            new_value = bnb.nn.Int8Params(value.to("cpu"), requires_grad=False, **kwargs).to(value_device)
            missing_keys.discard(f"{module_name}.weight_format")
            missing_keys.discard(f"{module_name}.SCB")
            return {target_key: new_value}
        else:
            missing_keys.discard(target_key)
            # useless key that gets saved for no reason
            if tensor_name.endswith("weight_format"):
                return {}
            # Save the states for later quantization when they are all gathered
            if not hasattr(self.hf_quantizer, "param_quant_stats"):
                self.hf_quantizer.param_quant_stats = defaultdict(dict)
            self.hf_quantizer.param_quant_stats[module_name].update({target_key: value})
            # We are ready for quantization in this case (SCB and the weight)
            if len(self.hf_quantizer.param_quant_stats[module_name]) == 2:
                weight = self.hf_quantizer.param_quant_stats[module_name].pop(f"{module_name}.weight")
                kwargs = getattr(module, "weight").__dict__
                weight_device = weight.device
                new_value = bnb.nn.Int8Params(weight.to("cpu"), requires_grad=False, **kwargs).to(weight_device)
                setattr(new_value, "SCB", self.hf_quantizer.param_quant_stats[module_name][f"{module_name}.SCB"])
                del self.hf_quantizer.param_quant_stats[module_name]
                # sometimes, weight_format is not saved so we need to remove it from missing keys ...
                missing_keys.discard(f"{module_name}.weight_format")
                return {f"{module_name}.weight": new_value}
            return {}


def _replace_with_bnb_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
    pre_quantized=False,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successful or not.
    """
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if (isinstance(module, (nn.Linear, Conv1D))) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                with init_empty_weights():
                    if isinstance(module, Conv1D):
                        in_features, out_features = module.weight.shape
                    else:
                        in_features = module.in_features
                        out_features = module.out_features

                    if quantization_config.quantization_method() == "llm_int8":
                        new_module = bnb.nn.Linear8bitLt(
                            in_features,
                            out_features,
                            module.bias is not None,
                            has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                            threshold=quantization_config.llm_int8_threshold,
                        )
                        # hack to create the correct keys in the state dict with the right dtype
                        new_module.weight.SCB = torch.empty(1, dtype=torch.float32)
                        if pre_quantized:
                            new_module.weight.data = new_module.weight.data.to(dtype=torch.int8)
                        model._modules[name] = new_module
                        has_been_replaced = True
                    else:
                        if (
                            quantization_config.llm_int8_skip_modules is not None
                            and name in quantization_config.llm_int8_skip_modules
                        ):
                            pass
                        else:
                            extra_kwargs = (
                                {"quant_storage": quantization_config.bnb_4bit_quant_storage}
                                if "quant_storage" in list(signature(bnb.nn.Linear4bit).parameters)
                                else {}
                            )
                            new_module = bnb.nn.Linear4bit(
                                in_features,
                                out_features,
                                module.bias is not None,
                                quantization_config.bnb_4bit_compute_dtype,
                                compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                                quant_type=quantization_config.bnb_4bit_quant_type,
                                **extra_kwargs,
                            )
                            from bitsandbytes.functional import QuantState

                            # hack to create the correct keys in the state dict with the right dtype
                            absmax_dtype = (
                                torch.uint8 if quantization_config.bnb_4bit_use_double_quant else torch.float32
                            )
                            new_module.weight.quant_state = QuantState(
                                absmax=torch.empty(1, dtype=absmax_dtype),
                                code=torch.empty(1, dtype=torch.float32),
                                shape=(1,),
                                offset=torch.empty(1),
                                quant_type=quantization_config.bnb_4bit_quant_type,
                                state2=QuantState(
                                    absmax=torch.empty(1, dtype=torch.float32),
                                    code=torch.empty(1, dtype=torch.float32),
                                )
                                if quantization_config.bnb_4bit_use_double_quant
                                else None,
                            )
                            if pre_quantized:
                                # this is kind of an edge case when supporting both loading and quantization ...
                                # we need to set the right dtype as we cast the checkpoint with the dtype of the meta model
                                new_module.weight.data = new_module.weight.data.to(dtype=torch.uint8)
                            model._modules[name] = new_module
                            has_been_replaced = True
                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_bnb_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
                pre_quantized=pre_quantized,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_bnb_linear(
    model, modules_to_not_convert=None, current_key_name=None, quantization_config=None, pre_quantized=False
):
    """
    A helper function to replace all `torch.nn.Linear` modules by `bnb.nn.Linear8bit` modules from the `bitsandbytes`
    library. This will enable running your models using mixed int8 precision as described by the paper `LLM.int8():
    8-bit Matrix Multiplication for Transformers at Scale`. Make sure `bitsandbytes` compiled with the correct CUDA
    version of your hardware is installed before running this function. `pip install -i https://test.pypi.org/simple/
    bitsandbytes`

    The function will be run recursively and replace all `torch.nn.Linear` modules except for the `lm_head` that should
    be kept as a `torch.nn.Linear` module. The replacement is done under `init_empty_weights` context manager so no
    CPU/GPU memory is required to run this function. Int8 mixed-precision matrix decomposition works by separating a
    matrix multiplication into two streams: (1) and systematic feature outlier stream matrix multiplied in fp16
    (0.01%), (2) a regular stream of int8 matrix multiplication (99.9%). With this method, int8 inference with no
    predictive degradation is possible for very large models (>=176B parameters).

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`list[`str`]`, *optional*, defaults to `["lm_head"]`):
            Names of the modules to not convert in `Linear8bitLt`. In practice we keep the `lm_head` in full precision
            for numerical stability reasons.
        current_key_name (`list[`str`]`, *optional*):
            An array to track the current key of the recursion. This is used to check whether the current key (part of
            it) is not in the list of modules to not convert (for instances modules that are offloaded to `cpu` or
            `disk`).
        quantization_config ('transformers.utils.quantization_config.BitsAndBytesConfig'):
            To configure and manage settings related to quantization, a technique used to compress neural network models
            by reducing the precision of the weights and activations, thus making models more efficient in terms of both
            storage and computation.
    """
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    model, has_been_replaced = _replace_with_bnb_linear(
        model, modules_to_not_convert, current_key_name, quantization_config, pre_quantized=pre_quantized
    )

    if not has_been_replaced:
        logger.warning(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )
    return model


# Copied from PEFT: https://github.com/huggingface/peft/blob/47b3712898539569c02ec5b3ed4a6c36811331a1/src/peft/utils/integrations.py#L41
def dequantize_bnb_weight(weight: "torch.nn.Parameter", dtype: "torch.dtype", state=None):
    """
    Helper function to dequantize 4bit or 8bit bnb weights.

    If the weight is not a bnb quantized weight, it will be returned as is.
    """
    if not isinstance(weight, torch.nn.Parameter):
        raise TypeError(f"Input weight should be of type nn.Parameter, got {type(weight)} instead")

    cls_name = weight.__class__.__name__
    if cls_name not in ("Params4bit", "Int8Params"):
        return weight

    if cls_name == "Params4bit":
        output_tensor = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        logger.warning_once(
            f"The model is going to be dequantized in {output_tensor.dtype} - if you want to upcast it to another dtype, make sure to pass the desired dtype when quantizing the model through `bnb_4bit_quant_type` argument of `BitsAndBytesConfig`"
        )
        return output_tensor.to(dtype)

    if state.SCB is None:
        state.SCB = weight.SCB

    if hasattr(bnb.functional, "int8_vectorwise_dequant"):
        # Use bitsandbytes API if available (requires v0.45.0+)
        dequantized = bnb.functional.int8_vectorwise_dequant(weight.data, state.SCB)
    else:
        # Multiply by (scale/127) to dequantize.
        dequantized = weight.data * state.SCB.view(-1, 1) * 7.874015718698502e-3

    return dequantized.to(dtype)


def _create_accelerate_new_hook(old_hook):
    r"""
    Creates a new hook based on the old hook. Use it only if you know what you are doing !
    This method is a copy of: https://github.com/huggingface/peft/blob/748f7968f3a31ec06a1c2b0328993319ad9a150a/src/peft/utils/other.py#L245
    with some changes
    """
    old_hook_cls = getattr(accelerate.hooks, old_hook.__class__.__name__)
    old_hook_attr = old_hook.__dict__
    filtered_old_hook_attr = {}
    old_hook_init_signature = inspect.signature(old_hook_cls.__init__)
    for k in old_hook_attr:
        if k in old_hook_init_signature.parameters:
            filtered_old_hook_attr[k] = old_hook_attr[k]
    new_hook = old_hook_cls(**filtered_old_hook_attr)
    return new_hook


def _dequantize_and_replace(
    model,
    dtype,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
):
    """
    Converts a quantized model into its dequantized original version. The newly converted model will have
    some performance drop compared to the original model before quantization - use it only for specific usecases
    such as QLoRA adapters merging.

    Returns the converted model and a boolean that indicates if the conversion has been successful or not.
    """
    quant_method = quantization_config.quantization_method()

    target_cls = bnb.nn.Linear8bitLt if quant_method == "llm_int8" else bnb.nn.Linear4bit

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, target_cls) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)

            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                bias = getattr(module, "bias", None)

                device = module.weight.device
                with init_empty_weights():
                    new_module = torch.nn.Linear(module.in_features, module.out_features, bias=bias is not None)

                if quant_method == "llm_int8":
                    state = module.state
                else:
                    state = None

                new_module.weight = torch.nn.Parameter(dequantize_bnb_weight(module.weight, dtype, state))

                if bias is not None:
                    new_module.bias = bias

                # Create a new hook and attach it in case we use accelerate
                if hasattr(module, "_hf_hook"):
                    old_hook = module._hf_hook
                    new_hook = _create_accelerate_new_hook(old_hook)

                    remove_hook_from_module(module)
                    add_hook_to_module(new_module, new_hook)

                new_module.to(device)
                model._modules[name] = new_module
                has_been_replaced = True
        if len(list(module.children())) > 0:
            _, has_been_replaced = _dequantize_and_replace(
                module,
                dtype,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def dequantize_and_replace(
    model,
    modules_to_not_convert=None,
    quantization_config=None,
):
    model, has_been_replaced = _dequantize_and_replace(
        model,
        model.dtype,
        modules_to_not_convert=modules_to_not_convert,
        quantization_config=quantization_config,
    )

    if not has_been_replaced:
        logger.warning(
            "For some reason the model has not been properly dequantized. You might see unexpected behavior."
        )

    return model


def validate_bnb_backend_availability(raise_exception=False):
    """
    Validates if the available devices are supported by bitsandbytes, optionally raising an exception if not.
    """
    import bitsandbytes as bnb

    bnb_supported_devices = getattr(bnb, "supported_torch_devices", set())
    available_devices = set(get_available_devices())

    if not available_devices.intersection(bnb_supported_devices):
        if raise_exception:
            err_msg = (
                f"None of the available devices `available_devices = {available_devices or None}` are supported by the bitsandbytes version you have installed: `bnb_supported_devices = {bnb_supported_devices}`. "
                "Please check the docs to see if the backend you intend to use is available and how to install it: https://huggingface.co/docs/bitsandbytes/main/en/installation"
            )
            raise RuntimeError(err_msg)

        logger.warning("No supported devices found for bitsandbytes")
        return False
    return True
