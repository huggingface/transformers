import torch

from ..quantizers.quantizers_utils import get_module_from_name
from ..utils import is_fouroversix_available


if is_fouroversix_available():
    from fouroversix import ModelQuantizationConfig

from transformers.utils.quantization_config import FourOverSixConfig

from ..core_model_loading import ConversionOps


class FourOverSixQuantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        missing_keys: list[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        We need to store some parameters to create the quantized weight. For example, fouroversix
        requires 4 values that are stored in the checkpoint to recover the quantized weight. So we
        store them in a dict that is stored in hf_quantizer for now as we can't save it in the op
        since we create an op per tensor.
        """

        if self.hf_quantizer.quantization_config.keep_master_weights:
            return input_dict

        module, _ = get_module_from_name(model, full_layer_name)
        module_name = full_layer_name.rsplit(".", 1)[0]

        full_parameter_name = list(input_dict.keys())[0]
        parameter_name = full_parameter_name.replace(f"{module_name}.", "", 1)
        parameter = input_dict[full_parameter_name][0]
        quantized_parameters = module.get_quantized_parameters(parameter_name, parameter)

        # Delete the high-precision parameters from the module after we used them to create
        # the quantized parameters
        for parameter_name in module.parameters_to_quantize:
            delattr(module, parameter_name)

        # Remove these keys from the missing_keys list since we've deleted them from the model
        for key in input_dict:
            missing_keys.discard(key)

        return {
            f"{module_name}.{quantized_key}": quantized_parameters[quantized_key]
            for quantized_key in quantized_parameters
        }


def adapt_fouroversix_config(config: FourOverSixConfig):
    return ModelQuantizationConfig(
        activation_scale_rule=config.activation_scale_rule,
        dtype=config.dtype,
        gradient_scale_rule=config.gradient_scale_rule,
        keep_master_weights=config.keep_master_weights,
        matmul_backend=config.matmul_backend,
        output_dtype=config.output_dtype,
        quantize_backend=config.quantize_backend,
        scale_rule=config.scale_rule,
        weight_scale_2d=config.weight_scale_2d,
        weight_scale_rule=config.weight_scale_rule,
        modules_to_not_convert=config.modules_to_not_convert,
        module_config_overrides=config.module_config_overrides,
    )
