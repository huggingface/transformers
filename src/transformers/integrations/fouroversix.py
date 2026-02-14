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

        target_keys, value = list(input_dict.items())[0]
        value = value[0]
        module, _ = get_module_from_name(model, full_layer_name)

        module_name = full_layer_name.rsplit(".", 1)[0]
        missing_keys.discard(target_keys)
        quantized_params = module.get_quantized_parameters(value)

        return {
            f"{module_name}.{quantized_key}": quantized_params[quantized_key]
            for quantized_key in quantized_params
        }


class FourOverSixDeserialize(ConversionOps):
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
        module_name = full_layer_name.rsplit(".", 1)[0]
        missing_keys.discard(f"{module_name}.weight")
        return {full_layer_name: input_dict["quantized_weight_values"][0].data}


class FourOverSixDequantize(ConversionOps):
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
        module_name = full_layer_name.rsplit(".", 1)[0]
        module = model.get_submodule(module_name)
        return {f"{module_name}.weight": module.quantized_weight().dequantize()}


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
