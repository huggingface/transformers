import torch

from ..quantizers.quantizers_utils import get_module_from_name
from ..utils import is_fouroversix_available

if is_fouroversix_available():
    from fouroversix import ModelQuantizationConfig, quantize_to_fp4

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
        we need to store some parameters to create the quantized weight. For example, fouroversix requires 6 values that are stored in the checkpoint to recover the quantized weight. So we store them in a dict that it stored in hf_quantizer for now as we can't save it in the op since we create an op per tensor.
        """
        value = list(input_dict.values())[0]
        value = value[0]
        module, _ = get_module_from_name(model, full_layer_name)
        module = module.to_empty(device=value.device)
        module.weight = torch.nn.Parameter(value, requires_grad=False)

        layer_name = full_layer_name.rsplit(".", 1)[0]
        missing_keys.discard(f"{layer_name}.weight")

        if module.weight.device.type != "meta":
            module.apply_ptq()

        return {
            f"{layer_name}.quantized_weight_amax": module.quantized_weight_amax,
            f"{layer_name}.quantized_weight_scale_factors": module.quantized_weight_scale_factors,
            f"{layer_name}.quantized_weight_values": module.quantized_weight_values,
            f"{layer_name}.quantized_weight_metadata": module.quantized_weight_metadata,
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
        layer_name = full_layer_name.rsplit(".", 1)[0]
        layer = model.get_submodule(layer_name)

        # Delete the randomly initialized high-precision weight before loading the quantized weight
        del layer.weight
        missing_keys.discard(f"{layer_name}.weight")

        return {full_layer_name: input_dict["quantized_weight_values"][0].data}


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
        exclude_layers=config.exclude_layers,
        layer_config_overrides=config.layer_config_overrides,
    )
