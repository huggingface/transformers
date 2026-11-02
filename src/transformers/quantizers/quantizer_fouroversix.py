from typing import TYPE_CHECKING

from ..utils.import_utils import is_fouroversix_available
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import FourOverSixConfig

from ..utils import (
    is_torch_available,
)


if is_torch_available():
    import torch


class FourOverSixHfQuantizer(HfQuantizer):
    """
    FP4 quantization with fouroversix.
    """

    requires_calibration = False
    # The buffers of `FourOverSixLinear`, they replace its `weight` unless `keep_master_weights`
    serialized_names = {
        "weight": [
            "quantized_weight_values",
            "quantized_weight_scale_factors",
            "quantized_weight_amax",
            "quantized_weight_metadata",
        ]
    }
    quantization_config: "FourOverSixConfig"

    def validate_environment(self, *args, **kwargs):
        if not is_fouroversix_available():
            raise ImportError(
                "Using `fouroversix` requires fouroversix: `pip install fouroversix --no-build-isolation`"
            )

    def param_element_size(
        self,
        model: "PreTrainedModel",
        param_name: str,
        param: "torch.Tensor",
    ) -> float:
        from fouroversix import QuantizedModule

        module, tensor_name = get_module_from_name(model, param_name)

        if QuantizedModule.is_quantized_module_type(type(module)):
            return module.get_element_size(tensor_name)

        return super().param_element_size(model, param_name, param)

    def param_needs_quantization(
        self,
        model: "PreTrainedModel",
        param_name: str,
        **kwargs,
    ) -> bool:
        from fouroversix import QuantizedModule

        module, tensor_name = get_module_from_name(model, param_name)

        return QuantizedModule.is_quantized_module_type(type(module)) and tensor_name in module.parameters_to_quantize

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        **kwargs,
    ):
        if self.dequantizing:
            # The model stays dense: the `dequantize` load operation runs on every checkpoint weight
            return
        from fouroversix import QuantizedModule, quantize_model

        from ..integrations.fouroversix import adapt_fouroversix_config

        quantize_model(
            model,
            adapt_fouroversix_config(self.quantization_config),
        )

        # If the model has already been quantized, we need to delete the weight tensor here so that
        # it's not expected when parameters are loaded from the checkpoint.
        if self.pre_quantized and not self.quantization_config.keep_master_weights:
            for _, module in model.named_modules():
                if QuantizedModule.is_quantized_module_type(type(module)):
                    for parameter_name in module.parameters_to_quantize:
                        delattr(module, parameter_name)

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return self.quantization_config.keep_master_weights

    def get_load_operations(self):
        from ..integrations.fouroversix import FourOverSixDequantize, FourOverSixQuantize

        return {"dequantize": FourOverSixDequantize, "quantize": FourOverSixQuantize}

    def get_weight_conversions(self):
        # Pre-quantized checkpoints of other formats are converted by fouroversix itself
        model_config_type = getattr(self.quantization_config, "pre_quantized_model_config_type", None)
        if model_config_type is None or self.quantization_config.dequantize:
            return super().get_weight_conversions()
        from fouroversix import WeightConversions

        return WeightConversions.get_weight_conversions(model_config_type)
