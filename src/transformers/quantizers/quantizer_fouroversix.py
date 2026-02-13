from typing import TYPE_CHECKING

from .base import HfQuantizer
from .quantizers_utils import get_module_from_name
from ..utils.import_utils import is_fouroversix_available

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import (
    is_torch_available,
)

if is_torch_available():
    import torch

    from ..core_model_loading import WeightConverter


class FourOverSixHfQuantizer(HfQuantizer):
    """
    FP4 quantization with fouroversix.
    """

    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

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
        if self.param_needs_quantization(model, param_name):
            # 4-bit quantization
            return 0.5

        return super().param_element_size(model, param_name, param)

    def param_needs_quantization(
        self,
        model: "PreTrainedModel",
        param_name: str,
        **kwargs,
    ) -> bool:
        from fouroversix import FourOverSixLinear

        module, name = get_module_from_name(model, param_name)
        return isinstance(module, FourOverSixLinear) and name != "bias"

    def adjust_max_memory(
        self, max_memory: dict[str, int | str]
    ) -> dict[str, int | str]:
        # need more space for buffers that are created during quantization
        max_memory = {key: val * 0.9 for key, val in max_memory.items()}
        return max_memory

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        **kwargs,
    ):
        from fouroversix import quantize_model

        from ..integrations.fouroversix import adapt_fouroversix_config

        quantize_model(
            model,
            adapt_fouroversix_config(self.quantization_config),
        )

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return self.quantization_config.keep_master_weights

    def get_quantize_ops(self):
        from ..integrations.fouroversix import FourOverSixQuantize

        return FourOverSixQuantize(self)

    def get_weight_conversions(self):
        from ..integrations.fouroversix import FourOverSixDeserialize

        if self.pre_quantized:
            return [
                WeightConverter(
                    source_patterns=["quantized_weight_values"],
                    target_patterns="quantized_weight_values",
                    operations=[FourOverSixDeserialize(self)],
                )
            ]

        return []
