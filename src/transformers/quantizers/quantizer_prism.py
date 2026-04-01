from typing import TYPE_CHECKING

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import PrismQuantConfig

from ..utils import is_accelerate_available, is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class PrismHfQuantizer(HfQuantizer):
    """
    Prism ML affine 1-bit quantization for pre-quantized checkpoints.

    Prism checkpoints store packed uint32 tensors alongside fp16 scales and
    biases for every group of 128 weights. The quantizer replaces linear and
    embedding modules with Prism-aware modules before state dict loading so the
    tensors can be loaded directly from the checkpoint.
    """

    requires_calibration = True
    quantization_config: "PrismQuantConfig"

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Loading a Prism quantized model requires accelerate (`pip install accelerate`)")

        if not torch.cuda.is_available():
            logger.warning_once(
                "You don't have a GPU available to load the model, Prism inference will work on CPU but be slow."
            )

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        if isinstance(dtype, torch.dtype):
            return dtype
        return torch.float16

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        from ..integrations import replace_with_prism_modules

        self.modules_to_not_convert = list(self.quantization_config.modules_to_not_convert or [])
        replace_with_prism_modules(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
        )

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return False
