from typing import TYPE_CHECKING

from ..utils import is_accelerate_available, is_torch_available, is_torch_xpu_available, logging
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch
    from torch import nn

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import FineGrainedFP8Config

logger = logging.get_logger(__name__)


class FineGrainedFP8HfQuantizer(HfQuantizer):
    """
    FP8 quantization implementation supporting both standard and MoE models.
    Supports both e4m3fn formats based on platform.
    """

    requires_calibration = False
    quantization_config: "FineGrainedFP8Config"

    @property
    def quantized_element_size(self) -> float:
        return 1

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Loading an FP8 quantized model requires accelerate (`pip install accelerate`)")

        if self.quantization_config.dequantize:
            return

        if not torch.cuda.is_available() and not is_torch_xpu_available():
            if self.pre_quantized:
                logger.warning_once(
                    "Using FP8 quantized models requires a GPU or XPU, we will default to dequantizing the model to bf16 since no GPU or XPU is available"
                )
                self.quantization_config.dequantize = True
                return
            else:
                raise RuntimeError("No GPU or XPU found. A GPU or XPU is needed for FP8 quantization.")

        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            major, minor = compute_capability
            if (major < 8) or (major == 8 and minor < 9):
                logger.warning_once(
                    "FP8 quantized models is only supported on GPUs with compute capability >= 8.9 (e.g 4090/H100)"
                    f", actual = `{major}.{minor}`. We will default to dequantizing the model to bf16. Feel free "
                    f"to use a different quantization method like bitsandbytes or torchao"
                )
                self.quantization_config.dequantize = True
                return

        device_map = kwargs.get("device_map")
        if device_map is None:
            logger.warning_once(
                "You have loaded an FP8 model on CPU and have a CUDA or XPU device available, make sure to set "
                "your model on a GPU or XPU device in order to run your model. To remove this warning, "
                "pass device_map = 'cuda' or 'xpu'. "
            )
        elif isinstance(device_map, dict):
            if (
                not self.pre_quantized
                and len(device_map) > 1
                and ("cpu" in device_map.values() or "disk" in device_map.values())
            ):
                raise ValueError(
                    "You are attempting to load an FP8 model with a device_map that contains a cpu/disk device."
                    "This is not supported when the model is quantized on the fly. "
                    "Please use a quantized checkpoint or remove the cpu/disk device from the device_map."
                )

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        # `FP8GroupedLinear` is a subclass of `FP8Linear`, so the tuple covers it implicitly.
        from ..integrations.finegrained_fp8 import FP8Experts, FP8Linear

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, (FP8Linear, FP8Experts)):
            if self.pre_quantized or tensor_name == "bias":
                return False
            else:
                return True
        return False

    def _normalize_modules_to_not_convert(self, model: "PreTrainedModel"):
        """Rewrite the skip-list to the model's own module tree.
        For models that were already released, if they have a list of modules to not quantize
        we need to apply the weight renaming / weight conversion opérations to get the actual
        layer name of the model in `transformers`.
        """
        skip = self.quantization_config.modules_to_not_convert
        if not skip:
            return

        from ..conversion_mapping import get_model_conversion_mapping

        renamings = get_model_conversion_mapping(model)
        remapped = []
        for name in skip:
            renamed = name
            for rename in renamings:
                renamed, _ = rename.rename_source_key(renamed)
            remapped.append(renamed)
        self.quantization_config.modules_to_not_convert = remapped

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        self._normalize_modules_to_not_convert(model)
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )
        self.text_config = model.config.get_text_config()

    def quantized_module(self, name, module):
        from ..integrations.finegrained_fp8 import (
            ALL_FP8_EXPERTS_FUNCTIONS,
            FP8Embedding,
            FP8Experts,
            FP8GroupedLinear,
            FP8Linear,
        )
        from ..integrations.moe import use_experts_implementation

        config = self.quantization_config
        if config.dequantize:
            return None
        kwargs = {
            "block_size": config.weight_block_size,
            "activation_scheme": config.activation_scheme,
            "scale_fmt": config.scale_fmt,
        }
        modules_to_convert = config.modules_to_convert if self.pre_quantized else None
        if modules_to_convert and type(module) is nn.Embedding and name.endswith(tuple(modules_to_convert)):
            new_module = FP8Embedding(module.num_embeddings, module.embedding_dim, module.padding_idx)
        elif name.endswith(".experts"):
            has_gate, has_bias = getattr(module, "has_gate", True), getattr(module, "has_bias", False)
            experts_class = use_experts_implementation(
                experts_class=FP8Experts,
                experts_interface=ALL_FP8_EXPERTS_FUNCTIONS,
                has_bias=has_bias,
                has_gate=has_gate,
            )
            experts_config = getattr(module, "config", self.text_config)
            new_module = experts_class(experts_config, has_bias=has_bias, has_gate=has_gate, **kwargs)
        elif type(module) is nn.Linear:
            new_module = FP8Linear(module.in_features, module.out_features, has_bias=module.bias is not None, **kwargs)
        elif isinstance(module, nn.Linear) and "GroupedLinear" in type(module).__name__:
            new_module = FP8GroupedLinear(
                module.in_features, module.out_features, module.n_groups, has_bias=module.bias is not None, **kwargs
            )
        else:
            return None
        new_module._hf_quantized_needs_local_tp = True
        return new_module

    def update_parallel_plan(self, model, plan):
        from ..distributed.parallel_utils import _get_parameter_plan, replace_layer_number_by_wildcard
        from ..integrations.finegrained_fp8 import FP8Experts

        super().update_parallel_plan(model, plan)
        # Per-tensor scales (`activation_scale`, `weight_scale_inv` without blocks) are scalars, not sharded
        scalars = {replace_layer_number_by_wildcard(n) for n, p in model.named_parameters() if p.ndim == 0}
        plan.update({name: "replicate" for name in scalars if _get_parameter_plan(name, plan) is not None})
        overwrite = FP8Experts._impl_tp_layer_overrides.get(getattr(model.config, "_experts_implementation", None), {})
        plan.update({key: overwrite.get(style, style) for key, style in plan.items()})

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return True

    @property
    def is_compileable(self) -> bool:
        return True

    def get_quantize_ops(self):
        from ..integrations.finegrained_fp8 import Fp8Quantize

        return Fp8Quantize(self)

    # `weight` first: its converter takes `*.down_proj.weight` before the experts' own-name `down_proj` does
    serialized_names = {
        "weight": ["weight", "weight_scale_inv", "activation_scale"],
        "gate_up_proj": ["gate_up_proj", "gate_up_proj_scale_inv"],
        "down_proj": ["down_proj", "down_proj_scale_inv"],
    }

    def get_load_operations(self):
        from ..integrations.finegrained_fp8 import Fp8Dequantize, Fp8Deserialize

        return {"deserialize": Fp8Deserialize, "dequantize": Fp8Dequantize}

    def get_weight_conversions(self):
        # Quantized, the experts load through the model's converters: only the linears need ours
        conversions = super().get_weight_conversions()
        if self.quantization_config.dequantize:
            return conversions
        return [c for c in conversions if c._original_target_patterns == ["weight"]]

    # DeepSeek-V4-Flash's per-block `.scale`, Mistral Large 4's `.qscale_weight`, ModelOpt's names (and FP8 embeddings')
    renamed_names = {
        "scale": "weight_scale_inv",
        "qscale_weight": "weight_scale_inv",
        "weight_scale": "weight_scale_inv",
        "input_scale": "activation_scale",
    }
