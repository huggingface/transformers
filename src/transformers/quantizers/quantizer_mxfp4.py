# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..core_model_loading import WeightTransform
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import Mxfp4Config

from ..utils import (
    is_accelerate_available,
    is_kernels_available,
    is_torch_available,
    is_triton_available,
    logging,
)
from ..utils.import_utils import KERNELS_MAX_VERSION, KERNELS_MIN_VERSION
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch

    from ..core_model_loading import WeightConverter

logger = logging.get_logger(__name__)
triton_kernels_hub = None


class Mxfp4HfQuantizer(HfQuantizer):
    """
    FP4 quantization using fbgemm kernels
    """

    requires_calibration = False
    quantization_config: "Mxfp4Config"

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.triton_kernels_hub = None
        # This tracks the experts modules which projections run packed. Set after inspecting the model.
        self.packed_experts_modules: list[str] = []

    def _lazy_import_kernels(self):
        """Lazy import and initialize kernels only when needed"""
        if self.triton_kernels_hub is None:
            try:
                from ..integrations.hub_kernels import get_kernel

                self.triton_kernels_hub = get_kernel("kernels-community/gpt-oss-triton-kernels", version=1)
            except ImportError:
                raise ImportError("kernels package is required for MXFP4 quantization")
        return self.triton_kernels_hub

    def validate_environment(self, *args, **kwargs):
        if not is_torch_available():
            raise ImportError(
                "Using mxfp4 quantization requires torch"
                "Please install the latest version of torch ( pip install --upgrade torch )"
            )

        if self.quantization_config.dequantize:
            return

        if not is_accelerate_available():
            raise ImportError("Using mxfp4 requires Accelerate: `pip install accelerate`")

        device = torch.accelerator.current_accelerator() or torch.device("cpu")
        if device.type not in ["cuda", "xpu", "cpu"]:
            if self.pre_quantized:
                logger.warning_once(
                    f"Using MXFP4 quantized models requires model on cuda/xpu/cpu, but found {device}, we will default to dequantizing the model to bf16. To use mxfp4, please disable the current accelerator."
                )
                self.quantization_config.dequantize = True
                return
            else:
                raise RuntimeError(
                    f"Quantizing a model using MXFP4 requires model on cuda/xpu/cpu, but found {device}. To use mxfp4, please disable the current accelerator."
                )

        if torch.xpu.is_available():
            is_device_supported_mxfp4 = True
            triton_available = is_triton_available("3.5.0")
            kernels_installed = is_kernels_available()
        elif torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            is_device_supported_mxfp4 = compute_capability >= (7, 5)
            triton_available = is_triton_available("3.4.0")
            kernels_installed = is_kernels_available()
        elif device.type == "cpu":
            is_device_supported_mxfp4 = True
            triton_available = is_triton_available("3.5.0")
            kernels_installed = is_kernels_available()
        else:
            is_device_supported_mxfp4 = False
            triton_available = False
            kernels_installed = False

        if self.pre_quantized:
            if not is_device_supported_mxfp4:
                logger.warning_once(
                    "MXFP4 quantization is only supported on GPUs with compute capability >= 7.5 "
                    "(e.g T4, A100, L4, H100, or B200) or XPUs (e.g Intel® Data Center GPU Max Series). "
                    "We will default to dequantizing the model to bf16."
                )
                self.quantization_config.dequantize = True
                return

            if not triton_available:
                logger.warning_once(
                    "MXFP4 quantization requires Triton: CUDA requires Triton >= 3.4.0, "
                    "XPU/CPU requires Triton >= 3.5.0. Please install triton: `pip install triton`. "
                    "We will default to dequantizing the model to bf16."
                )
                self.quantization_config.dequantize = True
                return

            if not kernels_installed:
                logger.warning_once(
                    "MXFP4 quantization requires the `kernels` package: "
                    f"Please install a compatible version ({KERNELS_MIN_VERSION} <= version < {KERNELS_MAX_VERSION}), "
                    f"e.g. `pip install kernels=={KERNELS_MIN_VERSION}`"
                    "We will default to dequantizing the model to bf16."
                )
                self.quantization_config.dequantize = True
                return
        elif not is_device_supported_mxfp4:
            raise ValueError(
                "MXFP4 quantization is only supported on GPUs with compute capability >= 7.5 "
                "(e.g T4, A100, L4, H100, or B200) or XPUs (e.g Intel® Data Center GPU Max Series) or CPU"
            )
        elif not triton_available:
            raise ValueError(
                "MXFP4 quantization requires Triton: CUDA requires Triton >= 3.4.0, "
                "XPU/CPU requires Triton >= 3.5.0. Please install triton: `pip install triton`"
            )
        elif not kernels_installed:
            raise ValueError("MXFP4 quantization requires the `kernels` package: `pip install kernels>=0.12.0`")

        if not self.pre_quantized:
            self._lazy_import_kernels()

        device_map = kwargs.get("device_map")
        if device_map is not None and isinstance(device_map, dict):
            if not self.pre_quantized and "disk" in device_map.values():
                raise ValueError(
                    "You are attempting to load an FP4 model with a device_map that contains a disk device."
                    "This is not supported when the model is quantized on the fly. "
                    "Please use a quantized checkpoint or remove the disk device from the device_map."
                )

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        from ..integrations import Mxfp4GptOssExperts

        # GPT-OSS has a legacy Mxfp4GptOssExperts that is tailored to the model, so treat the edge case first
        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, Mxfp4GptOssExperts):
            return tensor_name not in ["down_proj_bias", "gate_up_proj_bias"]

        # Every other MoE keeps its own experts module and uses `ExpertsInterface`. Only the expert projections are
        # quantized, biases and everything else stay dense.
        module_name = param_name.rpartition(".")[0]
        return module_name in self.packed_experts_modules and tensor_name in ("gate_up_proj", "up_proj", "down_proj")

    def param_element_size(self, model: "PreTrainedModel", param_name: str, param: "torch.Tensor") -> float:
        """Returns the bytes per weight value, used to size the caching-allocator warmup. Since generic MoE modules keep
        declaring their experts as dense parameters (only gpt-oss gets a module whose which are already blocks-shaped),
        we need to make sure that the warmup does not reserve for the the dense size for weights that load packed."""
        module_name, _, attribute = param_name.rpartition(".")
        # Only change the element size for pre-quantized packed experts modules
        if self.pre_quantized and module_name in self.packed_experts_modules:
            if attribute in ("gate_up_proj", "up_proj", "down_proj"):
                return 0.5 + 1 / 32  # half a byte per value plus one e8m0 scale per group of 32
        return param.element_size()

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        # clean cache due to triton ops
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.xpu.is_available():
            torch.xpu.empty_cache()

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        use_kernels: bool = False,
        **kwargs,
    ) -> None:
        """Performs several tasks related to mxfp4 before weight loading:
        - Turns off quantization if the device and `use_kernels` clash
        - If this is a gpt-oss model, replaces the module to use mxfp4 + gpt-oss specific modules
        - If not, sets the experts implementation to mxfp4 so the model can run with packed mxfp4 kernels
        """
        device = torch.accelerator.current_accelerator() or torch.device("cpu")
        device_is_cpu = device.type == "cpu"

        # The quantized model is compatible with the `kernels` package only if the device is CPU
        if use_kernels and not device_is_cpu:
            logger.warning_once(
                "You are using full precision kernels, we will dequantize the model to bf16. "
                "To use the quantized model with quantization kernels, please set use_kernels=False"
            )
            self.quantization_config.dequantize = True
        # Conversly, the quantized model cannot work on CPU without the `kernels` package
        if not use_kernels and device_is_cpu:
            logger.warning_once(
                "MXFP4 inference on CPU requires use_kernels=True, but use_kernels is disabled. "
                "We will dequantize the model to bf16. To run MXFP4 natively on CPU, please set use_kernels=True."
            )
            self.quantization_config.dequantize = True

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )

        from ..integrations import replace_with_mxfp4_linear

        # This replacement only affects gpt-oss models
        model = replace_with_mxfp4_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        # This is the general case replacement, which acts in place
        self._setup_packed_experts(model)

    def _setup_packed_experts(self, model: "PreTrainedModel") -> None:
        """Route the model's fused-experts modules through the packed mxfp4 runtime.

        gpt-oss models were already given `Mxfp4GptOssExperts` by `replace_with_mxfp4_linear` (its fused
        swiglu epilogue is specific to that architecture); every other MoE keeps its own experts module
        and dispatches through `ExpertsInterface`, exactly like packed compressed-tensors checkpoints do.
        The kernel-ready weights are attached at load time, by `Mxfp4Quantize` when quantizing on the fly
        and by `Mxfp4Deserialize` when the checkpoint already holds `blocks` / `scales`.
        """
        if self.quantization_config.dequantize:
            return

        from ..integrations import Mxfp4GptOssExperts
        from .quantizers_utils import is_packed_experts_module, should_convert_module

        has_gpt_oss_modules = False
        experts_modules = []
        for name, module in model.named_modules():
            # keep track of gpt-oss modules to avoid warning if this is a gpt-oss model
            if isinstance(module, Mxfp4GptOssExperts):
                has_gpt_oss_modules = True
            # find all the experts modules that support packed mxfp4 at runtime
            elif (
                name.endswith(".experts")
                and is_packed_experts_module(module)
                and should_convert_module(name, self.modules_to_not_convert)
            ):
                experts_modules.append(name)

        # Stop if no experts modules were found. This is expected for non-gpt-oss models, throw a warning otherwise
        if not experts_modules:
            if not has_gpt_oss_modules:
                logger.warning_once(
                    "You are loading your model in mxfp4 but no fused MoE experts modules were found: nothing "
                    "will be quantized. Please double check your model architecture, or submit an issue on "
                    "github if you think this is a bug."
                )
            return None

        # Try setting the experts implementation to mxfp4 and dequantize the model if that fails
        model.set_experts_implementation("mxfp4")
        failed_to_switch_implem = any(
            model.get_submodule(name).config._experts_implementation != "mxfp4"  # type: ignore
            for name in experts_modules
        )
        if failed_to_switch_implem:
            logger.warning_once(
                f"{model.__class__.__name__} cannot dispatch its MoE experts to the mxfp4 kernels, so its "
                "experts will not be quantized and stay in the model dtype."
            )
            if self.pre_quantized:
                self.quantization_config.dequantize = True
            return None

        # If successful, we can successfully route the experts modules to the mxfp4 kernels, and we keep track of those
        self.packed_experts_modules = experts_modules

    def update_tp_plan(self, config):
        if "GptOssConfig" in config.__class__.__name__:
            if getattr(config, "base_model_tp_plan", None) is not None:
                config.base_model_tp_plan.update(
                    {
                        "layers.*.mlp.experts.gate_up_proj_blocks": "grouped_gemm",
                        "layers.*.mlp.experts.gate_up_proj_scales": "grouped_gemm",
                        "layers.*.mlp.experts.down_proj_blocks": "grouped_gemm",
                        "layers.*.mlp.experts.down_proj_scales": "grouped_gemm",
                    }
                )
        return config

    def update_ep_plan(self, config):
        if "GptOssConfig" in config.__class__.__name__:
            if getattr(config, "base_model_ep_plan", None) is not None:
                config.base_model_ep_plan.update(
                    {
                        "layers.*.mlp.experts.gate_up_proj_blocks": "grouped_gemm",
                        "layers.*.mlp.experts.gate_up_proj_scales": "grouped_gemm",
                        "layers.*.mlp.experts.down_proj_blocks": "grouped_gemm",
                        "layers.*.mlp.experts.down_proj_scales": "grouped_gemm",
                    }
                )
        return config

    def get_state_dict_and_metadata(self, model: "PreTrainedModel") -> tuple[dict[str, "torch.Tensor"], dict]:
        from ..integrations.mxfp4 import make_packed_mxfp4_proj, unswizzle_mxfp4_proj

        state_dict = model.state_dict()
        for name, module in model.named_modules():
            for proj in ("gate_up_proj", "up_proj", "down_proj"):
                if make_packed_mxfp4_proj(module, proj)[1] is None:
                    continue
                blocks, scales = unswizzle_mxfp4_proj(module, proj)
                state_dict[f"{name}.{proj}_blocks"] = blocks
                state_dict[f"{name}.{proj}_scales"] = scales
                # The registered parameters hold the swizzled bytes, whose physical layout is
                # hardware-specific: checkpoints keep the canonical blocks/scales instead.
                state_dict.pop(f"{name}.{proj}_swizzled", None)
                state_dict.pop(f"{name}.{proj}_swizzled_scales", None)

        metadata = {}
        return state_dict, metadata

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        logger.warning_once(
            "MXFP4 quantization don't support training, please consider dequantizing the model first by passing quantization_config=Mxfp4Config(dequantize=True) to .from_pretrained()"
        )
        return False

    def get_quantize_ops(self):
        from ..integrations.mxfp4 import Mxfp4Quantize

        return Mxfp4Quantize(self)

    def update_weight_conversions(self, weight_conversions: list["WeightTransform"]) -> list["WeightTransform"]:
        """Updates the weight conversions for the experts modules.
        If the model is pre-quantized and it is being dequantized, this only adds the dequantize op at the start of the
        chain. If the model is running quantized, and is not a gpt-oss model, this replaces the experts weight
        conversion from the normal list of ops (usually merge + concatenate) to the packed mxfp4 ops, which applies the
        normal list of ops to packed weights and scales separately."""
        from ..integrations.mxfp4 import DequantizeMxfp4Experts, LoadPackedMxfp4Experts

        updated: list = []
        for conv in weight_conversions:
            # Only update the experts weight conversions
            is_experts_conv = all("experts" in p for p in conv.source_patterns)
            if not self.pre_quantized or not isinstance(conv, WeightConverter) or not is_experts_conv:
                updated.append(conv)
                continue

            # Extract the weight sources and other sources
            weight_sources, other_sources = [], []
            for p in conv.source_patterns:
                if p.endswith(".weight"):
                    weight_sources.append(p)
                else:
                    other_sources.append(p)
            # Infer the new sources, which differentiate between blocks and scales
            new_sources = (
                [p + "_blocks$" for p in weight_sources] + [p + "_scales$" for p in weight_sources] + other_sources
            )
            # Now update the weight conversions, only if weight sources were found
            if weight_sources:
                if self.quantization_config.dequantize:
                    new_ops = [DequantizeMxfp4Experts(self)] + list(conv.operations)
                else:
                    new_ops = [LoadPackedMxfp4Experts(self, operations=list(conv.operations))]
                conv = WeightConverter(
                    source_patterns=new_sources,
                    target_patterns=conv._original_target_patterns,
                    operations=new_ops,
                )
            updated.append(conv)

        # Adds quantizer weight conversions, like the ones that handle gpt-oss models
        updated.extend(self.get_weight_conversions())
        return updated

    def get_weight_conversions(self):
        from ..integrations.mxfp4 import Mxfp4Dequantize, Mxfp4Deserialize

        operation = Mxfp4Dequantize if self.pre_quantized and self.quantization_config.dequantize else Mxfp4Deserialize
        # Non-gated MoEs (those with `has_gate = False`) have a `up_proj` projection module, which we do not want to
        # match to "gate_up_proj", hence the negative lookbehind "(?<!gate_)"
        projections = (
            ("gate_up_proj", "gate_up_proj"),
            (r"(?<!gate_)up_proj", "up_proj"),
            ("down_proj", "down_proj"),
        )
        return [
            WeightConverter(
                source_patterns=[f"{source}_blocks", f"{source}_scales"],
                target_patterns=rf"{target}$",
                operations=[operation(self)],
            )
            for source, target in projections
        ]
