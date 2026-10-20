# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ..utils import is_torch_available, logging
from ..utils.quantization_config import QuantizationConfigMixin, QuantizationMethod
from .quantizers_utils import checkpoint_module_namer, get_module_from_name, should_convert_module


if TYPE_CHECKING:
    from torch.nn import ModuleList

    from ..modeling_utils import PreTrainedModel

if is_torch_available():
    import torch

    if not TYPE_CHECKING:
        from torch.nn import ModuleList
else:
    ModuleList = str

logger = logging.get_logger(__file__)


def get_keys_to_not_convert(model) -> list:
    r"""
    Function to automatically detect keys to not convert for usage like quantization. For example for CausalLM modules
    we may want to keep the lm_head in full precision for numerical stability reasons.
    """
    # remove tied weights
    tied_keys = set()
    if len(model.all_tied_weights_keys) > 0:
        tied_keys = set(model.all_tied_weights_keys.values()) | set(model.all_tied_weights_keys.keys())

    # remove last module
    last_module_key = {list(model.named_parameters())[-1][0]}

    # remove output emb
    output_emb_module = model.get_output_embeddings()
    output_emb_keys = {
        name
        for name, module in model.named_modules()
        if output_emb_module is not None and id(module) == id(output_emb_module)
    }
    modules_to_not_convert = tied_keys | last_module_key | output_emb_keys

    modules_to_not_convert = list({k.removesuffix(".weight") for k in modules_to_not_convert})

    return list(modules_to_not_convert)


def _assign_is_quantized(model):
    from ..modeling_utils import PreTrainedModel

    for module in model.modules():
        if isinstance(module, PreTrainedModel):
            module.config._is_quantized = True


class HfQuantizer(ABC):
    """
    Abstract class of the HuggingFace quantizer. Supports for now quantizing HF transformers models for inference and/or quantization.
    This class is used only for transformers.PreTrainedModel.from_pretrained and cannot be easily used outside the scope of that method
    yet.

    Attributes
        quantization_config (`transformers.utils.quantization_config.QuantizationConfigMixin`):
            The quantization config that defines the quantization parameters of your model that you want to quantize.
        requires_calibration (`bool`):
            Whether the quantization method requires to calibrate the model before using it.
        dequantized_saves_unquantized (`bool`):
            Whether a model loaded with `dequantize=True` saves like an unquantized model. The save then uses only
            the model's own conversions, not the quantizer's. It is off by default, so the other quantizers keep
            their current save.
    """

    requires_calibration = False
    dequantized_saves_unquantized = False

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        self.quantization_config = quantization_config
        self.pre_quantized = kwargs.pop("pre_quantized", True)

        if not self.pre_quantized and self.requires_calibration:
            raise ValueError(
                f"The quantization method {quantization_config.quant_method} does require the model to be pre-quantized."
                f" You explicitly passed `pre_quantized=False` meaning your model weights are not quantized. Make sure to "
                f"pass `pre_quantized=True` while knowing what you are doing."
            )

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        """
        Some quantization methods require to explicitly set the dtype of the model to a
        target dtype. You need to override this method in case you want to make sure that behavior is
        preserved

        Args:
            dtype (`torch.dtype`):
                The input dtype that is passed in `from_pretrained`
        """
        return dtype

    def update_device_map(self, device_map: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Override this method if you want to pass a override the existing device map with a new
        one. E.g. for bitsandbytes, since `accelerate` is a hard requirement, if no device_map is
        passed, the device_map is set to `"auto"``

        Args:
            device_map (`Union[dict, str]`, *optional*):
                The device_map that is passed through the `from_pretrained` method.
        """
        return device_map

    @property
    def quantized_element_size(self) -> float | None:
        "Bytes per element of a quantized weight, `None` to use the param's own element size."
        return None

    def param_element_size(self, model: "PreTrainedModel", param_name: str, param: "torch.Tensor") -> float:
        if self.quantized_element_size is not None and self.param_needs_quantization(model, param_name):
            return self.quantized_element_size
        return param.element_size()

    def get_new_dtype(self, converter, param, dtype: "torch.dtype | None") -> "torch.dtype | None":
        """
        This happens at materialization time, when the tensor is sent from disk to device.
        `dtype` to cast the pre-quantized checkpoint tensor `param` in.
        If that param is not quantized (float type) we use the input `dtype` that comes from the initialized `nn.Module`.

        If the quantization method has a `Deserializer` or `Dequantizer` (load function) we return None so it's handled by them.
        """
        is_float = (
            param.get_dtype().startswith(("F", "BF")) if hasattr(param, "get_dtype") else param.is_floating_point()
        )
        is_load_input = any(getattr(op, "hf_quantizer", None) for op in getattr(converter, "operations", ()))
        return dtype if is_float and not is_load_input else None

    def adjust_max_memory(self, max_memory: dict[str, int | str]) -> dict[str, int | str]:
        """adjust max_memory argument for infer_auto_device_map() if extra memory is needed for quantization"""
        return max_memory

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        """
        Check whether a given param needs to be quantized.
        """
        return False

    def validate_environment(self, *args, **kwargs):
        """
        This method is used to potentially check for potential conflicts with arguments that are
        passed in `from_pretrained`. You need to define it for all future quantizers that are integrated with transformers.
        If no explicit check are needed, simply return nothing.
        """
        return

    def update_parallel_plan(self, model, plan: dict[str, str]):
        """
        Updates in place `model.tp_plan` or `model.ep_plan` (the full plans, merged from the config and all
        sub-configs) to take into account the quantizer's custom keys like 'inv_scale'"""
        from ..distributed.tensor_parallel import replace_layer_number_by_wildcard

        params = {replace_layer_number_by_wildcard(name) for name, _ in model.named_parameters()}
        suffixes = {n.removeprefix(p) for p, names in self.serialized_names.items() for n in names}
        extra = {key + suffix: style for key, style in plan.items() for suffix in suffixes}
        plan.update({key: style for key, style in extra.items() if key in params and key not in plan})

    def update_attn_implementation(self, config):
        """Sees the requested `attn_implementation` before the model is built, so a quantizer can default it."""
        return config

    def _process_model_before_weight_loading(self, model, **kwargs):
        return model

    def preprocess_model(self, model: "PreTrainedModel", dtype=None, **kwargs):
        """
        Setting model attributes and/or converting model before weights loading. At this point
        the model should be initialized on the meta device so you can freely manipulate the skeleton
        of the model in order to replace modules in-place. Make sure to override the abstract method `_process_model_before_weight_loading`.

        Args:
            model (`~transformers.PreTrainedModel`):
                The model to quantize
            kwargs (`dict`, *optional*):
                The keyword arguments that are passed along `_process_model_before_weight_loading`.
        """
        setattr(model, "is_quantized", True)
        setattr(model, "quantization_method", self.quantization_config.quant_method)
        if self.pre_quantized:
            self._convert_model_for_quantization(model)
        self._process_model_before_weight_loading(model, **kwargs)
        self.update_parallel_plan(model, model.tp_plan)
        self.update_parallel_plan(model, model.ep_plan)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    def postprocess_model(self, model: "PreTrainedModel", **kwargs):
        """
        Post-process the model post weights loading.
        Make sure to override the abstract method `_process_model_after_weight_loading`.

        Args:
            model (`~transformers.PreTrainedModel`):
                The model to quantize
            kwargs (`dict`, *optional*):
                The keyword arguments that are passed along `_process_model_after_weight_loading`.
        """
        model.config.quantization_config = self.quantization_config

        if self.pre_quantized and getattr(self.quantization_config, "dequantize", False):
            self.remove_quantization_config(model)
            if self.dequantized_saves_unquantized and getattr(model, "_weight_conversions", None) is not None:
                # The loader kept the transforms that fired. Swap each for the model transform it derives from.
                fired = (conversion._unquantized_transform for conversion in model._weight_conversions)
                model._weight_conversions = list(
                    {id(transform): transform for transform in fired if transform is not None}.values()
                )
        else:
            _assign_is_quantized(model)

        return self._process_model_after_weight_loading(model, **kwargs)

    def remove_quantization_config(self, model):
        """
        Remove the quantization config from the model.
        """
        if hasattr(model, "hf_quantizer"):
            del model.hf_quantizer
        if hasattr(model.config, "quantization_config"):
            del model.config.quantization_config
        if hasattr(model, "quantization_method"):
            del model.quantization_method
        model.is_quantized = False

    def dequantize(self, model, dtype=None):
        """
        Potentially dequantize the model to retrieve the original model, with some loss in accuracy / performance.
        Note not all quantization schemes support this.
        """
        if dtype is None:
            # using the same dtype we used to load the model. If we don't do that, we might have issues with modules we didn't quantize.
            # or we need to upcast everything to the same dtype
            dtype = model.config.dtype
        model = self._dequantize(model, dtype=dtype)
        self.remove_quantization_config(model)

        return model

    def _dequantize(self, model, dtype=None):
        raise NotImplementedError(
            f"{self.quantization_config.quant_method} has no implementation of `dequantize`, please raise an issue on GitHub."
        )

    @staticmethod
    def get_modules_to_not_convert(
        model: "PreTrainedModel",
        skip_modules: list[str] | None = None,
        keep_in_fp32_modules: list[str] | None = None,
        add_default_skips: bool = False,
    ):
        if skip_modules is None or add_default_skips:
            modules_to_not_convert = get_keys_to_not_convert(model)
        else:
            modules_to_not_convert = []

        if skip_modules:
            modules_to_not_convert.extend(skip_modules)
            # The skip-list uses checkpoint names, e.g. a VLM's `language_model.model.*` that loads as
            # `model.language_model.*`. So also add the modules it covers under their model names.
            to_checkpoint = checkpoint_module_namer(model)
            renamed = []
            for module_name, _ in model.named_modules():
                if any(module_name.startswith(f"{parent}.") for parent in renamed):
                    continue
                checkpoint_name = to_checkpoint(module_name)
                if checkpoint_name != module_name and not should_convert_module(checkpoint_name, skip_modules):
                    renamed.append(module_name)
            modules_to_not_convert.extend(renamed)

        if keep_in_fp32_modules is not None:
            modules_to_not_convert.extend(keep_in_fp32_modules)

        modules_to_not_convert = list(set(modules_to_not_convert))

        return modules_to_not_convert

    @property
    def is_qat_trainable(self) -> bool:
        """Flag indicating whether the quantized model can carry out quantization aware training"""
        return False

    @property
    def is_compileable(self) -> bool:
        """Flag indicating whether the quantized model can be compiled"""
        return False

    def get_state_dict_and_metadata(self, model):
        """Get state dict and metadata. Useful when we need to modify a bit the state dict due to quantization"""
        return None, {}

    @abstractmethod
    def is_serializable(self): ...

    @property
    @abstractmethod
    def is_trainable(self): ...

    def _convert_model_for_quantization(self, model):
        for name, module in model.named_modules():
            module_class_name = module.__class__.__name__
            if module_class_name in MODULES_TO_PATCH_FOR_QUANTIZATION and (
                self.quantization_config.quant_method
                in MODULES_TO_PATCH_FOR_QUANTIZATION[module_class_name]["quantization_methods"]
            ):
                with torch.device("meta"):
                    parent_module, name = get_module_from_name(model, name)
                    parent_module._modules[name] = MODULES_TO_PATCH_FOR_QUANTIZATION[module_class_name]["module_name"](
                        model.config.get_text_config()
                    )

    def get_quantize_ops(self):
        raise NotImplementedError(
            f"{self.quantization_config.quant_method} is not available yet and will be supported soon."
        )

    # Maps a non quantized parameter name to its corresponding quantized names
    # e.g. `{"weight": ["weight", "weight_scale_inv"]}` means a weight becomes weight + scale inv.
    serialized_names: dict[str, list[str]] = {}

    def get_load_operations(self) -> dict:
        """
        Op classes loading the serialized tensors, e.g. `{"deserialize": Mxfp4Deserialize, "dequantize": Mxfp4Dequantize}`.
        A deserialization builds the params of one module, so it only runs on the general converters. A dequantization
        also runs before the ops of the model's converters, e.g. before a per-expert merge.
        """
        return {}

    def get_weight_conversions(self, model=None):
        """Return the loading to perform on any parameter that is not matched by the model's converters.

        `model` is the model being loaded, with its modules already replaced.
        """
        from ..core_model_loading import WeightConverter

        mode = "dequantize" if getattr(self.quantization_config, "dequantize", False) else "deserialize"
        operation = self.get_load_operations().get(mode) if self.pre_quantized else None
        if operation is None:
            return []
        return [WeightConverter(names, param, [operation(self)]) for param, names in self.serialized_names.items()]

    def get_checkpoint_renamings(self):
        """Return renamings of the checkpoint keys, applied before the model's conversion mapping.

        Use it when a producer names a tensor its own way, e.g. a block scale saved as `scale`.
        """
        return []

    def update_weight_conversions(self, weight_conversions, model):
        """
        Makes the model's converters (from the conversion mapping) load the serialized names. The model's params
        decide, e.g. for a merge of `experts.*.gate_proj.weight` into `experts.gate_up_proj`:
        - `experts.gate_up_proj_scale_inv` is a param: a copy merges `experts.*.gate_proj.weight_scale_inv` into it
        - otherwise the converter also collects `experts.*.gate_proj.weight_scale_inv`, the dequantize op runs first
        """
        from ..core_model_loading import WeightConverter

        param_names = "\n".join(name for name, _ in model.named_parameters())
        dequantize = self.get_load_operations().get("dequantize") if self.pre_quantized else None
        updated = []
        for converter in weight_conversions:
            updated.append(converter)
            if not isinstance(converter, WeightConverter):
                continue
            sources, targets = converter._original_source_patterns, converter._original_target_patterns
            param = sources[0].rpartition(".")[2]
            serialized = self.serialized_names.get(param)
            if not serialized or not all(source.endswith(f".{param}") for source in sources):
                continue
            # `weight_scale_inv` -> `_scale_inv`
            suffixes = [name.removeprefix(param) for name in serialized if name.startswith(param) and name != param]
            if suffixes and re.search(f"{targets[0]}{suffixes[0]}$", param_names, re.MULTILINE):
                # The module keeps the companions as params: each one gets its own copy of the converter
                for suffix in suffixes:
                    suffixed_sources = [source + suffix for source in sources]
                    suffixed_targets = [target + suffix for target in targets]
                    updated.append(
                        converter.copy(suffixed_sources, suffixed_targets, operations=list(converter.operations))
                    )
            elif dequantize is not None:
                # The module only has `param`: the converter also collects the companions and dequantizes first
                all_sources = [re.sub(rf"{param}$", name, source) for source in sources for name in serialized]
                updated[-1] = converter.copy(
                    all_sources, targets, operations=[dequantize(self)] + converter.operations
                )
        return self.get_checkpoint_renamings() + updated + self.get_weight_conversions(model)


class SequentialLlama4TextExperts(ModuleList):
    """
    A module that implements a compressed version of a list of expert modules.
    This is specifically designed to work with Llama4TextExperts in MoE layers.
    """

    def __init__(self, config):
        from transformers.models.llama4.modeling_llama4 import Llama4TextMLP

        super().__init__([Llama4TextMLP(config) for _ in range(config.num_local_experts)])
        self.num_experts = config.num_local_experts

    def forward(
        self,
        hidden_states: "torch.Tensor",
    ) -> "torch.Tensor":
        hidden_states = hidden_states.reshape(self.num_experts, -1, hidden_states.shape[-1])
        routed_out = torch.zeros_like(hidden_states)
        for expert_idx in range(self.num_experts):
            routed_out[expert_idx] = self[expert_idx](hidden_states[expert_idx])
        return routed_out


MODULES_TO_PATCH_FOR_QUANTIZATION = {
    "Llama4TextExperts": {
        "module_name": SequentialLlama4TextExperts,
        "quantization_methods": [
            QuantizationMethod.COMPRESSED_TENSORS,
            QuantizationMethod.BITS_AND_BYTES,
        ],
    }
}
