# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
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
import warnings
from collections.abc import Callable
from itertools import chain
from typing import TypeGuard

from ..models.auto.configuration_auto import AutoConfig
from ..utils import logging
from ..utils.quantization_config import (
    AqlmConfig,
    AutoRoundConfig,
    AwqConfig,
    BitNetQuantConfig,
    BitsAndBytesConfig,
    CompressedTensorsConfig,
    EetqConfig,
    FbgemmFp8Config,
    FineGrainedFP8Config,
    FourOverSixConfig,
    FPQuantConfig,
    GemmaQuantizationConfig,
    GgufConfig,
    GPTQConfig,
    HiggsConfig,
    HqqConfig,
    MetalConfig,
    Mxfp4Config,
    NVFP4Config,
    QuantizationConfigMixin,
    QuantizationMethod,
    QuantoConfig,
    QuarkConfig,
    SinqConfig,
    SpQRConfig,
    TorchAoConfig,
    VptqConfig,
)
from .base import HfQuantizer, _assign_is_quantized
from .quantizer_aqlm import AqlmHfQuantizer
from .quantizer_auto_round import AutoRoundQuantizer
from .quantizer_awq import AwqQuantizer
from .quantizer_bitnet import BitNetHfQuantizer
from .quantizer_bnb_4bit import Bnb4BitHfQuantizer
from .quantizer_bnb_8bit import Bnb8BitHfQuantizer
from .quantizer_compressed_tensors import CompressedTensorsHfQuantizer
from .quantizer_eetq import EetqHfQuantizer
from .quantizer_fbgemm_fp8 import FbgemmFp8HfQuantizer
from .quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
from .quantizer_fouroversix import FourOverSixHfQuantizer
from .quantizer_fp_quant import FPQuantHfQuantizer
from .quantizer_gemma import GemmaQuantizer
from .quantizer_gguf import GgufHfQuantizer
from .quantizer_gptq import GptqHfQuantizer
from .quantizer_higgs import HiggsHfQuantizer
from .quantizer_hqq import HqqHfQuantizer
from .quantizer_metal import MetalHfQuantizer
from .quantizer_mxfp4 import Mxfp4HfQuantizer
from .quantizer_nvfp4 import NVFP4HfQuantizer
from .quantizer_quanto import QuantoHfQuantizer
from .quantizer_quark import QuarkHfQuantizer
from .quantizer_sinq import SinqHfQuantizer
from .quantizer_spqr import SpQRHfQuantizer
from .quantizer_torchao import TorchAoHfQuantizer
from .quantizer_vptq import VptqHfQuantizer


AUTO_QUANTIZER_MAPPING = {
    "awq": AwqQuantizer,
    "bitsandbytes_4bit": Bnb4BitHfQuantizer,
    "bitsandbytes_8bit": Bnb8BitHfQuantizer,
    "gptq": GptqHfQuantizer,
    "aqlm": AqlmHfQuantizer,
    "quanto": QuantoHfQuantizer,
    "quark": QuarkHfQuantizer,
    "fouroversix": FourOverSixHfQuantizer,
    "fp_quant": FPQuantHfQuantizer,
    "eetq": EetqHfQuantizer,
    "higgs": HiggsHfQuantizer,
    "hqq": HqqHfQuantizer,
    "compressed-tensors": CompressedTensorsHfQuantizer,
    "fbgemm_fp8": FbgemmFp8HfQuantizer,
    "torchao": TorchAoHfQuantizer,
    "bitnet": BitNetHfQuantizer,
    "vptq": VptqHfQuantizer,
    "spqr": SpQRHfQuantizer,
    "fp8": FineGrainedFP8HfQuantizer,
    "gguf": GgufHfQuantizer,
    "nvfp4": NVFP4HfQuantizer,
    # MXFP8 = FP8 (E4M3 weights) with per-block ``[1, 32]`` E8M0 (uint8) scales —
    # reuses the FineGrainedFP8 dequant path, with the E8M0 byte→exponent
    # unpacking handled inside ``Fp8Dequantize._dequantize_one``.
    "mxfp8": FineGrainedFP8HfQuantizer,
    "auto-round": AutoRoundQuantizer,
    "mxfp4": Mxfp4HfQuantizer,
    "metal": MetalHfQuantizer,
    "sinq": SinqHfQuantizer,
    "gemma": GemmaQuantizer,
}

AUTO_QUANTIZATION_CONFIG_MAPPING = {
    "gguf": GgufConfig,
    "awq": AwqConfig,
    "bitsandbytes_4bit": BitsAndBytesConfig,
    "bitsandbytes_8bit": BitsAndBytesConfig,
    "eetq": EetqConfig,
    "gptq": GPTQConfig,
    "aqlm": AqlmConfig,
    "quanto": QuantoConfig,
    "quark": QuarkConfig,
    "fouroversix": FourOverSixConfig,
    "fp_quant": FPQuantConfig,
    "hqq": HqqConfig,
    "compressed-tensors": CompressedTensorsConfig,
    "fbgemm_fp8": FbgemmFp8Config,
    "higgs": HiggsConfig,
    "torchao": TorchAoConfig,
    "bitnet": BitNetQuantConfig,
    "vptq": VptqConfig,
    "spqr": SpQRConfig,
    "fp8": FineGrainedFP8Config,
    "nvfp4": NVFP4Config,
    "mxfp8": FineGrainedFP8Config,
    "auto-round": AutoRoundConfig,
    "mxfp4": Mxfp4Config,
    "metal": MetalConfig,
    "sinq": SinqConfig,
    "gemma": GemmaQuantizationConfig,
}

LOADING_ATTRIBUTES_CONFIG_TYPES = (
    GPTQConfig,
    AwqConfig,
    AutoRoundConfig,
    FbgemmFp8Config,
    CompressedTensorsConfig,
    Mxfp4Config,
    MetalConfig,
    FineGrainedFP8Config,
    NVFP4Config,
)

logger = logging.get_logger(__name__)


class AutoQuantizationConfig:
    """
    The Auto-HF quantization config class that takes care of automatically dispatching to the correct
    quantization config given a quantization config stored in a dictionary.
    """

    @classmethod
    def from_dict(cls, quantization_config_dict: dict):
        quant_method = quantization_config_dict.get("quant_method")
        # We need a special care for bnb models to make sure everything is BC ..
        if quantization_config_dict.get("load_in_8bit", False) or quantization_config_dict.get("load_in_4bit", False):
            suffix = "_4bit" if quantization_config_dict.get("load_in_4bit", False) else "_8bit"
            quant_method = QuantizationMethod.BITS_AND_BYTES + suffix
        elif quant_method is None:
            raise ValueError(
                "The model's quantization config from the arguments has no `quant_method` attribute. Make sure that the model has been correctly quantized"
            )

        if quant_method not in AUTO_QUANTIZATION_CONFIG_MAPPING:
            raise ValueError(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}"
            )

        target_cls = AUTO_QUANTIZATION_CONFIG_MAPPING[quant_method]
        return target_cls.from_dict(quantization_config_dict)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if getattr(model_config, "quantization_config", None) is None:
            raise ValueError(
                f"Did not found a `quantization_config` in {pretrained_model_name_or_path}. Make sure that the model is correctly quantized."
            )
        quantization_config_dict = model_config.quantization_config
        quantization_config = cls.from_dict(quantization_config_dict)
        # Update with potential kwargs that are passed through from_pretrained.
        quantization_config.update(**kwargs)
        return quantization_config


class AutoHfQuantizer:
    """
     The Auto-HF quantizer class that takes care of automatically instantiating to the correct
    `HfQuantizer` given the `QuantizationConfig`.
    """

    @classmethod
    def from_config(cls, quantization_config: QuantizationConfigMixin | dict, **kwargs):
        # Convert it to a QuantizationConfig if the q_config is a dict
        if isinstance(quantization_config, dict):
            quantization_config = AutoQuantizationConfig.from_dict(quantization_config)

        quant_method = quantization_config.quant_method

        # Again, we need a special care for bnb as we have a single quantization config
        # class for both 4-bit and 8-bit quantization
        if quant_method == QuantizationMethod.BITS_AND_BYTES:
            if not isinstance(quantization_config, BitsAndBytesConfig):
                raise TypeError(
                    "Found `quant_method=bitsandbytes` but `quantization_config` is not a `BitsAndBytesConfig`."
                )
            if quantization_config.load_in_8bit:
                quant_method += "_8bit"
            else:
                quant_method += "_4bit"

        if quant_method not in AUTO_QUANTIZER_MAPPING:
            raise ValueError(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}"
            )

        target_cls = AUTO_QUANTIZER_MAPPING[quant_method]
        return target_cls(quantization_config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        quantization_config = AutoQuantizationConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls.from_config(quantization_config)

    @classmethod
    def merge_quantization_configs(
        cls,
        quantization_config: dict | QuantizationConfigMixin,
        quantization_config_from_args: QuantizationConfigMixin | None,
    ):
        """
        handles situations where both quantization_config from args and quantization_config from model config are present.
        """
        if quantization_config_from_args is not None:
            warning_msg = (
                "You passed `quantization_config` or equivalent parameters to `from_pretrained` but the model you're loading"
                " already has a `quantization_config` attribute. The `quantization_config` from the model will be used."
            )
        else:
            warning_msg = ""

        if isinstance(quantization_config, dict):
            # Convert the config based on the type of quantization_config_from_args (e.g., AutoRoundConfig), which takes priority before automatic configuration dispatch.
            if isinstance(quantization_config_from_args, AutoRoundConfig):
                quantization_config = AutoRoundConfig.from_dict(quantization_config)
            else:
                quantization_config = AutoQuantizationConfig.from_dict(quantization_config)

        if (
            quantization_config_from_args is not None
            and quantization_config.__class__.__name__ != quantization_config_from_args.__class__.__name__
        ):
            raise ValueError(
                f"The model is quantized with {quantization_config.__class__.__name__} but you are passing a {quantization_config_from_args.__class__.__name__} config. "
                "Please make sure to pass the same quantization config class to `from_pretrained` with different loading attributes."
            )

        if isinstance(quantization_config, LOADING_ATTRIBUTES_CONFIG_TYPES) and isinstance(
            quantization_config_from_args, LOADING_ATTRIBUTES_CONFIG_TYPES
        ):
            loading_attr_dict = quantization_config_from_args.get_loading_attributes()
            for attr, val in loading_attr_dict.items():
                setattr(quantization_config, attr, val)

            if loading_attr_dict:
                warning_msg += f"However, loading attributes (e.g. {list(loading_attr_dict.keys())}) will be overwritten with the one you passed to `from_pretrained`. The rest will be ignored."

        if warning_msg != "" and not isinstance(quantization_config, (Mxfp4Config, MetalConfig, FineGrainedFP8Config)):
            warnings.warn(warning_msg)
        else:
            # in the case of mxfp4, we don't want to print the warning message, bit confusing for users
            logger.info(warning_msg)
        return quantization_config

    @staticmethod
    def supports_quant_method(quantization_config_dict):
        quant_method = quantization_config_dict.get("quant_method", None)
        if quantization_config_dict.get("load_in_8bit", False) or quantization_config_dict.get("load_in_4bit", False):
            suffix = "_4bit" if quantization_config_dict.get("load_in_4bit", False) else "_8bit"
            quant_method = QuantizationMethod.BITS_AND_BYTES + suffix
        elif quant_method is None:
            raise ValueError(
                "The model's quantization config from the arguments has no `quant_method` attribute. Make sure that the model has been correctly quantized"
            )

        if quant_method not in AUTO_QUANTIZATION_CONFIG_MAPPING:
            logger.warning(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}. Hence, we will skip the quantization. "
                "To remove the warning, you can delete the quantization_config attribute in config.json"
            )
            return False
        return True


def register_quantization_config(method: str):
    """Register a custom quantization configuration."""

    def register_config_fn(cls):
        if method in AUTO_QUANTIZATION_CONFIG_MAPPING:
            raise ValueError(f"Config '{method}' already registered")

        if not issubclass(cls, QuantizationConfigMixin):
            raise TypeError("Config must extend QuantizationConfigMixin")

        AUTO_QUANTIZATION_CONFIG_MAPPING[method] = cls
        return cls

    return register_config_fn


def register_quantizer(name: str):
    """Register a custom quantizer."""

    def register_quantizer_fn(cls):
        if name in AUTO_QUANTIZER_MAPPING:
            raise ValueError(f"Quantizer '{name}' already registered")

        if not issubclass(cls, HfQuantizer):
            raise TypeError("Quantizer must extend HfQuantizer")

        AUTO_QUANTIZER_MAPPING[name] = cls
        return cls

    return register_quantizer_fn


def is_quantization_plan(quantization_config) -> TypeGuard[dict]:
    "A plan maps module names or glob patterns to the config quantizing the modules they match (`None`: unquantized)"
    return (
        isinstance(quantization_config, dict)
        and len(quantization_config) > 0
        and all(
            isinstance(config, (dict, QuantizationConfigMixin, type(None))) for config in quantization_config.values()
        )
    )


# ModelOpt `quant_algo` -> its config. W4A16 NVFP4 loads with the W4A4 config, which quantizes activations
MODELOPT_ALGOS = {
    "NVFP4": NVFP4Config,
    "W4A16_NVFP4": NVFP4Config,
    "FP8": lambda: FineGrainedFP8Config(activation_scheme="static", weight_block_size=None),
    "FP8_PB_WO": lambda: FineGrainedFP8Config(activation_scheme="none", weight_block_size=(128, 128)),
}


def modelopt_plan(config: dict) -> dict | None:
    """
    ModelOpt's per-layer `quantized_layers`. The layers it leaves out keep the base `quant_method` if it is one (e.g.
    `fp8`), else stay unquantized like its `ignore`d ones.
    """
    if "quantized_layers" not in config:
        return None
    base = {key: value for key, value in config.items() if key != "quantized_layers"}
    base = base if base.get("quant_method") in AUTO_QUANTIZATION_CONFIG_MAPPING else None
    plan = {} if base else dict.fromkeys(config.get("ignore") or config.get("exclude_modules") or [])
    plan |= {name: MODELOPT_ALGOS[layer["quant_algo"]]() for name, layer in config["quantized_layers"].items()}
    return plan | {"*": base} if base else plan


def fp4_experts_plan(config: dict) -> dict | None:
    "DeepSeek-V4's FP8 checkpoints, whose routed experts are MXFP4 (`expert_dtype: fp4`)"
    if config.get("quant_method") != "fp8" or config.get("expert_dtype") != "fp4":
        return None
    return {
        "*.mlp.experts": Mxfp4Config(),
        "*": {key: value for key, value in config.items() if key != "expert_dtype"},
    }


# Saved formats that quantize per module, turned into a plan keyed by checkpoint module names. ModelOpt first: its
# per-layer configs (e.g. NVFP4 experts) override DeepSeek's `expert_dtype`
PLAN_CONVERTERS: dict[str, Callable[[dict], dict | None]] = {
    "modelopt": modelopt_plan,
    "fp4_experts": fp4_experts_plan,
}


class QuantizationPlanHfQuantizer(HfQuantizer):
    """
    The quantizers of a plan: like the dtype plan's, the glob pattern a module or param matches picks its quantizer,
    the others stay unquantized. While loading, a param goes through the converters and ops of its quantizer.
    """

    def __init__(self, quantizers: dict[str, HfQuantizer | None], checkpoint_names: bool = False):
        self.checkpoint_names = checkpoint_names  # patterns are checkpoint module names, renamed in `preprocess_model`
        self.set_quantizers(quantizers)
        self.pre_quantized = all(quantizer.pre_quantized for quantizer in self.all_quantizers)
        for quantizer in self.all_quantizers:
            quantizer.plan = self

    @property
    def all_quantizers(self):
        "Each quantizer once: patterns with equal configs share one"
        return [quantizer for quantizer in dict.fromkeys(self.quantizers.values()) if quantizer is not None]

    def set_quantizers(self, quantizers):
        from ..core_model_loading import build_glob_alternation

        self.quantizers = quantizers
        self.quantization_config = {
            pattern: None if quantizer is None else quantizer.quantization_config
            for pattern, quantizer in quantizers.items()
        }
        # Exact module names first, then globs, `"*"` last: each matches anywhere in the name, the first one wins
        patterns = sorted(quantizers, key=lambda pattern: (pattern == "*") + ("*" in pattern))
        globs = [
            (pattern if pattern.startswith("*") else f"*{pattern}")
            if "*" in pattern
            else rf"(?:^|.+\.){re.escape(pattern)}(?:\.|$)"
            for pattern in patterns
        ]
        self.alternation, groups, _ = build_glob_alternation(globs)
        self.group_to_pattern = dict(zip(groups, patterns))

    def get_quantizer(self, name):
        match = self.alternation.match(name)
        return self.quantizers[self.group_to_pattern[match.lastgroup]] if match else None

    def update_weight_conversions(self, weight_conversions, model):
        from ..core_model_loading import WeightConverter

        # Each model converter is updated by the quantizer of its target param only, e.g. the fp8 quantizer must not
        # add its companions to the merge of NVFP4 experts
        names = "\n".join(name for name, _ in chain(model.named_parameters(), model.named_buffers()))

        def owner(converter):
            target = converter._original_target_patterns[0].replace("*", ".*")
            match = re.search(rf"^(.*{target})$", names, re.MULTILINE)
            return self.get_quantizer(match.group(1)) if match else None

        owners = {id(c): owner(c) for c in weight_conversions if isinstance(c, WeightConverter)}
        general = {}
        for quantizer in self.all_quantizers:
            # Chained like a single quantizer (renamings keep their order), but blind to the converters of others
            hidden = [c for c in weight_conversions if owners.get(id(c), quantizer) is not quantizer]
            visible = [c for c in weight_conversions if owners.get(id(c), quantizer) is quantizer]
            own = quantizer.get_weight_conversions()
            # Taken out of `update_weight_conversions` by patterns, not position: renamings may come after them
            general_patterns = [(c._original_source_patterns, c._original_target_patterns) for c in own]
            updated = [
                c
                for c in quantizer.update_weight_conversions(visible, model)
                if (getattr(c, "_original_source_patterns", None), getattr(c, "_original_target_patterns", None))
                not in general_patterns
            ]
            for converter in updated:
                if isinstance(converter, WeightConverter):
                    owners.setdefault(id(converter), quantizer)
                elif not any(converter is c for c in visible):  # its own renamings: only on its own params
                    converter.applies_to = lambda key, quantizer=quantizer: self.get_quantizer(key) is quantizer
            weight_conversions = updated + hidden
            for converter in own:
                general.setdefault(converter._original_target_patterns[0], []).append(converter)
        # One general converter per param collects the names of every quantizer, before the own-name ones take them
        # (`down_proj` would take `shared_experts.down_proj.weight_sf_rowmajor`). Its quantizer gives a param its own
        # ops in `update_weight_transform`; the others pass through the params without companions
        # `weight` first: it takes `*.down_proj.weight` before the experts' own-name `down_proj` does
        general = dict(sorted(general.items(), key=lambda item: item[0] != "weight"))
        return weight_conversions + [
            WeightConverter(
                list(dict.fromkeys(name for c in converters for name in c._original_source_patterns)),
                target,
                converters[0].operations,
            )
            for target, converters in general.items()
        ]

    def update_weight_transform(self, transform, model, name):
        quantizer = self.get_quantizer(name)
        if quantizer is None:
            return transform
        # A general converter of another quantizer may have matched the first key of `name`: swap in its quantizer's
        targets = getattr(transform, "_original_target_patterns", None)
        own = next((c for c in quantizer.get_weight_conversions() if c._original_target_patterns == targets), None)
        return quantizer.update_weight_transform(own or transform, model, name)

    def param_element_size(self, model, param_name, param):
        quantizer = self.get_quantizer(param_name)
        return param.element_size() if quantizer is None else quantizer.param_element_size(model, param_name, param)

    def validate_environment(self, *args, **kwargs):
        for quantizer in self.all_quantizers:
            quantizer.validate_environment(*args, **kwargs)

    def update_dtype(self, dtype):
        for quantizer in self.all_quantizers:
            dtype = quantizer.update_dtype(dtype)
        return dtype

    def update_device_map(self, device_map):
        for quantizer in self.all_quantizers:
            device_map = quantizer.update_device_map(device_map)
        return device_map

    def adjust_max_memory(self, max_memory):
        for quantizer in self.all_quantizers:
            max_memory = quantizer.adjust_max_memory(max_memory)
        return max_memory

    def rename_patterns(self, model):
        "Renames checkpoint module names to the model's, the way the checkpoint keys are renamed on load"
        from ..conversion_mapping import get_model_conversion_mapping

        transforms = get_model_conversion_mapping(model)
        quantizers = {}
        for pattern, quantizer in self.quantizers.items():
            name = f"{pattern}.weight"  # the converters match params: per-expert ones collapse on the fused param
            for transform in transforms:
                name, _ = transform.rename_source_key(name)
            quantizers.setdefault(name.removesuffix(".weight"), quantizer)
        self.set_quantizers(quantizers)
        model.config.quantization_config = self.quantization_config

    def preprocess_model(self, model, **kwargs):
        if self.checkpoint_names:
            self.rename_patterns(model)
        for quantizer in self.all_quantizers:
            quantizer.preprocess_model(model, **kwargs)
        self.replace_modules(model)
        for quantizer in self.all_quantizers:
            quantizer.update_parallel_plan(model, model.tp_plan)
            quantizer.update_parallel_plan(model, model.ep_plan)

    def postprocess_model(self, model, **kwargs):
        """
        Runs the after-load hook of every quantizer. The patterns loaded with `dequantize=True` become unquantized
        (`None`): with none quantized left, the model is plain unquantized.
        """
        for quantizer in self.all_quantizers:
            quantizer._process_model_after_weight_loading(model, **kwargs)
        # Dequantized patterns stay in the plan as `None`, so a broader pattern (e.g. `"*"`) can't claim their modules
        self.set_quantizers(
            {
                pattern: None
                if quantizer is None
                or (quantizer.pre_quantized and getattr(quantizer.quantization_config, "dequantize", False))
                else quantizer
                for pattern, quantizer in self.quantizers.items()
            }
        )
        if not self.all_quantizers:
            self.remove_quantization_config(model)
            return model
        model.config.quantization_config = self.quantization_config
        _assign_is_quantized(model)
        return model

    def is_serializable(self):
        return all(quantizer.is_serializable() for quantizer in self.all_quantizers)

    @property
    def is_trainable(self):
        return all(quantizer.is_trainable for quantizer in self.all_quantizers)


def get_hf_quantizer(config, quantization_config, device_map, weights_only, user_agent, gguf_file=None):
    if gguf_file is not None:
        if quantization_config is None:
            quantization_config = GgufConfig(gguf_file=gguf_file)
        elif isinstance(quantization_config, GgufConfig):
            quantization_config.gguf_file = gguf_file
    elif isinstance(quantization_config, GgufConfig):
        raise ValueError(
            "Loading a GGUF checkpoint needs the file named as `from_pretrained(..., gguf_file=...)`. "
            "`GgufConfig` carries the loading options and does not have to repeat it."
        )

    quantization_params_from_config = getattr(config, "quantization_config", None) or getattr(
        config.get_text_config(decoder=True), "quantization_config", None
    )
    checkpoint_names = False
    if isinstance(quantization_params_from_config, dict) and not is_quantization_plan(quantization_params_from_config):
        for convert in PLAN_CONVERTERS.values():
            if (plan := convert(quantization_params_from_config)) is not None:
                quantization_params_from_config, checkpoint_names = plan, True
                break
    pre_quantized = quantization_params_from_config is not None
    plan_from_config = is_quantization_plan(quantization_params_from_config)
    if (
        pre_quantized
        and not plan_from_config
        and not AutoHfQuantizer.supports_quant_method(quantization_params_from_config)
    ):
        pre_quantized = False

    saved_plan = quantization_params_from_config if is_quantization_plan(quantization_params_from_config) else None
    user_plan = quantization_config if is_quantization_plan(quantization_config) else None
    if (plan := saved_plan or user_plan) is not None:
        # One quantizer per pattern. Loading attributes update the saved configs: for every pattern (e.g.
        # `{"dequantize": True}`), or per pattern (a plan of them)
        # A single config only brings its loading attributes: the saved plan may mix methods (e.g. fp8 + mxfp4)
        if isinstance(quantization_config, QuantizationConfigMixin):
            quantization_config = quantization_config.get_loading_attributes()
        loading_attributes = user_plan if user_plan is not None else dict.fromkeys(plan, quantization_config)
        quantizers, shared = {}, {}  # patterns with equal configs share one quantizer
        for pattern, pattern_config in plan.items():
            if pattern_config is None:
                quantizers[pattern] = None
                continue
            if plan_from_config:
                attributes = loading_attributes.get(pattern)
                pattern_config = AutoHfQuantizer.merge_quantization_configs(
                    pattern_config, None if isinstance(attributes, dict) else attributes
                )
                for key, value in (attributes if isinstance(attributes, dict) else {}).items():
                    setattr(pattern_config, key, value)
            elif isinstance(pattern_config, dict):
                pattern_config = AutoQuantizationConfig.from_dict(pattern_config)
            if (key := repr(pattern_config)) not in shared:
                shared[key] = AutoHfQuantizer.from_config(pattern_config, pre_quantized=plan_from_config)
            quantizers[pattern] = shared[key]
        hf_quantizer = QuantizationPlanHfQuantizer(quantizers, checkpoint_names)
        config.quantization_config = hf_quantizer.quantization_config
    elif pre_quantized or quantization_config is not None:
        if pre_quantized:
            config.quantization_config = AutoHfQuantizer.merge_quantization_configs(
                quantization_params_from_config, quantization_config
            )
        else:
            config.quantization_config = quantization_config

        hf_quantizer = AutoHfQuantizer.from_config(
            config.quantization_config,
            pre_quantized=pre_quantized,
        )
    else:
        hf_quantizer = None

    if hf_quantizer is not None:
        quant_method = getattr(hf_quantizer.quantization_config, "quant_method", "plan")
        if gguf_file is not None and quant_method != QuantizationMethod.GGUF:
            raise ValueError(
                "You cannot combine Quantization and loading a model from a GGUF file, try again by making sure "
                "you did not passed a `quantization_config` or that you did not load a quantized model from the Hub."
            )
        hf_quantizer.validate_environment(
            device_map=device_map,
            weights_only=weights_only,
        )
        device_map = hf_quantizer.update_device_map(device_map)
        config = hf_quantizer.update_attn_implementation(config)

        # In order to ensure popular quantization methods are supported. Can be disable with `disable_telemetry`
        if not getattr(hf_quantizer.quantization_config, "dequantize", False):
            user_agent["quant"] = getattr(quant_method, "value", quant_method)
    return hf_quantizer, config, device_map
