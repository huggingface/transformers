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
import warnings
from typing import Optional, Union

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
    FPQuantConfig,
    GPTQConfig,
    HiggsConfig,
    HqqConfig,
    Mxfp4Config,
    QuantizationConfigMixin,
    QuantizationMethod,
    QuantoConfig,
    QuarkConfig,
    SpQRConfig,
    TorchAoConfig,
    VptqConfig,
)
from .base import HfQuantizer
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
from .quantizer_fp_quant import FPQuantHfQuantizer
from .quantizer_gptq import GptqHfQuantizer
from .quantizer_higgs import HiggsHfQuantizer
from .quantizer_hqq import HqqHfQuantizer
from .quantizer_mxfp4 import Mxfp4HfQuantizer
from .quantizer_quanto import QuantoHfQuantizer
from .quantizer_quark import QuarkHfQuantizer
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
    "auto-round": AutoRoundQuantizer,
    "mxfp4": Mxfp4HfQuantizer,
}

AUTO_QUANTIZATION_CONFIG_MAPPING = {
    "awq": AwqConfig,
    "bitsandbytes_4bit": BitsAndBytesConfig,
    "bitsandbytes_8bit": BitsAndBytesConfig,
    "eetq": EetqConfig,
    "gptq": GPTQConfig,
    "aqlm": AqlmConfig,
    "quanto": QuantoConfig,
    "quark": QuarkConfig,
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
    "auto-round": AutoRoundConfig,
    "mxfp4": Mxfp4Config,
}

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
    def from_config(cls, quantization_config: Union[QuantizationConfigMixin, dict], **kwargs):
        # Convert it to a QuantizationConfig if the q_config is a dict
        if isinstance(quantization_config, dict):
            quantization_config = AutoQuantizationConfig.from_dict(quantization_config)

        quant_method = quantization_config.quant_method

        # Again, we need a special care for bnb as we have a single quantization config
        # class for both 4-bit and 8-bit quantization
        if quant_method == QuantizationMethod.BITS_AND_BYTES:
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
        quantization_config: Union[dict, QuantizationConfigMixin],
        quantization_config_from_args: Optional[QuantizationConfigMixin],
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

        if (
            isinstance(
                quantization_config,
                (GPTQConfig, AwqConfig, AutoRoundConfig, FbgemmFp8Config, CompressedTensorsConfig, Mxfp4Config),
            )
            and quantization_config_from_args is not None
        ):
            loading_attr_dict = quantization_config_from_args.get_loading_attributes()
            for attr, val in loading_attr_dict.items():
                setattr(quantization_config, attr, val)

            warning_msg += f"However, loading attributes (e.g. {list(loading_attr_dict.keys())}) will be overwritten with the one you passed to `from_pretrained`. The rest will be ignored."

        if warning_msg != "" and not isinstance(quantization_config, Mxfp4Config):
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
            raise ValueError("Quantizer must extend HfQuantizer")

        AUTO_QUANTIZER_MAPPING[name] = cls
        return cls

    return register_quantizer_fn


def get_hf_quantizer(config, quantization_config, dtype, from_tf, from_flax, device_map, weights_only, user_agent):
    pre_quantized = hasattr(config, "quantization_config")
    if pre_quantized and not AutoHfQuantizer.supports_quant_method(config.quantization_config):
        pre_quantized = False

    if pre_quantized or quantization_config is not None:
        if pre_quantized:
            config.quantization_config = AutoHfQuantizer.merge_quantization_configs(
                config.quantization_config, quantization_config
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
        hf_quantizer.validate_environment(
            dtype=dtype,
            from_tf=from_tf,
            from_flax=from_flax,
            device_map=device_map,
            weights_only=weights_only,
        )
        dtype = hf_quantizer.update_dtype(dtype)
        device_map = hf_quantizer.update_device_map(device_map)
        config = hf_quantizer.update_tp_plan(config)
        config = hf_quantizer.update_ep_plan(config)

        # In order to ensure popular quantization methods are supported. Can be disable with `disable_telemetry`
        if not getattr(hf_quantizer.quantization_config, "dequantize", False):
            quant_method = hf_quantizer.quantization_config.quant_method
            user_agent["quant"] = getattr(quant_method, "value", quant_method)
    return hf_quantizer, config, dtype, device_map
