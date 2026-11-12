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
from .schema_readers import read_quantization_schema


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
    The Auto-HF quantizer: picks the `HfQuantizer` of a `QuantizationConfig` (`from_config`), or of each pattern
    of a quantization plan (`from_plan`).
    """

    @classmethod
    def from_plan(cls, plan: dict, loading, pre_quantized: bool, checkpoint_names: bool = False):
        """
        The quantizer of each pattern of `plan`, one per distinct config. The saved plan may mix formats (e.g. fp8 +
        mxfp4).

        Loading a saved plan, the `quantization_config` passed to `from_pretrained` brings loading attributes: for
        every pattern (one config or dict, e.g. `{"dequantize": True}`), or per pattern (a plan of them).
        """
        if not is_quantization_plan(loading):
            loading = dict.fromkeys(plan, loading)
        quantizers, shared = {}, {}
        for pattern, config in plan.items():
            if config is None:
                quantizers[pattern] = None
                continue
            attributes = loading.get(pattern) if pre_quantized else None
            if isinstance(config, dict) and isinstance(attributes, AutoRoundConfig):
                config = AutoRoundConfig.from_dict(config)  # AutoRound loads GPTQ and AWQ checkpoints
            elif isinstance(config, dict):
                config = AutoQuantizationConfig.from_dict(config)
            if isinstance(attributes, QuantizationConfigMixin):
                if type(attributes).__name__ != type(config).__name__:
                    raise ValueError(
                        f"The model is quantized with {type(config).__name__} but you are passing a "
                        f"{type(attributes).__name__} config. Please make sure to pass the same quantization config class "
                        "to `from_pretrained` with different loading attributes."
                    )
                attributes = attributes.get_loading_attributes()
                logger.info(
                    "You passed `quantization_config` to `from_pretrained` but the model you're loading already has a "
                    f"`quantization_config`, which is used. Its loading attributes {list(attributes)} are "
                    "overwritten with the ones you passed, the rest is ignored."
                )
            if isinstance(attributes, dict):
                for key, value in attributes.items():
                    setattr(config, key, value)
            if (key := repr(config)) not in shared:
                if not pre_quantized and getattr(config, "dequantize", False):
                    logger.warning(
                        f"`dequantize=True` only applies to pre-quantized checkpoints: it is ignored for {pattern}."
                    )
                shared[key] = cls.from_config(config, pre_quantized=pre_quantized)
            quantizers[pattern] = shared[key]
        return QuantizationPlanHfQuantizer(quantizers, checkpoint_names)

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


class QuantizationPlanHfQuantizer(HfQuantizer):
    """
    A quantization plan: like the dtype plan's, the glob pattern a module or param matches picks its quantizer, the
    others stay unquantized.

    Every quantized model loads through one: a single quantizer is the plan `{"*": q}`. It knows every module's
    quantizer, so it builds the conversions and gives each param its ops; a quantizer only describes its format
    (`serialized_names`, `checkpoint_renames`, `get_load_operations`, `quantized_module`).
    """

    def __init__(self, quantizers: dict[str, HfQuantizer | None], checkpoint_names: bool = False):
        # The patterns are checkpoint module names, renamed in `map_module_to_quantizer`
        self.checkpoint_names = checkpoint_names
        self.patterns = None
        self.set_quantizers(quantizers)
        self.pre_quantized = all(quantizer.pre_quantized for quantizer in self.all_quantizers)
        # Mixed, each format swaps the modules the plan gives it (`quantized_module`): one swapping all of a model's
        # linears itself would take the others' too
        own_swaps = [type(q).__name__ for q in self.all_quantizers if not q.replaces_modules]
        if len(self.all_quantizers) > 1 and own_swaps:
            raise ValueError(f"{', '.join(own_swaps)} can't be mixed with other formats in a quantization plan yet")

    @property
    def all_quantizers(self):
        """Each quantizer once: patterns with equal configs share one"""
        return [quantizer for quantizer in dict.fromkeys(self.quantizers.values()) if quantizer is not None]

    def set_quantizers(self, quantizers):
        self.quantizers = quantizers
        self.quantization_config = {
            pattern: None if quantizer is None else quantizer.quantization_config
            for pattern, quantizer in quantizers.items()
        }
        if list(self.quantization_config) == ["*"]:  # a single quantizer keeps its config as is
            self.quantization_config = self.quantization_config["*"]

    @property
    def quantizer(self):
        """The quantizer of the whole model, if a single one"""
        return self.quantizers["*"] if list(self.quantizers) == ["*"] else None

    @property
    def __class__(self):
        # `isinstance(hf_quantizer, Bnb4BitHfQuantizer)` keeps working for a single quantizer
        return QuantizationPlanHfQuantizer if self.quantizer is None else type(self.quantizer)

    def __reduce__(self):  # copied and pickled as the plan, not as `__class__`
        return QuantizationPlanHfQuantizer, (self.quantizers,), self.__dict__

    def __getattr__(self, name):
        # A single quantizer's own attributes, e.g. GGUF's `metadata` or compressed-tensors' `compressor`
        if name.startswith("__") or "quantizers" not in self.__dict__ or self.quantizer is None:
            raise AttributeError(name)
        return getattr(self.quantizer, name)

    def map_module_to_quantizer(self, model):
        r"""
        The plan's patterns, in order, for `get_quantizer`. A checkpoint's plan names checkpoint modules: renamed
        to the model's like their params' keys (per-expert names onto the fused module), its own names kept too.

        E.g. the plan `{"layers.0.mlp.experts": nvfp4, "*": fp8}` gives:

            self.patterns = [("*layers.0.mlp.experts.*", nvfp4), ("**.*", fp8)]  # exact names first, `"*"` last
            self.matcher  = (?P<p0>.*layers\.0\.mlp\.experts\..*)|(?P<p1>.*\..*)  # one group per pattern
            get_quantizer("model.layers.0.mlp.experts.gate_up_proj")  # matches `p0` -> nvfp4
            get_quantizer("model.layers.0.self_attn.q_proj")          # matches `p1` -> fp8

        and, for FP8's `checkpoint_renames = {"scale": "weight_scale_inv", "input_scale": "activation_scale"}`:

            self.renamers[fp8] = ((?P<p0>.*\.scale)|(?P<p1>.*\.input_scale), ["weight_scale_inv", "activation_scale"])
            rename_layer("layers.0.self_attn.q_proj.scale")  # `p0` -> "layers.0.self_attn.q_proj.weight_scale_inv"
        """
        from ..conversion_mapping import get_model_conversion_mapping
        from ..core_model_loading import WeightConverter, WeightRenaming, rename_source_key

        checkpoint = {}
        if self.checkpoint_names:
            transforms = get_model_conversion_mapping(model)
            converters = [t for t in transforms if isinstance(t, WeightConverter)]
            renamings = [t for t in transforms if isinstance(t, WeightRenaming)]
            checkpoint = {pattern: quantizer for pattern, quantizer in self.quantizers.items() if "*" not in pattern}
            renamed = {
                rename_source_key(f"{pattern}.weight", renamings, converters)[0].rpartition(".")[0]: quantizer
                for pattern, quantizer in self.quantizers.items()
            }
            self.set_quantizers(renamed)
            self.checkpoint_names = False

        # Exact names first (the checkpoint's too, for `rename_layer` on its keys), then globs, `"*"` last. `P` (or
        # `P.*`) names the module `P` under any prefix, and everything under it
        patterns = self.quantizers | checkpoint
        order = sorted(patterns, key=lambda pattern: ("*" in pattern) + (pattern == "*"))
        self.patterns = [(f"*{pattern.removesuffix('.*')}.*", patterns[pattern]) for pattern in order]
        self.compile_matcher()

        # Each format's `checkpoint_renames` as one regex too, with our names in order, for `rename_layer`
        self.renamers = {}
        for quantizer in self.all_quantizers:
            names = quantizer.checkpoint_renames
            if quantizer.pre_quantized and names:
                renamer = re.compile("|".join(f"(?P<p{i}>.*\\.{n.replace('*', '.*')})" for i, n in enumerate(names)))
                self.renamers[quantizer] = (renamer, list(names.values()))

    def compile_matcher(self):
        """
        `self.patterns` as one regex, the first matching pattern wins. In `*P.*`, the dot after `P` is a module
        boundary: `mlp.gate` doesn't name `mlp.gate_proj`, nor `layers.1` `layers.11`.
        """
        groups = []
        for i, (glob, _) in enumerate(self.patterns):
            name = glob[1:].removesuffix(".*").replace("*", ".*")
            groups.append(f"(?P<p{i}>.*(?:{name})\\..*)")
        self.matcher = re.compile("|".join(groups))

    def get_quantizer(self, name):
        """
        The quantizer of the first plan pattern naming the module or key `name` (a module's name ends with a dot)
        """
        match = self.matcher.fullmatch(f"{name}.")
        return self.patterns[int(match.lastgroup[1:])][1] if match else None

    def update_weight_conversions(self, weight_conversions, model):
        """
        The model's conversions, then the formats' own converters. A model converter's `.weight` also takes the scales
        named after it (`weight_scale_inv`, see `rename_layer`), each into the target named like it. E.g. the experts'
        converter merging `gate_proj` and `up_proj` into `gate_up_proj` (FP8):

            mlp.experts.3.gate_proj.weight            -> mlp.experts.gate_up_proj            MergeModulelist, Concatenate,
            mlp.experts.3.up_proj.weight              ->                                      then Fp8Deserialize
            mlp.experts.3.gate_proj.weight_scale_inv  -> mlp.experts.gate_up_proj_scale_inv  same ops, its own target
            mlp.experts.3.up_proj.weight_scale_inv    ->

        Dequantized, the converter is widened to read the scales along (`*.gate_proj.weight_scale_inv`, ...), into the
        one target `gate_up_proj`: `Fp8Dequantize` folds each scale into its weight before the merge.
        """
        from ..core_model_loading import WeightConverter, _IdentityOp

        if self.patterns is None:
            self.map_module_to_quantizer(model)

        # Dequantized, a weight's scales fold into it before the model's ops: the model's `.weight` converters read
        # them along. Every format's names (as `rename_layer` leaves them): a layer's checkpoint only has its own
        names = []
        for quantizer in self.all_quantizers:
            weight_conversions = quantizer.update_weight_conversions(weight_conversions, model)
            if getattr(quantizer.quantization_config, "dequantize", False):
                names += [*quantizer.serialized_names["weight"], *quantizer.checkpoint_renames.values()]

        # A single format dequantizes on the converters: saving reverses it (e.g. FP8 quantizes back)
        quantizer = self.quantizer
        load = [quantizer.get_load_operation()(quantizer)] if quantizer is not None and names else []
        for i, converter in enumerate(weight_conversions):
            sources = converter._original_source_patterns
            if names and isinstance(converter, WeightConverter) and all(s.endswith(".weight") for s in sources):
                sources = [source.removesuffix("weight") + name for source in sources for name in names]
                weight_conversions[i] = converter.copy(
                    sources, converter._original_target_patterns, operations=[*load, *converter.operations]
                )
                # Reversed on save to the weights only: quantizing back names their scales
                weight_conversions[i]._original_source_patterns = converter._original_source_patterns

        general = [converter for quantizer in self.all_quantizers for converter in quantizer.get_weight_conversions()]
        if quantizer is None:  # mixed: each param gets its format's op, see `update_weight_transform`
            general = [
                c.copy(c._original_source_patterns, c._original_target_patterns, operations=[_IdentityOp()])
                for c in general
            ]
        return [*weight_conversions, *general]

    def rename_layer(self, key):
        """
        A checkpoint key in its layer's format names: the first of its format's `checkpoint_renames` (theirs -> ours)
        naming the key's last name. E.g. on an FP8 layer, `q_proj.scale` -> `q_proj.weight_scale_inv`, see
        `map_module_to_quantizer` for the regex. A key of a format without `checkpoint_renames` is kept.
        """
        renamer, ours = self.renamers.get(self.get_quantizer(key), (None, None))
        match = renamer.fullmatch(key) if renamer else None
        return key if match is None else f"{key.rpartition('.')[0]}.{ours[int(match.lastgroup[1:])]}"

    def update_weight_transform(self, transform, model, name):
        """
        The ops of the param `name`, once its first key matched a converter, from its layer's quantizer: a
        deserialization (after the converter's ops), a dequantization (before them), or a quantization of a full
        precision checkpoint.
        """
        from ..core_model_loading import WeightConverter

        quantizer = self.get_quantizer(name)
        if quantizer is None:
            return transform

        # The on-the-fly quantization moves to the load operations in #50386
        if not quantizer.pre_quantized:
            if quantizer.param_needs_quantization(model, name):
                transform.quantization_operation = quantizer.get_quantize_ops()
            return transform

        operation = quantizer.get_load_operation()
        tensors = {tensor for names in quantizer.serialized_names.values() for tensor in names}
        if operation is None or not isinstance(transform, WeightConverter) or name.rpartition(".")[2] not in tensors:
            return transform  # e.g. a merged expert bias: not one of the format's tensors
        # Dequantized, the scales fold into the weight before the model's ops
        load = operation(quantizer)
        ops = [op for op in transform.operations if not isinstance(op, operation)]  # already on a single format's
        dequantize = getattr(quantizer.quantization_config, "dequantize", False)
        transform.operations = [load, *ops] if dequantize else [*ops, load]
        return transform

    def param_element_size(self, model, param_name, param):
        if self.patterns is None:
            self.map_module_to_quantizer(model)
        quantizer = self.get_quantizer(param_name)
        return param.element_size() if quantizer is None else quantizer.param_element_size(model, param_name, param)

    def validate_environment(self, *args, **kwargs):
        for quantizer in self.all_quantizers:
            quantizer.validate_environment(*args, **kwargs)

    def update_attn_implementation(self, config):
        for quantizer in self.all_quantizers:
            config = quantizer.update_attn_implementation(config)
        return config

    def get_new_dtype(self, converter, param, dtype):
        return (self.quantizer or super()).get_new_dtype(converter, param, dtype)

    def dequantize(self, model, dtype=None):
        return (self.quantizer or super()).dequantize(model, dtype)

    def get_state_dict_and_metadata(self, model):
        return (self.quantizer or super()).get_state_dict_and_metadata(model)

    @property
    def is_compileable(self):
        return all(quantizer.is_compileable for quantizer in self.all_quantizers)

    @property
    def is_qat_trainable(self):
        return all(quantizer.is_qat_trainable for quantizer in self.all_quantizers)

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

    def preprocess_model(self, model, **kwargs):
        self.map_module_to_quantizer(model)
        for quantizer in self.all_quantizers:
            quantizer.plan_module_names = [pattern for pattern in self.quantizers if "*" not in pattern]
        super().preprocess_model(model, **kwargs)

    def replace_modules(self, model):
        """
        Each format's `modules_to_not_convert` (an exact name, a regex like `layers.*` or a suffix like `fc1`) is
        folded once into the plan as an unquantized pattern, so a module costs a single `get_quantizer`. They go after
        the plan's exact names, which win (e.g. `lm_head: fp8`), and before its globs.
        """
        skipped = []
        for quantizer in self.all_quantizers:
            if quantizer.replaces_modules:  # the others skip modules in their own swaps
                skipped += getattr(quantizer, "modules_to_not_convert", None) or []
        if skipped:
            exact = [glob for glob, _ in self.patterns if "*" not in glob[1:-2]]
            skips = [(f"*{name.removesuffix('$').removesuffix('.*')}.*", None) for name in dict.fromkeys(skipped)]
            self.patterns[len(exact) : len(exact)] = skips
            self.compile_matcher()
        super().replace_modules(model)

    def postprocess_model(self, model, **kwargs):
        """
        The patterns loaded with `dequantize=True` become unquantized (`None`, so a broader pattern like `"*"` can't
        claim their modules): with none quantized left, the model is plain unquantized.
        """
        for quantizer in self.all_quantizers:
            quantizer._process_model_after_weight_loading(model, **kwargs)
        dequantized = {
            quantizer
            for quantizer in self.all_quantizers
            if quantizer.pre_quantized and getattr(quantizer.quantization_config, "dequantize", False)
        }
        if self.quantizer is not None:  # a single format saves back in its format
            dequantized = set()
        self.set_quantizers({p: None if q in dequantized else q for p, q in self.quantizers.items()})
        self.patterns = [(glob, None if q in dequantized else q) for glob, q in self.patterns]
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
    """
    A plan maps module names or glob patterns to the config quantizing the modules they match (`None`: unquantized)
    """
    return (
        isinstance(quantization_config, dict)
        and len(quantization_config) > 0
        and all(
            isinstance(config, (dict, QuantizationConfigMixin, type(None))) for config in quantization_config.values()
        )
    )


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
        # Saved formats quantizing per module, as a plan of checkpoint module names, see `schema_readers`
        plan = read_quantization_schema(quantization_params_from_config)
        if plan is not None:
            quantization_params_from_config, checkpoint_names = plan, True
    pre_quantized = quantization_params_from_config is not None
    plan_from_config = is_quantization_plan(quantization_params_from_config)
    if (
        pre_quantized
        and not plan_from_config
        and not AutoHfQuantizer.supports_quant_method(quantization_params_from_config)
    ):
        pre_quantized = False

    saved_plan = quantization_params_from_config if plan_from_config else None
    user_plan = quantization_config if is_quantization_plan(quantization_config) else None
    if user_plan is not None and saved_plan is None:
        pre_quantized = False  # the passed plan quantizes the model, a saved single config is ignored
    # A single config is the plan `{"*": config}`: the saved one, else the passed one
    single_config = quantization_params_from_config if pre_quantized else quantization_config
    plan = saved_plan or user_plan
    if plan is None and single_config is not None:
        plan = {"*": single_config}

    hf_quantizer = None
    if plan is not None:
        hf_quantizer = AutoHfQuantizer.from_plan(plan, quantization_config, pre_quantized, checkpoint_names)
        if not checkpoint_names:  # else the checkpoint's own config is kept, e.g. ModelOpt's `quantized_layers`
            config.quantization_config = hf_quantizer.quantization_config

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
