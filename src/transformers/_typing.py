# Copyright 2026 The HuggingFace Inc. team.
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
"""Typing helpers shared across the Transformers library."""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias


if TYPE_CHECKING:
    import torch

    from .cache_utils import Cache


# A few helpful type aliases
Level: TypeAlias = int
ExcInfo: TypeAlias = (
    None
    | bool
    | BaseException
    | tuple[type[BaseException], BaseException, object]  # traceback is `types.TracebackType`, but keep generic here
)


class TransformersLogger(Protocol):
    # ---- Core Logger identity / configuration ----
    name: str
    level: int
    parent: logging.Logger | None
    propagate: bool
    disabled: bool
    handlers: list[logging.Handler]

    # Exists on Logger; default is True. (Not heavily used, but is part of API.)
    raiseExceptions: bool  # type: ignore[assignment]

    # ---- Standard methods ----
    def setLevel(self, level: Level) -> None: ...
    def isEnabledFor(self, level: Level) -> bool: ...
    def getEffectiveLevel(self) -> int: ...

    def getChild(self, suffix: str) -> logging.Logger: ...

    def addHandler(self, hdlr: logging.Handler) -> None: ...
    def removeHandler(self, hdlr: logging.Handler) -> None: ...
    def hasHandlers(self) -> bool: ...

    # ---- Logging calls ----
    def debug(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def info(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def warning(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def warn(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def error(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def exception(self, msg: object, *args: object, exc_info: ExcInfo = True, **kwargs: object) -> None: ...
    def critical(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def fatal(self, msg: object, *args: object, **kwargs: object) -> None: ...

    # The lowest-level primitive
    def log(self, level: Level, msg: object, *args: object, **kwargs: object) -> None: ...

    # ---- Record-level / formatting ----
    def makeRecord(
        self,
        name: str,
        level: Level,
        fn: str,
        lno: int,
        msg: object,
        args: tuple[object, ...] | Mapping[str, object],
        exc_info: ExcInfo,
        func: str | None = None,
        extra: Mapping[str, object] | None = None,
        sinfo: str | None = None,
    ) -> logging.LogRecord: ...

    def handle(self, record: logging.LogRecord) -> None: ...
    def findCaller(
        self,
        stack_info: bool = False,
        stacklevel: int = 1,
    ) -> tuple[str, int, str, str | None]: ...

    def callHandlers(self, record: logging.LogRecord) -> None: ...
    def getMessage(self) -> str: ...  # NOTE: actually on LogRecord; included rarely; safe to omit if you want

    def _log(
        self,
        level: Level,
        msg: object,
        args: tuple[object, ...] | Mapping[str, object],
        exc_info: ExcInfo = None,
        extra: Mapping[str, object] | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
    ) -> None: ...

    # ---- Filters ----
    def addFilter(self, filt: logging.Filter) -> None: ...
    def removeFilter(self, filt: logging.Filter) -> None: ...
    @property
    def filters(self) -> list[logging.Filter]: ...

    def filter(self, record: logging.LogRecord) -> bool: ...

    # ---- Convenience helpers ----
    def setFormatter(self, fmt: logging.Formatter) -> None: ...  # mostly on handlers; present on adapters sometimes
    def debugStack(self, msg: object, *args: object, **kwargs: object) -> None: ...  # not std; safe no-op if absent

    # ---- stdlib dictConfig-friendly / extra storage ----
    # Logger has `manager` and can have arbitrary attributes; Protocol can't express arbitrary attrs,
    # but we can at least include `__dict__` to make "extra attributes" less painful.
    __dict__: MutableMapping[str, Any]

    # ---- Transformers logger specific methods ----
    def warning_advice(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def warning_once(self, msg: object, *args: object, **kwargs: object) -> None: ...
    def info_once(self, msg: object, *args: object, **kwargs: object) -> None: ...


class GenerativePreTrainedModel(Protocol):
    """Protocol for the model interface that GenerationMixin expects.

    GenerationMixin is designed to be mixed into PreTrainedModel subclasses. This Protocol documents the
    attributes and methods the mixin relies on from its host class. It is *not* used at runtime — its
    purpose is to help the ``ty`` type checker resolve ``self.<attr>`` accesses inside the mixin.
    """

    config: Any  # PretrainedConfig — kept as Any to avoid circular imports
    device: torch.device
    dtype: torch.dtype
    main_input_name: str
    base_model_prefix: str
    _is_stateful: bool
    hf_quantizer: Any
    encoder: Any
    hf_device_map: dict[str, Any]
    _cache: Cache

    generation_config: Any  # GenerationConfig

    def __getattr__(self, name: str) -> Any: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def can_generate(self) -> bool: ...
    def get_encoder(self) -> Any: ...
    def get_output_embeddings(self) -> Any: ...
    def get_input_embeddings(self) -> Any: ...
    def set_output_embeddings(self, value: Any) -> None: ...
    def set_input_embeddings(self, value: Any) -> None: ...
    def get_compiled_call(self, compile_config: Any) -> Any: ...
    def set_experts_implementation(self, *args: Any, **kwargs: Any) -> Any: ...
    def _supports_logits_to_keep(self) -> bool: ...


class QuantizedModelLike(Protocol):
    """Protocol for model-level quantization flags patched during load."""

    is_quantized: bool
    quantization_method: Any


class Bnb4BitModelLike(Protocol):
    """Protocol for bitsandbytes 4-bit runtime flags set on loaded models."""

    is_loaded_in_4bit: bool
    is_4bit_serializable: bool


class Bnb8BitModelLike(Protocol):
    """Protocol for bitsandbytes 8-bit runtime flags set on loaded models."""

    is_loaded_in_8bit: bool
    is_8bit_serializable: bool


class HqqModelLike(Protocol):
    """Protocol for HQQ runtime flags set on loaded models."""

    is_hqq_quantized: bool
    is_hqq_serializable: bool


class LoadingAttributesConfigLike(Protocol):
    """Protocol for quantization configs exposing loading-only attributes."""

    def get_loading_attributes(self) -> dict[str, Any]: ...


class BnbLoadIn8BitConfigLike(Protocol):
    """Protocol for configs exposing bitsandbytes load mode switches."""

    load_in_8bit: bool


class AqlmConfigLike(Protocol):
    """Protocol for AQLM config fields used by quantizers."""

    linear_weights_not_to_quantize: list[str] | None


class BitsAndBytesConfigLike(Protocol):
    """Protocol for bitsandbytes config fields used by quantizers."""

    llm_int8_enable_fp32_cpu_offload: bool
    llm_int8_skip_modules: list[str] | None


class ModulesToNotConvertConfigLike(Protocol):
    """Protocol for configs exposing module skip lists."""

    modules_to_not_convert: list[str] | None


class AwqConfigLike(Protocol):
    """Protocol for AWQ config fields used by quantizers."""

    modules_to_not_convert: list[str] | None
    desc_act: bool
    backend: Any


class BitNetConfigLike(Protocol):
    """Protocol for BitNet config fields used by quantizers."""

    modules_to_not_convert: list[str] | None
    linear_class: str
    quantization_mode: str


class CompressedTensorsConfigLike(Protocol):
    """Protocol for compressed-tensors config fields used by quantizers."""

    is_quantization_compressed: bool
    is_sparsification_compressed: bool


class FbgemmFp8ConfigLike(Protocol):
    """Protocol for FBGEMM FP8 config fields used by quantizers."""

    modules_to_not_convert: list[str] | None
    activation_scale_ub: float


class DequantizeWithModulesConfigLike(Protocol):
    """Protocol for configs with dequantize and module skip fields."""

    dequantize: bool
    modules_to_not_convert: list[str] | None


class FourOverSixConfigLike(Protocol):
    """Protocol for FourOverSix config fields used by quantizers."""

    keep_master_weights: bool


class FPQuantConfigLike(Protocol):
    """Protocol for FP-Quant config fields used by quantizers."""

    pseudoquantization: bool
    store_master_weights: bool


class GPTQConfigLike(Protocol):
    """Protocol for GPTQ config fields used by quantizers."""

    tokenizer: Any

    def to_dict_optimum(self) -> dict[str, Any]: ...


class HiggsConfigLike(Protocol):
    """Protocol for HIGGS config fields used by quantizers."""

    modules_to_not_convert: list[str] | None
    tune_metadata: dict[str, Any]


class QuantoConfigLike(Protocol):
    """Protocol for Quanto config fields used by quantizers."""

    weights: str
    activations: Any
    modules_to_not_convert: list[str] | None


class QuarkConfigLike(Protocol):
    """Protocol for Quark config fields used by quantizers."""

    quant_config: Any
    custom_mode: str


class SinqConfigLike(Protocol):
    """Protocol for SINQ config fields used by quantizers."""

    method: str
    modules_to_not_convert: list[str] | None


class TorchAoConfigLike(Protocol):
    """Protocol for TorchAO config fields used by quantizers."""

    quant_type: Any
    modules_to_not_convert: list[str] | None
    include_input_output_embeddings: bool

    def _get_ao_version(self) -> Any: ...


class TorchNpuLike(Protocol):
    """Protocol for torch module exposing the `npu` namespace."""

    npu: Any


class TorchHpuLike(Protocol):
    """Protocol for torch module exposing the `hpu` namespace."""

    hpu: Any


class WhisperGenerationConfigLike(Protocol):
    """Protocol for Whisper-specific generation config fields accessed in generation internals."""

    no_timestamps_token_id: int
