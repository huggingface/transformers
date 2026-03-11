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
    raiseExceptions: bool

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


class WhisperGenerationConfigLike(Protocol):
    """Protocol for Whisper-specific generation config fields accessed in generation internals."""

    no_timestamps_token_id: int
