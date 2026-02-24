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
"""Protocol describing the model interface that GenerationMixin expects from its host class."""

from __future__ import annotations

from typing import Any, Protocol

import torch


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
    generation_config: Any  # GenerationConfig

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
