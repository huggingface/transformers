# Copyright 2026 The OpenBMB Team and the HuggingFace Inc. team. All rights reserved.
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

import math

from huggingface_hub.dataclasses import strict

from ...utils import auto_docstring
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import LlamaAttention
from ..minicpm3.modeling_minicpm3 import (
    MiniCPM3DecoderLayer,
    MiniCPM3ForCausalLM,
    MiniCPM3ForSequenceClassification,
    MiniCPM3Model,
    MiniCPM3PreTrainedModel,
)
from ..olmo.modeling_olmo import OlmoRotaryEmbedding, apply_rotary_pos_emb  # noqa: F401


@auto_docstring(checkpoint="openbmb/MiniCPM4-8B")
@strict
class MiniCPM4Config(LlamaConfig):
    r"""
    scale_emb (`int` or `float`, *optional*, defaults to 12):
        Multiplier applied to input embeddings.
    scale_depth (`int` or `float`, *optional*, defaults to 1.4):
        Multiplier for residual connections; the effective scaling is `scale_depth / sqrt(num_hidden_layers)`.
        If `None`, defaults to `sqrt(num_hidden_layers)` (no-op scaling).
    dim_model_base (`int`, *optional*, defaults to 256):
        Base model dimension used to scale logits before the language model head. If `None`,
        defaults to `hidden_size` (no-op scaling).

    Example:

    ```python
    >>> from transformers import MiniCPM4Config, MiniCPM4Model

    >>> configuration = MiniCPM4Config()
    >>> model = MiniCPM4Model(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "minicpm4"

    vocab_size: int = 73448
    intermediate_size: int = 16384
    num_key_value_heads: int | None = 2
    max_position_embeddings: int = 32768
    initializer_range: float = 0.1
    tie_word_embeddings: bool = True
    scale_emb: int | float = 12
    scale_depth: int | float | None = 1.4
    dim_model_base: int | None = 256

    def __post_init__(self, **kwargs):
        if self.scale_depth is None:
            self.scale_depth = math.sqrt(self.num_hidden_layers)
        if self.dim_model_base is None:
            self.dim_model_base = self.hidden_size
        super().__post_init__(**kwargs)

    @property
    def logits_scaling(self) -> float:
        return self.hidden_size / self.dim_model_base


class MiniCPM4RotaryEmbedding(OlmoRotaryEmbedding):
    pass


class MiniCPM4Attention(LlamaAttention):
    pass


class MiniCPM4DecoderLayer(MiniCPM3DecoderLayer):
    pass


class MiniCPM4PreTrainedModel(MiniCPM3PreTrainedModel):
    pass


class MiniCPM4Model(MiniCPM3Model):
    pass


class MiniCPM4ForCausalLM(MiniCPM3ForCausalLM):
    def forward(self, **super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, MiniCPM4ForCausalLM

        >>> model = MiniCPM4ForCausalLM.from_pretrained("openbmb/MiniCPM4-0.5B")
        >>> tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM4-0.5B")

        >>> prompt = "The capital of France is"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> generate_ids = model.generate(**inputs, max_new_tokens=10)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        ```"""
        return super().forward(**super_kwargs)


class MiniCPM4ForSequenceClassification(MiniCPM3ForSequenceClassification):
    pass


__all__ = [
    "MiniCPM4Config",
    "MiniCPM4PreTrainedModel",
    "MiniCPM4Model",
    "MiniCPM4ForCausalLM",
    "MiniCPM4ForSequenceClassification",
]
