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

import torch
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaPreTrainedModel,
)
from ..minicpm3.modeling_minicpm3 import MiniCPM3Model
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


class MiniCPM4ScaledWordEmbedding(Gemma3TextScaledWordEmbedding):
    pass


class MiniCPM4RotaryEmbedding(OlmoRotaryEmbedding):
    pass


class MiniCPM4Attention(LlamaAttention):
    pass


class MiniCPM4DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: MiniCPM4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.residual_scale = config.scale_depth / math.sqrt(config.num_hidden_layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states * self.residual_scale

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_scale
        return hidden_states


class MiniCPM4PreTrainedModel(LlamaPreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, MiniCPM4ScaledWordEmbedding):
            init.constant_(module.embed_scale, module.scalar_embed_scale)


class MiniCPM4Model(MiniCPM3Model):
    pass


@auto_docstring
class MiniCPM4ForCausalLM(LlamaForCausalLM):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
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
        ```
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state / self.config.logits_scaling
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]).float()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MiniCPM4ForSequenceClassification(LlamaForSequenceClassification):
    pass


__all__ = [
    "MiniCPM4Config",
    "MiniCPM4PreTrainedModel",
    "MiniCPM4Model",
    "MiniCPM4ForCausalLM",
    "MiniCPM4ForSequenceClassification",
]
