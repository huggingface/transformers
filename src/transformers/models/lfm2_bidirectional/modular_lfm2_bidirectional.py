# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ..lfm2.configuration_lfm2 import Lfm2Config
from ..lfm2.modeling_lfm2 import (
    Lfm2Attention,
    Lfm2DecoderLayer,
    Lfm2Model,
    Lfm2PreTrainedModel,
)


@auto_docstring(checkpoint="LiquidAI/LFM2.5-Embedding-350M")
@strict
class Lfm2BidirectionalConfig(Lfm2Config):
    r"""
    Configuration for the bidirectional (encoder) variant of LFM2, used for retrieval and embedding checkpoints such
    as [LiquidAI/LFM2.5-Embedding-350M](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M) and
    [LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M). It shares all fields with
    [`Lfm2Config`]; the model it points to applies bidirectional attention and a non-causal short convolution.

    conv_bias (`bool`, *optional*, defaults to `False`):
        Whether to use bias in the conv layers.
    conv_L_cache (`int`, *optional*, defaults to 3):
        L_cache dim in the conv layers.
    block_multiple_of (`int`, *optional*, defaults to 256):
        Multiple for the `intermediate_size`.
    block_ffn_dim_multiplier (`float`, *optional*, defaults to 1.0):
        Multiplier for the `intermediate_size`.
    block_auto_adjust_ff_dim (`bool`, *optional*, defaults to `True`):
        Whether to adjust the dim of the `intermediate_size`.
    full_attn_idxs (`Optional`, *optional*):
        Index of the layers which use attention.

    ```python
    >>> from transformers import Lfm2BidirectionalModel, Lfm2BidirectionalConfig

    >>> # Initializing a bidirectional LFM2 model
    >>> configuration = Lfm2BidirectionalConfig()

    >>> # Initializing a model from the LFM2.5-Embedding-350M style configuration
    >>> model = Lfm2BidirectionalModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "lfm2_bidirectional"


class Lfm2BidirectionalShortConv(nn.Module):
    """Non-causal short convolution: a centered depthwise conv1d, no cache / generation machinery."""

    def __init__(self, config: Lfm2BidirectionalConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.L_cache = config.conv_L_cache
        self.bias = config.conv_bias

        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.L_cache,
            groups=config.hidden_size,
            bias=self.bias,
            padding=self.L_cache - 1,
        )
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=self.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seqlen = hidden_states.shape[1]
        BCx = self.in_proj(hidden_states).transpose(-1, -2)
        B, C, x = BCx.chunk(3, dim=-2)
        Bx = B * x

        # centered conv (pad = kernel // 2 on each side) so each position mixes both neighbors
        pad = self.conv.weight.shape[-1] // 2
        conv_out = F.conv1d(Bx, self.conv.weight, self.conv.bias, stride=1, padding=pad, groups=Bx.shape[1])
        conv_out = conv_out[..., :seqlen]

        y = C * conv_out
        y = y.transpose(-1, -2).contiguous()
        return self.out_proj(y)


class Lfm2BidirectionalAttention(Lfm2Attention):
    def __init__(self, config: Lfm2BidirectionalConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = False


class Lfm2BidirectionalDecoderLayer(Lfm2DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if self.is_attention_layer:
            hidden_states, _ = self.self_attn(
                hidden_states=self.operator_norm(hidden_states),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
        else:
            hidden_states = self.conv(self.operator_norm(hidden_states))
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states


class Lfm2BidirectionalPreTrainedModel(Lfm2PreTrainedModel):
    config: Lfm2BidirectionalConfig
    _no_split_modules = ["Lfm2BidirectionalDecoderLayer"]
    # flash_attention_2 support is deferred; eager / sdpa only for now.
    _supports_flash_attn = False
    _can_record_outputs = {
        "hidden_states": Lfm2BidirectionalDecoderLayer,
        "attentions": Lfm2BidirectionalAttention,
    }


class Lfm2BidirectionalModel(Lfm2Model):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                **kwargs,
            )

        hidden_states = self.embedding_norm(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


__all__ = ["Lfm2BidirectionalConfig", "Lfm2BidirectionalModel", "Lfm2BidirectionalPreTrainedModel"]
